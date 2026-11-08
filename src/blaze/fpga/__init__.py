"""FPGA export: one generated Verilog top -> a Lattice ECP5 bitstream with yosys, nextpnr-ecp5 and ecppack.

Source only: this flow has not been run end to end, and no bitstream from it has been loaded on a board.
Pins and clocks come only from the user's LPF; blaze holds no board pinout. See docs/fpga.md.
"""
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from blaze.sim import WSL_TOOLS, module_ports

# The one target: LFE5U-85F-6BG381C, the ULX3S 85F part (https://github.com/emard/ulx3s README).
# nextpnr would default the package with a deprecation warning, so every flag is spelled out.
TARGET = "ecp5-85f"
NEXTPNR_DEVICE = ["--85k", "--package", "CABGA381", "--speed", "6"]
TOOLS_MISSING = ("not found. Install OSS CAD Suite (it has yosys, nextpnr-ecp5 and ecppack) and put its bin/ on "
                 "PATH; on Windows, unpack it in WSL at ~/tools/oss-cad-suite as in the README's Simulation setup")
SERIAL_TOP = "blaze_fpga_serial"


def _command(args: list[str]) -> list[str]:
    """The tool from PATH, or from OSS CAD Suite inside WSL on Windows, like sim-block."""
    exe = shutil.which(args[0])
    if exe:
        return [exe, *args[1:]]
    if os.name == "nt" and shutil.which("wsl"):
        return ["wsl", "-e", "bash", "-c", f'PATH="{WSL_TOOLS}:$PATH"; {shlex.join(args)}']
    raise RuntimeError(f"{args[0]} {TOOLS_MISSING}")


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    r = subprocess.run(_command(args), cwd=cwd, capture_output=True, text=True)
    if r.returncode == 127:  # WSL bash: command not found
        raise RuntimeError(f"{args[0]} {TOOLS_MISSING}")
    if r.returncode:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-15:])
        raise RuntimeError(f"FAILED: {args[0]} exited with {r.returncode}:\n{tail}")
    return r


def _version(args: list[str], cwd: Path) -> str:
    # nextpnr and ecppack print their version to stderr, yosys to stdout.
    r = _run(args, cwd)
    return (r.stdout.strip() or r.stderr.strip()).splitlines()[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _port_bits(ports: dict[str, tuple[str, int]]) -> list[str]:
    """Each top-level pin as nextpnr names it in an LPF: `x` for 1 bit, `x[0]`..`x[n-1]` for a bus."""
    return [name if w == 1 else f"{name}[{i}]" for name, (_, w) in ports.items() for i in range(w)]


def located_pins(lpf: str) -> set[str]:
    """COMP names of every `LOCATE COMP "name" SITE "X";` line, skipping # and // comments."""
    text = "\n".join(re.split(r"#|//", line, maxsplit=1)[0] for line in lpf.splitlines())
    return set(re.findall(r'\bLOCATE\s+COMP\s+"([^"]+)"\s+SITE\b', text, re.I))


def serial_wrapper(top: str, ports: dict[str, tuple[str, int]]) -> str:
    """A 5-pin shift-register top around a combinational block. Unvalidated: never simulated or run on a board.

    shift_en shifts shift_in into the input register (inputs concatenated in port order, MSB first) and the
    output register out at shift_out (MSB first). capture loads every output into the output register.
    """
    ins = [(n, w) for n, (d, w) in ports.items() if d == "input"]
    outs = [(n, w) for n, (d, w) in ports.items() if d == "output"]
    n_in, n_out = sum(w for _, w in ins), sum(w for _, w in outs)
    conns, hi = [], n_in
    for name, w in ins:
        conns.append(f".{name}(in_sr[{hi - 1}:{hi - w}])")
        hi -= w
    conns += [f".{name}(o_{name})" for name, _ in outs]
    return "\n".join([
        f"// {SERIAL_TOP}: written by blaze fpga --wrap serial around {top}. UNVALIDATED: never simulated and never",
        "// run on hardware. Shift the inputs in on shift_en, wait for the block to settle, pulse capture, then shift",
        "// the outputs out on shift_en. Capture may come any number of cycles after the last shift, so the block is",
        "// a multicycle path; nextpnr's Fmax treats it as single-cycle and is a conservative bound.",
        f"module {SERIAL_TOP} (",
        "    input  wire clk,",
        "    input  wire shift_in,",
        "    input  wire shift_en,",
        "    input  wire capture,",
        "    output wire shift_out",
        ");",
        f"    reg [{n_in - 1}:0] in_sr;",
        f"    reg [{n_out - 1}:0] out_sr;",
        *[f"    wire [{w - 1}:0] o_{name};" for name, w in outs],
        f"    {top} block ({', '.join(conns)});",
        "    always @(posedge clk) begin",
        "        if (shift_en) in_sr <= {in_sr, shift_in};  // truncates: the oldest bit drops off the top",
        f"        if (capture) out_sr <= {{{', '.join(f'o_{name}' for name, _ in outs)}}};",
        "        else if (shift_en) out_sr <= {out_sr, 1'b0};",
        "    end",
        f"    assign shift_out = out_sr[{n_out - 1}];",
        "endmodule",
        "",
    ])


def _nextpnr_report(path: Path) -> tuple[dict | str, dict | str]:
    """(fmax, utilization) from nextpnr's --report JSON, each "unavailable" if absent."""
    try:
        r = json.loads(path.read_text())
    except (OSError, ValueError):
        return "unavailable", "unavailable"
    return r.get("fmax") or "unavailable", r.get("utilization") or "unavailable"


def export(verilog: Path, target: str, lpf: Path, top: str | None, wrap: str | None, model: Path | None,
           out: Path) -> dict:
    """Synthesize, place and route `verilog` for `target` with `lpf`, then pack the bitstream.

    Raises FileNotFoundError when an input is absent, ValueError when the design is rejected (UNSUPPORTED
    interface, bad arguments) and RuntimeError when a tool is missing or fails. Returns the report, also
    written to <out>/<stem>.fpga.json, only after ecppack wrote the .bit. <stem> is the block's top, plus
    `.serial` when wrapped, so wrapping several blocks into one directory keeps each one's files.
    """
    if target != TARGET:
        raise ValueError(f"unknown target {target!r}; the only target is {TARGET}")
    for path in (verilog, lpf, *([model] if model else [])):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found")
    text = verilog.read_text()
    top = top or verilog.name.removesuffix(".v").removesuffix(".opt")
    block, stem = top, top + (".serial" if wrap else "")
    ports = module_ports(text, top)
    if wrap == "serial":
        if "clk" in ports:
            raise ValueError(f"--wrap serial is for combinational blocks; {top} has a clk, so pin its ports directly")
        source = text + "\n" + serial_wrapper(top, ports)
        top = SERIAL_TOP
        ports = module_ports(source, top)
    else:
        source = text

    pins, located = _port_bits(ports), located_pins(lpf.read_text())
    unpinned = [p for p in pins if p not in located]
    if unpinned:
        hint = ("; a combinational block's buses can't all be pinned out, so --wrap serial puts it behind 5 pins "
                "(unvalidated)" if "clk" not in ports and wrap is None else "")
        raise ValueError(f"UNSUPPORTED: {len(unpinned)} of {len(pins)} port bits of {top} have no LOCATE in {lpf} "
                         f"(first: {', '.join(unpinned[:5])}){hint}")

    out.mkdir(parents=True, exist_ok=True)
    versions = {"yosys": _version(["yosys", "-V"], out),
                "nextpnr-ecp5": _version(["nextpnr-ecp5", "--version"], out),
                "ecppack": _version(["ecppack", "--version"], out)}
    # The tools run in `out` with relative names so the same commands work through WSL.
    src, lpf_copy, report_path = out / f"{stem}.fpga.v", out / f"{stem}.lpf", out / f"{stem}.fpga.json"
    netlist, config, pnr_report, bit = (out / f"{stem}{s}" for s in (".json", ".config", ".nextpnr.json", ".bit"))
    src.write_text(source, newline="\n")
    lpf_copy.write_text(lpf.read_text(), newline="\n")
    for stale in (netlist, config, pnr_report, bit, report_path):
        stale.unlink(missing_ok=True)

    _run(["yosys", "-q", "-p", f"read_verilog {src.name}; synth_ecp5 -top {top} -json {netlist.name}"], out)
    _run(["nextpnr-ecp5", *NEXTPNR_DEVICE, "--json", netlist.name, "--lpf", lpf_copy.name,
          "--textcfg", config.name, "--report", pnr_report.name], out)
    _run(["ecppack", config.name, bit.name], out)
    if not bit.is_file() or bit.stat().st_size == 0:
        raise RuntimeError(f"FAILED: ecppack exited 0 but wrote no bitstream at {bit}")

    fmax, utilization = _nextpnr_report(pnr_report)
    report = {
        "status": "bitstream written (not run on hardware)",
        "bitstream": str(bit), "target": target, "device": "LFE5U-85F-6BG381C", "top": top,
        "block": block, "wrapper": "serial (unvalidated)" if wrap else None,
        "input": {"verilog": str(verilog), "sha256": _sha256(verilog), "header": text.splitlines()[0] if text else ""},
        "lpf": {"path": str(lpf), "sha256": _sha256(lpf)},
        "weights": {"path": str(model), "sha256": _sha256(model)} if model else "unavailable (pass --model)",
        "tools": versions,
        "fmax_mhz": fmax, "utilization": utilization,
        "files": {"source": str(src), "netlist": str(netlist), "config": str(config), "nextpnr_report": str(pnr_report)},
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", newline="\n")
    return report
