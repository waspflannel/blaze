"""`blaze` command line entry point."""
import argparse
import json
import math
import re
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from blaze.generate import generate_matrix, quant_error, quantize
from blaze.integrate import integrate as integrate_chip
from blaze.library import BUILDERS, build_op, measure_error
from blaze.report import ADDER_NS, SKY130_CLOCK_MHZ, estimate_spec, estimate_verilog
from blaze.sim import MATMUL_TENSORS, reference_block, simulate
from blaze.spec import load_matrix
from blaze.spec.scan import UNUSED_TENSORS, build_spec, count_params, read_checkpoint

# Karpathy's tiny Llama checkpoints: <name>/<file> inside the HF repo.
REPO = "karpathy/tinyllamas"
MODELS = {"stories260K": ["stories260K.bin", "tok512.bin"]}
DEFAULT_MODEL = Path("models/stories260K/stories260K.bin")


def fetch_model(name: str, out: str) -> None:
    for f in MODELS[name]:
        path = Path(hf_hub_download(REPO, f"{name}/{f}", local_dir=out))
        print(f"{path}  {path.stat().st_size} bytes")


def scan_model(checkpoint: Path, out: Path) -> None:
    cfg, tensors = read_checkpoint(checkpoint)
    spec = build_spec(checkpoint, cfg, tensors)
    spec_path = out / spec["model"] / "spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    # Force \n so the spec is byte-identical on every OS.
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", newline="\n")

    # One row per tensor kind; per-layer tensors ("layers.3.wq") collapse into one row with a count.
    rows: dict[str, list] = {}
    for name, t in tensors.items():
        kind = name.split(".")[-1]
        rows.setdefault(kind, [t.shape, 0])[1] += 1
    n_matmul = sum(op["op"] == "matmul" for op in spec["ops"])
    print(f"model    {spec['model']} ({spec['architecture']}), {checkpoint.stat().st_size} bytes")
    print(f"layers   {cfg.n_layers}  dim {cfg.dim}  hidden {cfg.hidden_dim}  heads {cfg.n_heads} (kv {cfg.n_kv_heads})"
          f"  vocab {cfg.vocab_size}  seq {cfg.seq_len}")
    print(f"params   {count_params(tensors):,} ({'shared' if cfg.shared_classifier else 'separate'} classifier)")
    print(f"ops      {len(spec['ops'])} ({n_matmul} matmul, {len(spec['ops']) - n_matmul} template)")
    print()
    print(f"{'tensor':<16} {'shape':<12} {'count':>5} {'params':>8}")
    for kind, (shape, n) in rows.items():
        params = "unused" if kind in UNUSED_TENSORS else f"{n * math.prod(shape):,}"
        print(f"{kind:<16} {' x '.join(map(str, shape)):<12} {n:>5} {params:>8}")
    print()
    print(f"wrote    {spec_path}")


def gen_matrix(model: Path, layer: int, tensor: str, out: Path, bits: int) -> None:
    weights = load_matrix(model, layer, tensor)
    q, scale = quantize(weights, bits)
    module = f"l{layer}_{tensor}_matvec"
    verilog, equations, stats = generate_matrix(q, scale, module, f"{model.name} layer {layer} {tensor}", bits)

    out.mkdir(parents=True, exist_ok=True)
    verilog_path = out / f"{module}.v"
    equations_path = out / f"{module}.txt"
    # Force \n so the generated files are byte-identical on every OS.
    verilog_path.write_text(verilog, newline="\n")
    equations_path.write_text(equations, newline="\n")

    print(f"shape    {stats['rows']} x {stats['cols']} (out x in)")
    print(f"nonzero  {stats['nonzero']} / {stats['rows'] * stats['cols']}")
    print(f"adders   {stats['adders']} ({stats['csd_terms']} CSD terms), tree depth {stats['depth']}")
    print(f"scale    {stats['scale']!r}; y is signed {stats['acc_width']}-bit")
    print(f"bits     {bits}; weight error {quant_error(weights, q, scale):.4f} (relative RMS)")
    print(f"wrote    {verilog_path}, {equations_path}")


def _emit(summary: dict, lines: list[str], as_json: bool) -> None:
    print(json.dumps(summary, indent=2) if as_json else "\n".join(lines))


def gen_op(model: Path, layer: int, op: str, out: Path, as_json: bool) -> None:
    module = f"l{layer}_{op}"
    block = build_op(model, layer, op, module)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{module}.v"
    path.write_text(block.verilog, newline="\n")
    err = measure_error(block)
    ins = ", ".join(f"{p.name} {p.count} x {p.bits}-bit" for p in block.inputs)
    summary = {"op": op, "layer": layer, "module": module, "file": str(path),
               "inputs": [{"name": p.name, "count": p.count, "bits": p.bits} for p in block.inputs],
               "output": {"count": block.output.count, "bits": block.output.bits}, "out_scale": block.out_scale,
               "formats": block.formats, "error_vs_float": err, "cost_estimate": block.cost}
    _emit(summary, [
        f"op       {op} (layer {layer}) -> {module}",
        f"ports    in {ins}; out y {block.output.count} x signed {block.output.bits}-bit",
        f"format   {block.formats}; y_real ~= y * {block.out_scale:.6g}",
        f"error    max |y_real - float| {err['max_abs_err']:.3g} (float peak {err['max_abs_float']:.3g}), "
        f"{err['vectors']} vectors seed {err['seed']}",
        f"cost     ~{block.cost['adders']} adders, depth {block.cost['depth']} (estimate)",
        f"wrote    {path}",
    ], as_json)


def sim_block(path: Path, ref: str, model: Path, vectors: int, seed: int, as_json: bool) -> bool:
    # A matmul's reference is quantized at the bit width gen-matrix wrote into the file's stats line.
    m = re.search(r"^// blaze-stats: (.*)$", path.read_text(), re.M)
    block = reference_block(model, ref, json.loads(m[1]).get("bits", 8) if m else 8)
    r = simulate(path, block, vectors, seed, Path("build") / "sim" / path.stem)
    r["ref"] = ref
    first = r["first_mismatch"]
    lines = [
        f"sim      {path} vs {ref} ({r['simulator']})",
        f"vectors  {vectors} (seed {seed}), {r['outputs_per_vector']} outputs each",
        f"match    {'yes' if r['match'] else 'NO'}"
        + ("" if r["match"] else f", {r['mismatched_vectors']} vectors differ"),
    ]
    if first:
        lines.append(f"first    vector {first['vector']} y[{first['element']}]: "
                     f"expected {first['expected']}, got {first['got']}")
    _emit(r, lines, as_json)
    return r["match"]


def estimate(path: Path, model: Path | None, clock_mhz: float, adder_ns: float, as_json: bool) -> None:
    if path.suffix == ".json":
        r = estimate_spec(path, model, clock_mhz, adder_ns)
    else:
        r = estimate_verilog(path, clock_mhz, adder_ns)
    a = r["assumptions"]
    lines = [
        f"ESTIMATES ONLY (no synthesis): {path}",
        f"adders   ~{r['adders']:,} adder-equivalents",
        f"depth    {r['depth']} adder levels" + (f" ({r['deepest_op']})" if "deepest_op" in r else ""),
        f"clock    SKY130 {a['sky130_clock_mhz']:g} MHz assumed, {a['adder_ns']:g} ns per level"
        f" -> {a['levels_per_cycle']} levels/cycle; 5nm = x{a['scale_5nm']:g} clock",
    ]
    if "tok_per_s" in r:
        lines += [
            f"cycles   {r['cycles_per_token']:,} per token (attention at full context {r['context']})",
            f"tok/s    ~{r['tok_per_s']['sky130']:,.0f} SKY130, ~{r['tok_per_s']['5nm']:,.0f} 5nm",
            "",
            f"{'op':<14} {'count':>5} {'adders':>10} {'depth':>5} {'cycles':>7}",
        ]
        lines += [f"{k:<14} {v['count']:>5} {v['adders']:>10,} {v['max_depth']:>5} {v['cycles']:>7,}"
                  for k, v in r["by_op"].items()]
    else:
        lines += [f"cycles   {r['cycles']} per evaluation",
                  f"rate     ~{r['evals_per_s']['sky130']:,.0f} evals/s SKY130, ~{r['evals_per_s']['5nm']:,.0f} 5nm "
                  "(this block alone; tok/s needs a spec.json)"]
    _emit(r, lines, as_json)


def integrate(model: Path, blocks: Path | None, layers: int | None, steps: int, out: Path | None, sim: bool,
              prompt: str, as_json: bool) -> bool:
    blocks = blocks or Path("build") / "agent" / model.stem
    try:
        tokens = [int(t) for t in prompt.split(",") if t.strip()]
    except ValueError:
        raise ValueError(f"--prompt must be comma-separated token ids, got {prompt!r}") from None
    r = integrate_chip(model, blocks, layers, steps, out or Path("build") / "chip" / model.stem, sim, tokens)
    bits = ", ".join(f"{layer} " + " ".join(f"{t}:{b}" for t, b in ts.items()) for layer, ts in r["bits"].items())
    lines = [
        f"chip     {r['chip']} ({r['layers']} of {r['model_layers']} layers, blocks from {r['blocks']})",
        f"bits     {bits}",
        f"inputs   {r['inputs']} (BOS + prompt), then greedy",
        f"ref      {r['steps']} tokens: {r['reference_tokens']}",
    ]
    if r["reference_text"] is not None:
        lines.append(f"text     {r['reference_text']!r}")
    lines.append(f"float    {r['float_agreement']}/{r['steps']} tokens equal the float model's (information only)")
    if sim:
        first = r["first_mismatch"]
        lines += [f"sim      {r['simulator']}",
                  f"match    {'yes, token-exact' if r['match'] else 'NO'}"
                  + ("" if r["match"] else f": step {first['step']} expected {first['expected']}, got {first['got']}")]
        if not r["match"]:
            lines.append(f"chip     {r['chip_tokens']}")
        c = r["cycles_per_token"]
        lines += [f"cycles   {r['mean_cycles_per_token']:,.1f} per token on average ({c[0]:,} at pos 0, "
                  f"{c[-1]:,} at pos {len(c) - 1}), from the simulation",
                  f"tok/s    ~{r['tok_per_s']['sky130']:,.0f} SKY130 at {r['assumptions']['sky130_clock_mhz']:g} MHz, "
                  f"~{r['tok_per_s']['5nm']:,.0f} 5nm (estimate)"]
    _emit(r, lines, as_json)
    return r.get("match", True)


def main() -> None:
    p = argparse.ArgumentParser(prog="blaze")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch-model", help="download a model checkpoint + tokenizer into models/<name>/")
    f.add_argument("name", choices=MODELS)
    f.add_argument("--models-dir", default="models", help="default: ./models")
    s = sub.add_parser("scan-model", help="read a checkpoint and write its model spec to build/<model>/spec.json")
    s.add_argument("checkpoint", type=Path, help="llama2.c .bin checkpoint, e.g. models/stories260K/stories260K.bin")
    s.add_argument("--out", type=Path, default=Path("build"), help="output directory (default: ./build)")
    g = sub.add_parser("gen-matrix", help="turn one weight matrix into shift/add equations and Verilog")
    g.add_argument("--model", type=Path, default=Path("models/stories260K/stories260K.bin"),
                   help="llama2.c .bin checkpoint (default: models/stories260K/stories260K.bin)")
    g.add_argument("--layer", type=int, default=0, help="default: 0")
    g.add_argument("--tensor", choices=MATMUL_TENSORS, default="wq", help="default: wq (query projection)")
    g.add_argument("--out", type=Path, default=Path("build"), help="output directory (default: ./build)")
    g.add_argument("--bits", type=int, choices=range(2, 9), default=8, metavar="2..8",
                   help="signed weight bits (default: 8)")
    o = sub.add_parser("gen-op", help="write a fixed-point op template (Verilog) for one layer")
    o.add_argument("op", choices=BUILDERS)
    o.add_argument("--model", type=Path, default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    o.add_argument("--layer", type=int, default=0, help="default: 0")
    o.add_argument("--out", type=Path, default=Path("build"), help="output directory (default: ./build)")
    o.add_argument("--json", action="store_true", help="print the summary as JSON")
    b = sub.add_parser("sim-block", help="simulate a generated block with Verilator against its reference")
    b.add_argument("file", type=Path, help="generated Verilog, e.g. build/l0_wq_matvec.v")
    b.add_argument("--ref", required=True, help="layer<N>.<tensor or op>, e.g. layer0.wq or layer0.rmsnorm")
    b.add_argument("--model", type=Path, default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    b.add_argument("--vectors", type=int, default=1000, help="random input vectors (default: 1000)")
    b.add_argument("--seed", type=int, default=0, help="default: 0")
    b.add_argument("--json", action="store_true", help="print the summary as JSON")
    e = sub.add_parser("estimate", help="estimate adders, depth and tok/s for a generated .v or a spec.json")
    e.add_argument("file", type=Path, help="generated .v or build/<model>/spec.json")
    e.add_argument("--model", type=Path, help="checkpoint for a spec (default: models/<model>/<source file>)")
    e.add_argument("--clock-mhz", type=float, default=SKY130_CLOCK_MHZ,
                   help=f"assumed SKY130 clock (default: {SKY130_CLOCK_MHZ:g})")
    e.add_argument("--adder-ns", type=float, default=ADDER_NS,
                   help=f"assumed delay of one adder level (default: {ADDER_NS:g})")
    e.add_argument("--json", action="store_true", help="print the summary as JSON")
    i = sub.add_parser("integrate", help="stitch verified blocks into one chip, simulate it and compare tokens")
    i.add_argument("--model", type=Path, default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    i.add_argument("--blocks", type=Path, help="verified block files (default: build/agent/<model file stem>)")
    i.add_argument("--layers", type=int, help="build the first N layers (default: all)")
    i.add_argument("--steps", type=int, default=32, help="positions to run, 1..seq_len (default: 32)")
    i.add_argument("--prompt", default="", help="comma-separated token ids fed after BOS before greedy decoding")
    i.add_argument("--out", type=Path, help="output directory (default: build/chip/<model file stem>)")
    i.add_argument("--no-sim", action="store_true", help="write the chip Verilog only")
    i.add_argument("--json", action="store_true", help="print the summary as JSON")
    a = p.parse_args()

    if a.cmd == "fetch-model":
        fetch_model(a.name, a.models_dir)
    elif a.cmd == "scan-model":
        try:
            scan_model(a.checkpoint, a.out)
        except (OSError, ValueError) as e:
            sys.exit(f"blaze scan-model: {e}")
    elif a.cmd == "gen-matrix":
        try:
            gen_matrix(a.model, a.layer, a.tensor, a.out, a.bits)
        except (OSError, ValueError) as e:
            sys.exit(f"blaze gen-matrix: {e}")
    elif a.cmd == "gen-op":
        try:
            gen_op(a.model, a.layer, a.op, a.out, a.json)
        except (OSError, ValueError) as e:
            sys.exit(f"blaze gen-op: {e}")
    elif a.cmd == "sim-block":
        try:
            ok = sim_block(a.file, a.ref, a.model, a.vectors, a.seed, a.json)
        except (OSError, ValueError, RuntimeError) as e:
            sys.exit(f"blaze sim-block: {e}")
        sys.exit(0 if ok else 1)
    elif a.cmd == "estimate":
        try:
            estimate(a.file, a.model, a.clock_mhz, a.adder_ns, a.json)
        except (OSError, ValueError, KeyError) as e:
            sys.exit(f"blaze estimate: {e}")
    elif a.cmd == "integrate":
        try:
            ok = integrate(a.model, a.blocks, a.layers, a.steps, a.out, not a.no_sim, a.prompt, a.json)
        except (OSError, ValueError, RuntimeError) as e:
            sys.exit(f"blaze integrate: {e}")
        sys.exit(0 if ok else 1)
