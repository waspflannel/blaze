# Plan: `blaze fpga` (FPGA bitstream export, last step)

Branch `feat/fpga-export` from `origin/main`, one PR, not stacked on PR #12 (OpenLane). Source only: no tests,
no synthesis / place-and-route / bitstream runs, no hardware.

## Goal

Close the agent path weights -> blocks -> integrate -> **bitstream file** with one command and one dsh tool.
Plain weights are never a target-independent bitstream: the export takes one generated Verilog top plus an
explicit target and a user-supplied board constraints file.

## Target (one profile)

`ecp5-85f`: Lattice ECP5 LFE5U-85F, CABGA381 package (the ULX3S 85F board), open flow
`yosys synth_ecp5` -> `nextpnr-ecp5 --85k --package CABGA381` -> `ecppack`. No device, pin or timing data is
invented: pins and clock come only from the user's `.lpf`. Docs point at the ULX3S project's published LPF as
the source to copy `LOCATE` lines from.

Fit, honestly: the whole stories260K chip (tens of thousands of adders per layer, combinational matrices,
embedding ROM, KV cache) is not expected to fit 84K LUTs. One matrix block, possibly at reduced bits, is the
realistic first fit. nextpnr decides; blaze reports its failure verbatim.

## Behavior

`blaze fpga FILE --target ecp5-85f --lpf BOARD.lpf [--top NAME] [--wrap serial] [--model CKPT] [--out DIR] [--json]`

1. Validate before side effects: file and LPF exist (absent -> error), target known, top module found
   (default: file stem without `.opt`), port list parsed with `blaze.sim.module_ports`.
2. Interface check (rejected -> `UNSUPPORTED`): every top-level port bit must have a `LOCATE COMP` in the LPF.
   - A gen-matrix block's packed buses (512 in, ~1.3K out bits) can't be pinned, so it is rejected unless
     `--wrap serial`.
   - `--wrap serial` (combinational tops only, i.e. no `clk` port) writes `blaze_fpga_serial`: 5 pins
     (`clk`, `shift_in`, `shift_en`, `capture`, `shift_out`), input shift register -> block -> parallel-load
     output shift register. Labeled unvalidated in the file header and the report.
   - `blaze_chip` (clk/rst/start/token/pos/logit/done, a few dozen bits) goes direct with the user's LPF.
3. Tools looked up before any run (absent -> error with OSS CAD Suite install pointer; WSL fallback on Windows
   like sim-block/view). Run yosys, nextpnr (`--report`), ecppack; a nonzero exit -> failed, with the tail.
4. Success only when ecppack wrote a non-empty `.bit`. Then write `<top>.fpga.json` report: status, bitstream
   path, target, input sha256, LPF sha256, weights path + sha256 (if `--model`) else header source, wrapper,
   tool versions, Fmax + utilization from the nextpnr report or `"unavailable"`.

Errors: absent (missing file/tool) vs rejected (UNSUPPORTED interface / bad args) vs failed (tool exit) are
distinct messages; CLI exits nonzero like the other commands.

## Owned files

- `src/blaze/fpga/__init__.py` (new): profile, LPF check, serial wrapper, tool runs, report.
- `src/blaze/cli.py`: `fpga` subcommand + summary printer.
- `src/blaze/sim/__init__.py`: `module_ports` also reads `output reg` / `signed` ports (blaze_chip's header), so the
  pin check sees the chip's real ports. Matrix/op headers parse as before.
- `plugins/fpga-export.ts` (new): `blaze_fpga_export` tool.
- `plugins/cordis.patch.yml`: register the row.
- `plugins/README.md`: tool table row.
- `.dsh/skills/block-agent.md`: optional step 7, FPGA export only with an explicit target + LPF.
- `docs/fpga.md` (new): target, prerequisites (install as docs), CLI + agent invocation, limitations,
  not-validated statement.
- `docs/tools.md`, `docs/block-agent.md`, `README.md`: short pointers.

## Out of scope

Other boards/families, UART wrapper, programming the board (openFPGALoader), any run of the flow, tests,
OpenLane/PR #12 code.

## Checks

`python -m py_compile` on changed Python; `npm run typecheck` in `plugins/` only if `node_modules` exists.
