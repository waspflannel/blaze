# Blaze (working title)

Weights-to-silicon compiler: takes open model weights and compiles them into a fused hardware design (Verilog -> GDSII) with the weights burned into the circuits, ready to export to a manufacturer. We stop before sending to a fab.

- Free toolchain: Verilator, Yosys, OpenROAD/OpenLane, SkyWater 130nm open PDK
- Token-exact simulation equivalence vs. the reference model (core product)
- Reports: cycle-accurate tokens/sec (projected to 5nm), area, power

See [docs/design.md](docs/design.md). Status: early. `blaze fetch-model` downloads a checkpoint, and `blaze gen-matrix` turns one weight matrix into shift/add Verilog (see [docs/generator.md](docs/generator.md)).

## Setup

Python 3.10+. From the repo root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
pip install -e .
blaze fetch-model stories260K     # -> models/stories260K/ (gitignored)
blaze gen-matrix --layer 0 --tensor wq --out build/   # -> build/l0_wq_matvec.v + .txt (gitignored)
```

## Layout

- `src/blaze/spec` - read a llama2.c checkpoint's config and weight matrices
- `src/blaze/reference` - reference results generated hardware must match (integer matvec now; model activations later)
- `src/blaze/library` - hardware component templates (Verilog in `library/verilog/`)
- `src/blaze/generate` - weights -> shift/add equations -> Verilog (`gen-matrix`)
- `src/blaze/sim` - Verilator block/chip compare
- `src/blaze/pd` - Yosys/OpenLane wrappers
- `src/blaze/report` - tokens/sec, area, power reports
- `src/blaze/cli.py` - `blaze` CLI
- `plugins/` - dsh TypeScript plugins exposing each `blaze` command as a tool (see its README)
- `models/` - downloaded checkpoints (gitignored)
