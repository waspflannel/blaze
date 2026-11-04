# Blaze (working title)

Weights-to-silicon compiler: takes open model weights and compiles them into a fused hardware design (Verilog -> GDSII) with the weights burned into the circuits, ready to export to a manufacturer. We stop before sending to a fab.

- Free toolchain: Verilator, Yosys, OpenROAD/OpenLane, SkyWater 130nm open PDK
- Token-exact simulation equivalence vs. the reference model (core product)
- Reports: cycle-accurate tokens/sec (projected to 5nm), area, power

See [docs/design.md](docs/design.md). Status: scaffold only; only `blaze fetch-model` works.

## Setup

Python 3.10+. From the repo root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
pip install -e .
blaze fetch-model stories260K     # -> models/stories260K/ (gitignored)
```

## Layout

- `src/blaze/spec` - read checkpoint + config into a model spec
- `src/blaze/reference` - run the reference model, capture per-layer activations
- `src/blaze/library` - hardware component templates (Verilog in `library/verilog/`)
- `src/blaze/generate` - weights -> Verilog generators
- `src/blaze/sim` - Verilator block/chip compare
- `src/blaze/pd` - Yosys/OpenLane wrappers
- `src/blaze/report` - tokens/sec, area, power reports
- `src/blaze/cli.py` - `blaze` CLI
- `plugins/` - future dsh TypeScript plugins (see its README)
- `models/` - downloaded checkpoints (gitignored)
