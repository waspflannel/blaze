# Blaze (working title)

Weights-to-silicon compiler: takes open model weights and compiles them into a fused hardware design (Verilog -> GDSII) with the weights burned into the circuits, ready to export to a manufacturer. We stop before sending to a fab.

- Free toolchain: Verilator, Yosys, OpenROAD/OpenLane, SkyWater 130nm open PDK
- Token-exact simulation equivalence vs. the reference model (core product)
- Reports: cycle-accurate tokens/sec (projected to 5nm), area, power

See [docs/design.md](docs/design.md). Status: early. The CLI can download a checkpoint, scan it into a spec, turn weight matrices and fixed-point op templates into Verilog, simulate a block against its Python reference with Verilator, and estimate adders and tok/s. Every command is described in [docs/tools.md](docs/tools.md).

## Setup

Python 3.10+. From the repo root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
pip install -e .
blaze fetch-model stories260K     # -> models/stories260K/ (gitignored)
blaze gen-matrix --layer 0 --tensor wq --out build/   # -> build/l0_wq_matvec.v + .txt (gitignored)
blaze sim-block build/l0_wq_matvec.v --ref layer0.wq  # needs Verilator, see below
```

### Simulation setup (Verilator)

`blaze sim-block` uses Verilator plus `make` and `g++`. If `verilator` is on PATH, blaze runs it directly; that's the usual case on Linux and macOS. On Windows, blaze runs it inside WSL from the free [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) unpacked in your WSL home. This needs no sudo and doesn't touch system paths:

```powershell
wsl --install -d Ubuntu            # only if WSL isn't set up yet
wsl -e bash -c "sudo apt install -y build-essential"   # g++ and make, if missing
wsl -e bash -c "mkdir -p ~/tools && cd ~/tools && curl -sSL https://github.com/YosysHQ/oss-cad-suite-build/releases/download/2026-11-04/oss-cad-suite-linux-x64-20261104.tgz | tar xz"
wsl -e bash -c "~/tools/oss-cad-suite/bin/verilator --version"
```

blaze adds `~/tools/oss-cad-suite/bin` to PATH for its WSL calls, so a newer release unpacked to the same place works too. The suite also includes Yosys, which the physical-design steps will use.

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
