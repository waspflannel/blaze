# Blaze

Blaze compiles open model weights into hardware. It takes a llama2.c-format checkpoint and turns it into fused Verilog with the weights burned into the circuits as fixed shift/add networks, so there is no weight memory to fetch from. Every generated block, and then the integrated chip, is simulated in Verilator and checked token-exact against an integer quantized reference model.

For the stories260K model, the integrated chip simulates at about 45,000 tokens/s at an assumed 50 MHz clock. The cycles per token are measured in simulation; the clock is an assumption, and nothing has been measured on real silicon or an FPGA. Blaze stops at verified Verilog and simulation; it does not produce anything for a fab.

Every command is described in [docs/tools.md](docs/tools.md). The design is in [docs/design.md](docs/design.md).

## Setup

Python 3.10+. From the repo root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
pip install -e .
```

### Verilator

`blaze sim-block` and `blaze integrate` use Verilator plus `make` and `g++`. If `verilator` is on PATH, blaze runs it directly; that's the usual case on Linux and macOS. On Windows, blaze runs it inside WSL from the free [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) unpacked in your WSL home. This needs no sudo and doesn't touch system paths:

```powershell
wsl --install -d Ubuntu            # only if WSL isn't set up yet
wsl -e bash -c "sudo apt install -y build-essential"   # g++ and make, if missing
wsl -e bash -c "mkdir -p ~/tools && cd ~/tools && curl -sSL https://github.com/YosysHQ/oss-cad-suite-build/releases/download/2026-11-04/oss-cad-suite-linux-x64-20261104.tgz | tar xz"
wsl -e bash -c "~/tools/oss-cad-suite/bin/verilator --version"
```

blaze adds `~/tools/oss-cad-suite/bin` to PATH for its WSL calls, so a newer release unpacked to the same place works too. The suite also includes Yosys, which `blaze view --schematic` uses.

## Usage

```powershell
blaze fetch-model stories260K     # -> models/stories260K/ (gitignored)
blaze gen-matrix --layer 0 --tensor wq --out build/stories260K   # -> build/stories260K/layer0/l0_wq_matvec.v + .txt
blaze sim-block build/stories260K/layer0/l0_wq_matvec.v --ref layer0.wq  # bit-exact vs the integer reference
blaze integrate --layers 1 --steps 32   # verified blocks -> build/stories260K/chip/blaze_chip.v, token-exact chip sim
```

`integrate` needs every block of the included layers, which the block agent below generates. `fetch-model` downloads checkpoints from Andrej Karpathy's [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas) on Hugging Face (stories260K by default).

## dsh plugins

[`plugins/`](plugins/README.md) is `blaze-dsh`, a DeepSeek Harness (`dsh`) bundle with one plugin per `blaze` command. Each tool runs `blaze` and returns JSON:

| tool | what it does |
|---|---|
| `blaze_scan_model` | scans a checkpoint into a spec: config, params, op counts |
| `blaze_gen_matrix` | turns one weight matrix into a shift/add Verilog block |
| `blaze_optimize_matrix` | shares common subexpressions in a generated block (`.opt.v`) |
| `blaze_sim_block` | simulates a block in Verilator against its integer reference |
| `blaze_gen_op` | generates a fixed-point op (rmsnorm, softmax, silu_mul, ...) as Verilog |
| `blaze_estimate` | estimates adders, depth, cycles and tok/s |
| `blaze_integrate` | stitches verified blocks into one chip and checks its tokens against the reference |
| `blaze_view` | renders the model, a block's adder graph, or one row's gate schematic as HTML |

A simulation or token mismatch comes back as a normal result with `match: false`; a failed command comes back as a tool error.

## Block agent

The block agent is a dsh skill, [`.dsh/skills/block-agent.md`](.dsh/skills/block-agent.md). Given a weights file, it scans the model, generates every layer's matrices and ops, simulates each block (and its optimized version) until it matches the reference bit for bit, then integrates the chip and checks it token-exact. It runs on Codex through your ChatGPT sign-in, not an API key.

One-time setup (`dsh` is `C:\deepseek harness\harness\resources\runtime\cli\bin\dsh.cmd` if it isn't on PATH):

```powershell
cd C:\blaze
dsh --profile blaze --from-default-profile headless --help      # creates ~/.dsh/profiles/blaze
dsh plugin --profile blaze add dsh-codex@0.3.2 link:C:/blaze/plugins
Copy-Item plugins\codex.patch.yml $HOME\.dsh\profiles\blaze\cordis.patch.yml
```

Sign in once from the dsh desktop app under **Settings -> OpenAI Codex -> Sign in with ChatGPT**. Then run:

```powershell
$env:BLAZE_BIN = "C:\blaze\.venv\Scripts\blaze.exe"
dsh --profile blaze "Use the block-agent skill on models/stories260K/stories260K.bin."
```

Blocks go to `build/<model>/layer<N>/` and the chip to `build/<model>/chip/`. Full setup, verification of the profile, and options are in [docs/block-agent.md](docs/block-agent.md). To use the tools without the agent profile (e.g. the `sdk-minimal` profile over JSON-RPC), see [plugins/README.md](plugins/README.md).

## Open PRs

These are not on `main` and have not produced verified output yet:

- [#12](https://github.com/waspflannel/blaze/pull/12): OpenLane physical design (`blaze pd`, GDSII). No verified GDS.
- [#13](https://github.com/waspflannel/blaze/pull/13): FPGA bitstream export (`blaze fpga`, `blaze_fpga_export`). No verified bitstream.

## Docs

- [docs/tools.md](docs/tools.md): every `blaze` command
- [docs/block-agent.md](docs/block-agent.md): the block agent and its dsh profile
- [plugins/README.md](plugins/README.md): the dsh bundle and how dsh loads it
- [docs/integrator.md](docs/integrator.md), [docs/optimizer.md](docs/optimizer.md), [docs/viewer.md](docs/viewer.md): chip integration, block optimization, the viewer
