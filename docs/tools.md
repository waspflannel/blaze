# Tools

Every `blaze` command is plain deterministic Python with no LLM. The same inputs always give the same files and numbers. The newer commands (`optimize-matrix`, `gen-op`, `sim-block`, `estimate`) print a short `key  value` summary, and `--json` swaps that for one JSON object an agent can parse. Generated files go under `build/`, which is gitignored.

## fetch-model

```powershell
blaze fetch-model stories260K     # -> models/stories260K/stories260K.bin + tok512.bin
```

This downloads a checkpoint and its tokenizer from Hugging Face (`karpathy/tinyllamas`) into `models/<name>/`. `--models-dir` changes the root.

## scan-model

```powershell
blaze scan-model models/stories260K/stories260K.bin   # -> build/stories260K/spec.json
```

This reads a llama2.c checkpoint and writes the model spec: the config plus every op of one forward pass, in order. See [scan-model.md](scan-model.md).

## gen-matrix

```powershell
blaze gen-matrix --layer 0 --tensor wq --out build/stories260K            # -> build/stories260K/layer0/l0_wq_matvec.v + .txt
blaze gen-matrix --layer 0 --tensor wq --bits 6 --out build/stories260K   # 6-bit weights
```

This turns one weight matrix into shift/add equations and a combinational Verilog module. See [generator.md](generator.md). `--bits` (2..8, default 8) sets the signed weight width. Fewer bits means fewer adders and more weight error, and the summary prints that error as relative RMS. The Verilog header has a `// blaze-stats: {"adders": ..., "depth": ..., "bits": ...}` line. `estimate` reads it, and `sim-block` quantizes a matmul's reference at its `bits`. Depth is the deepest row's balanced adder tree, `ceil(log2(CSD terms))`.

## optimize-matrix

```powershell
blaze optimize-matrix build/stories260K/layer0/l0_wq_matvec.v          # -> build/stories260K/layer0/l0_wq_matvec.opt.v
blaze optimize-matrix build/stories260K/layer0/l0_wq_matvec.v --json   # summary as JSON
blaze optimize-matrix build/stories260K/layer0/l0_wq_matvec.v --discard   # delete that .opt.v
```

This rewrites a gen-matrix block with fewer adders and the same math. It shares the shifted terms and partial sums that several rows use, such as `x_a + (x_b <<< k)`, as named wires, greedily, most frequent pair first. The input has to be exactly gen-matrix output for `--model` (default stories260K); the layer, tensor and bits come from its header and `blaze-stats` line. The output goes next to it as `<stem>.opt.v`, combinational Verilog-2005 with the same module name and ports, and its `blaze-stats` line adds `"optimized": true` and `"original_adders"`. The summary prints adders and depth before and after. `--discard` deletes the `.opt.v` instead, for one that failed `sim-block`. See [optimizer.md](optimizer.md). On stories260K layer 0 at 8 bits it cuts the 7 matrices from 93,247 adders to 40,954, with the same depth.

## gen-op

```powershell
blaze gen-op rmsnorm --layer 0 --out build/stories260K   # -> build/stories260K/layer0/l0_rmsnorm.v
```

This writes one fixed-point op as Verilog for one layer of the checkpoint (`--model`, default stories260K). Ops: `rmsnorm`, `silu_mul`, `softmax`, `rope`, `residual_add`. Each op is a template in `src/blaze/library/verilog/blaze_<op>.v` with its constants passed as parameters (LUTs, gamma), plus a thin top module `l<layer>_<op>` sized from the config. Each op has a Python integer reference in `blaze.library` that the Verilog matches bit for bit, and a float reference used to measure error. All blocks are combinational.

Every op input is signed int8 with 4 fractional bits (`real = v / 16`). Outputs are exact integers in a wider format, and the summary prints the `y_real ~= y * scale` factor. Error is the max `|y_real - float|` over 1000 seeded random vectors (seed 0) for stories260K layer 0:

| op | ports | method | max error vs float |
|---|---|---|---|
| `residual_add` | a, b: dim; y 9-bit | exact add | 0 |
| `silu_mul` | a (w1 x), b (w3 x): hidden_dim; y 20-bit | 256-entry silu lookup (8 frac bits), exact multiply | 0.0156 (outputs up to 63.5) |
| `softmax` | x: seq_len scores; y 17-bit, 15 frac bits | max-subtract, 256-entry exp lookup, sum, divide (floor) | 3.1e-5, about 1 LSB (outputs up to 0.045) |
| `rope` | x: dim, pos; y 24-bit | cos/sin ROM per position (14 frac bits), exact rotate | 4.4e-4 (outputs up to 11.2) |
| `rmsnorm` | x: dim; y 16-bit | exact sum of squares, bit-serial integer sqrt, truncating divide, int8 gamma | 0.019 (outputs up to 5.3), mostly gamma quantization |

Simplifications:

- `rmsnorm` drops eps. An all-zero input gives 0, and inputs with a mean square near eps (1e-5) come out larger than the float result. It uses the attention-norm weights (`rms_att`). `1/sqrt(dim)` is folded into the output scale.
- `softmax` has no mask: it covers all `seq_len` positions. The `1/sqrt(head_size)` score scale has to be applied before it.
- `rope` covers the `dim` query elements (all heads). `pos` must be below `seq_len`.

## sim-block

```powershell
blaze sim-block build/stories260K/layer0/l0_wq_matvec.v --ref layer0.wq
blaze sim-block build/stories260K/layer0/l0_rmsnorm.v --ref layer0.rmsnorm --vectors 1000 --seed 0 --json
```

This compiles the file with Verilator, drives `--vectors` seeded random inputs (default 1000, seed 0), and compares every output bit for bit with the Python integer reference. Data inputs are signed int8 and `rope`'s `pos` is drawn from `0..seq_len-1`. The first three vectors are all-minimum, all-maximum and all-zero.

- `--ref layer<N>.<name>`: `<name>` is a matmul tensor (`wq wk wv wo w1 w2 w3`, reference `blaze.reference.matvec_int` on the quantized weights) or a `gen-op` op. The reference is rebuilt from `--model` (default stories260K), so it has to be the checkpoint the file was generated from.
- The top module must be named after the file up to its first dot (`l0_wq_matvec.v` and `l0_wq_matvec.opt.v` -> `l0_wq_matvec`), and its ports must have the names and widths the reference expects. Otherwise the command stops before simulating.
- Output: match yes/no, vectors run, and the first mismatch (vector index, output element, expected, got). The exit code is 1 on a mismatch.
- sim-block compiles the Verilog to C++ and runs it, and Verilog can call `$system` or open files. Only simulate files you generated or trust.
- The work directory is `build/sim/<file stem>/`, holding a copy of the Verilog, a generated `tb.cpp` and Verilator's `obj/`.

Verilator setup on Windows (WSL + OSS CAD Suite) is in the [README](../README.md#simulation-setup-verilator).

stories260K layer 0, 1000 vectors, seed 0, Verilator 5.055: `wq`, `rmsnorm`, `silu_mul`, `softmax`, `rope` and `residual_add` all match, and so do all 7 `optimize-matrix` files.

## integrate

```powershell
blaze integrate --layers 1 --steps 32   # blocks from build/stories260K/layerN -> build/stories260K/chip/blaze_chip.v
```

This stitches the verified blocks of the first `--layers` layers (default: all) into one chip, `blaze_chip`. The chip adds the embedding ROM, attention with a KV cache per layer, the final norm, the classifier and a control FSM. The command then simulates the chip with Verilator, runs `--steps` positions (BOS, then the optional `--prompt` token ids, then each output token fed back, greedy), and compares every token and its logit with the Python integer reference. It reports the match, the first mismatch, the tokens as text, and cycles per token measured in the simulation. `--blocks` defaults to `build/<model file stem>` and has to hold all 12 block files of every included layer under `layer<N>/`, exactly as `gen-matrix` and `gen-op` wrote them. A matrix's `.opt.v` is used in place of its `.v` when it exists, checked the same way against what `optimize-matrix` writes, and the summary's `opt` line lists those matrices. A flat directory (the old `build/agent/<model file stem>/` layout) still works. `--out` defaults to `build/<model file stem>/chip`. `--no-sim` only writes the Verilog. The exit code is 1 on a mismatch. Numerics, control and results are in [integrator.md](integrator.md). Like `sim-block`, it compiles and runs Verilog, so only use blocks you generated.

## view

```powershell
blaze view --model build/stories260K                                       # -> build/stories260K/view/stories260K.html
blaze view build/stories260K/layer0/l0_wq_matvec.v                         # -> build/stories260K/layer0/view/l0_wq_matvec.html
blaze view build/stories260K/layer0/l0_wq_matvec.v --row 12 --schematic    # -> view/l0_wq_matvec.row12.svg (+ .v, .json)
```

This draws generated blocks for a person to look at, as one self-contained HTML page with no network access needed. `--model build/<model>/` (or its `chip/`) lays out the whole model as a pan/zoom chip workspace: embedding, every layer's matrices and ops in data-flow order, final norm, classifier and argmax, with matrix tiles sized by adder count and blocks not generated yet as dim placeholders. `--checkpoint` (default `models/<model>/<model>.bin`) gives the layer count and the placeholders' estimates. Clicking a matrix zooms into its adder graph. Given one block file instead, the page opens straight on that block's graph. The block is parsed from its Verilog assigns into an adder graph (inputs, shifts, add/sub nodes, shared intermediate wires, outputs), drawn whole with pan/zoom, row search, fan-in cone highlighting and nodes used by more than one output marked as shared. It reads plain gen-matrix blocks and optimized `.opt.v` blocks the same way. On a plain block its adders and max depth equal the file's `blaze-stats` line.

`--row N --schematic` extracts output row N's fan-in cone into a small standalone module, synthesizes it to generic gates with Yosys (`synth -flatten -noabc`, then `write_json`) and draws it with [netlistsvg](https://github.com/nturley/netlistsvg). Yosys comes from OSS CAD Suite like Verilator (native, or in WSL on Windows; see the [README](../README.md#simulation-setup-verilator)). netlistsvg is an npm package: `npm install -g netlistsvg`, or install it in any folder and put its `node_modules/.bin` on PATH. If either is missing, the command stops with that install message. `-o` sets the output file, `--json` prints the summary as JSON. Details and stories260K numbers are in [viewer.md](viewer.md).

## estimate

```powershell
blaze estimate build/stories260K/spec.json   # whole model: tok/s
blaze estimate build/stories260K/layer0/l0_wq_matvec.v   # one block: adders, depth, cycles
```

**All numbers are estimates from a simple model, not synthesis results.**

- **Adders** are adder-equivalents, meaning two-input add/subtract/compare. Matmuls count CSD terms - 1 per row from the quantized checkpoint weights (an `optimize-matrix` file adds its shared wires and counts fewer terms per row). In templates, an n-bit multiply counts as n - 1 adders, a divider or square root counts one adder per result bit, and lookups are free.
- **Depth** is the longest chain of adder levels in one op, assuming balanced trees. Dividers and square roots add one level per result bit.
- **Cycles**: each op is combinational with registered outputs, so it takes `ceil(depth / levels per cycle)` cycles, at least 1. Attention also streams the KV cache one row per cycle, twice (q.k, then p.v), at full context (`seq_len`). Ops run one after another, and cycles per token is the sum.
- **Clock**: SKY130 at 50 MHz (`--clock-mhz`) and 5 ns per adder level (`--adder-ns`), so 4 levels per cycle. 5nm assumes the same cycles at 10x the clock, from gate delay of roughly 45 ps at 130 nm vs 5 ps at 5 nm. Area and power aren't modeled.
- A spec needs its checkpoint (`--model`, default `models/<model>/<source file>`), and the checkpoint's sha256 has to match the spec. A `.v` is read from its `blaze-stats` line and reports evaluations per second for that block alone.

stories260K at the defaults:

| | |
|---|---|
| adders | ~975,290 (matmuls 545,288; attention 394,120; rmsnorm 22,022) |
| deepest op | attention, 45 levels (rmsnorm 44, matmul 9) |
| cycles per token | 5,399 (attention 5,180) |
| tok/s | ~9,300 SKY130, ~93,000 5nm |
