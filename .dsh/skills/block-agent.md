---
name: block-agent
description: Compile one transformer layer of a llama2.c checkpoint (default stories260K layer 0) into Verilog blocks with the blaze_* tools, verify every block against its reference with sim-block, then report the estimates.
whenToUse: The user asks to build, compile, generate or verify a layer (or "block agent", "layer 0") with blaze.
---

# Blaze block agent

You turn one layer of a model into hardware blocks. Every block must match its Python integer reference bit for bit in simulation. Use only the `blaze_*` tools. Don't edit files, write Verilog yourself, or run blaze from the shell. Only simulate files a blaze tool just wrote.

## Inputs

Take these from the task. Use the default for anything it doesn't say.

| input | default |
|---|---|
| checkpoint | `models/stories260K/stories260K.bin` |
| layer | `0` |
| matrix bits | `8` for every matrix (2..8) |
| output directory | `build/agent` |

## Blocks

One layer is 12 blocks, each generated and simulated once:

- 7 matrices, `blaze_gen_matrix` with `tensor` = `wq wk wv wo w1 w2 w3`, simulated with `ref` = `layer<N>.<tensor>`.
- 5 ops, `blaze_gen_op` with `op` = `rmsnorm silu_mul softmax rope residual_add`, simulated with `ref` = `layer<N>.<op>`.

Layer adders is the sum of each block's adders once: `adders` from gen-matrix plus `cost_estimate.adders` from gen-op. Ops have fixed int8 inputs and no bit-width setting.

## Steps

1. **Scan.** Call `blaze_scan_model` on the checkpoint. Note dim, hidden_dim, n_layers and the spec path. Stop if the layer index isn't below n_layers.
2. **Generate.** Call `blaze_gen_matrix` for each matrix with the matrix bits, and `blaze_gen_op` for each op. Pass the checkpoint, layer and output directory each time. Keep each returned file path, adders, depth and `weight_error` (matrices) or `error_vs_float` (ops).
3. **Verify.** Call `blaze_sim_block` on every returned file, using the default 1000 vectors and seed 0. A block passes only when `match` is true on its latest file. A regenerated file must be simulated again.
4. **Retry** until every block matches:
   - **Mismatch:** blaze output is deterministic, so a mismatch is a real fault. Regenerate the block once with the same settings and re-simulate it. If it still fails and it's a matrix, try one bit width up and one down (never outside 2..8). Ops have nothing to adjust. After 3 failed attempts on one block, stop retrying it and report its `first_mismatch` (vector, element, expected, got).
   - **Tool error** (bad path, missing Verilator): report blaze's message and stop. Don't work around it.
5. **Estimate.** Call `blaze_estimate` on each final block file for adders, depth, cycles and evals/s, and once on the spec path for the whole-model tok/s. The spec estimate prices every matmul at 8 bits, so label it as the int8 baseline.

## Report

End with this, in plain markdown:

1. **Result**: `PASS` (all 12 blocks match) or `FAIL`, with the reason.
2. **Blocks**: one table row per block with block, bits (matrices) or `int8 in` (ops), adders, depth, error (`weight_error` or `error_vs_float.max_abs_err`), sim match and vectors, cycles, and evals/s at SKY130 and 5nm.
3. **Layer**: total adders, the deepest block, and the max weight error. These are info only; there is no adder or error limit.
4. **Attempts**: every retry, in order, as what changed, why, and the outcome. Example: `wo regenerated at 8 bits: mismatch at vector 3; match`. Write `none` if every block matched first time.
5. **Whole model (int8 baseline)**: adders, cycles per token, and tok/s at SKY130 and 5nm from the spec estimate.
6. **Files**: the final `.v` paths.

All numbers are blaze estimates, not synthesis results. Say so once.
