# Integrator

`blaze integrate` stitches the block agent's verified blocks into one chip, `blaze_chip`, then simulates it with Verilator. The chip reads BOS, then the optional prompt tokens, then each output token fed back (greedy), and every token and its logit have to equal the Python integer reference of the same chip. That token-exact check is the core correctness check. The command also reports cycles per token measured in the simulation. Like every blaze command, it's deterministic Python with no LLM.

```powershell
blaze integrate --layers 1 --steps 32                  # stories260K, first layer only
blaze integrate --model models/stories260K/stories260K.bin --blocks build/stories260K --json
```

- `--model`: the checkpoint the blocks came from (default stories260K).
- `--blocks`: the verified block files (default `build/<model file stem>`, where the block agent writes them). Each layer's 12 files are in `layer<N>/`. A flat directory still works, and if the new tree has no blocks the old `build/agent/<model file stem>/` layout is used.
- `--layers N`: build the first N layers, then the final norm and classifier (default: all layers).
- `--steps`: positions to run, 1..seq_len (default 32). Each position produces one output token. Position 0 reads BOS (token 1).
- `--prompt`: comma-separated token ids fed after BOS; position p reads BOS and the prompt while they last, then the chip's previous output token (greedy). A prompt longer than `--steps` - 1 is an error.
- `--out`: output directory (default `build/<model file stem>/chip`). It gets `blaze_chip.v`, the testbench `tb.cpp` and Verilator's `obj/`.
- `--no-sim`: write the Verilog and compute the reference tokens only.
- `--json`: print the summary as one JSON object. The exit code is 1 on a token mismatch.

The dsh tool `blaze_integrate` runs the same command (see [plugins/README.md](../plugins/README.md)).

## Blocks it uses

Every included layer needs all 12 files the block agent verifies, under `layer<N>/`: `l<N>_<tensor>_matvec.v` for `wq wk wv wo w1 w2 w3` and `l<N>_<op>.v` for `rmsnorm silu_mul softmax rope residual_add`. The same names flat in the blocks directory are accepted too. Missing files stop the command with the list of what's missing.

- **Matrices** go into the chip verbatim, at whatever bit width the agent chose. Each file has to equal what `gen-matrix` writes for this checkpoint at the bits in its `blaze-stats` line. That catches stale files and files from another checkpoint.
- **Ops**: each gen-op file embeds its template, so the files can't be concatenated. Instead, each file has to equal `gen-op`'s output. The chip instantiates the same templates with the same parameters for `rmsnorm`, `rope`, `silu_mul` and `residual_add`. The verified `l<N>_softmax.v` is required and byte-checked but not instantiated: `blaze_attention` adds the causal mask and reuses only the softmax's exp LUT and floor divide, so the chip's softmax path is covered by the full-chip comparison. The FFN norm and the final norm use the rmsnorm template with the `rms_ffn` and `rms_final` weights. RoPE for k uses the rope template with n = kv_dim.
- **New parts** are checked only by the full-chip simulation: `blaze_requant`, `blaze_attention` (causal softmax with the KV cache), the embedding ROM, the classifier (`gen-matrix` on the classifier weights at 8 bits, which is the token embedding when it's shared) and the control FSM.

## Numerics

The integer reference (`blaze.integrate.reference_generate`) and the Verilog compute the same integers.

- **int8 between blocks.** Every tensor passed between blocks is signed int8 with its own real scale, `real ~= v * scale`. Each scale is the tensor's peak |value| / 127, taken from a 64-step greedy float run of the same layers from BOS. The calibration run is fixed, so the same checkpoint and layer count always give the same chip. Two inputs have fixed formats because their templates need them: the silu gate (w1 output) has 4 fractional bits for the silu LUT, and attention scores have 4 fractional bits for the exp LUT.
- **Requantize** (`blaze_requant`): `y = clamp(floor((x * M + 2^(S-1)) / 2^S))`, which is x times a real factor, rounded half up and saturated. `M = round(factor * 2^S)` sits in [2^15, 2^16]. Every block output passes through one requantizer on its way to the next block's int8 input.
- **Residual stream**: one scale for the embedding and every layer. Each residual add is the `residual_add` template plus a factor-1 requant, which just saturates.
- **Attention** (`blaze_attention`, one per layer): k (after RoPE) and v are written to cache row `pos`. Head h reads kv head `h / (n_heads / n_kv_heads)`. Score = q.k / sqrt(head_size), as a 16-bit value with 4 fractional bits. Softmax runs over positions 0..pos, so causal masking comes from the loop bounds. It reuses the `softmax` block's exp LUT and floor divide (the block itself isn't instantiated), with `min(max - score, 255)` as the LUT index (entry 255 is 0). The output `sum(p * v) / 2^15` keeps v's scale.
- **Classifier**: integer logits, then a balanced argmax tree (depth log2 vocab) in which the first maximum wins a tie.
- RMSNorm drops eps, as in the op template.

The integer model can differ from the float model. The summary reports how many tokens match the float model's greedy tokens. That's information only, not a pass criterion.

## Control and cycles

`blaze_chip` has `clk`, `rst`, `start`, `token_in`, `pos_in`, `token_out`, `logit_out` and `done`. A `start` pulse loads the token's embedding row in one cycle. The FSM then steps through each layer's stages and finishes with the final norm and classifier, pulsing `done`:

`attn_norm -> wq/wk/wv -> rope (q, k) -> attention -> wo + residual -> ffn_norm -> w1/w3 -> silu_mul -> w2 + residual`

Each combinational stage holds for `ceil(depth / levels per cycle)` cycles. Depth uses the cost model from `estimate`, plus 5 levels per requant, and runs at 50 MHz SKY130 with 5 ns per adder level, so 4 levels fit in a cycle. Attention takes 1 cycle to write the cache (the cycle `start` is high), then (R1 + R2 + R3)(pos + 1) cycles for the three passes over the cache, then 1 more cycle in which the FSM sees `done`: (R1 + R2 + R3)(pos + 1) + 2 cycles in total. Each pass holds every cache row for ceil(row depth / levels per cycle) cycles, all heads in parallel. Row depths: scores (R1) = q.k multiply and sum + score requant + max compare, 3 + log2(head_size) + 5 + 1; exp and sum (R2) = subtract + add, 2; accumulate (R3) = subtract + divider (one level per quotient bit, 16) + p * v multiply + add, 1 + 16 + 3 + 1 = 21. For stories260K (head_size 8) that is 12, 2 and 21 levels, so R1, R2, R3 = 3, 1, 6, and the summary lists them as `attention_row_cycles`. The testbench feeds BOS, then the prompt tokens, then each output token back in as the next `token_in`, and counts cycles from `start` to `done`. Those counts are measured, not estimated. tok/s converts the mean at the assumed clock and is an estimate, as is the 5nm figure.

## stories260K, first layer

This run used the 12 layer-0 blocks, all of which pass `sim-block`, with `wv` at 7 bits, `w2` at 6 and the rest at 8. The prompt is the first 20 tokens the full 5-layer float model generates from BOS:

```powershell
blaze integrate --layers 1 --steps 48 --prompt 403,407,261,378,432,383,286,261,376,298,315,421,395,317,426,338,401,396,267,337
```

| | |
|---|---|
| result | **token-exact**: all 48 tokens and their logits equal the integer reference (Verilator 5.055) |
| float model | 48/48 tokens equal the float model's (information only) |
| text | `Once Once upon a time, there was a little girl named Lily. She loved to play play play ...` (one layer mostly echoes its input) |
| cycles per token | 80 at pos 0, +10 per position (attention), 550 at pos 47; mean 315 |
| stages | embed 1, attn_norm 13, wq/wk/wv 4, rope 3, attention 12 at pos 0 (R1, R2, R3 = 3, 1, 6), wo + residual 5, ffn_norm 13, w1/w3 4, silu_mul 2, w2 + residual 5, final_norm 13, classifier + argmax 5 |
| tok/s (estimate) | ~159,000 SKY130 at 50 MHz, ~1,590,000 5nm, from the mean over these 48 positions |

Each extra layer adds its own stages, about 61 cycles at pos 0 (layer 0's count; it depends on the layer's matrix depths), plus 10 cycles per position. The first Verilator build of the 1-layer chip takes about 5 minutes, mostly g++ on the classifier. A rerun with an unchanged chip skips the rebuild.
