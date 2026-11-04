# Model scanner

`blaze scan-model` reads a checkpoint and writes a model spec: the config plus every op of one forward pass, in order. It is plain deterministic Python with no LLM, so the same checkpoint always gives a byte-identical `spec.json`.

```powershell
blaze scan-model models/stories260K/stories260K.bin   # -> build/stories260K/spec.json
```

`--out` changes the output root (default `build/`). The spec goes to `<out>/<file stem>/spec.json`. The command prints the layer count, the total params and one row per tensor kind with its shape.

## Inputs

Only llama2.c `.bin` checkpoints (Karpathy's legacy export) are read for now. Code is in `blaze.spec.scan`:

- `read_checkpoint(path)` returns `(Config, tensors)`. `tensors` maps a name like `token_embedding` or `layers.3.wq` to a `Tensor` that holds the shape, dtype and byte offset of its data in the file.
- `build_spec(path, cfg, tensors)` turns that into the spec. It never parses the file itself.

A config.json + safetensors reader only has to return the same `(Config, tensors)` and be added to `read_checkpoint`. The safetensors header already gives each tensor's dtype, shape and byte offsets.

## Spec

- `model`: the file stem. `architecture`: `llama`.
- `source`: file name, format, size in bytes and sha256.
- `config`: `dim`, `hidden_dim`, `n_layers`, `n_heads`, `n_kv_heads`, `head_size`, `kv_dim`, `vocab_size`, `seq_len`, `shared_classifier`.
- `params`: total weights, not counting `freq_cis_real`/`freq_cis_imag`. Legacy exports store those RoPE tables, but `run.c` ignores them and computes RoPE from theta = 10000.
- `ops`: one forward pass, in execution order:
  1. `embedding` (token table lookup)
  2. for each layer: `attn_norm` (rmsnorm) -> `wq`, `wk`, `wv` -> `rope` -> `attention` -> `wo` -> `attn_residual` -> `ffn_norm` (rmsnorm) -> `w1`, `w3` -> `silu_gate` (`silu(w1 x) * (w3 x)`) -> `w2` -> `ffn_residual`
  3. `final_norm` (rmsnorm) -> `classifier`

Each op has `name`, `op`, `template` and `layer` (`null` outside the layers).

- **Matmul ops** (`template: false`) have `tensor`, `shape` as `[out, in]`, `dtype`, `offset` (bytes from the file start) and `stats`: `min`, `max`, `mean_abs` and `pct_zero` (percent of weights that are exactly 0). When the classifier is shared, it points at `token_embedding`.
- **Template ops** (`template: true`) will be built from fixed hardware templates instead of generated from weights. Their `params` hold what the template needs. `rmsnorm` has `dim`, `eps` = 1e-5 and its `weight` tensor reference. `rope` has `head_size`, `theta` and `applies_to`. `attention` has the head counts, `score_scale` = 1/sqrt(head_size), the causal softmax and the KV cache shapes (`[seq_len, kv_dim]` each for k and v, float32).

stories260K gives 73 ops: 36 matmuls and 37 template ops.
