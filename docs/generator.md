# Matrix generator

`blaze gen-matrix` turns one weight matrix into hardwired shift/add equations and a combinational Verilog module. It is plain deterministic Python with no LLM, so the same checkpoint always gives byte-identical files.

```powershell
blaze gen-matrix --layer 0 --tensor wq --out build/stories260K
```

This writes `build/stories260K/layer0/l0_wq_matvec.v` and `build/stories260K/layer0/l0_wq_matvec.txt`, then prints the matrix shape, the nonzero weight count and the total adders. `--out` is the model root; the layer directory is created under it. `--tensor` accepts `wq wk wv wo w1 w2 w3`, and `--model` defaults to `models/stories260K/stories260K.bin`.

## Flow

1. **Load** (`blaze.spec.load_matrix`): read the 7-int llama2.c header (dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len), check that the file size matches the layout that header implies, then slice out one layer's float32 matrix as `(out, in)`.
2. **Quantize** (`blaze.generate.quantize`): use one symmetric scale per matrix, `scale = max|w| / qmax` and `q = round(w / scale)` (ties round to even), clipped to `-qmax..qmax`. `qmax = 2^(bits-1) - 1`, so the default `--bits 8` gives -127..127 and `--bits 4` gives -7..7. The real weight is `w ~= scale * q`, and the scale is written into both output files. The summary prints the relative RMS weight error, `rms(w - scale * q) / rms(w)`.
3. **Expand** (`blaze.generate.csd`): write each nonzero `q` in canonical signed digit form, a sum of `+-(x << k)` terms with no two adjacent nonzero digits. For example 7 = 8 - 1.
4. **Emit** (`blaze.generate.generate_matrix`): build one equation per output row, `y_i = sum_j q_ij * x_j`, skipping zero weights. It goes out as a text dump and as one Verilog module.
5. **Reference** (`blaze.reference.matvec_int`): the exact integer `q @ x`. Generated hardware has to match it bit for bit.

### Numbers

- **Inputs** `x[j]` are signed 8-bit (-128..127). The activation scale is chosen by whatever drives them, so `y_real ~= scale_w * scale_x * y`.
- **Outputs** `y[i]` are signed `ACC_W`-bit, with `ACC_W = bit_length(128 * max_i sum_j |q_ij|) + 1`. No input can produce a magnitude above `128 * sum_j |q_ij|`, so the sum is exact and can never overflow. This is computed from the actual weights; the shape-only worst case would be `8 + 8 + ceil(log2(in))` bits.
- **Adders** counts two-input add/subtract operations, which is (CSD terms - 1) for each non-empty row. A leading minus is a negation and isn't counted. gen-matrix shares no subexpressions; `optimize-matrix` does ([optimizer.md](optimizer.md)).

### Verilog shape

The module is called `l<layer>_<tensor>_matvec`, and the file stem matches. It's purely combinational, with packed ports so it stays plain Verilog-2005:

- `input wire [8*in-1:0] x`, where element `j` is `x[8*j+7:8*j]`
- `output wire [ACC_W*out-1:0] y`, where element `i` is `y[ACC_W*i+ACC_W-1:ACC_W*i]`

Each input is sign-extended to `ACC_W` bits in its own `wire signed` before any shift, so every operand in every sum is the same width and signed.

## 2x2 example

Start with `W = [[0.5, -0.25], [0.0, 0.875]]`.

- Scale: `0.875 / 127 = 0.0068898`, so `q = [[73, -36], [0, 127]]`.
- CSD: `73 = 64 + 8 + 1`, `-36 = -32 - 4`, `127 = 128 - 1`.
- Width: the largest row abs-sum is 127, and `bit_length(128 * 127 = 16256) + 1 = 15` bits.
- Cost: 3 nonzero weights and 7 CSD terms, so (5 - 1) + (2 - 1) = 5 adders.

The equation dump:

```text
y[0] = 73*x[0] - 36*x[1]
     = (x[0] << 6) + (x[0] << 3) + x[0] - (x[1] << 5) - (x[1] << 2)
y[1] = 127*x[1]
     = (x[1] << 7) - x[1]
```

The Verilog, with its header comments trimmed:

```verilog
module demo_matvec (
    input  wire [15:0] x,
    output wire [29:0] y
);
    wire signed [14:0] x_0 = $signed(x[7:0]);
    wire signed [14:0] x_1 = $signed(x[15:8]);

    assign y[14:0] = (x_0 <<< 6) + (x_0 <<< 3) + x_0 - (x_1 <<< 5) - (x_1 <<< 2);
    assign y[29:15] = (x_1 <<< 7) - x_1;
endmodule
```

Check it with `x = [3, -2]`: `matvec_int(q, x)` gives `[73*3 + 36*2, 127*-2] = [291, -254]`, and the module has to output exactly those two 15-bit values.
