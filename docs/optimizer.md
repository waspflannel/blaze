# Matrix optimizer

`blaze optimize-matrix` takes a gen-matrix block and writes the same `y = W x` with fewer adders. It is plain deterministic Python with no LLM, so the same input always gives a byte-identical file.

```powershell
blaze optimize-matrix build/stories260K/layer0/l0_wq_matvec.v   # -> build/stories260K/layer0/l0_wq_matvec.opt.v
```

The input has to be exactly what gen-matrix wrote. The command reads the layer and tensor from the header and the bits from the `// blaze-stats:` line, quantizes the matrix from `--model` again (default stories260K), and checks that gen-matrix would write the same bytes. Then it builds the optimized module from those quantized weights. The output keeps the module name and ports, so it's a drop-in replacement: `sim-block` checks it against the same `--ref`, and `integrate` uses it in place of the `.v` while it exists.

## Algorithm

gen-matrix writes every row as a sum of signed, shifted inputs, one per CSD digit, e.g. `(x_0 <<< 2) + x_0 + (x_1 <<< 2) - x_1`. Rows of the same matrix often contain the same two inputs at the same distance apart. That's common subexpression elimination over those terms (`blaze.optimize.share`):

1. **Terms.** Each row is a set of `sign * (signal <<< shift)` terms. At the start the signals are the inputs `x_j`.
2. **Pairs.** Any two terms of a row form a pair, normalized so the lower term (by signal, then shift) has sign `+` and the smaller shift is 0. `(x_0 <<< 3) + (x_1 <<< 3)` and `-x_0 - x_1` are both the pair `x_0 + x_1`, at shift 3 and shift 0 with sign `-`. Every pair in every row is counted.
3. **Share.** Take the most frequent pair (ties go to the smallest pair, so the result is deterministic). If it occurs at least twice without two copies sharing a term, it becomes a new wire `c_n`. Each occurrence's two terms are replaced by one term, `sign * (c_n <<< shift)`.
4. **Repeat** until no pair occurs twice. Wires are signals too, so later wires can contain earlier ones.

Every wire costs 1 adder and saves 1 adder in each row that uses it, so each step saves at least 1 adder. Pair counts are updated incrementally as terms are replaced, which keeps the run to seconds for stories260K's largest matrices.

### Numbers

- **Exact.** Every wire and row is `ACC_W` bits wide, the same signed width gen-matrix uses. Shifts and adds are exact modulo `2^ACC_W`, and the true output always fits in `ACC_W` bits, so every output equals gen-matrix's bit for bit even if an intermediate wire wraps.
- **Adders** are one per wire plus (terms - 1) per non-empty row. A leading minus is a negation and isn't counted, as in gen-matrix.
- **Depth** is the longest adder chain: an input is ready at level 0, a wire one level after its later operand, and each row is summed as a tree that always adds its two earliest-ready terms first.
- The header's `// blaze-stats:` line is `{"adders": ..., "depth": ..., "bits": ..., "optimized": true, "original_adders": ...}`. `estimate` reads it like any other block.

### Limits

Greedy pair sharing is the simple version. It doesn't look ahead, so a different order could share more. A pair whose copies overlap inside one row (`x`, `x <<< 2`, `x <<< 4`) and that has no second disjoint copy is dropped for good, even if later sharing would free one. Depth isn't bounded: a row that sums deeply nested wires could get deeper than gen-matrix's, and the summary prints both depths.

## Example

Start with `q = [[5, 3], [10, 6], [7, -1]]`. gen-matrix writes 8 adders:

```verilog
    assign y[12:0] = (x_0 <<< 2) + x_0 + (x_1 <<< 2) - x_1;
    assign y[25:13] = (x_0 <<< 3) + (x_0 <<< 1) + (x_1 <<< 3) - (x_1 <<< 1);
    assign y[38:26] = (x_0 <<< 3) - x_0 - x_1;
```

1. `x_0 + x_1` occurs 3 times: shift 2 in row 0, shift 3 in row 1, and negated in row 2. It becomes `c_0`.
2. `x_0 - x_1` now occurs twice: shift 0 in row 0, shift 1 in row 1. It becomes `c_1`.
3. Row 0 is `(c_0 <<< 2) + c_1` and row 1 is `(c_0 <<< 3) + (c_1 <<< 1)`. That's the same pair at shifts 0 and 1, so it becomes `c_2`.
4. No pair repeats, so it stops.

The optimized body has 3 wires and 1 row adder, 4 adders in all, at depth 2:

```verilog
    wire signed [12:0] c_0 = x_0 + x_1;
    wire signed [12:0] c_1 = x_0 - x_1;
    wire signed [12:0] c_2 = (c_0 <<< 2) + c_1;

    assign y[12:0] = c_2;
    assign y[25:13] = (c_2 <<< 1);
    assign y[38:26] = (x_0 <<< 3) - c_0;
```

Check it with `x = [3, -2]`: `c_0 = 1`, `c_1 = 5`, `c_2 = 9`, so `y = [9, 18, 23]`, the same as `q @ x`.

## stories260K layer 0

8-bit weights, then `sim-block` on each `.opt.v` (1000 vectors, seed 0, Verilator 5.055): all 7 match.

| matrix | shape | adders before | adders after | shared wires | depth |
|---|---|---|---|---|---|
| wq | 64 x 64 | 7,717 | 3,561 | 986 | 8 -> 8 |
| wk | 32 x 64 | 4,241 | 2,054 | 624 | 8 -> 8 |
| wv | 32 x 64 | 4,157 | 2,049 | 620 | 8 -> 8 |
| wo | 64 x 64 | 7,441 | 3,529 | 967 | 8 -> 8 |
| w1 | 172 x 64 | 22,961 | 9,782 | 2,322 | 8 -> 8 |
| w2 | 64 x 172 | 22,869 | 9,926 | 3,198 | 9 -> 9 |
| w3 | 172 x 64 | 23,861 | 10,053 | 2,409 | 8 -> 8 |
| layer 0 | | 93,247 | 40,954 | 11,126 | |

`blaze integrate --layers 1 --steps 32` with all 7 optimized files is token-exact against the integer reference.
