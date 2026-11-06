# Circuit viewer

`blaze view` shows what a generated matrix block computes, for a person to inspect. It reads the block's Verilog only, so it works on any block that `gen-matrix` (or an optimizer that keeps the same module and ports) wrote. Nothing in it changes a block or a pass criterion.

## Adder graph page

```powershell
blaze view build/stories260K/layer0/l0_wq_matvec.v        # -> build/stories260K/layer0/view/l0_wq_matvec.html
blaze view build/stories260K/layer0/l0_wq_matvec.opt.v    # -> .../view/l0_wq_matvec.opt.html
```

The parser reads every `wire ... = expr;` and `assign` statement. Expressions can use n-ary `+`/`-`, unary `-`, constant shifts (`<<`, `<<<`, `>>`, `>>>`), `$signed()`, parentheses, nets, input port bit-selects and integer constants. Anything else stops with an error naming the statement. The graph has five kinds of node:

- **input**: an input slice, named after the wire that sign-extends it (`x_3`).
- **shift**: a constant shift. It's wiring and costs no adder.
- **add/sub**: one sum of n signed terms, n - 1 adders. A plain block has one per row. An optimized block's shared intermediate wires are add/sub nodes too.
- **output**: one output row (`y[12]`).
- **constant**.

Identical subexpressions are one node, so `x_3 << 2` is drawn once however many rows use it. A node's depth is the fewest adder levels that can compute it as a balanced tree over its inputs' depths. On a plain block, the page's adders and max depth equal the file's `blaze-stats` line.

The page is one HTML file with the graph inlined as JSON and no network access needed. It opens as an overview:

- The header has inputs, outputs, adders, **shared adders** (adders in nodes that feed more than one output), max depth, shared intermediate wires and graph nodes.
- The left table has one line per output row: adders in its fan-in cone, shared adders, depth and cone size. Click a column to sort it. The search box takes a row number or `y[12]`, and Enter opens the first match.
- Click a row to draw its fan-in cone: inputs on the left, the output on the right, subtracted terms as dashed red edges. Click more rows to add them. Nodes they share are drawn once, and the last row clicked is highlighted while the rest dim. Click the row again to close it, or **Clear** to close all.
- Nodes used by more than one output have an orange outline and a `x<count>` badge.
- Drag to pan, scroll to zoom, **Fit** to frame the graph. Click a node for its kind, depth, inputs and the outputs that use it.

Only the rows you open are drawn, so a full layer-0 matrix stays responsive. Drawing more than 6,000 nodes at once asks first.

## Row gate schematic

```powershell
blaze view build/stories260K/layer0/l0_wq_matvec.v --row 12 --schematic
# -> build/stories260K/layer0/view/l0_wq_matvec.row12.v     the cone as a standalone module
#    build/stories260K/layer0/view/l0_wq_matvec.row12.json  yosys netlist
#    build/stories260K/layer0/view/l0_wq_matvec.row12.svg   netlistsvg drawing
```

This copies the input port and only the statements row 12 depends on into a module `<module>_row12` with one output `y`. Yosys synthesizes it to generic gates (`synth -flatten -noabc; opt_clean -purge; write_json`) and netlistsvg draws the JSON as SVG. ABC is skipped because it took over 3 minutes on one stories260K row, while the plain mapping takes under a second.

The page can't run tools, so a row's details panel shows this command and links the SVG at its default path. Render it from the CLI first, then reload the page.

Setup:

- **Yosys** is in OSS CAD Suite. blaze runs `yosys` from PATH, or on Windows from `~/tools/oss-cad-suite/bin` in WSL, the same install as Verilator in the [README](../README.md#simulation-setup-verilator).
- **netlistsvg** is an npm package that needs Node.js: `npm install -g netlistsvg`, or `npm install netlistsvg` in any folder with that folder's `node_modules/.bin` on PATH.

If either is missing, the command stops with an install message before writing anything.

## stories260K layer 0 wq

From `blaze gen-matrix --layer 0 --tensor wq` at 8 bits (64 x 64, 3,961 nonzero weights):

| | |
|---|---|
| inputs / outputs | 64 / 64 |
| adders | 7,717 (equal to `blaze-stats`) |
| shared adders | 0 (a plain block shares only shifts) |
| max depth | 8 (equal to `blaze-stats`) |
| graph nodes | 582 (64 inputs, 390 shifts, 64 add/sub, 64 outputs) |
| HTML | 73 KB |
| row 12 cone | 144 nodes, 117 adders, depth 7 |
| row 12 gates | 7,753 (2,932 AND, 421 NOT, 1,469 OR, 2,931 XOR); about 12 s for yosys + netlistsvg, 8 MB SVG |

The page inlines the whole graph, which suits stories260K-sized blocks. Blocks with millions of terms would need the graph loaded per row instead.
