# FPGA export

`blaze fpga` is the last step of the flow: weights -> verified blocks -> integrated chip -> **one bitstream file** for one FPGA board. It runs the free, open-source flow Yosys -> nextpnr -> ecppack on a generated Verilog top and records what it did.

> **Status: source only.** Neither this command nor the `blaze_fpga_export` tool has been run. No synthesis, place and route or bitstream was produced while writing them. No automated tests exist, and nothing has been loaded onto a board. The serial wrapper has never been simulated. Treat this whole page as unvalidated until someone runs the flow end to end.

Weights alone don't make a bitstream. A bitstream belongs to one generated top, one FPGA part and one board's pinout, so the target and the board constraints are always explicit inputs. The export claims no support for arbitrary models.

## Target: `ecp5-85f`

The only target is the Lattice ECP5 **LFE5U-85F-6BG381C**, the part on the [ULX3S](https://github.com/emard/ulx3s) 85F board. blaze runs:

```text
yosys -q -p "read_verilog <stem>.fpga.v; synth_ecp5 -top <top> -json <stem>.json"
nextpnr-ecp5 --85k --package CABGA381 --speed 6 --json <stem>.json --lpf <stem>.lpf --textcfg <stem>.config --report <stem>.nextpnr.json
ecppack <stem>.config <stem>.bit
```

These flags were checked against the upstream sources (nextpnr `ecp5/main.cc` and `common/kernel/command.cc`, prjtrellis `ecppack.cpp`, and the Yosys `synth_ecp5` docs). They have not been run here.

### Will it fit?

The ECP5-85F has about 84K LUTs. The whole stories260K chip almost certainly doesn't fit. Layer 0's seven matrices alone are 40,954 adders after `optimize-matrix`, each many bits wide, and the chip adds the embedding ROM, the KV caches and the classifier on top. Expect nextpnr to fail placement on the full chip; blaze passes its error through. One matrix block is the realistic first fit, possibly at fewer bits (`gen-matrix --bits`). Even that is for nextpnr's utilization report to decide.

## Board constraints (required)

blaze has no pinout and never guesses pins. Pass your board's LPF with `--lpf`. It needs one `LOCATE COMP "<port>" SITE "<ball>";` line for every bit of the top's ports, with bus bits written `name[i]`. Add a `FREQUENCY PORT "clk" <n> MHZ;` line if you want nextpnr to time against your clock; without one it targets 12 MHz.

For the ULX3S, take the sites from the board's published constraints, [`doc/constraints/ulx3s_v20.lpf`](https://github.com/emard/ulx3s/blob/master/doc/constraints/ulx3s_v20.lpf), and rename each `COMP` to the blaze port it drives. The board's 25 MHz oscillator is the `clk_25mhz` entry in that file, so its `LOCATE` and `FREQUENCY` lines get renamed to `"clk"`. You choose the header or button/LED pins for the other ports; blaze only checks that every port bit has a location.

## Which top

- **The chip**, `build/<model>/chip/blaze_chip.v` from `blaze integrate`. Its ports (`clk`, `rst`, `start`, `token_in`, `pos_in`, `token_out`, `logit_out`, `done`) total a few dozen bits, so you can pin it directly. Its stages hold their inputs for several cycles, which nextpnr doesn't know. The Fmax it reports is therefore the single-cycle bound, a conservative figure. The chip's clocking has been checked only in Verilator, never on an FPGA.
- **One matrix block** (`.v` or `.opt.v`). Its buses are hundreds to thousands of bits wide (`x` is 512 bits on stories260K), far more than the board has pins, so on its own it's rejected as UNSUPPORTED. `--wrap serial` adds `blaze_fpga_serial`, a top with 5 pins: `clk`, `shift_in`, `shift_en`, `capture` and `shift_out`. Shift the inputs in, wait for the block to settle, pulse `capture`, then shift the outputs out MSB first. **The wrapper is unvalidated** and is meant for fit and timing experiments; nothing drives it yet. blaze refuses it for tops that have a `clk` port.

## Run it

Prerequisites are Yosys, nextpnr-ecp5 and ecppack, all included in [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build); blaze never installs them. Put the suite's `bin/` on PATH. On Windows, when the tools aren't on the Windows PATH, blaze runs them inside WSL from `~/tools/oss-cad-suite/bin`, the same install the README's Simulation setup describes for Verilator. Loading the bitstream onto the board (e.g. with openFPGALoader) is outside blaze.

```powershell
blaze fpga build/stories260K/chip/blaze_chip.v --target ecp5-85f --lpf boards/my-ulx3s.lpf --model models/stories260K/stories260K.bin
blaze fpga build/stories260K/layer0/l0_wq_matvec.opt.v --target ecp5-85f --lpf boards/my-ulx3s-serial.lpf --wrap serial --json
```

- `--target` (required): `ecp5-85f`.
- `--lpf` (required): your constraints, copied into the output directory as `<stem>.lpf`.
- `--top`: the top module (default: the file name without `.v` or `.opt.v`).
- `--wrap serial`: see above.
- `--model`: the checkpoint the Verilog came from, recorded in the report with its sha256.
- `--out`: output directory (default: `fpga/` next to the file).
- `--json`: print the report.

From the agent: the block agent ([block-agent.md](block-agent.md)) exports only when the task names the target and gives an LPF, e.g. `Use the block-agent skill on models/stories260K/stories260K.bin, layer 0 only, then export l0 wq to ecp5-85f with boards/my-ulx3s-serial.lpf.` It calls the `blaze_fpga_export` dsh tool after the token-exact `integrate` check (for the chip) or the matching `sim-block` (for a block). That check is still the correctness proof; the export adds no verification of its own.

## Outcomes

A run ends in one of four outcomes, and only a packed bitstream counts as success:

- **absent**: the input Verilog, the LPF or the `--model` file doesn't exist, or a tool isn't installed. The message points at OSS CAD Suite.
- **rejected**: an unknown target, `--wrap serial` on a clocked top, or `UNSUPPORTED: k of n port bits of <top> have no LOCATE in <lpf>`. Nothing is written.
- **failed**: Yosys, nextpnr (for example, when the design doesn't fit) or ecppack exited nonzero, or ecppack wrote no file. The error shows the last lines of the tool's output. blaze deletes stale outputs, including the old report, before the run, so an old `.bit` never counts.
- **success**: `<stem>.bit` exists and isn't empty. `<stem>.fpga.json` then records:
  - the bitstream path, target and device, the top and the wrapper
  - the sha256 of the input Verilog, the LPF and the checkpoint (`unavailable` without `--model`)
  - the tool versions
  - Fmax and utilization from nextpnr's report, each `unavailable` if the report lacks it

`<stem>` is the top module's name, plus `.serial` when wrapped (`l0_wq_matvec.serial.bit`), so wrapped blocks don't overwrite each other. The output directory also keeps `<stem>.fpga.v` (the synthesized source, including the wrapper if one was added), the Yosys netlist, the nextpnr text config and nextpnr's report.
