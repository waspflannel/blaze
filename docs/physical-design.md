# Physical design

`blaze pd` takes one generated layer to GDSII with [OpenLane 2](https://openlane2.readthedocs.io) on SKY130
(`sky130A`, `sky130_fd_sc_hd`) and writes `pd.json` next to the run. None of this is a tapeout: see
[Foundry handoff gaps](#foundry-handoff-gaps).

```powershell
blaze pd build/stories260K/layer0 --dry-run   # wrapper + config.json only
blaze pd build/stories260K/layer0             # preflight, OpenLane run, pd.json
```

## Status today

On the development machine the toolchain isn't installed, so `blaze pd build/stories260K/layer0` returns
`status: "blocked"` and exits 1, naming `openlane`, `openroad`, `magic`, `netgen`, `klayout` and the `sky130A` PDK.
No area, timing, power, DRC or LVS number has been measured yet; every metric in that `pd.json` is `unavailable`.

## Setup (user-local only)

blaze installs nothing. OpenLane runs natively if `openlane` is on PATH, otherwise inside WSL on Windows (like
`sim-block`), with `~/tools/oss-cad-suite/bin` and `~/.local/bin` added to PATH. Inside WSL or Linux, without sudo:

- OpenLane 2 and its tools: follow OpenLane 2's documented install (upstream recommends Nix). `pip install openlane`
  alone gives the Python flow but not OpenROAD, Magic, Netgen or KLayout, which also have to be on PATH.
- PDK: `ciel` (or the older `volare`) fetches `sky130A` into `~/.ciel` (or `~/.volare`). blaze looks in `$PDK_ROOT`,
  `~/.ciel` and `~/.volare` for a `sky130A` directory with `libs.ref/sky130_fd_sc_hd`, and passes `--manual-pdk` so
  OpenLane never downloads a PDK itself.

`blaze pd` checks `openlane openroad magic netgen klayout yosys` and the PDK before anything heavy, and lists each
missing one with the command or path it checked.

## Resources

Preflight reads free RAM and CPU load on the Windows host (WMI) and in WSL (`/proc/meminfo`, `nproc`,
`/proc/loadavg`). The run is `deferred` (exit 1, numbers in `pd.json`) when the smaller free-memory figure is below
`--max-mem-gb` (default 8). The flow gets `-j` and `DRT_THREADS` from `--threads` (default min(4, cores / 2)) and
`ulimit -v` at `--max-mem-gb`, which caps address space, not resident memory.

A full stories260K layer is large for a laptop: about 41k shared adders after `optimize-matrix` plus 40,601 wrapper
flops, likely 10^5 to 10^6 standard cells. Detailed routing at that size can take hours and more than 15 GB. Running
from `/mnt/c` in WSL is also slow; an `--out` on the WSL filesystem helps. If a layer run is deferred or fails for
resources, `--block l0_wk_matvec.opt.v` runs one real block instead. Its `pd.json` has `"scope": "block-substitute"`
and the summary says "substitute, not a layer result". blaze refuses `--block` unless the layer's `pd.json` says
`deferred` or `failed`.

## The wrapper and why it exists

A layer's blocks are combinational, with about 13k input and 27k output bits: far more than any pin budget. The
generated top (`l<N>_pd_top`) has six pins: `clk rst_n si shift_en capture so`. A shift-in chain drives every block
input from registers, `capture` registers every block output, and a shift-out chain reads them out. All real block
logic stays observable (nothing is optimized away), timing is register to register through each block, and the pin
count is bounded. It's a test harness, not functional I/O. Its flop count is reported (`wrapper.wrapper_flops`) so
its area share is visible; it isn't subtracted from any area metric.

## pd.json

- `status`: `complete`, `blocked`, `deferred`, `failed` or `dry-run`. `scope`: `layer` or `block-substitute`. Also
  `blockers` (each `item` and what was `checked`), `reason`, `gds`, `run_dir` and `resources`.
- `metrics`: area (die, core, std-cell um^2, cell count, utilization), timing (WNS, TNS, clock period; worst of all
  STA corners), power (internal, switching, leakage, total; OpenSTA default activity, no VCD/SAIF), DRC (Magic and
  KLayout counts) and LVS (Netgen error count and `match`). Each is `{"value", "unit", "source", "from"}`, read from
  the run's `final/metrics.json` or `resolved.json`. A missing key is `unavailable` with the reason. blaze never
  computes or estimates a value into these fields.
- `provenance`: blaze git SHA and dirty flag, input files with sha256, model and layer, OpenLane version, PDK path
  and version hash, std-cell library, tool versions, threads and memory cap, start and end time, command, host OS.

OpenLane metric names change between versions. The mapping is `METRICS` in `src/blaze/pd/__init__.py`; check it
against the installed OpenLane's `metrics.json` on the first real run.

## Limitations

- One flat macro per layer: no hierarchy, no hard macros, and OpenLane's default floorplan with `FP_CORE_UTIL` 35,
  `PL_TARGET_DENSITY_PCT` 45 and a 20 ns clock.
- Weights are hard-wired into the logic, so area grows with the weight count.
- Power is vectorless. Timing is setup only, at the worst corner OpenLane reports.

## Path to the full model

- One hardened macro per layer (or per block), then a top-level floorplan placing the layer macros, embedding ROM,
  classifier, KV caches and control, using OpenLane's macro flow.
- Hierarchy kept through synthesis, so each matvec is timed and placed as a unit.
- Weights: hard-wired logic (now) or SRAM/ROM macros with a shared multiplier array. The second trades area for
  cycles and is the only way to models much bigger than stories260K.
- The KV caches need SRAM macros (OpenRAM or a foundry compiler), not flops.

## Foundry handoff gaps

GDSII from this flow alone is not manufacturer-ready. A real tapeout also needs:

- I/O pads and a padframe, ESD protection, a seal ring, and a harness (for example Caravel for SKY130 shuttles).
- Chip-level power delivery: a power grid sized for the real current, with IR-drop and electromigration sign-off.
- Fill, density and antenna fixes to the foundry's rules at chip level.
- Multi-corner, multi-mode timing sign-off (all PVT corners, hold, signal integrity), not one worst-corner setup
  number.
- A clean run of the foundry's own DRC/LVS sign-off decks, beyond OpenLane's Magic, KLayout and Netgen checks.
- Functional sign-off of the chip with its real interface; the wrapper here is only a test harness.
