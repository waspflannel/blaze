# Blaze (working title)

Weights-to-silicon compiler: takes open model weights and compiles them into a fused hardware design (Verilog -> GDSII) with the weights burned into the circuits, ready to export to a manufacturer. We stop before sending to a fab.

- Free toolchain: Verilator, Yosys, OpenROAD/OpenLane, SkyWater 130nm open PDK
- Token-exact simulation equivalence vs. the reference model (core product)
- Reports: cycle-accurate tokens/sec (projected to 5nm), area, power

See [docs/design.md](docs/design.md). Status: v0 design draft, not building yet.
