# Blaze v0 design draft

## Goal
Open weights in (Hugging Face), manufacturing-ready chip package out: GDSII + timing/area/power reports + verification evidence. Stop before fab submission.

## Pipeline
1. **Analyze** model architecture (layers, attention type, dense vs MoE).
2. **Quantize** per layer (2-4 bit), chosen by accuracy testing against the reference model.
3. **Plan** (LLM lead agent): map each layer/block to proven hardware templates (constant-weight multiply arrays, attention, norms, activations), pipeline layers as stages.
4. **Swarm**: dispatch one agent per layer/block to generate Verilog, simulate it with Verilator against reference activations, and iterate until it matches. No piece is accepted until its simulation matches.
5. **Integrate**: final agent stitches blocks, simulates the full chip token-for-token vs the reference model.
6. **Physical design**: Yosys synthesis + OpenROAD/OpenLane place-and-route on SKY130 -> GDSII, timing, area, power.
7. **Report**: cycle-accurate tokens/sec at achieved clock, projection to 5nm, area, power, estimated cost.
8. **Export** manufacturing package (no fab submission).

## Verification
Token-exact simulation equivalence vs the reference model is a core product feature (correctness verification), not optional automated tests.

## Targets
1. Tiny transformer (Karpathy stories260K).
2. Qwen3-0.6B (dense, single chip).
3. Later: Qwen3-32B (multi-chip partitioning + inter-chip links).

## Open questions
Templates library, quantization scheme, how weights are fused (constant logic vs ROM), multi-chip interconnect, LLM provider for agents.
