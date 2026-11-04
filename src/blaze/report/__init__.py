"""Estimates: adder count, adder chain depth and a rough tokens/sec projection. Not synthesis results.

Timing model (all assumptions, override with the CLI flags):
- SKY130 clock SKY130_CLOCK_MHZ, and one adder level costs ADDER_NS, so floor(period / ADDER_NS)
  chained adder levels fit in one cycle.
- Each op is a combinational block with registered outputs: ceil(depth / levels per cycle) cycles,
  at least 1, plus any streaming cycles (attention reads the KV cache one row per cycle).
- Ops run one after another, so cycles per token is the sum over the spec's ops.
- 5nm: the same cycle count at SCALE_5NM times the clock.
"""
import hashlib
import json
import math
import re
from pathlib import Path

from blaze.generate import matrix_cost, quantize
from blaze.library import spec_op_cost
from blaze.spec import Config, Tensor

SKY130_CLOCK_MHZ = 50.0  # typical OpenLane/Caravel target for sky130_fd_sc_hd designs
ADDER_NS = 5.0  # one ~20-bit add level in sky130_fd_sc_hd, routing included
SCALE_5NM = 10.0  # gate delay ~45 ps at 130 nm vs ~5 ps at 5 nm


def _levels(clock_mhz: float, adder_ns: float) -> int:
    levels = math.floor(1000.0 / clock_mhz / adder_ns)
    if levels < 1:
        raise ValueError(f"no adder level fits a {clock_mhz} MHz cycle at {adder_ns} ns per level")
    return levels


def _cycles(cost: dict, levels: int) -> int:
    return max(1, math.ceil(cost["depth"] / levels)) + cost.get("stream", 0)


def _assumptions(clock_mhz: float, adder_ns: float, levels: int) -> dict:
    return {"sky130_clock_mhz": clock_mhz, "adder_ns": adder_ns, "levels_per_cycle": levels, "scale_5nm": SCALE_5NM}


def estimate_verilog(path: Path, clock_mhz: float, adder_ns: float) -> dict:
    """One block, from the `// blaze-stats:` line that gen-matrix and gen-op write."""
    m = re.search(r"^// blaze-stats: (.*)$", path.read_text(), re.M)
    if not m:
        raise ValueError(f"{path}: no '// blaze-stats:' line; regenerate it with gen-matrix or gen-op")
    cost = json.loads(m[1])
    levels = _levels(clock_mhz, adder_ns)
    cycles = _cycles(cost, levels)
    rate = clock_mhz * 1e6 / cycles
    return {"estimate": True, "file": str(path), "adders": cost["adders"], "depth": cost["depth"],
            "cycles": cycles, "evals_per_s": {"sky130": rate, "5nm": rate * SCALE_5NM},
            "assumptions": _assumptions(clock_mhz, adder_ns, levels)}


def estimate_spec(spec_path: Path, checkpoint: Path | None, clock_mhz: float, adder_ns: float) -> dict:
    """Whole model from a scan-model spec; matmul costs come from the quantized checkpoint weights."""
    spec = json.loads(spec_path.read_text())
    checkpoint = checkpoint or Path("models") / spec["model"] / spec["source"]["file"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != spec["source"]["sha256"]:
        raise ValueError(f"{checkpoint} does not match the spec's sha256; pass the right --model")
    c = spec["config"]
    cfg = Config(c["dim"], c["hidden_dim"], c["n_layers"], c["n_heads"], c["n_kv_heads"], c["vocab_size"],
                 c["seq_len"], c["shared_classifier"])
    levels = _levels(clock_mhz, adder_ns)

    kinds: dict[str, dict] = {}
    deepest = {"op": None, "depth": -1}
    for op in spec["ops"]:
        if op["op"] == "matmul":
            w = Tensor(checkpoint, tuple(op["shape"]), op["dtype"], op["offset"]).load()
            cost = {**matrix_cost(quantize(w)[0]), "stream": 0}
        else:
            cost = spec_op_cost(op, cfg)
        k = kinds.setdefault(op["op"], {"count": 0, "adders": 0, "max_depth": 0, "cycles": 0})
        k["count"] += 1
        k["adders"] += cost["adders"]
        k["max_depth"] = max(k["max_depth"], cost["depth"])
        k["cycles"] += _cycles(cost, levels)
        if cost["depth"] > deepest["depth"]:
            deepest = {"op": op["name"], "depth": cost["depth"]}

    cycles = sum(k["cycles"] for k in kinds.values())
    tok_s = clock_mhz * 1e6 / cycles
    return {"estimate": True, "file": str(spec_path), "model": spec["model"],
            "adders": sum(k["adders"] for k in kinds.values()), "depth": deepest["depth"],
            "deepest_op": deepest["op"], "cycles_per_token": cycles, "context": cfg.seq_len,
            "tok_per_s": {"sky130": tok_s, "5nm": tok_s * SCALE_5NM}, "by_op": kinds,
            "assumptions": _assumptions(clock_mhz, adder_ns, levels)}
