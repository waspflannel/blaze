"""Fixed-point op templates (Verilog in verilog/) and their bit-exact Python integer references.

Every op input is int8 with IN_FRAC fractional bits (real = v / 16). Each builder returns a Block:
its ports, an integer reference the Verilog must match bit for bit, a float reference to measure
fixed-point error against, and a cost estimate. Deterministic; no LLM involved.

Cost is in adder-equivalents (two-input add/subtract/compare) with the depth of the longest
chain, assuming balanced trees: an a x b multiply is min(a, b) - 1 adders, depth ceil(log2 min);
a divider or square root producing q bits is q adders in series; lookups are free.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from blaze.generate import quantize
from blaze.spec import Config, read_llama2c

VERILOG_DIR = Path(__file__).with_name("verilog")
IN_FRAC = 4
RMS_K = 8  # fractional bits of the rms estimate
RMS_FO = 8  # rmsnorm output fractional bits
SILU_SW = 12  # silu LUT width; values carry 8 fractional bits
EXP_FRAC = 15  # softmax exp LUT and output fractional bits
ROPE_CW = 16  # cos/sin width; values carry 14 fractional bits
ROPE_FRAC = 14


@dataclass(frozen=True)
class Port:
    name: str
    count: int
    bits: int
    lo: int = -128
    hi: int = 127

    @property
    def width(self) -> int:
        return self.count * self.bits


@dataclass
class Block:
    inputs: list[Port]
    output: Port  # elements are always signed
    reference: Callable[[dict[str, np.ndarray]], np.ndarray]  # one input vector -> exact ints
    real: Callable[[dict[str, np.ndarray]], np.ndarray] | None = None  # float reference, real units
    out_scale: float = 1.0  # y_real ~= y * out_scale
    cost: dict | None = None
    verilog: str = ""
    formats: str = ""


def random_inputs(ports: list[Port], n: int, seed: int) -> list[dict[str, np.ndarray]]:
    """n seeded input vectors; the first three are all-lo, all-hi and all-zero (clipped) corners."""
    rng = np.random.default_rng(seed)
    cols = {p.name: rng.integers(p.lo, p.hi + 1, size=(n, p.count), dtype=np.int64) for p in ports}
    for p in ports:
        for row, v in enumerate([p.lo, p.hi, min(max(0, p.lo), p.hi)][:n]):
            cols[p.name][row] = v
    return [{name: c[i] for name, c in cols.items()} for i in range(n)]


def measure_error(block: Block, n: int = 1000, seed: int = 0) -> dict:
    """Max |y * out_scale - float reference| over n seeded random vectors."""
    err = peak = 0.0
    for v in random_inputs(block.inputs, n, seed):
        f = block.real(v)
        err = max(err, float(np.max(np.abs(block.reference(v) * block.out_scale - f))))
        peak = max(peak, float(np.max(np.abs(f))))
    return {"max_abs_err": err, "max_abs_float": peak, "vectors": n, "seed": seed}


# --- cost model -----------------------------------------------------------------------------------

def _clog2(n: int) -> int:
    return max(n - 1, 0).bit_length()


def _mul(bits: int) -> tuple[int, int]:
    return bits - 1, _clog2(bits)


def residual_add_cost(n: int) -> dict:
    return {"adders": n, "depth": 1, "stream": 0}


def silu_mul_cost(n: int) -> dict:
    a, d = _mul(8)
    return {"adders": n * a, "depth": d, "stream": 0}


def _rms_widths(n: int) -> tuple[int, int, int]:
    sw = (n * 128 * 128).bit_length()
    aw = sw + 2 * RMS_K
    aw += aw % 2
    ow = (127 << RMS_FO).bit_length() + 1
    return sw, aw, ow


def rmsnorm_cost(n: int) -> dict:
    _, aw, ow = _rms_widths(n)
    a, d = _mul(8)
    # squares + sum tree + isqrt, in series; x*gamma in parallel; then one divider per element.
    adders = 2 * n * a + (n - 1) + aw // 2 + n * ow
    return {"adders": adders, "depth": d + _clog2(n) + aw // 2 + ow, "stream": 0}


def softmax_cost(n: int) -> dict:
    q = EXP_FRAC + 1
    # max tree, subtract, exp lookup, sum tree, one divider per element.
    return {"adders": (n - 1) + n + (n - 1) + n * q, "depth": _clog2(n) + 1 + _clog2(n) + q, "stream": 0}


def rope_cost(n: int) -> dict:
    a, d = _mul(8)
    return {"adders": n // 2 * (4 * a + 2), "depth": d + 1, "stream": 0}


def attention_cost(cfg: Config) -> dict:
    """Decode step at full context: one KV-cache row per cycle for q.k, then again for p.v, all heads
    in parallel, with one softmax per head between the two passes."""
    a, d = _mul(8)
    hs, heads = cfg.head_size, cfg.n_heads
    sm = softmax_cost(cfg.seq_len)
    dot = hs * a + hs - 1
    acc = hs * a + hs
    return {
        "adders": heads * (dot + acc + sm["adders"]),
        "depth": (d + _clog2(hs)) + sm["depth"] + (d + 1),
        "stream": 2 * cfg.seq_len,
    }


def spec_op_cost(op: dict, cfg: Config) -> dict:
    """Cost of one template op from a model spec."""
    kind = op["op"]
    if kind == "embedding":
        return {"adders": 0, "depth": 0, "stream": 0}
    if kind == "rmsnorm":
        return rmsnorm_cost(op["params"]["dim"])
    if kind == "residual_add":
        return residual_add_cost(op["params"]["dim"])
    if kind == "silu_mul":
        return silu_mul_cost(op["params"]["dim"])
    if kind == "rope":  # q and k rotate in parallel
        q, k = rope_cost(cfg.dim), rope_cost(cfg.kv_dim)
        return {"adders": q["adders"] + k["adders"], "depth": max(q["depth"], k["depth"]), "stream": 0}
    if kind == "attention":
        return attention_cost(cfg)
    raise ValueError(f"no cost model for op {kind!r}")


# --- Verilog emission -----------------------------------------------------------------------------

def _hex(values, width: int) -> str:
    """Pack values (element 0 in the low bits) into a Verilog literal of len(values) * width bits."""
    mask = (1 << width) - 1
    packed = 0
    for i, v in enumerate(values):
        packed |= (int(v) & mask) << (width * i)
    bits = width * len(values)
    return f"{bits}'h{packed:0{(bits + 3) // 4}x}"


def _emit(module: str, template: str, block: Block, summary: list[str], params: dict, extra: list[str] = (),
          connect: dict | None = None) -> str:
    ports = [f"    input  wire [{p.width - 1}:0] {p.name}," for p in block.inputs]
    ports.append(f"    output wire [{block.output.width - 1}:0] {block.output.name}")
    connect = connect or {p.name: p.name for p in [*block.inputs, block.output]}
    plist = ", ".join(f".{k}({v})" for k, v in params.items())
    clist = ", ".join(f".{k}({v})" for k, v in connect.items())
    stats = json.dumps({"adders": block.cost["adders"], "depth": block.cost["depth"]})
    return "\n".join(
        ["// Generated by blaze gen-op; do not edit."]
        + [f"// {line}" for line in summary]
        + [f"// blaze-stats: {stats}", ""]
        + [(VERILOG_DIR / f"{template}.v").read_text().rstrip("\n"), ""]
        + ["`default_nettype none", f"module {module} ("]
        + ports
        + [");"]
        + list(extra)
        + [f"    {template} #({plist}) core ({clist});", "endmodule", "`default_nettype wire"]
    ) + "\n"


def _summary(module: str, op: str, source: str, block: Block) -> list[str]:
    ins = ", ".join(f"{p.name}: {p.count} x {p.bits}-bit" for p in block.inputs)
    return [
        f"{module}: {op} from {source}",
        f"inputs {ins}; output y: {block.output.count} x signed {block.output.bits}-bit",
        block.formats,
        f"y_real ~= y * {block.out_scale!r}",
    ]


# --- ops ------------------------------------------------------------------------------------------

def _tdiv(a: int, b: int) -> int:
    """Integer division truncating toward zero, like Verilog's signed `/`."""
    q = abs(a) // b
    return q if a >= 0 else -q


# Each builder returns (block, template params, extra wrapper lines).

def residual_add(cfg: Config, gamma: np.ndarray) -> tuple[Block, dict, list[str]]:
    n = cfg.dim
    block = Block(
        inputs=[Port("a", n, 8), Port("b", n, 8)],
        output=Port("y", n, 9),
        reference=lambda v: v["a"] + v["b"],
        real=lambda v: (v["a"] + v["b"]) / 2**IN_FRAC,
        out_scale=2.0**-IN_FRAC,
        cost=residual_add_cost(n),
        formats="a, b, y: 4 fractional bits; exact",
    )
    return block, {"N": n}, []


def _silu_lut() -> np.ndarray:
    """Entry u is silu of the int8 whose bit pattern is u, with 8 fractional bits."""
    lut = []
    for u in range(256):
        a = (u - 256 if u >= 128 else u) / 2**IN_FRAC
        lut.append(round(a / (1 + math.exp(-a)) * 2**(SILU_SW - IN_FRAC)))
    return np.array(lut, dtype=np.int64)


def silu_mul(cfg: Config, gamma: np.ndarray) -> tuple[Block, dict, list[str]]:
    n = cfg.hidden_dim
    lut = _silu_lut()
    frac = SILU_SW - IN_FRAC

    def real(v):
        a = v["a"] / 2**IN_FRAC
        return a / (1 + np.exp(-a)) * (v["b"] / 2**IN_FRAC)

    block = Block(
        inputs=[Port("a", n, 8), Port("b", n, 8)],
        output=Port("y", n, SILU_SW + 8),
        reference=lambda v: lut[v["a"] & 0xFF] * v["b"],
        real=real,
        out_scale=2.0**-(frac + IN_FRAC),
        cost=silu_mul_cost(n),
        formats=f"a, b: 4 fractional bits; silu(a) lookup: {frac} fractional bits; y: {frac + IN_FRAC}",
    )
    return block, {"N": n, "SW": SILU_SW, "LUT": _hex(lut, SILU_SW)}, []


def rmsnorm(cfg: Config, gamma: np.ndarray) -> tuple[Block, dict, list[str]]:
    n = cfg.dim
    gq, gscale = quantize(gamma)
    g = gq.astype(np.int64)
    sw, aw, ow = _rms_widths(n)

    def ref(v):
        x = v["x"]
        d = math.isqrt(int((x * x).sum()) << (2 * RMS_K))
        if d == 0:
            return np.zeros(n, dtype=np.int64)
        return np.array([_tdiv(int(xi * gi) << (RMS_K + RMS_FO), d) for xi, gi in zip(x, g)], dtype=np.int64)

    def real(v):
        x = v["x"] / 2**IN_FRAC
        return x / np.sqrt(np.mean(x * x) + 1e-5) * gamma

    block = Block(
        inputs=[Port("x", n, 8)],
        output=Port("y", n, ow),
        reference=ref,
        real=real,
        out_scale=gscale * math.sqrt(n) / 2**RMS_FO,
        cost=rmsnorm_cost(n),
        formats=f"x: 4 fractional bits; gamma int8, scale {gscale!r}; eps dropped (zero input gives 0)",
    )
    return block, {"N": n, "K": RMS_K, "FO": RMS_FO, "SW": sw, "AW": aw, "OW": ow, "G": _hex(g, 8)}, []


def softmax(cfg: Config, gamma: np.ndarray) -> tuple[Block, dict, list[str]]:
    n = cfg.seq_len
    lut = np.array([round(math.exp(-k / 2**IN_FRAC) * 2**EXP_FRAC) for k in range(256)], dtype=np.int64)

    def ref(v):
        e = lut[v["x"].max() - v["x"]]
        return (e << EXP_FRAC) // int(e.sum())

    def real(v):
        z = np.exp((v["x"] - v["x"].max()) / 2**IN_FRAC)
        return z / z.sum()

    block = Block(
        inputs=[Port("x", n, 8)],
        output=Port("y", n, EXP_FRAC + 2),
        reference=ref,
        real=real,
        out_scale=2.0**-EXP_FRAC,
        cost=softmax_cost(n),
        formats=f"x: 4 fractional bits (scores pre-scaled by 1/sqrt(head_size)); y: {EXP_FRAC} fractional bits; "
                f"no mask, all {n} positions",
    )
    params = {"N": n, "EW": EXP_FRAC + 1, "FP": EXP_FRAC, "SUMW": (n << EXP_FRAC).bit_length(),
              "LUT": _hex(lut, EXP_FRAC + 1)}
    return block, params, []


def _rope_angle(cfg: Config, pos: int, f: int) -> float:
    """llama2.c run.c: pos * theta^(-2f / head_size), theta = 10000."""
    return pos * 10000.0 ** (-2 * f / cfg.head_size)


def rope(cfg: Config, gamma: np.ndarray) -> tuple[Block, dict, list[str]]:
    n, half = cfg.dim, cfg.head_size // 2
    table = np.zeros((cfg.seq_len, half, 2), dtype=np.int64)  # cos, sin with 14 fractional bits
    for pos in range(cfg.seq_len):
        for f in range(half):
            a = _rope_angle(cfg, pos, f)
            table[pos, f] = round(math.cos(a) * 2**ROPE_FRAC), round(math.sin(a) * 2**ROPE_FRAC)
    pos_bits = max(cfg.seq_len - 1, 1).bit_length()

    def rotate(x, cs):
        y = np.zeros(n, dtype=cs.dtype)
        for p in range(n // 2):
            c, s = cs[p % half]
            x0, x1 = x[2 * p], x[2 * p + 1]
            y[2 * p], y[2 * p + 1] = x0 * c - x1 * s, x0 * s + x1 * c
        return y

    def real(v):
        pos = int(v["pos"][0])
        cs = np.array([[math.cos(_rope_angle(cfg, pos, f)), math.sin(_rope_angle(cfg, pos, f))] for f in range(half)])
        return rotate(v["x"] / 2**IN_FRAC, cs)

    block = Block(
        inputs=[Port("x", n, 8), Port("pos", 1, pos_bits, 0, cfg.seq_len - 1)],
        output=Port("y", n, 8 + ROPE_CW),
        reference=lambda v: rotate(v["x"], table[int(v["pos"][0])]),
        real=real,
        out_scale=2.0**-(IN_FRAC + ROPE_FRAC),
        cost=rope_cost(n),
        formats=f"x: 4 fractional bits; cos/sin ROM {cfg.seq_len} x {half}, {ROPE_FRAC} fractional bits; "
                f"head_size {cfg.head_size}",
    )
    # The cos/sin table is a case ROM in the wrapper, indexed by pos.
    row_bits = ROPE_CW * cfg.head_size
    rom = [f"    function [{row_bits - 1}:0] table_row(input [{pos_bits - 1}:0] p);", "        case (p)"]
    rom += [f"            {pos_bits}'d{i}: table_row = {_hex(r.ravel(), ROPE_CW)};" for i, r in enumerate(table)]
    rom += [f"            default: table_row = {row_bits}'h0;", "        endcase", "    endfunction",
            f"    wire [{row_bits - 1}:0] cs = table_row(pos);"]
    return block, {"N": n, "HS": cfg.head_size, "CW": ROPE_CW}, rom


BUILDERS = {"rmsnorm": rmsnorm, "silu_mul": silu_mul, "softmax": softmax, "rope": rope, "residual_add": residual_add}


def build_op(checkpoint: Path, layer: int, op: str, module: str) -> Block:
    """The op for one layer of a llama2.c checkpoint, with its Verilog in block.verilog."""
    if op not in BUILDERS:
        raise ValueError(f"unknown op {op!r}; expected one of {', '.join(BUILDERS)}")
    cfg, tensors = read_llama2c(checkpoint)
    if not 0 <= layer < cfg.n_layers:
        raise ValueError(f"layer {layer} out of range 0..{cfg.n_layers - 1}")
    # rmsnorm uses the attention norm weights (rms_att); the other ops have no weights.
    gamma = tensors[f"layers.{layer}.rms_att"].load().astype(np.float64)
    block, params, extra = BUILDERS[op](cfg, gamma)
    connect = {"x": "x", "cs": "cs", "y": "y"} if op == "rope" else None
    summary = _summary(module, op, f"{checkpoint.name} layer {layer}", block)
    block.verilog = _emit(module, f"blaze_{op}", block, summary, params, extra, connect)
    return block
