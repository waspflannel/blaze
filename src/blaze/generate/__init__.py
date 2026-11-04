"""Weights -> hardwired equations -> combinational Verilog. Deterministic; no LLM involved.

One weight matrix W (out x in) becomes y = W x with every weight a constant:
1. `quantize`: float W -> signed 8-bit integers with one per-matrix scale.
2. `csd`: each integer weight -> canonical signed digits, i.e. a few +-(x << k) terms.
3. `generate_matrix`: one shift/add equation per output row, as text and as a Verilog module.
"""
import json

import numpy as np

INPUT_BITS = 8
WEIGHT_MAX = 127  # symmetric int8 range -127..127; -128 is unused so the scale is the same both ways
INPUT_PEAK = 1 << (INPUT_BITS - 1)  # largest input magnitude, |-128|


def quantize(w: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (q, scale) with q int8 and w ~= scale * q; scale = max|w| / 127, ties round to even."""
    peak = float(np.max(np.abs(w)))
    scale = peak / WEIGHT_MAX if peak else 1.0
    q = np.clip(np.rint(w.astype(np.float64) / scale), -WEIGHT_MAX, WEIGHT_MAX).astype(np.int8)
    return q, scale


def csd(c: int) -> list[tuple[int, int]]:
    """Canonical signed digit (non-adjacent form) of c as (sign, shift) pairs, highest shift first.

    c == sum(sign << shift), with no two adjacent nonzero digits, so 7 is 8 - 1 rather than 4 + 2 + 1.
    """
    digits = []
    shift = 0
    while c:
        if c & 1:
            sign = 2 - (c & 3)  # +1 when c mod 4 == 1, -1 when c mod 4 == 3
            digits.append((sign, shift))
            c -= sign
        c >>= 1
        shift += 1
    return digits[::-1]


# CSD term count of every int8 weight -127..127, indexed by weight + WEIGHT_MAX.
CSD_TERMS = np.array([len(csd(c)) for c in range(-WEIGHT_MAX, WEIGHT_MAX + 1)])


def matrix_cost(q: np.ndarray) -> dict:
    """Adders (CSD terms - 1 per non-empty row) and the deepest row's balanced adder tree depth."""
    terms = CSD_TERMS[q.astype(np.int64) + WEIGHT_MAX].sum(axis=1)
    depth = max((max(int(t) - 1, 0).bit_length() for t in terms), default=0)
    return {"adders": int(np.maximum(terms - 1, 0).sum()), "depth": depth}


def acc_width(q: np.ndarray) -> int:
    """Signed bit width that holds every row's exact sum for any signed 8-bit inputs.

    |y_i| <= 128 * sum_j |q_ij|, so bit_length(128 * max row abs-sum) magnitude bits plus a sign bit
    can never overflow.
    """
    worst_row = int(np.abs(q.astype(np.int64)).sum(axis=1).max())
    return (INPUT_PEAK * worst_row).bit_length() + 1


def _join_terms(terms: list[tuple[int, str]]) -> str:
    """Render signed terms as `a + b - c`; an empty sum is 0."""
    if not terms:
        return "0"
    first_sign, first = terms[0]
    parts = [first if first_sign > 0 else f"-{first}"]
    parts += [("+ " if sign > 0 else "- ") + term for sign, term in terms[1:]]
    return " ".join(parts)


def generate_matrix(q: np.ndarray, scale: float, module: str, source: str) -> tuple[str, str, dict]:
    """Return (verilog, equations, stats) for y = q x, skipping zero weights.

    Adders counts two-input add/subtract operations: (CSD terms - 1) per non-empty row. A leading
    minus is a negation and is not counted. Depth is the longest row's adder tree, ceil(log2 terms).
    """
    rows, cols = q.shape
    width = acc_width(q)
    nonzero = int(np.count_nonzero(q))
    csd_terms = 0
    cost = matrix_cost(q)
    equation_lines = []
    verilog_assigns = []

    for i in range(rows):
        weight_terms = []
        shift_terms = []
        verilog_terms = []
        for j in range(cols):
            weight = int(q[i, j])
            if weight == 0:
                continue
            weight_terms.append((1 if weight > 0 else -1, f"{abs(weight)}*x[{j}]"))
            for sign, shift in csd(weight):
                shift_terms.append((sign, f"x[{j}]" if shift == 0 else f"(x[{j}] << {shift})"))
                verilog_terms.append((sign, f"x_{j}" if shift == 0 else f"(x_{j} <<< {shift})"))
        csd_terms += len(shift_terms)
        equation_lines.append(f"y[{i}] = {_join_terms(weight_terms)}")
        equation_lines.append(f"{' ' * len(f'y[{i}]')} = {_join_terms(shift_terms)}")
        lsb = i * width
        verilog_assigns.append(f"    assign y[{lsb + width - 1}:{lsb}] = {_join_terms(verilog_terms)};")

    stats = {"rows": rows, "cols": cols, "nonzero": nonzero, "csd_terms": csd_terms, "adders": cost["adders"],
             "depth": cost["depth"], "acc_width": width, "scale": scale}
    summary = [
        f"{module}: y = W x, W {rows}x{cols} int8 from {source}",
        f"w_real ~= scale * W, scale = {scale!r}",
        f"x[j]: signed {INPUT_BITS}-bit; y[i]: signed {width}-bit, exact (no rounding, cannot overflow)",
        f"nonzero weights {nonzero}/{rows * cols}, CSD terms {csd_terms}, adders {cost['adders']}",
    ]

    equations = "\n".join([f"# {line}" for line in summary] + [""] + equation_lines) + "\n"

    sign_extends = [
        f"    wire signed [{width - 1}:0] x_{j} = $signed(x[{j * INPUT_BITS + INPUT_BITS - 1}:{j * INPUT_BITS}]);"
        for j in range(cols)
    ]
    verilog = "\n".join(
        ["// Generated by blaze gen-matrix; do not edit."]
        + [f"// {line}" for line in summary]
        + [f"// blaze-stats: {json.dumps(cost)}"]
        + [
            f"// Packed ports: x[j] = x[{INPUT_BITS}*j+{INPUT_BITS - 1}:{INPUT_BITS}*j], "
            f"y[i] = y[{width}*i+{width - 1}:{width}*i], both two's complement.",
            "`default_nettype none",
            f"module {module} (",
            f"    input  wire [{cols * INPUT_BITS - 1}:0] x,",
            f"    output wire [{rows * width - 1}:0] y",
            ");",
            "    // Sign-extend every input to the output width first so each shift and add below is exact.",
        ]
        + sign_extends
        + [""]
        + verilog_assigns
        + ["endmodule", "`default_nettype wire"]
    ) + "\n"
    return verilog, equations, stats
