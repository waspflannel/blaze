"""Common subexpression elimination for gen-matrix blocks: same y = q x, fewer adders. See docs/optimizer.md.

Every row of a gen-matrix block is a sum of signed, shifted inputs, `+-(x_j <<< k)`, one per CSD digit.
Two terms that several rows share, up to one common shift and sign (`x_a + (x_b <<< 2)` in one row,
`-(x_a <<< 3) - (x_b <<< 5)` in another), become one wire `c_n = x_a + (x_b <<< 2)` computed once, and
each of those rows uses `(c_n <<< k)` in their place. Wires can contain earlier wires. Greedy: share the
most frequent pair first and repeat until no pair occurs twice. Deterministic; no LLM involved.
"""
import heapq

import numpy as np

from blaze.generate import CSD_TERMS, WEIGHT_MAX, _join_terms, acc_width, csd, matrix_cost, matrix_summary, matvec_module

Term = tuple[int, int]  # (signal, shift): signals 0..cols-1 are x_j, cols + n is wire c_n
Pair = tuple[int, int, int, int, int]  # (a, ka, b, kb, sign): x_a << ka + sign * (x_b << kb), min shift 0


def _pair(row: dict[Term, int], t: Term, u: Term) -> Pair:
    """The pair p with row[t] * t + row[u] * u == row[lo] * (p << m), lo the lower term and m its lower shift."""
    lo, hi = sorted((t, u))
    m = min(lo[1], hi[1])
    return lo[0], lo[1] - m, hi[0], hi[1] - m, row[lo] * row[hi]


def _matches(row: dict[Term, int], shifts: dict[int, set[int]], p: Pair) -> list[tuple[Term, Term, int, int]]:
    """Disjoint occurrences of pair p in one row as (t, u, shift, sign), lowest shift first."""
    a, ka, b, kb, s = p
    used: set[Term] = set()
    out = []
    for k in sorted(shifts.get(a, ())):
        t, u = (a, k), (b, kb + k - ka)
        if k < ka or t in used or u in used or u not in row or row[t] * row[u] != s:
            continue
        used |= {t, u}
        out.append((t, u, k - ka, row[t]))
    return out


def share(q: np.ndarray) -> tuple[list[Pair], list[dict[Term, int]]]:
    """(wires, rows): wire n is signal cols + n; each row maps its remaining terms to their sign."""
    rows_n, cols = q.shape
    rows = [{(j, k): s for j in range(cols) if q[i, j] for s, k in csd(int(q[i, j]))} for i in range(rows_n)]
    shifts = [{} for _ in rows]
    for row, sh in zip(rows, shifts):
        for j, k in row:
            sh.setdefault(j, set()).add(k)

    count: dict[Pair, int] = {}
    for row in rows:
        terms = list(row)
        for x, t in enumerate(terms):
            for u in terms[x + 1:]:
                p = _pair(row, t, u)
                count[p] = count.get(p, 0) + 1
    heap = [(-n, p) for p, n in count.items() if n > 1]
    heapq.heapify(heap)

    def drop(r: int, t: Term) -> None:
        row = rows[r]
        for u in row:
            if u != t:
                count[_pair(row, t, u)] -= 1
        del row[t]
        shifts[r][t[0]].discard(t[1])

    def add(r: int, t: Term, sign: int) -> None:
        row = rows[r]
        row[t] = sign
        for u in row:
            if u != t:
                p = _pair(row, t, u)
                count[p] = count.get(p, 0) + 1
                if count[p] > 1:
                    heapq.heappush(heap, (-count[p], p))
        shifts[r].setdefault(t[0], set()).add(t[1])

    wires: list[Pair] = []
    dead: set[Pair] = set()
    while heap:
        n, p = heapq.heappop(heap)
        if p in dead:
            continue
        if -n != count[p]:  # stale: the count changed since this entry was pushed
            if count[p] > 1:
                heapq.heappush(heap, (-count[p], p))
            continue
        found = [(r, ms) for r in range(rows_n) if (ms := _matches(rows[r], shifts[r], p))]
        if sum(len(ms) for _, ms in found) < 2:
            # ponytail: pairs that overlap inside one row (x << 0, << 2, << 4) are dropped for good here,
            # even if later sharing would free a second disjoint copy.
            dead.add(p)
            continue
        signal = cols + len(wires)
        wires.append(p)
        for r, ms in found:
            for t, u, k, sign in ms:
                drop(r, t)
                drop(r, u)
                add(r, (signal, k), sign)
    return wires, rows


def _depth(levels: list[int]) -> int:
    """Adder levels to sum operands that are ready at `levels`: always add the two earliest."""
    if not levels:
        return 0
    heapq.heapify(levels)
    while len(levels) > 1:
        heapq.heappush(levels, max(heapq.heappop(levels), heapq.heappop(levels)) + 1)
    return levels[0]


def optimize_matrix(q: np.ndarray, scale: float, module: str, source: str, bits: int = 8) -> tuple[str, dict]:
    """Return (verilog, stats) for the same y = q x as generate_matrix, with shared subexpressions.

    Adders counts one per wire plus (terms - 1) per non-empty row. Depth assumes each row is summed as a
    tree that adds its earliest-ready terms first, a wire being ready one level after its later operand.
    """
    rows_n, cols = q.shape
    width = acc_width(q)
    wires, rows = share(q)
    names = [f"x_{j}" for j in range(cols)] + [f"c_{n}" for n in range(len(wires))]
    level = [0] * cols

    def ref(t: Term) -> str:
        return names[t[0]] if t[1] == 0 else f"({names[t[0]]} <<< {t[1]})"

    body = []
    for n, (a, ka, b, kb, s) in enumerate(wires):
        level.append(max(level[a], level[b]) + 1)
        body.append(f"    wire signed [{width - 1}:0] c_{n} = {_join_terms([(1, ref((a, ka))), (s, ref((b, kb)))])};")
    body.append("")
    adders = len(wires)
    depth = 0
    for i, row in enumerate(rows):
        terms = sorted(row, key=lambda t: (t[0], -t[1]))
        adders += max(len(terms) - 1, 0)
        depth = max(depth, _depth([level[t[0]] for t in terms]))
        lsb = i * width
        body.append(f"    assign y[{lsb + width - 1}:{lsb}] = {_join_terms([(row[t], ref(t)) for t in terms])};")

    csd_terms = int(CSD_TERMS[q.astype(np.int64) + WEIGHT_MAX].sum())
    original = matrix_cost(q)["adders"]
    summary = matrix_summary(q, scale, module, source, bits, csd_terms, original) + [
        f"optimized: {len(wires)} shared wires c_n, adders {original} -> {adders}, same y as gen-matrix"]
    stats = {"adders": adders, "depth": depth, "bits": bits, "optimized": True, "original_adders": original}
    verilog = matvec_module("optimize-matrix", summary, stats, module, rows_n, cols, width, body)
    return verilog, {**stats, "wires": len(wires), "acc_width": width}
