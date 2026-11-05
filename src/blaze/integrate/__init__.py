"""Integrator: verified blocks -> one chip top level (blaze_chip), its integer reference, and a Verilator
run that generates tokens and compares them with the reference token for token. See docs/integrator.md.

The chip is the block agent's matmul files verbatim plus the library templates the verified op files
wrap, with the same parameters, glued by requantizers, an attention unit with a KV cache per layer,
an embedding ROM, a classifier and a control FSM. Deterministic; no LLM involved.
"""
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from blaze.generate import generate_matrix, quantize
from blaze.integrate.model import BOS, decode, float_generate
from blaze.library import (EXP_FRAC, IN_FRAC, ROPE_FRAC, SILU_SW, VERILOG_DIR, Block, _clog2, _hex, _mul, build_op,
                           exp_lut, residual_add, rmsnorm, rope, silu_mul)
from blaze.reference import matvec_int
from blaze.report import ADDER_NS, SCALE_5NM, SKY130_CLOCK_MHZ, _levels
from blaze.sim import MATMUL_TENSORS, _run
from blaze.spec import Config, load_matrix, read_llama2c

OPS = ["rmsnorm", "silu_mul", "softmax", "rope", "residual_add"]  # the op files the block agent verifies
TEMPLATES = ["blaze_rmsnorm", "blaze_rope", "blaze_silu_mul", "blaze_residual_add", "blaze_requant",
             "blaze_attention"]
CAL_STEPS = 64  # greedy float steps from BOS whose peak activations set the int8 scales
SCORE_BITS = 16  # attention scores: signed, IN_FRAC fractional bits, saturated
REQUANT_DEPTH = _mul(16)[1] + 1  # multiply by a 16-bit constant, then the rounding add


@dataclass(frozen=True)
class Requant:
    """y = clamp(floor((x * m + 2^(s-1)) / 2^s)) to signed `bits`: x * factor, rounded half up."""
    m: int
    s: int
    bits: int = 8

    @staticmethod
    def of(factor: float, bits: int = 8) -> "Requant":
        if not factor > 0:
            raise ValueError(f"requant factor must be positive, got {factor}")
        s = 15 - math.floor(math.log2(factor))  # m = round(factor * 2^s) lands in [2^15, 2^16]
        if not 1 <= s <= 46:
            raise ValueError(f"requant factor {factor} out of range")
        return Requant(round(factor * 2**s), s, bits)

    def __call__(self, x) -> np.ndarray:
        lo, hi = -(1 << (self.bits - 1)), (1 << (self.bits - 1)) - 1
        return np.clip((np.asarray(x, dtype=np.int64) * self.m + (1 << (self.s - 1))) >> self.s, lo, hi)


@dataclass
class Mat:
    module: str
    q: np.ndarray
    scale: float
    bits: int
    width: int  # signed output element width
    depth: int
    verilog: str


@dataclass
class Layer:
    mats: dict[str, Mat]
    att_norm: tuple[Block, dict]
    ffn_norm: tuple[Block, dict]
    rope_q: tuple[Block, dict]
    rope_k: tuple[Block, dict]
    silu: tuple[Block, dict]
    residual: tuple[Block, dict]
    rq: dict[str, Requant] = field(default_factory=dict)


@dataclass
class Chip:
    cfg: Config
    layers: list[Layer]
    emb: np.ndarray  # vocab x dim int8 at the residual scale
    final_norm: tuple[Block, dict]
    rq_xf: Requant
    cls: Mat
    rope_rom: list[str]
    scales: dict[str, float]


# --- planning -------------------------------------------------------------------------------------

def _scale(peak: dict[str, float], name: str) -> float:
    return peak[name] / 127 if peak.get(name) else 2.0**-IN_FRAC


def _block_path(blocks: Path, layer: int, name: str) -> Path:
    """layer<N>/<name>. A flat file in the same directory is the old layout."""
    nested = blocks / f"layer{layer}" / name
    flat = blocks / name
    return flat if flat.is_file() and not nested.is_file() else nested


def _matrix(checkpoint: Path, blocks: Path, layer: int, tensor: str) -> Mat:
    """A verified gen-matrix file, checked to be exactly what gen-matrix writes for this checkpoint."""
    module = f"l{layer}_{tensor}_matvec"
    path = _block_path(blocks, layer, f"{module}.v")
    text = path.read_text()
    m = re.search(r"^// blaze-stats: (.*)$", text, re.M)
    bits = json.loads(m[1]).get("bits", 8) if m else 8
    q, scale = quantize(load_matrix(checkpoint, layer, tensor), bits)
    verilog, _, stats = generate_matrix(q, scale, module, f"{checkpoint.name} layer {layer} {tensor}", bits)
    if verilog != text:
        raise ValueError(f"{path} is not gen-matrix output for {checkpoint.name} layer {layer} {tensor} at {bits} "
                         "bits; regenerate and re-verify it")
    return Mat(module, q.astype(np.int64), scale, bits, stats["acc_width"], stats["depth"], text)


def _check_ops(checkpoint: Path, blocks: Path, layer: int) -> None:
    """The verified op files must be exactly what gen-op writes; the chip instantiates the same templates."""
    for op in OPS:
        path = _block_path(blocks, layer, f"l{layer}_{op}.v")
        if path.read_text() != build_op(checkpoint, layer, op, f"l{layer}_{op}").verilog:
            raise ValueError(f"{path} is not gen-op output for {checkpoint.name} layer {layer}; regenerate and "
                             "re-verify it")


def plan_chip(checkpoint: Path, blocks: Path, n_layers: int) -> Chip:
    cfg, tensors = read_llama2c(checkpoint)
    if not 1 <= n_layers <= cfg.n_layers:
        raise ValueError(f"--layers must be 1..{cfg.n_layers}, got {n_layers}")
    missing = []
    for i in range(n_layers):
        names = [f"l{i}_{t}_matvec.v" for t in MATMUL_TENSORS] + [f"l{i}_{op}.v" for op in OPS]
        missing += [f"layer{i}/{name}" for name in names if not _block_path(blocks, i, name).is_file()]
    if missing:
        raise ValueError(f"{blocks} is missing {len(missing)} verified block files: {', '.join(missing)}")

    _, peak = float_generate(cfg, tensors, n_layers, CAL_STEPS)
    s_res = _scale(peak, "res")
    scales = {"res": s_res}
    kv_cfg = Config(cfg.kv_dim, cfg.hidden_dim, cfg.n_layers, cfg.n_kv_heads, cfg.n_kv_heads, cfg.vocab_size,
                    cfg.seq_len, cfg.shared_classifier)  # same head size and rope table, n = kv_dim
    layers = []
    for i in range(n_layers):
        p = f"layers.{i}."
        _check_ops(checkpoint, blocks, i)
        mats = {t: _matrix(checkpoint, blocks, i, t) for t in MATMUL_TENSORS}
        ly = Layer(
            mats=mats,
            att_norm=rmsnorm(cfg, tensors[p + "rms_att"].load().astype(np.float64))[:2],
            ffn_norm=rmsnorm(cfg, tensors[p + "rms_ffn"].load().astype(np.float64))[:2],
            rope_q=rope(cfg, None)[:2],
            rope_k=rope(kv_cfg, None)[:2],
            silu=silu_mul(cfg, None)[:2],
            residual=residual_add(cfg, None)[:2],
        )
        s = {n: _scale(peak, f"l{i}.{n}") for n in ("xn", "q", "k", "v", "qr", "kr", "xn2", "h3", "g")}
        scales.update({f"l{i}.{n}": v for n, v in s.items()})
        rot = 2.0**-ROPE_FRAC
        ly.rq = {
            "xn": Requant.of(ly.att_norm[0].out_scale / s["xn"]),
            "q": Requant.of(mats["wq"].scale * s["xn"] / s["q"]),
            "k": Requant.of(mats["wk"].scale * s["xn"] / s["k"]),
            "v": Requant.of(mats["wv"].scale * s["xn"] / s["v"]),
            "qr": Requant.of(s["q"] * rot / s["qr"]),
            "kr": Requant.of(s["k"] * rot / s["kr"]),
            # q.k in real units, / sqrt(head_size), to IN_FRAC fractional bits
            "score": Requant.of(s["qr"] * s["kr"] / math.sqrt(cfg.head_size) * 2**IN_FRAC, SCORE_BITS),
            "att": Requant.of(2.0**-EXP_FRAC),  # sum(p) ~= 2^EXP_FRAC, so the output keeps v's scale
            "wo": Requant.of(mats["wo"].scale * s["v"] / s_res),
            "res": Requant.of(1.0),  # residual sum -> int8: saturate
            "xn2": Requant.of(ly.ffn_norm[0].out_scale / s["xn2"]),
            "h1": Requant.of(mats["w1"].scale * s["xn2"] * 2**IN_FRAC),  # silu LUT input: 4 fractional bits
            "h3": Requant.of(mats["w3"].scale * s["xn2"] / s["h3"]),
            "g": Requant.of(2.0**-(SILU_SW - IN_FRAC) * s["h3"] / s["g"]),
            "w2": Requant.of(mats["w2"].scale * s["g"] / s_res),
        }
        layers.append(ly)

    emb = tensors["token_embedding"].load().astype(np.float64)
    wcls = tensors["wcls"].load() if "wcls" in tensors else emb
    q, scale = quantize(wcls)
    verilog, _, stats = generate_matrix(q, scale, "cls_matvec", f"{checkpoint.name} classifier")
    cls = Mat("cls_matvec", q.astype(np.int64), scale, 8, stats["acc_width"], stats["depth"], verilog)
    final_norm = rmsnorm(cfg, tensors["rms_final"].load().astype(np.float64))[:2]
    scales["xf"] = _scale(peak, "xf")
    return Chip(
        cfg=cfg,
        layers=layers,
        emb=np.clip(np.rint(emb / s_res), -127, 127).astype(np.int64),
        final_norm=final_norm,
        rq_xf=Requant.of(final_norm[0].out_scale / scales["xf"]),
        cls=cls,
        rope_rom=rope(cfg, None)[2],
        scales=scales,
    )


# --- integer reference ----------------------------------------------------------------------------

def _attention(cfg: Config, rq: dict[str, Requant], q: np.ndarray, kc: np.ndarray, vc: np.ndarray) -> np.ndarray:
    """blaze_attention for one step: kc, vc hold cache rows 0..pos."""
    hs, kvm = cfg.head_size, cfg.n_heads // cfg.n_kv_heads
    lut = exp_lut()
    acc = np.zeros(cfg.dim, dtype=np.int64)
    for h in range(cfg.n_heads):
        kv = slice((h // kvm) * hs, (h // kvm + 1) * hs)
        s = rq["score"](kc[:, kv] @ q[h * hs:(h + 1) * hs])
        e = lut[np.minimum(s.max() - s, 255)]
        p = (e << EXP_FRAC) // int(e.sum())
        acc[h * hs:(h + 1) * hs] = p @ vc[:, kv]
    return rq["att"](acc)


def reference_generate(chip: Chip, steps: int, inputs: list[int]) -> list[tuple[int, int]]:
    """(token, its logit) per step: what blaze_chip must produce. Position p reads inputs[p] while
    there is one, then the previous output token (greedy)."""
    cfg = chip.cfg
    kc = np.zeros((len(chip.layers), steps, cfg.kv_dim), dtype=np.int64)
    vc = np.zeros_like(kc)
    out = []
    for pos in range(steps):
        token = inputs[pos] if pos < len(inputs) else out[-1][0]
        x = chip.emb[token]
        at = np.array([pos])
        for i, ly in enumerate(chip.layers):
            r, w = ly.rq, {t: m.q for t, m in ly.mats.items()}
            xn = r["xn"](ly.att_norm[0].reference({"x": x}))
            q, k, v = r["q"](matvec_int(w["wq"], xn)), r["k"](matvec_int(w["wk"], xn)), r["v"](matvec_int(w["wv"], xn))
            kc[i, pos] = r["kr"](ly.rope_k[0].reference({"x": k, "pos": at}))
            vc[i, pos] = v
            qr = r["qr"](ly.rope_q[0].reference({"x": q, "pos": at}))
            att = _attention(cfg, r, qr, kc[i, :pos + 1], vc[i, :pos + 1])
            x = r["res"](ly.residual[0].reference({"a": x, "b": r["wo"](matvec_int(w["wo"], att))}))
            xn = r["xn2"](ly.ffn_norm[0].reference({"x": x}))
            h1, h3 = r["h1"](matvec_int(w["w1"], xn)), r["h3"](matvec_int(w["w3"], xn))
            g = r["g"](ly.silu[0].reference({"a": h1, "b": h3}))
            x = r["res"](ly.residual[0].reference({"a": x, "b": r["w2"](matvec_int(w["w2"], g))}))
        logits = matvec_int(chip.cls.q, chip.rq_xf(chip.final_norm[0].reference({"x": x})))
        best = int(np.argmax(logits))  # first maximum, like the chip
        out.append((best, int(logits[best])))
    return out


# --- Verilog --------------------------------------------------------------------------------------

def _inst(template: str, params: dict, name: str, ports: dict) -> str:
    plist = ", ".join(f".{k}({v})" for k, v in params.items())
    clist = ", ".join(f".{k}({v})" for k, v in ports.items())
    return f"    {template} #({plist}) {name} ({clist});"


def _requant(name: str, n: int, iw: int, r: Requant, src: str) -> list[str]:
    """Wire `name` = requant of src (n elements of iw bits)."""
    return [f"    wire [{r.bits * n - 1}:0] {name};",
            _inst("blaze_requant", {"N": n, "IW": iw, "OB": r.bits, "M": r.m, "S": r.s}, f"{name}_rq",
                  {"x": src, "y": name})]


def _attention_rows(cfg: Config) -> tuple[int, int, int]:
    """Cycles per cache row in blaze_attention's three passes (R1, R2, R3): row depth over levels per cycle."""
    mul = _mul(8)[1]
    depths = (mul + _clog2(cfg.head_size) + REQUANT_DEPTH + 1,  # q.k, score requant, max compare
              1 + 1,                                            # max - score (the LUT is free), sum add
              1 + (EXP_FRAC + 1) + mul + 1)                     # max - score, divider, p * v, accumulate add
    levels = _levels(SKY130_CLOCK_MHZ, ADDER_NS)
    return tuple(max(1, math.ceil(d / levels)) for d in depths)


def _stages(chip: Chip) -> list[dict]:
    """Control steps after the embedding cycle: name, depth (adder levels) or attention, and the
    registers loaded when the step ends."""
    cfg, rqd = chip.cfg, REQUANT_DEPTH
    out = []
    for i, ly in enumerate(chip.layers):
        m, p = ly.mats, f"l{i}_"
        out += [
            {"name": p + "attn_norm", "depth": ly.att_norm[0].cost["depth"] + rqd, "load": {p + "xn": p + "xn_d"}},
            {"name": p + "qkv", "depth": max(m[t].depth for t in ("wq", "wk", "wv")) + rqd,
             "load": {p + "q": p + "q_d", p + "k": p + "k_d", p + "v": p + "v_d"}},
            {"name": p + "rope", "depth": max(ly.rope_q[0].cost["depth"], ly.rope_k[0].cost["depth"]) + rqd,
             "load": {p + "qr": p + "qr_d", p + "kr": p + "kr_d"}},
            {"name": p + "attention", "attention": p + "att"},
            {"name": p + "wo_residual", "depth": m["wo"].depth + rqd + 1 + rqd, "load": {"x": p + "x1_d"}},
            {"name": p + "ffn_norm", "depth": ly.ffn_norm[0].cost["depth"] + rqd, "load": {p + "xn2": p + "xn2_d"}},
            {"name": p + "w1_w3", "depth": max(m["w1"].depth, m["w3"].depth) + rqd,
             "load": {p + "h1": p + "h1_d", p + "h3": p + "h3_d"}},
            {"name": p + "silu_mul", "depth": ly.silu[0].cost["depth"] + rqd, "load": {p + "g": p + "g_d"}},
            {"name": p + "w2_residual", "depth": m["w2"].depth + rqd + 1 + rqd, "load": {"x": p + "x2_d"}},
        ]
    out += [
        {"name": "final_norm", "depth": chip.final_norm[0].cost["depth"] + rqd, "load": {"xf": "xf_d"}},
        {"name": "classifier", "depth": chip.cls.depth + _clog2(cfg.vocab_size),
         "load": {"token_out": "best", "logit_out": "best_logit", "done": "1'b1"}},
    ]
    levels = _levels(SKY130_CLOCK_MHZ, ADDER_NS)
    for s in out:
        if "depth" in s:
            s["cycles"] = max(1, math.ceil(s["depth"] / levels))
    return out


def emit_chip(chip: Chip, source: str) -> str:
    cfg = chip.cfg
    d, kvd, hd = cfg.dim, cfg.kv_dim, cfg.hidden_dim
    tb = max(cfg.vocab_size - 1, 1).bit_length()
    pb = max(cfg.seq_len - 1, 1).bit_length()
    lw = chip.cls.width
    stages = _stages(chip)
    rows = _attention_rows(cfg)
    sb = max(len(stages), 1).bit_length()
    cw = max(max(s.get("cycles", 1) for s in stages), 2).bit_length()

    body = [
        "`default_nettype none",
        "module blaze_chip (",
        "    input  wire clk,",
        "    input  wire rst,",
        "    input  wire start,                // one cycle: run one token",
        f"    input  wire [{tb - 1}:0] token_in,",
        f"    input  wire [{pb - 1}:0] pos_in,",
        f"    output reg  [{tb - 1}:0] token_out,",
        f"    output reg  [{lw - 1}:0] logit_out,  // signed logit of token_out",
        "    output reg  done                  // one cycle: token_out is valid",
        ");",
        f"    reg [{sb - 1}:0] st;",
        f"    reg [{cw - 1}:0] cnt;",
        f"    reg [{pb - 1}:0] pos;",
        f"    reg [{8 * d - 1}:0] x;  // residual stream, int8, scale {chip.scales['res']!r}",
        "",
        f"    function [{8 * d - 1}:0] emb_row(input [{tb - 1}:0] tk);",
        "        case (tk)",
    ]
    body += [f"            {tb}'d{t}: emb_row = {_hex(row, 8)};" for t, row in enumerate(chip.emb)]
    body += [f"            default: emb_row = {8 * d}'h0;", "        endcase", "    endfunction", ""]
    body += chip.rope_rom + [""]

    for i, ly in enumerate(chip.layers):
        p, r, m = f"l{i}_", ly.rq, ly.mats
        body += [f"    // layer {i}"]
        body += [f"    reg [{8 * n - 1}:0] {p}{name};" for name, n in
                 (("xn", d), ("q", d), ("k", kvd), ("v", kvd), ("qr", d), ("kr", kvd), ("xn2", d), ("h1", hd),
                  ("h3", hd), ("g", hd))]
        body += [f"    wire [{8 * d - 1}:0] {p}att;", f"    wire {p}att_done;", f"    wire {p}att_go;"]
        rms_bits = ly.att_norm[0].output.bits
        rope_bits = ly.rope_q[0].output.bits
        att_params = {
            "NH": cfg.n_heads, "NKV": cfg.n_kv_heads, "HS": cfg.head_size, "SEQ": cfg.seq_len, "PB": pb,
            "DW": (cfg.head_size << 14).bit_length() + 1, "SB": SCORE_BITS, "SM": r["score"].m,
            "SS": r["score"].s, "EW": EXP_FRAC + 1, "FP": EXP_FRAC,
            "SUMW": (cfg.seq_len << EXP_FRAC).bit_length(), "LUT": _hex(exp_lut(), EXP_FRAC + 1),
            "AW": (128 << EXP_FRAC).bit_length() + 1, "OM": r["att"].m, "OS": r["att"].s,
            "R1": rows[0], "R2": rows[1], "R3": rows[2],
        }
        body += [f"    wire [{d * rms_bits - 1}:0] {p}an;",
                 _inst("blaze_rmsnorm", ly.att_norm[1], f"{p}attn_norm", {"x": "x", "y": f"{p}an"})]
        body += _requant(f"{p}xn_d", d, rms_bits, r["xn"], f"{p}an")
        for t, dst in (("wq", "q"), ("wk", "k"), ("wv", "v"), ("wo", "o1"), ("w1", "h1"), ("w3", "h3"),
                       ("w2", "o2")):
            src = {"wq": "xn", "wk": "xn", "wv": "xn", "wo": "att", "w1": "xn2", "w3": "xn2", "w2": "g"}[t]
            mt = m[t]
            body += [f"    wire [{mt.q.shape[0] * mt.width - 1}:0] {p}{t}_y;",
                     f"    {mt.module} {p}{t} (.x({p}{src}), .y({p}{t}_y));"]
            body += _requant(f"{p}{dst}_d", mt.q.shape[0], mt.width, r[t if t in ("wo", "w2") else dst], f"{p}{t}_y")
        body += [f"    wire [{d * rope_bits - 1}:0] {p}qr_y;",
                 _inst("blaze_rope", ly.rope_q[1], f"{p}rope_q", {"x": f"{p}q", "cs": "cs", "y": f"{p}qr_y"}),
                 f"    wire [{kvd * rope_bits - 1}:0] {p}kr_y;",
                 _inst("blaze_rope", ly.rope_k[1], f"{p}rope_k", {"x": f"{p}k", "cs": "cs", "y": f"{p}kr_y"})]
        body += _requant(f"{p}qr_d", d, rope_bits, r["qr"], f"{p}qr_y")
        body += _requant(f"{p}kr_d", kvd, rope_bits, r["kr"], f"{p}kr_y")
        body += [_inst("blaze_attention", att_params, f"{p}attention",
                       {"clk": "clk", "rst": "rst", "start": f"{p}att_go", "pos": "pos", "q": f"{p}qr", "k": f"{p}kr",
                        "v": f"{p}v", "y": f"{p}att", "done": f"{p}att_done"})]
        for n, b, nxt in (("1", "o1_d", "x1_d"), ("2", "o2_d", "x2_d")):
            body += [f"    wire [{9 * d - 1}:0] {p}r{n};",
                     _inst("blaze_residual_add", ly.residual[1], f"{p}residual{n}",
                           {"a": "x", "b": f"{p}{b}", "y": f"{p}r{n}"})]
            body += _requant(f"{p}{nxt}", d, 9, r["res"], f"{p}r{n}")
        body += [f"    wire [{d * rms_bits - 1}:0] {p}fn;",
                 _inst("blaze_rmsnorm", ly.ffn_norm[1], f"{p}ffn_norm", {"x": "x", "y": f"{p}fn"})]
        body += _requant(f"{p}xn2_d", d, rms_bits, r["xn2"], f"{p}fn")
        silu_bits = ly.silu[0].output.bits
        body += [f"    wire [{hd * silu_bits - 1}:0] {p}sm;",
                 _inst("blaze_silu_mul", ly.silu[1], f"{p}silu_mul", {"a": f"{p}h1", "b": f"{p}h3", "y": f"{p}sm"})]
        body += _requant(f"{p}g_d", hd, silu_bits, r["g"], f"{p}sm")
        body += [""]

    rms_bits = chip.final_norm[0].output.bits
    body += ["    // final norm, classifier, argmax tree (first maximum wins)",
             f"    reg [{8 * d - 1}:0] xf;",
             f"    wire [{d * rms_bits - 1}:0] fnorm;",
             _inst("blaze_rmsnorm", chip.final_norm[1], "final_norm", {"x": "x", "y": "fnorm"})]
    body += _requant("xf_d", d, rms_bits, chip.rq_xf, "fnorm")
    v = cfg.vocab_size
    body += [f"    wire [{v * lw - 1}:0] logits;", "    cls_matvec classifier (.x(xf), .y(logits));"]
    nodes = [(f"logits[{lw * i} +: {lw}]", f"{tb}'d{i}") for i in range(v)]  # (value, index)
    lvl = 0
    while len(nodes) > 1:  # the right node wins only if strictly greater, so a tie keeps the lower index
        nxt = []
        for j in range(0, len(nodes) - 1, 2):
            (lv, li), (rv, ri) = nodes[j], nodes[j + 1]
            n = f"am{lvl}_{j // 2}"
            body += [f"    wire {n}_r = $signed({rv}) > $signed({lv});",
                     f"    wire signed [{lw - 1}:0] {n}_v = {n}_r ? {rv} : {lv};",
                     f"    wire [{tb - 1}:0] {n}_i = {n}_r ? {ri} : {li};"]
            nxt.append((f"{n}_v", f"{n}_i"))
        nodes = nxt + nodes[-1:] if len(nodes) % 2 else nxt  # an odd last node passes through
        lvl += 1
    body += [f"    wire [{tb - 1}:0] best = {nodes[0][1]};",
             f"    wire signed [{lw - 1}:0] best_logit = {nodes[0][0]};", ""]

    # Control: state 0 waits for start and loads the embedding row; each later state is one stage.
    fsm = ["    // control: one state per stage; a stage of n cycles holds its inputs for n cycles"]
    fsm += [f"    assign {s['attention']}_go = (st == {k}) && (cnt == 0);"
            for k, s in enumerate(stages, 1) if "attention" in s]
    fsm += [
        "    always @(posedge clk) begin",
        "        done <= 1'b0;",
        "        if (rst) begin",
        "            st <= 0;",
        "            cnt <= 0;",
        "        end else if (st == 0) begin",
        "            if (start) begin",
        "                pos <= pos_in;",
        "                x <= emb_row(token_in);",
        "                st <= 1;",
        "                cnt <= 0;",
        "            end",
        "        end else begin",
        "            case (st)",
    ]
    for k, s in enumerate(stages, 1):
        nxt = 0 if k == len(stages) else k + 1
        if "attention" in s:
            fsm += [f"                {k}: if (cnt == 0) cnt <= 1;  // {s['name']}: start pulse, then wait",
                    f"                   else if ({s['attention']}_done) begin st <= {nxt}; cnt <= 0; end"]
        else:
            loads = " ".join(f"{dst} <= {src};" for dst, src in s["load"].items())
            fsm += [f"                {k}: if (cnt == {s['cycles'] - 1}) begin  // {s['name']}, {s['cycles']} cycles",
                    f"                       {loads} st <= {nxt}; cnt <= 0;",
                    "                   end else cnt <= cnt + 1'b1;"]
    fsm += ["                default: st <= 0;", "            endcase", "        end", "    end",
            "endmodule", "`default_nettype wire"]

    header = [
        "// Generated by blaze integrate; do not edit.",
        f"// blaze_chip: {source}, {len(chip.layers)} of {cfg.n_layers} layers, greedy decode one token per start.",
        f"// int8 activation scales (real ~= v * scale): {json.dumps(chip.scales)}",
        "",
    ]
    templates = [(VERILOG_DIR / f"{t}.v").read_text().rstrip("\n") + "\n" for t in TEMPLATES]
    mats = [m.verilog for ly in chip.layers for m in ly.mats.values()] + [chip.cls.verilog]
    return "\n".join(header) + "\n".join(templates + mats) + "\n" + "\n".join(body + fsm) + "\n"


# --- simulation -----------------------------------------------------------------------------------

TESTBENCH = """// Generated by blaze integrate: reset, then per step pulse start with the input token and the
// position, clock until done and print "token logit cycles". Usage: Vchip <steps> <input tokens...>;
// position p reads input token p while there is one, then the chip's previous output token.
#include <cstdio>
#include <cstdlib>
#include <memory>
#include "verilated.h"
#include "Vblaze_chip.h"

int main(int argc, char** argv) {
    auto ctx = std::make_unique<VerilatedContext>();
    auto top = std::make_unique<Vblaze_chip>(ctx.get());
    long steps = std::atol(argv[1]);
    long forced = argc - 2;
    unsigned long token = 0;
    auto tick = [&] { top->clk = 0; top->eval(); top->clk = 1; top->eval(); };
    top->start = 0;
    top->rst = 1;
    tick();
    top->rst = 0;
    for (long pos = 0; pos < steps; pos++) {
        if (pos < forced) token = std::strtoul(argv[2 + pos], nullptr, 10);
        top->token_in = token;
        top->pos_in = pos;
        top->start = 1;
        tick();
        top->start = 0;
        unsigned long cycles = 1;
        while (!top->done) {
            tick();
            if (++cycles > @LIMIT@UL) {
                std::fprintf(stderr, "no done after %lu cycles at step %ld\\n", cycles, pos);
                return 2;
            }
        }
        token = top->token_out;
        std::printf("%lu %llu %lu\\n", token, (unsigned long long)top->logit_out, cycles);
    }
    top->final();
    return 0;
}
"""


def _write(path: Path, text: str) -> None:
    """Write only on a change, so Verilator skips an identical rebuild (a big chip takes minutes)."""
    if not path.is_file() or path.read_text() != text:
        path.write_text(text, newline="\n")


def simulate_chip(chip: Chip, chip_path: Path, steps: int, inputs: list[int]) -> tuple[str, list[tuple[int, int, int]]]:
    """Build blaze_chip with Verilator next to chip_path and run it; (version, [(token, logit, cycles)])."""
    workdir = chip_path.parent
    stages = _stages(chip)
    limit = 10 * (sum(s.get("cycles", 0) for s in stages) + len(chip.layers) * (sum(_attention_rows(chip.cfg)) * chip.cfg.seq_len + 3)) + 100
    _write(workdir / "tb.cpp", TESTBENCH.replace("@LIMIT@", str(limit)))
    version = _run(["verilator", "--version"], workdir).strip()
    _run(["verilator", "--cc", "--exe", "--build", "-j", "0", "-Wno-fatal", "--top-module", "blaze_chip",
          "-Mdir", "obj", "-o", "Vchip", chip_path.name, "tb.cpp"], workdir)
    lines = _run(["./obj/Vchip", str(steps), *map(str, inputs)], workdir).split("\n")
    lw = chip.cls.width
    out = []
    for line in filter(None, lines):
        t, logit, cycles = (int(v) for v in line.split())
        out.append((t, logit - (1 << lw) if logit >> (lw - 1) else logit, cycles))
    if len(out) != steps:
        raise RuntimeError(f"chip simulation returned {len(out)} tokens, expected {steps}")
    return version, out


def integrate(checkpoint: Path, blocks: Path, n_layers: int | None, steps: int, out: Path, sim: bool = True,
              prompt: list[int] = ()) -> dict:
    """Write <out>/blaze_chip.v; with sim, run `steps` positions on it (BOS, then the prompt tokens, then
    greedy) and compare every output token and logit with the reference."""
    cfg, tensors = read_llama2c(checkpoint)
    n_layers = cfg.n_layers if n_layers is None else n_layers
    if not 1 <= steps <= cfg.seq_len:
        raise ValueError(f"--steps must be 1..{cfg.seq_len} (seq_len), got {steps}")
    bad = [t for t in prompt if not 0 <= t < cfg.vocab_size]
    if bad:
        raise ValueError(f"prompt tokens must be 0..{cfg.vocab_size - 1}, got {bad}")
    inputs = [BOS, *prompt]
    if len(inputs) > steps:
        raise ValueError(f"--prompt has {len(prompt)} tokens, so --steps must be at least {len(inputs)}")
    chip =plan_chip(checkpoint, blocks, n_layers)
    out.mkdir(parents=True, exist_ok=True)
    chip_path = out / "blaze_chip.v"
    _write(chip_path, emit_chip(chip, checkpoint.name))

    ref = reference_generate(chip, steps, inputs)
    float_tokens, _ = float_generate(cfg, tensors, n_layers, steps, inputs)
    tokenizer = checkpoint.parent / f"tok{cfg.vocab_size}.bin"
    text = (lambda ts: decode(tokenizer, ts)) if tokenizer.is_file() else (lambda ts: None)
    stages = _stages(chip)
    r = {
        "chip": str(chip_path), "model": checkpoint.name, "layers": n_layers, "model_layers": cfg.n_layers,
        "blocks": str(blocks),
        "bits": {f"l{i}": {t: m.bits for t, m in ly.mats.items()} for i, ly in enumerate(chip.layers)},
        "steps": steps, "inputs": inputs,
        "reference_tokens": [t for t, _ in ref], "reference_text": text([t for t, _ in ref]),
        "float_tokens": float_tokens,
        "float_agreement": sum(a == b for a, b in zip(float_tokens, (t for t, _ in ref))),
        "stage_cycles": {s["name"]: s["cycles"] for s in stages if "cycles" in s},
        "attention_row_cycles": dict(zip(("scores", "exp_sum", "accumulate"), _attention_rows(cfg))),
        "scales": chip.scales,
        "simulated": sim,
    }
    if not sim:
        return r
    version, got = simulate_chip(chip, chip_path, steps, inputs)
    first = next(({"step": i, "expected": {"token": e[0], "logit": e[1]}, "got": {"token": g[0], "logit": g[1]}}
                  for i, (e, g) in enumerate(zip(ref, got)) if (e[0], e[1]) != (g[0], g[1])), None)
    cycles = [c for _, _, c in got]
    mean = sum(cycles) / len(cycles)
    tok_s = SKY130_CLOCK_MHZ * 1e6 / mean
    r.update({
        "simulator": version, "match": first is None, "first_mismatch": first,
        "chip_tokens": [t for t, _, _ in got], "chip_text": text([t for t, _, _ in got]),
        "cycles_per_token": cycles, "mean_cycles_per_token": mean,
        "tok_per_s": {"sky130": tok_s, "5nm": tok_s * SCALE_5NM},
        "assumptions": {"sky130_clock_mhz": SKY130_CLOCK_MHZ, "adder_ns": ADDER_NS,
                        "levels_per_cycle": _levels(SKY130_CLOCK_MHZ, ADDER_NS), "scale_5nm": SCALE_5NM},
    })
    return r
