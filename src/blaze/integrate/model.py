"""Float reference of the model (llama2.c run.c forward, float64) and the llama2.c tokenizer decoder.

The chip's activation scales are calibrated from the float model's peak activations, and its greedy
tokens are reported next to the chip's for information. The correctness check is the integer
reference in blaze.integrate, not this.
"""
import math
import struct
from pathlib import Path

import numpy as np

from blaze.spec import Config, Tensor

BOS = 1  # llama2.c tokenizer: 1 is <s>


def _rmsnorm(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x) + 1e-5) * g


def _rope(x: np.ndarray, pos: int, head_size: int) -> np.ndarray:
    """run.c RoPE: pair (x[2p], x[2p+1]) rotates by pos * 10000^(-(2p mod head_size) / head_size)."""
    f = (np.arange(len(x) // 2) * 2) % head_size
    a = pos * 10000.0 ** (-f / head_size)
    c, s = np.cos(a), np.sin(a)
    y = np.empty_like(x)
    y[0::2] = x[0::2] * c - x[1::2] * s
    y[1::2] = x[0::2] * s + x[1::2] * c
    return y


def float_generate(cfg: Config, tensors: dict[str, Tensor], n_layers: int, steps: int,
                   inputs: list[int] = (BOS,)) -> tuple[list[int], dict[str, float]]:
    """Greedy tokens from the first n_layers layers, plus the peak |value| of every tensor the chip
    stores as int8: "res" (embedding and residual stream) and "l<N>.<name>" per layer. Position p
    reads inputs[p] while there is one, then the previous output token, like run.c with a prompt."""
    w = {name: t.load().astype(np.float64) for name, t in tensors.items()}
    hs, kvm = cfg.head_size, cfg.n_heads // cfg.n_kv_heads
    emb = w["token_embedding"]
    cls = w.get("wcls", emb)
    kc = np.zeros((n_layers, steps, cfg.kv_dim))
    vc = np.zeros_like(kc)
    peak: dict[str, float] = {}

    def note(name: str, v: np.ndarray) -> None:
        peak[name] = max(peak.get(name, 0.0), float(np.max(np.abs(v))))

    tokens = []
    for pos in range(steps):
        token = inputs[pos] if pos < len(inputs) else tokens[-1]
        x = emb[token].copy()
        note("res", x)
        for i in range(n_layers):
            p = f"layers.{i}."
            name = f"l{i}."
            xn = _rmsnorm(x, w[p + "rms_att"])
            q, k, v = w[p + "wq"] @ xn, w[p + "wk"] @ xn, w[p + "wv"] @ xn
            qr, kr = _rope(q, pos, hs), _rope(k, pos, hs)
            for n, a in (("xn", xn), ("q", q), ("k", k), ("v", v), ("qr", qr), ("kr", kr)):
                note(name + n, a)
            kc[i, pos], vc[i, pos] = kr, v
            att = np.zeros(cfg.dim)
            for h in range(cfg.n_heads):
                kv = slice((h // kvm) * hs, (h // kvm + 1) * hs)
                s = kc[i, :pos + 1, kv] @ qr[h * hs:(h + 1) * hs] / math.sqrt(hs)
                e = np.exp(s - s.max())
                att[h * hs:(h + 1) * hs] = (e / e.sum()) @ vc[i, :pos + 1, kv]
            x = x + w[p + "wo"] @ att
            note("res", x)
            xn = _rmsnorm(x, w[p + "rms_ffn"])
            h1, h3 = w[p + "w1"] @ xn, w[p + "w3"] @ xn
            g = h1 / (1 + np.exp(-h1)) * h3
            for n, a in (("xn2", xn), ("h3", h3), ("g", g)):
                note(name + n, a)
            x = x + w[p + "w2"] @ g
            note("res", x)
        xf = _rmsnorm(x, w["rms_final"])
        note("xf", xf)
        tokens.append(int(np.argmax(cls @ xf)))
    return tokens, peak


def decode(tokenizer: Path, tokens: list[int], prev: int = BOS) -> str:
    """Text for tokens with a llama2.c tokenizer file (tok<vocab>.bin), as run.c prints it."""
    data = tokenizer.read_bytes()
    pieces, off = [], 4  # int32 max_token_length, then per token: float32 score, int32 length, bytes
    while off < len(data):
        (n,) = struct.unpack_from("<i", data, off + 4)
        pieces.append(data[off + 8:off + 8 + n])
        off += 8 + n
    out = bytearray()
    for t in tokens:
        piece = pieces[t]
        if prev == BOS and piece.startswith(b" "):
            piece = piece[1:]
        if len(piece) == 6 and piece.startswith(b"<0x") and piece.endswith(b">"):
            piece = bytes([int(piece[3:5], 16)])
        out += piece
        prev = t
    return out.decode("utf-8", "replace")
