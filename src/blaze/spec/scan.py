"""Scan a checkpoint into a model spec: config plus every op of one forward pass, in order.

A reader returns (Config, tensors): tensors maps a name ("token_embedding", "layers.0.wq", ...) to a
Tensor (shape, dtype, byte offset in the file). Only the llama2.c `.bin` reader exists. A
config.json + safetensors reader can return the same two things, because the safetensors header
already gives each tensor's dtype, shape and offsets.
"""
import hashlib
import math
from pathlib import Path

import numpy as np

from blaze.spec import Config, Tensor, read_llama2c

# llama2.c run.c hardcodes both.
RMS_EPS = 1e-5
ROPE_THETA = 10000.0
# Precomputed RoPE tables in legacy exports; run.c ignores them and computes RoPE from ROPE_THETA.
UNUSED_TENSORS = {"freq_cis_real", "freq_cis_imag"}


def read_checkpoint(path: Path) -> tuple[Config, dict[str, Tensor]]:
    if path.suffix == ".bin":
        return read_llama2c(path)
    raise ValueError(f"{path}: unsupported checkpoint; only llama2.c .bin is supported so far")


def count_params(tensors: dict[str, Tensor]) -> int:
    return sum(math.prod(t.shape) for name, t in tensors.items() if name not in UNUSED_TENSORS)


def build_spec(path: Path, cfg: Config, tensors: dict[str, Tensor]) -> dict:
    """The model spec as a JSON-ready dict. Same checkpoint in, same dict out."""

    def ref(name: str) -> dict:
        t = tensors[name]
        return {"tensor": name, "shape": list(t.shape), "dtype": t.dtype, "offset": t.offset}

    def template(name: str, op: str, layer: int | None, **params) -> dict:
        return {"name": name, "op": op, "template": True, "layer": layer, "params": params}

    def matmul(name: str, tensor: str, layer: int | None) -> dict:
        w = tensors[tensor].load()
        stats = {
            "min": float(w.min()),
            "max": float(w.max()),
            "mean_abs": float(np.abs(w.astype(np.float64)).mean()),
            "pct_zero": 100.0 * int(np.count_nonzero(w == 0)) / w.size,
        }
        return {"name": name, "op": "matmul", "template": False, "layer": layer, **ref(tensor), "stats": stats}

    attention = {
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
        "head_size": cfg.head_size,
        "score_scale": 1 / math.sqrt(cfg.head_size),
        "softmax": "causal, over positions 0..pos",
        "kv_cache": {"k": [cfg.seq_len, cfg.kv_dim], "v": [cfg.seq_len, cfg.kv_dim], "dtype": "float32"},
    }
    rope = {"head_size": cfg.head_size, "theta": ROPE_THETA, "applies_to": ["q", "k"]}

    ops = [template("embedding", "embedding", None, vocab_size=cfg.vocab_size, dim=cfg.dim, table=ref("token_embedding"))]
    for i in range(cfg.n_layers):
        p = f"layers.{i}."
        ops += [
            template(p + "attn_norm", "rmsnorm", i, dim=cfg.dim, eps=RMS_EPS, weight=ref(p + "rms_att")),
            matmul(p + "wq", p + "wq", i),
            matmul(p + "wk", p + "wk", i),
            matmul(p + "wv", p + "wv", i),
            template(p + "rope", "rope", i, **rope),
            template(p + "attention", "attention", i, **attention),
            matmul(p + "wo", p + "wo", i),
            template(p + "attn_residual", "residual_add", i, dim=cfg.dim),
            template(p + "ffn_norm", "rmsnorm", i, dim=cfg.dim, eps=RMS_EPS, weight=ref(p + "rms_ffn")),
            matmul(p + "w1", p + "w1", i),
            matmul(p + "w3", p + "w3", i),
            template(p + "silu_gate", "silu_mul", i, dim=cfg.hidden_dim, formula="silu(w1 x) * (w3 x)"),
            matmul(p + "w2", p + "w2", i),
            template(p + "ffn_residual", "residual_add", i, dim=cfg.dim),
        ]
    ops += [
        template("final_norm", "rmsnorm", None, dim=cfg.dim, eps=RMS_EPS, weight=ref("rms_final")),
        matmul("classifier", "token_embedding" if cfg.shared_classifier else "wcls", None),
    ]

    return {
        "model": path.stem,
        "architecture": "llama",
        "source": {
            "file": path.name,
            "format": "llama2c",
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "config": {
            "dim": cfg.dim,
            "hidden_dim": cfg.hidden_dim,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "n_kv_heads": cfg.n_kv_heads,
            "head_size": cfg.head_size,
            "kv_dim": cfg.kv_dim,
            "vocab_size": cfg.vocab_size,
            "seq_len": cfg.seq_len,
            "shared_classifier": cfg.shared_classifier,
        },
        "params": count_params(tensors),
        "ops": ops,
    }
