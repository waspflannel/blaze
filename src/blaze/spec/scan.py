"""Scan a checkpoint into a model spec: config plus every op of one forward pass, in order.

A reader returns (Config, tensors): tensors maps a name ("token_embedding", "layers.0.wq", ...) to a
Tensor (shape, dtype, byte offset in the file). Only the llama2.c `.bin` reader exists. A
config.json + safetensors reader can return the same two things, because the safetensors header
already gives each tensor's dtype, shape and offsets.
"""
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blaze.spec import FLOAT_BYTES, HEADER_BYTES, Config, _tensor_sizes, read_config

# llama2.c run.c hardcodes both.
RMS_EPS = 1e-5
ROPE_THETA = 10000.0
# Precomputed RoPE tables in legacy exports; run.c ignores them and computes RoPE from ROPE_THETA.
UNUSED_TENSORS = {"freq_cis_real", "freq_cis_imag"}


@dataclass(frozen=True)
class Tensor:
    file: Path
    shape: tuple[int, ...]
    dtype: str  # numpy dtype name, stored little-endian in the file
    offset: int  # byte offset of the first element

    def load(self) -> np.ndarray:
        dtype = np.dtype(self.dtype).newbyteorder("<")
        data = np.fromfile(self.file, dtype=dtype, count=math.prod(self.shape), offset=self.offset)
        return data.reshape(self.shape)


def read_llama2c(path: Path) -> tuple[Config, dict[str, Tensor]]:
    cfg = read_config(path)
    sizes = _tensor_sizes(cfg)
    expected_bytes = HEADER_BYTES + FLOAT_BYTES * sum(n for _, n in sizes)
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"{path}: {actual_bytes} bytes, header implies {expected_bytes}; not a legacy llama2.c export")

    matmuls = cfg.matmul_shapes()
    shapes = {
        "token_embedding": (cfg.vocab_size, cfg.dim),
        "rms_att": (cfg.dim,),
        "rms_ffn": (cfg.dim,),
        "rms_final": (cfg.dim,),
        "freq_cis_real": (cfg.seq_len, cfg.head_size // 2),
        "freq_cis_imag": (cfg.seq_len, cfg.head_size // 2),
        "wcls": (cfg.vocab_size, cfg.dim),
        **matmuls,
    }
    per_layer = {"rms_att", "rms_ffn", *matmuls}

    tensors = {}
    offset = HEADER_BYTES
    for name, count in sizes:
        shape = shapes[name]
        if name in per_layer:
            # All layers of one tensor are stored back to back.
            layer_bytes = FLOAT_BYTES * math.prod(shape)
            for i in range(cfg.n_layers):
                tensors[f"layers.{i}.{name}"] = Tensor(path, shape, "float32", offset + i * layer_bytes)
        else:
            tensors[name] = Tensor(path, shape, "float32", offset)
        offset += FLOAT_BYTES * count
    return cfg, tensors


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
