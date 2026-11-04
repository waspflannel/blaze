"""Read a llama2.c checkpoint (Karpathy's legacy `.bin` export) into a config and weight matrices.

File layout: a 7 x int32 header, then float32 tensors back to back in this order:
token_embedding, rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3, rms_final, freq_cis_real,
freq_cis_imag, and wcls only when the classifier is not shared (negative vocab_size in the header).
Matmul weights are stored PyTorch-style as (out_features, in_features), row-major, per layer.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_BYTES = 7 * 4
FLOAT_BYTES = 4


@dataclass(frozen=True)
class Config:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
    shared_classifier: bool

    @property
    def head_size(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.head_size * self.n_kv_heads

    def matmul_shapes(self) -> dict[str, tuple[int, int]]:
        """(out, in) shape of each per-layer matmul weight."""
        d, h, kv = self.dim, self.hidden_dim, self.kv_dim
        return {"wq": (d, d), "wk": (kv, d), "wv": (kv, d), "wo": (d, d), "w1": (h, d), "w2": (d, h), "w3": (h, d)}


def read_config(path: Path) -> Config:
    header = np.fromfile(path, dtype="<i4", count=7)
    if header.size != 7:
        raise ValueError(f"{path}: too short for a llama2.c header")
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = (int(v) for v in header)
    if min(dim, hidden_dim, n_layers, n_heads, n_kv_heads, seq_len, abs(vocab_size)) <= 0 or dim % n_heads:
        raise ValueError(f"{path}: implausible header {header.tolist()}")
    return Config(dim, hidden_dim, n_layers, n_heads, n_kv_heads, abs(vocab_size), seq_len, vocab_size > 0)


def _tensor_sizes(cfg: Config) -> list[tuple[str, int]]:
    """(name, float count) of every tensor in file order; per-layer tensors cover all layers."""
    per_layer = {name: cfg.n_layers * rows * cols for name, (rows, cols) in cfg.matmul_shapes().items()}
    freq = cfg.seq_len * cfg.head_size // 2
    sizes = [
        ("token_embedding", cfg.vocab_size * cfg.dim),
        ("rms_att", cfg.n_layers * cfg.dim),
        ("wq", per_layer["wq"]),
        ("wk", per_layer["wk"]),
        ("wv", per_layer["wv"]),
        ("wo", per_layer["wo"]),
        ("rms_ffn", cfg.n_layers * cfg.dim),
        ("w1", per_layer["w1"]),
        ("w2", per_layer["w2"]),
        ("w3", per_layer["w3"]),
        ("rms_final", cfg.dim),
        ("freq_cis_real", freq),
        ("freq_cis_imag", freq),
    ]
    if not cfg.shared_classifier:
        sizes.append(("wcls", cfg.vocab_size * cfg.dim))
    return sizes


def load_matrix(path: Path, layer: int, tensor: str) -> np.ndarray:
    """Return one layer's matmul weight as a float32 (out, in) array."""
    cfg = read_config(path)
    shapes = cfg.matmul_shapes()
    if tensor not in shapes:
        raise ValueError(f"unknown tensor {tensor!r}; expected one of {', '.join(shapes)}")
    if not 0 <= layer < cfg.n_layers:
        raise ValueError(f"layer {layer} out of range 0..{cfg.n_layers - 1}")

    sizes = _tensor_sizes(cfg)
    expected_bytes = HEADER_BYTES + FLOAT_BYTES * sum(n for _, n in sizes)
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"{path}: {actual_bytes} bytes, header implies {expected_bytes}; not a legacy llama2.c export")

    tensor_start = HEADER_BYTES
    for name, count in sizes:
        if name == tensor:
            break
        tensor_start += FLOAT_BYTES * count
    rows, cols = shapes[tensor]
    offset = tensor_start + FLOAT_BYTES * layer * rows * cols
    weights = np.fromfile(path, dtype="<f4", count=rows * cols, offset=offset)
    return weights.reshape(rows, cols)
