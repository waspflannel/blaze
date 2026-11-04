"""`blaze` command line entry point."""
import argparse
import json
import math
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from blaze.generate import generate_matrix, quantize
from blaze.spec import load_matrix
from blaze.spec.scan import UNUSED_TENSORS, build_spec, count_params, read_checkpoint

# Karpathy's tiny Llama checkpoints: <name>/<file> inside the HF repo.
REPO = "karpathy/tinyllamas"
MODELS = {"stories260K": ["stories260K.bin", "tok512.bin"]}
MATMUL_TENSORS = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]


def fetch_model(name: str, out: str) -> None:
    for f in MODELS[name]:
        path = Path(hf_hub_download(REPO, f"{name}/{f}", local_dir=out))
        print(f"{path}  {path.stat().st_size} bytes")


def scan_model(checkpoint: Path, out: Path) -> None:
    cfg, tensors = read_checkpoint(checkpoint)
    spec = build_spec(checkpoint, cfg, tensors)
    spec_path = out / spec["model"] / "spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    # Force \n so the spec is byte-identical on every OS.
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", newline="\n")

    # One row per tensor kind; per-layer tensors ("layers.3.wq") collapse into one row with a count.
    rows: dict[str, list] = {}
    for name, t in tensors.items():
        kind = name.split(".")[-1]
        rows.setdefault(kind, [t.shape, 0])[1] += 1
    n_matmul = sum(op["op"] == "matmul" for op in spec["ops"])
    print(f"model    {spec['model']} ({spec['architecture']}), {checkpoint.stat().st_size} bytes")
    print(f"layers   {cfg.n_layers}  dim {cfg.dim}  hidden {cfg.hidden_dim}  heads {cfg.n_heads} (kv {cfg.n_kv_heads})"
          f"  vocab {cfg.vocab_size}  seq {cfg.seq_len}")
    print(f"params   {count_params(tensors):,} ({'shared' if cfg.shared_classifier else 'separate'} classifier)")
    print(f"ops      {len(spec['ops'])} ({n_matmul} matmul, {len(spec['ops']) - n_matmul} template)")
    print()
    print(f"{'tensor':<16} {'shape':<12} {'count':>5} {'params':>8}")
    for kind, (shape, n) in rows.items():
        params = "unused" if kind in UNUSED_TENSORS else f"{n * math.prod(shape):,}"
        print(f"{kind:<16} {' x '.join(map(str, shape)):<12} {n:>5} {params:>8}")
    print()
    print(f"wrote    {spec_path}")


def gen_matrix(model: Path, layer: int, tensor: str, out: Path) -> None:
    weights = load_matrix(model, layer, tensor)
    q, scale = quantize(weights)
    module = f"l{layer}_{tensor}_matvec"
    verilog, equations, stats = generate_matrix(q, scale, module, f"{model.name} layer {layer} {tensor}")

    out.mkdir(parents=True, exist_ok=True)
    verilog_path = out / f"{module}.v"
    equations_path = out / f"{module}.txt"
    # Force \n so the generated files are byte-identical on every OS.
    verilog_path.write_text(verilog, newline="\n")
    equations_path.write_text(equations, newline="\n")

    print(f"shape    {stats['rows']} x {stats['cols']} (out x in)")
    print(f"nonzero  {stats['nonzero']} / {stats['rows'] * stats['cols']}")
    print(f"adders   {stats['adders']} ({stats['csd_terms']} CSD terms)")
    print(f"scale    {stats['scale']!r}; y is signed {stats['acc_width']}-bit")
    print(f"wrote    {verilog_path}, {equations_path}")


def main() -> None:
    p = argparse.ArgumentParser(prog="blaze")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch-model", help="download a model checkpoint + tokenizer into models/<name>/")
    f.add_argument("name", choices=MODELS)
    f.add_argument("--models-dir", default="models", help="default: ./models")
    s = sub.add_parser("scan-model", help="read a checkpoint and write its model spec to build/<model>/spec.json")
    s.add_argument("checkpoint", type=Path, help="llama2.c .bin checkpoint, e.g. models/stories260K/stories260K.bin")
    s.add_argument("--out", type=Path, default=Path("build"), help="output directory (default: ./build)")
    g = sub.add_parser("gen-matrix", help="turn one weight matrix into shift/add equations and Verilog")
    g.add_argument("--model", type=Path, default=Path("models/stories260K/stories260K.bin"),
                   help="llama2.c .bin checkpoint (default: models/stories260K/stories260K.bin)")
    g.add_argument("--layer", type=int, default=0, help="default: 0")
    g.add_argument("--tensor", choices=MATMUL_TENSORS, default="wq", help="default: wq (query projection)")
    g.add_argument("--out", type=Path, default=Path("build"), help="output directory (default: ./build)")
    a = p.parse_args()

    if a.cmd == "fetch-model":
        fetch_model(a.name, a.models_dir)
    elif a.cmd == "scan-model":
        try:
            scan_model(a.checkpoint, a.out)
        except (OSError, ValueError) as e:
            sys.exit(f"blaze scan-model: {e}")
    elif a.cmd == "gen-matrix":
        try:
            gen_matrix(a.model, a.layer, a.tensor, a.out)
        except (OSError, ValueError) as e:
            sys.exit(f"blaze gen-matrix: {e}")
