"""`blaze` command line entry point."""
import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from blaze.generate import generate_matrix, quantize
from blaze.spec import load_matrix

# Karpathy's tiny Llama checkpoints: <name>/<file> inside the HF repo.
REPO = "karpathy/tinyllamas"
MODELS = {"stories260K": ["stories260K.bin", "tok512.bin"]}
MATMUL_TENSORS = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]


def fetch_model(name: str, out: str) -> None:
    for f in MODELS[name]:
        path = Path(hf_hub_download(REPO, f"{name}/{f}", local_dir=out))
        print(f"{path}  {path.stat().st_size} bytes")


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
    g = sub.add_parser("gen-matrix", help="turn one weight matrix into shift/add equations and Verilog")
    g.add_argument("--model", type=Path, default=Path("models/stories260K/stories260K.bin"),
                   help="llama2.c .bin checkpoint (default: models/stories260K/stories260K.bin)")
    g.add_argument("--layer", type=int, default=0, help="default: 0")
    g.add_argument("--tensor", choices=MATMUL_TENSORS, default="wq", help="default: wq (query projection)")
    g.add_argument("--out", type=Path, default=Path("build"), help="output directory (default: ./build)")
    a = p.parse_args()

    if a.cmd == "fetch-model":
        fetch_model(a.name, a.models_dir)
    elif a.cmd == "gen-matrix":
        try:
            gen_matrix(a.model, a.layer, a.tensor, a.out)
        except (OSError, ValueError) as e:
            sys.exit(f"blaze gen-matrix: {e}")
