"""`blaze` command line entry point."""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

# Karpathy's tiny Llama checkpoints: <name>/<file> inside the HF repo.
REPO = "karpathy/tinyllamas"
MODELS = {"stories260K": ["stories260K.bin", "tok512.bin"]}


def fetch_model(name: str, out: str) -> None:
    for f in MODELS[name]:
        path = Path(hf_hub_download(REPO, f"{name}/{f}", local_dir=out))
        print(f"{path}  {path.stat().st_size} bytes")


def main() -> None:
    p = argparse.ArgumentParser(prog="blaze")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch-model", help="download a model checkpoint + tokenizer into models/<name>/")
    f.add_argument("name", choices=MODELS)
    f.add_argument("--models-dir", default="models", help="default: ./models")
    a = p.parse_args()
    fetch_model(a.name, a.models_dir)
