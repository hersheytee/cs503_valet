"""
Model downloader for Upper_Bound VLM experiments.

Usage
-----
    # Download all models
    python download_models.py

    # Download specific models
    python download_models.py qwen3b qwen7b

    # List available models without downloading
    python download_models.py --list

    # Custom cache directory
    python download_models.py --cache_dir /scratch/your_username/vlm_models
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ── Model registry ─────────────────────────────────────────────────────────────
# Single source of truth — imported by vlm_oracle.py

MODELS = {
    "qwen3b": {
        "name":    "Qwen2.5-VL 3B",
        "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "size_gb": 6.5,
        "dtype":   "float16",
        "gpu_mem": 0.40,
    },
    "qwen7b": {
        "name":    "Qwen2.5-VL 7B",
        "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "size_gb": 15.0,
        "dtype":   "float16",
        "gpu_mem": 0.70,
    },
    "internvl": {
        "name":    "InternVL2.5 4B",
        "repo_id": "OpenGVLab/InternVL2_5-4B",
        "size_gb": 8.0,
        "dtype":   "float16",
        "gpu_mem": 0.80,
    },
    "smolvlm": {
        "name":    "SmolVLM 2B",
        "repo_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "size_gb": 4.5,
        "dtype":   "float16",
        "gpu_mem": 0.40,
    },
    "internvl3": {
        "name":    "InternVL3 8B",
        "repo_id": "OpenGVLab/InternVL3-8B",
        "size_gb": 16.0,
        "dtype":   "float16",
        "gpu_mem": 0.90,
    },
    "wethink": {
        "name":    "WeThink-Qwen2.5VL 7B",
        "repo_id": "yangjie-cv/WeThink-Qwen2.5VL-7B",
        "size_gb": 15.0,
        "dtype":   "float16",
        "gpu_mem": 0.70,
    },
    "qwen2vl": {
        "name":    "Qwen2-VL 7B",
        "repo_id": "Qwen/Qwen2-VL-7B-Instruct",
        "size_gb": 15.0,
        "dtype":   "float16",
        "gpu_mem": 0.70,
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def is_downloaded(repo_id: str, cache_dir: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, cache_dir=cache_dir, local_files_only=True)
        return True
    except Exception:
        return False


def download(repo_id: str, cache_dir: str, model_name: str) -> bool:
    from huggingface_hub import snapshot_download

    print(f"  Downloading {model_name} ({repo_id}) ...")
    t0 = time.time()
    try:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*",
                             "rust_model*", "*.ot", "*.onnx"],
        )
        print(f"  Done in {(time.time() - t0) / 60:.1f} min.")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Download VLMs for Upper_Bound experiments")
    p.add_argument(
        "models", nargs="*",
        help=f"Model keys to download. Defaults to all. Choices: {list(MODELS)}",
    )
    p.add_argument(
        "--cache_dir", default=os.environ.get("HF_HOME", "./hf_cache"),
        help="HuggingFace cache directory (default: $HF_HOME or ./hf_cache)",
    )
    p.add_argument(
        "--list", action="store_true",
        help="List available models and their cache status, then exit.",
    )
    args = p.parse_args()

    cache_dir = str(Path(args.cache_dir).resolve())
    os.environ["HF_HOME"]               = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

    # Validate requested keys
    keys = args.models if args.models else list(MODELS)
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        print(f"Unknown model key(s): {unknown}")
        print(f"Available: {list(MODELS)}")
        sys.exit(1)

    selected = {k: MODELS[k] for k in keys}

    print("=" * 62)
    print("  Upper_Bound — VLM Model Downloader")
    print("=" * 62)
    print(f"  Cache : {cache_dir}")
    print()

    for key, m in selected.items():
        cached = is_downloaded(m["repo_id"], cache_dir)
        status = "cached" if cached else "missing"
        print(f"  {'OK' if cached else '--'}  {key:12s}  {m['name']:25s}  ~{m['size_gb']:.0f} GB  [{status}]")

    if args.list:
        return

    print()
    to_download = {k: m for k, m in selected.items()
                   if not is_downloaded(m["repo_id"], cache_dir)}

    if not to_download:
        print("  All selected models already in cache.")
        return

    total_gb = sum(m["size_gb"] for m in to_download.values())
    print(f"  Downloading {len(to_download)} model(s) (~{total_gb:.0f} GB) ...\n")

    results = {}
    for key, m in to_download.items():
        print(f"[{key}] {m['name']}")
        results[key] = download(m["repo_id"], cache_dir, m["name"])
        print()

    print("=" * 62)
    for key, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {MODELS[key]['name']}")

    if not all(results.values()):
        print("\n  Some downloads failed. Check your connection and HF login.")
        sys.exit(1)

    print(f"\n  All models ready in: {cache_dir}")
    print(f"  Run before training:  export HF_HOME={cache_dir}")


if __name__ == "__main__":
    main()
