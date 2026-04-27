import argparse
import os
import sys
import time
from pathlib import Path

MODELS = {
    # "gemma": {
    #     "name":     "Gemma 3 4B",
    #     "repo_id":  "google/gemma-3-4b-it",
    #     "size_gb":  8.5,
    #     "gated":    True,   # requires HF login + license acceptance
    #     "notes":    "Accept license at huggingface.co/google/gemma-3-4b-it",
    # },
    "qwen3b": {
        "name":     "Qwen2.5-VL 3B",
        "repo_id":  "Qwen/Qwen2.5-VL-3B-Instruct",
        "size_gb":  6.5,
        "gated":    False,
        "notes":    "",
    },
    "qwen7b": {
        "name":     "Qwen2.5-VL 7B",
        "repo_id":  "Qwen/Qwen2.5-VL-7B-Instruct",
        "size_gb":  15.0,
        "gated":    False,
        "notes":    "",
    },
    "internvl": {
        "name":     "InternVL2.5 4B",
        "repo_id":  "OpenGVLab/InternVL2_5-4B",
        "size_gb":  8.0,
        "gated":    False,
        "notes":    "",
    },
    "smolvlm": {
        "name":     "SmolVLM 2B",
        "repo_id":  "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "size_gb":  4.5,
        "gated":    False,
        "notes":    "",
    },
    "internvl3": {
        "name":     "InternVL3 8B",
        "repo_id":  "OpenGVLab/InternVL3-8B",
        "size_gb":  16.0,
        "gated":    False,
        "notes":    "",
    },
    "internvl8b_mpo": {
        "name":    "InternVL2.5 8B MPO",
        "repo_id": "OpenGVLab/InternVL2_5-8B-MPO",
        "size_gb": 16.0,
        "gated":   False,
        "notes":   "",
    },
    "wethink": {
        "name":    "WeThink-Qwen2.5VL 7B",
        "repo_id": "yangjie-cv/WeThink-Qwen2.5VL-7B",
        "size_gb": 15.0,
        "gated":   False,
        "notes":   "",
    },
    "qwen2vl": {
        "name":     "Qwen2-VL 7B",
        "repo_id":  "Qwen/Qwen2-VL-7B-Instruct",
        "size_gb":  15.0,
        "gated":    False,
        "notes":    "",
    },
    # "llava": {
    #     "name":     "LLaVA-OneVision 7B",
    #     "repo_id":  "lmms-lab/llava-onevision-qwen2-7b-ov",
    #     "size_gb":  15.0,
    #     "gated":    False,
    #     "notes":    "",
    # },
}


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _check_hf_login():
    """Warns if the user is not logged in to HuggingFace."""
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"  ✅  Logged in as: {user['name']}")
        return True
    except Exception:
        print("  ⚠️   Not logged in to HuggingFace.")
        print("      Run: huggingface-cli login")
        print("      (required for gated models like Gemma 3)")
        return False


def _is_already_downloaded(repo_id: str, cache_dir: str) -> bool:
    """
    Checks whether a model appears to already be in the cache.
    Uses snapshot_download with local_files_only=True as a probe.
    """
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        return True
    except Exception:
        return False


def _download_model(repo_id: str, cache_dir: str, model_name: str) -> bool:
    """
    Downloads a model from HuggingFace Hub using snapshot_download.
    Returns True on success, False on failure.
    """
    from huggingface_hub import snapshot_download

    print(f"\n  Downloading {model_name} ({repo_id}) ...")
    t0 = time.time()

    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            # Only download model weights + tokenizer, skip large extra files
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*",
                             "rust_model*", "*.ot", "*.onnx"],
        )
        elapsed = time.time() - t0
        print(f"  ✅  Done in {elapsed/60:.1f} min  →  {local_path}")
        return True

    except Exception as e:
        print(f"  ❌  Failed: {e}")
        return False



# main

def download_all(model_keys: list[str], cache_dir: str, dry_run: bool,
                 skip_existing: bool):

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Set HF_HOME so vLLM and transformers find the cache automatically
    os.environ["HF_HOME"]            = str(cache_path)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_path)

    selected = {k: MODELS[k] for k in model_keys if k in MODELS}
    if not selected:
        print("No valid model keys provided. Available keys:")
        for k, m in MODELS.items():
            print(f"  {k:10s} — {m['name']}")
        sys.exit(1)

    total_gb = sum(m["size_gb"] for m in selected.values())

    print("=" * 62)
    print("  MiniGrid VLM Benchmark — Model Downloader")
    print("=" * 62)
    print(f"  Cache directory : {cache_path.resolve()}")
    print(f"  Models selected : {len(selected)}")
    print(f"  Estimated size  : ~{total_gb:.1f} GB")
    print()

    for key, meta in selected.items():
        gated_tag = " [GATED — login required]" if meta["gated"] else ""
        notes_tag = f"\n      ↳ {meta['notes']}" if meta["notes"] else ""
        print(f"  • {meta['name']:25s}  ~{meta['size_gb']:.1f} GB{gated_tag}{notes_tag}")

    if dry_run:
        print("\n  Dry run — nothing downloaded.")
        return

    print()

    # Check HF login (warn only — non-gated models don't need it)
    _check_hf_login()

    results = {}

    for key, meta in selected.items():
        repo_id    = meta["repo_id"]
        model_name = meta["name"]

        print(f"\n{'─'*62}")
        print(f"  [{key}] {model_name}")
        print(f"  Repo : {repo_id}")

        # Skip if already cached
        if skip_existing and _is_already_downloaded(repo_id, cache_dir):
            print("Already in cache — skipping.")
            results[key] = "skipped"
            continue

        if meta["gated"]:
            print(f"Gated model — make sure you accepted the license at:")
            print(f"huggingface.co/{repo_id}")

        success = _download_model(repo_id, cache_dir, model_name)
        results[key] = "ok" if success else "failed"

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  Download summary")
    print(f"{'='*62}")
    for key, status in results.items():
        icon = {"ok": "ok", "skipped": "skipped", "failed": "failed"}.get(status, "?")
        print(f"  {icon}  {MODELS[key]['name']:25s}  [{status}]")

    failed = [k for k, s in results.items() if s == "failed"]
    if failed:
        print(f"\n  Failed  {len(failed)} model(s) failed to download.")
        print("      Check your internet connection and HF login status.")
        sys.exit(1)
    else:
        print(f"\n  All models ready in: {cache_path.resolve()}")
        print()
        print("  Next step — set this in your environment before running vLLM:")
        print(f"    export HF_HOME={cache_path.resolve()}")
        print(f"    export HUGGINGFACE_HUB_CACHE={cache_path.resolve()}")


# CLI

def parse_args():
    p = argparse.ArgumentParser(
        description="Download VLMs for the MiniGrid benchmark (run on login node)"
    )
    p.add_argument(
        "--models", nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help=(
            "Which models to download. "
            "Choices: gemma qwen3b qwen7b internvl smolvlm llava. "
            "Default: all."
        ),
    )
    p.add_argument(
        "--cache_dir", type=str,
        default=os.environ.get("HF_HOME", "./hf_cache"),
        help=(
            "Local directory for the HuggingFace model cache. "
            "On Izar, use your scratch: /scratch/izar/your_username/vlm_models. "
            f"Default: $HF_HOME or ./hf_cache"
        ),
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be downloaded without actually downloading.",
    )
    p.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="Skip models that are already in the cache (default: True).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_all(
        model_keys=args.models,
        cache_dir=args.cache_dir,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
    )