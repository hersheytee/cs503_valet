"""
MiniGrid VLM Benchmark — Main Runner
=====================================
Runs all VLMs sequentially on the dataset using vLLM, testing several prompt
variants (if_then, negative_rules, verbose) and saving separate outputs
per variant for cross-variant analysis.

Output structure
----------------
results/
  raw/
    qwen3b_global_variant.json
    qwen3b_partial_variant.json
    ...
  benchmark_results.csv
  benchmark_summary.json

Usage:
    python run_benchmark.py \\
        --dataset  ./dataset/dataset.json \\
        --results  ./results \\
        --cache_dir /scratch/your_username/vlm_models

    # Quick test — one model, global view, first 10 samples
    python run_benchmark.py \\
        --models   qwen3b \\
        --views    global \\
        --max_samples 10

Install:
    pip install vllm openai tqdm pandas
"""

import argparse
import json
import os
import sys
import time
import subprocess
import signal
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import sys
sys.path.append("../Benchmark")
from utils.prompt_builder import (
    build_prompt,
    parse_response,
    SYSTEM_PROMPT,
    VIEW_CONTEXT,
    SIMPLE_FORMAT_MODELS,
)
from utils.debug import _debug_probe

# Prompts variants 
from prompt_variants import PROMPT_VARIANTS



# Models configurations for vLLM
MODELS = {
    "qwen3b": {
        "name":    "Qwen2.5-VL 3B",
        "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "dtype":   "float16",
        "gpu_mem": 0.40,
    },
    "qwen7b": {
        "name":    "Qwen2.5-VL 7B",
        "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "dtype":   "float16",
        "gpu_mem": 0.70,
    },
    "internvl": {
        "name":    "InternVL2.5 4B",
        "repo_id": "OpenGVLab/InternVL2_5-4B",
        "dtype":   "float16",
        "gpu_mem": 0.80,
    },
    "smolvlm": {
        "name":    "SmolVLM 2B",
        "repo_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "dtype":   "float16",
        "gpu_mem": 0.30,
    },
    "internvl3": {
        "name":    "InternVL3 8B",
        "repo_id": "OpenGVLab/InternVL3-8B",
        "dtype":   "float16",
        "gpu_mem": 0.85,
    },
    "internvl8b_mpo": {
        "name":    "InternVL2.5 8B MPO",
        "repo_id": "OpenGVLab/InternVL2_5-8B-MPO",
        "dtype":   "float16",
        "gpu_mem": 0.85,
    },
    "wethink": {
        "name":    "WeThink-Qwen2.5VL 7B",
        "repo_id": "yangjie-cv/WeThink-Qwen2.5VL-7B",
        "dtype":   "float16",   # ← BF16 obligatoire
        "gpu_mem": 0.75,
    },
    "qwen2vl": {
        "name":    "Qwen2-VL 7B",
        "repo_id": "Qwen/Qwen2-VL-7B-Instruct",
        "dtype":   "float16",
        "gpu_mem": 0.70,
    },
}


VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_URL  = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"

# vLLM server management

def start_vllm_server(model_cfg: dict, cache_dir: str) -> subprocess.Popen:
    """Starts a vLLM OpenAI-compatible server as a subprocess."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  model_cfg["repo_id"],
        "--dtype",                  model_cfg["dtype"],
        "--gpu-memory-utilization", str(model_cfg["gpu_mem"]),
        "--host",                   VLLM_HOST,
        "--port",                   str(VLLM_PORT),
        "--trust-remote-code",
        "--max-model-len",          "4096",
        "--download-dir",           cache_dir,
    ]

    env = os.environ.copy()
    env["HF_HOME"]               = cache_dir
    env["HUGGINGFACE_HUB_CACHE"] = cache_dir
    env["TRANSFORMERS_OFFLINE"]  = "1"
    env["HF_DATASETS_OFFLINE"]   = "1"

    print(f"  Starting vLLM server for {model_cfg['name']} ...")
    print(f"  CMD: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=sys.stderr,   # vLLM logs → .err file for debugging
        stderr=sys.stderr,
    )
    return proc


def wait_for_server(timeout: int = 180) -> bool:
    """Polls the vLLM /health endpoint until ready or timeout."""
    import urllib.request

    health_url = f"http://{VLLM_HOST}:{VLLM_PORT}/health"
    t0 = time.time()

    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(health_url, timeout=2)
            return True
        except Exception:
            time.sleep(3)

    return False


def stop_vllm_server(proc: subprocess.Popen):
    """Terminates the vLLM server subprocess."""
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    time.sleep(5)   # wait for GPU memory to be released

# Inference

def run_inference(
    samples: list[dict],
    model_key: str,
    view: str,
    variant: dict,
    dataset_root: Path,
    results_raw_dir: Path,
    max_samples: int | None,
    debug: bool = False,
) -> list[dict]:
    """
    Sends all samples to the running vLLM server using the given prompt variant.
    Evaluates against optimal_actions (list) to handle equivalent actions.
    Saves raw results to {model_key}_{view}_{variant_name}.json.

    The prompt sent to the VLM is:
        SYSTEM_PROMPT (separate or merged depending on model)
        + VIEW_CONTEXT[view]   ← critical: explains what the image shows
        + variant-specific user text
    """
    from openai import OpenAI

    client    = OpenAI(base_url=VLLM_URL, api_key="vllm-local")
    model_cfg = MODELS[model_key]

    try:
        served_model_name = client.models.list().data[0].id
    except Exception:
        served_model_name = model_cfg["repo_id"]

    subset  = samples[:max_samples] if max_samples else samples
    results = []
    n_ok    = n_fail = n_parse = 0

    # Debug probe runs once per (model, view) — only for the first variant
    if debug and variant["id"] == PROMPT_VARIANTS[0]["id"]:
        valid_samples = [s for s in subset if s.get("oracle_valid", True)][:5]
        if valid_samples:
            _debug_probe(client, served_model_name, valid_samples, view, dataset_root, model_key)

    DEBUG_SHOW = 5

    pbar = tqdm(
        subset,
        desc=f"  {model_cfg['name']} [{view}] [{variant['name']}]",
        unit="sample",
    )

    for sample in pbar:
        if not sample.get("oracle_valid", True):
            continue

        # Build prompt to get image block + oracle metadata
        base_messages, meta = build_prompt(
            sample,
            view=view,
            model_key=model_key,
            dataset_root=dataset_root,
        )

        # Extract image block from base_messages
        img_block = None
        for _msg in base_messages:
            if isinstance(_msg["content"], list):
                for _part in _msg["content"]:
                    if _part["type"] == "image_url":
                        img_block = _part
                        break
                if img_block:
                    break

        # ── Build the full prompt: VIEW_CONTEXT + variant-specific text ──
        variant_text = variant["fn"](sample, view)
        full_user_text = f"{VIEW_CONTEXT[view]}\n\n{variant_text}"

        # ── Choose message format depending on the model ──
        # Some VLMs (InternVL, LLaVA, SmolVLM) cannot handle a separate
        # system message when content is a list → merge SYSTEM_PROMPT into user.
        if model_key in SIMPLE_FORMAT_MODELS:
            messages = [{
                "role": "user",
                "content": [
                    img_block,
                    {"type": "text",
                     "text": SYSTEM_PROMPT + "\n\n" + full_user_text},
                ] if img_block else SYSTEM_PROMPT + "\n\n" + full_user_text,
            }]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": [
                     img_block,
                     {"type": "text", "text": full_user_text},
                 ] if img_block else full_user_text},
            ]

        t_start = time.time()

        try:
            response = client.chat.completions.create(
                model=served_model_name,
                messages=messages,
                max_tokens=variant["max_out"],
                temperature=0.0,
                timeout=30,
            )
            raw_output = response.choices[0].message.content or ""
            latency_ms = int((time.time() - t_start) * 1000)
            n_ok += 1

        except Exception:
            raw_output = ""
            latency_ms = -1
            n_fail    += 1

        predicted_action = parse_response(raw_output)
        if predicted_action is None:
            n_parse += 1

        optimal_actions = meta["optimal_actions"]
        correct = predicted_action in optimal_actions

        if debug and len(results) < DEBUG_SHOW:
            icon = "✅" if correct else "❌"
            print(f"  {icon} [{meta['sample_id']}] raw={repr(raw_output[:60])} "
                  f"→ predicted={predicted_action}  oracle={optimal_actions}")

        result = {
            # Identification
            "sample_id":        meta["sample_id"],
            "model":            model_key,
            "model_name":       model_cfg["name"],
            "view":             view,
            "variant":          variant["name"],
            # Sample metadata
            "env":              meta["env"],
            "complexity":       meta["complexity"],
            "mission":          meta["mission"],
            "agent_carrying":   meta["agent_carrying"],
            # Oracle
            "optimal_actions":  optimal_actions,
            "action_names":     meta.get("action_names", []),
            # Prediction
            "raw_output":       raw_output.strip(),
            "predicted_action": predicted_action,
            "correct":          correct,
            "parse_failed":     predicted_action is None,
            "latency_ms":       latency_ms,
        }
        results.append(result)

        n_done    = len(results)
        n_correct = sum(r["correct"] for r in results)
        pbar.set_postfix(
            acc=f"{100*n_correct/max(n_done,1):.1f}%",
            fail=n_fail,
            parse_err=n_parse,
        )

    pbar.close()

    out_path = results_raw_dir / f"{model_key}_{view}_{variant['name']}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    n_done    = len(results)
    n_correct = sum(r["correct"] for r in results)
    print(f"  → [{variant['name']}] Accuracy: {n_correct}/{n_done} "
          f"({100*n_correct/max(n_done,1):.1f}%)"
          f"  |  API failures: {n_fail}  |  Parse errors: {n_parse}")
    print(f"  → Saved to {out_path}")

    return results

# Summary

def compute_summary(all_results: list[dict]) -> dict:
    """Accuracy breakdowns per (model, view, variant), complexity, and action."""
    summary = {}
    df      = pd.DataFrame(all_results)

    for (model, view, variant), grp in df.groupby(["model", "view", "variant"]):
        key = f"{model}_{view}_{variant}"

        by_complexity = {
            c: {"accuracy": round(s["correct"].mean(), 4), "n": len(s)}
            for c, s in grp.groupby("complexity")
        }
        exploded = grp.explode("action_names")
        by_action = {
            a: {"accuracy": round(s["correct"].mean(), 4), "n": len(s)}
            for a, s in exploded.groupby("action_names")
            if a and str(a) != "nan"
        }

        valid_lat = grp[grp["latency_ms"] > 0]["latency_ms"]

        summary[key] = {
            "model":           model,
            "model_name":      MODELS[model]["name"],
            "view":            view,
            "variant":         variant,
            "accuracy":        round(grp["correct"].mean(), 4),
            "n_samples":       len(grp),
            "n_correct":       int(grp["correct"].sum()),
            "n_parse_failed":  int(grp["parse_failed"].sum()),
            "latency_mean_ms": int(valid_lat.mean()) if len(valid_lat) else -1,
            "latency_p95_ms":  int(valid_lat.quantile(0.95)) if len(valid_lat) else -1,
            "by_complexity":   by_complexity,
            "by_action":       by_action,
        }

    return summary


def print_summary(summary: dict):
    """Prints a leaderboard to stdout."""
    W = 78

    rows = sorted(
        summary.values(),
        key=lambda x: (x["model"], x["view"], x["variant"]),
    )

    # Full table
    print(f"\n{'═'*W}")
    print("  BENCHMARK RESULTS")
    print(f"{'═'*W}")
    print(f"  {'Model':<25} {'View':<8} {'Variant':<16} {'Accuracy':>9}  {'N':>5}  {'Lat(ms)':>8}")
    print(f"  {'─'*25} {'─'*8} {'─'*16} {'─'*9}  {'─'*5}  {'─'*8}")
    for r in rows:
        print(
            f"  {r['model_name']:<25} {r['view']:<8} {r['variant']:<16} "
            f"{100*r['accuracy']:>8.1f}%  {r['n_samples']:>5}  "
            f"{r['latency_mean_ms']:>7}ms"
        )

    # Per-variant leaderboard 
    print(f"\n{'═'*W}")
    print("  PER-VARIANT LEADERBOARD  (models ranked within each variant × view)")
    print(f"{'═'*W}")
    for variant in sorted({r["variant"] for r in rows}):
        for view in sorted({r["view"] for r in rows}):
            subset = sorted(
                [r for r in rows if r["variant"] == variant and r["view"] == view],
                key=lambda x: -x["accuracy"],
            )
            if not subset:
                continue
            print(f"\n  Variant: {variant}  |  View: {view}")
            print(f"  {'Model':<25} {'Accuracy':>9}  {'N':>5}  {'Parse errors':>13}")
            print(f"  {'─'*25} {'─'*9}  {'─'*5}  {'─'*13}")
            for r in subset:
                print(f"  {r['model_name']:<25} {100*r['accuracy']:>8.1f}%  "
                      f"{r['n_samples']:>5}  {r['n_parse_failed']:>13}")

    # Per-action accuracy (global view)
    global_rows = [r for r in rows if r["view"] == "global"]
    all_actions = sorted({a for r in global_rows for a in r["by_action"]})
    if all_actions:
        print(f"\n{'═'*W}")
        print("  PER-ACTION ACCURACY  (global view)")
        print(f"{'═'*W}")
        label_w = 44
        print(f"  {'Model / Variant':<{label_w}}" + "".join(f"{a:>12}" for a in all_actions))
        print(f"  {'─'*label_w}" + "".join("─" * 12 for _ in all_actions))
        for r in global_rows:
            label = f"{r['model_name']} / {r['variant']}"
            row   = f"  {label:<{label_w}}"
            for a in all_actions:
                acc = r["by_action"].get(a, {}).get("accuracy")
                row += f"{100*acc:>11.1f}%" if acc is not None else f"{'N/A':>12}"
            print(row)
    print()


# main

def run_benchmark(
    dataset_path: str,
    results_dir: str,
    cache_dir: str,
    model_keys: list[str],
    views: list[str],
    max_samples: int | None,
    debug: bool = False,
):
    dataset_root    = Path(dataset_path).parent
    results_path    = Path(results_dir)
    results_raw_dir = results_path / "raw"
    results_raw_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"]               = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

    with open(dataset_path) as f:
        samples = json.load(f)
    print(f"\n  Dataset loaded: {len(samples)} samples from {dataset_path}")
    print(f"  Prompt variants: {[v['name'] for v in PROMPT_VARIANTS]}")

    all_results = []

    for model_key in model_keys:
        model_cfg = MODELS[model_key]
        print(f"\n{'═'*65}")
        print(f"  MODEL: {model_cfg['name']}  ({model_cfg['repo_id']})")
        print(f"{'═'*65}")

        proc = start_vllm_server(model_cfg, cache_dir)

        print("  Waiting for server to be ready (up to 10 min) ...")
        ready = wait_for_server(timeout=600)

        if not ready:
            if proc.poll() is not None:
                _, err = proc.communicate()
                if err:
                    print(f"  vLLM stderr: {err.decode()[:500]}", file=sys.stderr)
            print("Server did not start in time. Skipping this model.")
            stop_vllm_server(proc)
            continue

        print("  ✅  Server ready.")

        for view in views:
            for variant in PROMPT_VARIANTS:
                results = run_inference(
                    samples=samples,
                    model_key=model_key,
                    view=view,
                    variant=variant,
                    dataset_root=dataset_root,
                    results_raw_dir=results_raw_dir,
                    max_samples=max_samples,
                    debug=debug,
                )
                all_results.extend(results)

        print(f"  Stopping vLLM server ...")
        stop_vllm_server(proc)
        print(f"  GPU memory released.")

    if not all_results:
        print("\n  No results collected — exiting.")
        return

    df       = pd.DataFrame(all_results)
    csv_path = results_path / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved → {csv_path}")

    summary      = compute_summary(all_results)
    summary_path = results_path / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved → {summary_path}")

    print_summary(summary)



# CLI

def parse_args():
    p = argparse.ArgumentParser(
        description="Run VLM benchmark on MiniGrid dataset using vLLM"
    )
    p.add_argument("--dataset",     type=str, default="./dataset/dataset.json")
    p.add_argument("--results",     type=str, default="./results")
    p.add_argument("--cache_dir",   type=str,
                   default=os.environ.get("HF_HOME", "./hf_cache"))
    p.add_argument("--models",      nargs="+", default=list(MODELS.keys()),
                   choices=list(MODELS.keys()))
    p.add_argument("--views",       nargs="+", default=["global", "partial"],
                   choices=["global", "partial"])
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument(
        "--debug", action="store_true",
        help=(
            "Debug mode: before each (model, view) run, prints the full text prompt, "
            "compares output with vs without image (identical = image not processed), "
            "and tests all prompt variants on the first 5 samples."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        dataset_path=args.dataset,
        results_dir=args.results,
        cache_dir=args.cache_dir,
        model_keys=args.models,
        views=args.views,
        max_samples=args.max_samples,
        debug=args.debug,
    )
