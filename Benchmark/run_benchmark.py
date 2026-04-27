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

from utils.prompt_builder import build_prompt, parse_response
from utils.debug import _debug_probe

# Prompts variants

_DIR_WORD = {"→": "RIGHT", "↓": "DOWN", "←": "LEFT", "↑": "UP"}


def _v_ifthen(s, view):
    """~170 tokens — decision tree / if-then rules."""
    d = s.get("agent_dir_str", "?")
    fw = _DIR_WORD.get(d, d)
    carrying = s.get("agent_carrying")
    carry_line = f"You are currently carrying a {carrying}." if carrying else ""
    return (
        f"Grid navigation. Agent faces {fw}. Mission: {s['mission']}\n"
        f"{carry_line}\n\n"
        f"Decision rules (apply the first that matches):\n"
        f"  → If the GREEN GOAL is directly {fw}: action 2 (forward)\n"
        f"  → If a KEY is directly {fw} and you need it: action 3 (pickup)\n"
        f"  → If a DOOR is directly {fw} and you have the key: action 5 (toggle)\n"
        f"  → If turning right gets you closer to your target: action 1\n"
        f"  → If turning left gets you closer to your target: action 0\n"
        f"  → Otherwise move forward: action 2\n\n"
        f"Based on the image, which action applies? YOU HAVE TO REPLY WITH ONE INTEGER ONLY."
    )


def _v_negative(s, view):
    """~180 tokens — tells what NOT to do + what to do."""
    d = s.get("agent_dir_str", "?")
    fw = _DIR_WORD.get(d, d)
    carrying = s.get("agent_carrying")
    carry_line = f"You carry a {carrying}." if carrying else "You carry nothing."
    return (
        f"Grid agent (red triangle) faces {fw}. {carry_line}\n"
        f"Mission: {s['mission']}\n\n"
        f"Rules:\n"
        f"  ✗ Do NOT pick up (action 3) unless a key is DIRECTLY {fw} of you.\n"
        f"  ✗ Do NOT toggle (action 5) unless a door is DIRECTLY {fw} of you AND you hold the key.\n"
        f"  ✗ Do NOT move forward (action 2) into a wall or closed door.\n"
        f"  ✓ DO pick up the key if it is directly ahead.\n"
        f"  ✓ DO open the door if you face it and hold the key.\n"
        f"  ✓ DO move forward when the path is clear and it brings you closer.\n"
        f"  ✓ DO turn (0 or 1) to face your next target.\n\n"
        f"Actions: 0=turn_left 1=turn_right 2=forward({fw}) 3=pickup 5=toggle\n"
        f"YOU HAVE TO REPLY WITH ONE INTEGER ONLY."
    )


def _v_verbose(s, view):
    """~310 tokens — full MiniGrid mechanics explanation."""
    d = s.get("agent_dir_str", "?")
    fw = _DIR_WORD.get(d, d)
    carrying = s.get("agent_carrying")
    carry_line = (
        f"The agent is currently carrying: a {carrying}."
        if carrying else "The agent is not carrying anything."
    )
    view_ctx = (
        "The image is a full top-down view of the grid. "
        f"Every cell is visible. The red triangle is the agent, looking {fw}."
    ) if view == "global" else (
        f"The image shows a 7×7 egocentric view. The agent is at the bottom-center, looking {fw}."
    )
    return (
        f"{view_ctx}\n\n"
        f"GAME MECHANICS:\n"
        f"- The grid contains: walls (dark border cells), floor (empty cells), "
        f"keys (yellow objects), doors (colored gates), and a green goal square.\n"
        f"- To win: reach the green goal square.\n"
        f"- If there is a door: you must first PICK UP the key (action 3 when facing it), "
        f"then OPEN the door (action 5 when facing it with key in hand), then go through.\n"
        f"- Walls and closed doors block movement.\n\n"
        f"CURRENT STATE:\n"
        f"- Mission: {s['mission']}\n"
        f"- Agent facing: {fw}\n"
        f"- {carry_line}\n\n"
        f"ACTIONS:\n"
        f"  0 = turn_left   — rotate 90° counter-clockwise (no movement)\n"
        f"  1 = turn_right  — rotate 90° clockwise (no movement)\n"
        f"  2 = forward     — move one cell {fw} (blocked by walls/closed doors)\n"
        f"  3 = pickup      — grab object in the cell directly {fw} of you\n"
        f"  5 = toggle      — open/close door in the cell directly {fw} of you\n\n"
        f"WHAT IS THE OPTIMAL BEST ACTION? REPLY WITH ONE INTEGER ONLY."
    )


PROMPT_VARIANTS = [
    {"id": 0, "name": "if_then",        "approx_tokens": 170, "fn": _v_ifthen,   "max_out": 16},
    {"id": 1, "name": "negative_rules", "approx_tokens": 180, "fn": _v_negative, "max_out": 16},
    {"id": 2, "name": "verbose",        "approx_tokens": 310, "fn": _v_verbose,  "max_out": 16},
]

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

        # Extract image block
        img_block = None
        for _msg in base_messages:
            if isinstance(_msg["content"], list):
                for _part in _msg["content"]:
                    if _part["type"] == "image_url":
                        img_block = _part
                        break
                if img_block:
                    break

        # Build variant-specific message (single user turn with image + variant text)
        user_text = variant["fn"](sample, view)
        if img_block:
            messages = [{"role": "user", "content": [
                img_block,
                {"type": "text", "text": user_text},
            ]}]
        else:
            messages = [{"role": "user", "content": user_text}]

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
