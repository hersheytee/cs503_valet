"""
Batch zero-shot transfer evaluation across all available checkpoints.

Discovers every run with a final.pt checkpoint, evaluates it on one or more
target environments, and appends results to a CSV. Already-evaluated
(run_name, env_id) pairs are skipped so the script is safe to re-run.

Outputs:
    gym_sokoban/figures/final/transfer_results.csv

Usage:
    python gym_sokoban/eval_transfer_all.py
    python gym_sokoban/eval_transfer_all.py --n-episodes 300
    python gym_sokoban/eval_transfer_all.py --env-ids FixedTarget-Sokoban-v2 Sokoban-small-v0
"""

from __future__ import annotations

import os
import warnings

# Suppress gym deprecation warnings in this process and all spawned workers.
# PYTHONWARNINGS is inherited by child processes before Python initialises,
# so it fires before any import can trigger the warning.
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:gym,ignore::DeprecationWarning:gym")
warnings.filterwarnings("ignore", message=".*unmaintained.*")
warnings.filterwarnings("ignore", message=".*Gymnasium.*")
warnings.filterwarnings("ignore", category=UserWarning, module="gym")

import argparse
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RUN_ROOTS = [
    REPO_ROOT / "runs",
    SCRIPT_DIR / "sokoban_vast_results" / "runs",
]
DEFAULT_OUT = SCRIPT_DIR / "figures" / "final" / "transfer_results.csv"

sys.path.insert(0, str(SCRIPT_DIR))
from model import CNNPolicy
from env_wrapper import SokobanOracleWrapper

import numpy as _np
if not hasattr(_np, "bool8"):
    _np.bool8 = _np.bool_


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def as_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def discover_checkpoints(run_roots: list[Path], min_steps: int = 0) -> list[dict]:
    raw = []
    for root in run_roots:
        if not root.exists():
            continue
        for ckpt in sorted(root.glob("*/checkpoints/final.pt")):
            run_dir = ckpt.parent.parent
            config_path = run_dir / "config.yaml"
            if not config_path.exists():
                continue
            try:
                cfg = read_yaml(config_path)
            except Exception:
                continue
            args = cfg.get("args", {})
            run_name = str(cfg.get("run", {}).get("name") or run_dir.name)

            # Use actual completed steps from metrics.csv, not planned total
            metrics_path = run_dir / "data" / "metrics.csv"
            try:
                max_step = int(pd.read_csv(metrics_path)["global_step"].max())
            except Exception:
                max_step = 0

            raw.append({
                "checkpoint": ckpt,
                "run_name": run_name,
                "exp_name": str(args.get("exp_name") or run_name),
                "seed": int(args.get("seed", 0)),
                "no_oracle": bool(args.get("no_oracle", False)),
                "oracle_accuracy": as_float(args.get("oracle_accuracy")),
                "oracle_cost": as_float(args.get("oracle_cost")),
                "oracle_cost_final": as_float(args.get("oracle_cost_final")),
                "total_timesteps": int(args.get("total_timesteps", 0)),
                "hidden_dim": int(args.get("hidden_dim", 256)),
                "obs_size": int(args.get("obs_size", 56)),
                "max_step": max_step,
            })

    # Deduplicate by (exp_name, seed), keeping the longest run
    best: dict[tuple, dict] = {}
    for rec in raw:
        key = (rec["exp_name"], rec["seed"])
        if key not in best or rec["max_step"] > best[key]["max_step"]:
            best[key] = rec

    return [r for r in best.values() if r["max_step"] >= min_steps]


def load_results(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def already_evaluated(df: pd.DataFrame, run_name: str, env_id: str) -> bool:
    if df.empty:
        return False
    return not df[(df["run_name"] == run_name) & (df["env_id"] == env_id)].empty


def evaluate(
    ckpt: Path,
    no_oracle: bool,
    hidden_dim: int,
    obs_size: int,
    env_id: str,
    n_episodes: int,
    max_episode_steps: int,
    seed: int,
    oracle_accuracy: float = 1.0,
    progress_q=None,
    progress_label: str = "",
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = 9 if no_oracle else 10
    obs_shape = (obs_size, obs_size, 3)

    model = CNNPolicy(obs_shape=obs_shape, n_actions=n_actions, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(str(ckpt), map_location=device, weights_only=True))
    model.eval()

    env = SokobanOracleWrapper(
        env_id,
        oracle_cost=0.0,
        oracle_accuracy=oracle_accuracy,
        no_oracle=no_oracle,
        max_episode_steps=max_episode_steps,
        obs_size=obs_size,
        seed=seed,
    )

    successes, guided_pcts, queries = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        steps = guided = 0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.uint8).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(obs_t)[0].argmax(dim=-1).item()
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            if info.get("guided", False):
                guided += 1
        successes.append(float(info.get("success", False)))
        guided_pcts.append(100.0 * guided / steps if steps > 0 else 0.0)
        queries.append(guided)
        if progress_q is not None:
            try:
                progress_q.put_nowait((progress_label, ep + 1, n_episodes))
            except Exception:
                pass

    env.close()
    return {
        "success_rate": float(np.mean(successes)),
        "success_std": float(np.std(successes)),
        "oracle_usage_pct": float(np.mean(guided_pcts)),
        "oracle_usage_pct_std": float(np.std(guided_pcts)),
        "queries_per_ep": float(np.mean(queries)),
        "queries_per_ep_std": float(np.std(queries)),
        "n_episodes": n_episodes,
    }


# ── Worker process helpers (top-level for multiprocessing pickle) ─────────────

def _suppress_gym_warnings():
    """Initializer run in each worker process before any tasks execute."""
    import warnings
    warnings.filterwarnings("ignore", message=".*unmaintained.*")
    warnings.filterwarnings("ignore", message=".*Gymnasium.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="gym")

def _eval_task(task: dict) -> tuple[dict, dict | None]:
    try:
        metrics = evaluate(
            ckpt=task["ckpt"],
            no_oracle=task["no_oracle"],
            hidden_dim=task["hidden_dim"],
            obs_size=task["obs_size"],
            env_id=task["env_id"],
            n_episodes=task["n_episodes"],
            max_episode_steps=task["max_episode_steps"],
            seed=task["seed"],
            oracle_accuracy=task["oracle_accuracy"],
            progress_q=task.get("_progress_q"),
            progress_label=task["meta"]["exp_name"],
        )
        return task["meta"], metrics
    except Exception:
        traceback.print_exc()
        return task["meta"], None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-roots", nargs="+", default=[str(p) for p in DEFAULT_RUN_ROOTS],
    )
    parser.add_argument(
        "--env-ids", nargs="+", default=["FixedTarget-Sokoban-v2"],
        help="Target environments to evaluate on.",
    )
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-steps", type=int, default=450_000,
                        help="Skip checkpoints from runs shorter than this many env steps.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel worker processes (default: min(cpu_count, 16)).")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_roots = [Path(r) for r in args.run_roots]
    checkpoints = discover_checkpoints(run_roots, min_steps=args.min_steps)
    if not checkpoints:
        raise SystemExit(f"No checkpoints with >= {args.min_steps} actual steps. Check --run-roots or lower --min-steps.")

    results_df = load_results(out_path)

    # Build task list, skipping already-evaluated pairs
    tasks = []
    for rec in checkpoints:
        for env_id in args.env_ids:
            if already_evaluated(results_df, rec["run_name"], env_id):
                print(f"  [skip] {rec['exp_name']} on {env_id}")
                continue
            tasks.append({
                "ckpt": rec["checkpoint"],
                "no_oracle": rec["no_oracle"],
                "hidden_dim": rec["hidden_dim"],
                "obs_size": rec["obs_size"],
                "env_id": env_id,
                "n_episodes": args.n_episodes,
                "max_episode_steps": args.max_episode_steps,
                "seed": args.seed,
                "oracle_accuracy": rec["oracle_accuracy"] if rec["oracle_accuracy"] is not None else 1.0,
                "meta": {
                    "run_name": rec["run_name"],
                    "exp_name": rec["exp_name"],
                    "seed": rec["seed"],
                    "no_oracle": rec["no_oracle"],
                    "oracle_accuracy": rec["oracle_accuracy"],
                    "oracle_cost": rec["oracle_cost"],
                    "oracle_cost_final": rec["oracle_cost_final"],
                    "total_timesteps": rec["total_timesteps"],
                    "env_id": env_id,
                },
            })

    if not tasks:
        print("Nothing to evaluate.")
    else:
        workers = min(args.workers, len(tasks))
        total = len(tasks)
        print(f"\nQueued {total} eval(s) across {workers} worker(s)  ({args.n_episodes} episodes each)")
        for i, t in enumerate(tasks, 1):
            print(f"  [{i:2d}/{total}]  {t['meta']['exp_name']}  acc={t['oracle_accuracy']:g}")
        print()

        # Shared queue: workers push (label, ep, total_ep) after each episode
        manager = Manager()
        progress_q = manager.Queue()
        for t in tasks:
            t["_progress_q"] = progress_q

        # Overall task-completion bar
        overall_bar = tqdm(total=total, desc="tasks done", position=0, leave=True, ncols=100)

        # Per-run episode bars, one slot per worker (tasks rotate through slots)
        n_slots = workers
        slot_bars = [
            tqdm(total=args.n_episodes, desc=" " * 30, position=i + 1,
                 leave=False, ncols=100, bar_format="{desc} {bar} {n}/{total}")
            for i in range(n_slots)
        ]
        # Map label → slot (assigned on first progress message)
        label_to_slot: dict[str, int] = {}
        slot_free = list(range(n_slots))

        def _monitor():
            while True:
                item = progress_q.get()
                if item is None:
                    break
                label, ep, total_ep = item
                if label not in label_to_slot:
                    if slot_free:
                        slot = slot_free.pop(0)
                    else:
                        slot = ep % n_slots  # fallback if all slots taken
                    label_to_slot[label] = slot
                slot = label_to_slot[label]
                bar = slot_bars[slot]
                bar.set_description(f"{label[:30]:30s}")
                bar.n = ep
                bar.total = total_ep
                bar.refresh()

        monitor_thread = threading.Thread(target=_monitor, daemon=True)
        monitor_thread.start()

        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=workers, initializer=_suppress_gym_warnings) as pool:
            futures = {pool.submit(_eval_task, t): t for t in tasks}
            for future in as_completed(futures):
                meta, metrics = future.result()
                done += 1
                elapsed = time.time() - t0
                eta = (elapsed / done) * (total - done) if done < total else 0

                # Free the slot this label was using
                label = meta["exp_name"]
                if label in label_to_slot:
                    slot = label_to_slot.pop(label)
                    slot_bars[slot].set_description(" " * 30)
                    slot_bars[slot].n = 0
                    slot_bars[slot].refresh()
                    slot_free.append(slot)

                overall_bar.update(1)
                if metrics is None:
                    overall_bar.write(f"  [FAILED] {meta['exp_name']}  ({elapsed:.0f}s)")
                    continue
                row = {**meta, **metrics}
                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
                results_df.to_csv(out_path, index=False)
                overall_bar.write(
                    f"  [done {done:2d}/{total}]  {meta['exp_name']}"
                    f"  success={metrics['success_rate']:.3f}"
                    f"  oracle={metrics['oracle_usage_pct']:.0f}%"
                    f"  queries/ep={metrics['queries_per_ep']:.1f}"
                    f"  ({elapsed:.0f}s, ETA {eta:.0f}s)"
                )

        progress_q.put(None)
        monitor_thread.join()
        for bar in slot_bars:
            bar.close()
        overall_bar.close()
        manager.shutdown()

    print(f"\nResults saved to {out_path}")
    print(results_df[["exp_name", "env_id", "success_rate", "oracle_usage_pct", "queries_per_ep"]].to_string(index=False))


if __name__ == "__main__":
    main()
