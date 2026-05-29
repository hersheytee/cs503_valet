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

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

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


def discover_checkpoints(run_roots: list[Path]) -> list[dict]:
    records = []
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
            records.append({
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
            })
    return records


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
        oracle_accuracy=1.0,
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

    env.close()
    return {
        "success_rate": float(np.mean(successes)),
        "success_std": float(np.std(successes)),
        "oracle_usage_pct": float(np.mean(guided_pcts)),
        "queries_per_ep": float(np.mean(queries)),
        "queries_per_ep_std": float(np.std(queries)),
        "n_episodes": n_episodes,
    }


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
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_roots = [Path(r) for r in args.run_roots]
    checkpoints = discover_checkpoints(run_roots)
    if not checkpoints:
        raise SystemExit("No checkpoints found. Check --run-roots.")

    print(f"Found {len(checkpoints)} checkpoint(s) across {len(run_roots)} run root(s).")
    results_df = load_results(out_path)

    for rec in checkpoints:
        for env_id in args.env_ids:
            run_name = rec["run_name"]

            if already_evaluated(results_df, run_name, env_id):
                print(f"  [skip] {rec['exp_name']} on {env_id}")
                continue

            print(f"  [eval] {rec['exp_name']} on {env_id} ...", end="", flush=True)
            try:
                metrics = evaluate(
                    ckpt=rec["checkpoint"],
                    no_oracle=rec["no_oracle"],
                    hidden_dim=rec["hidden_dim"],
                    obs_size=rec["obs_size"],
                    env_id=env_id,
                    n_episodes=args.n_episodes,
                    max_episode_steps=args.max_episode_steps,
                    seed=args.seed,
                )
            except Exception:
                print(f" FAILED")
                traceback.print_exc()
                continue

            row = {
                "run_name": run_name,
                "exp_name": rec["exp_name"],
                "seed": rec["seed"],
                "no_oracle": rec["no_oracle"],
                "oracle_accuracy": rec["oracle_accuracy"],
                "oracle_cost": rec["oracle_cost"],
                "oracle_cost_final": rec["oracle_cost_final"],
                "total_timesteps": rec["total_timesteps"],
                "env_id": env_id,
                **metrics,
            }
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
            results_df.to_csv(out_path, index=False)

            print(
                f" success={metrics['success_rate']:.3f}"
                f"  oracle_usage={metrics['oracle_usage_pct']:.1f}%"
                f"  queries/ep={metrics['queries_per_ep']:.1f}"
            )

    print(f"\nResults saved to {out_path}")
    print(results_df[["exp_name", "env_id", "success_rate", "oracle_usage_pct", "queries_per_ep"]].to_string(index=False))


if __name__ == "__main__":
    main()
