import argparse
import json
import os
import re

import pandas as pd
import wandb


ENTITY = "hersheytee"
PROJECT = "cs503-sokoban"
GROUPS = [
    "sokoban_randomized_oracle_500k",
    "sokoban_randomized_oracle_extra_500k",
    "sokoban_linear_cost_500k",
    "worldcoder_2m",
]

DEFAULT_KEYS = [
    "charts/global_step",
    "episode/return",
    "episode/success_rate",
    "episode/guided_pct",
    "episode/queries_per_ep",
    "episode/oracle_correct_rate",
    "episode/oracle_cost",
    "losses/value_loss",
    "losses/policy_loss",
    "losses/explained_variance",
    "losses/entropy",
]


def safe_name(text):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    return text.strip("_")[:180]


def export_run(run, out_dir, keys, overwrite=False, page_size=500):
    cfg = run.config
    summary = run.summary._json_dict
    run_dir = os.path.join(out_dir, safe_name(run.name or run.id))
    os.makedirs(run_dir, exist_ok=True)

    metrics_path = os.path.join(run_dir, "history.csv")
    metadata_path = os.path.join(run_dir, "metadata.json")

    if os.path.exists(metrics_path) and not overwrite:
        print(f"[skip] {run.name} already exported")
        return metrics_path

    metadata = {
        "run_id": run.id,
        "run_name": run.name,
        "group": run.group,
        "state": run.state,
        "url": run.url,
        "config": dict(cfg),
        "summary": dict(summary),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, default=str)

    rows = []
    scan_kwargs = {"page_size": page_size}
    if keys:
        scan_kwargs["keys"] = keys

    print(f"[scan] {run.name}")
    for row in run.scan_history(**scan_kwargs):
        rows.append(row)
        if len(rows) % 5000 == 0:
            print(f"  {len(rows)} rows...")

    df = pd.DataFrame(rows)

    for key, value in {
        "run_id": run.id,
        "run_name": run.name,
        "group": run.group,
        "state": run.state,
        "env_id": cfg.get("env_id"),
        "seed": cfg.get("seed"),
        "no_oracle": cfg.get("no_oracle"),
        "oracle_cost": cfg.get("oracle_cost"),
        "oracle_cost_final": cfg.get("oracle_cost_final"),
        "oracle_accuracy": cfg.get("oracle_accuracy"),
        "total_timesteps": cfg.get("total_timesteps"),
    }.items():
        df.insert(0, key, value)

    df.to_csv(metrics_path, index=False)
    print(f"[saved] {metrics_path} ({len(df)} rows)")
    return metrics_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--groups", nargs="+", default=GROUPS)
    parser.add_argument("--out-dir", default="wandb_history")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--all-keys", action="store_true")
    parser.add_argument("--page-size", type=int, default=500)
    args = parser.parse_args()

    api = wandb.Api(timeout=60)
    os.makedirs(args.out_dir, exist_ok=True)

    exported = []
    for group in args.groups:
        print(f"[group] {group}")
        runs = api.runs(f"{args.entity}/{args.project}", filters={"group": group})
        for run in runs:
            keys = None if args.all_keys else DEFAULT_KEYS
            path = export_run(
                run,
                args.out_dir,
                keys=keys,
                overwrite=args.overwrite,
                page_size=args.page_size,
            )
            exported.append(path)

    manifest = pd.DataFrame({"history_csv": exported})
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"[done] exported {len(exported)} runs")
    print(f"[done] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
