import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
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

HISTORY_KEYS = [
    "charts/global_step",
    "episode/success_rate",
    "episode/guided_pct",
    "episode/queries_per_ep",
]


def as_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def label_float(value):
    if value is None:
        return "missing"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def run_label(row, group_by):
    value = row[group_by]
    if group_by == "oracle_cost":
        return f"cost={label_float(value)}"
    if group_by == "oracle_accuracy":
        return f"accuracy={label_float(value)}"
    return f"{group_by}={label_float(value)}"


def fetch_histories(entity, project, groups):
    api = wandb.Api()
    rows = []

    for group in groups:
        runs = api.runs(f"{entity}/{project}", filters={"group": group})
        for run in runs:
            cfg = run.config
            if cfg.get("no_oracle"):
                continue

            oracle_cost = as_float(cfg.get("oracle_cost_final"))
            if oracle_cost is None:
                oracle_cost = as_float(cfg.get("oracle_cost"))

            oracle_accuracy = as_float(cfg.get("oracle_accuracy"))
            if oracle_accuracy is None:
                continue

            history = []
            for item in run.scan_history(keys=HISTORY_KEYS):
                step = item.get("charts/global_step")
                success = item.get("episode/success_rate")
                guided = item.get("episode/guided_pct")
                queries = item.get("episode/queries_per_ep")
                if step is None or success is None:
                    continue
                history.append(
                    {
                        "global_step": step,
                        "success_rate": success,
                        "guided_pct": guided,
                        "queries_per_ep": queries,
                    }
                )

            if not history:
                continue

            df = pd.DataFrame(history).sort_values("global_step")
            rows.append(
                {
                    "run_name": run.name,
                    "group": group,
                    "oracle_cost": oracle_cost,
                    "oracle_accuracy": oracle_accuracy,
                    "history": df,
                }
            )

    return rows


def nearly_equal(a, b):
    return a is not None and b is not None and abs(a - b) < 1e-9


def filter_rows(rows, fixed_cost=None, fixed_accuracy=None):
    filtered = []
    for row in rows:
        if fixed_cost is not None and not nearly_equal(row["oracle_cost"], fixed_cost):
            continue
        if fixed_accuracy is not None and not nearly_equal(row["oracle_accuracy"], fixed_accuracy):
            continue
        filtered.append(row)
    return filtered


def aggregate_group(rows, metric, n_points=300):
    max_step = min(row["history"]["global_step"].max() for row in rows)
    grid = np.linspace(0, max_step, n_points)
    values = []

    for row in rows:
        hist = row["history"].dropna(subset=["global_step", metric])
        if hist.empty:
            continue
        values.append(np.interp(grid, hist["global_step"], hist[metric]))

    if not values:
        return None, None
    return grid, np.mean(np.stack(values), axis=0)


def plot_curves(rows, group_by, title, out_path):
    if not rows:
        raise ValueError(f"No runs available for {title}")

    grouped = {}
    for row in rows:
        grouped.setdefault(row[group_by], []).append(row)

    grouped = dict(sorted(grouped.items(), key=lambda item: item[0]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    metrics = [
        ("success_rate", "Success Rate", (0, 1)),
        ("guided_pct", "Oracle Usage (% steps)", (0, 100)),
    ]

    colors = plt.cm.viridis(np.linspace(0.05, 0.9, max(len(grouped), 1)))

    for color, (_, group_rows) in zip(colors, grouped.items()):
        example = group_rows[0]
        label = f"{run_label(example, group_by)} (n={len(group_rows)})"
        for ax, (metric, ylabel, ylim) in zip(axes, metrics):
            grid, mean = aggregate_group(group_rows, metric)
            if grid is None:
                continue
            ax.plot(grid, mean, color=color, linewidth=2.2, label=label)
            ax.set_title(ylabel)
            ax.set_xlabel("Environment Steps")
            ax.set_ylabel(ylabel)
            ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.25, linestyle="--")

    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--groups", nargs="+", default=GROUPS)
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--cost-plot-accuracy", type=float, default=None)
    parser.add_argument("--accuracy-plot-cost", type=float, default=None)
    args = parser.parse_args()

    rows = fetch_histories(args.entity, args.project, args.groups)
    print(f"Loaded {len(rows)} oracle runs with history")

    cost_rows = filter_rows(rows, fixed_accuracy=args.cost_plot_accuracy)
    cost_title = "Sokoban Oracle Cost Comparison"
    if args.cost_plot_accuracy is not None:
        cost_title += f" (accuracy={label_float(args.cost_plot_accuracy)})"
    plot_curves(
        cost_rows,
        "oracle_cost",
        cost_title,
        os.path.join(args.out_dir, "wandb_sokoban_cost_curves.png"),
    )

    accuracy_rows = filter_rows(rows, fixed_cost=args.accuracy_plot_cost)
    accuracy_title = "Sokoban Oracle Accuracy Comparison"
    if args.accuracy_plot_cost is not None:
        accuracy_title += f" (cost={label_float(args.accuracy_plot_cost)})"
    plot_curves(
        accuracy_rows,
        "oracle_accuracy",
        accuracy_title,
        os.path.join(args.out_dir, "wandb_sokoban_accuracy_curves.png"),
    )


if __name__ == "__main__":
    main()
