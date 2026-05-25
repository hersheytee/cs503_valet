import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


DEFAULT_SUCCESS_CSV = "logs/wandb_export_2026-05-25T20_20_18.805+02_00.csv"
DEFAULT_USAGE_CSV = "logs/wandb_export_2026-05-25T20_20_24.201+02_00.csv"
DEFAULT_LOGS_DIR = "logs"


def parse_acc(token):
    return int(token) / 100.0


def parse_cost(token):
    if token == "0":
        return 0.0
    if token == "1":
        return 1.0
    if token.startswith("0"):
        return int(token) / 10.0
    return float(token)


def parse_local_cost(filename):
    if filename.startswith("oracle_free__"):
        return 0.0
    match = re.search(r"oracle_paid_(\d+)__", filename)
    if not match:
        return None
    return parse_cost(match.group(1))


def parse_run(column):
    run_name = column.split(" - ", 1)[0]
    match = re.search(r"oracle_acc(\d+)_cost(\d+)_", run_name)
    if not match:
        return None
    return {
        "run_name": run_name,
        "accuracy": parse_acc(match.group(1)),
        "cost": parse_cost(match.group(2)),
    }


def read_metric_csv(path, metric_name):
    df = pd.read_csv(path)
    step = pd.to_numeric(df["charts/global_step"], errors="coerce")
    rows = []

    for column in df.columns:
        if metric_name not in column:
            continue
        if column.endswith("__MIN") or column.endswith("__MAX"):
            continue

        meta = parse_run(column)
        if meta is None:
            continue

        values = pd.to_numeric(df[column], errors="coerce")
        metric_df = pd.DataFrame(
            {
                "global_step": step,
                "value": values,
                "metric": metric_name,
                **meta,
            }
        ).dropna(subset=["global_step", "value"])
        rows.append(metric_df)

    if not rows:
        raise ValueError(f"No columns found for metric {metric_name} in {path}")

    return pd.concat(rows, ignore_index=True)


def load_data(success_csv, usage_csv):
    success = read_metric_csv(success_csv, "success_rate_50ep")
    success["metric_label"] = "Success Rate"

    usage = read_metric_csv(usage_csv, "guided_pct_40ep")
    usage["metric_label"] = "Oracle Usage (% steps)"

    return pd.concat([success, usage], ignore_index=True)


def load_local_logs(logs_dir):
    oracle_rows = []
    baseline_rows = []

    for path in glob.glob(os.path.join(logs_dir, "*.csv")):
        filename = os.path.basename(path)
        if filename.startswith("wandb_export_"):
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        needed = {"global_step", "success", "guided_pct"}
        if df.empty or not needed.issubset(df.columns):
            continue
        if df["global_step"].max() < 100000:
            continue

        run_name = os.path.splitext(filename)[0]
        local = df[["global_step", "success", "guided_pct"]].copy()
        local["global_step"] = pd.to_numeric(local["global_step"], errors="coerce")
        local["success"] = pd.to_numeric(local["success"], errors="coerce")
        local["guided_pct"] = pd.to_numeric(local["guided_pct"], errors="coerce")
        local = local.dropna(subset=["global_step"])

        if filename.startswith("baseline__"):
            if local["guided_pct"].max() != 0:
                continue
            success = local[["global_step", "success"]].copy()
            success["value"] = success["success"].rolling(50, min_periods=1).mean()
            success["metric_label"] = "Success Rate"
            success["run_name"] = run_name
            baseline_rows.append(success[["global_step", "value", "metric_label", "run_name"]])
            continue

        cost = parse_local_cost(filename)
        if cost is None:
            continue

        success = local[["global_step", "success"]].copy()
        success["value"] = success["success"].rolling(50, min_periods=1).mean()
        success["metric_label"] = "Success Rate"

        usage = local[["global_step", "guided_pct"]].copy()
        usage["value"] = usage["guided_pct"].rolling(40, min_periods=1).mean()
        usage["metric_label"] = "Oracle Usage (% steps)"

        for metric_df in (success, usage):
            metric_df["run_name"] = run_name
            metric_df["accuracy"] = 1.0
            metric_df["cost"] = cost
            oracle_rows.append(
                metric_df[["global_step", "value", "metric_label", "run_name", "accuracy", "cost"]]
            )

    oracle = pd.concat(oracle_rows, ignore_index=True) if oracle_rows else pd.DataFrame()
    baseline = pd.concat(baseline_rows, ignore_index=True) if baseline_rows else pd.DataFrame()
    return oracle, baseline


def label_value(name, value):
    if name == "cost":
        return f"{value:g}"
    if name == "accuracy":
        return f"{value:g}"
    return f"{name}={value:g}"


def aggregate(df, group_by, metric_label):
    curves = []
    metric_df = df[df["metric_label"] == metric_label]
    for group_value, group_df in metric_df.groupby(group_by):
        pivots = []
        for _, run_df in group_df.groupby("run_name"):
            run_df = run_df.sort_values("global_step")
            pivots.append(run_df[["global_step", "value"]])

        if not pivots:
            continue
        max_step = min(p["global_step"].max() for p in pivots)
        min_step = max(p["global_step"].min() for p in pivots)
        if max_step <= min_step:
            continue
        common_steps = np.linspace(min_step, max_step, 300)

        values = []
        for pivot in pivots:
            pivot = pivot.dropna(subset=["global_step", "value"])
            aligned = np.interp(common_steps, pivot["global_step"], pivot["value"])
            values.append(aligned)

        curves.append(
            {
                "group_value": group_value,
                "steps": common_steps,
                "values": np.mean(np.stack(values), axis=0),
                "n": len(pivots),
            }
        )
    return sorted(curves, key=lambda x: x["group_value"])


def baseline_curve(baseline_df):
    if baseline_df is None or baseline_df.empty:
        return None

    pivots = []
    for _, run_df in baseline_df.groupby("run_name"):
        run_df = run_df.sort_values("global_step").dropna(subset=["global_step", "value"])
        if not run_df.empty:
            pivots.append(run_df[["global_step", "value"]])
    if not pivots:
        return None

    max_step = min(p["global_step"].max() for p in pivots)
    min_step = max(p["global_step"].min() for p in pivots)
    if max_step <= min_step:
        return None

    steps = np.linspace(min_step, max_step, 300)
    values = [
        np.interp(steps, pivot["global_step"], pivot["value"])
        for pivot in pivots
    ]
    return steps, np.mean(np.stack(values), axis=0)


def plot_pair(df, group_by, out_path, baseline_df=None, title=None):
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, constrained_layout=True)
    metrics = ["Success Rate", "Oracle Usage (% steps)"]
    group_values = sorted(df[group_by].dropna().unique())
    colors = plt.cm.plasma(np.linspace(0.08, 0.9, len(group_values)))

    for ax, metric_label in zip(axes, metrics):
        curves = aggregate(df, group_by, metric_label)
        for color, curve in zip(colors, curves):
            label = label_value(group_by, curve["group_value"])
            ax.plot(curve["steps"], curve["values"], linewidth=2.4, color=color, label=label)

        ax.set_xlabel("Environment Steps")
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)

        if metric_label == "Success Rate":
            baseline = baseline_curve(baseline_df)
            if baseline is not None:
                steps, values = baseline
                ax.plot(steps, values, color="black", linewidth=2.6, label="baseline")
                ax.legend(frameon=False, fontsize=8)
            ax.set_ylim(0, 1)
        if metric_label == "Oracle Usage (% steps)":
            ax.set_ylim(0, 100)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--success-csv", default=DEFAULT_SUCCESS_CSV)
    parser.add_argument("--usage-csv", default=DEFAULT_USAGE_CSV)
    parser.add_argument("--logs-dir", default=DEFAULT_LOGS_DIR)
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--cost-accuracy", type=float, default=1.0)
    parser.add_argument("--accuracy-cost", type=float, default=0.5)
    parser.add_argument("--only-requested", action="store_true")
    args = parser.parse_args()

    df = load_data(args.success_csv, args.usage_csv)
    local_oracle, local_baseline = load_local_logs(args.logs_dir)

    cost_df = df[np.isclose(df["accuracy"], args.cost_accuracy)]
    if cost_df.empty:
        raise ValueError(f"No runs found with accuracy={args.cost_accuracy}")
    plot_pair(
        cost_df,
        "cost",
        os.path.join(args.out_dir, "wandb_export_cost_usage_curves_acc100.png"),
        baseline_df=local_baseline,
        title="Varying Query Cost (100% Oracle Accuracy)",
    )

    accuracy_df = df[np.isclose(df["cost"], args.accuracy_cost)]
    if accuracy_df.empty:
        raise ValueError(f"No runs found with cost={args.accuracy_cost}")
    plot_pair(
        accuracy_df,
        "accuracy",
        os.path.join(args.out_dir, "wandb_export_accuracy_usage_curves_cost05.png"),
        baseline_df=local_baseline,
        title="Varying Oracle Accuracy (Cost = 0.5)",
    )

    if args.only_requested:
        return

    plot_pair(
        df,
        "cost",
        os.path.join(args.out_dir, "wandb_export_cost_usage_curves_all.png"),
        baseline_df=local_baseline,
        title="Varying Query Cost",
    )
    plot_pair(
        df,
        "accuracy",
        os.path.join(args.out_dir, "wandb_export_accuracy_usage_curves_all.png"),
        baseline_df=local_baseline,
        title="Varying Oracle Accuracy",
    )


if __name__ == "__main__":
    main()
