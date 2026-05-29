"""Plot final Sokoban oracle experiments from local run artifacts.

The script reads `runs/*/config.yaml` and `runs/*/data/metrics.csv` folders,
including the downloaded Vast bundle under `gym_sokoban/sokoban_vast_results`.

Default outputs:
  - `cost_curves_acc100.png`: success and oracle usage vs steps for 100% oracle
    accuracy across query costs.
  - `accuracy_endpoint_cost05.png`: final success and oracle usage vs oracle
    accuracy at cost 0.5.
  - `linear_cost_schedule.png`: success, oracle usage, and scheduled cost vs
    steps for the 0 -> 1 linear cost run.

Curves use a 50-episode rolling average by default. Endpoint ablations use the
mean of the final 50 episodes, which is more stable than the final row alone.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RUN_ROOTS = [
    REPO_ROOT / "runs",
    SCRIPT_DIR / "sokoban_vast_results" / "runs",
]


@dataclass
class RunRecord:
    """Small bundle of config metadata plus episode-level metrics."""

    run_dir: Path
    run_name: str
    exp_name: str
    seed: int
    no_oracle: bool
    oracle_accuracy: float | None
    oracle_cost: float | None
    oracle_cost_final: float | None
    oracle_cost_anneal_steps: int | None
    total_timesteps: int
    max_step: float
    metrics: pd.DataFrame

    @property
    def is_linear_cost(self) -> bool:
        return self.oracle_cost_final is not None

    @property
    def fixed_cost(self) -> float | None:
        if self.is_linear_cost:
            return None
        return self.oracle_cost


def label_float(value: float | None) -> str:
    if value is None:
        return "missing"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def coerce_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = ["episode", "global_step", "success", "guided_pct", "oracle_cost"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["global_step"]).sort_values(["global_step", "episode"])
    return df


def discover_runs(run_roots: Iterable[Path], min_steps: int) -> list[RunRecord]:
    """Load runs and keep the latest/longest artifact for duplicate exp names."""

    records: list[RunRecord] = []
    for root in run_roots:
        if not root.exists():
            continue
        for config_path in root.glob("*/config.yaml"):
            run_dir = config_path.parent
            metrics_path = run_dir / "data" / "metrics.csv"
            if not metrics_path.exists():
                continue

            try:
                config = read_yaml(config_path)
                metrics = coerce_metrics(metrics_path)
            except Exception as exc:
                print(f"Skipping {run_dir}: {exc}")
                continue

            if metrics.empty:
                continue

            args = config.get("args", {})
            run_name = str(config.get("run", {}).get("name") or run_dir.name)
            exp_name = str(args.get("exp_name") or parse_exp_name(run_name))
            max_step = float(metrics["global_step"].max())

            # Smoke tests and cancelled fragments are useful for debugging, not
            # for final figures. The de-duplication below also removes earlier
            # partial attempts for the same exp_name.
            if max_step < min_steps:
                continue

            records.append(
                RunRecord(
                    run_dir=run_dir,
                    run_name=run_name,
                    exp_name=exp_name,
                    seed=int(args.get("seed", 0)),
                    no_oracle=bool(args.get("no_oracle", False)),
                    oracle_accuracy=as_float(args.get("oracle_accuracy")),
                    oracle_cost=as_float(args.get("oracle_cost")),
                    oracle_cost_final=as_float(args.get("oracle_cost_final")),
                    oracle_cost_anneal_steps=int(args["oracle_cost_anneal_steps"]) if args.get("oracle_cost_anneal_steps") is not None else None,
                    total_timesteps=int(args.get("total_timesteps", 0)),
                    max_step=max_step,
                    metrics=metrics,
                )
            )

    best_by_key: dict[tuple[str, int], RunRecord] = {}
    for record in records:
        key = (record.exp_name, record.seed)
        current = best_by_key.get(key)
        if current is None:
            best_by_key[key] = record
            continue
        if (record.max_step, record.run_name) > (current.max_step, current.run_name):
            best_by_key[key] = record

    return sorted(best_by_key.values(), key=lambda r: r.run_name)


def parse_exp_name(run_name: str) -> str:
    parts = run_name.split("__")
    return parts[1] if len(parts) >= 2 else run_name


def as_float(value) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def smooth_metric(metrics: pd.DataFrame, column: str, window: int, n_interp: int = 500) -> pd.DataFrame:
    out = metrics[["global_step", column]].copy()
    out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["global_step", column]).sort_values("global_step")
    if len(out) < 2:
        return pd.DataFrame(columns=["global_step", "value"])
    steps = np.linspace(out["global_step"].iloc[0], out["global_step"].iloc[-1], n_interp)
    values = np.interp(steps, out["global_step"].values, out[column].values)
    smoothed = pd.Series(values).rolling(window=window, min_periods=1).mean().values
    return pd.DataFrame({"global_step": steps, "value": smoothed})


def smooth_metric_with_std(metrics: pd.DataFrame, column: str, window: int, n_interp: int = 500) -> pd.DataFrame:
    """Like smooth_metric but also returns a rolling std column (for single-run shading)."""
    out = metrics[["global_step", column]].copy()
    out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["global_step", column]).sort_values("global_step")
    if len(out) < 2:
        return pd.DataFrame(columns=["global_step", "value", "std"])
    steps = np.linspace(out["global_step"].iloc[0], out["global_step"].iloc[-1], n_interp)
    values = np.interp(steps, out["global_step"].values, out[column].values)
    series = pd.Series(values)
    smoothed = series.rolling(window=window, min_periods=1).mean().values
    std = series.rolling(window=window, min_periods=1).std().fillna(0).values
    return pd.DataFrame({"global_step": steps, "value": smoothed, "std": std})


def smooth_metric_band(
    records_group: list, column: str, window: int, n_interp: int = 500
):
    """Interpolate smoothed curves across multiple runs, return (steps, mean, std)."""
    curves = []
    for rec in records_group:
        s = smooth_metric(rec.metrics, column, window)
        if not s.empty:
            curves.append(s)
    if not curves:
        return None, None, None
    min_step = max(c["global_step"].min() for c in curves)
    max_step = min(c["global_step"].max() for c in curves)
    if min_step >= max_step:
        return None, None, None
    steps = np.linspace(min_step, max_step, n_interp)
    vals = np.array(
        [np.interp(steps, c["global_step"].values, c["value"].values) for c in curves]
    )
    return steps, vals.mean(axis=0), vals.std(axis=0)


def endpoint(record: RunRecord, column: str, window: int) -> float:
    values = pd.to_numeric(record.metrics[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.tail(window).mean())


def fixed_cost_runs(records: list[RunRecord], accuracy: float) -> list[RunRecord]:
    runs = [
        r
        for r in records
        if not r.no_oracle
        and not r.is_linear_cost
        and r.oracle_accuracy is not None
        and np.isclose(r.oracle_accuracy, accuracy)
    ]
    return sorted(runs, key=lambda r: (r.fixed_cost if r.fixed_cost is not None else -1))


def accuracy_runs(records: list[RunRecord], cost: float) -> list[RunRecord]:
    runs = [
        r
        for r in records
        if not r.no_oracle
        and not r.is_linear_cost
        and r.fixed_cost is not None
        and np.isclose(r.fixed_cost, cost)
    ]
    return sorted(runs, key=lambda r: (r.oracle_accuracy if r.oracle_accuracy is not None else -1))


def baseline_run(records: list[RunRecord]) -> RunRecord | None:
    baselines = [r for r in records if r.no_oracle]
    if not baselines:
        return None
    return max(baselines, key=lambda r: (r.max_step, r.run_name))


def linear_runs(records: list[RunRecord]) -> list[RunRecord]:
    return sorted(
        [r for r in records if r.is_linear_cost],
        key=lambda r: (r.max_step, r.run_name),
        reverse=True,
    )


def style_plots():
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.titlesize": 18,
        }
    )


def finish_axis(ax):
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Environment Steps")


def plot_cost_curves(
    records: list[RunRecord],
    out_dir: Path,
    window: int,
    accuracy: float,
    shade: bool = False,
):
    runs = fixed_cost_runs(records, accuracy=accuracy)
    if not runs:
        print(f"No fixed-cost runs found for accuracy={accuracy}")
        return

    from collections import defaultdict
    cost_groups: dict = defaultdict(list)
    for run in runs:
        cost_groups[run.fixed_cost].append(run)
    unique_costs = sorted(cost_groups.keys(), key=lambda c: c if c is not None else -1)

    baseline = baseline_run(records)
    colors = plt.cm.plasma(np.linspace(0.08, 0.9, len(unique_costs)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)

    for color, cost_val in zip(colors, unique_costs):
        group = cost_groups[cost_val]
        label = f"cost={label_float(cost_val)}"
        for col, ax in zip(["success", "guided_pct"], axes):
            if shade and len(group) > 1:
                steps, mean, std = smooth_metric_band(group, col, window)
                if steps is None:
                    continue
                ax.plot(steps, mean, color=color, linewidth=2.4, label=label)
                ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.25)
            elif shade:
                s = smooth_metric_with_std(group[0].metrics, col, window)
                ax.plot(s["global_step"], s["value"], color=color, linewidth=2.4, label=label)
                ax.fill_between(
                    s["global_step"], s["value"] - s["std"], s["value"] + s["std"],
                    color=color, alpha=0.25,
                )
            else:
                run = max(group, key=lambda r: r.max_step)
                s = smooth_metric(run.metrics, col, window)
                ax.plot(s["global_step"], s["value"], color=color, linewidth=2.4, label=label)

    if baseline is not None:
        base_success = smooth_metric(baseline.metrics, "success", window)
        axes[0].plot(
            base_success["global_step"],
            base_success["value"],
            color="black",
            linewidth=2.8,
            linestyle="-",
            label="baseline",
        )

    axes[0].set_title("Success Rate")
    axes[0].set_ylabel("")
    axes[0].set_ylim(-0.02, 1.02)

    axes[1].set_title("Oracle Usage (% steps)")
    axes[1].set_ylabel("")
    axes[1].set_ylim(-2, 102)

    for ax in axes:
        finish_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    seen = set(labels)
    for h, l in zip(h1, l1):
        if l not in seen:
            handles.append(h)
            labels.append(l)
            seen.add(l)
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False, fontsize=11)

    fig.tight_layout()
    out_path = out_dir / "cost_curves_acc100.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_accuracy_lines(
    records: list[RunRecord],
    out_dir: Path,
    window: int,
    shade: bool = False,
):
    """Line plot of final metrics vs oracle accuracy, one line per cost value."""
    from collections import defaultdict

    all_runs = [
        r for r in records
        if not r.no_oracle and not r.is_linear_cost
        and r.oracle_accuracy is not None and r.fixed_cost is not None
    ]
    if not all_runs:
        print("No accuracy-ablation runs found")
        return

    costs = sorted({r.fixed_cost for r in all_runs})
    palette = plt.cm.plasma(np.linspace(0.08, 0.9, len(costs)))
    cost_color = dict(zip(costs, palette))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    for cost_val in costs:
        color = cost_color[cost_val]
        label = f"cost={label_float(cost_val)}"

        by_acc: dict = defaultdict(lambda: {"success": [], "usage": [], "queries": []})
        for run in all_runs:
            if not np.isclose(run.fixed_cost, cost_val):
                continue
            svals = pd.to_numeric(run.metrics["success"], errors="coerce").dropna().tail(window).values
            uvals = pd.to_numeric(run.metrics["guided_pct"], errors="coerce").dropna().tail(window).values
            by_acc[run.oracle_accuracy]["success"].extend(svals.tolist())
            by_acc[run.oracle_accuracy]["usage"].extend(uvals.tolist())
            if "queries_per_ep" in run.metrics.columns:
                qvals = pd.to_numeric(run.metrics["queries_per_ep"], errors="coerce").dropna().tail(window).values
                by_acc[run.oracle_accuracy]["queries"].extend(qvals.tolist())

        if not by_acc:
            continue
        accs = sorted(by_acc.keys())
        x = np.array(accs)
        smean = np.array([np.mean(by_acc[a]["success"]) for a in accs])
        sstd = np.array([np.std(by_acc[a]["success"]) for a in accs])
        umean = np.array([np.mean(by_acc[a]["usage"]) for a in accs])
        ustd = np.array([np.std(by_acc[a]["usage"]) for a in accs])
        qmean = np.array([np.mean(by_acc[a]["queries"]) if by_acc[a]["queries"] else np.nan for a in accs])
        qstd  = np.array([np.std(by_acc[a]["queries"])  if by_acc[a]["queries"] else np.nan for a in accs])

        axes[0].plot(x, smean, color=color, linewidth=2, marker="o", label=label)
        axes[1].plot(x, umean, color=color, linewidth=2, marker="o", label=label)
        valid = ~np.isnan(qmean)
        if valid.any():
            axes[2].plot(x[valid], qmean[valid], color=color, linewidth=2, marker="o", label=label)
        if shade:
            axes[0].fill_between(x, smean - sstd, smean + sstd, color=color, alpha=0.25)
            axes[1].fill_between(x, umean - ustd, umean + ustd, color=color, alpha=0.25)
            if valid.any():
                axes[2].fill_between(x[valid], qmean[valid] - qstd[valid], qmean[valid] + qstd[valid], color=color, alpha=0.25)

    axes[0].set_title("Success Rate")
    axes[0].set_ylabel("")
    axes[0].set_ylim(0, 1.02)

    axes[1].set_title("Oracle Usage (% steps)")
    axes[1].set_ylabel("")
    axes[1].set_ylim(0, 102)

    axes[2].set_title("Oracle Queries per Episode")
    axes[2].set_ylabel("")

    for ax in axes:
        ax.set_xlabel("Oracle Accuracy")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False, fontsize=11)

    fig.tight_layout()
    out_path = out_dir / "accuracy_lines.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_linear_schedule(records: list[RunRecord], out_dir: Path, window: int, shade: bool = False):
    runs = linear_runs(records)
    if not runs:
        print("No linear cost schedule run found")
        return

    run = runs[0]
    cost_max = float(run.oracle_cost_final) if run.oracle_cost_final is not None else float(run.oracle_cost or 1.0)

    # Build derived metrics (efficiency ratios) on raw data then smooth
    m = run.metrics.copy()
    guided_frac = pd.to_numeric(m["guided_pct"], errors="coerce") / 100
    success = pd.to_numeric(m["success"], errors="coerce")
    m["oracle_efficiency"] = success / guided_frac.clip(lower=0.001)
    if "queries_per_ep" in m.columns:
        q = pd.to_numeric(m["queries_per_ep"], errors="coerce")
        m["success_per_query"] = success / q.clip(lower=0.1)

    RED = "#e53935"

    def _plot(ax, col, title, ylabel, ylim, raw=False):
        if col not in m.columns:
            ax.set_visible(False)
            return
        if raw:
            # No smoothing — plot the raw interpolated signal directly
            out = m[["global_step", col]].copy()
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out = out.dropna().sort_values("global_step")
            ax.plot(out["global_step"], out[col], color=RED, linewidth=2.5)
        elif shade:
            s = smooth_metric_with_std(m, col, window)
            ax.plot(s["global_step"], s["value"], color=RED, linewidth=2.5)
            ax.fill_between(s["global_step"], s["value"] - s["std"], s["value"] + s["std"],
                            color=RED, alpha=0.25)
        else:
            s = smooth_metric(m, col, window)
            ax.plot(s["global_step"], s["value"], color=RED, linewidth=2.5)
        ax.set_title(title)
        ax.set_ylabel("")
        if ylim is not None:
            ax.set_ylim(*ylim)
        finish_axis(ax)
        ax.tick_params(labelbottom=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)

    _plot(axes[0, 0], "oracle_cost",      "Query Cost",                                None, (-0.02 * cost_max, cost_max * 1.08), raw=True)
    _plot(axes[0, 1], "success",          "Success Rate",                              None, (-0.02, 1.02))
    _plot(axes[0, 2], "ep_return",        "Episodic Reward",                           None, None)
    _plot(axes[1, 0], "guided_pct",       "Oracle Usage (%)",                          None, (-2, 102))
    _plot(axes[1, 1], "queries_per_ep",   "Oracle Queries per Episode",                None, None)
    _plot(axes[1, 2], "success_per_query","Success Rate / Oracle Queries per Episode", None, (0, 0.5))

    cost_start = label_float(run.oracle_cost)
    cost_end = label_float(run.oracle_cost_final)
    anneal_m = f"{int(run.oracle_cost_anneal_steps) // 1_000_000}M" if run.oracle_cost_anneal_steps else f"{int(run.total_timesteps) // 1_000_000}M"
    total_m = f"{int(run.total_timesteps) // 1_000_000}M"
    fig.suptitle(f"Increasing Query Cost Schedule: {cost_start} → {cost_end} over {anneal_m} steps", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / "linear_cost_schedule.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_transfer(out_dir: Path, min_timesteps: int = 450_000):
    """Transfer results: accuracy on x-axis, lines colored by oracle cost."""
    csv_path = out_dir / "transfer_results.csv"
    if not csv_path.exists():
        print(f"No transfer_results.csv found at {csv_path} — skipping transfer plot")
        return

    df = pd.read_csv(csv_path)
    df = df[df["total_timesteps"] >= min_timesteps].copy()
    df = df[~df["no_oracle"]].copy()
    if df.empty:
        print("No eligible transfer results found — skipping transfer plot")
        return

    df = df[df["env_id"].str.contains("FixedTarget")].copy()
    if df.empty:
        print("No FixedTarget transfer results found — skipping transfer plot")
        return

    costs = sorted(df["oracle_cost"].dropna().unique())
    palette = plt.cm.plasma(np.linspace(0.08, 0.9, len(costs)))
    cost_color = dict(zip(costs, palette))

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.8))

    for cost_val in costs:
        rows = df[np.isclose(df["oracle_cost"], cost_val)].sort_values("oracle_accuracy")
        if rows.empty:
            continue
        color = cost_color[cost_val]
        ax.plot(rows["oracle_accuracy"], rows["success_rate"],
                color=color, linewidth=2, marker="o", label=f"cost={cost_val:g}")
        ax.fill_between(
            rows["oracle_accuracy"],
            rows["success_rate"] - rows["success_std"],
            rows["success_rate"] + rows["success_std"],
            color=color, alpha=0.2,
        )

    ax.set_title("FixedTarget-Sokoban-v2 (zero-shot transfer)")
    ax.set_xlabel("Oracle Accuracy")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False, fontsize=11)
    fig.suptitle("Zero-Shot Transfer: Success Rate by Oracle Accuracy & Cost", fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / "transfer_results.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def print_run_summary(records: list[RunRecord], window: int):
    print("\nLoaded runs used for plotting:")
    for run in records:
        kind = "baseline" if run.no_oracle else "oracle"
        if run.is_linear_cost:
            cost = f"{label_float(run.oracle_cost)}->{label_float(run.oracle_cost_final)}"
        else:
            cost = label_float(run.oracle_cost)
        print(
            f"  {run.exp_name:38s} "
            f"kind={kind:8s} acc={label_float(run.oracle_accuracy):>4s} "
            f"cost={cost:>6s} max_step={int(run.max_step):6d} "
            f"final_success_{window}ep={endpoint(run, 'success', window):.3f} "
            f"final_usage_{window}ep={endpoint(run, 'guided_pct', window):.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-roots",
        nargs="+",
        default=[str(path) for path in DEFAULT_RUN_ROOTS],
        help="Directories containing run subdirectories with config.yaml and data/metrics.csv.",
    )
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "figures" / "final"))
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--min-steps", type=int, default=450_000)
    parser.add_argument("--cost-curve-accuracy", type=float, default=1.0)
    parser.add_argument(
        "--shade",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw solid shaded error bands (±1 std) around curves (default: on). Use --no-shade to disable.",
    )
    args = parser.parse_args()

    style_plots()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_roots = [Path(path) for path in args.run_roots]
    records = discover_runs(run_roots, min_steps=args.min_steps)
    if not records:
        raise SystemExit("No eligible runs found. Lower --min-steps or check --run-roots.")

    print_run_summary(records, window=args.window)
    plot_cost_curves(records, out_dir, window=args.window, accuracy=args.cost_curve_accuracy, shade=args.shade)
    plot_accuracy_lines(records, out_dir, window=args.window, shade=args.shade)
    plot_linear_schedule(records, out_dir, window=args.window, shade=args.shade)
    plot_transfer(out_dir, min_timesteps=args.min_steps)


if __name__ == "__main__":
    main()
