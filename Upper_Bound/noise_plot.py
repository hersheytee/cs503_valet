"""
noise_plot.py — Noisy oracle x cost comparison.

One row per noise level. In each row, one curve per cost.
The PPO baseline (--base-csv) is overlaid as dashed lines on each row.

Usage — single cost per noise level:
    python noise_plot.py \
        --run 0.75 0.0 "logs/noisy_oracle_noise075_cost00_seed*.csv" \
        --run 0.50 0.0 "logs/noisy_oracle_noise05_cost00_seed*.csv"  \
        --base-csv "logs/baseline_seed*.csv"                         \
        --out figures/noisy_oracle.png

Usage — multiple costs per noise level:
    python noise_plot.py \
        --run 0.75 0.0  "logs/noisy_oracle_noise075_cost00_seed*.csv"  \
        --run 0.75 0.01 "logs/noisy_oracle_noise075_cost001_seed*.csv" \
        --run 0.75 0.02 "logs/noisy_oracle_noise075_cost002_seed*.csv" \
        --run 0.50 0.0  "logs/noisy_oracle_noise05_cost00_seed*.csv"   \
        --run 0.50 0.01 "logs/noisy_oracle_noise05_cost001_seed*.csv"  \
        --base-csv "logs/baseline_seed*.csv"                           \
        --out figures/noisy_oracle_costs.png

Usage — single row: fix a cost, compare all noise levels:
    python noise_plot.py \
        --run 0.75 0.01 "logs/noisy_oracle_noise075_cost001_seed*.csv" \
        --run 0.50 0.01 "logs/noisy_oracle_noise05_cost001_seed*.csv"  \
        --run 0.25 0.01 "logs/noisy_oracle_noise025_cost001_seed*.csv" \
        --base-csv "logs/baseline_seed*.csv"                           \
        --cost-line 0.01                                               \
        --out figures/noisy_oracle_cost001.png
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.titleweight':  'bold',
    'axes.labelsize':    10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
    'lines.linewidth':   2.0,
    'figure.dpi':        150,
})

BASELINE_COLOR = '#9C27B0'

COLS = [
    ('ep_return',      'Episodic Return',   None, (None, None)),
    ('success',        'Success Rate (%)',   100,  (0, 100)),
    ('guided_pct',     'Oracle Usage (%)',   None, (0, 100)),
    ('queries_per_ep', 'Queries / Episode',  None, (0, None)),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csvs(patterns: list[str], min_steps: int = 0) -> list[pd.DataFrame]:
    files = []
    for p in patterns:
        files += glob.glob(p)
    if not files:
        raise FileNotFoundError(f"No CSV found for: {patterns}")

    seed_to_file: dict = {}
    for f in files:
        m = re.search(r'seed(\d+)__(\d+)\.csv$', os.path.basename(f))
        if m:
            seed, ts = int(m.group(1)), int(m.group(2))
            if seed not in seed_to_file or ts > seed_to_file[seed][1]:
                seed_to_file[seed] = (f, ts)
        else:
            seed_to_file[f] = (f, 0)

    dfs = []
    for seed, (f, _) in sorted(seed_to_file.items()):
        df = pd.read_csv(f)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        max_step = df['global_step'].max()
        if max_step >= min_steps:
            dfs.append(df)
            print(f"  [load] seed={seed}  steps={int(max_step):,}  {os.path.basename(f)}")
        else:
            print(f"  [skip] seed={seed}  steps={int(max_step):,} < {min_steps}  {os.path.basename(f)}")
    if not dfs:
        raise ValueError(f"All CSVs filtered out (min_steps={min_steps})")
    return dfs


def align_and_aggregate(dfs: list[pd.DataFrame], n_points: int = 500):
    cols     = [c for c, *_ in COLS]
    max_step = min(df['global_step'].max() for df in dfs)
    grid     = np.linspace(0, max_step, n_points)

    interpolated: dict[str, list] = {c: [] for c in cols}
    for df in dfs:
        x = df['global_step'].values
        for c in cols:
            if c in df.columns:
                y    = df[c].values.astype(float)
                mask = ~np.isnan(y)
                if mask.sum() >= 2:
                    interpolated[c].append(np.interp(grid, x[mask], y[mask]))
                else:
                    interpolated[c].append(np.full(n_points, np.nan))
            else:
                interpolated[c].append(np.full(n_points, np.nan))

    mean = {c: np.nanmean(np.stack(interpolated[c]), axis=0) for c in cols}
    std  = {c: np.nanstd( np.stack(interpolated[c]), axis=0) for c in cols}
    return grid, mean, std


def smooth(x: np.ndarray, w: int = 50) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=1).mean().values


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_curve(axes, dfs, color, label, smooth_w: int = 50):
    grid, mean, std = align_and_aggregate(dfs)
    for ax, (col, _, scale, ylim) in zip(axes, COLS):
        m = smooth(mean[col], smooth_w) * (scale or 1)
        s = smooth(std[col],  smooth_w) * (scale or 1)
        valid = ~np.isnan(m)
        if not valid.any():
            continue
        ax.plot(grid[valid], m[valid], color=color, label=label)
        lo, hi = ylim
        lo_arr = m - s if lo is None else np.clip(m - s, lo, None)
        hi_arr = m + s if hi is None else np.clip(m + s, None, hi)
        ax.fill_between(grid[valid], lo_arr[valid], hi_arr[valid],
                        alpha=0.15, color=color)
        if ylim[0] is not None or ylim[1] is not None:
            lo = ylim[0] if ylim[0] is not None else ax.get_ylim()[0]
            hi = ylim[1] if ylim[1] is not None else ax.get_ylim()[1]
            ax.set_ylim(lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo))


def draw_baseline(axes, base_dfs, smooth_w: int = 50):
    bg, bm, bs = align_and_aggregate(base_dfs)
    for ax, (col, _, scale, ylim) in zip(axes, COLS):
        bvals = smooth(bm[col], smooth_w) * (scale or 1)
        bstd  = smooth(bs[col], smooth_w) * (scale or 1)
        valid = ~np.isnan(bvals)
        if not valid.any():
            continue
        ax.plot(bg[valid], bvals[valid], color=BASELINE_COLOR, linestyle='--',
                linewidth=1.4, alpha=0.7, label='Baseline PPO')
        lo, hi = ylim
        lo_arr = bvals - bstd if lo is None else np.clip(bvals - bstd, lo, None)
        hi_arr = bvals + bstd if hi is None else np.clip(bvals + bstd, None, hi)
        ax.fill_between(bg[valid], lo_arr[valid], hi_arr[valid],
                        alpha=0.10, color=BASELINE_COLOR)


# ── Color palettes ────────────────────────────────────────────────────────────

def noise_colors(n: int) -> list[str]:
    """Green (perfect oracle) → red (random oracle), one colour per row."""
    cmap = cm.get_cmap('RdYlGn', n)
    return [matplotlib.colors.to_hex(cmap(i)) for i in range(n - 1, -1, -1)]


def cost_colors(n: int) -> list[str]:
    """Distinct categorical colours, one per cost level (no yellow)."""
    cmap = cm.get_cmap('tab10', max(n, 2))
    return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(rows: list[dict], base_dfs, title: str, out: str, smooth_w: int = 50):
    """
    rows : list of dicts  { 'noise': float, 'costs': [(cost, dfs), ...] }
           sorted by noise descending.
    """
    n_rows = len(rows) + (1 if base_dfs is not None else 0)
    n_cols = len(COLS)

    fig = plt.figure(figsize=(n_cols * 4.2, n_rows * 3.8))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.55, wspace=0.35,
                           left=0.07, right=0.98,
                           top=0.93,  bottom=0.05)

    for row_idx, row in enumerate(rows):
        axes = [fig.add_subplot(gs[row_idx, c]) for c in range(n_cols)]
        for ax in axes:
            ax.set_facecolor('white')

        costs_sorted = sorted(row['costs'], key=lambda x: x[0])
        colors = cost_colors(len(costs_sorted))

        for (cost, dfs), color in zip(costs_sorted, colors):
            label = f'cost={cost}'
            draw_curve(axes, dfs, color, label, smooth_w)

        if base_dfs is not None:
            draw_baseline(axes, base_dfs, smooth_w)

        if row_idx == 0:
            for ax, (_, col_title, _, _) in zip(axes, COLS):
                ax.set_title(col_title)

        noise = row['noise']
        pct   = int(noise * 100)
        axes[0].set_ylabel(f'noise={noise}\n({pct}% correct)', labelpad=6)

        for ax in axes:
            ax.set_xlabel('Global Step')
            ax.legend(fontsize=8, loc='upper left')

    # Baseline row at the bottom
    if base_dfs is not None:
        row_idx = len(rows)
        axes = [fig.add_subplot(gs[row_idx, c]) for c in range(n_cols)]
        for ax in axes:
            ax.set_facecolor('white')
        draw_curve(axes, base_dfs, BASELINE_COLOR, 'Baseline PPO', smooth_w)
        axes[0].set_ylabel('Baseline PPO\n(no oracle)', labelpad=6)
        for ax in axes:
            ax.set_xlabel('Global Step')
            ax.legend(fontsize=8, loc='upper left')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else '.', exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


# ── Single-row mode: fixed cost, one curve per noise level ────────────────────

def make_figure_cost_line(
    noise_runs: list[tuple[float, list]],
    base_dfs,
    cost: float,
    title: str,
    out: str,
    smooth_w: int = 50,
):
    """
    Single row with one curve per noise level for a fixed cost.
    Oracle Usage (guided_pct) is excluded from this view.

    noise_runs : list of (noise, dfs), sorted by noise descending.
    """
    global COLS
    cols = [c for c in COLS if c[0] != 'guided_pct']
    n_cols = len(cols)
    fig = plt.figure(figsize=(n_cols * 4.2, 3.8))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(1, n_cols, figure=fig,
                           hspace=0.55, wspace=0.35,
                           left=0.07, right=0.98,
                           top=0.85,  bottom=0.14)

    axes = [fig.add_subplot(gs[0, c]) for c in range(n_cols)]
    for ax in axes:
        ax.set_facecolor('white')

    noise_sorted = sorted(noise_runs, key=lambda x: x[0], reverse=True)
    colors = noise_colors(max(len(noise_sorted), 2))

    # temporarily restrict COLS to the subset used here
    _saved_cols, COLS = COLS, cols

    for (noise, dfs), color in zip(noise_sorted, colors):
        pct   = int(noise * 100)
        label = f'noise={noise} ({pct}% correct)'
        draw_curve(axes, dfs, color, label, smooth_w)

    if base_dfs is not None:
        draw_baseline(axes, base_dfs, smooth_w)

    COLS = _saved_cols  # restore

    for ax, (_, col_title, _, _) in zip(axes, cols):
        ax.set_title(col_title)

    axes[0].set_ylabel(f'cost = {cost}', labelpad=6)
    for ax in axes:
        ax.set_xlabel('Global Step')
        ax.legend(fontsize=8, loc='upper left')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else '.', exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


# ── Summary mode: final performance vs noise, one line per cost ───────────────

def final_value(dfs: list, col: str, scale, tail: float = 0.2) -> tuple[float, float]:
    """Mean ± std of `col` over the last `tail` fraction of training, across seeds."""
    vals = []
    for df in dfs:
        if col not in df.columns:
            continue
        n = max(1, int(len(df) * tail))
        v = df[col].iloc[-n:].dropna()
        if len(v):
            vals.append(float(v.mean()) * (scale or 1))
    if not vals:
        return float('nan'), float('nan')
    return float(np.mean(vals)), float(np.std(vals))


def make_summary_figure(
    grouped: dict,          # {noise: [(cost, dfs), ...]}
    base_dfs,
    title: str,
    out: str,
    tail: float = 0.2,
):
    """
    2-panel summary: Success Rate and Queries/Episode as a function of noise level.
    One line per cost, baseline as horizontal dashed line.
    """
    summary_cols = [
        ('success',        'Success Rate (%)',  100,  'Final Success Rate vs Oracle Accuracy'),
        ('queries_per_ep', 'Queries / Episode', None, 'Oracle Queries vs Oracle Accuracy'),
    ]

    # Collect all costs across all noise levels
    all_costs = sorted({cost for runs in grouped.values() for cost, _ in runs})
    noise_levels = sorted(grouped.keys())

    colors = cost_colors(max(len(all_costs), 2))
    cost_color = dict(zip(all_costs, colors))

    fig, axes = plt.subplots(len(summary_cols), 1,
                             figsize=(6, len(summary_cols) * 4))
    fig.patch.set_facecolor('white')
    for ax in axes:
        ax.set_facecolor('white')

    for ax, (col, ylabel, scale, ax_title) in zip(axes, summary_cols):
        # One line per cost
        for cost in all_costs:
            ys, errs = [], []
            for noise in noise_levels:
                runs = dict(grouped[noise])
                if cost not in runs:
                    ys.append(float('nan'))
                    errs.append(0.0)
                    continue
                m, s = final_value(runs[cost], col, scale, tail)
                ys.append(m)
                errs.append(s)
            xs = noise_levels
            color = cost_color[cost]
            ax.plot(xs, ys, marker='o', color=color, label=f'cost={cost}')
            ax.fill_between(xs,
                            [y - e for y, e in zip(ys, errs)],
                            [y + e for y, e in zip(ys, errs)],
                            alpha=0.15, color=color)

        # Baseline horizontal line
        if base_dfs is not None:
            bm_val, bs_val = final_value(base_dfs, col, scale, tail)
            ax.axhline(bm_val, color=BASELINE_COLOR, linestyle='--',
                       linewidth=1.4, alpha=0.8, label='Baseline PPO')
            ax.axhspan(bm_val - bs_val, bm_val + bs_val,
                       alpha=0.08, color=BASELINE_COLOR)

        # VLM threshold vertical line
        ax.axvline(0.4, color='red', linestyle=':', linewidth=1.6, alpha=0.8,
                   label='VLM threshold')

        ax.set_title(ax_title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Oracle Accuracy')
        ax.set_ylabel(ylabel)
        ax.set_xticks(noise_levels)
        ax.set_xticklabels([f'{n:.2f}' for n in noise_levels])
        ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else '.', exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Noisy oracle comparison plot (noise × cost)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--run', nargs=3, action='append',
                   metavar=('NOISE', 'COST', 'GLOB'),
                   help='Noise level, cost and CSV glob. Repeat for each combination.')
    p.add_argument('--base-csv',  nargs='+', default=None)
    p.add_argument('--out',       type=str, default='figures/noisy_oracle.png')
    p.add_argument('--title',     type=str, default='Noisy Oracle x Cost')
    p.add_argument('--smooth',    type=int, default=50)
    p.add_argument('--min-steps', type=int, default=0)
    p.add_argument('--cost-line', type=float, default=None,
                   metavar='COST',
                   help=(
                       'Single-row mode: fix a cost and plot one curve per noise level. '
                       'Only --run entries matching this cost are used.'
                   ))
    p.add_argument('--summary', action='store_true', default=False,
                   help=(
                       'Summary mode: plot final performance (last 20%% of training) '
                       'vs noise level, one line per cost. 2-panel figure.'
                   ))
    p.add_argument('--tail', type=float, default=0.2,
                   metavar='FRAC',
                   help='Fraction of training used to compute final performance (default: 0.2).')
    args = p.parse_args()

    if not args.run:
        p.error('Provide at least one --run NOISE COST GLOB.')

    base_dfs = None
    if args.base_csv:
        print('\n[Baseline PPO]')
        base_dfs = load_csvs(args.base_csv, args.min_steps)

    if args.summary:
        # ── Summary mode ─────────────────────────────────────────────────────
        grouped: dict[float, list] = defaultdict(list)
        for noise_str, cost_str, pattern in args.run:
            noise = float(noise_str)
            cost  = float(cost_str)
            print(f'\n[noise={noise}  cost={cost}]')
            dfs = load_csvs([pattern], args.min_steps)
            grouped[noise].append((cost, dfs))
        make_summary_figure(
            grouped, base_dfs,
            title=args.title, out=args.out, tail=args.tail,
        )

    elif args.cost_line is not None:
        # ── Single-row mode ───────────────────────────────────────────────────
        target = args.cost_line
        noise_runs: list[tuple[float, list]] = []
        for noise_str, cost_str, pattern in args.run:
            noise = float(noise_str)
            cost  = float(cost_str)
            if abs(cost - target) < 1e-9:
                print(f'\n[noise={noise}  cost={cost}]')
                dfs = load_csvs([pattern], args.min_steps)
                noise_runs.append((noise, dfs))
        if not noise_runs:
            p.error(
                f'No --run found with cost={target}. '
                f'Available costs: {sorted({float(c) for _, c, _ in args.run})}'
            )
        default_title = f'Noisy Oracle — cost={target}, all noise levels'
        title = args.title if args.title != 'Noisy Oracle x Cost' else default_title
        make_figure_cost_line(
            noise_runs, base_dfs,
            cost=target, title=title, out=args.out, smooth_w=args.smooth,
        )

    else:
        # ── Grid mode (original behaviour) ───────────────────────────────────
        grouped: dict[float, list] = defaultdict(list)
        for noise_str, cost_str, pattern in args.run:
            noise = float(noise_str)
            cost  = float(cost_str)
            print(f'\n[noise={noise}  cost={cost}]')
            dfs = load_csvs([pattern], args.min_steps)
            grouped[noise].append((cost, dfs))

        rows = [{'noise': n, 'costs': grouped[n]}
                for n in sorted(grouped, reverse=True)]

        make_figure(rows, base_dfs, title=args.title, out=args.out, smooth_w=args.smooth)
