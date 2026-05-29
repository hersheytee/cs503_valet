"""
merged_plot.py — All conditions on the same subplots.

Usage (full obs 16x16):
    python merged_plot.py \
        --tag 16 \
        --log-dir logs \
        --out figures/merged_16x16_full.png

Usage (partial obs 16x16):
    python merged_plot.py \
        --tag 16_partial \
        --log-dir logs \
        --out figures/merged_16x16_partial.png

Usage (8x8):
    python merged_plot.py \
        --tag "" \
        --log-dir logs \
        --out figures/merged_8x8.png

Manual CSV override:
    python merged_plot.py \
        --free-csv      logs/oracle_free_16__*.csv \
        --paid-csv      logs/oracle_paid_001_16__*.csv \
                        logs/oracle_paid_002_16__*.csv \
        --base-csv      logs/baseline_16__*.csv \
        --out figures/merged_16x16_full.png
"""

import argparse
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        16,
    'axes.titlesize':   18,
    'axes.titleweight': 'bold',
    'axes.labelsize':   16,
    'xtick.labelsize':  14,
    'ytick.labelsize':  14,
    'legend.fontsize':  13,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
    'lines.linewidth':  2.5,
    'figure.dpi':       150,
})

# All oracle costs we train with (excluding 0.0 = free)
ORACLE_COSTS = [0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.015, 0.018, 0.02, 0.03, 0.04, 0.05]
COST_LABELS  = {c: f'Oracle cost={c:.3f}'.rstrip('0').rstrip('.') for c in ORACLE_COSTS}

def cost_to_str(c):
    """Convert oracle cost float to the 3-digit string used in exp names."""
    v = int(round(c * 1000))
    if v % 10 == 0:   # 0.01→10, 0.02→20 … use ×100 convention → '001','002'…
        return f'{v // 10:03d}'
    return f'{v:03d}' # 0.007→7, 0.011→11 … use ×1000 convention → '007','011'…

# Colormap: plasma across all paid conditions
_paid_colors = cm.plasma(np.linspace(0.15, 0.85, len(ORACLE_COSTS)))
COST_COLORS  = {c: tuple(_paid_colors[i]) for i, c in enumerate(ORACLE_COSTS)}

FREE_COLOR     = cm.viridis(0.0)
BASELINE_COLOR = '#333333'


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csvs(patterns, min_steps=0):
    files = []
    for p in patterns:
        files += glob.glob(p)
    if not files:
        raise FileNotFoundError(f"No CSV found for: {patterns}")

    seed_to_file = {}
    for f in files:
        m = re.search(r'seed(\d+)__(\d+)\.csv$', os.path.basename(f))
        if m:
            seed, ts = int(m.group(1)), int(m.group(2))
            if seed not in seed_to_file or ts > seed_to_file[seed][1]:
                seed_to_file[seed] = (f, ts)

    dfs = []
    for seed, (f, ts) in sorted(seed_to_file.items()):
        df = pd.read_csv(f)
        max_s = df['global_step'].max()
        if max_s >= min_steps:
            dfs.append(df)
            print(f"  [load] seed={seed} max_step={max_s:,} — {os.path.basename(f)}")
        else:
            print(f"  [skip] seed={seed} max_step={max_s:,} < {min_steps} — {os.path.basename(f)}")
    if not dfs:
        raise ValueError(f"All CSVs filtered out (min_steps={min_steps})")
    return dfs


def align_and_aggregate(dfs, n_points=500):
    keys     = ['ep_return', 'success', 'guided_pct', 'queries_per_ep']
    max_step = min(df['global_step'].max() for df in dfs)
    grid     = np.linspace(0, max_step, n_points)

    interpolated = {k: [] for k in keys}
    for df in dfs:
        for k in keys:
            interpolated[k].append(
                np.interp(grid, df['global_step'].values, df[k].values)
            )

    arrays = {k: np.stack(interpolated[k]) for k in keys}
    return grid, {k: arrays[k].mean(0) for k in keys}, {k: arrays[k].std(0) for k in keys}


def smooth(x, w=40):
    return pd.Series(x).rolling(w, min_periods=1).mean().values


def plot_condition(ax, grid, mean, std, color, label, w=40, clip_lo=None, clip_hi=None, linestyle='-'):
    m = smooth(mean, w)
    s = smooth(std,  w)
    if clip_lo is not None:
        m = np.clip(m, clip_lo, np.inf)
    if clip_hi is not None:
        s = np.clip(s, 0, (clip_hi - (clip_lo or 0)) / 2)
    lo = m - s
    hi = m + s
    if clip_lo is not None:
        lo = np.clip(lo, clip_lo, np.inf)
    if clip_hi is not None:
        hi = np.clip(hi, -np.inf, clip_hi)
    ax.plot(grid, m, color=color, label=label, linestyle=linestyle)
    ax.fill_between(grid, lo, hi, alpha=0.15, color=color)


# ── Main figure ───────────────────────────────────────────────────────────────

def make_merged(all_data, out, title='PPO + Oracle vs Baseline'):
    """
    all_data: list of (key, color, label, grid, mean_dict, std_dict)
    """
    fig, axes = plt.subplots(1, 4, figsize=(32, 8))
    fig.patch.set_facecolor('#FAFAFA')
    for ax in axes:
        ax.set_facecolor('#F5F5F5')

    special_handles = []

    cost_min = ORACLE_COSTS[0]
    cost_max = ORACLE_COSTS[-1]
    norm     = matplotlib.colors.Normalize(vmin=cost_min, vmax=cost_max)
    sm       = matplotlib.cm.ScalarMappable(cmap='plasma', norm=norm)
    sm.set_array([])

    for key, color, label, grid, mean, std in all_data:
        is_baseline = key == 'baseline'
        is_free     = key == 'free'

        ls = '--' if is_baseline else '-'
        plot_condition(axes[0], grid, mean['ep_return'], std['ep_return'],
                       color, label, clip_lo=0, clip_hi=2.5, w=40, linestyle=ls)
        plot_condition(axes[1], grid, mean['success'] * 100, std['success'] * 100,
                       color, label, clip_lo=0, clip_hi=100, w=40, linestyle=ls)

        if not is_baseline:
            plot_condition(axes[2], grid, mean['guided_pct'], std['guided_pct'],
                           color, label, clip_lo=0, clip_hi=100, w=40)
            plot_condition(axes[3], grid, mean['queries_per_ep'], std['queries_per_ep'],
                           color, label, clip_lo=0, w=40)

        if is_baseline:
            special_handles.append(
                Line2D([0], [0], color=color, lw=2.5, linestyle='--', label='Baseline PPO'))
        elif is_free:
            special_handles.append(
                Line2D([0], [0], color=color, lw=2.5, linestyle='-', label='Oracle free (cost=0)'))

    axes[0].set_title('Episodic Return')
    axes[0].set_ylabel('Return')
    axes[0].set_ylim(0, 2.5)

    axes[1].set_title('Success Rate (%)')
    axes[1].set_ylabel('Success (%)')
    axes[1].set_ylim(-2, 105)

    axes[2].set_title('Oracle Usage (% of steps)')
    axes[2].set_ylabel('Guided Steps (%)')
    axes[2].set_ylim(-2, 105)
    axes[2].axhline(0, color=BASELINE_COLOR, linestyle='--', linewidth=1.0, alpha=0.5)

    axes[3].set_title('Oracle Queries per Episode')
    axes[3].set_ylabel('Queries / episode')
    axes[3].set_ylim(bottom=-0.1)
    axes[3].axhline(0, color=BASELINE_COLOR, linestyle='--', linewidth=1.0, alpha=0.5)

    for ax in axes:
        ax.set_xlabel('Environment Steps')
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{int(x/1e3)}k'))

    # ── Colorbar for oracle costs ──────────────────────────────────────────────
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.78, wspace=0.3)
    cbar_ax = fig.add_axes([0.15, 0.87, 0.55, 0.03])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.ax.set_title('Oracle cost  →  increasing', fontsize=13, pad=6)
    cbar.ax.tick_params(labelsize=11)

    # Small legend for baseline + oracle free
    if special_handles:
        fig.legend(handles=special_handles, loc='upper right',
                   bbox_to_anchor=(0.98, 0.96), fontsize=12,
                   frameon=True, framealpha=0.9)

    fig.suptitle(title, fontsize=17, fontweight='bold', y=0.99)
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else '.', exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def build_all_data_from_tag(tag, log_dir, min_steps):
    """Auto-discover CSVs by exp-name tag (e.g. '16', '16_partial', '')."""
    suffix = f'_{tag}' if tag else ''

    def pattern(name):
        return os.path.join(log_dir, f'{name}{suffix}__*.csv')

    all_data = []

    # Free oracle
    p = pattern('oracle_free')
    try:
        dfs = load_csvs([p], min_steps)
        grid, mean, std = align_and_aggregate(dfs)
        all_data.append(('free', tuple(FREE_COLOR), 'Oracle free (cost=0.0)', grid, mean, std))
    except (FileNotFoundError, ValueError) as e:
        print(f"  [skip] oracle_free: {e}")

    # Paid oracle costs
    for cost in ORACLE_COSTS:
        cost_str = cost_to_str(cost)
        p = pattern(f'oracle_paid_{cost_str}')
        try:
            dfs = load_csvs([p], min_steps)
            grid, mean, std = align_and_aggregate(dfs)
            all_data.append((f'paid_{cost_str}', COST_COLORS[cost],
                             COST_LABELS[cost], grid, mean, std))
        except (FileNotFoundError, ValueError) as e:
            print(f"  [skip] oracle_paid_{cost_str}: {e}")

    # Baseline
    p = pattern('baseline')
    try:
        dfs = load_csvs([p], min_steps)
        grid, mean, std = align_and_aggregate(dfs)
        all_data.append(('baseline', BASELINE_COLOR, 'Baseline PPO (no oracle)', grid, mean, std))
    except (FileNotFoundError, ValueError) as e:
        print(f"  [skip] baseline: {e}")

    return all_data


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib.ticker

    parser = argparse.ArgumentParser()
    # Auto-discovery mode
    parser.add_argument('--tag',       type=str, default=None,
                        help='Exp-name tag suffix, e.g. "16" → oracle_free_16__*.csv')
    parser.add_argument('--log-dir',   type=str, default='logs')
    # Manual override
    parser.add_argument('--free-csv',  nargs='+', default=None)
    parser.add_argument('--paid-csv',  nargs='+', action='append', default=None,
                        help='Repeatable: one group of CSVs per --paid-csv')
    parser.add_argument('--base-csv',  nargs='+', default=None)
    # Common
    parser.add_argument('--min-steps', type=int, default=0)
    parser.add_argument('--title',     type=str, default=None)
    parser.add_argument('--out',       type=str, default='figures/merged.png')
    args = parser.parse_args()

    import matplotlib.ticker

    if args.tag is not None:
        tag_clean = args.tag if args.tag else ''
        print(f"\nAuto-discovering CSVs with tag='{tag_clean}' in {args.log_dir}/")
        all_data = build_all_data_from_tag(tag_clean, args.log_dir, args.min_steps)
        title = args.title or f'PPO + Oracle — MiniGrid-DoorKey (tag={tag_clean or "8x8"})'
    else:
        if not args.free_csv or not args.base_csv:
            parser.error('Provide --tag, or both --free-csv and --base-csv')
        all_data = []
        dfs = load_csvs(args.free_csv, args.min_steps)
        grid, mean, std = align_and_aggregate(dfs)
        all_data.append(('free', tuple(FREE_COLOR), 'Oracle free (cost=0.0)', grid, mean, std))

        if args.paid_csv:
            for i, paid_group in enumerate(args.paid_csv):
                dfs = load_csvs(paid_group, args.min_steps)
                grid, mean, std = align_and_aggregate(dfs)
                # Try to infer cost from filename
                m = re.search(r'paid_(\d+)', paid_group[0])
                cost_str = m.group(1) if m else f'{i+1:03d}'
                cost_val = int(cost_str) / 1000
                color = COST_COLORS.get(cost_val, '#888888')
                label = COST_LABELS.get(cost_val, f'Oracle cost={cost_str}')
                all_data.append((f'paid_{cost_str}', color, label, grid, mean, std))

        dfs = load_csvs(args.base_csv, args.min_steps)
        grid, mean, std = align_and_aggregate(dfs)
        all_data.append(('baseline', BASELINE_COLOR, 'Baseline PPO (no oracle)', grid, mean, std))
        title = args.title or 'PPO + Oracle vs Baseline'

    if not all_data:
        print("No data found — nothing to plot.")
    else:
        make_merged(all_data, out=args.out, title=title)
