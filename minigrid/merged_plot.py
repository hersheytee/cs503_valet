"""
merged_plot.py — All conditions on the same 5 subplots.

Usage:
    python merged_plot.py \
        --free-csv      logs/oracle_free__*.csv \
        --paid-low-csv  logs/oracle_paid_001__*.csv \
        --paid-mid-csv  logs/oracle_paid_002__*.csv \
        --paid-high-csv logs/oracle_paid_005__*.csv \
        --base-csv      logs/baseline__*.csv \
        --out figures/merged_doorkey.png
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
    'legend.fontsize':  14,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
    'lines.linewidth':  2.5,
    'figure.dpi':       150,
})

# Oracle conditions: viridis colormap
_reds = cm.viridis(np.linspace(0, 0.9, 4))

C = {
    'free':     tuple(_reds[0]),
    'low':      tuple(_reds[1]),
    'mid':      tuple(_reds[2]),
    'high':     tuple(_reds[3]),
    'baseline': '#000000',  # noir
}

LABELS = {
    'free':     'Oracle free (cost=0.0)',
    'low':      'Oracle cost=0.01',
    'mid':      'Oracle cost=0.02',
    'high':     'Oracle cost=0.05',
    'baseline': 'Baseline PPO (no oracle)',
}


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
        if df['global_step'].max() >= min_steps:
            dfs.append(df)
            print(f"  [load] seed={seed} — {os.path.basename(f)}")
        else:
            print(f"  [skip] seed={seed} — max_step={df['global_step'].max()} < {min_steps}")
    if not dfs:
        raise ValueError(f"All CSVs filtered (min_steps={min_steps})")
    return dfs


def align_and_aggregate(dfs, n_points=500):
    keys     = ['ep_return', 'success', 'guided_pct', 'queries_per_ep']
    max_step = min(df['global_step'].max() for df in dfs)
    grid     = np.linspace(0, max_step, n_points)

    interpolated = {k: [] for k in keys}
    for df in dfs:
        eff = df['ep_return'].values / (1 + df['queries_per_ep'].values)
        interpolated.setdefault('query_efficiency', []).append(
            np.interp(grid, df['global_step'].values, eff)
        )
        for k in keys:
            interpolated[k].append(
                np.interp(grid, df['global_step'].values, df[k].values)
            )

    all_keys = keys + ['query_efficiency']
    arrays   = {k: np.stack(interpolated[k]) for k in all_keys}
    return grid, {k: arrays[k].mean(0) for k in all_keys}, {k: arrays[k].std(0) for k in all_keys}


def smooth(x, w=50):
    return pd.Series(x).rolling(w, min_periods=1).mean().values


def plot_condition(ax, grid, mean, std, color, label, w=50, clip_lo=None, clip_hi=None):
    m = smooth(mean, w)
    s = smooth(std,  w)
    if clip_lo is not None:
        m = np.clip(m, clip_lo, np.inf)
    if clip_hi is not None:
        m = np.clip(m, -np.inf, clip_hi)
        s = np.clip(s, 0, (clip_hi - (clip_lo or 0)) / 2)
    lo = np.clip(m - s, clip_lo or -np.inf, clip_hi or np.inf)
    hi = np.clip(m + s, clip_lo or -np.inf, clip_hi or np.inf)
    ax.plot(grid, m, color=color, label=label)
    ax.fill_between(grid, lo, hi, alpha=0.15, color=color)


# ── Main figure ───────────────────────────────────────────────────────────────

def make_merged(all_data, out):
    """
    all_data: dict with keys 'free','low','mid','high','baseline'
              each value is (grid, mean_dict, std_dict)
    """
    fig, axes = plt.subplots(4, 1, figsize=(8, 22))
    fig.patch.set_facecolor('#FAFAFA')
    for ax in axes:
        ax.set_facecolor('#F5F5F5')

    for key, (grid, mean, std) in all_data.items():
        color = C[key]
        label = LABELS[key]

        # Return
        plot_condition(axes[0], grid, mean['ep_return'], std['ep_return'],
                       color, label, clip_lo=0, clip_hi=2.2)

        # Success rate
        plot_condition(axes[1], grid, mean['success']*100, std['success']*100,
                       color, label, clip_lo=0, clip_hi=100)

        # Oracle usage — baseline is always 0, skip to avoid clutter
        if key != 'baseline':
            plot_condition(axes[2], grid, mean['guided_pct'], std['guided_pct'],
                           color, label, clip_lo=0, clip_hi=100)

        # Queries per episode — baseline is always 0, skip
        if key != 'baseline':
            plot_condition(axes[3], grid, mean['queries_per_ep'], std['queries_per_ep'],
                           color, label, clip_lo=0)

    # Titles & labels
    axes[0].set_title('Episodic Return')
    axes[0].set_ylabel('Return')
    axes[0].set_ylim(0, 2.2)

    axes[1].set_title('Success Rate (%)')
    axes[1].set_ylabel('Success (%)')
    axes[1].set_ylim(-2, 105)

    axes[2].set_title('Oracle Usage (%)')
    axes[2].set_ylabel('Guided Steps (%)')
    axes[2].set_ylim(-2, 105)
    axes[2].axhline(0, color=C['baseline'], linestyle='--', linewidth=1.2, label='Baseline (0%)')

    axes[3].set_title('Queries per Episode')
    axes[3].set_ylabel('Queries')
    axes[3].set_ylim(bottom=-0.1)
    axes[3].axhline(0, color=C['baseline'], linestyle='--', linewidth=1.2, label='Baseline (0)')

    for ax in axes:
        ax.set_xlabel('Global Step')

    # Shared legend
    handles = [
        Line2D([0], [0], color=C['free'],     lw=2, label=LABELS['free']),
        Line2D([0], [0], color=C['low'],      lw=2, label=LABELS['low']),
        Line2D([0], [0], color=C['mid'],      lw=2, label=LABELS['mid']),
        Line2D([0], [0], color=C['high'],     lw=2, label=LABELS['high']),
        Line2D([0], [0], color=C['baseline'], lw=2, label=LABELS['baseline'], linestyle='--'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2,
               fontsize=14, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.0))

    fig.suptitle('PPO + Oracle vs Baseline — MiniGrid-DoorKey-8x8-v0',
                 fontsize=16, fontweight='bold', y=1.04)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else '.', exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--free-csv',      nargs='+', required=True)
    parser.add_argument('--paid-low-csv',  nargs='+', required=True)
    parser.add_argument('--paid-mid-csv',  nargs='+', required=True)
    parser.add_argument('--paid-high-csv', nargs='+', required=True)
    parser.add_argument('--base-csv',      nargs='+', required=True)
    parser.add_argument('--min-steps',     type=int,  default=0)
    parser.add_argument('--out',           type=str,  default='figures/merged.png')
    args = parser.parse_args()

    ms = args.min_steps
    all_data = {
        'free':     align_and_aggregate(load_csvs(args.free_csv,      ms)),
        'low':      align_and_aggregate(load_csvs(args.paid_low_csv,  ms)),
        'mid':      align_and_aggregate(load_csvs(args.paid_mid_csv,  ms)),
        'high':     align_and_aggregate(load_csvs(args.paid_high_csv, ms)),
        'baseline': align_and_aggregate(load_csvs(args.base_csv,      ms)),
    }

    make_merged(all_data, out=args.out)
