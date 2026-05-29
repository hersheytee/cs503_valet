"""
per_cost_plot.py — Un plot 1×4 par oracle cost (partial obs, 16x16).

Pour chaque coût : Return | Success | Oracle Usage % | Queries/ep
Baseline superposée en tirets sur chaque plot.
Un plot dédié baseline seul.

Usage:
    python per_cost_plot.py --log-dir logs --out-dir figures/per_cost
"""

import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.titleweight': 'bold',
    'axes.labelsize':   10,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
    'lines.linewidth':  2.0,
    'figure.dpi':       150,
})

C_ORACLE   = '#2196F3'
C_BASELINE = '#F44336'
C_GUIDED   = '#FF9800'

# ── cost tag → actual float value ─────────────────────────────────────────────

COSTS = [
    ('free',  0.000, 'oracle_free'),
    ('008',   0.008, 'oracle_paid_008'),
    ('001',   0.010, 'oracle_paid_001'),
    ('002',   0.020, 'oracle_paid_002'),
    ('003',   0.030, 'oracle_paid_003'),
    ('004',   0.040, 'oracle_paid_004'),
    ('005',   0.050, 'oracle_paid_005'),
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_csvs(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    seed_to_file = {}
    for f in files:
        m = re.search(r'seed(\d+)__(\d+)\.csv$', os.path.basename(f))
        if m:
            seed, ts = int(m.group(1)), int(m.group(2))
            if seed not in seed_to_file or ts > seed_to_file[seed][1]:
                seed_to_file[seed] = (f, ts)
    dfs = []
    for seed, (f, _) in sorted(seed_to_file.items()):
        dfs.append(pd.read_csv(f))
    return dfs if dfs else None


def align_and_aggregate(dfs, n_points=500):
    keys     = ['ep_return', 'success', 'guided_pct', 'queries_per_ep']
    max_step = min(df['global_step'].max() for df in dfs)
    grid     = np.linspace(0, max_step, n_points)
    arrays   = {k: [] for k in keys}
    for df in dfs:
        for k in keys:
            arrays[k].append(np.interp(grid, df['global_step'].values, df[k].values))
    stacked = {k: np.stack(arrays[k]) for k in keys}
    return grid, {k: stacked[k].mean(0) for k in keys}, {k: stacked[k].std(0) for k in keys}


def smooth(x, w=40):
    return pd.Series(x).rolling(w, min_periods=1).mean().values


def plot_metric(ax, steps, mean, std, color, label, ls='-', w=40, ylim=None):
    m = smooth(mean, w)
    s = smooth(std,  w)
    ax.plot(steps, m, color=color, label=label, linestyle=ls)
    ax.fill_between(steps, m - s, m + s, alpha=0.15, color=color)
    if ylim:
        ax.set_ylim(*ylim)


# ── Single plot ────────────────────────────────────────────────────────────────

def make_plot(tag, cost_val, oracle_dfs, base_dfs, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.patch.set_facecolor('#FAFAFA')
    for ax in axes:
        ax.set_facecolor('#F5F5F5')

    cost_str = f'Oracle cost = {cost_val:.3f}'.rstrip('0').rstrip('.')
    if tag == 'free':
        cost_str = 'Oracle free (cost = 0)'
    fig.suptitle(f'DoorKey-16x16 Partial Obs — {cost_str}',
                 fontsize=13, fontweight='bold', y=1.01)

    eps_o, mean_o, std_o = align_and_aggregate(oracle_dfs)
    eps_b, mean_b, std_b = align_and_aggregate(base_dfs)

    # ── Return ────────────────────────────────────────────────────────────────
    ax = axes[0]
    plot_metric(ax, eps_o, mean_o['ep_return'], std_o['ep_return'],
                C_ORACLE, 'Oracle PPO')
    plot_metric(ax, eps_b, mean_b['ep_return'], std_b['ep_return'],
                C_BASELINE, 'Baseline PPO', ls='--')
    ax.set_title('Episodic Return')
    ax.set_ylabel('Return')
    ax.set_ylim(-0.1, 2.2)
    ax.legend(fontsize=9)
    ax.set_xlabel('Global Step')

    # ── Success ───────────────────────────────────────────────────────────────
    ax = axes[1]
    plot_metric(ax, eps_o, mean_o['success'] * 100, std_o['success'] * 100,
                C_ORACLE, 'Oracle PPO', ylim=(-2, 105))
    plot_metric(ax, eps_b, mean_b['success'] * 100, std_b['success'] * 100,
                C_BASELINE, 'Baseline PPO', ls='--', ylim=(-2, 105))
    ax.set_title('Success Rate (%)')
    ax.set_ylabel('Success (%)')
    ax.legend(fontsize=9)
    ax.set_xlabel('Global Step')

    # ── Oracle Usage % ────────────────────────────────────────────────────────
    ax = axes[2]
    plot_metric(ax, eps_o, mean_o['guided_pct'], std_o['guided_pct'],
                C_GUIDED, 'Oracle PPO', ylim=(-2, 105))
    ax.axhline(0, color=C_BASELINE, linestyle='--', linewidth=1.5, label='Baseline (0%)')
    ax.set_title('Oracle Usage (%)')
    ax.set_ylabel('Guided Steps (%)')
    ax.legend(fontsize=9)
    ax.set_xlabel('Global Step')

    # ── Queries per episode ───────────────────────────────────────────────────
    ax = axes[3]
    qmax = max(mean_o['queries_per_ep'].max(), 1) * 1.3
    plot_metric(ax, eps_o, mean_o['queries_per_ep'], std_o['queries_per_ep'],
                C_GUIDED, 'Oracle PPO')
    ax.axhline(0, color=C_BASELINE, linestyle='--', linewidth=1.5, label='Baseline (0)')
    ax.set_title('Queries per Episode')
    ax.set_ylabel('Queries')
    ax.set_ylim(bottom=-0.5)
    ax.legend(fontsize=9)
    ax.set_xlabel('Global Step')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out_path}')
    plt.close(fig)


def make_baseline_plot(base_dfs, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.patch.set_facecolor('#FAFAFA')
    for ax in axes:
        ax.set_facecolor('#F5F5F5')

    fig.suptitle('DoorKey-16x16 Partial Obs — Baseline PPO (no oracle)',
                 fontsize=13, fontweight='bold', y=1.01)

    eps_b, mean_b, std_b = align_and_aggregate(base_dfs)

    titles  = ['Episodic Return', 'Success Rate (%)', 'Oracle Usage (%)', 'Queries per Episode']
    keys    = ['ep_return', 'success', 'guided_pct', 'queries_per_ep']
    ylabels = ['Return', 'Success (%)', 'Guided Steps (%)', 'Queries']
    scales  = [1, 100, 1, 1]
    ylims   = [(-0.1, 2.2), (-2, 105), (-2, 105), None]

    for ax, title, k, ylabel, scale, ylim in zip(axes, titles, keys, ylabels, scales, ylims):
        m = mean_b[k] * scale
        s = std_b[k]  * scale
        plot_metric(ax, eps_b, m, s, C_BASELINE, 'Baseline PPO',
                    ylim=ylim if ylim else (None, None))
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.set_xlabel('Global Step')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out_path}')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log-dir', type=str, default='logs')
    p.add_argument('--out-dir', type=str, default='figures/per_cost')
    p.add_argument('--env-tag', type=str, default='16_partial__MiniGrid-DoorKey-16x16-v0')
    args = p.parse_args()

    base_pattern = os.path.join(args.log_dir, f'baseline_{args.env_tag}__seed*__*.csv')
    base_dfs = load_csvs(base_pattern)
    if base_dfs is None:
        raise FileNotFoundError(f'Baseline CSVs not found: {base_pattern}')
    print(f'Baseline: {len(base_dfs)} seeds')

    # Baseline standalone
    make_baseline_plot(base_dfs, os.path.join(args.out_dir, 'cost_baseline.png'))

    # One plot per cost
    for tag, cost_val, prefix in COSTS:
        pattern = os.path.join(args.log_dir, f'{prefix}_{args.env_tag}__seed*__*.csv')
        dfs = load_csvs(pattern)
        if dfs is None:
            print(f'[skip] {prefix} — no CSVs found')
            continue
        print(f'{prefix}: {len(dfs)} seeds')
        fname = f'cost_{tag}.png'
        make_plot(tag, cost_val, dfs, base_dfs, os.path.join(args.out_dir, fname))

    print('All done.')


if __name__ == '__main__':
    main()
