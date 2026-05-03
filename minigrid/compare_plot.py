"""
compare_plot.py — Vue d'ensemble : PPO+Oracle vs Baseline PPO.

Génère 1 figure par env avec 2 lignes (oracle gratuit / oracle payant)
et 4 colonnes de métriques.

Usage:
    python compare_plot.py \
        --free-csv   logs/oracle_free_*.csv  \
        --paid-csv   logs/oracle_paid_*.csv  \
        --base-csv   logs/baseline_*.csv     \
        --env-id     MiniGrid-DoorKey-8x8-v0 \
        --oracle-cost 0.01 \
        --out        figures/overview_doorkey.png
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'axes.titlesize':   11,
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

C = {
    'oracle':   '#2196F3',  # bleu  – oracle
    'baseline': '#F44336',  # rouge – baseline PPO
    'guided':   '#FF9800',  # orange – guided%
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csvs(patterns):
    """Charge un ou plusieurs CSV (glob patterns), retourne un DataFrame."""
    files = []
    for p in patterns:
        files += glob.glob(p)
    if not files:
        raise FileNotFoundError(f"Aucun CSV trouvé pour : {patterns}")
    dfs = [pd.read_csv(f) for f in sorted(files)]
    return dfs


def align_and_aggregate(dfs, n_points=500):
    """
    Aligne plusieurs runs sur un axe global_step commun (interpolation),
    puis calcule mean ± std entre seeds.
    """
    keys     = ['ep_return', 'success', 'guided_pct', 'queries_per_ep']
    max_step = min(df['global_step'].max() for df in dfs)
    grid     = np.linspace(0, max_step, n_points)

    interpolated = {k: [] for k in keys}
    for df in dfs:
        # query_efficiency calculée par épisode avant interpolation
        eff = df['success'].values / (df['queries_per_ep'].values + 1e-6)
        interpolated.setdefault('query_efficiency', []).append(
            np.interp(grid, df['global_step'].values, eff)
        )
        for k in keys:
            interp = np.interp(grid, df['global_step'].values, df[k].values)
            interpolated[k].append(interp)

    all_keys = keys + ['query_efficiency']
    arrays   = {k: np.stack(interpolated[k]) for k in all_keys}
    return grid, {k: arrays[k].mean(0) for k in all_keys}, {k: arrays[k].std(0) for k in all_keys}


def smooth(x, w=50):
    return pd.Series(x).rolling(w, min_periods=1).mean().values


def plot_metric(ax, eps, mean, std, color, label, w=50, clip=None):
    m = smooth(mean, w)
    s = smooth(std,  w)
    if clip is not None:
        lo, hi = clip
        m = np.clip(m, lo, hi)
        s = np.clip(s, 0, (hi - lo) / 2)
    ax.plot(eps, m, color=color, label=label)
    ax.fill_between(eps, np.clip(m - s, *clip) if clip else m - s,
                         np.clip(m + s, *clip) if clip else m + s,
                    alpha=0.15, color=color)


# ── Row builder ───────────────────────────────────────────────────────────────

def build_row(axes, oracle_dfs, base_dfs, row_label):
    """Remplit une ligne de 5 subplots."""
    eps_o, mean_o, std_o = align_and_aggregate(oracle_dfs)
    eps_b, mean_b, std_b = align_and_aggregate(base_dfs)

    # ── Return ────────────────────────────────────────────────────────────
    ax = axes[0]
    plot_metric(ax, eps_o, mean_o['ep_return'], std_o['ep_return'],
                C['oracle'],   'Oracle PPO')
    plot_metric(ax, eps_b, mean_b['ep_return'], std_b['ep_return'],
                C['baseline'], 'Baseline PPO')
    ax.set_title('Episodic Return')
    ax.set_ylabel(f'{row_label}\nReturn')
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=8)

    # ── Success rate ──────────────────────────────────────────────────────
    ax = axes[1]
    plot_metric(ax, eps_o, mean_o['success'] * 100, std_o['success'] * 100,
                C['oracle'],   'Oracle PPO',   clip=(0, 100))
    plot_metric(ax, eps_b, mean_b['success'] * 100, std_b['success'] * 100,
                C['baseline'], 'Baseline PPO', clip=(0, 100))
    ax.set_title('Success Rate (%)')
    ax.set_ylabel('Success (%)')
    ax.set_ylim(-2, 105)
    ax.legend(fontsize=8)

    # ── Guided % ──────────────────────────────────────────────────────────
    ax = axes[2]
    plot_metric(ax, eps_o, mean_o['guided_pct'], std_o['guided_pct'],
                C['guided'], 'Oracle PPO', clip=(0, 100))
    ax.axhline(0, color=C['baseline'], linestyle='--', linewidth=1.2,
               label='Baseline (0%)')
    ax.set_title('Oracle Usage (%)')
    ax.set_ylabel('Guided Steps (%)')
    ax.set_ylim(-2, 105)
    ax.legend(fontsize=8)

    # ── Queries per episode ───────────────────────────────────────────────
    ax = axes[3]
    plot_metric(ax, eps_o, mean_o['queries_per_ep'], std_o['queries_per_ep'],
                C['guided'], 'Oracle PPO')
    ax.axhline(0, color=C['baseline'], linestyle='--', linewidth=1.2,
               label='Baseline (0)')
    ax.set_title('Queries per Episode')
    ax.set_ylabel('Queries')
    ax.set_ylim(bottom=-0.1)
    ax.legend(fontsize=8)

    # ── Query efficiency (oracle only) ────────────────────────────────────
    ax = axes[4]
    plot_metric(ax, eps_o, mean_o['query_efficiency'], std_o['query_efficiency'],
                C['oracle'], 'Oracle PPO', clip=(0, None))
    ax.set_title('Query Efficiency\n(Success / Query)')
    ax.set_ylabel('Success / Query')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel('Global Step')


# ── Main figure ───────────────────────────────────────────────────────────────

def make_overview(free_dfs, paid_low_dfs, paid_high_dfs, base_dfs,
                  env_id, cost_low, cost_high, out):
    fig = plt.figure(figsize=(25, 14))
    fig.patch.set_facecolor('#FAFAFA')

    gs = gridspec.GridSpec(
        3, 5, figure=fig,
        hspace=0.55, wspace=0.35,
        left=0.06, right=0.98,
        top=0.90,  bottom=0.07,
    )

    row0 = [fig.add_subplot(gs[0, c]) for c in range(5)]
    row1 = [fig.add_subplot(gs[1, c]) for c in range(5)]
    row2 = [fig.add_subplot(gs[2, c]) for c in range(5)]

    for ax in row0 + row1 + row2:
        ax.set_facecolor('#F5F5F5')


    build_row(row0, free_dfs,      base_dfs, f'Free Oracle\n(cost=0.0)')
    build_row(row1, paid_low_dfs,  base_dfs, f'Paid Oracle\n(cost={cost_low})')
    build_row(row2, paid_high_dfs, base_dfs, f'Paid Oracle\n(cost={cost_high})')

    handles = [
        Line2D([0], [0], color=C['oracle'],   lw=2, label='PPO + Oracle'),
        Line2D([0], [0], color=C['baseline'], lw=2, label='Baseline PPO (no oracle)', linestyle='--'),
        Line2D([0], [0], color=C['guided'],   lw=2, label='Oracle usage metric'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=3,
               fontsize=10, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.97))

    fig.suptitle(f'PPO + Oracle vs Baseline — {env_id}',
                 fontsize=14, fontweight='bold', y=1.0)

    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else '.', exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--free-csv',      nargs='+', required=True)
    parser.add_argument('--paid-low-csv',  nargs='+', required=True)
    parser.add_argument('--paid-high-csv', nargs='+', required=True)
    parser.add_argument('--base-csv',      nargs='+', required=True)
    parser.add_argument('--env-id',        type=str,   default='MiniGrid-DoorKey-8x8-v0')
    parser.add_argument('--cost-low',      type=float, default=0.01)
    parser.add_argument('--cost-high',     type=float, default=0.05)
    parser.add_argument('--out',           type=str,   default='figures/overview.png')
    args = parser.parse_args()

    free_dfs      = load_csvs(args.free_csv)
    paid_low_dfs  = load_csvs(args.paid_low_csv)
    paid_high_dfs = load_csvs(args.paid_high_csv)
    base_dfs      = load_csvs(args.base_csv)

    make_overview(free_dfs, paid_low_dfs, paid_high_dfs, base_dfs,
                  env_id=args.env_id,
                  cost_low=args.cost_low,
                  cost_high=args.cost_high,
                  out=args.out)
