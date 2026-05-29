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

def load_csvs(patterns, min_steps=0):
    """Charge un ou plusieurs CSV (glob patterns).
    Pour chaque seed, garde uniquement le run le plus récent (timestamp max).
    Filtre les runs qui n'atteignent pas min_steps global_step."""
    import re
    files = []
    for p in patterns:
        files += glob.glob(p)
    if not files:
        raise FileNotFoundError(f"Aucun CSV trouvé pour : {patterns}")

    # Groupe par seed, garde le timestamp le plus élevé
    seed_to_file = {}
    for f in files:
        m = re.search(r'seed(\d+)__(\d+)\.csv$', os.path.basename(f))
        if m:
            seed, ts = int(m.group(1)), int(m.group(2))
            if seed not in seed_to_file or ts > seed_to_file[seed][1]:
                seed_to_file[seed] = (f, ts)
        else:
            seed_to_file[f] = (f, 0)  # fallback si nom inattendu

    dfs = []
    for seed, (f, ts) in sorted(seed_to_file.items()):
        df = pd.read_csv(f)
        if df['global_step'].max() >= min_steps:
            dfs.append(df)
            print(f"  [load] seed={seed} — {os.path.basename(f)}")
        else:
            print(f"  [skip] seed={seed} — {os.path.basename(f)} (max_step={df['global_step'].max()} < {min_steps})")
    if not dfs:
        raise ValueError(f"Tous les CSVs filtrés (min_steps={min_steps})")
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
        # return / (1 + queries) — ne peut pas exploser, valide pour baseline aussi
        eff = df['ep_return'].values / (1 + df['queries_per_ep'].values)
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
        lo = clip[0] if clip[0] is not None else -np.inf
        hi = clip[1] if clip[1] is not None else  np.inf
        m  = np.clip(m, lo, hi)
        s  = np.clip(s, 0, (hi - lo) / 2 if np.isfinite(hi - lo) else np.inf)
    ax.plot(eps, m, color=color, label=label)
    ax.fill_between(eps,
                    np.clip(m - s, lo, hi) if clip else m - s,
                    np.clip(m + s, lo, hi) if clip else m + s,
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
    ax.set_ylim(-0.1, 2.2)
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

    # ── Query efficiency ──────────────────────────────────────────────────
    ax = axes[4]
    plot_metric(ax, eps_o, mean_o['query_efficiency'], std_o['query_efficiency'],
                C['oracle'], 'Oracle PPO', clip=(0, None))
    ax.set_title('Query Efficiency\nReturn / (1 + Queries)')
    ax.set_ylabel('Return / (1 + Queries)')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel('Global Step')


# ── Main figure ───────────────────────────────────────────────────────────────

def make_overview(conditions, base_dfs, env_id, out):
    """
    conditions: list of (label, dfs) — one row per oracle condition
    base_dfs:   baseline dfs (shown as reference in every row)
    """
    n_rows = len(conditions) + 1  # +1 for baseline row
    fig = plt.figure(figsize=(25, n_rows * 4.5))
    fig.patch.set_facecolor('#FAFAFA')

    gs = gridspec.GridSpec(
        n_rows, 5, figure=fig,
        hspace=0.55, wspace=0.35,
        left=0.06, right=0.98,
        top=0.95,  bottom=0.04,
    )

    rows = [[fig.add_subplot(gs[r, c]) for c in range(5)] for r in range(n_rows)]
    for row in rows:
        for ax in row:
            ax.set_facecolor('#F5F5F5')

    for i, (label, dfs) in enumerate(conditions):
        build_row(rows[i], dfs, base_dfs, label)
    build_row(rows[-1], base_dfs, base_dfs, 'Baseline PPO\n(no oracle)')

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
    import re as _re
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag',      type=str, default=None,
                        help='Exp-name suffix, e.g. "16_partial"')
    parser.add_argument('--costs',    nargs='+', default=None,
                        help='Explicit cost strings to include, e.g. 001 012 015 018 002')
    parser.add_argument('--log-dir',  type=str, default='logs')
    parser.add_argument('--base-csv', nargs='+', default=None)
    parser.add_argument('--env-id',   type=str, default='MiniGrid-DoorKey-16x16-v0')
    parser.add_argument('--out',      type=str, default='figures/overview.png')
    parser.add_argument('--min-steps',type=int, default=0)
    args = parser.parse_args()

    ms = args.min_steps

    if args.tag is None:
        parser.error('--tag is required')

    suffix = f'_{args.tag}' if args.tag else ''
    def p(name): return os.path.join(args.log_dir, f'{name}{suffix}__*.csv')

    # Which costs to show
    cost_strings = args.costs if args.costs else ['001', '002', '003', '004', '005']

    conditions = []
    for cs in cost_strings:
        try:
            dfs = load_csvs([p(f'oracle_paid_{cs}')], ms)
            v = int(cs)
            cost_val = f'{v/100:.2f}' if v <= 5 else f'{v/1000:.3f}'
            conditions.append((f'Paid Oracle\n(cost={cost_val})', dfs))
        except (FileNotFoundError, ValueError) as e:
            print(f'  [skip] oracle_paid_{cs}: {e}')

    if args.base_csv:
        base_dfs = load_csvs(args.base_csv, ms)
    else:
        base_dfs = load_csvs([p('baseline')], ms)

    make_overview(conditions, base_dfs, env_id=args.env_id, out=args.out)
