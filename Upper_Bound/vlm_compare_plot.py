"""
vlm_compare_plot.py — Comparaison des runs VLM oracle par coût.

Une ligne par condition (coût oracle), 6 colonnes de métriques :
    ep_return | success | guided_pct | queries_per_ep | vlm_accuracy | vlm_fallback_rate

Toutes les conditions sont optionnelles — inclure ce qu'on a.

Usage:
    # Avec une seule condition
    python vlm_compare_plot.py \
        --cost00-csv  "logs/vlm_wethink_baseline_cost00_seed*.csv" \
        --out figures/vlm_wethink.png

    # Sweep complet (quand toutes les seeds sont disponibles)
    python vlm_compare_plot.py \
        --cost00-csv  "logs/vlm_wethink_baseline_cost00_seed*.csv" \
        --cost001-csv "logs/vlm_wethink_baseline_cost001_seed*.csv" \
        --cost002-csv "logs/vlm_wethink_baseline_cost002_seed*.csv" \
        --cost005-csv "logs/vlm_wethink_baseline_cost005_seed*.csv" \
        --base-csv    "logs/baseline_seed*.csv" \
        --out figures/vlm_wethink_full.png
"""

import argparse
import glob
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

C = {
    'cost00':   '#2196F3',   # bleu  — oracle gratuit
    'cost001':  '#4CAF50',   # vert
    'cost002':  '#FF9800',   # orange
    'cost005':  '#F44336',   # rouge
    'baseline': '#9C27B0',   # violet — no oracle
    'vlm_acc':  '#00BCD4',   # cyan  — VLM accuracy
    'vlm_fb':   '#FF5722',   # deep orange — fallback
}

COLS = [
    ('ep_return',        'Episodic Return',            None,       (None, None)),
    ('success',          'Success Rate (%)',            100,        (0, 100)),
    ('guided_pct',       'Oracle Usage (%)',            None,       (0, 100)),
    ('queries_per_ep',   'Queries / Episode',           None,       (0, None)),
    ('vlm_accuracy',     'VLM Accuracy vs BFS',        None,       (0, 1)),
    ('vlm_fallback_rate','VLM Fallback Rate',           None,       (0, 1)),
]

# Colonnes pour le mode single-row (plus compact)
COLS_SINGLE = [
    ('ep_return',      'Episodic Return',   None, (None, None)),
    ('success',        'Success Rate (%)',   100,  (0, 100)),
    ('queries_per_ep', 'Queries / Episode',  None, (0, None)),
    ('vlm_accuracy',   'VLM Accuracy vs BFS', None, (0, 1)),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csvs(patterns: list[str], min_steps: int = 0) -> list[pd.DataFrame]:
    files = []
    for p in patterns:
        files += glob.glob(p)
    if not files:
        raise FileNotFoundError(f"Aucun CSV pour : {patterns}")

    # Garde le run le plus récent par seed
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
        raise ValueError(f"Tous les CSVs filtrés (min_steps={min_steps})")
    return dfs


def align_and_aggregate(dfs: list[pd.DataFrame], n_points: int = 500,
                        cols_def=None) -> tuple:
    """Interpole chaque seed sur une grille global_step commune, retourne (grid, mean, std)."""
    if cols_def is None:
        cols_def = COLS
    cols   = [c for c, *_ in cols_def]
    max_step = min(df['global_step'].max() for df in dfs)
    grid   = np.linspace(0, max_step, n_points)

    interpolated: dict[str, list] = {c: [] for c in cols}
    for df in dfs:
        x = df['global_step'].values
        for c in cols:
            if c in df.columns:
                y = df[c].values.astype(float)
                # Remplace NaN par la valeur voisine la plus proche pour l'interpolation
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


# ── Row drawing ───────────────────────────────────────────────────────────────

def draw_row(axes, dfs: list[pd.DataFrame], color: str, label: str,
             base_dfs: list[pd.DataFrame] | None, smooth_w: int = 50):
    grid, mean, std = align_and_aggregate(dfs)

    for ax, (col, title, scale, ylim) in zip(axes, COLS):
        m = smooth(mean[col], smooth_w) * (scale or 1)
        s = smooth(std[col],  smooth_w) * (scale or 1)

        # Seulement plotter si on a des données non-NaN
        valid = ~np.isnan(m)
        if valid.any():
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

    # Superpose la baseline en tirets sur les 4 premières colonnes si disponible
    if base_dfs is not None:
        bg, bm, bs = align_and_aggregate(base_dfs)
        for ax, (col, _, scale, ylim) in zip(axes[:4], COLS[:4]):
            bvals = smooth(bm[col], smooth_w) * (scale or 1)
            bstd  = smooth(bs[col], smooth_w) * (scale or 1)
            valid = ~np.isnan(bvals)
            if valid.any():
                ax.plot(bg[valid], bvals[valid],
                        color=C['baseline'], linestyle='--',
                        linewidth=1.4, alpha=0.7, label='Baseline PPO')
                lo, hi = ylim
                lo_arr = bvals - bstd if lo is None else np.clip(bvals - bstd, lo, None)
                hi_arr = bvals + bstd if hi is None else np.clip(bvals + bstd, None, hi)
                ax.fill_between(bg[valid], lo_arr[valid], hi_arr[valid],
                                alpha=0.10, color=C['baseline'])


# ── Figure builder — multi-lignes (une ligne par condition) ──────────────────

def make_figure(conditions: list[dict], title: str, out: str | None,
                smooth_w: int = 50, min_steps: int = 0):
    n_rows = len(conditions)
    n_cols = len(COLS)

    fig = plt.figure(figsize=(n_cols * 4.2, n_rows * 3.8))
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.55, wspace=0.35,
        left=0.06, right=0.98,
        top=0.93,  bottom=0.05,
    )

    for row_idx, cond in enumerate(conditions):
        axes = [fig.add_subplot(gs[row_idx, c]) for c in range(n_cols)]
        for ax in axes:
            ax.set_facecolor('white')

        draw_row(axes, cond['dfs'], cond['color'], cond['label'],
                 cond.get('base_dfs'), smooth_w)

        if row_idx == 0:
            for ax, (_, col_title, _, _) in zip(axes, COLS):
                ax.set_title(col_title)

        axes[0].set_ylabel(f"{cond['label']}\n{axes[0].get_ylabel()}", labelpad=6)

        for ax in axes:
            ax.set_xlabel('Global Step')
            ax.legend(fontsize=8, loc='upper left')

    fig.suptitle(title, fontsize=13, fontweight='bold')

    if out:
        os.makedirs(os.path.dirname(out) if os.path.dirname(out) else '.', exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f'Saved → {out}')
    else:
        plt.show()
    plt.close(fig)


# ── Figure builder — ligne unique (toutes conditions superposées) ─────────────

def make_single_row_figure(conditions: list[dict], base_dfs, title: str,
                           out: str | None, smooth_w: int = 50):
    n_cols = len(COLS_SINGLE)

    fig = plt.figure(figsize=(n_cols * 4.2, 4.5))
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(
        1, n_cols, figure=fig,
        hspace=0.3, wspace=0.35,
        left=0.06, right=0.98,
        top=0.88,  bottom=0.12,
    )

    axes = [fig.add_subplot(gs[0, c]) for c in range(n_cols)]
    for ax in axes:
        ax.set_facecolor('white')

    # Toutes les conditions sur les mêmes axes
    for cond in conditions:
        grid, mean, std = align_and_aggregate(cond['dfs'], cols_def=COLS_SINGLE)
        for ax, (col, col_title, scale, ylim) in zip(axes, COLS_SINGLE):
            m = smooth(mean[col], smooth_w) * (scale or 1)
            s = smooth(std[col],  smooth_w) * (scale or 1)
            valid = ~np.isnan(m)
            if not valid.any():
                continue
            ax.plot(grid[valid], m[valid], color=cond['color'], label=cond['label'])
            lo, hi = ylim
            lo_arr = m - s if lo is None else np.clip(m - s, lo, None)
            hi_arr = m + s if hi is None else np.clip(m + s, None, hi)
            ax.fill_between(grid[valid], lo_arr[valid], hi_arr[valid],
                            alpha=0.12, color=cond['color'])
            if ylim[0] is not None or ylim[1] is not None:
                lo = ylim[0] if ylim[0] is not None else ax.get_ylim()[0]
                hi = ylim[1] if ylim[1] is not None else ax.get_ylim()[1]
                ax.set_ylim(lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo))

    # Baseline en tirets avec incertitude
    if base_dfs is not None:
        bg, bm, bs = align_and_aggregate(base_dfs, cols_def=COLS_SINGLE)
        for ax, (col, _, scale, ylim) in zip(axes, COLS_SINGLE):
            bvals = smooth(bm[col], smooth_w) * (scale or 1)
            bstd  = smooth(bs[col], smooth_w) * (scale or 1)
            valid = ~np.isnan(bvals)
            if not valid.any():
                continue
            ax.plot(bg[valid], bvals[valid], color=C['baseline'], linestyle='--',
                    linewidth=1.4, alpha=0.8, label='Baseline PPO')
            lo, hi = ylim
            lo_arr = bvals - bstd if lo is None else np.clip(bvals - bstd, lo, None)
            hi_arr = bvals + bstd if hi is None else np.clip(bvals + bstd, None, hi)
            ax.fill_between(bg[valid], lo_arr[valid], hi_arr[valid],
                            alpha=0.10, color=C['baseline'])

    for ax, (_, col_title, _, _) in zip(axes, COLS_SINGLE):
        ax.set_title(col_title)
        ax.set_xlabel('Global Step')
        ax.legend(fontsize=8, loc='upper left')

    fig.suptitle(title, fontsize=13, fontweight='bold')

    if out:
        os.makedirs(os.path.dirname(out) if os.path.dirname(out) else '.', exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f'Saved → {out}')
    else:
        plt.show()
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='VLM oracle comparison plot (per-cost rows, VLM metrics included)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--cost00-csv',  nargs='+', default=None,
                   help='Glob(s) pour cost=0.00 (oracle gratuit)')
    p.add_argument('--cost001-csv', nargs='+', default=None,
                   help='Glob(s) pour cost=0.01')
    p.add_argument('--cost002-csv', nargs='+', default=None,
                   help='Glob(s) pour cost=0.02')
    p.add_argument('--cost005-csv', nargs='+', default=None,
                   help='Glob(s) pour cost=0.05')
    p.add_argument('--base-csv',    nargs='+', default=None,
                   help='Glob(s) pour baseline PPO (no oracle)')
    p.add_argument('--out',         type=str, default='figures/vlm_overview.png')
    p.add_argument('--title',       type=str, default='VLM Oracle — Training Metrics')
    p.add_argument('--smooth',      type=int, default=50,
                   help='Fenêtre de rolling mean (défaut: 50)')
    p.add_argument('--min-steps',   type=int, default=0,
                   help='Ignorer les CSVs avec moins de N global_steps')
    p.add_argument('--single-row',  action='store_true', default=False,
                   help='Superpose toutes les conditions sur une seule ligne de graphes')
    args = p.parse_args()

    CONDITION_SPECS = [
        (args.cost00_csv,  'cost=0.00 (free)',  C['cost00']),
        (args.cost001_csv, 'cost=0.01',          C['cost001']),
        (args.cost002_csv, 'cost=0.02',          C['cost002']),
        (args.cost005_csv, 'cost=0.05',          C['cost005']),
    ]

    base_dfs = None
    if args.base_csv:
        print('\n[Baseline]')
        base_dfs = load_csvs(args.base_csv, args.min_steps)

    conditions = []
    for globs, label, color in CONDITION_SPECS:
        if globs is None:
            continue
        print(f'\n[{label}]')
        dfs = load_csvs(globs, args.min_steps)
        conditions.append({
            'label':    label,
            'color':    color,
            'dfs':      dfs,
            'base_dfs': base_dfs,
        })

    if base_dfs and not conditions:
        # Baseline seule — on la plotte quand même
        conditions.append({
            'label':    'Baseline PPO',
            'color':    C['baseline'],
            'dfs':      base_dfs,
            'base_dfs': None,
        })

    if not conditions:
        p.error('Aucun CSV fourni. Utilise au moins --cost00-csv ou --base-csv.')

    if args.single_row:
        make_single_row_figure(conditions, base_dfs, title=args.title,
                               out=args.out, smooth_w=args.smooth)
    else:
        make_figure(conditions, title=args.title, out=args.out,
                    smooth_w=args.smooth, min_steps=args.min_steps)
