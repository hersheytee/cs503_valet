"""
plot.py — Generate all training metric plots.

Can be used in two modes:
    1. From a real CSV log:
         python plot.py --csv logs/run.csv --out figures/

    2. Demo mode with simulated curves (default):
         python plot.py --demo
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.labelsize':    11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
    'lines.linewidth':   2.2,
    'figure.dpi':        150,
})

# ── Colour palette ──────────────────────────────────────────────────────────
C = {
    'oracle':   '#2196F3',   # blue  – oracle-guided agent
    'ppo':      '#F44336',   # red   – vanilla PPO baseline
    'shade':    '#2196F3',
    'accent':   '#FF9800',   # orange – special markers
    'green':    '#4CAF50',
    'purple':   '#9C27B0',
}

# ── Rolling statistics ───────────────────────────────────────────────────────
def smooth(x, w=30):
    """Rolling mean."""
    return pd.Series(x).rolling(w, min_periods=1).mean().values

def shade(ax, x, y, w=30, color=C['shade'], alpha=0.15):
    """Rolling std shading."""
    ser  = pd.Series(y)
    mean = ser.rolling(w, min_periods=1).mean().values
    std  = ser.rolling(w, min_periods=1).std().fillna(0).values
    ax.fill_between(x, mean - std, mean + std, alpha=alpha, color=color)


# ── Simulate realistic training curves ───────────────────────────────────────
def simulate(n_eps=2000, seed=42):
    rng = np.random.default_rng(seed)
    eps = np.arange(n_eps)
    T   = n_eps

    # ── Helpers ──────────────────────────────────────────────────────────
    def sigmoid(x, k=1, x0=0):
        return 1 / (1 + np.exp(-k * (x - x0)))

    def decay(x, rate=0.003, floor=0.0):
        return floor + (1 - floor) * np.exp(-rate * x)

    # ── Guided agent ─────────────────────────────────────────────────────
    # Episodic return: rises from ~0 to ~0.85, with noise
    base_return = 0.85 * sigmoid(eps, k=0.005, x0=600)
    ep_return   = np.clip(base_return + rng.normal(0, 0.08, T), 0, 1)

    # Success rate: smoother version of return > 0.5
    base_sr  = sigmoid(eps, k=0.006, x0=550)
    sr_noise = rng.normal(0, 0.05, T)
    success  = np.clip(base_sr + sr_noise, 0, 1)

    # Guided pct: starts 85%, decays to ~5%
    guided_pct = decay(eps, rate=0.004, floor=0.05) * 0.85
    guided_pct += rng.normal(0, 0.03, T)
    guided_pct  = np.clip(guided_pct, 0, 1) * 100

    # Queries per episode
    queries_per_ep = guided_pct / 100 * (4 + 2 * decay(eps, 0.003))
    queries_per_ep = np.clip(queries_per_ep + rng.normal(0, 0.2, T), 0, None)

    # BC loss: starts high, decays
    bc_loss = 1.8 * decay(eps, 0.006) + rng.normal(0, 0.08, T)
    bc_loss = np.clip(bc_loss, 0, None)

    # Agreement rate: starts 0.1, rises to 0.9
    agreement = 0.9 * sigmoid(eps, k=0.006, x0=600)
    agreement += rng.normal(0, 0.05, T)
    agreement  = np.clip(agreement, 0, 1)

    # First unguided success episode
    # Find where rolling success is decent AND guided_pct is below 15%
    smooth_sr  = smooth(success, 50)
    smooth_gp  = smooth(guided_pct, 50)
    candidates = np.where((smooth_sr > 0.5) & (smooth_gp < 15))[0]
    first_ug   = candidates[0] if len(candidates) > 0 else int(T * 0.7)

    # Cumulative oracle calls (for Pareto plot)
    cum_queries = np.cumsum(queries_per_ep)

    # ── Vanilla PPO baseline (no oracle) ─────────────────────────────────
    base_ppo   = 0.45 * sigmoid(eps, k=0.003, x0=900)
    ppo_return = np.clip(base_ppo + rng.normal(0, 0.07, T), 0, 1)

    base_ppo_sr  = sigmoid(eps, k=0.003, x0=950)
    ppo_sr_noise = rng.normal(0, 0.06, T)
    ppo_success  = np.clip(0.5 * base_ppo_sr + ppo_sr_noise, 0, 1)

    return {
        'eps':            eps,
        'ep_return':      ep_return,
        'success':        success,
        'guided_pct':     guided_pct,
        'queries_per_ep': queries_per_ep,
        'bc_loss':        bc_loss,
        'agreement':      agreement,
        'first_ug':       first_ug,
        'cum_queries':    cum_queries,
        'ppo_return':     ppo_return,
        'ppo_success':    ppo_success,
    }


# ── Individual plot functions ────────────────────────────────────────────────

def plot_episodic_return(ax, d):
    x = d['eps']
    ax.plot(x, smooth(d['ep_return']),  color=C['oracle'], label='Oracle PPO')
    ax.plot(x, smooth(d['ppo_return']), color=C['ppo'],    label='Vanilla PPO', linestyle='--')
    shade(ax, x, d['ep_return'],  color=C['oracle'])
    shade(ax, x, d['ppo_return'], color=C['ppo'])
    ax.set_title('Episodic Return')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)


def plot_success_rate(ax, d):
    x = d['eps']
    ax.plot(x, smooth(d['success'],   50) * 100, color=C['oracle'], label='Oracle PPO')
    ax.plot(x, smooth(d['ppo_success'], 50) * 100, color=C['ppo'],  label='Vanilla PPO', linestyle='--')
    shade(ax, x, d['success']     * 100, color=C['oracle'])
    shade(ax, x, d['ppo_success'] * 100, color=C['ppo'])
    ax.set_title('Success Rate')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim(-2, 102)
    ax.legend(fontsize=9)


def plot_guided_pct(ax, d):
    x = d['eps']
    ax.plot(x, smooth(d['guided_pct'], 40), color=C['oracle'])
    shade(ax, x, d['guided_pct'], color=C['oracle'])
    ax.axhline(15, color=C['accent'], linestyle=':', linewidth=1.5,
               label='15% threshold')
    ax.set_title('Oracle Guidance Usage')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Guided Steps (%)')
    ax.set_ylim(-2, 95)
    ax.legend(fontsize=9)


def plot_queries_per_ep(ax, d):
    x = d['eps']
    ax.plot(x, smooth(d['queries_per_ep'], 40), color=C['purple'])
    shade(ax, x, d['queries_per_ep'], color=C['purple'])
    ax.set_title('Oracle Queries per Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Queries')
    ax.set_ylim(bottom=0)


def plot_first_unguided(ax, d):
    """Highlight the first fully autonomous success."""
    x      = d['eps']
    first  = d['first_ug']
    sr     = smooth(d['success'], 50) * 100

    ax.plot(x, sr, color=C['oracle'])
    shade(ax, x, d['success'] * 100, color=C['oracle'])

    ax.axvline(first, color=C['accent'], linewidth=2.0,
               linestyle='--', label=f'First unguided success\n(ep {first})')

    ax.scatter([first], [sr[first]], color=C['accent'],
               s=80, zorder=5)

    ax.set_title('First Fully Autonomous Success')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim(-2, 102)
    ax.legend(fontsize=9)


def plot_bc_loss(ax, d):
    x = d['eps']
    ax.plot(x, smooth(d['bc_loss'], 30), color=C['green'])
    shade(ax, x, d['bc_loss'], color=C['green'])
    ax.set_title('Behavior Cloning Loss')
    ax.set_xlabel('Episode')
    ax.set_ylabel('BC Loss')
    ax.set_ylim(bottom=0)


def plot_agreement_rate(ax, d):
    x = d['eps']
    ax.plot(x, smooth(d['agreement'], 40) * 100, color=C['purple'])
    shade(ax, x, d['agreement'] * 100, color=C['purple'])
    ax.set_title('Policy–Oracle Agreement Rate')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Agreement (%)')
    ax.set_ylim(-2, 102)
    # Annotate final value
    final = smooth(d['agreement'], 40)[-1] * 100
    ax.annotate(f'{final:.0f}%', xy=(x[-1], final),
                xytext=(-40, -15), textcoords='offset points',
                fontsize=10, color=C['purple'],
                arrowprops=dict(arrowstyle='->', color=C['purple'], lw=1.2))


def plot_pareto(ax, d):
    """Return vs cumulative oracle calls — sample efficiency curve."""
    cum_q  = d['cum_queries']
    ret    = smooth(d['ep_return'], 30)

    # Oracle PPO trace
    ax.plot(cum_q, ret, color=C['oracle'], label='Oracle PPO')

    # Vanilla PPO: no oracle calls, just a horizontal progression
    ppo_ret = smooth(d['ppo_return'], 30)
    # Map PPO's episodes to a fake "0 queries" x axis for comparison
    max_q = cum_q[-1]
    ax.axhline(ppo_ret[-1], color=C['ppo'], linestyle='--',
               label=f'Vanilla PPO final ({ppo_ret[-1]:.2f})')

    # Mark key milestones
    for threshold, label in [(0.5, '50%'), (0.75, '75%')]:
        idx = np.where(ret >= threshold)[0]
        if len(idx):
            q_at = cum_q[idx[0]]
            ax.scatter([q_at], [threshold], color=C['accent'], s=70, zorder=5)
            ax.annotate(f'{label} @ {int(q_at)} calls',
                        xy=(q_at, threshold),
                        xytext=(15, -12), textcoords='offset points',
                        fontsize=8.5, color=C['accent'])

    ax.set_title('Sample Efficiency\n(Return vs Oracle Calls)')
    ax.set_xlabel('Cumulative Oracle Calls')
    ax.set_ylabel('Episodic Return')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)


# ── Master figure ────────────────────────────────────────────────────────────

def make_figure(d, title='Oracle-Guided PPO — MiniGrid Training Metrics', out=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor('#FAFAFA')
    axes = axes.flatten()

    for ax in axes:
        ax.set_facecolor('#F5F5F5')

    plot_episodic_return(axes[0], d)
    plot_success_rate(axes[1], d)
    plot_guided_pct(axes[2], d)
    plot_queries_per_ep(axes[3], d)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(title, fontsize=13, fontweight='bold')

    if out:
        os.makedirs(os.path.dirname(out) if os.path.dirname(out) else '.', exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'Saved → {out}')
    else:
        plt.show()

    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',  type=str, default=None,
                        help='Path to training CSV log')
    parser.add_argument('--demo', action='store_true', default=True,
                        help='Use simulated data (default)')
    parser.add_argument('--out',  type=str, default='figures/metrics.png',
                        help='Output image path')
    parser.add_argument('--env',  type=str, default='MiniGrid-Empty-5x5',
                        help='Env name for plot title')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.csv:
        df   = pd.read_csv(args.csv)
        data = {
            'eps':            df['episode'].values,
            'ep_return':      df['ep_return'].values,
            'success':        df['success'].values,
            'guided_pct':     df['guided_pct'].values,
            'queries_per_ep': df['queries_per_ep'].values,
            'ppo_return':     np.zeros(len(df)),
            'ppo_success':    np.zeros(len(df)),
        }
    else:
        data = simulate(n_eps=2000, seed=args.seed)

    make_figure(data,
                title=f'Oracle-Guided PPO — {args.env} Training Metrics',
                out=args.out)