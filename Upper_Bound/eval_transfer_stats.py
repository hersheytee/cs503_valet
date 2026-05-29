"""
eval_transfer_stats.py — Run N episodes of zero-shot transfer and collect stats.

Usage (single run):
    python eval_transfer_stats.py \
        --checkpoint checkpoints/best__oracle_paid_001_16_partial__MiniGrid-DoorKey-16x16-v0.pt \
        --env-id MiniGrid-Fetch-16x16-N3-v0 --env-type fetch \
        --partial-obs --n-episodes 100 --label "Oracle 0.01 (argmax)" \
        --out figures/transfer_stats.png

Usage (comparison — multiple runs saved to same CSV then plotted):
    Called automatically by submit_eval_transfer_stats.sh
"""

import argparse
import os
import numpy as np
import torch
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 13,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
    'figure.dpi': 150,
})

from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
from model_partial import CNNPolicy as CNNPolicyPartial
from model_large   import CNNPolicy as CNNPolicyLarge
from model         import CNNPolicy as CNNPolicySmall
from oracle_transfer import get_optimal_steps


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  type=str, required=True)
    p.add_argument('--env-id',      type=str, default='MiniGrid-Fetch-16x16-N3-v0')
    p.add_argument('--env-type',    type=str, default='fetch')
    p.add_argument('--no-oracle',   action='store_true', default=False)
    p.add_argument('--partial-obs', action='store_true', default=False)
    p.add_argument('--large-model', action='store_true', default=False)
    p.add_argument('--stochastic',  action='store_true', default=False)
    p.add_argument('--hidden-dim',  type=int, default=256)
    p.add_argument('--tile-size',   type=int, default=8)
    p.add_argument('--n-episodes',  type=int, default=100)
    p.add_argument('--seed',        type=int, default=200)
    p.add_argument('--label',       type=str, default=None)
    p.add_argument('--csv-out',     type=str, default=None,
                   help='Append results to this CSV (for multi-run comparison)')
    p.add_argument('--out',         type=str, default='figures/transfer_stats.png')
    return p.parse_args()


def make_env(env_id, env_type, tile_size, no_oracle, partial_obs):
    from env_wrapper import get_oracle_action
    from minigrid.wrappers import RGBImgPartialObsWrapper

    inner = gym.make(env_id, render_mode=None)
    if partial_obs:
        inner = RGBImgPartialObsWrapper(inner, tile_size=tile_size)
    else:
        inner = FullyObsWrapper(inner)
        inner = RGBImgObsWrapper(inner, tile_size=tile_size)

    class StatsWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            import gymnasium.spaces as spaces
            h, w, c = env.observation_space['image'].shape
            self.observation_space = spaces.Box(0, 255, (h, w, c), dtype=np.uint8)
            if no_oracle:
                self.n_actions    = env.action_space.n
                self.QUERY_ACTION = None
            else:
                self.n_actions    = env.action_space.n + 1
                self.QUERY_ACTION = env.action_space.n
            self.action_space  = spaces.Discrete(self.n_actions)
            self._env_type     = env_type

        def _get_obs(self, obs_dict):
            return obs_dict['image']

        def reset(self, **kwargs):
            obs_dict, info = self.env.reset(**kwargs)
            return self._get_obs(obs_dict), info

        def step(self, action):
            guided = False
            if self.QUERY_ACTION is not None and action == self.QUERY_ACTION:
                action = get_oracle_action(self.env.unwrapped, self._env_type)
                guided = True
            obs_dict, reward, terminated, truncated, info = self.env.step(action)
            info['guided'] = guided
            return self._get_obs(obs_dict), reward, terminated, truncated, info

    return StatsWrapper(inner)


def run_episodes(env, model, device, n_episodes, seed0, stochastic, env_type):
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep)

        # BFS optimal path length from start state (before any action)
        try:
            opt_steps = get_optimal_steps(env.env.unwrapped, env_type)
        except Exception:
            opt_steps = -1

        total_ret, steps, queries, success = 0.0, 0, 0, False

        while True:
            obs_t = torch.tensor(obs[None], dtype=torch.uint8).to(device)
            with torch.no_grad():
                logits, _ = model(obs_t)
                if stochastic:
                    action = torch.distributions.Categorical(logits=logits).sample().item()
                else:
                    action = logits.argmax(dim=-1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            total_ret += reward
            steps     += 1
            if info.get('guided', False):
                queries += 1

            if terminated or truncated:
                # total_ret > 0 ensures the correct object was picked up
                # (FetchEnv gives reward=0 if wrong object picked up)
                success = bool(terminated) and total_ret > 0
                break

        oracle_pct = (queries / steps * 100) if steps > 0 else 0.0
        efficiency = (opt_steps / steps) if (success and steps > 0 and opt_steps > 0) else float('nan')

        results.append({
            'episode':    ep,
            'success':    success,
            'return':     total_ret,
            'steps':      steps,
            'queries':    queries,
            'oracle_pct': oracle_pct,
            'opt_steps':  opt_steps,
            'efficiency': efficiency,
        })
        status = 'SUCCESS' if success else 'FAIL'
        eff_str = f'{efficiency:.2f}' if not np.isnan(efficiency) else ' N/A'
        print(f"  ep={ep+1:3d}  {status}  ret={total_ret:.3f}  "
              f"steps={steps}  opt={opt_steps}  eff={eff_str}  "
              f"queries={queries}  oracle%={oracle_pct:.1f}%")

    return pd.DataFrame(results)


def make_comparison_plot(csv_path, out):
    """Plot all runs stored in a comparison CSV."""
    df   = pd.read_csv(csv_path)
    runs = df['label'].unique()
    n    = len(runs)

    # 4 panels — each at least 5" wide
    fig, axes_2d = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes_2d.flatten()
    fig.patch.set_facecolor('#FAFAFA')
    for ax in axes:
        ax.set_facecolor('#F5F5F5')

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    labels      = []
    sr_vals     = []
    ret_means   = []
    ret_stds    = []
    qpct_means  = []
    qpct_stds   = []
    eff_means   = []
    eff_stds    = []

    for i, label in enumerate(runs):
        sub = df[df['label'] == label]
        labels.append(label)
        sr_vals.append(sub['success'].mean() * 100)
        ret_means.append(sub['return'].mean())
        ret_stds.append(sub['return'].std())
        qpct_means.append(sub['oracle_pct'].mean())
        qpct_stds.append(sub['oracle_pct'].std())
        eff_sub = sub['efficiency'].dropna()
        eff_means.append(eff_sub.mean() if len(eff_sub) > 0 else float('nan'))
        eff_stds.append(eff_sub.std()   if len(eff_sub) > 0 else 0.0)

    x = np.arange(n)
    w = 0.5

    # Oracle-only indices (exclude Baseline PPO)
    oracle_idx = [i for i, l in enumerate(labels) if 'baseline' not in l.lower()]
    x_or = np.arange(len(oracle_idx))
    labels_or  = [labels[i]     for i in oracle_idx]
    colors_or  = [colors[i]     for i in oracle_idx]
    qpct_m_or  = [qpct_means[i] for i in oracle_idx]
    qpct_s_or  = [qpct_stds[i]  for i in oracle_idx]
    eff_m_or   = [eff_means[i]  for i in oracle_idx]
    eff_s_or   = [eff_stds[i]   for i in oracle_idx]

    # ── Success rate ──────────────────────────────────────────────────────────
    bars = axes[0].bar(x, sr_vals, width=w, color=colors[:n], edgecolor='white')
    axes[0].set_title('Success Rate (%)')
    axes[0].set_ylabel('%')
    axes[0].set_ylim(0, 115)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha='right', fontsize=11)
    for bar, v in zip(bars, sr_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 1, f'{v:.0f}%',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    # ── Mean return ───────────────────────────────────────────────────────────
    axes[1].bar(x, ret_means, width=w, yerr=ret_stds, color=colors[:n],
                edgecolor='white', capsize=5)
    axes[1].set_title('Mean Return ± std')
    axes[1].set_ylabel('Return')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha='right', fontsize=11)

    # ── Oracle query % (oracle conditions only) ───────────────────────────────
    axes[2].bar(x_or, qpct_m_or, width=w, yerr=qpct_s_or, color=colors_or,
                edgecolor='white', capsize=5)
    axes[2].set_title('Oracle Queries (% of steps)\n— oracle conditions only —')
    axes[2].set_ylabel('%')
    ymax = max(qpct_m_or) * 1.3 + 1 if max(qpct_m_or) > 0 else 5
    axes[2].set_ylim(0, ymax)
    axes[2].set_xticks(x_or)
    axes[2].set_xticklabels(labels_or, rotation=15, ha='right', fontsize=11)

    # ── Trajectory efficiency (oracle conditions only) ────────────────────────
    eff_plot_or = [v if not np.isnan(v) else 0.0 for v in eff_m_or]
    bars4 = axes[3].bar(x_or, eff_plot_or, width=w, color=colors_or,
                        edgecolor='white')
    axes[3].set_title('Trajectory Efficiency\n— oracle conditions only —')
    axes[3].set_ylabel('Efficiency (opt / agent steps)')
    eff_ymax = max(1.2, max(eff_plot_or) * 1.15 + 0.05) if eff_plot_or else 1.2
    axes[3].set_ylim(0, eff_ymax)
    axes[3].axhline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    axes[3].set_xticks(x_or)
    axes[3].set_xticklabels(labels_or, rotation=15, ha='right', fontsize=11)
    for bar, v in zip(bars4, eff_m_or):
        if not np.isnan(v) and v > 0:
            axes[3].text(bar.get_x() + bar.get_width()/2,
                         min(v, eff_ymax - 0.05) + 0.02,
                         f'{v:.2f}', ha='center', va='bottom',
                         fontsize=11, fontweight='bold')

    fig.suptitle(
        f'Zero-shot Transfer — {n} conditions, 100 episodes each',
        fontsize=13, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else '.', exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {out}')
    plt.close(fig)


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = make_env(args.env_id, args.env_type, args.tile_size,
                   args.no_oracle, args.partial_obs)

    if args.partial_obs:
        CNNPolicy = CNNPolicyPartial
    elif args.large_model:
        CNNPolicy = CNNPolicyLarge
    else:
        CNNPolicy = CNNPolicySmall

    model = CNNPolicy(env.observation_space.shape, env.n_actions, args.hidden_dim).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd    = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()

    label = args.label or os.path.basename(args.checkpoint).replace('.pt', '')
    print(f"\nRunning {args.n_episodes} episodes — {label}")
    print(f"  stochastic={args.stochastic}, no_oracle={args.no_oracle}")

    df          = run_episodes(env, model, device, args.n_episodes, args.seed,
                               args.stochastic, args.env_type)
    df['label'] = label
    env.close()

    eff_ok = df['efficiency'].dropna()
    print(f"\n── Summary : {label} ─────────────────────────────────")
    print(f"  Success rate : {df['success'].mean()*100:.1f}%")
    print(f"  Mean return  : {df['return'].mean():.4f} ± {df['return'].std():.4f}")
    print(f"  Mean steps   : {df['steps'].mean():.1f}")
    print(f"  Opt steps    : {df['opt_steps'].mean():.1f}")
    print(f"  Efficiency   : {eff_ok.mean():.3f} ± {eff_ok.std():.3f}"
          f"  (n={len(eff_ok)} successful)")
    print(f"  Oracle %     : {df['oracle_pct'].mean():.1f}% ± {df['oracle_pct'].std():.1f}%")

    if args.csv_out:
        header = not os.path.exists(args.csv_out)
        df.to_csv(args.csv_out, mode='a', header=header, index=False)
        print(f'Appended → {args.csv_out}')

    if args.csv_out and os.path.exists(args.csv_out):
        all_df = pd.read_csv(args.csv_out)
        if all_df['label'].nunique() > 1:
            make_comparison_plot(args.csv_out, args.out)


if __name__ == '__main__':
    main()
