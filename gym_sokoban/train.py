"""
PPO training for Sokoban with an optional oracle-query action.
Adapted for WorldCoder-style full-sprite observations (56x56 RGB).
Style: CleanRL single-file.

Metrics logged per episode to CSV:
    episode, global_step, ep_return, success,
    guided_pct, queries_per_ep,
    agreement_rate, cum_queries, first_unguided_success
"""

import argparse
import csv
import os
import random
import re
import shlex
import subprocess
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import yaml

# Assumes env_wrapper.py and model.py are in the same directory
from env_wrapper import make_env
from model import CNNPolicy

# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    
    # --- Environment Settings ---
    p.add_argument('--env-id',          type=str,   default='Sokoban-small-v0')    # The exact name of the Gym environment to load
    p.add_argument('--env-type',        type=str,   default='empty',               # Legacy MiniGrid argument (kept so your old bash scripts don't crash)
                   choices=['empty', 'doorkey']) 
    
    # --- Oracle & Shaping Settings ---
    p.add_argument('--oracle-cost',     type=float, default=0.0)                   # Negative reward penalty applied every time the agent asks the Oracle for a move
    p.add_argument('--oracle-accuracy', type=float, default=1.0)                   # Probability that a queried oracle returns the BFS-optimal action; otherwise returns a random native action
    p.add_argument('--no-oracle',       action='store_true', default=False)        # If True, removes the Oracle action entirely (runs as a pure PPO baseline)
    p.add_argument('--reward-shaping',  action='store_true', default=False)        # Enables potential-based dense rewards
    p.add_argument('--warmup-steps',    type=int,   default=0)                     # Forces the agent to ONLY use the Oracle for the first N steps to build a good starting buffer
    p.add_argument('--max-episode-steps', type=int, default=50)                    # WorldCoder-style short Sokoban horizon
    p.add_argument('--obs-size',        type=int,   default=56)                    # 7x7 board rendered at 8 pixels per cell
    
    # --- Training Loop Dimensions ---
    p.add_argument('--total-timesteps', type=int,   default=500_000)               # Total number of environment frames the agent will experience during the entire run
    p.add_argument('--n-envs',          type=int,   default=8)                     # Number of parallel environments running at the same time (speeds up data collection)
    p.add_argument('--n-steps',         type=int,   default=256)                   # Number of steps each parallel environment takes before pausing to update the neural network
    p.add_argument('--n-minibatches',   type=int,   default=4)                     # How many chunks the collected data (n_envs * n_steps) is split into for gradient descent
    p.add_argument('--n-epochs',        type=int,   default=10)                    # How many times the network iterates over the collected batch of data per update phase
    
    # --- Standard PPO Hyperparameters ---
    p.add_argument('--gamma',           type=float, default=0.99)                  # Discount factor: How much the agent cares about future rewards vs immediate rewards (0.99 is standard)
    p.add_argument('--gae-lambda',      type=float, default=0.95)                  # Smoothing parameter for Advantage estimation (balances bias vs variance in reward predictions)
    p.add_argument('--clip-coef',       type=float, default=0.2)                   # PPO's core feature: prevents the policy from changing more than 20% in a single update step
    p.add_argument('--ent-coef',        type=float, default=0.0)                   # Entropy bonus; WorldCoder-style PPO baseline uses no entropy bonus
    p.add_argument('--vf-coef',         type=float, default=0.5)                   # Value function coefficient: scales how much the Value head's errors impact the overall network loss
    p.add_argument('--max-grad-norm',   type=float, default=0.5)                   # Gradient clipping threshold: prevents "exploding gradients" from destroying the network weights
    
    # --- Learning Rate ---
    p.add_argument('--lr',              type=float, default=3e-4)                  # The starting step size for the Adam optimizer
    p.add_argument('--anneal-lr',       action='store_true', default=True)         # If True, linearly decreases the learning rate to 0 by the end of training
    
    # --- Architecture & Reproducibility ---
    p.add_argument('--seed',            type=int,   default=1)                     # Random seed to ensure you get the exact same results if you run the script twice
    p.add_argument('--hidden-dim',      type=int,   default=64)                    # Size of the FC layer after the WorldCoder-style CNN
    p.add_argument('--tile-size',       type=int,   default=8)                     # Legacy MiniGrid argument (kept for bash script compatibility)
    
    # --- Output/Saving ---
    p.add_argument('--save-model',      action='store_true', default=False)        # If True, saves the final PyTorch weights (.pt file) in the /checkpoints folder
    p.add_argument('--exp-name',        type=str,   default='worldcoder_ppo')      # The prefix name used for saving CSV logs, PNG plots, and Model checkpoints

    # --- Optional W&B Tracking ---
    p.add_argument('--track',            action=argparse.BooleanOptionalAction, default=True,
                   help='Enable W&B logging by default; use --no-track to disable.')
    p.add_argument('--wandb-project',    type=str, default='cs503-sokoban')
    p.add_argument('--wandb-entity',     type=str, default='')
    p.add_argument('--wandb-group',      type=str, default='')
    p.add_argument('--wandb-tags',       type=str, default='')
    p.add_argument('--wandb-mode',       type=str, default='online',
                   choices=['online', 'offline', 'disabled'])
    p.add_argument('--upload-checkpoint', action='store_true', default=False)
    return p.parse_args()


# ── CSV Logger ───────────────────────────────────────────────────────────────

CSV_FIELDS = [
    'episode', 'global_step',
    'ep_return', 'success',
    'guided_pct', 'queries_per_ep',
    'agreement_rate', 'oracle_correct_rate',
    'cum_queries', 'first_unguided_success',
]

class CSVLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        self._f      = open(path, 'w', newline='')
        self._writer = csv.DictWriter(self._f, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        print(f"  CSV → {path}")

    def write(self, row: dict):
        self._writer.writerow({k: row.get(k, '') for k in CSV_FIELDS})
        self._f.flush()

    def close(self):
        self._f.close()


class TeeStream:
    """Write console output to both the terminal and the run log."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, 'isatty', lambda: False)() for stream in self.streams)


def _safe_name(text: str) -> str:
    text = re.sub(r'[^A-Za-z0-9_.-]+', '-', text.strip())
    return text.strip('-') or 'run'


def write_yaml(path: str, data: dict):
    with open(path, 'w') as f:
        yaml.safe_dump(to_plain_config(data), f, sort_keys=False)


def to_plain_config(value):
    if isinstance(value, dict):
        return {str(k): to_plain_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_config(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def get_git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ''


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if not 0.0 <= args.oracle_accuracy <= 1.0:
        raise ValueError(f"oracle_accuracy must be in [0, 1], got {args.oracle_accuracy}")

    batch_size     = args.n_envs * args.n_steps
    minibatch_size = batch_size // args.n_minibatches
    n_updates      = args.total_timesteps // batch_size
    if n_updates < 1:
        raise ValueError(
            f"total_timesteps={args.total_timesteps} is smaller than one rollout "
            f"batch={batch_size}. Increase timesteps or reduce n-envs/n-steps."
        )

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_desc = _safe_name(f"{args.exp_name}__{args.env_id}__seed{args.seed}")
    run_name = f"{timestamp}__{run_desc}"
    run_dir = os.path.join('runs', run_name)
    logs_dir = os.path.join(run_dir, 'logs')
    data_dir = os.path.join(run_dir, 'data')
    figures_dir = os.path.join(run_dir, 'figures')
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    for path in (logs_dir, data_dir, figures_dir, checkpoints_dir):
        os.makedirs(path, exist_ok=True)

    stdout_log = open(os.path.join(logs_dir, 'stdout.log'), 'a', buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, stdout_log)
    sys.stderr = TeeStream(sys.__stderr__, stdout_log)

    print(f"Run: {run_name}")
    print(f"Run directory: {run_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  device={device}, batch={batch_size}, updates={n_updates}")

    metrics_path = os.path.join(data_dir, 'metrics.csv')
    logger = CSVLogger(metrics_path)

    # ── Environments ─────────────────────────────────────────────────────────
    envs = gym.vector.SyncVectorEnv([
        make_env(
            args.env_id, 
            oracle_cost=args.oracle_cost, 
            oracle_accuracy=args.oracle_accuracy,
            seed=args.seed * 1000 + i, 
            reward_shaping=args.reward_shaping, 
            no_oracle=args.no_oracle,
            max_episode_steps=args.max_episode_steps,
            obs_size=args.obs_size,
            )
        for i in range(args.n_envs)
    ])

    obs_shape    = envs.single_observation_space.shape
    n_actions    = envs.single_action_space.n
    
    # ---> SAFEGUARD THE QUERY ACTION INDEX <---
    QUERY_ACTION = n_actions - 1 if not args.no_oracle else None
    print(f"  obs={obs_shape}, n_actions={n_actions}, query_action={QUERY_ACTION}")


    # ── Model ─────────────────────────────────────────────────────────────────
    model     = CNNPolicy(obs_shape=obs_shape, n_actions=n_actions, hidden_dim=args.hidden_dim).to(device)
    optimiser = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  params={param_count:,}")

    git_commit = get_git_commit()
    config_path = os.path.join(run_dir, 'config.yaml')
    config_payload = {
        'run': {
            'name': run_name,
            'timestamp': timestamp,
            'description': run_desc,
            'directory': run_dir,
            'command': ' '.join(shlex.quote(part) for part in [sys.executable] + sys.argv),
            'git_commit': git_commit,
        },
        'args': vars(args).copy(),
        'derived': {
            'batch_size': batch_size,
            'minibatch_size': minibatch_size,
            'n_updates': n_updates,
            'obs_shape': tuple(obs_shape),
            'n_actions': n_actions,
            'query_action': QUERY_ACTION,
            'parameter_count': param_count,
            'device': str(device),
        },
        'artifacts': {
            'metrics_csv': metrics_path,
            'stdout_log': os.path.join(logs_dir, 'stdout.log'),
            'figure': os.path.join(figures_dir, 'training_metrics.png'),
            'checkpoint': os.path.join(checkpoints_dir, 'final.pt'),
        },
    }
    write_yaml(config_path, config_payload)
    print(f"  config → {config_path}")

    # ── Optional W&B ──────────────────────────────────────────────────────────
    wandb_run = None
    if args.track and args.wandb_mode != 'disabled':
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B tracking was requested with --track, but wandb is not installed. "
                "Install it with: pip install wandb"
            ) from exc

        wandb_config = vars(args).copy()
        wandb_config.update({
            'run_dir': run_dir,
            'run_name': run_name,
            'batch_size': batch_size,
            'minibatch_size': minibatch_size,
            'n_updates': n_updates,
            'obs_shape': tuple(obs_shape),
            'n_actions': n_actions,
            'parameter_count': param_count,
            'device': str(device),
            'git_commit': git_commit,
        })
        tags = [t.strip() for t in args.wandb_tags.split(',') if t.strip()]

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            group=args.wandb_group or None,
            name=run_name,
            config=wandb_config,
            tags=tags,
            mode=args.wandb_mode,
            save_code=True,
        )
        wandb.define_metric('episode/count')
        wandb.define_metric('episode/*', step_metric='episode/count')
        wandb.define_metric('raw_episode/*', step_metric='episode/count')
        wandb.define_metric('charts/global_step')
        wandb.define_metric('charts/*', step_metric='charts/global_step')
        wandb.define_metric('losses/*', step_metric='charts/global_step')

    # ── Rollout buffers ───────────────────────────────────────────────────────
    obs_buf            = torch.zeros((args.n_steps, args.n_envs) + obs_shape,
                                     dtype=torch.uint8).to(device)
    actions_buf        = torch.zeros((args.n_steps, args.n_envs), dtype=torch.long).to(device)
    logprobs_buf       = torch.zeros((args.n_steps, args.n_envs)).to(device)
    rewards_buf        = torch.zeros((args.n_steps, args.n_envs)).to(device)
    dones_buf          = torch.zeros((args.n_steps, args.n_envs)).to(device)
    values_buf         = torch.zeros((args.n_steps, args.n_envs)).to(device)
    guided_buf         = torch.zeros((args.n_steps, args.n_envs), dtype=torch.bool).to(device)
    oracle_actions_buf = torch.zeros((args.n_steps, args.n_envs), dtype=torch.long).to(device)

    # ── Episode accumulators (one per parallel env) ───────────────────────────
    ep_ret       = np.zeros(args.n_envs)
    ep_len       = np.zeros(args.n_envs, dtype=int)
    ep_n_queries = np.zeros(args.n_envs, dtype=int)   # oracle calls this ep
    ep_agree     = np.zeros(args.n_envs, dtype=int)   # guided steps where greedy==oracle
    ep_oracle_correct = np.zeros(args.n_envs, dtype=int)

    # Global counters
    episode_count          = 0
    cum_queries            = 0
    first_unguided_success = None
    # Rolling windows for console
    ep_returns    = deque(maxlen=100)
    ep_successes  = deque(maxlen=50)
    ep_guided_pct = deque(maxlen=40)
    ep_queries    = deque(maxlen=40)
    ep_lengths    = deque(maxlen=100)
    ep_agreements = deque(maxlen=40)
    ep_oracle_correct_rates = deque(maxlen=40)

    # ── Initial reset ─────────────────────────────────────────────────────────
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs  = torch.tensor(next_obs_np, dtype=torch.uint8).to(device)
    next_done = torch.zeros(args.n_envs).to(device)

    global_step = 0
    t0          = time.time()

    # ── Training loop ─────────────────────────────────────────────────────────
    for update in range(1, n_updates + 1):
        frac = 1.0 - (update - 1) / n_updates

        if args.anneal_lr:
            optimiser.param_groups[0]['lr'] = frac * args.lr

        # ── Rollout collection ────────────────────────────────────────────────
        for step in range(args.n_steps):
            global_step += args.n_envs
            obs_buf[step]   = next_obs
            dones_buf[step] = next_done


            with torch.no_grad():
                logits, value = model(next_obs)
                dist = torch.distributions.Categorical(logits=logits)
                
                if args.warmup_steps > 0 and global_step < args.warmup_steps and not args.no_oracle:
                    action = torch.full((args.n_envs,), QUERY_ACTION, dtype=torch.long, device=device)
                else:
                    action = dist.sample()
                logprob = dist.log_prob(action)
            
                logits_masked = logits.clone()
                if not args.no_oracle:
                    logits_masked[:, QUERY_ACTION] = -float('inf')
                
                greedy = logits_masked.argmax(dim=-1)

            actions_buf[step]  = action
            logprobs_buf[step] = logprob
            values_buf[step]   = value

            next_obs_np, reward_np, term_np, trunc_np, infos = envs.step(action.cpu().numpy())

            rewards_buf[step] = torch.tensor(reward_np, dtype=torch.float32).to(device)
            next_done = torch.tensor((term_np | trunc_np), dtype=torch.float32).to(device)
            next_obs  = torch.tensor(next_obs_np, dtype=torch.uint8).to(device)

            # Parse guided flags ──────────────────────────────────────────────
            guided_np     = np.zeros(args.n_envs, dtype=bool)
            oracle_act_np = np.zeros(args.n_envs, dtype=np.int64)
            oracle_correct_np = np.zeros(args.n_envs, dtype=bool)
            success_np    = np.zeros(args.n_envs, dtype=bool)

            for i in range(args.n_envs):
                if (term_np[i] or trunc_np[i]) and 'final_info' in infos and infos['final_info'][i] is not None:
                    step_info = infos['final_info'][i]
                elif 'guided' in infos and isinstance(infos['guided'], np.ndarray):
                    step_info = {k: infos[k][i] for k in infos if isinstance(infos[k], np.ndarray)}
                elif isinstance(infos, list):
                    step_info = infos[i]
                else:
                    step_info = {}

                guided_np[i]     = step_info.get('guided', False)
                oa               = step_info.get('oracle_action', -1)
                oracle_act_np[i] = oa if oa >= 0 else 0
                oracle_correct_np[i] = bool(step_info.get('oracle_correct', False))
                success_np[i]    = bool(step_info.get('success', False))

            guided_buf[step]         = torch.tensor(guided_np).to(device)
            oracle_actions_buf[step] = torch.tensor(oracle_act_np).to(device)
            cum_queries             += int(guided_np.sum())

            # Accumulate per-env stats ────────────────────────────────────────
            ep_ret       += reward_np
            ep_len       += 1
            ep_n_queries += guided_np.astype(int)
            ep_oracle_correct += (guided_np & oracle_correct_np).astype(int)

            greedy_np = greedy.cpu().numpy()
            for i in range(args.n_envs):
                if guided_np[i] and greedy_np[i] == oracle_act_np[i]:
                    ep_agree[i] += 1

            # Episode end ─────────────────────────────────────────────────────
            for i in range(args.n_envs):
                if not (term_np[i] or trunc_np[i]):
                    continue

                episode_count += 1
                ret     = float(ep_ret[i])
                success = float(success_np[i])
                n_q     = int(ep_n_queries[i])
                pct     = n_q / max(int(ep_len[i]), 1) * 100
                agree   = (ep_agree[i] / ep_n_queries[i]
                           if ep_n_queries[i] > 0 else float('nan'))
                oracle_correct_rate = (ep_oracle_correct[i] / ep_n_queries[i]
                                       if ep_n_queries[i] > 0 else float('nan'))

                # First unguided success
                fug = ''
                if success and n_q == 0 and first_unguided_success is None:
                    first_unguided_success = episode_count
                    fug = episode_count
                    print(f"  *** First unguided success — episode {episode_count} ***")

                logger.write({
                    'episode':               episode_count,
                    'global_step':           global_step,
                    'ep_return':             round(ret, 4),
                    'success':               success,
                    'guided_pct':            round(pct, 2),
                    'queries_per_ep':        n_q,
                    'agreement_rate':        round(agree, 4) if not np.isnan(agree) else '',
                    'oracle_correct_rate':   round(oracle_correct_rate, 4) if not np.isnan(oracle_correct_rate) else '',
                    'cum_queries':           cum_queries,
                    'first_unguided_success': fug,
                })

                ep_returns.append(ret)
                ep_successes.append(success)
                ep_guided_pct.append(pct)
                ep_queries.append(n_q)
                ep_lengths.append(int(ep_len[i]))
                if not np.isnan(agree):
                    ep_agreements.append(agree)
                if not np.isnan(oracle_correct_rate):
                    ep_oracle_correct_rates.append(oracle_correct_rate)

                if wandb_run is not None:
                    wandb.log({
                        'episode/count': episode_count,
                        'episode/return': float(np.mean(ep_returns)),
                        'episode/success_rate': float(np.mean(ep_successes)),
                        'episode/length': float(np.mean(ep_lengths)),
                        'episode/guided_pct': float(np.mean(ep_guided_pct)),
                        'episode/queries_per_ep': float(np.mean(ep_queries)),
                        'episode/agreement_rate': (
                            float(np.mean(ep_agreements)) if ep_agreements else None
                        ),
                        'episode/oracle_correct_rate': (
                            float(np.mean(ep_oracle_correct_rates))
                            if ep_oracle_correct_rates else None
                        ),
                        'episode/cum_queries': cum_queries,
                        'episode/first_success_episode': first_unguided_success or 0,
                        'raw_episode/return': ret,
                        'raw_episode/success': success,
                        'raw_episode/length': int(ep_len[i]),
                        'raw_episode/guided_pct': pct,
                        'raw_episode/queries_per_ep': n_q,
                        'raw_episode/agreement_rate': agree if not np.isnan(agree) else None,
                        'raw_episode/oracle_correct_rate': (
                            oracle_correct_rate if not np.isnan(oracle_correct_rate) else None
                        ),
                    }, step=global_step)

                # Reset accumulators
                ep_ret[i] = ep_len[i] = ep_n_queries[i] = ep_agree[i] = ep_oracle_correct[i] = 0

        # ── GAE ───────────────────────────────────────────────────────────────
        with torch.no_grad():
            next_value = model.get_value(next_obs)
            advantages = torch.zeros_like(rewards_buf).to(device)
            last_gae   = 0

            for t in reversed(range(args.n_steps)):
                nxt_nterm = 1.0 - (next_done if t == args.n_steps - 1 else dones_buf[t + 1])
                nxt_val   = next_value if t == args.n_steps - 1 else values_buf[t + 1]
                delta      = rewards_buf[t] + args.gamma * nxt_val * nxt_nterm - values_buf[t]
                last_gae   = delta + args.gamma * args.gae_lambda * nxt_nterm * last_gae
                advantages[t] = last_gae

            returns = advantages + values_buf

        # ── PPO update ────────────────────────────────────────────────────────
        b_obs         = obs_buf.reshape((-1,) + obs_shape)
        b_actions     = actions_buf.reshape(-1)
        b_logprobs    = logprobs_buf.reshape(-1)
        b_advantages  = advantages.reshape(-1)
        b_returns     = returns.reshape(-1)
        b_values      = values_buf.reshape(-1)
        b_guided      = guided_buf.reshape(-1)

        approx_kl = torch.tensor(0.0, device=device)
        clipfracs = []
        last_pg_loss = torch.tensor(0.0, device=device)
        last_v_loss = torch.tensor(0.0, device=device)
        last_entropy = torch.tensor(0.0, device=device)
        for _ in range(args.n_epochs):
            for mb in np.array_split(np.random.permutation(batch_size), args.n_minibatches):

                _, new_lp, entropy, new_val = model.get_action_and_value(b_obs[mb], b_actions[mb])

                ratio  = (new_lp - b_logprobs[mb]).exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (new_lp - b_logprobs[mb])).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                mb_adv = b_advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
                ).mean()

                v_clip  = b_values[mb] + (new_val - b_values[mb]).clamp(-args.clip_coef, args.clip_coef)
                v_loss  = 0.5 * torch.max((new_val - b_returns[mb])**2,
                                           (v_clip  - b_returns[mb])**2).mean()

                mb_g    = b_guided[mb]
                ent_loss = entropy[~mb_g].mean() if (~mb_g).any() else torch.tensor(0.0, device=device)

                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss
                last_pg_loss = pg_loss.detach()
                last_v_loss = v_loss.detach()
                last_entropy = ent_loss.detach()

                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimiser.step()

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # ── Console log ───────────────────────────────────────────────────────
        if update % 10 == 0 or update == 1:
            sps = int(global_step / (time.time() - t0))
            current_lr = optimiser.param_groups[0]['lr']
            print(
                f"[{update:4d}/{n_updates}] step={global_step:7d} | "
                f"ep={episode_count:5d} | "
                f"return={np.mean(ep_returns) if ep_returns else float('nan'):6.3f} | "
                f"guided%={np.mean(ep_guided_pct) if ep_guided_pct else float('nan'):5.1f} | "
                f"cumQ={cum_queries:6d} | "
                f"kl={approx_kl.item():.4f} | "
                f"clip={np.mean(clipfracs):.3f} | ev={explained_var:.3f} | sps={sps}"
            )

        if wandb_run is not None:
            wandb.log({
                'charts/global_step': global_step,
                'charts/SPS': int(global_step / (time.time() - t0)),
                'charts/learning_rate': optimiser.param_groups[0]['lr'],
                'charts/episodes': episode_count,
                'charts/return_mean_100ep': float(np.mean(ep_returns)) if ep_returns else None,
                'charts/success_rate_50ep': float(np.mean(ep_successes)) if ep_successes else None,
                'charts/guided_pct_40ep': float(np.mean(ep_guided_pct)) if ep_guided_pct else None,
                'charts/queries_per_ep_40ep': float(np.mean(ep_queries)) if ep_queries else None,
                'charts/oracle_correct_rate_40ep': (
                    float(np.mean(ep_oracle_correct_rates)) if ep_oracle_correct_rates else None
                ),
                'losses/policy_loss': last_pg_loss.item(),
                'losses/value_loss': last_v_loss.item(),
                'losses/entropy': last_entropy.item(),
                'losses/approx_kl': approx_kl.item(),
                'losses/clipfrac': float(np.mean(clipfracs)) if clipfracs else 0.0,
                'losses/explained_variance': explained_var,
            }, step=global_step)

    # ── Save & plot ───────────────────────────────────────────────────────────
    if args.save_model:
        ckpt = os.path.join(checkpoints_dir, 'final.pt')
        torch.save(model.state_dict(), ckpt)
        print(f"Saved → {ckpt}")
        if wandb_run is not None and args.upload_checkpoint:
            artifact = wandb.Artifact(f'{run_name}-checkpoint', type='model')
            artifact.add_file(ckpt)
            wandb.log_artifact(artifact)

    logger.close()
    envs.close()

    log_path = metrics_path
    fig_path = os.path.join(figures_dir, 'training_metrics.png')
    print("Generating plots...")
    plot_script = os.path.join(os.getcwd(), "plot.py")
    if os.path.exists(plot_script):
        subprocess.run(
            [sys.executable, plot_script, "--csv", log_path, "--out", fig_path, "--env", args.env_id],
            check=False,
        )
    else:
        print("Skipping plot generation: plot.py not found in current working directory.")
    print(f"Plots → {fig_path}")

    if wandb_run is not None:
        try:
            import wandb
            if os.path.exists(fig_path):
                wandb.log({'figures/training_metrics': wandb.Image(fig_path)}, step=global_step)
                artifact = wandb.Artifact(f'{run_name}-figures', type='figure')
                artifact.add_file(fig_path)
                artifact.add_file(config_path)
                artifact.add_file(metrics_path)
                wandb.log_artifact(artifact)
        except Exception as exc:
            print(f"Skipping W&B figure upload: {exc}")
        wandb.finish()

if __name__ == '__main__':
    main()
