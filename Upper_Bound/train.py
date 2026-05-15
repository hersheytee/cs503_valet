"""
PPO + Behavior Cloning training — Oracle-guided MiniGrid agent.
Style: CleanRL single-file.

Metrics logged per episode to CSV:
    episode, global_step, ep_return, success,
    guided_pct, queries_per_ep, bc_loss,
    agreement_rate, cum_queries, first_unguided_success

Usage:
    python train.py --env-id MiniGrid-Empty-5x5-v0   --env-type empty
    python train.py --env-id MiniGrid-DoorKey-5x5-v0 --env-type doorkey
"""

import argparse
import csv
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from env_wrapper import make_env
from model import CNNPolicy as CNNPolicySmall
from model_large import CNNPolicy as CNNPolicyLarge


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env-id',      type=str,   default='MiniGrid-Empty-5x5-v0')
    p.add_argument('--env-type',    type=str,   default='empty',
                   choices=['empty', 'doorkey'])
    p.add_argument('--oracle-cost', type=float, default=0.0)
    p.add_argument('--total-timesteps', type=int,   default=500_000)
    p.add_argument('--n-envs',          type=int,   default=8)
    p.add_argument('--n-steps',         type=int,   default=128)
    p.add_argument('--n-minibatches',   type=int,   default=4)
    p.add_argument('--n-epochs',        type=int,   default=4)
    p.add_argument('--gamma',           type=float, default=0.99)
    p.add_argument('--gae-lambda',      type=float, default=0.95)
    p.add_argument('--clip-coef',       type=float, default=0.2)
    p.add_argument('--ent-coef',        type=float, default=0.01)
    p.add_argument('--vf-coef',         type=float, default=0.5)
    p.add_argument('--max-grad-norm',   type=float, default=0.5)
    p.add_argument('--lr',              type=float, default=2.5e-4)
    p.add_argument('--anneal-lr',       action='store_true', default=True)
    p.add_argument('--bc-coef',         type=float, default=0.0)
    p.add_argument('--bc-anneal',       action='store_true', default=False)
    p.add_argument('--no-oracle',        action='store_true', default=False)
    p.add_argument('--warmup-steps',     type=int,   default=0)
    p.add_argument('--reward-shaping',   action='store_true', default=False)
    # oracle_cost=0.0  → upper bound (oracle gratuit)
    # oracle_cost>0.0  → VLM réel (agent apprend quand consulter)
    # --no-oracle      → baseline PPO pur (pas d'action query)
    # --warmup-steps N → force oracle pendant les N premiers steps
    p.add_argument('--seed',            type=int,   default=1)
    p.add_argument('--hidden-dim',      type=int,   default=256)
    p.add_argument('--tile-size',       type=int,   default=8)
    p.add_argument('--save-model',      action='store_true', default=False)
    p.add_argument('--exp-name',        type=str,   default='oracle_ppo')
    p.add_argument('--large-model',     action='store_true', default=False,
                   help='Use strided CNN for large obs (e.g. 16x16 grid)')
    p.add_argument('--vlm-model',       type=str,   default='',
                   help='VLM key to use as oracle (e.g. qwen3b). Empty = BFS oracle.')
    p.add_argument('--cache-dir',       type=str,
                   default=os.environ.get('HF_HOME', './hf_cache'),
                   help='HuggingFace model cache directory')
    return p.parse_args()


# ── CSV Logger ───────────────────────────────────────────────────────────────

CSV_FIELDS = [
    'episode', 'global_step',
    'ep_return', 'success',
    'guided_pct', 'queries_per_ep',
    'bc_loss', 'agreement_rate',
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    batch_size     = args.n_envs * args.n_steps
    minibatch_size = batch_size // args.n_minibatches
    n_updates      = args.total_timesteps // batch_size

    run_name = f"{args.exp_name}__{args.env_id}__seed{args.seed}__{int(time.time())}"
    print(f"Run: {run_name}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  device={device}, batch={batch_size}, updates={n_updates}")

    logger = CSVLogger(f"logs/{run_name}.csv")

    # ── VLM server (optional) ─────────────────────────────────────────────────
    vlm_proc         = None
    vlm_client       = None
    vlm_served_name  = ''

    if args.vlm_model and not args.no_oracle:
        from vlm_oracle import (start_vlm_server, wait_for_server,
                                make_vlm_client, get_served_model_name)
        print(f"  VLM oracle : {args.vlm_model}  (cache: {args.cache_dir})")
        vlm_proc = start_vlm_server(args.vlm_model, args.cache_dir)
        if not wait_for_server(timeout=900):
            vlm_proc.kill()
            raise RuntimeError("vLLM server failed to start within 900s.")
        vlm_client      = make_vlm_client()
        vlm_served_name = get_served_model_name(vlm_client)
        print(f"  vLLM ready : {vlm_served_name}")
    else:
        print("  Oracle     : BFS")

    # ── Environments ─────────────────────────────────────────────────────────
    envs = gym.vector.SyncVectorEnv([
        make_env(args.env_id, args.env_type, args.tile_size,
                 args.oracle_cost, seed=args.seed + i,
                 no_oracle=args.no_oracle,
                 reward_shaping=args.reward_shaping,
                 vlm_client=vlm_client,
                 vlm_model_key=args.vlm_model or 'qwen3b',
                 vlm_served_name=vlm_served_name)
        for i in range(args.n_envs)
    ])

    obs_shape    = envs.single_observation_space.shape
    n_actions    = envs.single_action_space.n
    QUERY_ACTION = n_actions - 1
    print(f"  obs={obs_shape}, n_actions={n_actions}, query_action={QUERY_ACTION}")

    # ── Model ─────────────────────────────────────────────────────────────────
    CNNPolicy = CNNPolicyLarge if args.large_model else CNNPolicySmall
    model     = CNNPolicy(obs_shape, n_actions, args.hidden_dim).to(device)
    optimiser = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    print(f"  params={sum(p.numel() for p in model.parameters()):,}")

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

    # Global counters
    episode_count          = 0
    cum_queries            = 0
    first_unguided_success = None
    last_bc_loss           = 0.0

    # Rolling windows for console
    ep_returns    = deque(maxlen=100)
    ep_guided_pct = deque(maxlen=100)

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

        bc_coef = args.bc_coef * frac if args.bc_anneal else args.bc_coef

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
            # SyncVectorEnv stores terminal-step info in infos['final_info'][i]
            guided_np     = np.zeros(args.n_envs, dtype=bool)
            oracle_act_np = np.zeros(args.n_envs, dtype=np.int64)

            for i in range(args.n_envs):
                # For terminated/truncated envs the step info is in final_info
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

            guided_buf[step]         = torch.tensor(guided_np).to(device)
            oracle_actions_buf[step] = torch.tensor(oracle_act_np).to(device)
            cum_queries             += int(guided_np.sum())

            # Accumulate per-env stats ────────────────────────────────────────
            ep_ret       += reward_np
            ep_len       += 1
            ep_n_queries += guided_np.astype(int)

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
                success = float(term_np[i])  # terminated = reached goal, not timeout
                n_q     = int(ep_n_queries[i])
                pct     = n_q / max(int(ep_len[i]), 1) * 100
                agree   = (ep_agree[i] / ep_n_queries[i]
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
                    'bc_loss':               round(last_bc_loss, 5),
                    'agreement_rate':        round(agree, 4) if not np.isnan(agree) else '',
                    'cum_queries':           cum_queries,
                    'first_unguided_success': fug,
                })

                ep_returns.append(ret)
                ep_guided_pct.append(pct)

                # Reset accumulators
                ep_ret[i] = ep_len[i] = ep_n_queries[i] = ep_agree[i] = 0

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
        b_oracle_acts = oracle_actions_buf.reshape(-1)

        for _ in range(args.n_epochs):
            for mb in np.array_split(np.random.permutation(batch_size), args.n_minibatches):

                _, new_lp, entropy, new_val = model.get_action_and_value(b_obs[mb], b_actions[mb])

                ratio  = (new_lp - b_logprobs[mb]).exp()
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

                # ── BC loss — only on guided steps ────────────────────────────
                if mb_g.any() and bc_coef > 0:
                    _, bc_lp, _, _ = model.get_action_and_value(
                        b_obs[mb][mb_g], b_oracle_acts[mb][mb_g])
                    bc_loss = -bc_lp.mean()
                else:
                    bc_loss = torch.tensor(0.0, device=device)

                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss + bc_coef * bc_loss

                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimiser.step()

            last_bc_loss = bc_loss.item()

        # ── Console log ───────────────────────────────────────────────────────
        if update % 10 == 0 or update == 1:
            sps = int(global_step / (time.time() - t0))
            print(
                f"[{update:4d}/{n_updates}] step={global_step:7d} | "
                f"ep={episode_count:5d} | "
                f"return={np.mean(ep_returns) if ep_returns else float('nan'):6.3f} | "
                f"guided%={np.mean(ep_guided_pct) if ep_guided_pct else float('nan'):5.1f} | "
                f"bc={last_bc_loss:.4f} | cumQ={cum_queries:6d} | "
                f"bc_coef={bc_coef:.3f} | sps={sps}"
            )

    # ── Save & plot ───────────────────────────────────────────────────────────
    if args.save_model:
        os.makedirs('checkpoints', exist_ok=True)
        ckpt = f'checkpoints/{run_name}.pt'
        torch.save(model.state_dict(), ckpt)
        print(f"Saved → {ckpt}")

    logger.close()
    envs.close()

    if vlm_proc is not None:
        from vlm_oracle import stop_vlm_server
        stop_vlm_server(vlm_proc)

    log_path = f"logs/{run_name}.csv"
    fig_path = f"figures/{run_name}.png"
    print("Generating plots...")
    os.system(f"python plot.py --csv {log_path} --out {fig_path} --env {args.env_id}")
    print(f"Plots → {fig_path}")


if __name__ == '__main__':
    main()