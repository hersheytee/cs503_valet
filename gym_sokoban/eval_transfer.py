"""
Zero-shot transfer evaluation: run a trained Sokoban oracle policy on a
different environment variant (default: FixedTarget-Sokoban-v2, 7x7 2-box)
without any fine-tuning, and report success rate and oracle usage.

Usage:
    # Auto-discover the most recent linear-cost checkpoint:
    python gym_sokoban/eval_transfer.py

    # Explicit checkpoint:
    python gym_sokoban/eval_transfer.py --checkpoint runs/.../checkpoints/final.pt

    # Test without oracle (pure policy transfer):
    python gym_sokoban/eval_transfer.py --no-oracle

    # More episodes for tighter estimates:
    python gym_sokoban/eval_transfer.py --n-episodes 500
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from model import CNNPolicy
from env_wrapper import SokobanOracleWrapper

# numpy 2.0 compat
import numpy as _np
if not hasattr(_np, 'bool8'):
    _np.bool8 = _np.bool_


def read_yaml(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def find_linear_checkpoint(run_root: Path) -> Path | None:
    """Return the most recently modified final.pt from a linear-cost run."""
    candidates = []
    for ckpt in run_root.glob('*/checkpoints/final.pt'):
        config_path = ckpt.parent.parent / 'config.yaml'
        if not config_path.exists():
            continue
        try:
            cfg = read_yaml(config_path)
            if cfg.get('args', {}).get('oracle_cost_final') is not None:
                candidates.append((ckpt.stat().st_mtime, ckpt))
        except Exception:
            pass
    if not candidates:
        return None
    return max(candidates)[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Path to model checkpoint (.pt). Auto-discovers linear-cost run if omitted.')
    p.add_argument('--run-root', type=str, default=str(SCRIPT_DIR.parent / 'runs'),
                   help='Root directory to search for checkpoints.')
    p.add_argument('--env-id', type=str, default='FixedTarget-Sokoban-v2',
                   help='Target environment for zero-shot transfer.')
    p.add_argument('--n-episodes', type=int, default=200)
    p.add_argument('--max-episode-steps', type=int, default=50)
    p.add_argument('--obs-size', type=int, default=56)
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--oracle-cost', type=float, default=0.0,
                   help='Oracle query cost during eval (0 = free).')
    p.add_argument('--oracle-accuracy', type=float, default=1.0)
    p.add_argument('--no-oracle', action='store_true',
                   help='Remove oracle action (9-action policy). Use when checkpoint was trained without oracle.')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Checkpoint discovery ──────────────────────────────────────────────────
    if args.checkpoint is None:
        ckpt_path = find_linear_checkpoint(Path(args.run_root))
        if ckpt_path is None:
            raise SystemExit(
                'No linear-cost checkpoint found. Pass --checkpoint explicitly or check --run-root.'
            )
        print(f'Auto-selected checkpoint: {ckpt_path}')
    else:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise SystemExit(f'Checkpoint not found: {ckpt_path}')

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_actions = 9 if args.no_oracle else 10
    obs_shape = (args.obs_size, args.obs_size, 3)
    model = CNNPolicy(obs_shape=obs_shape, n_actions=n_actions, hidden_dim=args.hidden_dim).to(device)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device, weights_only=True))
    model.eval()
    print(f'Loaded: {ckpt_path}')
    print(f'Device: {device} | Actions: {n_actions} | Env: {args.env_id}')

    # ── Environment ───────────────────────────────────────────────────────────
    env = SokobanOracleWrapper(
        args.env_id,
        oracle_cost=args.oracle_cost,
        oracle_accuracy=args.oracle_accuracy,
        no_oracle=args.no_oracle,
        max_episode_steps=args.max_episode_steps,
        obs_size=args.obs_size,
        seed=args.seed,
    )

    # ── Evaluation loop ───────────────────────────────────────────────────────
    successes = []
    guided_pcts = []
    queries_per_ep = []

    for ep in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        total_steps = 0
        guided_steps = 0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.uint8).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(obs_t)
                action = logits.argmax(dim=-1).item()

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_steps += 1
            if info.get('guided', False):
                guided_steps += 1

        successes.append(float(info.get('success', False)))
        pct = 100.0 * guided_steps / total_steps if total_steps > 0 else 0.0
        guided_pcts.append(pct)
        queries_per_ep.append(guided_steps)

        if (ep + 1) % 50 == 0:
            print(
                f'  [{ep+1:4d}/{args.n_episodes}] '
                f'success={np.mean(successes[-50:]):.3f}  '
                f'oracle_usage={np.mean(guided_pcts[-50:]):.1f}%  '
                f'queries/ep={np.mean(queries_per_ep[-50:]):.1f}'
            )

    env.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print('=' * 52)
    print(f'Zero-shot transfer: {Path(ckpt_path).parent.parent.name}')
    print(f'  -> {args.env_id}  ({args.n_episodes} episodes)')
    print('=' * 52)
    print(f'  Success rate:      {np.mean(successes):.3f}  (±{np.std(successes):.3f})')
    print(f'  Oracle usage:      {np.mean(guided_pcts):.1f}%')
    print(f'  Queries / episode: {np.mean(queries_per_ep):.1f}  (±{np.std(queries_per_ep):.1f})')
    print('=' * 52)


if __name__ == '__main__':
    main()
