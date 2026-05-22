"""
Evaluate a trained vanilla PPO baseline on Sokoban.
Loads a checkpoint, runs episodes, and prints the action taken at each step
along with saving rendered frames as a GIF.

Usage:
    python eval_baseline.py --checkpoint checkpoints/baseline__Sokoban-small-v0__seed1__1778149804.pt
"""

import argparse
import os
import numpy as np
import torch
import cv2

import gym as old_gym
import gym_sokoban

from model import CNNPolicy
from env_wrapper import SokobanOracleWrapper

# Monkeypatch for numpy 2.0
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

ACTION_NAMES = {
    0: "No-op",
    1: "Push Up",
    2: "Push Down",
    3: "Push Left",
    4: "Push Right",
    5: "Move Up",
    6: "Move Down",
    7: "Move Left",
    8: "Move Right",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str,
                   default='checkpoints/baseline__Sokoban-small-v0__seed1__1778149804.pt')
    p.add_argument('--env-id', type=str, default='Sokoban-small-v0')
    p.add_argument('--n-episodes', type=int, default=5)
    p.add_argument('--max-steps', type=int, default=120)
    p.add_argument('--obs-size', type=int, default=128)
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--oracle-enabled', action='store_true',
                   help='Use 10-action model/env with query_oracle action available')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-gif', action='store_true', default=True,
                   help='Save episode frames as GIF')
    p.add_argument('--out-dir', type=str, default='eval_output')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_actions = 10 if args.oracle_enabled else 9
    obs_shape = (args.obs_size, args.obs_size, 3)
    model = CNNPolicy(obs_shape=obs_shape, n_actions=n_actions, hidden_dim=args.hidden_dim).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Device: {device}, Actions: {n_actions}")

    # Create a single environment (no oracle)
    env = SokobanOracleWrapper(
        args.env_id,
        oracle_cost=0.0,
        reward_shaping=False,
        no_oracle=not args.oracle_enabled,
        max_episode_steps=args.max_steps,
        obs_size=args.obs_size,
    )

    for ep in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        frames = []
        total_reward = 0.0
        done = False
        step_count = 0

        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}")
        print(f"{'='*60}")
        print(f"{'Step':<6} {'Action':<12} {'Reward':<8} {'Cumulative':<12}")
        print(f"{'-'*60}")

        # Save initial frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        while not done and step_count < args.max_steps:
            obs_tensor = torch.tensor(obs, dtype=torch.uint8).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, value = model(obs_tensor)
                # Greedy action (deterministic eval)
                action = logits.argmax(dim=-1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

            print(f"{step_count:<6} {ACTION_NAMES.get(action, '?'):<12} {reward:<8.3f} {total_reward:<12.3f}")

            frame = env.render()
            if frame is not None:
                frames.append(frame)

        success = bool(info.get('success', False))
        print(f"{'-'*60}")
        print(f"Result: {'SUCCESS' if success else 'FAILED'} | "
              f"Steps: {step_count} | Total Reward: {total_reward:.3f}")

        # Save GIF
        if args.save_gif and frames:
            gif_path = os.path.join(args.out_dir, f"episode_{ep+1}.gif")
            try:
                from PIL import Image
                pil_frames = [Image.fromarray(f) for f in frames]
                pil_frames[0].save(
                    gif_path, save_all=True, append_images=pil_frames[1:],
                    duration=200, loop=0
                )
                print(f"Saved GIF: {gif_path}")
            except ImportError:
                print("(PIL not available, skipping GIF save)")

    print(f"\nDone. Evaluated {args.n_episodes} episodes.")


if __name__ == '__main__':
    main()
