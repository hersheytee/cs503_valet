"""
eval.py — Run a trained agent and save a GIF.

Usage:
    python eval.py \
        --checkpoint "checkpoints/oracle_free__MiniGrid-DoorKey-8x8-v0__seed4__*.pt" \
        --env-id MiniGrid-DoorKey-8x8-v0 --env-type doorkey \
        --n-episodes 3 --out figures/eval.gif
"""

import argparse
import glob
import os
import numpy as np
import torch
import gymnasium as gym
import imageio
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper

from oracle_transfer import get_oracle_action as _get_oracle_action_transfer
from oracle import get_oracle_action as _get_oracle_action_doorkey

_TRANSFER_ENV_TYPES = {'fetch', 'gotodoor', 'gotoobject'}

def get_oracle_action(env_unwrapped, env_type):
    if env_type in _TRANSFER_ENV_TYPES:
        return _get_oracle_action_transfer(env_unwrapped, env_type)
    return _get_oracle_action_doorkey(env_unwrapped, env_type)

from model import CNNPolicy as CNNPolicySmall
from model_large import CNNPolicy as CNNPolicyLarge
from model_partial import CNNPolicy as CNNPolicyPartial


ACTION_NAMES = {0: 'Left', 1: 'Right', 2: 'Forward',
                3: 'Pickup', 4: 'Drop', 5: 'Toggle', 6: 'Done', 7: 'Oracle'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',     type=str, required=True)
    p.add_argument('--env-id',         type=str, default='MiniGrid-DoorKey-8x8-v0')
    p.add_argument('--env-type',       type=str, default='doorkey')
    p.add_argument('--no-oracle',      action='store_true', default=False)
    p.add_argument('--oracle-cost',    type=float, default=0.0)
    p.add_argument('--reward-shaping', action='store_true', default=False)
    p.add_argument('--n-episodes',     type=int, default=3)
    p.add_argument('--hidden-dim',     type=int, default=256)
    p.add_argument('--tile-size',      type=int, default=8)
    p.add_argument('--fps',            type=int, default=6)
    p.add_argument('--out',            type=str, default='figures/eval.gif')
    p.add_argument('--seed',           type=int, default=42)
    p.add_argument('--large-model',    action='store_true', default=False,
                   help='Use large CNN (must match training setting)')
    p.add_argument('--partial-obs',    action='store_true', default=False,
                   help='Use partial observability (must match training setting)')
    return p.parse_args()


def make_eval_env(env_id, env_type, tile_size, no_oracle, oracle_cost,
                  reward_shaping, partial_obs=False):
    """Build env with render_mode='rgb_array' for visualization."""
    from env_wrapper import RewardShaper
    from minigrid.wrappers import RGBImgPartialObsWrapper

    inner = gym.make(env_id, render_mode='rgb_array')
    if partial_obs:
        inner = RGBImgPartialObsWrapper(inner, tile_size=tile_size)
    else:
        inner = FullyObsWrapper(inner)
        inner = RGBImgObsWrapper(inner, tile_size=tile_size)

    class EvalWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self._shaper = RewardShaper() if reward_shaping else None
            if no_oracle:
                self.n_actions    = env.action_space.n
                self.QUERY_ACTION = None
            else:
                self.n_actions    = env.action_space.n + 1
                self.QUERY_ACTION = env.action_space.n
            self._env_type = env_type

            import gymnasium.spaces as spaces
            h, w, c = env.observation_space['image'].shape
            self.observation_space = spaces.Box(0, 255, (h, w, c), dtype=np.uint8)
            self.action_space      = spaces.Discrete(self.n_actions)

        def _get_obs(self, obs_dict):
            return obs_dict['image']

        def reset(self, **kwargs):
            obs_dict, info = self.env.reset(**kwargs)
            if self._shaper:
                self._shaper.reset(self.env.unwrapped)
            return self._get_obs(obs_dict), info

        def step(self, action):
            guided = False
            if self.QUERY_ACTION is not None and action == self.QUERY_ACTION:
                action = get_oracle_action(self.env.unwrapped, self._env_type)
                guided = True

            obs_dict, reward, terminated, truncated, info = self.env.step(action)
            if self._shaper:
                reward = self._shaper.shape(self.env.unwrapped, reward)
            info['guided'] = guided
            return self._get_obs(obs_dict), reward, terminated, truncated, info

        def render(self):
            return self.env.unwrapped.render()

    return EvalWrapper(inner)


def run_episode(env, model, device, no_oracle, seed):
    obs, _ = env.reset(seed=seed)
    frames, total_ret, step = [], 0.0, 0
    QUERY_ACTION = env.QUERY_ACTION

    while True:
        frame = env.render()
        if frame is not None:
            frames.append(frame.copy())

        obs_t = torch.tensor(obs[None], dtype=torch.uint8).to(device)
        with torch.no_grad():
            logits, _ = model(obs_t)
            if no_oracle:
                action = logits.argmax(dim=-1).item()
            else:
                # Let model decide — including potentially querying oracle
                action = logits.argmax(dim=-1).item()

        obs, reward, terminated, truncated, info = env.step(action)
        total_ret += reward
        step += 1
        guided = info.get('guided', False)

        name = 'Oracle' if guided else ACTION_NAMES.get(action, str(action))
        print(f"  step={step:3d}  action={name:8s}  reward={reward:+.3f}  "
              f"guided={guided}  ret={total_ret:.3f}")

        if terminated or truncated:
            frame = env.render()
            if frame is not None:
                frames.append(frame.copy())
            status = 'SUCCESS' if terminated else 'TIMEOUT'
            print(f"  → {status} in {step} steps, return={total_ret:.3f}")
            break

    return frames, total_ret, step


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_files = glob.glob(args.checkpoint)
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint: {args.checkpoint}")
    ckpt_path = sorted(ckpt_files)[-1]
    print(f"Checkpoint: {ckpt_path}")

    env = make_eval_env(args.env_id, args.env_type, args.tile_size,
                        args.no_oracle, args.oracle_cost, args.reward_shaping,
                        partial_obs=args.partial_obs)

    if args.partial_obs:
        CNNPolicy = CNNPolicyPartial
    elif args.large_model:
        CNNPolicy = CNNPolicyLarge
    else:
        CNNPolicy = CNNPolicySmall
    model = CNNPolicy(env.observation_space.shape, env.n_actions, args.hidden_dim).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Model loaded — obs={env.observation_space.shape}, n_actions={env.n_actions}")

    all_frames = []
    for ep in range(args.n_episodes):
        print(f"\n── Episode {ep+1}/{args.n_episodes} ──")
        frames, ret, steps = run_episode(env, model, device, args.no_oracle,
                                         seed=args.seed + ep)
        separator = [np.zeros_like(frames[0])] * 4
        all_frames += frames + separator

    env.close()

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    imageio.mimsave(args.out, all_frames, fps=args.fps, loop=0)
    print(f"\nSaved → {args.out}")


if __name__ == '__main__':
    main()
