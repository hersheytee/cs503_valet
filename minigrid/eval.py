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
from PIL import Image, ImageDraw, ImageFont
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper

from oracle_transfer import get_oracle_action as _get_oracle_action_transfer
from oracle import get_oracle_action as _get_oracle_action_doorkey

_TRANSFER_ENV_TYPES = {'fetch', 'gotodoor', 'gotoobject', 'multiroom'}

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
    p.add_argument('--stochastic',     action='store_true', default=False,
                   help='Sample actions from policy distribution instead of argmax')
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


def annotate_frame(frame, step, action_name, reward, total_ret, guided, n_queries, probs=None):
    """Overlay HUD + action probability bars on a rendered frame."""
    try:
        font    = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
        font_sm = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
    except Exception:
        font = font_sm = ImageFont.load_default()

    img = Image.fromarray(frame).convert('RGB')
    W, H = img.size

    # ── Probability panel (appended below) ───────────────────────────────────
    BAR_H    = 18   # height per action row
    LABEL_W  = 58   # label column width
    BAR_MAXW = W - LABEL_W - 6  # max bar width
    N_ACTS   = len(ACTION_NAMES)
    PANEL_H  = N_ACTS * BAR_H + 6

    panel = Image.new('RGB', (W, PANEL_H), (30, 30, 30))
    pd_   = ImageDraw.Draw(panel)

    if probs is not None:
        for i, (act_id, act_name) in enumerate(ACTION_NAMES.items()):
            p    = float(probs[act_id]) if act_id < len(probs) else 0.0
            y0   = 3 + i * BAR_H
            y1   = y0 + BAR_H - 3

            is_oracle   = (act_id == N_ACTS - 1)
            is_selected = (act_name == action_name) or (guided and is_oracle)

            # Bar colour
            if is_oracle:
                bar_col = (220, 60, 60) if is_selected else (160, 60, 60)
            elif is_selected:
                bar_col = (60, 200, 100)
            else:
                bar_col = (70, 130, 200)

            bar_w = max(2, int(p * BAR_MAXW))
            pd_.rectangle([LABEL_W, y0, LABEL_W + bar_w, y1], fill=bar_col)

            # Label + pct
            label = f'{act_name[:6]:<6s}'
            pd_.text((2, y0), label,    font=font_sm, fill=(200, 200, 200))
            pd_.text((LABEL_W + bar_w + 3, y0), f'{p*100:4.1f}%',
                     font=font_sm, fill=(200, 200, 200))

    # ── Combine frame + panel ─────────────────────────────────────────────────
    combined = Image.new('RGB', (W, H + PANEL_H))
    combined.paste(img,   (0, 0))
    combined.paste(panel, (0, H))
    draw = ImageDraw.Draw(combined, 'RGBA')

    # ── Red border + banner when oracle called ────────────────────────────────
    if guided:
        for t in range(4):
            draw.rectangle([t, t, W - 1 - t, H - 1 - t], outline=(220, 30, 30, 255))
        draw.rectangle([0, 0, W, 18], fill=(220, 30, 30, 210))
        draw.text((4, 2), '◆ ORACLE QUERY', font=font, fill=(255, 255, 255, 255))
    else:
        draw.rectangle([0, 0, W - 1, H - 1], outline=(80, 80, 80, 180))

    # ── Top HUD ───────────────────────────────────────────────────────────────
    hud_h = 16
    draw.rectangle([0, H - hud_h, W, H], fill=(0, 0, 0, 170))
    hud_text = (f'step={step}  act={action_name:<7s}  '
                f'r={reward:+.2f}  R={total_ret:.3f}  queries={n_queries}')
    draw.text((4, H - hud_h + 2), hud_text, font=font_sm, fill=(220, 220, 220, 255))

    return np.array(combined)


def run_episode(env, model, device, no_oracle, seed, stochastic=False):
    obs, _ = env.reset(seed=seed)
    frames, total_ret, step, n_queries = [], 0.0, 0, 0

    while True:
        obs_t = torch.tensor(obs[None], dtype=torch.uint8).to(device)
        with torch.no_grad():
            logits, _ = model(obs_t)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            if stochastic:
                action = torch.distributions.Categorical(logits=logits).sample().item()
            else:
                action = logits.argmax(dim=-1).item()

        obs, reward, terminated, truncated, info = env.step(action)
        total_ret += reward
        step += 1
        guided = info.get('guided', False)
        if guided:
            n_queries += 1

        name = 'Oracle' if guided else ACTION_NAMES.get(action, str(action))
        print(f"  step={step:3d}  action={name:8s}  reward={reward:+.3f}  "
              f"guided={guided}  queries={n_queries}  ret={total_ret:.3f}")

        frame = env.render()
        if frame is not None:
            frames.append(annotate_frame(frame, step, name, reward,
                                         total_ret, guided, n_queries, probs=probs))

        if terminated or truncated:
            status = 'SUCCESS' if terminated else 'TIMEOUT'
            print(f"  → {status} in {step} steps, return={total_ret:.3f}, "
                  f"oracle_queries={n_queries}")
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
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    if isinstance(ckpt, dict):
        print(f"Model loaded — best_return={ckpt.get('best_return', '?'):.4f}, "
              f"seed={ckpt.get('seed', '?')}")
    print(f"  obs={env.observation_space.shape}, n_actions={env.n_actions}")

    all_frames = []
    for ep in range(args.n_episodes):
        print(f"\n── Episode {ep+1}/{args.n_episodes} ──")
        frames, ret, steps = run_episode(env, model, device, args.no_oracle,
                                         seed=args.seed + ep,
                                         stochastic=args.stochastic)
        separator = [np.zeros_like(frames[0])] * 4
        all_frames += frames + separator

    env.close()

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    imageio.mimsave(args.out, all_frames, fps=args.fps, loop=0)
    print(f"\nSaved → {args.out}")


if __name__ == '__main__':
    main()
