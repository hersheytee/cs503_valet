"""
Visualise a trained Sokoban policy and save a GIF.

Renders each episode with:
  - upscaled board (4×)
  - red border + banner on oracle-query steps
  - action probability bars below the board
  - HUD overlay (step / action / reward / queries)

Works with standard (3-channel) and budget-aware (4-channel) checkpoints;
channel count is auto-detected from the saved weights.

Usage:
    python gym_sokoban/eval_gif.py \\
        --checkpoint runs/.../checkpoints/final.pt \\
        --n-episodes 5 \\
        --out gym_sokoban/figures/final/eval.gif

    # Budget run (will auto-detect 4-channel obs):
    python gym_sokoban/eval_gif.py \\
        --checkpoint runs/.../checkpoints/final.pt \\
        --max-oracle-queries 3

    # No oracle (baseline):
    python gym_sokoban/eval_gif.py \\
        --checkpoint runs/.../checkpoints/final.pt \\
        --no-oracle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from model import CNNPolicy
from env_wrapper import SokobanOracleWrapper

import numpy as _np
if not hasattr(_np, "bool8"):
    _np.bool8 = _np.bool_

# ── Action names ──────────────────────────────────────────────────────────────

ACTION_NAMES = {
    0: "No-op",
    1: "Push Up",
    2: "Push Dn",
    3: "Push L",
    4: "Push R",
    5: "Move Up",
    6: "Move Dn",
    7: "Move L",
    8: "Move R",
    9: "Oracle",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def detect_channels(state_dict: dict) -> int:
    """Read input channel count from the first conv weight."""
    key = next(k for k in state_dict if "cnn" in k and "weight" in k)
    return state_dict[key].shape[1]


def load_font(size: int):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def annotate_frame(
    frame: np.ndarray,
    step: int,
    action_name: str,
    reward: float,
    total_ret: float,
    guided: bool,
    n_queries: int,
    budget_remaining: int | None,
    probs: np.ndarray | None,
    n_actions: int,
    scale: int = 4,
) -> np.ndarray:
    # Upscale board
    img = Image.fromarray(frame).resize(
        (frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST
    )
    W, H = img.size

    font    = load_font(13)
    font_sm = load_font(11)

    # ── Action probability panel ──────────────────────────────────────────────
    BAR_H    = 20
    LABEL_W  = 68
    BAR_MAXW = W - LABEL_W - 8
    PANEL_H  = n_actions * BAR_H + 8

    panel = Image.new("RGB", (W, PANEL_H), (28, 28, 28))
    pd_   = ImageDraw.Draw(panel)

    if probs is not None:
        for act_id in range(n_actions):
            name = ACTION_NAMES.get(act_id, str(act_id))
            p    = float(probs[act_id]) if act_id < len(probs) else 0.0
            y0   = 4 + act_id * BAR_H
            y1   = y0 + BAR_H - 4

            is_oracle   = act_id == n_actions - 1 and act_id == 9
            is_selected = (ACTION_NAMES.get(act_id) == action_name) or (guided and is_oracle)

            if is_oracle:
                col = (220, 60, 60) if is_selected else (140, 50, 50)
            elif is_selected:
                col = (60, 200, 100)
            else:
                col = (70, 120, 190)

            bar_w = max(2, int(p * BAR_MAXW))
            pd_.rectangle([LABEL_W, y0, LABEL_W + bar_w, y1], fill=col)
            pd_.text((3, y0), f"{name[:8]:<8s}", font=font_sm, fill=(190, 190, 190))
            pd_.text((LABEL_W + bar_w + 4, y0), f"{p*100:4.1f}%", font=font_sm, fill=(190, 190, 190))

    # ── Combine ───────────────────────────────────────────────────────────────
    combined = Image.new("RGB", (W, H + PANEL_H))
    combined.paste(img,   (0, 0))
    combined.paste(panel, (0, H))
    draw = ImageDraw.Draw(combined, "RGBA")

    # Oracle border + banner
    if guided:
        for t in range(4):
            draw.rectangle([t, t, W - 1 - t, H - 1 - t], outline=(220, 30, 30, 255))
        draw.rectangle([0, 0, W, 20], fill=(220, 30, 30, 210))
        draw.text((4, 3), "◆ ORACLE QUERY", font=font, fill=(255, 255, 255, 255))
    else:
        draw.rectangle([0, 0, W - 1, H - 1], outline=(70, 70, 70, 160))

    # HUD strip at bottom of board
    hud_h = 18
    draw.rectangle([0, H - hud_h, W, H], fill=(0, 0, 0, 175))
    budget_str = f"  budget={budget_remaining}" if budget_remaining is not None else ""
    hud = (f"step={step}  act={action_name:<8s}  "
           f"r={reward:+.2f}  R={total_ret:.3f}  queries={n_queries}{budget_str}")
    draw.text((4, H - hud_h + 2), hud, font=font_sm, fill=(210, 210, 210, 255))

    return np.array(combined)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    env: SokobanOracleWrapper,
    model: CNNPolicy,
    device: torch.device,
    seed: int,
    stochastic: bool,
    scale: int,
) -> tuple[list[np.ndarray], float, int]:
    obs, _ = env.reset(seed=seed)
    frames, total_ret, step, n_queries = [], 0.0, 0, 0
    n_actions = env.action_space.n

    while True:
        obs_t = torch.tensor(obs[None], dtype=torch.uint8).to(device)
        with torch.no_grad():
            logits, _ = model(obs_t)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            action = (torch.distributions.Categorical(logits=logits).sample().item()
                      if stochastic else logits.argmax(dim=-1).item())

        obs, reward, terminated, truncated, info = env.step(action)
        total_ret += reward
        step += 1
        guided = info.get("guided", False)
        if guided:
            n_queries += 1

        # Budget remaining (from 4th channel if present)
        budget_remaining = None
        if obs.shape[-1] == 4:
            budget_val = int(obs[0, 0, 3])
            max_q = env.max_oracle_queries
            if max_q:
                budget_remaining = round((budget_val / 255) * max_q)

        action_name = "Oracle" if guided else ACTION_NAMES.get(action, str(action))
        print(f"  step={step:3d}  {action_name:<10s}  r={reward:+.3f}  "
              f"guided={guided}  queries={n_queries}  ret={total_ret:.3f}")

        raw_frame = env.render()
        if raw_frame is not None:
            frame = annotate_frame(
                raw_frame, step, action_name, reward, total_ret,
                guided, n_queries, budget_remaining, probs, n_actions, scale=scale,
            )
            frames.append(frame)

        if terminated or truncated:
            status = "SUCCESS" if info.get("success", False) else "TIMEOUT"
            print(f"  → {status} | steps={step}  return={total_ret:.3f}  queries={n_queries}")
            break

    return frames, total_ret, step


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--env-id", type=str, default="Sokoban-small-v0")
    p.add_argument("--no-oracle", action="store_true")
    p.add_argument("--oracle-cost", type=float, default=0.0)
    p.add_argument("--oracle-accuracy", type=float, default=1.0)
    p.add_argument("--max-oracle-queries", type=int, default=None)
    p.add_argument("--max-episode-steps", type=int, default=50)
    p.add_argument("--obs-size", type=int, default=56)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--scale", type=int, default=4, help="Upscale factor for board rendering")
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=str(SCRIPT_DIR / "figures" / "final" / "eval.gif"))
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    obs_channels = detect_channels(state_dict)
    print(f"Checkpoint: {ckpt_path}  (obs_channels={obs_channels})")

    # If checkpoint is 4-channel but no --max-oracle-queries given, infer budget mode
    max_oracle_queries = args.max_oracle_queries
    if obs_channels == 4 and max_oracle_queries is None:
        raise SystemExit(
            "Checkpoint has 4 input channels (budget-aware) but --max-oracle-queries not set. "
            "Pass --max-oracle-queries N matching the training value."
        )

    env = SokobanOracleWrapper(
        args.env_id,
        oracle_cost=args.oracle_cost,
        oracle_accuracy=args.oracle_accuracy,
        no_oracle=args.no_oracle,
        max_episode_steps=args.max_episode_steps,
        obs_size=args.obs_size,
        max_oracle_queries=max_oracle_queries,
        seed=args.seed,
    )

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    model = CNNPolicy(obs_shape=obs_shape, n_actions=n_actions, hidden_dim=args.hidden_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Device: {device}  obs={obs_shape}  n_actions={n_actions}")

    all_frames: list[np.ndarray] = []
    for ep in range(args.n_episodes):
        print(f"\n── Episode {ep + 1}/{args.n_episodes} ──")
        frames, ret, steps = run_episode(
            env, model, device, seed=args.seed + ep,
            stochastic=args.stochastic, scale=args.scale,
        )
        # Black separator between episodes
        if frames:
            separator = [np.zeros_like(frames[0])] * 3
            all_frames += frames + separator

    env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out), all_frames, fps=args.fps, loop=0)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
