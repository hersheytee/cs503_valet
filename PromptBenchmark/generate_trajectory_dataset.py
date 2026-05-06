"""
Génère un dataset avec historique d'actions.
Simule un agent semi-aléatoire (70% optimal, 30% random) pour
créer des trajectoires réalistes.
"""
import json
import random
import argparse
import sys
import traceback
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper

sys.path.append("../Benchmark")
from utils.generate_dataset import (
    bfs_oracle, randomize_agent, ENV_POOL, ACTION_NAMES, TILE_SIZE, DIR_STR
)

ORACLE_PROB = 0.7
HISTORY_LEN = 5
TRAJ_LEN    = 5


def make_env(env_name, seed):
    env = gym.make(env_name, render_mode="rgb_array", tile_size=TILE_SIZE)
    env = FullyObsWrapper(env)
    env = RGBImgObsWrapper(env, tile_size=TILE_SIZE)
    env.reset(seed=seed)
    return env


def random_or_oracle(env):
    if random.random() < ORACLE_PROB:
        opt_actions, _ = bfs_oracle(env)
        if opt_actions:
            return random.choice(opt_actions)
    return random.choice([0, 1, 2])


def generate(n_samples, output_dir, seed_offset=0):
    out_path = Path(output_dir)
    img_path = out_path / "images"
    img_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed_offset)
    samples = []

    for i in tqdm(range(n_samples), desc="Generating trajectories"):
        env_name, complexity = rng.choice(ENV_POOL)   # ← tuple unpacking
        seed = seed_offset + i

        try:
            env = make_env(env_name, seed)
            randomize_agent(env, rng)

            mission = env.unwrapped.mission   # avant le close()

            history = []
            terminated = False
            for _ in range(TRAJ_LEN):
                action = random_or_oracle(env)
                history.append(ACTION_NAMES[action])
                _, _, terminated, _, _ = env.step(action)
                if terminated:
                    break

            if terminated:
                env.close()
                continue

            agent_pos = list(map(int, env.unwrapped.agent_pos))
            agent_dir = int(env.unwrapped.agent_dir)
            opt_actions, oracle_info = bfs_oracle(env)
            if not opt_actions:
                env.close()
                continue

            global_rgb = env.render()
            env.close()

            img_id = f"{i:05d}"
            Image.fromarray(global_rgb).save(img_path / f"{img_id}_global.png")

            samples.append({
                "id":              img_id,
                "env":             env_name,
                "complexity":      complexity,
                "seed":            seed,
                "mission":         mission,
                "agent_pos":       agent_pos,
                "agent_dir":       agent_dir,
                "agent_dir_str":   DIR_STR[agent_dir],
                "global_image":    f"images/{img_id}_global.png",
                "action_history":  history[-HISTORY_LEN:],
                "optimal_actions": opt_actions,
                "action_names":    [ACTION_NAMES[a] for a in opt_actions],
                "agent_carrying":  None,
                "oracle_valid":    True,
            })
        except Exception as e:
            print(f"  ⚠ skip sample {i}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

    with open(out_path / "trajectory_dataset.json", "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  ✅ {len(samples)} samples saved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", type=str, default="./trajectory_dataset")
    p.add_argument("--seed_offset", type=int, default=0)
    args = p.parse_args()
    generate(args.n, args.out, args.seed_offset)