"""
Appends pickup and toggle samples to an existing dataset.json, or creates
a standalone file if no existing dataset is provided.

The trick: instead of hoping the random agent ends up facing the key or door,
we CONSTRUCT the situation directly by placing the agent one step in front of
the target object, facing it.

Usage:
    # Append 50 pickup + 50 toggle samples to an existing dataset
    python sample_rare_actions.py --n 50 --out ./dataset

    # Standalone (no existing dataset.json required)
    python sample_rare_actions.py --n 100 --out ./rare_dataset
"""

import argparse
import json
import random
import warnings
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import gymnasium as gym
import minigrid  # noqa
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper

# Constants
_DIR_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]   # 0=right 1=down 2=left 3=up
DIR_STR  = {0: "→", 1: "↓", 2: "←", 3: "↑"}
ACTION_NAMES = {
    0: "turn_left", 1: "turn_right", 2: "forward",
    3: "pickup",    4: "drop",       5: "toggle",  6: "done",
}
TILE_SIZE = 32

# Only DoorKey envs — they have both a key and a door
DOORKEY_ENVS = [
    ("MiniGrid-DoorKey-5x5-v0",   "medium"),
    ("MiniGrid-DoorKey-8x8-v0",   "medium"),
    ("MiniGrid-DoorKey-16x16-v0", "hard"),
]

# Helpers

def find_objects(env):
    """Returns (goal_pos, key_pos, door_pos) as (x,y) tuples or None."""
    raw  = env.unwrapped
    grid = raw.grid
    goal_pos = key_pos = door_pos = None
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is None:
                continue
            if cell.type == "goal":
                goal_pos = (x, y)
            elif cell.type == "key":
                key_pos = (x, y)
            elif cell.type == "door":
                door_pos = (x, y)
    return goal_pos, key_pos, door_pos


def free_cells(env):
    """Returns all empty cells (no object)."""
    raw   = env.unwrapped
    grid  = raw.grid
    cells = []
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is None or cell.type == "floor":
                cells.append((x, y))
    return cells

def _goal_side_cells(grid, goal_pos, W, H) -> set:
    """
    Flood-fill from goal WITHOUT crossing walls or doors.
    Returns cells on the same side as the goal.
    """
    if goal_pos is None:
        return set()

    reachable = {goal_pos}
    queue     = deque([goal_pos])
    dir_vecs  = [(1,0),(0,1),(-1,0),(0,-1)]

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in dir_vecs:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if (nx, ny) in reachable:
                continue
            cell = grid.get(nx, ny)
            if cell is not None and cell.type in ("wall", "door", "lava"):
                continue
            reachable.add((nx, ny))
            queue.append((nx, ny))

    return reachable




#  Construct pickup situation
#  Agent placed one step in front of the key, facing it

def make_pickup_situation(env_name: str, seed: int):
    """
    Resets the env, then places the agent adjacent to the key, facing it.
    Returns (agent_pos, agent_dir) or raises if no valid position exists.
    """
    env = gym.make(env_name, render_mode=None, tile_size=TILE_SIZE)
    env = FullyObsWrapper(env)
    env.reset(seed=seed)
    _, key_pos, _ = find_objects(env)
    env.close()

    if key_pos is None:
        raise ValueError("No key found in this env/seed")

    kx, ky = key_pos

    # Try all 4 directions: find a free cell adjacent to the key
    # such that the agent can stand there and face the key
    #
    # If agent is at (kx - dx, ky - dy) facing direction d,
    # then the cell in front = (kx, ky) ✓
    candidates = []
    for d, (dx, dy) in enumerate(_DIR_VEC):
        stand_pos = (kx - dx, ky - dy)  # cell behind the key from direction d

        # Verify stand_pos is a free cell
        env2 = gym.make(env_name, render_mode=None, tile_size=TILE_SIZE)
        env2 = FullyObsWrapper(env2)
        env2.reset(seed=seed)
        raw  = env2.unwrapped
        grid = raw.grid
        W, H = grid.width, grid.height

        if not (0 <= stand_pos[0] < W and 0 <= stand_pos[1] < H):
            env2.close()
            continue
        cell = grid.get(*stand_pos)
        if cell is None or cell.type == "floor":
            candidates.append((stand_pos, d))
        env2.close()

    if not candidates:
        raise ValueError(f"No valid stand position adjacent to key at {key_pos}")

    # Filter out candidates on the goal side — agent must stay on key side
    env_check = gym.make(env_name, render_mode=None, tile_size=TILE_SIZE)
    env_check = FullyObsWrapper(env_check)
    env_check.reset(seed=seed)
    raw_c  = env_check.unwrapped
    grid_c = raw_c.grid
    W_c, H_c = grid_c.width, grid_c.height
    goal_c = None
    for x in range(W_c):
        for y in range(H_c):
            cell = grid_c.get(x, y)
            if cell is not None and cell.type == "goal":
                goal_c = (x, y)
                break
    goal_side = _goal_side_cells(grid_c, goal_c, W_c, H_c)
    env_check.close()

    valid = [(p, d) for p, d in candidates if p not in goal_side]
    if not valid:
        valid = candidates  # fallback: keep all if none survive filter

    return valid[0]


#  Construct toggle situation
#  Agent placed adjacent to the door, facing it, with the key already in hand

def make_toggle_situation(env_name: str, seed: int):
    """
    Resets the env, places the agent adjacent to the door (facing it),
    and gives it the key. Returns (agent_pos, agent_dir) or raises.
    """
    env = gym.make(env_name, render_mode=None, tile_size=TILE_SIZE)
    env = FullyObsWrapper(env)
    env.reset(seed=seed)
    _, key_pos, door_pos = find_objects(env)
    env.close()

    if door_pos is None:
        raise ValueError("No door found in this env/seed")

    dx_d, dy_d = door_pos

    candidates = []
    for d, (dx, dy) in enumerate(_DIR_VEC):
        stand_pos = (dx_d - dx, dy_d - dy)

        env2 = gym.make(env_name, render_mode=None, tile_size=TILE_SIZE)
        env2 = FullyObsWrapper(env2)
        env2.reset(seed=seed)
        raw  = env2.unwrapped
        grid = raw.grid
        W, H = grid.width, grid.height

        if not (0 <= stand_pos[0] < W and 0 <= stand_pos[1] < H):
            env2.close()
            continue
        cell = grid.get(*stand_pos)
        if cell is None or cell.type == "floor":
            candidates.append((stand_pos, d))
        env2.close()

    if not candidates:
        raise ValueError(f"No valid stand position adjacent to door at {door_pos}")

    return candidates[0]


#  Capture images

def _render_both_views(env_name: str, seed: int, agent_pos, agent_dir: int,
                       give_key: bool = False):
    """
    Renders global + partial views with a specific agent position/direction.
    If give_key=True, removes the key from the grid and sets agent.carrying.
    Returns (global_rgb, partial_rgb, mission).
    """
    # Global view 
    env_g = gym.make(env_name, render_mode="rgb_array", tile_size=TILE_SIZE)
    env_g = FullyObsWrapper(env_g)
    env_g = RGBImgObsWrapper(env_g, tile_size=TILE_SIZE)
    env_g.reset(seed=seed)

    raw  = env_g.unwrapped
    raw.agent_pos = np.array(agent_pos)
    raw.agent_dir = agent_dir

    if give_key:
        # Find and remove the key from the grid, give it to the agent
        grid = raw.grid
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None and cell.type == "key":
                    raw.carrying = cell
                    grid.set(x, y, None)
                    break

    mission    = raw.mission
    global_rgb = env_g.render()
    env_g.close()

    # Partial view 
    env_p = gym.make(env_name, render_mode="rgb_array", tile_size=TILE_SIZE)
    env_p = RGBImgPartialObsWrapper(env_p, tile_size=TILE_SIZE)
    env_p.reset(seed=seed)

    raw_p = env_p.unwrapped
    raw_p.agent_pos = np.array(agent_pos)
    raw_p.agent_dir = agent_dir

    if give_key:
        grid_p = raw_p.grid
        for x in range(grid_p.width):
            for y in range(grid_p.height):
                cell = grid_p.get(x, y)
                if cell is not None and cell.type == "key":
                    raw_p.carrying = cell
                    grid_p.set(x, y, None)
                    break

    obs         = env_p.observation(env_p.unwrapped.gen_obs())
    partial_rgb = obs["image"]
    env_p.close()

    return global_rgb, partial_rgb, mission


def capture_pickup(env_name: str, seed: int):
    agent_pos, agent_dir = make_pickup_situation(env_name, seed)
    global_rgb, partial_rgb, mission = _render_both_views(
        env_name, seed, agent_pos, agent_dir, give_key=False
    )
    return agent_pos, agent_dir, global_rgb, partial_rgb, mission


def capture_toggle(env_name: str, seed: int):
    agent_pos, agent_dir = make_toggle_situation(env_name, seed)
    global_rgb, partial_rgb, mission = _render_both_views(
        env_name, seed, agent_pos, agent_dir, give_key=True
    )
    return agent_pos, agent_dir, global_rgb, partial_rgb, mission

# Main 

def generate_rare(n_per_action: int, out_dir: str, seed_offset: int):
    out     = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Load existing dataset if present (to append)
    json_path = out / "dataset.json"
    if json_path.exists():
        with open(json_path) as f:
            records = json.load(f)
        start_id = len(records)
        print(f"Appending to existing dataset ({start_id} samples already present)")
    else:
        records  = []
        start_id = 0
        print("Creating new dataset")

    sample_id   = start_id
    global_seed = seed_offset
    skipped     = 0

    tasks = [
        ("pickup", capture_pickup, 3),
        ("toggle", capture_toggle, 5),
    ]

    for action_name, capture_fn, action_int in tasks:
        print(f"\n  Generating {n_per_action} × {action_name} samples...")
        generated = 0
        attempts  = 0
        rng_env   = random.Random(seed_offset + action_int * 1000)

        while generated < n_per_action:
            attempts += 1
            if attempts > n_per_action * 20:
                warnings.warn(f"Too many attempts for {action_name}, stopping early.")
                break

            env_name, complexity = rng_env.choice(DOORKEY_ENVS)

            try:
                agent_pos, agent_dir, global_rgb, partial_rgb, mission = \
                    capture_fn(env_name, global_seed)
            except Exception as e:
                global_seed += 1
                skipped     += 1
                continue

            sid          = f"{sample_id:05d}"
            global_path  = img_dir / f"{sid}_global.png"
            partial_path = img_dir / f"{sid}_partial.png"
            Image.fromarray(global_rgb).save(global_path)
            Image.fromarray(partial_rgb).save(partial_path)

            records.append({
                "id":             sid,
                "env":            env_name,
                "seed":           global_seed,
                "complexity":     complexity,
                "mission":        mission,
                "agent_pos":      list(map(int, agent_pos)),
                "agent_dir":      agent_dir,
                "agent_dir_str":  DIR_STR[agent_dir],
                "global_image":   f"images/{sid}_global.png",
                "partial_image":  f"images/{sid}_partial.png",
                "optimal_action": action_int,
                "action_name":    action_name,
                "oracle_info":     "constructed",
                "agent_carrying":  "key" if action_int == 5 else None,
                "optimal_actions": [action_int],   # single optimal action by construction
                "action_names":    [action_name],
                "oracle_valid":    True,
            })

            sample_id   += 1
            generated   += 1
            global_seed += 1
            print(f"    [{sid}] {env_name:35s}  pos={list(map(int,agent_pos))}  "
                  f"dir={DIR_STR[agent_dir]}  → {action_name}")

    #  Save
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    try:
        import pandas as pd
        pd.DataFrame(records).to_parquet(out / "dataset.parquet", index=False)
    except ImportError:
        pass

    # Summary 
    from collections import Counter
    action_counts = Counter(a for r in records for a in r["action_names"])
    n_added = len(records) - start_id

    print(f"\n{'─'*55}")
    print(f"{n_added} samples added  →  {len(records)} total")
    print(f" Skipped : {skipped}")
    print(f"{'─'*55}")
    print("\n  Full action distribution in dataset:")
    max_c = max(action_counts.values(), default=1)
    for name, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
        bar = "█" * max(1, cnt * 28 // max_c)
        print(f"    {name:13s} {bar} {cnt}")
    print(f"\n  Metadata → {json_path}\n")


# CLI

def parse_args():
    p = argparse.ArgumentParser(
        description="Append targeted pickup & toggle samples to a MiniGrid dataset"
    )
    p.add_argument("--n",           type=int, default=50,
                   help="Number of samples PER rare action (default: 50)")
    p.add_argument("--out",         type=str, default="./dataset",
                   help="Dataset directory — appends to existing dataset.json if present")
    p.add_argument("--seed_offset", type=int, default=5000,
                   help="Seed offset to avoid overlap with main dataset (default: 5000)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_rare(
        n_per_action=args.n,
        out_dir=args.out,
        seed_offset=args.seed_offset,
    )