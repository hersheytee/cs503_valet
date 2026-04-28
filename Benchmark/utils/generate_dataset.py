"""
Generates N samples (global view + partial view + validated oracle action).

----------------
Each JSON entry:
  {
    "id":             "00042",
    "env":            "MiniGrid-DoorKey-8x8-v0",
    "seed":           42,
    "complexity":     "medium",
    "mission":        "use the key to open the door and then get to the goal",
    "agent_pos":      [2, 3],
    "agent_dir":      1,
    "agent_dir_str":  "↓",
    "global_image":   "images/00042_global.png",
    "partial_image":  "images/00042_partial.png",
    "optimal_action": 3,
    "action_name":    "pickup",
    "oracle_info":    "path_len=14",
    "oracle_valid":   true       ← verified by simulation
  }
Usage:
    python generate_dataset.py
    python generate_dataset.py --n 200 --out ./dataset --seed_offset 0
"""

import argparse
import json
import random
import warnings
from collections import Counter, deque
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import gymnasium as gym
import minigrid  # noqa — registers all environments
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper

# Constants
_DIR_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]   # 0=right 1=down 2=left 3=up
DIR_STR  = {0: "→", 1: "↓", 2: "←", 3: "↑"}
ACTION_NAMES = {
    0: "turn_left", 1: "turn_right", 2: "forward",
    3: "pickup",    4: "drop",       5: "toggle",  6: "done",
}
TILE_SIZE = 32   # pixels per cell — large enough for VLMs to read clearly

# Only goal-reaching and key+door environments
ENV_POOL = [
    ("MiniGrid-Empty-8x8-v0",      "simple"),
    ("MiniGrid-Empty-16x16-v0",    "simple"),
    ("MiniGrid-DoorKey-5x5-v0",    "medium"),
    ("MiniGrid-DoorKey-8x8-v0",    "medium"),
    ("MiniGrid-DoorKey-16x16-v0",  "hard"),
]



# Goal-side cells = cells reachable from the goal without crossing a door

def _goal_side_cells(env) -> set:
    """
    Returns the set of cells that are on the same side as the goal,
    reachable WITHOUT passing through any door (flood-fill ignoring doors).

    Used by randomize_agent() to ensure the agent is never placed on the
    goal side in DoorKey environments — which would make the key/door
    sub-tasks irrelevant and produce a misleading mission prompt.
    """
    raw  = env.unwrapped
    grid = raw.grid
    W, H = grid.width, grid.height

    # Find goal
    goal_pos = None
    for x in range(W):
        for y in range(H):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                goal_pos = (x, y)
                break
        if goal_pos:
            break

    if goal_pos is None:
        return set()

    # Flood-fill from goal, treating doors and walls as impassable
    reachable = set()
    queue     = deque([goal_pos])
    reachable.add(goal_pos)

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in _DIR_VEC:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if (nx, ny) in reachable:
                continue
            cell = grid.get(nx, ny)
            # Stop at walls AND doors (treat closed door as wall)
            if cell is not None and cell.type in ("wall", "door", "lava"):
                continue
            reachable.add((nx, ny))
            queue.append((nx, ny))

    return reachable

# Position randomization

def randomize_agent(env, rng: random.Random):
    """
    Moves the agent to a random free cell with a random direction, in-place.

    Must be called after env.reset(). Works by directly modifying the internal
    state — necessary because Empty-* always places the agent at (1,1) facing
    right regardless of the seed.

    Free cells = cells with no object (wall, door, key, goal, lava).
    """
    raw  = env.unwrapped
    grid = raw.grid
    W, H = grid.width, grid.height

    # Cells reachable from the goal without crossing a door
    # excluded so the agent always starts on the key/door side
    goal_side = _goal_side_cells(env)

    free_cells = []
    for x in range(W):
        for y in range(H):
            if (x, y) in goal_side:
                continue          # skip goal-side cells in DoorKey envs
            cell = grid.get(x, y)
            if cell is None or cell.type == "floor":
                free_cells.append((x, y))

    if not free_cells:
        # Fallback: if all free cells happen to be on the goal side
        # (e.g. Empty env with no door), allow any free cell
        for x in range(W):
            for y in range(H):
                cell = grid.get(x, y)
                if cell is None or cell.type == "floor":
                    free_cells.append((x, y))

    if not free_cells:
        return  # truly no free cell — keep default position

    new_pos = rng.choice(free_cells)
    new_dir = rng.randint(0, 3)

    raw.agent_pos = np.array(new_pos)
    raw.agent_dir = new_dir


# BFS Helpers

def _get_successors(state, grid, W, H, key_pos, door_pos):
    """
    Returns list of (action, next_state) from a given state.
    State = (pos, dir, has_key, door_open).
    """
    pos, d, has_key, door_open = state
    dx, dy  = _DIR_VEC[d]
    fwd     = (pos[0] + dx, pos[1] + dy)
    in_grid = 0 <= fwd[0] < W and 0 <= fwd[1] < H

    succs = [
        (0, (pos, (d - 1) % 4, has_key, door_open)),
        (1, (pos, (d + 1) % 4, has_key, door_open)),
    ]

    if in_grid:
        fwd_cell = grid.get(*fwd)
        blocked  = fwd_cell is not None and (
            fwd_cell.type == "wall"
            or fwd_cell.type == "lava"
            or (fwd_cell.type == "door" and not door_open)
        )
        if not blocked:
            succs.append((2, (fwd, d, has_key, door_open)))

        if not has_key and key_pos is not None and fwd == key_pos:
            succs.append((3, (pos, d, True, door_open)))

        if door_pos is not None and not door_open and fwd == door_pos and has_key:
            succs.append((5, (pos, d, has_key, True)))

    return succs


def _bfs_dist(start, goal_pos, grid, W, H, key_pos, door_pos):
    """Returns shortest number of steps from start to goal_pos, or None."""
    if start[0] == goal_pos:
        return 0
    queue   = deque([(start, 0)])
    visited = {start}
    while queue:
        state, dist = queue.popleft()
        for _, next_state in _get_successors(state, grid, W, H, key_pos, door_pos):
            if next_state in visited:
                continue
            if next_state[0] == goal_pos:
                return dist + 1
            visited.add(next_state)
            queue.append((next_state, dist + 1))
    return None


# bfs_oracle

def bfs_oracle(env):
    """
    Returns ALL optimal first actions as a list.

    Two-phase approach to avoid the visited-set bug that collapses symmetric
    paths (e.g. 180° pivot: turn_leftx2 and turn_rightx2 share the same
    intermediate state, so a single BFS would only keep the first path found):

      Phase 1 — find best_len with a standard BFS from start.
      Phase 2 — for each feasible first action, check if the resulting state
                reaches the goal in exactly best_len-1 more steps.

    Exact MiniGrid mechanics:
      - turn_left / turn_right : change direction, do not move
      - forward  : move one cell ahead (blocked by wall or closed door)
      - pickup   : pick up the object on the cell DIRECTLY IN FRONT
      - toggle   : open the door DIRECTLY IN FRONT (key required)
    """
    raw  = env.unwrapped
    grid = raw.grid
    pos0 = tuple(map(int, raw.agent_pos))
    dir0 = int(raw.agent_dir)
    W, H = grid.width, grid.height

    goal_pos = key_pos = door_pos = None
    for x in range(W):
        for y in range(H):
            cell = grid.get(x, y)
            if cell is None:
                continue
            if cell.type == "goal":
                goal_pos = (x, y)
            elif cell.type == "key":
                key_pos = (x, y)
            elif cell.type == "door":
                door_pos = (x, y)

    if goal_pos is None:
        return [], "no goal found"

    has_key0   = raw.carrying is not None and raw.carrying.type == "key"
    door_open0 = (door_pos is None) or (
        grid.get(*door_pos) is not None and grid.get(*door_pos).is_open
    )

    start    = (pos0, dir0, has_key0, door_open0)
    best_len = _bfs_dist(start, goal_pos, grid, W, H, key_pos, door_pos)

    if best_len is None:
        return [], "no path found"

    if best_len == 0:
        return [2], "path_len=0"

    # Phase 2: each feasible first action is optimal iff it leads to a state
    # from which the goal is reachable in exactly best_len-1 steps.
    best_actions = []
    for action, next_state in _get_successors(start, grid, W, H, key_pos, door_pos):
        remaining = _bfs_dist(next_state, goal_pos, grid, W, H, key_pos, door_pos)
        if remaining is not None and remaining == best_len - 1:
            best_actions.append(action)

    if best_actions:
        actions_list = sorted(best_actions)
        return actions_list, f"path_len={best_len}"
    return [], "no path found"


# Oracle validation

def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def validate_oracle(env_name: str, seed: int, agent_pos, agent_dir: int,
                    action: int) -> bool:
    """
    Replays the oracle action in a fresh environment with the exact same
    agent position and direction, then checks for consistency:

      - pickup  → the cell in front must contain a key
      - toggle  → the cell in front must be a door + agent holds a key
      - forward → agent must get closer to the current sub-goal
      - turn    → distance to sub-goal must not increase by more than 1
                  (a pivot is sometimes necessary before moving forward)

    Returns True if the action appears valid.
    """
    try:
        env = gym.make(env_name, render_mode=None, tile_size=TILE_SIZE)
        env = FullyObsWrapper(env)
        env.reset(seed=seed)

        # Restore the exact randomized agent state
        raw = env.unwrapped
        raw.agent_pos = np.array(agent_pos)
        raw.agent_dir = agent_dir

        grid = raw.grid
        W, H = grid.width, grid.height

        goal_pos = key_pos = door_pos = None
        for x in range(W):
            for y in range(H):
                cell = grid.get(x, y)
                if cell is None:
                    continue
                if cell.type == "goal":
                    goal_pos = (x, y)
                elif cell.type == "key":
                    key_pos = (x, y)
                elif cell.type == "door":
                    door_pos = (x, y)

        def current_target():
            """Returns the current sub-goal: key → door → goal."""
            hk = raw.carrying is not None and raw.carrying.type == "key"
            if not hk and key_pos is not None:
                return key_pos
            if door_pos is not None:
                c = grid.get(*door_pos)
                if c is not None and not c.is_open:
                    return door_pos
            return goal_pos

        pos_before  = tuple(map(int, raw.agent_pos))
        target      = current_target()
        dist_before = _manhattan(pos_before, target) if target else 0

        dx, dy   = _DIR_VEC[raw.agent_dir]
        fwd      = (pos_before[0] + dx, pos_before[1] + dy)
        fwd_cell = grid.get(*fwd) if (0 <= fwd[0] < W and 0 <= fwd[1] < H) else None

        # Static pre-execution checks for pickup / toggle
        if action == 3:   # pickup
            valid = fwd_cell is not None and fwd_cell.type == "key"
            env.close()
            return valid

        if action == 5:   # toggle
            hk    = raw.carrying is not None and raw.carrying.type == "key"
            valid = fwd_cell is not None and fwd_cell.type == "door" and hk
            env.close()
            return valid

        # Execute action and check resulting distance
        _, _, terminated, _, _ = env.step(action)
        env.close()

        if terminated:
            return True  # reached the goal — always valid

        pos_after  = tuple(map(int, raw.agent_pos))
        dist_after = _manhattan(pos_after, target) if target else 0

        if action == 2:        # forward must bring agent closer
            return dist_after < dist_before
        if action in (0, 1):   # turn — allow tolerance of +1
            return dist_after <= dist_before + 1

    except Exception as e:
        warnings.warn(f"Validation error ({env_name} seed={seed}): {e}")

    return False


# Sample capture

def capture(env_name: str, seed: int, rng: random.Random) -> dict:
    """
    Resets the environment, randomizes the agent, captures both views,
    and computes the oracle action. Returns a dict or raises on failure.
    """
    # Global view (also used for oracle computation)
    env_g = gym.make(env_name, render_mode="rgb_array", tile_size=TILE_SIZE)
    env_g = FullyObsWrapper(env_g)
    env_g = RGBImgObsWrapper(env_g, tile_size=TILE_SIZE)
    env_g.reset(seed=seed)

    randomize_agent(env_g, rng)

    mission   = env_g.unwrapped.mission
    agent_pos = list(map(int, env_g.unwrapped.agent_pos))
    agent_dir      = int(env_g.unwrapped.agent_dir)
    _carrying       = env_g.unwrapped.carrying
    agent_carrying  = _carrying.type if _carrying is not None else None

    oracle_actions, oracle_info = bfs_oracle(env_g)
    global_rgb = env_g.render()
    env_g.close()

    if not oracle_actions:
        raise ValueError(f"No oracle action found ({oracle_info})")

    # Partial view (same seed + same agent position) 
    env_p = gym.make(env_name, render_mode="rgb_array", tile_size=TILE_SIZE)
    env_p = RGBImgPartialObsWrapper(env_p, tile_size=TILE_SIZE)
    env_p.reset(seed=seed)

    # Restore the same randomized agent state
    env_p.unwrapped.agent_pos = np.array(agent_pos)
    env_p.unwrapped.agent_dir = agent_dir

    # Force re-render with the updated position
    obs = env_p.observation(env_p.unwrapped.gen_obs())
    partial_rgb = obs["image"]
    env_p.close()

    return {
        "env":            env_name,
        "seed":           seed,
        "mission":        mission,
        "agent_pos":      agent_pos,
        "agent_dir":      agent_dir,
        "agent_dir_str":  DIR_STR[agent_dir],
        "oracle_actions": oracle_actions,   # full list of equally optimal actions
        "oracle_info":    oracle_info,
        "agent_carrying": agent_carrying,
        "global_rgb":     global_rgb,
        "partial_rgb":    partial_rgb,
    }


# Main

def generate(n_samples: int, out_dir: str, seed_offset: int):
    out     = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Dedicated RNG for agent position randomization
    rng = random.Random(seed_offset)

    records     = []
    skipped     = 0
    n_invalid   = 0
    sample_id   = 0
    global_seed = seed_offset   # continuous — never reset between envs

    per_env   = n_samples // len(ENV_POOL)
    remainder = n_samples  % len(ENV_POOL)

    pbar = tqdm(total=n_samples, desc="Generating", unit="sample")

    for env_idx, (env_name, complexity) in enumerate(ENV_POOL):
        n_this    = per_env + (1 if env_idx < remainder else 0)
        generated = 0
        attempts  = 0

        while generated < n_this:
            attempts += 1
            if attempts > n_this * 30:
                warnings.warn(f"Too many failed attempts for {env_name}, moving on.")
                break

            try:
                data = capture(env_name, global_seed, rng)
            except Exception as e:
                global_seed += 1
                skipped     += 1
                continue

            oracle_actions = data["oracle_actions"]   # list of equally optimal actions
            agent_pos      = data["agent_pos"]
            agent_dir      = data["agent_dir"]

            #  Oracle validation — valid if ANY optimal action passes 
            is_valid = any(
                validate_oracle(env_name, global_seed, agent_pos, agent_dir, a)
                for a in oracle_actions
            )
            if not is_valid:
                n_invalid += 1

            # Save images 
            sid          = f"{sample_id:05d}"
            global_path  = img_dir / f"{sid}_global.png"
            partial_path = img_dir / f"{sid}_partial.png"
            Image.fromarray(data["global_rgb"]).save(global_path)
            Image.fromarray(data["partial_rgb"]).save(partial_path)

            records.append({
                "id":              sid,
                "env":             env_name,
                "seed":            global_seed,
                "complexity":      complexity,
                "mission":         data["mission"],
                "agent_pos":       agent_pos,
                "agent_dir":       agent_dir,
                "agent_dir_str":   data["agent_dir_str"],
                "global_image":    f"images/{sid}_global.png",
                "partial_image":   f"images/{sid}_partial.png",
                # Oracle — only optimal_actions (list), no single oracle_action
                "optimal_actions": oracle_actions,
                "action_names":    [ACTION_NAMES[a] for a in oracle_actions],
                "oracle_info":     data["oracle_info"],
                "agent_carrying":  data["agent_carrying"],
                "oracle_valid":    is_valid,
            })

            sample_id   += 1
            generated   += 1
            global_seed += 1
            pbar.update(1)

    pbar.close()

    # Save metadata
    json_path = out / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    try:
        import pandas as pd
        df = pd.DataFrame(records)
        df.to_parquet(out / "dataset.parquet", index=False)
        print(f"   Parquet  → {out / 'dataset.parquet'}")
    except ImportError:
        pass

    # Summary
    n_total = len(records)
    n_valid = sum(r["oracle_valid"] for r in records)

    print(f"\n{'─'*58}")
    print(f"{n_total} samples saved  →  {out}")
    print(f"Oracle validated : {n_valid}/{n_total} "
          f"({100*n_valid/max(n_total,1):.1f}%)")
    print(f"Oracle suspect  : {n_invalid}  (oracle_valid=false, kept in dataset)")
    print(f"Skipped          : {skipped}")
    print(f"{'─'*58}")

    action_counts = Counter(a for r in records for a in r["action_names"])
    max_c = max(action_counts.values(), default=1)
    print("\n  Oracle action distribution:")
    for name, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
        bar = "█" * max(1, cnt * 28 // max_c)
        print(f"    {name:13s} {bar} {cnt}")

    print("\n  Agent starting positions (first 10 samples):")
    for r in records[:10]:
        print(f"    [{r['id']}] {r['env']:42s}  "
              f"pos={r['agent_pos']}  dir={r['agent_dir_str']}")

    print(f"\n  Metadata → {json_path}\n")

# CLI

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a MiniGrid dataset for VLM benchmarking"
    )
    p.add_argument("--n",           type=int, default=200,
                   help="Total number of samples (default: 200)")
    p.add_argument("--out",         type=str, default="./dataset",
                   help="Output directory (default: ./dataset)")
    p.add_argument("--seed_offset", type=int, default=0,
                   help="Seed offset — use to shard generation across Izar nodes")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(n_samples=args.n, out_dir=args.out, seed_offset=args.seed_offset)