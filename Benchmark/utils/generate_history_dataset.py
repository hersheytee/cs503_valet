"""
Generates N samples where each sample includes the current state + history_len
preceding states, with a configurable mix of optimal and suboptimal histories.
Rare-action samples (pickup / toggle) are constructed explicitly so their
frequency is controlled via n_rare, exactly like in the original dataset.

  Optimal history   : the history_len states that immediately precede the
                      current state on the BFS shortest path.
  Suboptimal history: the agent takes history_len random steps from
                      {turn_left, turn_right, forward}; the optimal action
                      from the resulting state is computed fresh by BFS.
  Rare-action history (pickup / toggle current state):
                      the target situation is constructed explicitly (agent
                      placed in front of key / door), then a BFS-to-target
                      finds the history_len preceding states on an optimal
                      approach path.  Always labelled history_type="optimal".

action_sequence field (flat list at root level):
  [a_{-N}, a_{-N+1}, ..., a_{-1}]
  where a_{-k} is the action taken FROM state step=-k TO the next state.
  a_{-1} is therefore the action that produced the current state.

Dataset JSON structure per entry:
  {
    "id":              "00042",
    "env":             "MiniGrid-DoorKey-8x8-v0",
    "seed":            42,
    "complexity":      "medium",
    "mission":         "...",
    "agent_pos":       [3, 4],
    "agent_dir":       1,
    "agent_dir_str":   "↓",
    "agent_carrying":  null,
    "global_image":    "images/00042_global.png",
    "partial_image":   "images/00042_partial.png",
    "optimal_actions": [2],
    "action_names":    ["forward"],
    "oracle_info":     "remaining=6",
    "oracle_valid":    true,
    "history_type":    "optimal",
    "action_sequence": [2, 0, 2, 2, 1],   <- flat list of actions in history
    "history": [
      {
        "step":           -5,
        "agent_pos":      [1, 2],
        "agent_dir":      0,
        "agent_dir_str":  "→",
        "agent_carrying": null,
        "global_image":   "images/00042_h5_global.png",
        "partial_image":  "images/00042_h5_partial.png",
        "action_taken":   2,
        "action_name":    "forward"
      },
      ...
    ]
  }
"""

import json
import random
import re
import warnings
from collections import Counter, deque
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import gymnasium as gym
import minigrid  # noqa
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper

from utils.generate_dataset import (
    ACTION_NAMES,
    DIR_STR,
    ENV_POOL,
    TILE_SIZE,
    _get_successors,
    bfs_oracle,
    randomize_agent,
)
from utils.sample_rare_actions import (
    DOORKEY_ENVS,
    _goal_side_cells,
    make_pickup_situation,
    make_toggle_situation,
)

_RANDOM_ACTIONS = [0, 1, 2]   # turn_left, turn_right, forward


# ── BFS helpers ──────────────────────────────────────────────────────────────

def bfs_full_path(env):
    """
    Returns one shortest path from the current agent state to the goal.

    path[0] = (None,     start_state)
    path[i] = (action_i, state_i)   — action_i was taken from path[i-1]

    state = (pos, dir, has_key, door_open)
    Returns (path, info_str); path is [] on failure.
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
            if cell.type == "goal":   goal_pos = (x, y)
            elif cell.type == "key":  key_pos  = (x, y)
            elif cell.type == "door": door_pos = (x, y)

    if goal_pos is None:
        return [], "no goal found"

    has_key0   = raw.carrying is not None and raw.carrying.type == "key"
    door_open0 = (door_pos is None) or (
        grid.get(*door_pos) is not None and grid.get(*door_pos).is_open
    )

    start = (pos0, dir0, has_key0, door_open0)
    if start[0] == goal_pos:
        return [(None, start)], "path_len=0"

    parent = {start: (None, None)}
    queue  = deque([start])
    found  = None

    while queue and found is None:
        state = queue.popleft()
        for action, next_state in _get_successors(state, grid, W, H, key_pos, door_pos):
            if next_state not in parent:
                parent[next_state] = (state, action)
                if next_state[0] == goal_pos:
                    found = next_state
                    break
                queue.append(next_state)

    if found is None:
        return [], "no path found"

    path = []
    cur  = found
    while cur is not None:
        par_state, act = parent[cur]
        path.append((act, cur))
        cur = par_state
    path.reverse()
    return path, f"path_len={len(path) - 1}"


def _bfs_to_target(start_state, target_state, grid, W, H,
                   key_pos, door_pos, goal_pos):
    """
    BFS from start_state to target_state (never stepping onto goal_pos).
    Returns the full path [(action, state), ...] or [] if unreachable.
    Used to build the history leading up to an explicitly constructed
    pickup / toggle current state.
    """
    if start_state == target_state:
        return [(None, start_state)]

    parent = {start_state: (None, None)}
    queue  = deque([start_state])
    found  = None

    while queue and found is None:
        state = queue.popleft()
        for action, next_state in _get_successors(state, grid, W, H, key_pos, door_pos):
            if next_state in parent:
                continue
            if next_state[0] == goal_pos:
                continue   # don't pass through the terminal goal cell
            parent[next_state] = (state, action)
            if next_state == target_state:
                found = next_state
                break
            queue.append(next_state)

    if found is None:
        return []

    path = []
    cur  = found
    while cur is not None:
        par_state, act = parent[cur]
        path.append((act, cur))
        cur = par_state
    path.reverse()
    return path


# ── Live environment state ────────────────────────────────────────────────────

def _get_env_state(env):
    """Returns (pos, dir, has_key, door_open) from a live environment."""
    raw  = env.unwrapped
    grid = raw.grid
    W, H = grid.width, grid.height

    pos     = tuple(map(int, raw.agent_pos))
    dir_    = int(raw.agent_dir)
    has_key = raw.carrying is not None and raw.carrying.type == "key"

    door_open = True
    for x in range(W):
        for y in range(H):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "door":
                door_open = cell.is_open
                break

    return pos, dir_, has_key, door_open


# ── Render a specific environment state ──────────────────────────────────────

def _render_state(env_name: str, seed: int, pos, dir_: int,
                  has_key: bool, door_open: bool):
    """
    Renders global and partial views for a specific (agent, world) state.
    Reconstructs the state by patching agent_pos/dir, key carrying, door open.
    """
    def _configure(raw, grid):
        raw.agent_pos = np.array(pos)
        raw.agent_dir = dir_
        if has_key:
            for x in range(grid.width):
                for y in range(grid.height):
                    cell = grid.get(x, y)
                    if cell is not None and cell.type == "key":
                        raw.carrying = cell
                        grid.set(x, y, None)
                        break
        if door_open:
            for x in range(grid.width):
                for y in range(grid.height):
                    cell = grid.get(x, y)
                    if cell is not None and cell.type == "door":
                        cell.is_open   = True
                        cell.is_locked = False
                        break

    env_g = gym.make(env_name, render_mode="rgb_array", tile_size=TILE_SIZE)
    env_g = FullyObsWrapper(env_g)
    env_g = RGBImgObsWrapper(env_g, tile_size=TILE_SIZE)
    env_g.reset(seed=seed)
    _configure(env_g.unwrapped, env_g.unwrapped.grid)
    global_rgb = env_g.render()
    env_g.close()

    env_p = gym.make(env_name, render_mode="rgb_array", tile_size=TILE_SIZE)
    env_p = RGBImgPartialObsWrapper(env_p, tile_size=TILE_SIZE)
    env_p.reset(seed=seed)
    _configure(env_p.unwrapped, env_p.unwrapped.grid)
    obs         = env_p.observation(env_p.unwrapped.gen_obs())
    partial_rgb = obs["image"]
    env_p.close()

    return global_rgb, partial_rgb


# ── Capture helpers ───────────────────────────────────────────────────────────

def _extract_history(path, current_idx, history_len):
    """
    Extracts history_len entries from path ending just before current_idx.
    Returns list of dicts with step/pos/dir/has_key/door_open/action_taken.
    """
    history = []
    for i in range(current_idx - history_len, current_idx):
        _, h_state  = path[i]
        h_action    = path[i + 1][0]
        h_pos, h_dir, h_has_key, h_door_open = h_state
        history.append({
            "step":         i - current_idx,
            "pos":          list(map(int, h_pos)),
            "dir":          int(h_dir),
            "has_key":      bool(h_has_key),
            "door_open":    bool(h_door_open),
            "action_taken": int(h_action),
        })
    return history


def capture_optimal_history(env_name, seed, rng, history_len=5):
    """
    Places the agent randomly, finds the BFS optimal path, picks a random
    current index with >= history_len predecessors and >= 1 step to goal.
    history_type = "optimal"
    """
    env_g = gym.make(env_name, render_mode="rgb_array", tile_size=TILE_SIZE)
    env_g = FullyObsWrapper(env_g)
    env_g = RGBImgObsWrapper(env_g, tile_size=TILE_SIZE)
    env_g.reset(seed=seed)
    randomize_agent(env_g, rng)
    mission = env_g.unwrapped.mission

    path, oracle_info = bfs_full_path(env_g)
    env_g.close()

    if not path:
        raise ValueError(f"BFS failed: {oracle_info}")

    path_len = len(path) - 1
    if path_len - 1 < history_len:
        raise ValueError(f"Path too short ({path_len} steps) for history_len={history_len}")

    current_idx = rng.randint(history_len, path_len - 1)
    _, current_state = path[current_idx]
    cur_pos, cur_dir, cur_has_key, cur_door_open = current_state

    return {
        "env":             env_name,
        "seed":            seed,
        "mission":         mission,
        "remaining_steps": path_len - current_idx,
        "agent_pos":       list(map(int, cur_pos)),
        "agent_dir":       int(cur_dir),
        "has_key":         bool(cur_has_key),
        "door_open":       bool(cur_door_open),
        "optimal_action":  int(path[current_idx + 1][0]),
        "history":         _extract_history(path, current_idx, history_len),
        "history_type":    "optimal",
    }


def capture_suboptimal_history(env_name, seed, rng, history_len=5):
    """
    Places the agent randomly, takes history_len random steps from
    {turn_left, turn_right, forward}, then computes the optimal action
    from the resulting state via BFS.
    history_type = "suboptimal"
    """
    env = gym.make(env_name, render_mode=None, tile_size=TILE_SIZE)
    env = FullyObsWrapper(env)
    env.reset(seed=seed)
    randomize_agent(env, rng)
    mission = env.unwrapped.mission
    history = []

    for i in range(history_len):
        pos, dir_, has_key, door_open = _get_env_state(env)
        action = rng.choice(_RANDOM_ACTIONS)
        history.append({
            "step":         i - history_len,
            "pos":          list(pos),
            "dir":          dir_,
            "has_key":      has_key,
            "door_open":    door_open,
            "action_taken": action,
        })
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            env.close()
            raise ValueError("Agent reached goal during suboptimal history")

    cur_pos, cur_dir, cur_has_key, cur_door_open = _get_env_state(env)
    optimal_actions, oracle_info = bfs_oracle(env)
    env.close()

    if not optimal_actions:
        raise ValueError(f"No BFS solution from suboptimal end state: {oracle_info}")

    m = re.search(r"path_len=(\d+)", oracle_info)
    remaining = int(m.group(1)) if m else None

    return {
        "env":             env_name,
        "seed":            seed,
        "mission":         mission,
        "remaining_steps": remaining,
        "agent_pos":       list(cur_pos),
        "agent_dir":       cur_dir,
        "has_key":         cur_has_key,
        "door_open":       cur_door_open,
        "optimal_action":  optimal_actions[0],
        "history":         history,
        "history_type":    "suboptimal",
    }


def capture_rare_with_history(env_name, seed, rng, history_len=5,
                               action_type="pickup"):
    """
    Constructs an explicit pickup or toggle current state (agent placed
    directly in front of the key / door), then finds history_len preceding
    states on an optimal approach path using BFS-to-target.

    history_type = "optimal"  (the history leading to the situation is optimal)
    """
    assert action_type in ("pickup", "toggle")

    # --- Construct the target situation ---
    if action_type == "pickup":
        agent_pos, agent_dir = make_pickup_situation(env_name, seed)
        optimal_action  = 3   # pickup
        target_has_key  = False
        target_door_open = False
    else:
        agent_pos, agent_dir = make_toggle_situation(env_name, seed)
        optimal_action   = 5  # toggle
        target_has_key   = True    # agent already has the key
        target_door_open = False

    target_state = (tuple(map(int, agent_pos)), int(agent_dir),
                    target_has_key, target_door_open)

    # --- Load grid once to run BFS-to-target ---
    env = gym.make(env_name, render_mode=None, tile_size=TILE_SIZE)
    env = FullyObsWrapper(env)
    env.reset(seed=seed)
    raw  = env.unwrapped
    grid = raw.grid
    W, H = grid.width, grid.height
    mission = raw.mission

    goal_pos = key_pos = door_pos = None
    for x in range(W):
        for y in range(H):
            cell = grid.get(x, y)
            if cell is None:
                continue
            if cell.type == "goal":   goal_pos = (x, y)
            elif cell.type == "key":  key_pos  = (x, y)
            elif cell.type == "door": door_pos = (x, y)

    # Free cells on the agent side (key side)
    goal_side      = _goal_side_cells(grid, goal_pos, W, H)
    agent_side_free = [
        (x, y) for x in range(W) for y in range(H)
        if (grid.get(x, y) is None or grid.get(x, y).type == "floor")
        and (x, y) not in goal_side
    ] or [
        (x, y) for x in range(W) for y in range(H)
        if grid.get(x, y) is None or grid.get(x, y).type == "floor"
    ]

    env.close()

    # --- Find a starting state whose BFS-to-target path is long enough ---
    history_path = []
    for _ in range(100):
        start_pos   = rng.choice(agent_side_free)
        start_dir   = rng.randint(0, 3)
        start_state = (start_pos, start_dir, False, False)

        if start_state == target_state:
            continue

        path = _bfs_to_target(
            start_state, target_state,
            grid, W, H, key_pos, door_pos, goal_pos
        )
        if len(path) >= history_len + 1:
            history_path = path
            break

    if not history_path:
        raise ValueError(
            f"Could not find an approach path of length >= {history_len} "
            f"to {action_type} state for {env_name} seed={seed}"
        )

    # --- Extract history from the last history_len steps of the approach path ---
    current_idx = len(history_path) - 1   # index of target_state in path
    history     = _extract_history(history_path, current_idx, history_len)

    return {
        "env":             env_name,
        "seed":            seed,
        "mission":         mission,
        "remaining_steps": None,   # not meaningful here; agent is 1 action away from key/door
        "agent_pos":       list(map(int, agent_pos)),
        "agent_dir":       int(agent_dir),
        "has_key":         target_has_key,
        "door_open":       target_door_open,
        "optimal_action":  optimal_action,
        "history":         history,
        "history_type":    "optimal",
    }


# ── Image + record builder (shared between main and rare loops) ───────────────

def _save_and_build(data, env_name, seed, sid, img_dir, complexity):
    """
    Renders all images for one sample, saves them, and returns the JSON record.
    """
    # Current state
    g_rgb, p_rgb = _render_state(
        env_name, seed,
        data["agent_pos"], data["agent_dir"],
        data["has_key"], data["door_open"],
    )
    Image.fromarray(g_rgb).save(img_dir / f"{sid}_global.png")
    Image.fromarray(p_rgb).save(img_dir / f"{sid}_partial.png")

    # History
    history_records = []
    for h in data["history"]:
        step_num = abs(h["step"])
        hg_rgb, hp_rgb = _render_state(
            env_name, seed,
            h["pos"], h["dir"],
            h["has_key"], h["door_open"],
        )
        Image.fromarray(hg_rgb).save(img_dir / f"{sid}_h{step_num}_global.png")
        Image.fromarray(hp_rgb).save(img_dir / f"{sid}_h{step_num}_partial.png")

        history_records.append({
            "step":           h["step"],
            "agent_pos":      h["pos"],
            "agent_dir":      h["dir"],
            "agent_dir_str":  DIR_STR[h["dir"]],
            "agent_carrying": "key" if h["has_key"] else None,
            "global_image":   f"images/{sid}_h{step_num}_global.png",
            "partial_image":  f"images/{sid}_h{step_num}_partial.png",
            "action_taken":   h["action_taken"],
            "action_name":    ACTION_NAMES[h["action_taken"]],
        })

    optimal_action = data["optimal_action"]
    remaining      = data["remaining_steps"]

    return {
        "id":              sid,
        "env":             env_name,
        "seed":            seed,
        "complexity":      complexity,
        "mission":         data["mission"],
        "agent_pos":       data["agent_pos"],
        "agent_dir":       data["agent_dir"],
        "agent_dir_str":   DIR_STR[data["agent_dir"]],
        "agent_carrying":  "key" if data["has_key"] else None,
        "door_open":       data["door_open"],
        "global_image":    f"images/{sid}_global.png",
        "partial_image":   f"images/{sid}_partial.png",
        "optimal_actions": [optimal_action],
        "action_names":    [ACTION_NAMES[optimal_action]],
        "oracle_info":     f"remaining={remaining}",
        "oracle_valid":    True,
        "history_type":    data["history_type"],
        "action_sequence": [h["action_taken"] for h in data["history"]],
        "history":         history_records,
    }


# ── Main generation function ──────────────────────────────────────────────────

def generate_with_history(n_samples: int, out_dir: str, seed_offset: int,
                           history_len: int = 5, p_suboptimal: float = 0.5,
                           n_rare: int = 50):
    """
    Generates n_samples + 2*n_rare samples total:
      - n_samples main samples  (mix of optimal/suboptimal history)
      - n_rare pickup samples   (explicit construction, optimal history)
      - n_rare toggle samples   (explicit construction, optimal history)

    p_suboptimal : fraction of main samples with suboptimal history.
    n_rare       : number of rare-action samples PER action type (0 = skip).
    """
    out     = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed_offset)

    records     = []
    skipped     = 0
    n_short     = 0
    sample_id   = 0
    global_seed = seed_offset

    per_env   = n_samples // len(ENV_POOL)
    remainder = n_samples  % len(ENV_POOL)

    pbar = tqdm(total=n_samples, desc="Main samples", unit="sample")

    # ── Main samples ─────────────────────────────────────────────────────────
    for env_idx, (env_name, complexity) in enumerate(ENV_POOL):
        n_this    = per_env + (1 if env_idx < remainder else 0)
        generated = 0
        attempts  = 0

        while generated < n_this:
            attempts += 1
            if attempts > n_this * 40:
                warnings.warn(f"Too many failed attempts for {env_name}, moving on.")
                break

            use_suboptimal = rng.random() < p_suboptimal
            capture_fn     = capture_suboptimal_history if use_suboptimal \
                             else capture_optimal_history

            try:
                data = capture_fn(env_name, global_seed, rng, history_len)
            except ValueError as e:
                if "too short" in str(e).lower():
                    n_short += 1
                global_seed += 1
                skipped     += 1
                continue
            except Exception:
                global_seed += 1
                skipped     += 1
                continue

            sid    = f"{sample_id:05d}"
            record = _save_and_build(data, env_name, global_seed, sid,
                                     img_dir, complexity)
            records.append(record)

            sample_id   += 1
            generated   += 1
            global_seed += 1
            pbar.update(1)

    pbar.close()

    # ── Rare-action samples (pickup + toggle) ─────────────────────────────────
    if n_rare > 0:
        rare_seed = seed_offset + 10_000   # separate seed space
        rare_rng  = random.Random(seed_offset + 77_777)

        tasks = [
            ("pickup", 3),
            ("toggle", 5),
        ]

        for action_name, action_int in tasks:
            print(f"\n  Generating {n_rare} × {action_name} samples with history…")
            generated = 0
            attempts  = 0
            env_rng   = random.Random(seed_offset + action_int * 1_000)

            pbar_rare = tqdm(total=n_rare, desc=f"  {action_name}", unit="sample")

            while generated < n_rare:
                attempts += 1
                if attempts > n_rare * 40:
                    warnings.warn(f"Too many attempts for rare {action_name}.")
                    break

                env_name, complexity = env_rng.choice(DOORKEY_ENVS)

                try:
                    data = capture_rare_with_history(
                        env_name, rare_seed, rare_rng,
                        history_len, action_name
                    )
                except Exception:
                    rare_seed += 1
                    skipped   += 1
                    continue

                sid    = f"{sample_id:05d}"
                record = _save_and_build(data, env_name, rare_seed, sid,
                                         img_dir, complexity)
                records.append(record)

                sample_id += 1
                generated += 1
                rare_seed += 1
                pbar_rare.update(1)

            pbar_rare.close()

    # ── Save metadata ─────────────────────────────────────────────────────────
    json_path = out / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    try:
        import pandas as pd
        flat = [{k: v for k, v in r.items() if k != "history"} for r in records]
        pd.DataFrame(flat).to_parquet(out / "dataset.parquet", index=False)
        print(f"   Parquet  → {out / 'dataset.parquet'}")
    except ImportError:
        pass

    # ── Summary ───────────────────────────────────────────────────────────────
    n_total = len(records)
    n_opt   = sum(1 for r in records if r["history_type"] == "optimal")
    n_sub   = sum(1 for r in records if r["history_type"] == "suboptimal")
    action_counts = Counter(r["action_names"][0] for r in records)
    max_c = max(action_counts.values(), default=1)

    print(f"\n{'─'*58}")
    print(f"{n_total} samples saved  →  {out}")
    print(f"  Main samples       : {n_samples}")
    print(f"  Rare pickup        : {sum(1 for r in records if r['action_names'] == ['pickup'])}")
    print(f"  Rare toggle        : {sum(1 for r in records if r['action_names'] == ['toggle'])}")
    print(f"  Optimal history    : {n_opt}  ({100*n_opt/max(n_total,1):.0f}%)")
    print(f"  Suboptimal history : {n_sub}  ({100*n_sub/max(n_total,1):.0f}%)")
    print(f"Skipped total  : {skipped}  (path-too-short: {n_short})")
    print(f"History length : {history_len} steps  |  action_sequence length : {history_len}")
    print(f"{'─'*58}")
    print("\n  Current-state oracle action distribution:")
    for name, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
        bar = "█" * max(1, cnt * 28 // max_c)
        print(f"    {name:13s} {bar} {cnt}")
    print(f"\n  Metadata → {json_path}\n")
