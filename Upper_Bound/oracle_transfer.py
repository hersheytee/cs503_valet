"""
BFS oracles for transfer environments (same 8x8 obs as DoorKey).

Supported env_types:
    'fetch'      — MiniGrid-Fetch-8x8-N3-v0   : pick up target object
    'gotodoor'   — MiniGrid-GoToDoor-8x8-v0   : go adjacent to target door
    'gotoobject' — MiniGrid-GoToObject-8x8-N2-v0 : go adjacent to target object
"""

from collections import deque

DIR_TO_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

ACTION_LEFT    = 0
ACTION_RIGHT   = 1
ACTION_FORWARD = 2
ACTION_PICKUP  = 3
ACTION_TOGGLE  = 5

COLORS = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
TYPES  = ['ball', 'box', 'key', 'door']


def _turn_left(d):  return (d - 1) % 4
def _turn_right(d): return (d + 1) % 4

def _front_pos(pos, d):
    dx, dy = DIR_TO_VEC[d]
    return (pos[0] + dx, pos[1] + dy)


def _parse_mission(mission):
    """Extract (color, type) from mission string."""
    mission = mission.lower()
    color = next((c for c in COLORS if c in mission), None)
    obj   = next((t for t in TYPES  if t in mission), None)
    return color, obj


def _find_target(grid, width, height, color, obj_type):
    """Find position of target object in grid."""
    for x in range(width):
        for y in range(height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == obj_type and cell.color == color:
                return (x, y)
    return None


def _can_walk(grid, x, y, width, height):
    """True if agent can walk onto (x, y)."""
    if not (0 <= x < width and 0 <= y < height):
        return False
    cell = grid.get(x, y)
    if cell is None:
        return True
    return cell.type in ('empty', 'goal', 'floor')


# ── Fetch oracle ──────────────────────────────────────────────────────────────

def bfs_fetch(env_unwrapped):
    """
    Navigate to the target object and pick it up.
    State: (x, y, direction, has_object)
    Goal:  has_object = True
    """
    if env_unwrapped.carrying is not None:
        return None  # already done

    grid   = env_unwrapped.grid
    W, H   = env_unwrapped.width, env_unwrapped.height
    color, obj_type = _parse_mission(env_unwrapped.mission)
    target = _find_target(grid, W, H, color, obj_type)

    if target is None:
        return ACTION_FORWARD  # fallback

    start_pos = tuple(int(v) for v in env_unwrapped.agent_pos)
    start_dir = int(env_unwrapped.agent_dir)
    start     = (*start_pos, start_dir, False)

    queue   = deque([(start, None)])
    visited = {start}

    while queue:
        (x, y, d, ho), fa = queue.popleft()

        for action in [ACTION_LEFT, ACTION_RIGHT, ACTION_FORWARD, ACTION_PICKUP]:
            nx, ny, nd, nho = x, y, d, ho

            if action == ACTION_LEFT:
                nd = _turn_left(d)

            elif action == ACTION_RIGHT:
                nd = _turn_right(d)

            elif action == ACTION_FORWARD:
                fx, fy = _front_pos((x, y), d)
                if not _can_walk(grid, fx, fy, W, H):
                    continue
                nx, ny = fx, fy

            elif action == ACTION_PICKUP:
                fx, fy = _front_pos((x, y), d)
                if (fx, fy) != target:
                    continue
                nho = True

            ns = (nx, ny, nd, nho)
            if ns in visited:
                continue
            visited.add(ns)
            nfa = action if fa is None else fa

            if nho:
                return nfa

            queue.append((ns, nfa))

    return ACTION_FORWARD


# ── GoToDoor oracle ───────────────────────────────────────────────────────────

def bfs_gotodoor(env_unwrapped):
    """
    Navigate to be adjacent to the target door.
    State: (x, y, direction)
    Goal:  Chebyshev distance ≤ 1 from target door
    """
    grid   = env_unwrapped.grid
    W, H   = env_unwrapped.width, env_unwrapped.height
    color, _ = _parse_mission(env_unwrapped.mission)
    target   = _find_target(grid, W, H, color, 'door')

    if target is None:
        return ACTION_FORWARD

    start_pos = tuple(int(v) for v in env_unwrapped.agent_pos)
    start_dir = int(env_unwrapped.agent_dir)

    # Already adjacent?
    if max(abs(start_pos[0] - target[0]), abs(start_pos[1] - target[1])) <= 1:
        return None

    start = (*start_pos, start_dir)
    queue   = deque([(start, None)])
    visited = {start}

    while queue:
        (x, y, d), fa = queue.popleft()

        for action in [ACTION_LEFT, ACTION_RIGHT, ACTION_FORWARD]:
            nx, ny, nd = x, y, d

            if action == ACTION_LEFT:
                nd = _turn_left(d)
            elif action == ACTION_RIGHT:
                nd = _turn_right(d)
            elif action == ACTION_FORWARD:
                fx, fy = _front_pos((x, y), d)
                if not _can_walk(grid, fx, fy, W, H):
                    continue
                nx, ny = fx, fy

            ns = (nx, ny, nd)
            if ns in visited:
                continue
            visited.add(ns)
            nfa = action if fa is None else fa

            if max(abs(nx - target[0]), abs(ny - target[1])) <= 1:
                return nfa

            queue.append((ns, nfa))

    return ACTION_FORWARD


# ── GoToObject oracle ─────────────────────────────────────────────────────────

def bfs_gotoobject(env_unwrapped):
    """
    Navigate to be adjacent to the target object.
    State: (x, y, direction)
    Goal:  Chebyshev distance ≤ 1 from target
    """
    grid   = env_unwrapped.grid
    W, H   = env_unwrapped.width, env_unwrapped.height
    color, obj_type = _parse_mission(env_unwrapped.mission)
    target = _find_target(grid, W, H, color, obj_type)

    if target is None:
        return ACTION_FORWARD

    start_pos = tuple(int(v) for v in env_unwrapped.agent_pos)
    start_dir = int(env_unwrapped.agent_dir)

    if max(abs(start_pos[0] - target[0]), abs(start_pos[1] - target[1])) <= 1:
        return None

    start   = (*start_pos, start_dir)
    queue   = deque([(start, None)])
    visited = {start}

    while queue:
        (x, y, d), fa = queue.popleft()

        for action in [ACTION_LEFT, ACTION_RIGHT, ACTION_FORWARD]:
            nx, ny, nd = x, y, d

            if action == ACTION_LEFT:
                nd = _turn_left(d)
            elif action == ACTION_RIGHT:
                nd = _turn_right(d)
            elif action == ACTION_FORWARD:
                fx, fy = _front_pos((x, y), d)
                if not _can_walk(grid, fx, fy, W, H):
                    continue
                nx, ny = fx, fy

            ns = (nx, ny, nd)
            if ns in visited:
                continue
            visited.add(ns)
            nfa = action if fa is None else fa

            if max(abs(nx - target[0]), abs(ny - target[1])) <= 1:
                return nfa

            queue.append((ns, nfa))

    return ACTION_FORWARD


# ── Entry point ───────────────────────────────────────────────────────────────

def get_oracle_action(env_unwrapped, env_type):
    if env_type == 'fetch':
        action = bfs_fetch(env_unwrapped)
    elif env_type == 'gotodoor':
        action = bfs_gotodoor(env_unwrapped)
    elif env_type == 'gotoobject':
        action = bfs_gotoobject(env_unwrapped)
    else:
        raise ValueError(f"Unknown env_type: {env_type}")

    return 6 if action is None else action  # 6 = done


# ── Multi-action BFS (two-phase) ─────────────────────────────────────────────

def _successors_fetch(state, grid, W, H, target):
    x, y, d, ho = state
    succs = [
        (ACTION_LEFT,  (x, y, (d - 1) % 4, ho)),
        (ACTION_RIGHT, (x, y, (d + 1) % 4, ho)),
    ]
    fx, fy = _front_pos((x, y), d)
    if _can_walk(grid, fx, fy, W, H):
        succs.append((ACTION_FORWARD, (fx, fy, d, ho)))
    if not ho and target is not None and (fx, fy) == target:
        succs.append((ACTION_PICKUP, (x, y, d, True)))
    return succs


def _bfs_len_fetch(start, grid, W, H, target):
    if start[3]:  # has_object
        return 0
    queue   = deque([(start, 0)])
    visited = {start}
    while queue:
        state, dist = queue.popleft()
        for _, nxt in _successors_fetch(state, grid, W, H, target):
            if nxt in visited:
                continue
            if nxt[3]:
                return dist + 1
            visited.add(nxt)
            queue.append((nxt, dist + 1))
    return None


def bfs_all_fetch(env_unwrapped):
    if env_unwrapped.carrying is not None:
        return []
    grid   = env_unwrapped.grid
    W, H   = env_unwrapped.width, env_unwrapped.height
    color, obj_type = _parse_mission(env_unwrapped.mission)
    target = _find_target(grid, W, H, color, obj_type)
    if target is None:
        return []

    start_pos = tuple(int(v) for v in env_unwrapped.agent_pos)
    start     = (*start_pos, int(env_unwrapped.agent_dir), False)

    best_len = _bfs_len_fetch(start, grid, W, H, target)
    if best_len is None or best_len == 0:
        return []

    best_actions = []
    for action, nxt in _successors_fetch(start, grid, W, H, target):
        rem = _bfs_len_fetch(nxt, grid, W, H, target)
        if rem is not None and rem == best_len - 1:
            best_actions.append(action)
    return sorted(best_actions)


def _successors_nav(state, grid, W, H):
    x, y, d = state
    succs = [
        (ACTION_LEFT,  (x, y, (d - 1) % 4)),
        (ACTION_RIGHT, (x, y, (d + 1) % 4)),
    ]
    fx, fy = _front_pos((x, y), d)
    if _can_walk(grid, fx, fy, W, H):
        succs.append((ACTION_FORWARD, (fx, fy, d)))
    return succs


def _bfs_len_nav(start, grid, W, H, target):
    if max(abs(start[0] - target[0]), abs(start[1] - target[1])) <= 1:
        return 0
    queue   = deque([(start, 0)])
    visited = {start}
    while queue:
        state, dist = queue.popleft()
        for _, nxt in _successors_nav(state, grid, W, H):
            if nxt in visited:
                continue
            if max(abs(nxt[0] - target[0]), abs(nxt[1] - target[1])) <= 1:
                return dist + 1
            visited.add(nxt)
            queue.append((nxt, dist + 1))
    return None


def _bfs_all_nav(env_unwrapped, target):
    if target is None:
        return []
    grid  = env_unwrapped.grid
    W, H  = env_unwrapped.width, env_unwrapped.height

    start_pos = tuple(int(v) for v in env_unwrapped.agent_pos)
    if max(abs(start_pos[0] - target[0]), abs(start_pos[1] - target[1])) <= 1:
        return []

    start    = (*start_pos, int(env_unwrapped.agent_dir))
    best_len = _bfs_len_nav(start, grid, W, H, target)
    if best_len is None or best_len == 0:
        return []

    best_actions = []
    for action, nxt in _successors_nav(start, grid, W, H):
        rem = _bfs_len_nav(nxt, grid, W, H, target)
        if rem is not None and rem == best_len - 1:
            best_actions.append(action)
    return sorted(best_actions)


def get_all_oracle_actions(env_unwrapped, env_type):
    """
    Returns a list of all equally-optimal first actions (may contain several).
    Used to evaluate VLM correctness: vlm_action in get_all_oracle_actions(...).
    """
    if env_type == 'fetch':
        return bfs_all_fetch(env_unwrapped)
    elif env_type == 'gotodoor':
        grid   = env_unwrapped.grid
        W, H   = env_unwrapped.width, env_unwrapped.height
        color, _ = _parse_mission(env_unwrapped.mission)
        target   = _find_target(grid, W, H, color, 'door')
        return _bfs_all_nav(env_unwrapped, target)
    elif env_type == 'gotoobject':
        grid   = env_unwrapped.grid
        W, H   = env_unwrapped.width, env_unwrapped.height
        color, obj_type = _parse_mission(env_unwrapped.mission)
        target = _find_target(grid, W, H, color, obj_type)
        return _bfs_all_nav(env_unwrapped, target)
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
