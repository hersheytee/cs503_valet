"""
BFS oracles for transfer environments (same 8x8 obs as DoorKey).

Supported env_types:
    'fetch'      — MiniGrid-Fetch-8x8-N3-v0   : pick up target object
    'gotodoor'   — MiniGrid-GoToDoor-8x8-v0   : go adjacent to target door
    'gotoobject' — MiniGrid-GoToObject-8x8-N2-v0 : go adjacent to target object
    'multiroom'  — MiniGrid-MultiRoom-N6-v0   : navigate to goal through doors

Each BFS returns (first_action, optimal_steps).
  first_action = None  → already at goal (optimal_steps = 0)
  first_action = int   → first step of optimal path
Use get_oracle_action()  to get just the action.
Use get_optimal_steps()  to get just the path length.
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
    Returns: (first_action, optimal_steps)
    """
    if env_unwrapped.carrying is not None:
        return (None, 0)

    grid   = env_unwrapped.grid
    W, H   = env_unwrapped.width, env_unwrapped.height
    color, obj_type = _parse_mission(env_unwrapped.mission)
    target = _find_target(grid, W, H, color, obj_type)

    if target is None:
        return (ACTION_FORWARD, -1)

    start_pos = tuple(int(v) for v in env_unwrapped.agent_pos)
    start_dir = int(env_unwrapped.agent_dir)
    start     = (*start_pos, start_dir, False)

    queue   = deque([(start, None, 0)])
    visited = {start}

    while queue:
        (x, y, d, ho), fa, depth = queue.popleft()

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
                return (nfa, depth + 1)

            queue.append((ns, nfa, depth + 1))

    return (ACTION_FORWARD, -1)


# ── GoToDoor oracle ───────────────────────────────────────────────────────────

def bfs_gotodoor(env_unwrapped):
    """
    Navigate to be adjacent to the target door.
    State: (x, y, direction)
    Returns: (first_action, optimal_steps)
    """
    grid   = env_unwrapped.grid
    W, H   = env_unwrapped.width, env_unwrapped.height
    color, _ = _parse_mission(env_unwrapped.mission)
    target   = _find_target(grid, W, H, color, 'door')

    if target is None:
        return (ACTION_FORWARD, -1)

    start_pos = tuple(int(v) for v in env_unwrapped.agent_pos)
    start_dir = int(env_unwrapped.agent_dir)

    if max(abs(start_pos[0] - target[0]), abs(start_pos[1] - target[1])) <= 1:
        return (None, 0)

    start = (*start_pos, start_dir)
    queue   = deque([(start, None, 0)])
    visited = {start}

    while queue:
        (x, y, d), fa, depth = queue.popleft()

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
                return (nfa, depth + 1)

            queue.append((ns, nfa, depth + 1))

    return (ACTION_FORWARD, -1)


# ── GoToObject oracle ─────────────────────────────────────────────────────────

def bfs_gotoobject(env_unwrapped):
    """
    Navigate to be adjacent to the target object.
    State: (x, y, direction)
    Returns: (first_action, optimal_steps)
    """
    grid   = env_unwrapped.grid
    W, H   = env_unwrapped.width, env_unwrapped.height
    color, obj_type = _parse_mission(env_unwrapped.mission)
    target = _find_target(grid, W, H, color, obj_type)

    if target is None:
        return (ACTION_FORWARD, -1)

    start_pos = tuple(int(v) for v in env_unwrapped.agent_pos)
    start_dir = int(env_unwrapped.agent_dir)

    if max(abs(start_pos[0] - target[0]), abs(start_pos[1] - target[1])) <= 1:
        return (None, 0)

    start   = (*start_pos, start_dir)
    queue   = deque([(start, None, 0)])
    visited = {start}

    while queue:
        (x, y, d), fa, depth = queue.popleft()

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
                return (nfa, depth + 1)

            queue.append((ns, nfa, depth + 1))

    return (ACTION_FORWARD, -1)


# ── MultiRoom oracle ──────────────────────────────────────────────────────────

def bfs_multiroom(env_unwrapped):
    """
    Navigate from start to goal by toggling doors through N rooms.
    State: (x, y, direction, frozenset_of_open_door_positions)
    Returns: (first_action, optimal_steps)
    """
    grid = env_unwrapped.grid
    W, H = env_unwrapped.width, env_unwrapped.height

    goal_pos = None
    for x in range(W):
        for y in range(H):
            cell = grid.get(x, y)
            if cell is not None and cell.type == 'goal':
                goal_pos = (x, y)
                break
        if goal_pos:
            break

    if goal_pos is None:
        return (ACTION_FORWARD, -1)

    start_pos = tuple(int(v) for v in env_unwrapped.agent_pos)
    start_dir = int(env_unwrapped.agent_dir)

    if start_pos == goal_pos:
        return (None, 0)

    initial_open = frozenset(
        (x, y) for x in range(W) for y in range(H)
        if grid.get(x, y) is not None
        and grid.get(x, y).type == 'door'
        and grid.get(x, y).is_open
    )

    start = (*start_pos, start_dir, initial_open)
    queue = deque([(start, None, 0)])
    visited = {start}

    while queue:
        (x, y, d, open_doors), fa, depth = queue.popleft()

        for action in [ACTION_LEFT, ACTION_RIGHT, ACTION_FORWARD, ACTION_TOGGLE]:
            nx, ny, nd = x, y, d
            n_open = open_doors

            if action == ACTION_LEFT:
                nd = _turn_left(d)

            elif action == ACTION_RIGHT:
                nd = _turn_right(d)

            elif action == ACTION_FORWARD:
                fx, fy = _front_pos((x, y), d)
                if not (0 <= fx < W and 0 <= fy < H):
                    continue
                cell = grid.get(fx, fy)
                if cell is None or cell.type in ('empty', 'floor', 'goal'):
                    pass
                elif cell.type == 'door' and (fx, fy) in open_doors:
                    pass
                else:
                    continue
                nx, ny = fx, fy

            elif action == ACTION_TOGGLE:
                fx, fy = _front_pos((x, y), d)
                if not (0 <= fx < W and 0 <= fy < H):
                    continue
                cell = grid.get(fx, fy)
                if cell is None or cell.type != 'door':
                    continue
                if (fx, fy) in open_doors:
                    continue
                n_open = open_doors | frozenset([(fx, fy)])

            ns = (nx, ny, nd, n_open)
            if ns in visited:
                continue
            visited.add(ns)
            nfa = action if fa is None else fa

            if (nx, ny) == goal_pos:
                return (nfa, depth + 1)

            queue.append((ns, nfa, depth + 1))

    return (ACTION_FORWARD, -1)


# ── Entry points ──────────────────────────────────────────────────────────────

def _dispatch(env_unwrapped, env_type):
    if env_type == 'fetch':
        return bfs_fetch(env_unwrapped)
    elif env_type == 'gotodoor':
        return bfs_gotodoor(env_unwrapped)
    elif env_type == 'gotoobject':
        return bfs_gotoobject(env_unwrapped)
    elif env_type == 'multiroom':
        return bfs_multiroom(env_unwrapped)
    else:
        raise ValueError(f"Unknown env_type: {env_type}")


def get_oracle_action(env_unwrapped, env_type):
    action, _ = _dispatch(env_unwrapped, env_type)
    return 6 if action is None else action  # 6 = done


def get_optimal_steps(env_unwrapped, env_type):
    """Return the BFS-optimal number of steps from current state. -1 if unreachable."""
    _, steps = _dispatch(env_unwrapped, env_type)
    return steps
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
