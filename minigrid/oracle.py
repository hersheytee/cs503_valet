"""
BFS Oracle for MiniGrid environments.
Computes the optimal action given full knowledge of the environment.

Actions: 0=turn_left, 1=turn_right, 2=move_forward, 3=pickup, 4=drop, 5=toggle
Directions: 0=right, 1=down, 2=left, 3=up
"""

from collections import deque

# Direction vectors: (dx, dy)
DIR_TO_VEC = {
    0: (1, 0),   # right
    1: (0, 1),   # down
    2: (-1, 0),  # left
    3: (0, -1),  # up
}

ACTION_LEFT     = 0
ACTION_RIGHT    = 1
ACTION_FORWARD  = 2
ACTION_PICKUP   = 3
ACTION_TOGGLE   = 5


def _turn_left(d):
    return (d - 1) % 4

def _turn_right(d):
    return (d + 1) % 4

def _front_pos(pos, direction):
    dx, dy = DIR_TO_VEC[direction]
    return (pos[0] + dx, pos[1] + dy)


def bfs_empty(env_unwrapped):
    """
    BFS for MiniGrid-Empty.
    State: (x, y, direction)
    Goal: reach the 'goal' cell.
    Returns the first action to take, or None if already at goal.
    """
    grid   = env_unwrapped.grid
    width  = env_unwrapped.width
    height = env_unwrapped.height

    # Find goal position
    goal_pos = None
    for x in range(width):
        for y in range(height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == 'goal':
                goal_pos = (x, y)
                break

    assert goal_pos is not None, "No goal found in grid"

    start_pos = tuple(int(v) for v in env_unwrapped.agent_pos)
    start_dir = int(env_unwrapped.agent_dir)
    start     = (start_pos[0], start_pos[1], start_dir)

    if (start_pos[0], start_pos[1]) == goal_pos:
        return None  # already at goal

    # BFS: state = (x, y, direction)
    queue   = deque()
    visited = set()
    # Store (state, first_action)
    queue.append((start, None))
    visited.add(start)

    while queue:
        state, first_action = queue.popleft()
        x, y, d = state

        # Generate successors
        for action in [ACTION_LEFT, ACTION_RIGHT, ACTION_FORWARD]:
            if action == ACTION_LEFT:
                new_d = _turn_left(d)
                new_x, new_y = x, y
            elif action == ACTION_RIGHT:
                new_d = _turn_right(d)
                new_x, new_y = x, y
            else:  # FORWARD
                fx, fy = _front_pos((x, y), d)
                cell = grid.get(fx, fy)
                # Can move forward if cell is empty or goal
                if cell is not None and cell.type not in ('empty', 'goal'):
                    continue
                if not (0 <= fx < width and 0 <= fy < height):
                    continue
                new_x, new_y, new_d = fx, fy, d

            new_state = (new_x, new_y, new_d)
            if new_state in visited:
                continue
            visited.add(new_state)

            fa = action if first_action is None else first_action

            if (new_x, new_y) == goal_pos:
                return fa

            queue.append((new_state, fa))

    return None  # no path found


def bfs_doorkey(env_unwrapped):
    """
    BFS for MiniGrid-DoorKey.
    State: (x, y, direction, has_key, door_open)
    Goal: reach the 'goal' cell.
    Returns the first action to take, or None if already at goal.
    """
    grid   = env_unwrapped.grid
    width  = env_unwrapped.width
    height = env_unwrapped.height

    # Find key, door and goal positions
    key_pos  = None
    door_pos = None
    goal_pos = None

    for x in range(width):
        for y in range(height):
            cell = grid.get(x, y)
            if cell is None:
                continue
            if cell.type == 'key':
                key_pos = (x, y)
            elif cell.type == 'door':
                door_pos = (x, y)
            elif cell.type == 'goal':
                goal_pos = (x, y)

    assert goal_pos is not None, "No goal found"
    assert door_pos is not None, "No door found"

    start_pos = tuple(int(v) for v in env_unwrapped.agent_pos)
    start_dir = int(env_unwrapped.agent_dir)
    has_key   = env_unwrapped.carrying is not None
    # Check if door is already open
    door_cell = grid.get(*door_pos)
    door_open = door_cell is None or door_cell.is_open if door_cell else True

    start = (start_pos[0], start_pos[1], start_dir, has_key, door_open)

    if (start_pos[0], start_pos[1]) == goal_pos:
        return None

    queue   = deque()
    visited = set()
    queue.append((start, None))
    visited.add(start)

    while queue:
        state, first_action = queue.popleft()
        x, y, d, hk, do = state

        for action in [ACTION_LEFT, ACTION_RIGHT, ACTION_FORWARD, ACTION_PICKUP, ACTION_TOGGLE]:
            new_x, new_y, new_d = x, y, d
            new_hk, new_do = hk, do

            if action == ACTION_LEFT:
                new_d = _turn_left(d)

            elif action == ACTION_RIGHT:
                new_d = _turn_right(d)

            elif action == ACTION_FORWARD:
                fx, fy = _front_pos((x, y), d)
                if not (0 <= fx < width and 0 <= fy < height):
                    continue
                cell = grid.get(fx, fy)
                # Blocked by wall
                if cell is not None and cell.type == 'wall':
                    continue
                # Blocked by closed door
                if cell is not None and cell.type == 'door' and not new_do:
                    continue
                # Key is picked up when we move onto its cell (already handled by pickup)
                new_x, new_y = fx, fy

            elif action == ACTION_PICKUP:
                # Can only pick up key if in front and not already carrying
                if hk:
                    continue
                fx, fy = _front_pos((x, y), d)
                if key_pos is None or (fx, fy) != key_pos:
                    continue
                new_hk = True

            elif action == ACTION_TOGGLE:
                # Can only toggle door if in front, has key, door is closed
                if not hk or do:
                    continue
                fx, fy = _front_pos((x, y), d)
                if (fx, fy) != door_pos:
                    continue
                new_do = True

            new_state = (new_x, new_y, new_d, new_hk, new_do)
            if new_state in visited:
                continue
            visited.add(new_state)

            fa = action if first_action is None else first_action

            if (new_x, new_y) == goal_pos:
                return fa

            queue.append((new_state, fa))

    return None  # no path found


def get_oracle_action(env_unwrapped, env_type='empty'):
    """
    Main entry point.
    env_type: 'empty' or 'doorkey'
    Returns the optimal action index (int), or 6 (done) if no path.
    """
    if env_type == 'empty':
        action = bfs_empty(env_unwrapped)
    elif env_type == 'doorkey':
        action = bfs_doorkey(env_unwrapped)
    else:
        raise ValueError(f"Unknown env_type: {env_type}")

    if action is None:
        return 6  # done action
    return action


# ── Multi-action BFS (two-phase) ─────────────────────────────────────────────
# Returns ALL equally-optimal first actions (there can be several, e.g. two
# symmetric 180° pivots). Used to evaluate VLM correctness fairly.

def _successors_empty(state, grid, width, height):
    x, y, d = state
    succs = [
        (ACTION_LEFT,  (x, y, (d - 1) % 4)),
        (ACTION_RIGHT, (x, y, (d + 1) % 4)),
    ]
    fx, fy = _front_pos((x, y), d)
    if 0 <= fx < width and 0 <= fy < height:
        cell = grid.get(fx, fy)
        if cell is None or cell.type in ('empty', 'goal', 'floor'):
            succs.append((ACTION_FORWARD, (fx, fy, d)))
    return succs


def _bfs_len_empty(start, goal_pos, grid, width, height):
    if (start[0], start[1]) == goal_pos:
        return 0
    queue   = deque([(start, 0)])
    visited = {start}
    while queue:
        state, dist = queue.popleft()
        for _, nxt in _successors_empty(state, grid, width, height):
            if nxt in visited:
                continue
            if (nxt[0], nxt[1]) == goal_pos:
                return dist + 1
            visited.add(nxt)
            queue.append((nxt, dist + 1))
    return None


def bfs_all_empty(env_unwrapped):
    grid   = env_unwrapped.grid
    width  = env_unwrapped.width
    height = env_unwrapped.height

    goal_pos = None
    for x in range(width):
        for y in range(height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == 'goal':
                goal_pos = (x, y)
                break
        if goal_pos:
            break
    if goal_pos is None:
        return []

    start_pos = tuple(int(v) for v in env_unwrapped.agent_pos)
    start     = (start_pos[0], start_pos[1], int(env_unwrapped.agent_dir))

    if (start[0], start[1]) == goal_pos:
        return []

    best_len = _bfs_len_empty(start, goal_pos, grid, width, height)
    if best_len is None or best_len == 0:
        return []

    best_actions = []
    for action, nxt in _successors_empty(start, grid, width, height):
        rem = _bfs_len_empty(nxt, goal_pos, grid, width, height)
        if rem is not None and rem == best_len - 1:
            best_actions.append(action)
    return sorted(best_actions)


def _successors_doorkey(state, grid, width, height, key_pos, door_pos):
    x, y, d, hk, do = state
    succs = [
        (ACTION_LEFT,  (x, y, (d - 1) % 4, hk, do)),
        (ACTION_RIGHT, (x, y, (d + 1) % 4, hk, do)),
    ]
    fx, fy = _front_pos((x, y), d)
    if 0 <= fx < width and 0 <= fy < height:
        cell    = grid.get(fx, fy)
        blocked = cell is not None and (
            cell.type == 'wall' or (cell.type == 'door' and not do)
        )
        if not blocked:
            succs.append((ACTION_FORWARD, (fx, fy, d, hk, do)))
        if not hk and key_pos is not None and (fx, fy) == key_pos:
            succs.append((ACTION_PICKUP, (x, y, d, True, do)))
        if hk and not do and door_pos is not None and (fx, fy) == door_pos:
            succs.append((ACTION_TOGGLE, (x, y, d, hk, True)))
    return succs


def _bfs_len_doorkey(start, goal_pos, grid, width, height, key_pos, door_pos):
    if (start[0], start[1]) == goal_pos:
        return 0
    queue   = deque([(start, 0)])
    visited = {start}
    while queue:
        state, dist = queue.popleft()
        for _, nxt in _successors_doorkey(state, grid, width, height, key_pos, door_pos):
            if nxt in visited:
                continue
            if (nxt[0], nxt[1]) == goal_pos:
                return dist + 1
            visited.add(nxt)
            queue.append((nxt, dist + 1))
    return None


def bfs_all_doorkey(env_unwrapped):
    grid   = env_unwrapped.grid
    width  = env_unwrapped.width
    height = env_unwrapped.height

    key_pos = door_pos = goal_pos = None
    for x in range(width):
        for y in range(height):
            cell = grid.get(x, y)
            if cell is None:
                continue
            if cell.type == 'key':
                key_pos = (x, y)
            elif cell.type == 'door':
                door_pos = (x, y)
            elif cell.type == 'goal':
                goal_pos = (x, y)
    if goal_pos is None:
        return []

    start_pos = tuple(int(v) for v in env_unwrapped.agent_pos)
    has_key   = env_unwrapped.carrying is not None
    door_cell = grid.get(*door_pos) if door_pos else None
    door_open = (door_cell is None or door_cell.is_open) if door_cell else True

    start = (start_pos[0], start_pos[1], int(env_unwrapped.agent_dir), has_key, door_open)

    if (start[0], start[1]) == goal_pos:
        return []

    best_len = _bfs_len_doorkey(start, goal_pos, grid, width, height, key_pos, door_pos)
    if best_len is None or best_len == 0:
        return []

    best_actions = []
    for action, nxt in _successors_doorkey(start, grid, width, height, key_pos, door_pos):
        rem = _bfs_len_doorkey(nxt, goal_pos, grid, width, height, key_pos, door_pos)
        if rem is not None and rem == best_len - 1:
            best_actions.append(action)
    return sorted(best_actions)


def get_all_oracle_actions(env_unwrapped, env_type='empty'):
    """
    Returns a list of all equally-optimal first actions (may contain several).
    Used to evaluate VLM correctness: vlm_action in get_all_oracle_actions(...).
    """
    if env_type == 'empty':
        return bfs_all_empty(env_unwrapped)
    elif env_type == 'doorkey':
        return bfs_all_doorkey(env_unwrapped)
    else:
        raise ValueError(f"Unknown env_type: {env_type}")