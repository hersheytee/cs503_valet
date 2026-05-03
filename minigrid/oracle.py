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