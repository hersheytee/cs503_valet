"""
BFS Oracle for gym-sokoban environments.
Finds the guaranteed shortest path by brute-forcing the state space.
"""

import numpy as np
from collections import deque
import time

DIRECTIONS = {
    'up':    (-1, 0),
    'down':  (1, 0),
    'left':  (0, -1),
    'right': (0, 1)
}

def get_action_id(direction_name, is_push):
    action_base = {'up': 1, 'down': 2, 'left': 3, 'right': 4}
    act = action_base[direction_name]
    if not is_push:
        act += 4 
    return act

def is_deadlock(box, walls, targets):
    if box in targets:
        return False
    
    y, x = box
    up_wall    = (y - 1, x) in walls
    down_wall  = (y + 1, x) in walls
    left_wall  = (y, x - 1) in walls
    right_wall = (y, x + 1) in walls
    
    if (up_wall or down_wall) and (left_wall or right_wall):
        return True
    return False

def extract_state(env_unwrapped):
    # FIXED: 0 is Wall, 1 is Empty. 
    walls = set(zip(*np.where(env_unwrapped.room_fixed == 0)))
    targets = frozenset(zip(*np.where(env_unwrapped.room_fixed == 2)))
    
    boxes = frozenset(zip(*np.where((env_unwrapped.room_state == 3) | (env_unwrapped.room_state == 4))))
    p_loc = np.where((env_unwrapped.room_state == 5) | (env_unwrapped.room_state == 6))
    
    if len(p_loc[0]) == 0:
        return None, boxes, walls, targets 
        
    player = (int(p_loc[0][0]), int(p_loc[1][0]))
    return player, boxes, walls, targets

def bfs_sokoban(env_unwrapped, debug=False):
    start_player, start_boxes, walls, targets = extract_state(env_unwrapped)
    
    if start_player is None or start_boxes == targets:
        return [] 
        
    start_state = (start_player, start_boxes)
    # Queue stores: (current_state, path_to_get_here)
    # UPDATED: path now stores tuples of (action, resulting_state)
    queue = deque([(start_state, [])])
    visited = set()
    visited.add(start_state)
    
    states_explored = 0
    
    max_y = env_unwrapped.room_fixed.shape[0]
    max_x = env_unwrapped.room_fixed.shape[1]
    
    while queue:
        current_state, path = queue.popleft()
        player, boxes = current_state
        states_explored += 1
        
        if boxes == targets:
            if debug: print(f"  [Oracle] BFS Solved! Path length: {len(path)}")
            return path
            
        py, px = player
        
        for dir_name, (dy, dx) in DIRECTIONS.items():
            ny, nx = py + dy, px + dx
            n_pos = (ny, nx)
            
            if ny < 0 or ny >= max_y or nx < 0 or nx >= max_x:
                continue
            if n_pos in walls:
                continue
                
            new_boxes = boxes
            is_push = False
            
            if n_pos in boxes:
                bny, bnx = ny + dy, nx + dx
                bn_pos = (bny, bnx)
                
                if bny < 0 or bny >= max_y or bnx < 0 or bnx >= max_x:
                    continue
                if bn_pos in walls or bn_pos in boxes:
                    continue
                if is_deadlock(bn_pos, walls, targets):
                    continue
                    
                new_boxes_list = list(boxes)
                new_boxes_list.remove(n_pos)
                new_boxes_list.append(bn_pos)
                new_boxes = frozenset(new_boxes_list)
                is_push = True
                
            new_state = (n_pos, new_boxes)
            
            if new_state not in visited:
                visited.add(new_state)
                action = get_action_id(dir_name, is_push)
                # UPDATED: Store the action AND the state it leads to
                queue.append((new_state, path + [(action, new_state)]))
                
    if debug: print(f"  [Oracle Error] Unsolvable state reached.")
    return None 

def get_oracle_action(env_unwrapped):
    """
    State-aware Oracle. If the agent deviates from the path, the cache breaks
    and it recomputes a new optimal path.
    """
    if is_fixed_target_env(env_unwrapped):
        return _get_oracle_action_fixed_target(env_unwrapped)
    return _get_oracle_action_standard(env_unwrapped)


def _get_oracle_action_standard(env_unwrapped):
    current_state = extract_state(env_unwrapped)
    if current_state[0] is None:
        return 0

    state_key = (current_state[0], current_state[1])

    if hasattr(env_unwrapped, '_oracle_cache') and getattr(env_unwrapped, '_expected_state', None) == state_key:
        if env_unwrapped._oracle_cache:
            action, next_state_key = env_unwrapped._oracle_cache.pop(0)
            env_unwrapped._expected_state = next_state_key
            return action

    path_data = bfs_sokoban(env_unwrapped, debug=False)
    if path_data and len(path_data) > 0:
        action, next_state_key = path_data.pop(0)
        env_unwrapped._oracle_cache = path_data
        env_unwrapped._expected_state = next_state_key
        return action

    return 0


# ── Fixed-target BFS ──────────────────────────────────────────────────────────

def is_fixed_target_env(env_unwrapped) -> bool:
    return hasattr(env_unwrapped, 'box_mapping')


def _extract_ft_state(env_unwrapped):
    """Return (player, assignment) where assignment = frozenset of (target, box) pairs."""
    box_mapping = env_unwrapped.box_mapping
    assignment = frozenset(box_mapping.items())
    p_loc = np.where((env_unwrapped.room_state == 5) | (env_unwrapped.room_state == 6))
    if len(p_loc[0]) == 0:
        return None, assignment
    player = (int(p_loc[0][0]), int(p_loc[1][0]))
    return player, assignment


def bfs_sokoban_fixed_target(env_unwrapped, debug=False):
    """BFS for FixedTarget Sokoban: each box must reach its paired target."""
    walls = set(zip(*np.where(env_unwrapped.room_fixed == 0)))
    targets_set = frozenset(env_unwrapped.box_mapping.keys())

    start_player, start_assignment = _extract_ft_state(env_unwrapped)
    if start_player is None:
        return []
    if all(t == b for t, b in start_assignment):
        return []

    max_y, max_x = env_unwrapped.room_fixed.shape

    start_state = (start_player, start_assignment)
    queue = deque([(start_state, [])])
    visited = {start_state}

    while queue:
        (player, assignment), path = queue.popleft()
        py, px = player
        box_positions = {b for _, b in assignment}

        for dir_name, (dy, dx) in DIRECTIONS.items():
            ny, nx = py + dy, px + dx
            n_pos = (ny, nx)

            if ny < 0 or ny >= max_y or nx < 0 or nx >= max_x:
                continue
            if n_pos in walls:
                continue

            new_assignment = assignment
            is_push = False

            if n_pos in box_positions:
                bny, bnx = ny + dy, nx + dx
                bn_pos = (bny, bnx)
                if bny < 0 or bny >= max_y or bnx < 0 or bnx >= max_x:
                    continue
                if bn_pos in walls or bn_pos in box_positions:
                    continue
                if is_deadlock(bn_pos, walls, targets_set):
                    continue
                new_assignment = frozenset(
                    (t, bn_pos) if b == n_pos else (t, b)
                    for t, b in assignment
                )
                is_push = True

            new_state = (n_pos, new_assignment)
            if new_state in visited:
                continue
            visited.add(new_state)

            if all(t == b for t, b in new_assignment):
                if debug:
                    print(f"  [FT Oracle] Solved! Path length: {len(path) + 1}")
                action = get_action_id(dir_name, is_push)
                return path + [(action, new_state)]

            action = get_action_id(dir_name, is_push)
            queue.append((new_state, path + [(action, new_state)]))

    if debug:
        print("  [FT Oracle] Unsolvable state.")
    return None


def _get_oracle_action_fixed_target(env_unwrapped):
    start_player, assignment = _extract_ft_state(env_unwrapped)
    if start_player is None:
        return 0

    state_key = (start_player, assignment)

    if hasattr(env_unwrapped, '_ft_oracle_cache') and getattr(env_unwrapped, '_ft_expected_state', None) == state_key:
        if env_unwrapped._ft_oracle_cache:
            action, next_state_key = env_unwrapped._ft_oracle_cache.pop(0)
            env_unwrapped._ft_expected_state = next_state_key
            return action

    path_data = bfs_sokoban_fixed_target(env_unwrapped, debug=False)
    if path_data and len(path_data) > 0:
        action, next_state_key = path_data.pop(0)
        env_unwrapped._ft_oracle_cache = path_data
        env_unwrapped._ft_expected_state = next_state_key
        return action

    return 0