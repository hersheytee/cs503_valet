"""
A* Oracle for gym-sokoban environments.
Computes the optimal action sequence to push all boxes onto targets.

Action Map (gym-sokoban):
0: noop
1: push up    | 5: move up
2: push down  | 6: move down
3: push left  | 7: move left
4: push right | 8: move right
"""

import numpy as np
import heapq

# Directions: (dy, dx) matching (row, col) numpy indexing
DIRECTIONS = {
    'up':    (-1, 0),
    'down':  (1, 0),
    'left':  (0, -1),
    'right': (0, 1)
}

def get_action_id(direction_name, is_push):
    """Maps a direction and interaction type to gym-sokoban action ints."""
    action_base = {'up': 1, 'down': 2, 'left': 3, 'right': 4}
    act = action_base[direction_name]
    if not is_push:
        act += 4 # move actions are offset by 4
    return act

def heuristic(boxes, targets):
    """
    Computes the heuristic: sum of Manhattan distances from each box 
    to its nearest unassigned target. (Admissible for small grids).
    """
    total_h = 0
    for b in boxes:
        # Minimum distance from this box to any target
        total_h += min(abs(b[0] - t[0]) + abs(b[1] - t[1]) for t in targets)
    return total_h

def is_deadlock(box, walls, targets):
    """
    Corner deadlock detection. 
    If a box is against two orthogonal walls and is NOT on a target, it is stuck forever.
    """
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
    """Parses the gym-sokoban room state."""
    # room_fixed: 1 = wall, 2 = target
    # room_state: 3 = box_on_target, 4 = box, 5 = player
    
    walls = set(zip(*np.where(env_unwrapped.room_fixed == 1)))
    targets = frozenset(zip(*np.where(env_unwrapped.room_fixed == 2)))
    
    boxes_off = set(zip(*np.where(env_unwrapped.room_state == 4)))
    boxes_on  = set(zip(*np.where(env_unwrapped.room_state == 3)))
    boxes = frozenset(boxes_off | boxes_on)
    
    # Player position
    py, px = np.where(env_unwrapped.room_state == 5)
    player = (int(py[0]), int(px[0]))
    
    return player, boxes, walls, targets

def astar_sokoban(env_unwrapped):
    """
    Main A* Search Algorithm.
    Returns a list of actions to solve the current state.
    """
    start_player, start_boxes, walls, targets = extract_state(env_unwrapped)
    
    if start_boxes == targets:
        return [] # Already solved
        
    # Priority Queue: (f_score, g_score, (player_pos, boxes), path_of_actions)
    start_state = (start_player, start_boxes)
    queue = [(heuristic(start_boxes, targets), 0, start_state, [])]
    
    visited = set()
    visited.add(start_state)
    
    while queue:
        _, g_score, current_state, path = heapq.heappop(queue)
        player, boxes = current_state
        
        if boxes == targets:
            return path
            
        py, px = player
        
        for dir_name, (dy, dx) in DIRECTIONS.items():
            ny, nx = py + dy, px + dx
            n_pos = (ny, nx)
            
            # Hit a wall
            if n_pos in walls:
                continue
                
            new_boxes = boxes
            is_push = False
            
            # Hit a box
            if n_pos in boxes:
                # Where would the box go?
                bny, bnx = ny + dy, nx + dx
                bn_pos = (bny, bnx)
                
                # Can't push into a wall or another box
                if bn_pos in walls or bn_pos in boxes:
                    continue
                
                # Check for deadlocks immediately after pushing
                if is_deadlock(bn_pos, walls, targets):
                    continue
                    
                # Valid push! Update boxes
                new_boxes_list = list(boxes)
                new_boxes_list.remove(n_pos)
                new_boxes_list.append(bn_pos)
                new_boxes = frozenset(new_boxes_list)
                is_push = True
                
            new_state = (n_pos, new_boxes)
            
            if new_state not in visited:
                visited.add(new_state)
                action = get_action_id(dir_name, is_push)
                new_path = path + [action]
                f_score = g_score + 1 + heuristic(new_boxes, targets)
                
                heapq.heappush(queue, (f_score, g_score + 1, new_state, new_path))
                
    return None # No path found (unsolvable state)

def get_oracle_action(env_unwrapped):
    """
    Wrapper function to get the next optimal action.
    Caches the path to avoid running A* on every single step.
    """
    # 1. Check if we have a valid cached path
    if hasattr(env_unwrapped, '_oracle_path') and env_unwrapped._oracle_path:
        # Pop the next action
        return env_unwrapped._oracle_path.pop(0)
        
    # 2. No valid cache, compute full A* path
    print("  [Oracle] Computing A* Path... (this might take a second)")
    path = astar_sokoban(env_unwrapped)
    
    if path and len(path) > 0:
        env_unwrapped._oracle_path = path # Cache the rest of the path
        return env_unwrapped._oracle_path.pop(0)
        
    # 3. If A* fails (agent pushed a box into an unrecoverable state)
    # We return 0 (noop) or 6 (done) depending on how you want to handle it.
    return 0