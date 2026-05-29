"""
Environment wrapper for gym-sokoban.
Translates old Gym v0.21 environments into modern Gymnasium v0.28+ environments.
Adds the `query_oracle` action and resizes RGB images to 56x56 by default
for 7x7 boards rendered at 8 pixels per cell.
"""

import gym as old_gym          # The old library where Sokoban lives
import gymnasium as gym        # The modern library your RL agent needs
from gymnasium import spaces
import numpy as np
import cv2 
import gym_sokoban 
from bfs_oracle import get_oracle_action

# --- MONKEYPATCH FOR NUMPY 2.0 vs OLD GYM ---
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
# --------------------------------------------

class SokobanRewardShaper:
    """
    Potential-based reward shaper for Sokoban.
    Rewards the agent for reducing the Manhattan distance between boxes and targets.
    Penalizes pushing boxes away from targets.
    """
    def __init__(self, distance_scale=0.1):
        self.distance_scale = distance_scale
        self._previous_box_dist = None

    def _get_state_coordinates(self, env_unwrapped):
        # 3 and 4 are boxes (4 is a box on a target)
        boxes = np.argwhere((env_unwrapped.room_state == 3) | (env_unwrapped.room_state == 4))
        # 2 is a target
        targets = np.argwhere(env_unwrapped.room_fixed == 2)
        return boxes, targets

    def _calculate_total_box_distance(self, boxes, targets):
        """Calculates the sum of distances from each box to its *nearest* target."""
        if len(boxes) == 0 or len(targets) == 0:
            return 0.0
            
        total_dist = 0.0
        for b in boxes:
            # Manhattan distance from this box to all targets
            dists = np.sum(np.abs(targets - b), axis=1)
            total_dist += np.min(dists) # Only care about the closest target
        return total_dist

    def reset(self, env_unwrapped):
        boxes, targets = self._get_state_coordinates(env_unwrapped)
        self._previous_box_dist = self._calculate_total_box_distance(boxes, targets)

    def shape(self, env_unwrapped, base_reward):
        bonus = 0.0
        boxes, targets = self._get_state_coordinates(env_unwrapped)
        current_box_dist = self._calculate_total_box_distance(boxes, targets)

        # Potential-based shaping: Reward the difference in state potential
        if self._previous_box_dist is not None:
            dist_diff = self._previous_box_dist - current_box_dist
            # If dist_diff > 0, boxes got closer (+ bonus)
            # If dist_diff < 0, boxes got further (- penalty)
            bonus += dist_diff * self.distance_scale

        self._previous_box_dist = current_box_dist

        # Note: gym-sokoban natively gives +1 for pushing a box onto a target
        # and -1 for pushing it off. We leave that base_reward intact.
        return base_reward + bonus

class SokobanOracleWrapper(gym.Env):
    """
    A modern Gymnasium environment that wraps the old Gym Sokoban environment.
    """
    metadata = {'render_modes': ['rgb_array']}

    def __init__(
        self,
        env_id: str = 'Sokoban-small-v0',
        oracle_cost: float = 0.0,
        oracle_accuracy: float = 1.0,
        reward_shaping: bool = False,
        no_oracle: bool = False,
        max_episode_steps: int = 120,
        obs_size: int = 56,
        seed: int = 0,
    ):
        # Build the environment using the OLD gym
        self.env = old_gym.make(env_id)
        self.oracle_cost = oracle_cost
        self.oracle_accuracy = float(np.clip(oracle_accuracy, 0.0, 1.0))
        self.max_episode_steps = max_episode_steps
        self.elapsed_steps = 0
        self._oracle_rng = np.random.default_rng(seed)
        
        # include no oracle flag
        self.no_oracle = no_oracle

        # Initialize the reward shaper if enabled
        self.reward_shaping = reward_shaping
        self._shaper = SokobanRewardShaper(distance_scale=0.1) if reward_shaping else None
        
        # Extend action space: 9 original actions + 1 oracle action
        n_original = 9
        self.n_original_actions = n_original

        if self.no_oracle:
            self.action_space = spaces.Discrete(n_original)
            self.QUERY_ACTION = -1  # Kill the Oracle index entirely
        else:
            self.action_space = spaces.Discrete(n_original + 1)
            self.QUERY_ACTION = n_original
        
        # Resize observation space from the raw render to MiniGrid-large size.
        self.obs_size = obs_size
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(self.obs_size, self.obs_size, 3), 
            dtype=np.uint8
        )

    def _process_obs(self):
        """Grabs the raw RGB image from the old env and resizes it."""
        obs = self.env.render(mode='rgb_array')
        resized = cv2.resize(obs, (self.obs_size, self.obs_size), interpolation=cv2.INTER_AREA)
        return resized

    def _is_solved(self):
        """Return True only when every box is currently on a target square."""
        unwrapped = self.env.unwrapped
        if not hasattr(unwrapped, 'room_state') or not hasattr(unwrapped, 'room_fixed'):
            return False

        room_state = np.asarray(unwrapped.room_state)
        room_fixed = np.asarray(unwrapped.room_fixed)
        box_mask = (room_state == 3) | (room_state == 4)

        if not np.any(box_mask):
            return False

        return bool(np.all(room_fixed[box_mask] == 2))

    def reset(self, seed=None, options=None):
        # Old gym doesn't always handle seeds cleanly in reset()
        if seed is not None:
            self.env.seed(seed)
            self._oracle_rng = np.random.default_rng(seed)
            
        self.env.reset()
        self.elapsed_steps = 0
        
        # Clear the state-aware cache
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, '_oracle_cache'): unwrapped._oracle_cache = None
        if hasattr(unwrapped, '_expected_state'): unwrapped._expected_state = None
        
        # Reset the reward shaper so it can calculate the initial board potential
        if self._shaper:
            self._shaper.reset(unwrapped)
            
        # Return modern format: obs, info
        return self._process_obs(), {}

    def step(self, action: int):
        guided = False
        oracle_action = None

        if action == self.QUERY_ACTION:
            unwrapped = self.env.unwrapped
            optimal_action = get_oracle_action(unwrapped)
            guided = True
            
            # Fatal deadlock
            if optimal_action == 0:
                self.env.reset() 
                info = {
                    'guided': True,
                    'oracle_action': 0,
                    'oracle_optimal_action': 0,
                    'oracle_correct': False,
                    'oracle_accuracy': self.oracle_accuracy,
                    'oracle_cost': self.oracle_cost,
                    'fatal_deadlock': True,
                    'success': False,
                }
                # Return modern format: obs, reward, terminated, truncated, info
                return self._process_obs(), -1.0, True, False, info

            if self._oracle_rng.random() <= self.oracle_accuracy:
                oracle_action = optimal_action
                oracle_correct = True
            else:
                non_optimal_actions = [
                    a for a in range(self.n_original_actions)
                    if a != optimal_action
                ]
                oracle_action = int(self._oracle_rng.choice(non_optimal_actions))
                oracle_correct = False
                
            action = oracle_action
        else:
            optimal_action = None
            oracle_correct = False

        # Take the step in the old environment (which returns 4 values)
        _, reward, done, info = self.env.step(action)
        self.elapsed_steps += 1
        base_reward = reward

        # Apply potential-based reward shaping
        if self._shaper:
            reward = self._shaper.shape(self.env.unwrapped, reward)

        if guided and not info.get('fatal_deadlock', False):
            reward -= self.oracle_cost

        info['guided'] = guided
        info['oracle_action'] = int(oracle_action) if oracle_action is not None else -1
        info['oracle_optimal_action'] = int(optimal_action) if optimal_action is not None else -1
        info['oracle_correct'] = bool(oracle_correct)
        info['oracle_accuracy'] = self.oracle_accuracy
        info['oracle_cost'] = self.oracle_cost
        info['success'] = bool(done and self._is_solved())

        # Convert old 'done' to modern 'terminated' and 'truncated'
        terminated = bool(done)
        truncated = bool(not terminated and self.elapsed_steps >= self.max_episode_steps)

        return self._process_obs(), reward, terminated, truncated, info

    def render(self):
        return self.env.render(mode='rgb_array')

    def close(self):
        self.env.close()


def make_env(
    env_id: str,
    oracle_cost: float = 0.0,
    oracle_accuracy: float = 1.0,
    seed: int = 0,
    reward_shaping: bool = False,
    no_oracle: bool = False,
    max_episode_steps: int = 120,
    obs_size: int = 56,
):
    """Factory function for SyncVectorEnv."""
    def _init():
        env = SokobanOracleWrapper(
            env_id,
            oracle_cost,
            oracle_accuracy=oracle_accuracy,
            reward_shaping=reward_shaping,
            no_oracle=no_oracle,
            max_episode_steps=max_episode_steps,
            obs_size=obs_size,
            seed=seed,
        )
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return _init
