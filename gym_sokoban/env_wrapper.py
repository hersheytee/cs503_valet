"""
Environment wrapper for gym-sokoban.
Translates old Gym v0.21 environments into modern Gymnasium v0.28+ environments.
Adds the `query_oracle` action and downsamples RGB images to 84x84.
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

class SokobanOracleWrapper(gym.Env):
    """
    A modern Gymnasium environment that wraps the old Gym Sokoban environment.
    """
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, env_id: str = 'Sokoban-small-v0', oracle_cost: float = 0.0):
        # Build the environment using the OLD gym
        self.env = old_gym.make(env_id)
        self.oracle_cost = oracle_cost
        
        # Extend action space: 9 original actions + 1 oracle action
        n_original = 9
        self.QUERY_ACTION = n_original
        self.action_space = spaces.Discrete(n_original + 1)
        
        # Downsample observation space from 160x160 to 84x84
        self.obs_size = 84
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(self.obs_size, self.obs_size, 3), 
            dtype=np.uint8
        )

    def _process_obs(self):
        """Grabs the raw RGB image from the old env and resizes it to 84x84."""
        obs = self.env.render(mode='rgb_array')
        resized = cv2.resize(obs, (self.obs_size, self.obs_size), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, seed=None, options=None):
        # Old gym doesn't always handle seeds cleanly in reset()
        if seed is not None:
            self.env.seed(seed)
            
        self.env.reset()
        
        # Clear the state-aware cache
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, '_oracle_cache'): unwrapped._oracle_cache = None
        if hasattr(unwrapped, '_expected_state'): unwrapped._expected_state = None
            
        # Return modern format: obs, info
        return self._process_obs(), {}

    def step(self, action: int):
        guided = False
        oracle_action = None

        if action == self.QUERY_ACTION:
            unwrapped = self.env.unwrapped
            oracle_action = get_oracle_action(unwrapped)
            guided = True
            
            # Fatal deadlock
            if oracle_action == 0:
                self.env.reset() 
                info = {'guided': True, 'oracle_action': 0, 'fatal_deadlock': True}
                # Return modern format: obs, reward, terminated, truncated, info
                return self._process_obs(), -1.0, True, False, info
                
            action = oracle_action

        # Take the step in the old environment (which returns 4 values)
        _, reward, done, info = self.env.step(action)

        if guided and not info.get('fatal_deadlock', False):
            reward -= self.oracle_cost

        info['guided'] = guided
        info['oracle_action'] = int(oracle_action) if oracle_action is not None else -1

        # Convert old 'done' to modern 'terminated' and 'truncated'
        terminated = bool(done)
        truncated = False

        return self._process_obs(), reward, terminated, truncated, info

    def render(self):
        return self.env.render(mode='rgb_array')


def make_env(env_id: str, oracle_cost: float = 0.0, seed: int = 0):
    """Factory function for SyncVectorEnv."""
    def _init():
        env = SokobanOracleWrapper(env_id, oracle_cost)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return _init