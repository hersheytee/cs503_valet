"""
Environment wrapper for MiniGrid that adds a `query_oracle` action.

The extended action space is:
    0..N-1 : original MiniGrid actions
    N      : query_oracle → oracle computes optimal action via BFS,
             executes it, marks the transition as guided (gt=1)

The wrapper also applies:
    - FullyObsWrapper  : full grid visibility
    - RGBImgObsWrapper : RGB image observation (40x40 for tile_size=8)

Info dict always contains:
    'guided'        : bool  — was this step guided by the oracle?
    'oracle_action' : int   — which action the oracle chose (if guided)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper

from oracle import get_oracle_action


class OracleWrapper(gym.Wrapper):
    """
    Wraps a MiniGrid environment to add a query_oracle action.

    Args:
        env_id   : Gymnasium env id, e.g. 'MiniGrid-Empty-5x5-v0'
        env_type : 'empty' or 'doorkey' — selects the BFS oracle
        tile_size: pixel size of each grid tile (default 8)
        oracle_cost: reward penalty for querying oracle (default 0.0)
    """

    def __init__(self, env_id: str, env_type: str = 'empty',
                 tile_size: int = 8, oracle_cost: float = 0.0):
        # Build the inner env with full obs + RGB
        inner = gym.make(env_id)
        inner = FullyObsWrapper(inner)
        inner = RGBImgObsWrapper(inner, tile_size=tile_size)

        super().__init__(inner)

        self.env_type    = env_type
        self.oracle_cost = oracle_cost

        # Extend action space: add one extra action = query_oracle
        n_original       = inner.action_space.n
        self.QUERY_ACTION = n_original          # index of the new action
        self.action_space = spaces.Discrete(n_original + 1)

        # Observation space: just the RGB image
        h, w, c = inner.observation_space['image'].shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, c), dtype=np.uint8
        )

    def _get_obs(self, obs_dict):
        """Extract RGB image from the obs dict returned by RGBImgObsWrapper."""
        return obs_dict['image']

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        return self._get_obs(obs_dict), info

    def step(self, action: int):
        guided       = False
        oracle_action = None

        if action == self.QUERY_ACTION:
            # Ask the BFS oracle for the optimal action
            inner_unwrapped = self.env.unwrapped
            oracle_action   = get_oracle_action(inner_unwrapped, self.env_type)
            guided          = True
            action          = oracle_action  # oracle plays instead of agent

        obs_dict, reward, terminated, truncated, info = self.env.step(action)

        # Apply oracle cost (0 by default for upper bound experiments)
        if guided:
            reward -= self.oracle_cost

        info['guided']        = guided
        info['oracle_action'] = int(oracle_action) if oracle_action is not None else -1

        return self._get_obs(obs_dict), reward, terminated, truncated, info


class BaselineWrapper(gym.Wrapper):
    """Env sans action oracle — baseline PPO pur."""

    def __init__(self, env_id: str, tile_size: int = 8):
        inner = gym.make(env_id)
        inner = FullyObsWrapper(inner)
        inner = RGBImgObsWrapper(inner, tile_size=tile_size)
        super().__init__(inner)

        h, w, c = inner.observation_space['image'].shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, c), dtype=np.uint8
        )

    def _get_obs(self, obs_dict):
        return obs_dict['image']

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        return self._get_obs(obs_dict), info

    def step(self, action: int):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        info['guided']        = False
        info['oracle_action'] = -1
        return self._get_obs(obs_dict), reward, terminated, truncated, info


def make_env(env_id: str, env_type: str = 'empty',
             tile_size: int = 8, oracle_cost: float = 0.0,
             seed: int = 0, no_oracle: bool = False):
    """
    Factory function — returns a callable that creates a wrapped env.
    Compatible with gymnasium's SyncVectorEnv.
    """
    def _init():
        if no_oracle:
            env = BaselineWrapper(env_id, tile_size)
        else:
            env = OracleWrapper(env_id, env_type, tile_size, oracle_cost)
        env.reset(seed=seed)
        return env
    return _init