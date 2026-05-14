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

from oracle import get_oracle_action as _get_oracle_action_doorkey
from oracle_transfer import get_oracle_action as _get_oracle_action_transfer

_TRANSFER_ENV_TYPES = {'fetch', 'gotodoor', 'gotoobject'}

def get_oracle_action(env_unwrapped, env_type):
    if env_type in _TRANSFER_ENV_TYPES:
        return _get_oracle_action_transfer(env_unwrapped, env_type)
    return _get_oracle_action_doorkey(env_unwrapped, env_type)


# ── Reward shaping ────────────────────────────────────────────────────────────

class RewardShaper:
    """
    Donne des rewards intermédiaires pour DoorKey :
        +0.5 quand l'agent ramasse la clé
        +0.5 quand l'agent ouvre la porte
    Réinitialise à chaque épisode.
    """
    KEY_REWARD  = 0.5
    DOOR_REWARD = 0.5

    def __init__(self):
        self._has_key   = False
        self._door_open = False
        self._door_pos  = None  # position de la porte, trouvée au reset

    def reset(self, env_unwrapped):
        self._has_key   = False
        self._door_open = False
        # Trouve la position de la porte une seule fois par épisode
        self._door_pos  = None
        grid = env_unwrapped.grid
        for x in range(env_unwrapped.width):
            for y in range(env_unwrapped.height):
                cell = grid.get(x, y)
                if cell is not None and cell.type == 'door':
                    self._door_pos = (x, y)
                    break
            if self._door_pos:
                break

    def shape(self, env_unwrapped, reward):
        bonus = 0.0

        # Clé ramassée
        if not self._has_key and env_unwrapped.carrying is not None:
            bonus += self.KEY_REWARD
            self._has_key = True

        # Porte ouverte — accès direct via position connue
        if not self._door_open and self._door_pos is not None:
            cell = env_unwrapped.grid.get(*self._door_pos)
            if cell is None or cell.is_open:
                bonus += self.DOOR_REWARD
                self._door_open = True

        return reward + bonus


# ── Oracle Wrapper ────────────────────────────────────────────────────────────

class OracleWrapper(gym.Wrapper):
    """
    Wraps a MiniGrid environment to add a query_oracle action.

    Args:
        env_id        : Gymnasium env id, e.g. 'MiniGrid-DoorKey-8x8-v0'
        env_type      : 'empty' or 'doorkey' — selects the BFS oracle
        tile_size     : pixel size of each grid tile (default 8)
        oracle_cost   : reward penalty for querying oracle (default 0.0)
        reward_shaping: add intermediate rewards for key/door (default False)
    """

    def __init__(self, env_id: str, env_type: str = 'empty',
                 tile_size: int = 8, oracle_cost: float = 0.0,
                 reward_shaping: bool = False):
        inner = gym.make(env_id)
        inner = FullyObsWrapper(inner)
        inner = RGBImgObsWrapper(inner, tile_size=tile_size)

        super().__init__(inner)

        self.env_type       = env_type
        self.oracle_cost    = oracle_cost
        self.reward_shaping = reward_shaping
        self._shaper        = RewardShaper() if reward_shaping else None

        n_original        = inner.action_space.n
        self.QUERY_ACTION = n_original
        self.action_space = spaces.Discrete(n_original + 1)

        h, w, c = inner.observation_space['image'].shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, c), dtype=np.uint8
        )

    def _get_obs(self, obs_dict):
        return obs_dict['image']

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        if self._shaper:
            self._shaper.reset(self.env.unwrapped)
        return self._get_obs(obs_dict), info

    def step(self, action: int):
        guided        = False
        oracle_action = None

        if action == self.QUERY_ACTION:
            inner_unwrapped = self.env.unwrapped
            oracle_action   = get_oracle_action(inner_unwrapped, self.env_type)
            guided          = True
            action          = oracle_action

        obs_dict, reward, terminated, truncated, info = self.env.step(action)

        if guided:
            reward -= self.oracle_cost

        if self._shaper:
            reward = self._shaper.shape(self.env.unwrapped, reward)

        info['guided']        = guided
        info['oracle_action'] = int(oracle_action) if oracle_action is not None else -1

        return self._get_obs(obs_dict), reward, terminated, truncated, info


# ── Baseline Wrapper ──────────────────────────────────────────────────────────

class BaselineWrapper(gym.Wrapper):
    """Env sans action oracle — baseline PPO pur."""

    def __init__(self, env_id: str, tile_size: int = 8,
                 reward_shaping: bool = False):
        inner = gym.make(env_id)
        inner = FullyObsWrapper(inner)
        inner = RGBImgObsWrapper(inner, tile_size=tile_size)
        super().__init__(inner)

        self.reward_shaping = reward_shaping
        self._shaper        = RewardShaper() if reward_shaping else None

        h, w, c = inner.observation_space['image'].shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, c), dtype=np.uint8
        )

    def _get_obs(self, obs_dict):
        return obs_dict['image']

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        if self._shaper:
            self._shaper.reset(self.env.unwrapped)
        return self._get_obs(obs_dict), info

    def step(self, action: int):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)

        if self._shaper:
            reward = self._shaper.shape(self.env.unwrapped, reward)

        info['guided']        = False
        info['oracle_action'] = -1
        return self._get_obs(obs_dict), reward, terminated, truncated, info


# ── Factory ───────────────────────────────────────────────────────────────────

def make_env(env_id: str, env_type: str = 'empty',
             tile_size: int = 8, oracle_cost: float = 0.0,
             seed: int = 0, no_oracle: bool = False,
             reward_shaping: bool = False):
    """
    Factory function compatible avec gymnasium's SyncVectorEnv.
    """
    def _init():
        if no_oracle:
            env = BaselineWrapper(env_id, tile_size, reward_shaping)
        else:
            env = OracleWrapper(env_id, env_type, tile_size,
                                oracle_cost, reward_shaping)
        env.reset(seed=seed)
        return env
    return _init
