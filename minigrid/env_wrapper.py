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

import random as _random

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper

from oracle import get_oracle_action as _get_oracle_action_doorkey
from oracle import get_all_oracle_actions as _get_all_oracle_actions_doorkey
from oracle_transfer import get_oracle_action as _get_oracle_action_transfer
from oracle_transfer import get_all_oracle_actions as _get_all_oracle_actions_transfer

_TRANSFER_ENV_TYPES = {'fetch', 'gotodoor', 'gotoobject'}

def get_oracle_action(env_unwrapped, env_type):
    if env_type in _TRANSFER_ENV_TYPES:
        return _get_oracle_action_transfer(env_unwrapped, env_type)
    return _get_oracle_action_doorkey(env_unwrapped, env_type)

def get_all_oracle_actions(env_unwrapped, env_type):
    if env_type in _TRANSFER_ENV_TYPES:
        return _get_all_oracle_actions_transfer(env_unwrapped, env_type)
    return _get_all_oracle_actions_doorkey(env_unwrapped, env_type)


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

_ACTION_NAMES = {0: "turn_left", 1: "turn_right", 2: "forward",
                 3: "pickup", 4: "drop", 5: "toggle"}
_HISTORY_LEN  = 5   # number of past actions kept for VLM context


class OracleWrapper(gym.Wrapper):
    """
    Wraps a MiniGrid environment to add a query_oracle action.

    Args:
        env_id          : Gymnasium env id, e.g. 'MiniGrid-DoorKey-8x8-v0'
        env_type        : 'empty' or 'doorkey' — selects the BFS oracle
        tile_size       : pixel size of each grid tile (default 8)
        oracle_cost     : reward penalty for querying oracle (default 0.0)
        reward_shaping  : add intermediate rewards for key/door (default False)
        vlm_client      : OpenAI client for VLM queries (None = use BFS oracle)
        vlm_model_key   : key in vlm_oracle.MODELS (e.g. 'qwen3b')
        vlm_served_name : model ID as reported by the vLLM server
    """

    def __init__(self, env_id: str, env_type: str = 'empty',
                 tile_size: int = 8, oracle_cost: float = 0.0,
                 reward_shaping: bool = False,
                 vlm_client=None, vlm_model_key: str = 'qwen3b',
                 vlm_served_name: str = '', vlm_mode: str = 'baseline',
                 partial_obs: bool = False,
                 oracle_noise: float = 1.0):
        inner = gym.make(env_id)
        if partial_obs:
            inner = RGBImgPartialObsWrapper(inner, tile_size=tile_size)
        else:
            inner = FullyObsWrapper(inner)
            inner = RGBImgObsWrapper(inner, tile_size=tile_size)

        super().__init__(inner)

        self.env_type        = env_type
        self.oracle_cost     = oracle_cost
        self.reward_shaping  = reward_shaping
        self._shaper         = RewardShaper() if reward_shaping else None

        self._vlm_client      = vlm_client
        self._vlm_model_key   = vlm_model_key
        self._vlm_served_name = vlm_served_name
        self._vlm_mode        = vlm_mode
        self._oracle_noise    = oracle_noise  # 1.0 = perfect BFS, 0.5 = 50% accuracy
        self._last_obs        = None   # current RGB obs, updated at reset/step
        self._action_history  = []     # rolling window of past action names

        n_original        = inner.action_space.n
        self.QUERY_ACTION = n_original
        self.action_space = spaces.Discrete(n_original + 1)

        h, w, c = inner.observation_space['image'].shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, c), dtype=np.uint8
        )

    def _get_obs(self, obs_dict):
        return obs_dict['image']

    def _record_action(self, action: int):
        name = _ACTION_NAMES.get(action, str(action))
        self._action_history.append(name)
        if len(self._action_history) > _HISTORY_LEN:
            self._action_history.pop(0)

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        if self._shaper:
            self._shaper.reset(self.env.unwrapped)
        self._last_obs       = self._get_obs(obs_dict)
        self._action_history = []
        return self._last_obs, info

    def step(self, action: int):
        guided           = False
        oracle_action    = None
        vlm_correct      = -1    # -1 = not applicable (BFS mode or non-guided step)
        vlm_parse_failed = False  # True only when VLM response was unparsable/failed

        if action == self.QUERY_ACTION:
            if self._vlm_client is not None:
                from vlm_oracle import query_vlm
                oracle_action, vlm_parse_failed = query_vlm(
                    obs_img           = self._last_obs,
                    env_unwrapped     = self.env.unwrapped,
                    client            = self._vlm_client,
                    served_model_name = self._vlm_served_name,
                    model_key         = self._vlm_model_key,
                    action_history    = list(self._action_history),
                    mode              = self._vlm_mode,
                )
                # Compare VLM action against all BFS-optimal actions.
                # vlm_correct=1 if the VLM chose any optimal action, 0 otherwise.
                bfs_optimal = get_all_oracle_actions(self.env.unwrapped, self.env_type)
                vlm_correct = int(oracle_action in bfs_optimal) if bfs_optimal else -1
            else:
                # BFS oracle — optionally noisy
                bfs_action  = get_oracle_action(self.env.unwrapped, self.env_type)
                bfs_optimal = get_all_oracle_actions(self.env.unwrapped, self.env_type)
                if self._oracle_noise < 1.0 and _random.random() > self._oracle_noise:
                    # Return a random non-optimal action
                    all_acts   = set(range(self.QUERY_ACTION))
                    wrong_acts = list(all_acts - set(bfs_optimal or [bfs_action]))
                    oracle_action = _random.choice(wrong_acts) if wrong_acts else bfs_action
                    vlm_correct   = 0
                else:
                    oracle_action = bfs_action
                    vlm_correct   = 1 if self._oracle_noise < 1.0 else -1
            guided = True
            action = oracle_action

        self._record_action(action)
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = self._get_obs(obs_dict)

        if guided:
            reward -= self.oracle_cost

        if self._shaper:
            reward = self._shaper.shape(self.env.unwrapped, reward)

        info['guided']           = guided
        info['oracle_action']    = int(oracle_action) if oracle_action is not None else -1
        info['vlm_correct']      = vlm_correct      # 1=correct, 0=wrong, -1=N/A
        info['vlm_parse_failed'] = vlm_parse_failed # True when VLM response was unparsable

        return self._last_obs, reward, terminated, truncated, info


# ── Baseline Wrapper ──────────────────────────────────────────────────────────

class BaselineWrapper(gym.Wrapper):
    """Env sans action oracle — baseline PPO pur."""

    def __init__(self, env_id: str, tile_size: int = 8,
                 reward_shaping: bool = False, partial_obs: bool = False):
        inner = gym.make(env_id)
        if partial_obs:
            inner = RGBImgPartialObsWrapper(inner, tile_size=tile_size)
        else:
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

        info['guided']           = False
        info['oracle_action']    = -1
        info['vlm_correct']      = -1
        info['vlm_parse_failed'] = False
        return self._get_obs(obs_dict), reward, terminated, truncated, info


# ── Factory ───────────────────────────────────────────────────────────────────

def make_env(env_id: str, env_type: str = 'empty',
             tile_size: int = 8, oracle_cost: float = 0.0,
             seed: int = 0, no_oracle: bool = False,
             reward_shaping: bool = False,
             vlm_client=None, vlm_model_key: str = 'qwen3b',
             vlm_served_name: str = '', vlm_mode: str = 'baseline',
             partial_obs: bool = False,
             oracle_noise: float = 1.0):
    """
    Factory function compatible avec gymnasium's SyncVectorEnv.
    """
    def _init():
        if no_oracle:
            env = BaselineWrapper(env_id, tile_size, reward_shaping, partial_obs)
        else:
            env = OracleWrapper(env_id, env_type, tile_size,
                                oracle_cost, reward_shaping,
                                vlm_client, vlm_model_key, vlm_served_name,
                                vlm_mode, partial_obs, oracle_noise)
        env.reset(seed=seed)
        return env
    return _init
