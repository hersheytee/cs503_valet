"""
CNN policy for MiniGrid RGB observations (40x40x3).
Architecture inspired by SAGE / standard RL vision papers.

Input : (B, 3, H, W) float32 in [0, 1]
Outputs:
    - logits : (B, n_actions)   action logits
    - value  : (B,)             state value estimate
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal init as used in CleanRL / PPO papers."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class CNNPolicy(nn.Module):
    """
    Shared CNN backbone + separate policy and value heads.

    Args:
        obs_shape  : (H, W, C) — raw obs shape (channels last)
        n_actions  : total number of actions (including query_oracle)
        hidden_dim : size of the FC layer after CNN
    """

    def __init__(self, obs_shape, n_actions: int, hidden_dim: int = 256):
        super().__init__()

        H, W, C = obs_shape

        # CNN backbone — 3 conv layers
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(C, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            cnn_out = self.cnn(dummy).shape[1]

        # Shared FC layer
        self.fc = nn.Sequential(
            layer_init(nn.Linear(cnn_out, hidden_dim)),
            nn.ReLU(),
        )

        # Policy head — small std init for stable early training
        self.policy_head = layer_init(nn.Linear(hidden_dim, n_actions), std=0.01)

        # Value head
        self.value_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def _preprocess(self, obs):
        """(B, H, W, C) uint8 → (B, C, H, W) float32 in [0, 1]."""
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
        # channels last → channels first
        return obs.permute(0, 3, 1, 2)

    def forward(self, obs):
        """Returns (logits, value)."""
        x = self._preprocess(obs)
        x = self.cnn(x)
        x = self.fc(x)
        return self.policy_head(x), self.value_head(x).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        """
        Sample action (or evaluate given action) and return:
            action, log_prob, entropy, value
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, obs):
        _, value = self.forward(obs)
        return value