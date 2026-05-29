"""
CNN policy for MiniGrid partial-obs RGB observations (56x56x3).
One strided conv (stride=2) instead of two, keeping 28x28 feature maps
before the AdaptiveAvgPool — better suited for the smaller input.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class CNNPolicy(nn.Module):
    def __init__(self, obs_shape, n_actions: int, hidden_dim: int = 256):
        super().__init__()

        H, W, C = obs_shape

        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(C, 32, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            cnn_out = self.cnn(dummy).shape[1]

        self.fc = nn.Sequential(
            layer_init(nn.Linear(cnn_out, hidden_dim)),
            nn.ReLU(),
        )

        self.policy_head = layer_init(nn.Linear(hidden_dim, n_actions), std=0.01)
        self.value_head  = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def _preprocess(self, obs):
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
        return obs.permute(0, 3, 1, 2)

    def forward(self, obs):
        x = self._preprocess(obs)
        x = self.cnn(x)
        x = self.fc(x)
        return self.policy_head(x), self.value_head(x).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, obs):
        _, value = self.forward(obs)
        return value
