import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class CNNPolicy(nn.Module):
    def __init__(self, n_actions: int, hidden_dim: int = 512):
        super().__init__()
        
        # The classic "Nature CNN" architecture for 84x84 RGB images
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            # 64 channels * 7 * 7 spatial dimensions = 3136 flat features
            layer_init(nn.Linear(3136, hidden_dim)),
            nn.ReLU(),
        )
        
        # Action distribution and Value estimation
        self.policy_head = layer_init(nn.Linear(hidden_dim, n_actions), std=0.01)
        self.value_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, obs):
        # Convert from numpy/gym format (B, 84, 84, 3) to PyTorch format (B, 3, 84, 84)
        # Also normalize pixel values from [0, 255] to [0.0, 1.0]
        x = obs.float() / 255.0
        
        if len(x.shape) == 3: # Single observation (unbatched)
            x = x.unsqueeze(0)
            
        x = x.permute(0, 3, 1, 2)
        
        hidden = self.network(x)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
            
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, obs):
        _, value = self.forward(obs)
        return value