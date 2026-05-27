"""
Smoke test the Sokoban CNNPolicy tensor shapes.
"""

import torch
from model import CNNPolicy


def check_model(obs_shape, n_actions):
    model = CNNPolicy(obs_shape=obs_shape, n_actions=n_actions)

    batch_size = 4
    dummy_obs = torch.randint(0, 256, (batch_size,) + obs_shape, dtype=torch.uint8)
    action, log_prob, entropy, value = model.get_action_and_value(dummy_obs)

    assert action.shape == (batch_size,)
    assert log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert value.shape == (batch_size,)

    single_obs = torch.randint(0, 256, obs_shape, dtype=torch.uint8)
    single_action, _, _, single_value = model.get_action_and_value(single_obs)
    assert single_action.shape == (1,)
    assert single_value.shape == (1,)

    print(f"OK: obs_shape={obs_shape}, n_actions={n_actions}")


def main():
    obs_shape = (56, 56, 3)
    check_model(obs_shape, n_actions=9)
    check_model(obs_shape, n_actions=10)
    print("[SUCCESS] Model smoke test passed.")


if __name__ == "__main__":
    main()
