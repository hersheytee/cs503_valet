"""
Tests the SokobanOracleWrapper and saves a GIF of the run.
"""

from env_wrapper import SokobanOracleWrapper
import env_wrapper
import numpy as np
import imageio
import os

def main():
    print("Initializing Wrapper...")
    env = SokobanOracleWrapper(env_id='Sokoban-small-v0', oracle_cost=0.5, reward_shaping=True)
    
    obs, info = env.reset(seed=12)
    assert obs.shape == (56, 56, 3), obs.shape
    assert env.action_space.n == 10, env.action_space.n
    
    # List to store our frames for the GIF
    frames = []
    
    # We grab the high-resolution render for the GIF rather than the resized policy obs.
    frames.append(env.render())

    QUERY_ACTION = env.action_space.n - 1

    print("\n--- RUNNING ORACLE ---")
    total_reward = 0.0
    
    # Let the oracle play until it wins or hits 20 steps
    for step in range(1, 21):
        obs, reward, terminated, truncated, info = env.step(QUERY_ACTION)
        total_reward += reward
        
        # Capture the frame after the step
        frames.append(env.render())
        
        oracle_action = info.get('oracle_action')
        current_dist = env._shaper._previous_box_dist if env._shaper else None

        print(f"Step {step:2d} | Oracle chose: {oracle_action} | "
              f"Reward: {reward:+.2f} | Current Dist: {current_dist}")

        if terminated or info.get('fatal_deadlock'):
            print(f"\n[!] Episode ended at step {step}")
            # Add a few duplicate frames at the end so the GIF pauses on the victory screen
            for _ in range(5):
                frames.append(env.render())
            break

    print(f"\nTotal shaped reward: {total_reward:.2f}")
    env.close()

    baseline_env = SokobanOracleWrapper(env_id='Sokoban-small-v0', no_oracle=True)
    baseline_obs, _ = baseline_env.reset(seed=12)
    assert baseline_obs.shape == (56, 56, 3), baseline_obs.shape
    assert baseline_env.action_space.n == 9, baseline_env.action_space.n
    baseline_env.close()

    original_oracle = env_wrapper.get_oracle_action
    try:
        env_wrapper.get_oracle_action = lambda _env: 4

        noisy_env = SokobanOracleWrapper(
            env_id='Sokoban-small-v0',
            oracle_accuracy=0.0,
            seed=12,
        )
        noisy_env.reset(seed=12)
        noisy_query = noisy_env.action_space.n - 1
        for _ in range(20):
            _, _, terminated, truncated, info = noisy_env.step(noisy_query)
            assert info['oracle_optimal_action'] == 4
            assert info['oracle_action'] != 4
            assert info['oracle_correct'] is False
            if terminated or truncated:
                noisy_env.reset(seed=12)
        noisy_env.close()
    finally:
        env_wrapper.get_oracle_action = original_oracle

    # Save the GIF
    os.makedirs("gym_sokoban/figures", exist_ok=True)
    gif_path = "gym_sokoban/figures/sokoban_oracle_test.gif"
    print(f"\nSaving visual to {gif_path}...")
    try:
        imageio.mimsave(gif_path, frames, fps=4)
        print("Done! Open the GIF to see the agent move.")
    except PermissionError as exc:
        print(f"Could not write GIF ({exc}); wrapper/oracle checks already passed.")

if __name__ == "__main__":
    main()
