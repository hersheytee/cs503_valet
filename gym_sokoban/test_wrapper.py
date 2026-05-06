"""
Tests the SokobanOracleWrapper and saves a GIF of the run.
"""

from env_wrapper import SokobanOracleWrapper
import numpy as np
import imageio
import os

def main():
    print("Initializing Wrapper...")
    env = SokobanOracleWrapper(env_id='Sokoban-small-v0', oracle_cost=0.5, reward_shaping=True)
    
    obs, info = env.reset(seed=12)
    
    # List to store our frames for the GIF
    frames = []
    
    # We grab the high-resolution render for the GIF (160x160) rather than the 84x84 obs
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

    # Save the GIF
    gif_path = "sokoban_oracle_test.gif"
    print(f"\nSaving visual to {gif_path}...")
    imageio.mimsave(gif_path, frames, fps=4) # fps=4 gives a nice, readable speed
    print("Done! Open the GIF to see the agent move.")

if __name__ == "__main__":
    main()