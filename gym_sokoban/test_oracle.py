"""
Tests the BFS oracle on a raw gym-sokoban environment using the legacy gym API.
"""

import argparse
import gym 
import gym_sokoban
import matplotlib.pyplot as plt
import numpy as np
from bfs_oracle import get_oracle_action

# --- MONKEYPATCH FOR NUMPY 2.0 vs OLD GYM ---
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
# --------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="Sokoban-small-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    # Use old gym registry
    env = gym.make(args.env_id)
    env.seed(args.seed)
    
    # Old gym reset returns just the observation, not (obs, info)
    obs = env.reset()
    
    # Setup Matplotlib for live rendering
    if not args.no_window:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis('off')
    
    # Old gym render requires the mode argument here
        img_display = ax.imshow(env.render(mode='rgb_array'))
        plt.title(f"BFS oracle solving {args.env_id}")
        plt.tight_layout()
    
    done = False
    steps = 0
    
    print("Starting Oracle Test...")

    reward = 0
    
    while not done and steps < args.max_steps:
        # Pass the raw base environment to the oracle
        unwrapped_env = env.unwrapped
        
        # Ask the Oracle for the next best action
        action = get_oracle_action(unwrapped_env)
        
        if action == 0:
            print("Oracle returned no-op (0). It either solved the map or got stuck!")
            break
            
        # Old gym step returns 4 values (no 'truncated')
        obs, reward, done, info = env.step(action)
        steps += 1
        
        print(f"Step: {steps:02d} | Action: {action} | Reward: {reward}")
        
        # Update the visual frame
        if not args.no_window:
            img_display.set_data(env.render(mode='rgb_array'))
        
        # Pause to make it human-watchable
            plt.pause(0.2)
        
    print(f"Episode finished. Total steps taken: {steps}")
    if reward > 0:
        print("Success! The Oracle solved it.")
    else:
        print("Failure. The Oracle couldn't solve it.")
        
    if not args.no_window:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()
