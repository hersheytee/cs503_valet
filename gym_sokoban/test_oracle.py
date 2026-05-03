"""
Tests the A* Oracle on a raw gym-sokoban environment using the legacy gym API.
"""

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
    # Use old gym registry
    env = gym.make('Sokoban-small-v0')
    
    # Old gym reset returns just the observation, not (obs, info)
    obs = env.reset()
    
    # Setup Matplotlib for live rendering
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    
    # Old gym render requires the mode argument here
    img_display = ax.imshow(env.render(mode='rgb_array'))
    plt.title("A* Oracle solving Sokoban-small-v0")
    plt.tight_layout()
    
    done = False
    steps = 0
    
    print("Starting Oracle Test...")

    reward = 0
    
    while not done:
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
        img_display.set_data(env.render(mode='rgb_array'))
        
        # Pause to make it human-watchable
        plt.pause(0.2)
        
    print(f"Episode finished. Total steps taken: {steps}")
    if reward > 0:
        print("Success! The Oracle solved it.")
    else:
        print("Failure. The Oracle couldn't solve it.")
        
    # Keep window open at the end
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()