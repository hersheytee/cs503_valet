import gymnasium as gym
from stable_baselines3 import PPO
import time

# Create a new environment meant for human viewing
eval_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")

# Load your trained baseline model
model = PPO.load("ppo_frozenlake_baseline")

obs, info = eval_env.reset()
done = False

print("Watching the trained agent...")
while not done:
    # The agent predicts the best action (returns a numpy array)
    action, _states = model.predict(obs, deterministic=True)
    
    # Extract the integer from the array using .item()
    action_int = action.item() 
    
    # Take the step using the integer
    obs, reward, terminated, truncated, info = eval_env.step(action_int)
    
    # Add a tiny sleep so you can actually see the agent move
    time.sleep(0.5)
    
    done = terminated or truncated

eval_env.close()