import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Start with is_slippery=False for debugging VLM queries, 
# then turn it to True to test stochasticity.
env = gym.make(
    "FrozenLake-v1", 
    map_name="4x4", 
    is_slippery=False, 
    render_mode="rgb_array"

)

# initialize PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.001,
    n_steps=1024,
    batch_size=64
)

print("Starting baseline training...")
model.learn(total_timesteps=50_000)

model.save("ppo_frozenlake_baseline")
print("Training finished and model saved!")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

env.close()