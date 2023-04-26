import os
import gym
import pybullet_envs
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Create the environment
env = gym.make("HumanoidBulletEnv-v0")
env = DummyVecEnv([lambda: env])

# Train the model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2000000)

# Save the model
model.save("ppo_humanoid")

# Load the trained model
model = PPO.load("ppo_humanoid")

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

# Generate a video of the trained model
video_path = "PPO_humanoid_video.mp4"
n_episodes = 10
frames = []

for episode in range(n_episodes):
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        frame = env.envs[0].render(mode="rgb_array")
        frames.append(frame)
        if done:
            break

imageio.mimwrite(video_path, [np.array(frame) for frame in frames], fps=30)
print(f"Video saved at {video_path}")
