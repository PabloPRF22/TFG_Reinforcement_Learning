import gym
import pybullet_envs
import numpy as np
import imageio
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Crear el entorno
env = gym.make("HumanoidBulletEnv-v0")
env = DummyVecEnv([lambda: env])

# Crear y entrenar el modelo
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)

# Evaluar el modelo entrenado
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

# Generar un video del modelo entrenado
video_path = "ac_humanoid_video.mp4"
n_episodes = 1
frames = []

for episode in range(n_episodes):
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        frame = env.render(mode="rgb_array")
        frames.append(frame)
        if done:
            break

imageio.mimwrite(video_path, [np.array(frame) for frame in frames], fps=30)
print(f"Video saved at {video_path}")
