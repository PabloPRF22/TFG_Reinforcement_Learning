import gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from plot_rewards import plot_rewards
from RewardLogger import RewardLogger


# Crea el entorno
env = gym.make("LunarLander-v2")
env = Monitor(env)

# Crea un entorno de evaluación
eval_env = gym.make("LunarLander-v2")
eval_env = Monitor(eval_env)

# Envuelve el entorno en un DummyVecEnv
env = DummyVecEnv([lambda: env])
eval_env = DummyVecEnv([lambda: eval_env])

# Crea un callback de evaluación
eval_callback = EvalCallback(eval_env, best_model_save_path="./best_model",
                             log_path="./logs", eval_freq=500, deterministic=True, render=False)

# Crea el modelo DQN
model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, batch_size=64, target_update_interval=1000,
            exploration_fraction=0.1, tensorboard_log="./tensorboard_logs/")

reward_logger = RewardLogger()

# Entrena el agente
model.learn(total_timesteps=100000, callback=[eval_callback, reward_logger])

# Guarda el modelo entrenado
model.save("lunarlander_dqn")

# Prueba el agente entrenado
env = gym.make("LunarLander-v2")
obs = env.reset()
total_reward = 0

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        break

print("Recompensa final:", total_reward)
env.close()
plot_rewards(reward_logger.rewards, "rewards_plot.png")
