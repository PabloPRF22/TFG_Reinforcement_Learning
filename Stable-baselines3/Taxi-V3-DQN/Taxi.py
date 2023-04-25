import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_rewards(rewards, save_path, window_size=100):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Recompensa por episodio')
    plt.plot(np.arange(window_size - 1, len(rewards)), moving_average(rewards, window_size),
             label=f'Media móvil de {window_size} episodios')
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa acumulada')
    plt.title('Recompensas a lo largo del entrenamiento')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

class RewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLogger, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        reward = self.training_env.envs[0].get_episode_rewards()
        if reward:
            self.rewards.append(reward[-1])
        return True

# Crea el entorno
env = gym.make("Taxi-v3")
env = Monitor(env)

# Envuelve el entorno en un DummyVecEnv
env = DummyVecEnv([lambda: env])

# Crea el modelo DQN
model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, batch_size=64, target_update_interval=1000,
            exploration_fraction=0.1, tensorboard_log="./tensorboard_logs/")

reward_logger = RewardLogger()

# Entrena el agente
model.learn(total_timesteps=100000, callback=reward_logger)

# Guarda el modelo entrenado
model.save("taxi_dqn")

# Prueba el agente entrenado
env = gym.make("Taxi-v3")
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

# Grafica las recompensas a lo largo del entrenamiento
plot_rewards(reward_logger.rewards, "rewards_plot.png")
