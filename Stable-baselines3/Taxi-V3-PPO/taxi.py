import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

# Función para graficar las recompensas
def plot_rewards(rewards, title='Recompensas del agente PPO'):
    plt.plot(rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa')
    plt.title(title)
    plt.show()

# Crear el entorno
env = gym.make('Taxi-v3')
env = Monitor(env)

# Envolver el entorno en un DummyVecEnv
env = DummyVecEnv([lambda: env])

# Crear el modelo PPO
model = PPO('MlpPolicy', env, verbose=1)

# Entrenar el agente
model.learn(total_timesteps=200000)

# Guardar el modelo entrenado
model.save("ppo_taxi-v3")

# Probar el agente entrenado
rewards = []
for episode in range(10):
    obs = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
    
    rewards.append(episode_reward[0])

print("Recompensas de prueba:", rewards)

# Graficar las recompensas
plot_rewards(rewards)
