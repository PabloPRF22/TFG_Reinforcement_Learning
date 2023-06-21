from stable_baselines3 import PPO
import gym

# Crea el primer entorno
env1 = gym.make('DemonAttackNoFrameskip-v4')

# Entrena un modelo en el primer entorno
model = PPO("MlpPolicy", env1, verbose=1)
model.learn(total_timesteps=10000)

# Guarda el modelo
model.save("ppo_lunar")

# Crea el segundo entorno
env2 = gym.make('MountainCar-v0')

# Carga el modelo guardado
model = PPO.load("ppo_lunar", env=env2)

# Continúa entrenando el modelo en el segundo entorno
model.learn(total_timesteps=10000)
