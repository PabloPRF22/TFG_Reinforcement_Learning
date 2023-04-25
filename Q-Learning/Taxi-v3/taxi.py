import numpy as np
import random
import gym
import matplotlib.pyplot as plt

# Hiperparámetros
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento
epsilon = 1.0  # Probabilidad de exploración
min_epsilon = 0.01  # Mínima probabilidad de exploración
decay_rate = 0.001  # Tasa de decaimiento de la exploración

def choose_action(q_table, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explorar
    else:
        return np.argmax(q_table[state])  # Explotar

def update_q_table(q_table, state, action, next_state, reward, alpha, gamma):
    q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

def train_agent(env, episodes, alpha, gamma, epsilon, min_epsilon, decay_rate):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = choose_action(q_table, state, epsilon)
            next_state, reward, done, _ = env.step(action)
            update_q_table(q_table, state, action, next_state, reward, alpha, gamma)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        epsilon = max(min_epsilon, epsilon - decay_rate)

    return q_table, rewards

def plot_rewards(rewards, window_size=100):
    plt.figure(figsize=(12, 6))
    moving_average = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    plt.plot(np.arange(len(moving_average)), moving_average)
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa media')
    plt.title(f'Recompensa media de cada {window_size} episodios')
    plt.grid()
    plt.show()

env = gym.make("Taxi-v3")
episodes = 1500

q_table, rewards = train_agent(env, episodes, alpha, gamma, epsilon, min_epsilon, decay_rate)

# Grafica las recompensas a lo largo del entrenamiento
plot_rewards(rewards)

# Prueba el agente entrenado
state = env.reset()
done = False
total_reward = 0

while not done:
    action = np.argmax(q_table[state])
    state, reward, done, _ = env.step(action)
    total_reward += reward
    env.render()

print("Recompensa final:", total_reward)
env.close()
