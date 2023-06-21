import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

env = gym.make('Taxi-v3')

# Define el tamaño de entrada para la red neuronal
input_size = env.observation_space.n
# Define el tamaño de salida para la red neuronal
output_size = env.action_space.n

# Define la arquitectura de la red neuronal
model = Sequential()
model.add(Dense(128, input_dim=input_size, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(output_size, activation='linear'))
model.compile(loss='mse', optimizer=Adam(lr=0.001))

# Entrena la red neuronal
num_episodes = 50
num_timesteps = 100
epsilon = 1.0
decay_rate = 0.995
min_epsilon = 0.01
for i in range(num_episodes):
    state = env.reset()
    state = state[0]
    total_reward = 0
    for t in range(num_timesteps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(np.eye(input_size)[state:state+1]))
        next_state, reward, done, truncated, info = env.step(action)
        target = reward + 0.99 * np.max(model.predict(np.eye(input_size)[next_state:next_state+1]))
        target_f = model.predict(np.eye(input_size)[state:state+1])[0]
        target_f[action] = target
        model.fit(np.eye(input_size)[state:state+1], np.reshape(target_f, (1, output_size)), epochs=1, verbose=0)
        state = next_state
        total_reward += reward
        if done or truncated:
            break
    epsilon = max(epsilon * decay_rate, min_epsilon)
    if i % 100 == 0:
        print('Episode {}: Total reward = {}, Epsilon = {}'.format(i, total_reward, epsilon))
env = gym.make('Taxi-v3',render_mode = "human")
state = env.reset()
state= state[0]
done = False
truncated = False
totalReward = 0
while not done and not truncated:
    action = np.argmax(model.predict(np.eye(input_size)[state:state+1]))
    new_state, reward, done,truncated, info = env.step(action)
    totalReward+= reward
    state= new_state
print(totalReward)
