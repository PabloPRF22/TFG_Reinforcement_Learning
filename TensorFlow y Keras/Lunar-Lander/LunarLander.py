from Agente import Agente
import numpy as np
import gym
import tensorflow as tf
from utils import plotLearning
from tf_agents.environments import tf_py_environment
import imageio
from pyvirtualdisplay import Display

# Crea un display virtual
display = Display(visible=0, size=(1400, 900))
display.start()

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    env = gym.make('LunarLander-v2')

    lr = 0.001
    episodios = 400
    agente = Agente(gamma=0.99, epsilon=1.0, lr=lr,
                  input_dims=env.observation_space.shape,
                  n_actions=env.action_space.n, mem_size=1000000,
                  batch_size=64, epsilon_end=0.10)

    scores = []
    eps_history = []

    for i in range(episodios):
        done = False
        score = 0
        observation = env.reset()
        images = []
        while not done:
            action = agente.elegir_accion(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agente.guardar_transicion(observation, action, reward, observation_, done)
            observation = observation_
            agente.aprender()

            img = env.render(mode='rgb_array')
            images.append(img)
        eps_history.append(agente.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print("Episodio : ", i)
        print("Recompensa ", score)
        print("Recompensa Media : ", avg_score)
        print("Epsilon : ", agente.epsilon)
        print("-------------------------------------\n")

        # Guarda el video del episodio actual
        if(i%100 ==0):imageio.mimsave(f'./video/episode_{i}.mp4', images, fps=30)
    images = []

    for i in range(10):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agente.elegir_accion(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            observation = observation_

            img = env.render(mode='rgb_array')
            images.append(img)
        #eps_history.append(agente.epsilon)
        #scores.append(score)

        avg_score = np.mean(scores[-100:])


        # Guarda el video del episodio actual
    imageio.mimsave(f'./video/episode_final.mp4', images, fps=30)       
    env.close()

    filename = 'Estadisticas64.png'
    x = [i + 1 for i in range(episodios)]
    plotLearning(x, scores, eps_history, filename)

# Detén el display virtual
display.stop()
