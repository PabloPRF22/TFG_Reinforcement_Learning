from ReplayBuffer import ReplayBuffer
from DQN import build_dqn
import numpy as np
from tensorflow.keras.models import load_model

class Agente():

    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                input_dims, epsilon_dec=1e-3, epsilon_end=0.01,
                mem_size=1000000, fname='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        print(n_actions)
        print(input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 256, 256)


    def guardar_transicion(self, state, action, reward, new_state, done):
        # Guardamos la informacion en la memoria
        self.memory.guardar_transincion(state, action, reward, new_state, done)


    def elegir_accion(self, observation):
        if np.random.random() < self.epsilon:
            # Exploramos
            action = np.random.choice(self.action_space)
        else:
            #Pasamos a explotar usando la red neuronal entrenada
            # en vez seguir explorar
            state = np.array([observation])
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return action


    def aprender(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # Recuperar datos de la memoria de episodios anteriores
        states, actions, rewards, states_, dones = \
                self.memory.sample_buffer(self.batch_size)

        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)

        # Entrenamos el modelo
        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, actions] = rewards + \
                        self.gamma * np.max(q_next, axis=1)*dones
        self.q_eval.train_on_batch(states, q_target)

        # Actualizamos la epsilo usand
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                self.eps_min else self.eps_min


    def save_model(self):
        self.q_eval.save(self.model_file)


    def load_model(self):
        self.q_eval = load_model(self.model_file)