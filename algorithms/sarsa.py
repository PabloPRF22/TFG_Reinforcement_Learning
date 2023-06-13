import gym
import numpy as np

from tqdm import tqdm
from typing import List


class SARSAAgent:

    def __init__(
            self,
            env: gym.Env,
            alpha: float = 0.1,
            gamma: float = 0.99,
            epsilon: float = 0.1
    ):
        self.env = env
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        print("entor")

    def choose_action(self, state, deterministic: bool = False) -> int:
        if not deterministic and np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_table[state]))

    def sarsa_update(self, s, a, r, ns, na) -> None:
        next_action = self.choose_action(ns)
        td_target = r + self.gamma * self.q_table[ns, next_action]
        td_error = td_target - self.q_table[s, a]
        self.q_table[s, a] += self.alpha * td_error

    def learn(self, total_episodes: int) -> None:
        for _ in tqdm(range(total_episodes), desc='Training'):
            state, done = self.env.reset(), False
            action = self.choose_action(state, deterministic=False)
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.choose_action(next_state, deterministic=False)
                self.sarsa_update(state, action, reward, next_state, next_action)
                state, action = next_state, next_action

    def predict(self, state, deterministic: bool = True) -> List[int]:
        return [self.choose_action(state, deterministic=deterministic)]
