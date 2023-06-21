import gym
import numpy as np

from tqdm import tqdm
from typing import List


class QAgent:

    def __init__(
            self,
            env: gym.Env,
            env_name: str,
            alpha: float = 0.1,
            gamma: float = 0.99,
            epsilon: float = 0.1,
    ):
        self.env = env
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.env_name = env_name

    def choose_action(self, state, deterministic: bool = False) -> int:
        if not deterministic and np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_table[state]))

    def q_update(self, s, a, r, ns) -> float:
        return self.alpha * \
            (r + self.gamma * np.max(self.q_table[ns]) - self.q_table[s, a])

    def learn(self, total_episodes: int) -> None:
        for _ in tqdm(range(total_episodes), desc='Training Q-Agent para el entorno '+self.env_name):
            state, done = self.env.reset(), False
            while not done:
                action = self.choose_action(state, deterministic=False)
                next_state, reward, done, _ = self.env.step(action)
                self.q_table[state, action] += self.q_update(state, action, reward, next_state)
                # self.q_table = self.q_update(state, action, reward, next_state)
                state = next_state

    def predict(self, state, deterministic: bool = True) -> List[int]:
        return [self.choose_action(state, deterministic=deterministic)]
