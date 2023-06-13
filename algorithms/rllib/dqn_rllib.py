import torch

from tqdm import tqdm
from typing import List,Dict
from ray.rllib.algorithms.dqn import DQNConfig


class DQNrllib:

    def __init__(self, env_name: str, config: Dict[str, any]) -> None:
        self.config = DQNConfig().training(double_q=True).environment(env_name).framework('torch')
        self.dqn = self.config.build()

    def learn(self) -> None:
        for _ in tqdm(range(10), desc='Training DQN'):
            self.dqn.train()


