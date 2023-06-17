import torch

from tqdm import tqdm
from typing import List
from ray.rllib.algorithms.dqn import DQNConfig


class DQNrllib:

    def __init__(self, env_name: str) -> None:
        self.config = DQNConfig().environment(env_name).framework('torch')
        self.dqn = self.config.build()

    def learn(self) -> None:
        for _ in tqdm(range(10), desc='Training DQN RLLIB'):
            self.dqn.train()

    def predict(self, state, deterministic: bool = True) -> List[int]:
        # Taxi-v3
        if type(state) == int:
            state_arr = torch.zeros(500).unsqueeze(0)
            state_arr[0, state] = 1
        # other environments
        else:
            state_arr = torch.tensor(state).unsqueeze(0)
        policy = self.dqn.get_policy()
        out = policy.compute_single_action(state_arr, explore=not deterministic)
        q_values = out[2]['q_values']
        return [q_values.argmax()]
