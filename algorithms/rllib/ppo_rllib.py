import torch

from tqdm import tqdm
from typing import List
from ray.rllib.algorithms.ppo import PPOConfig


class PPOrllib:

    def __init__(self, env_name: str) -> None:
        self.config = PPOConfig().environment(env_name).framework('torch')
        self.ppo = self.config.build()

    def learn(self) -> None:
        for _ in tqdm(range(20), desc='Training PPO'):
            self.ppo.train()

    def predict(self, state, deterministic: bool = True) -> List[int]:
        # Taxi-v3
        if type(state) == int:
            state_arr = torch.zeros(500).unsqueeze(0)
            state_arr[0, state] = 1
        # other environments
        else:
            state_arr = torch.tensor(state).unsqueeze(0)
        policy = self.ppo.get_policy()
        out = policy.compute_single_action(state_arr, explore=not deterministic)
        logits = out[2]['action_dist_inputs']
        return [logits.argmax()]
