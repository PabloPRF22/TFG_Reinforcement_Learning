import gym

from sb3_contrib import TRPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

class RewardLoggerCallback(BaseCallback):
    def __init__(self, check_freq: int):
        super(RewardLoggerCallback, self).__init__()
        self.check_freq = check_freq
        self.rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            rewards = []
            for _ in range(50):
                r = evaluate_policy(self.model, self.training_env, n_eval_episodes=1, deterministic=True)[0]
                rewards.append(r)
            self.rewards.append(sum(rewards) / len(rewards))

        return True


def train_sb3_trpo(env: gym.Env, **kwargs) -> BaseAlgorithm:
    reward_logger = RewardLoggerCallback(check_freq=1000)
    sb3_trpo = TRPO('MlpPolicy', env, verbose=1, **kwargs)
    sb3_trpo.learn(total_timesteps=10_000, callback=reward_logger)
    return sb3_trpo, reward_logger.rewards
