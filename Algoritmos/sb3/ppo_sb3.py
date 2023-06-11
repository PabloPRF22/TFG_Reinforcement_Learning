import gym

from stable_baselines3 import PPO
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

def train_sb3_ppo(env: gym.Env, **kwargs) -> BaseAlgorithm:
    reward_logger = RewardLoggerCallback(check_freq=1000)
    sb3_ppo = PPO(env = env,policy='MlpPolicy', verbose=1, **kwargs)
    sb3_ppo.learn(total_timesteps=10_000, callback=reward_logger)
    return sb3_ppo, reward_logger.rewards
