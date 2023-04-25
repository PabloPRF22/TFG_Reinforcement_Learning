from stable_baselines3.common.callbacks import BaseCallback

class RewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLogger, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        reward = self.training_env.envs[0].get_episode_rewards()
        if reward:
            self.rewards.append(reward[-1])
        return True