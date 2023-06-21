
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

class CustomEvalCallBack(BaseCallback):
    def __init__(self,eval_env,reward_threshold,num_episodes, verbose=0):
        
        super(CustomEvalCallBack, self).__init__(verbose)
        # List used to store total rewards per episode
        self.episode_eval_count = 0
        self.episode_eval_reward = []
        self.eval_env = eval_env
        self.num_episodes =num_episodes
        self.reward_threshold = reward_threshold

    def _on_step(self):
        evaluate = True
        long = len(self.locals['infos'])
        for i in range(long):
            if(len(self.locals['infos'][i]) != 0 and 'episode' in self.locals['infos'][i]):
                self.episode_eval_count = self.episode_eval_count+1
                evaluate = self._evaluate()
        return evaluate
    
    def _evaluate(self):
        if(self.episode_eval_count%100 == 0):
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.num_episodes)
            print(f"La recompensa media en el episodio {self.episode_eval_count} es de {mean_reward}" )
            self.episode_eval_reward.append(mean_reward)
            if(mean_reward>self.reward_threshold): return False
        return True

            