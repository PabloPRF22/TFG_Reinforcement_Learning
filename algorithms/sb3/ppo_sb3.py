import gym

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import csv
import numpy as np
import shutil

def get_rewards():
    archivo_csv = './logs/.monitor.csv'
    episodios = []

    with open(archivo_csv, 'r') as f:
        lector = csv.reader(f)
        
        next(lector)
        next(lector)
        for i, row in enumerate(lector):
            episodio = float(row[0])
            
            episodios.append(episodio)
                
    return np.array(episodios, dtype=float)

def train_sb3_ppo(env: gym.Env, eval_env: gym.Env,reward_threshold:int, **kwargs) -> BaseAlgorithm:
    monitor_env = VecMonitor(env, "logs/", info_keywords=['episode'])
    name = eval_env.unwrapped.spec.id
    total = 10_000_000
    if name != "HumanoidPyBulletEnv-v0":
        total = 1_000_000
        
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, eval_freq=2000,n_eval_episodes=10, verbose=1)

    sb3_ppo = PPO(env = monitor_env,policy='MlpPolicy', verbose=1, **kwargs)
    sb3_ppo.learn(total_timesteps=total,callback=eval_callback)
    rewards = get_rewards()
    shutil.rmtree("logs/", ignore_errors=True)
    return sb3_ppo, rewards
