import gym

from stable_baselines3 import A2C
from stable_baselines3.common.base_class import BaseAlgorithm
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


def train_sb3_a2c(env: gym.Env,eval_env: gym.Env,reward_threshold:int,**kwargs) -> BaseAlgorithm:
    print(eval_env.unwrapped.spec.id)
    shutil.rmtree("logs/", ignore_errors=True)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, eval_freq=2500,n_eval_episodes=10, verbose=1)
    monitor_env = VecMonitor(env, "logs/", info_keywords=['episode'])
    
    sb3_a2c = A2C('MlpPolicy',monitor_env, verbose=1, **kwargs)
    sb3_a2c.learn(total_timesteps=500000,callback=eval_callback,progress_bar=True)
    rewards = get_rewards()
    return sb3_a2c, rewards

def get_trained_model_a2c(path):
    loaded_model = A2C.load(path)
    return loaded_model
