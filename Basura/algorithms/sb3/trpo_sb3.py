import gym

from sb3_contrib import TRPO
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


def train_sb3_trpo(env: gym.Env,eval_env: gym.Env,reward_threshold:int,env_name:str,**kwargs) -> BaseAlgorithm:
    monitor_env2 = VecMonitor(eval_env, "logsEval/", info_keywords=['episode'])

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, eval_freq=2500,n_eval_episodes=10, verbose=1)
    monitor_env = VecMonitor(env, "logs/", info_keywords=['episode'])
    total = 10_000_000
    if env_name != "HumanoidPyBulletEnv-v0":
        total = 1_000_000
    sb3_trpo = TRPO('MlpPolicy', monitor_env, verbose=1, **kwargs)
    sb3_trpo.learn(total_timesteps=total,callback=eval_callback,progress_bar=True)
    rewards = get_rewards()
    return sb3_trpo, rewards

def get_trained_model_trpo(path):
    loaded_model = TRPO.load(path)
    return loaded_model
