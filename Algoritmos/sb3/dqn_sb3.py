from typing import Tuple
from typing import List
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, configure
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor
import shutil
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

import csv
import numpy as np

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
        #primera_columna = np.array([float(row[0]) for row in lector], dtype=float)
        
    return np.array(episodios, dtype=float)

# Ruta del archivo CSV

def train_sb3_dqn(env: gym.Env, eval_env: gym.Env, reward_threshold:int, **kwargs:dict) -> BaseAlgorithm:
    shutil.rmtree("logs/", ignore_errors=True)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, eval_freq=1500, n_eval_episodes=10,verbose=1)
    monitor_env = VecMonitor(env, "logs/")
    sb3_dqn = DQN(env = monitor_env,policy='MlpPolicy', verbose=1, **kwargs)
    sb3_dqn.learn(total_timesteps=250_000,callback=eval_callback)
    rewards = get_rewards()
    shutil.rmtree("logs/", ignore_errors=True)


    return sb3_dqn, rewards


