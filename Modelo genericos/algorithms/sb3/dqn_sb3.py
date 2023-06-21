from typing import Tuple
from typing import List
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.base_class import BaseAlgorithm
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecMonitor
from algorithms.sb3.customCallbacks.CustomEvalCallBack import CustomEvalCallBack
from algorithms.sb3.customCallbacks.LossTrainCallBack import LossTrainCallBack


def train_sb3_dqn(env: gym.Env, eval_env: gym.Env, reward_threshold:int, **kwargs:dict) -> BaseAlgorithm:
    eval_callback = CustomEvalCallBack(eval_env,reward_threshold,10)
    loss_callback = LossTrainCallBack()

    monitor_env = VecMonitor(env, "logs/")
    try:
        if 'policy' in kwargs:
            sb3_dqn = DQN(**kwargs,env = monitor_env, verbose=1)
        else:
            sb3_dqn = DQN(env = monitor_env,policy='MlpPolicy', verbose=1, **kwargs)
    except Exception as e:
        msg = f"Se produjo un error al crear el agente de DQN: {e}"
        raise ValueError(msg)
    sb3_dqn.learn(total_timesteps=1_000_000,callback=[eval_callback,loss_callback],progress_bar=True)
    data_of_training = {
        'agent': sb3_dqn,
        'episodes': eval_callback.episode_eval_count,
        'rewardsEvaluateTraining':eval_callback.episode_eval_reward ,
        'lossTrain': loss_callback.loss_train
         
         
    }
    
    return sb3_dqn, eval_callback.episode_eval_count,eval_callback.episode_eval_reward ,loss_callback.loss_train

def get_trained_model_dqn(path):
    loaded_model = DQN.load(path)
    return loaded_model




