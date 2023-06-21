import gym

from stable_baselines3 import A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecMonitor
from algorithms.sb3.customCallbacks.CustomEvalCallBack import CustomEvalCallBack
from algorithms.sb3.customCallbacks.LossTrainCallBack import LossTrainCallBack

def train_sb3_a2c(env: gym.Env,eval_env: gym.Env,reward_threshold:int,**kwargs) -> BaseAlgorithm:
    print(eval_env.unwrapped.spec.id)
    eval_callback = CustomEvalCallBack(eval_env,reward_threshold,10)
    loss_callback = LossTrainCallBack()
    monitor_env = VecMonitor(env, "logs/", info_keywords=['episode'])
    try:
        if 'policy' in kwargs:
            sb3_a2c = A2C(**kwargs,env = monitor_env, verbose=1)
        else:
            sb3_a2c = A2C('MlpPolicy',monitor_env, verbose=1, **kwargs)
    except Exception as e:
        msg = f"Se produjo un error al crear el agente de A2C: {e}"
        raise ValueError(msg)
    sb3_a2c.learn(total_timesteps=1_000_000,callback=[eval_callback,loss_callback],progress_bar=True)
    return sb3_a2c, eval_callback.episode_eval_count,eval_callback.episode_eval_reward ,loss_callback.loss_train

def get_trained_model_a2c(path):
    loaded_model = A2C.load(path)
    return loaded_model
