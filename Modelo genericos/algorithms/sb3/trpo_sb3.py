import gym

from sb3_contrib import TRPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from algorithms.sb3.customCallbacks.CustomEvalCallBack import CustomEvalCallBack
from algorithms.sb3.customCallbacks.LossTrainCallBack import LossTrainCallBack

def train_sb3_trpo(env: gym.Env,eval_env: gym.Env,reward_threshold:int,env_name:str,**kwargs) -> BaseAlgorithm:
    monitor_env2 = VecMonitor(eval_env, "logsEval/", info_keywords=['episode'])
    eval_callback = CustomEvalCallBack(eval_env,reward_threshold,10)
    loss_callback = LossTrainCallBack()
    monitor_env = VecMonitor(env, "logs/", info_keywords=['episode'])
    total = 10_000_000
    if env_name != "HumanoidPyBulletEnv-v0":
        total = 1_000_000
    try:
        if 'policy' in kwargs:
            sb3_trpo = PPO(**kwargs,env = env, verbose=1)
        else:
            sb3_trpo = TRPO('MlpPolicy', monitor_env, verbose=1, **kwargs)
    except Exception as e:
        msg = f"Se produjo un error al crear el agente de TRPO: {e}"
        raise ValueError(msg)
    sb3_trpo.learn(total_timesteps=total,callback=[eval_callback,loss_callback],progress_bar=True)
    return sb3_trpo, eval_callback.episode_eval_count,eval_callback.episode_eval_reward ,loss_callback.loss_train


def get_trained_model_trpo(path):
    loaded_model = TRPO.load(path)
    return loaded_model
