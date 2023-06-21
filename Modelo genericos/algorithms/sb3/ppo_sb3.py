import gym

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from algorithms.sb3.customCallbacks.CustomEvalCallBack import CustomEvalCallBack
from algorithms.sb3.customCallbacks.LossTrainCallBack import LossTrainCallBack
from stable_baselines3.common.env_util import make_vec_env

def train_sb3_ppo(env: gym.Env, eval_env: gym.Env,reward_threshold:int,env_name:str, **kwargs) -> BaseAlgorithm:
    total = 10_000_000
    print("ememem")
    if env_name != "HumanoidPyBulletEnv-v0":
        total = 1_000_000
        
    eval_callback = CustomEvalCallBack(eval_env,reward_threshold,10)
    loss_callback = LossTrainCallBack()
    try:
        if 'policy' in kwargs:
            sb3_ppo = PPO(**kwargs,env = env, verbose=1)
        else:
            sb3_ppo = PPO(env = env,policy='MlpPolicy', verbose=1, **kwargs)
    except Exception as e:
        msg = f"Se produjo un error al crear el agente de PPO: {e}"
        raise ValueError(msg)
    sb3_ppo.learn(total_timesteps=total,callback=[eval_callback,loss_callback],progress_bar =True)
    return sb3_ppo, eval_callback.episode_eval_count,eval_callback.episode_eval_reward ,loss_callback.loss_train

def get_trained_model_ppo(path):
    loaded_model = PPO.load(path)
    return loaded_model

def transferLearning(pathLunar,env_name):
    lunar_model = get_trained_model_ppo(pathLunar)

    # Crea el nuevo entorno
    mountain_env = make_vec_env(env_name, n_envs=8)

    # Crea un nuevo modelo para el entorno MountainCar
    mountain_model = PPO('MlpPolicy', mountain_env, verbose=1)

    # Transfiere los pesos de las primeras capas del modelo LunarLander al modelo MountainCar
    for lunar_param, mountain_param in zip(lunar_model.policy.parameters(), mountain_model.policy.parameters()):
        print(f'lunar_param: {lunar_param.shape}')
        print(f'mountain_param: {mountain_param.shape}')
        if lunar_param.shape == mountain_param.shape:
            mountain_param.data.copy_(lunar_param.data)
            print("ememememem")
        else:
            break  # Solo copiamos los pesos de las capas que tienen la misma forma

    
    
