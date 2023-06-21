

from algorithms.sarsa import SARSAAgent

from algorithms.sb3.dqn_sb3 import train_sb3_dqn
from algorithms.sb3.dqn_sb3 import get_trained_model_dqn
from algorithms.sb3.ppo_sb3  import transferLearning

from algorithms.sb3.ppo_sb3 import train_sb3_ppo
from algorithms.sb3.ppo_sb3 import get_trained_model_ppo

from algorithms.sb3.trpo_sb3 import train_sb3_trpo
from algorithms.sb3.trpo_sb3 import get_trained_model_trpo

from algorithms.sb3.a2c_sb3 import train_sb3_a2c
from algorithms.sb3.a2c_sb3 import get_trained_model_a2c

from algorithms.rllib.dqn_rllib import DQNrllib
from algorithms.rllib.ppo_rllib import PPOrllib

from algorithms.sarsa import SARSAAgent
from algorithms.q_learning import QAgent


from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from utils import plot_train_results
from utils import plot_mean_reward_per_algorithm
from utils import plot_train_eval_results
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from pyvirtualdisplay import Display
from gym import wrappers


import os
import ast
from pathlib import Path
import pybulletgym  # register PyBullet enviroments with open ai gym
import gym
from tqdm import tqdm

def evaluate_agent(env_name,
                  agent,
                  num_episodes: int = 50,
                  seed: int = 0) -> float:
   env = gym.make(env_name)
   set_random_seed(seed)
   env.seed(seed)
   total_rewards = []
   for _ in tqdm(range(num_episodes), desc='Evaluating agent'):
       state, done = env.reset(), False
       total_reward = 0
       while not done:
           if(env_name != "HumanoidPyBulletEnv-v0"):
              action = int(agent.predict(state, deterministic=True)[0])
           else:
              action = agent.predict(state, deterministic=True)[0]
           state, reward, done, _ = env.step(action)
           total_reward += reward
       total_rewards.append(total_reward)
   return sum(total_rewards) / len(total_rewards)


def train_and_evaluate_classic_rl_agents(env_name):
   env = gym.make(env_name)


   # ------------------------ Q LEARNING ------------------------ #
   q_agent = QAgent(env=env,env_name=env_name)
   q_agent.learn(total_episodes=10_000)
   q_agent_mean_reward = evaluate_agent(env_name, agent=q_agent)


   # ------------------------ SARSA ------------------------ #
   sarsa_agent = SARSAAgent(env=env,env_name=env_name)
   sarsa_agent.learn(total_episodes=10_000)
   sarsa_agent_mean_reward = evaluate_agent(env_name, agent=sarsa_agent)


   print(f'Q Learning mean reward: {q_agent_mean_reward}')
   print(f'SARSA mean reward: {sarsa_agent_mean_reward}')


   output = [
       ('Q Learning', q_agent_mean_reward),
       ('SARSA', sarsa_agent_mean_reward)
   ]
   return q_agent_mean_reward,sarsa_agent_mean_reward




def train_and_evaluate_sb3_agents(env_name,reward_threshold,a2c_hyperparams,n_envs_a2c,dqn_hyperparams,n_envs_dqn,ppo_hyperparams,n_envs_ppo,trpo_hyperparams,n_envs_trpo):
   print("Training SB3 Agents for " + env_name)
   rewardsAverageArray = []
   titulos = []
   lossArray = []
   episodes = []
   rewardEvalArray = []
   # ------------------------ stable-baselines3 A2C ------------------------ #
   if(a2c_hyperparams != None):
      print("--------*-*-*-*   A2C TRAINING   *-*-*-*--------")
      vec_env_dqn = make_vec_env(env_name, n_envs=n_envs_a2c)
      eval_env = gym.make(env_name)
      sb3_a2c, episodes_a2c,eval_reward_a2c,loss_train_a2c= train_sb3_a2c(env=vec_env_dqn,eval_env=eval_env,reward_threshold = reward_threshold,**a2c_hyperparams)
      sb3_a2c_mean_reward = evaluate_agent(env_name, agent=sb3_a2c)
      print(f'stable-baselines3 A2C mean reward: {sb3_a2c_mean_reward}')
      path = "algorithms/sb3/trained_models/"+ env_name + "_trained_a2c.zip"
      sb3_a2c.save(path)
      titulos.append("A2C")
      episodes.append(episodes_a2c)
      rewardEvalArray.append(eval_reward_a2c)
      lossArray.append(loss_train_a2c)
      rewardsAverageArray.append(sb3_a2c_mean_reward)
   
   
   # ------------------------ stable-baselines3 DQN ------------------------ #
   if(dqn_hyperparams !=None):
      print("--------*-*-*-*   DQN TRAINING   *-*-*-*--------")
      vec_env_dqn = make_vec_env(env_name, n_envs=n_envs_dqn)
      sb3_dqn, episodes_dqn,eval_reward_dqn,loss_train_dqn = train_sb3_dqn(env=vec_env_dqn,eval_env=eval_env,reward_threshold = reward_threshold,**dqn_hyperparams)
      sb3_dqn_mean_reward = evaluate_agent(env_name, agent=sb3_dqn)
      print(f'stable-baselines3 DQN mean reward: {sb3_dqn_mean_reward}')
      path = "algorithms/sb3/trained_models/"+ env_name + "_trained_dqn.zip"
      sb3_dqn.save(path)  
      titulos.append("DQN")
      episodes.append(episodes_dqn)
      rewardEvalArray.append(eval_reward_dqn)
      lossArray.append(loss_train_dqn)
      rewardsAverageArray.append(sb3_dqn_mean_reward)
   
   # ------------------------ stable-baselines3 PPO ------------------------ #
   if(ppo_hyperparams !=None):
      print("--------*-*-*-*   PPO TRAINING   *-*-*-*--------")
      vec_env_ppo = make_vec_env(env_name, n_envs=n_envs_ppo)
      eval_env = gym.make(env_name)
      # Aplicar la normalización
      if isinstance(eval_env.observation_space,gym.spaces.Box):
         vec_env_ppo = VecNormalize(vec_env_ppo)
      sb3_ppo, episodes_ppo,eval_reward_ppo,loss_train_ppo = train_sb3_ppo(env=vec_env_ppo,eval_env=eval_env,reward_threshold = reward_threshold,env_name = env_name,**ppo_hyperparams)
      sb3_ppo_mean_reward = evaluate_agent(env_name, agent=sb3_ppo)
      print(f'stable-baselines3 PPO mean reward: {sb3_ppo_mean_reward}')
      path = "algorithms/sb3/trained_models/"+ env_name + "trained_ppo.zip"
      sb3_ppo.save(path)
      titulos.append("PPO")
      episodes.append(episodes_ppo)
      rewardEvalArray.append(eval_reward_ppo)
      lossArray.append(loss_train_ppo)
      rewardsAverageArray.append(sb3_ppo_mean_reward)
   
   # ------------------------ stable-baselines3 TRPO ------------------------ #
   if(trpo_hyperparams !=None):
      print("--------*-*-*-*   TRPO TRAINING   *-*-*-*--------")
      vec_env_trpo = make_vec_env(env_name, n_envs=n_envs_trpo)
      eval_env = make_vec_env(env_name, n_envs=8)
      sb3_trpo, episodes_trpo,eval_reward_trpo,loss_train_trpo = train_sb3_trpo(env=vec_env_trpo,eval_env=eval_env,reward_threshold = reward_threshold,env_name = env_name,**trpo_hyperparams)
      sb3_trpo_mean_reward = evaluate_agent(env_name, agent=sb3_trpo)
      print(f'stable-baselines3 TRPO mean reward: {sb3_trpo_mean_reward}')
      path = "algorithms/sb3/trained_models/"+ env_name + "trained_trpo.zip"
      sb3_trpo.save(path)
      titulos.append("TRPO")
      episodes.append(episodes_trpo)
      rewardEvalArray.append(eval_reward_trpo)
      lossArray.append(loss_train_trpo)
      rewardsAverageArray.append(sb3_trpo_mean_reward)
   
   plot_train_results(lossArray,'results/' +env_name + "_loss.png",titulos,env_name)
   plot_train_eval_results(rewardEvalArray,'results/' +env_name + "_rewards.png",titulos,env_name)
   return titulos,rewardEvalArray,rewardsAverageArray


def train_and_evaluate_rllib_agents(env_name,reward_threshold):
   results: List[Tuple[str, float]] = []
   savepath = Path('results/' +env_name+'_average_rewards_rllib.png')
   
   # ------------------------ RLLIB DQN ------------------------ #
   rllib_dqn = DQNrllib(env_name)
   rllib_dqn.learn()
   rllib_dqn_mean_reward = evaluate_agent(env_name, agent=rllib_dqn)
   # ------------------------ rllib PPO ------------------------ #
   rllib_ppo = PPOrllib(env_name)
   rllib_ppo.learn()
   rllib_ppo_mean_reward = evaluate_agent(env_name, agent=rllib_ppo)


   print(f'rllib DQN mean reward: {rllib_dqn_mean_reward}')
   print(f'rllib PPO mean reward: {rllib_ppo_mean_reward}')
   results.append(("DQN", rllib_dqn_mean_reward))
   results.append(("PPO", rllib_ppo_mean_reward))
   
   plot_mean_reward_per_algorithm(results, env_name, savepath, reward_threshold)

   
def lunarlander_v2_experiment():
   # https://www.gymlibrary.dev/environments/classic_control/acrobot/
   env_name = 'LunarLander-v2'
   hyperparameters_sets_dqn = [
        {
            'num_iterations': 2000,
            'initial_collect_steps': 1000,
            'collect_steps_per_iteration': 10,
            'replay_buffer_max_length': 1_000_000,
            'batch_size': 64,
            'learning_rate': 0.0001,
            'epsilon_decay_duration':500,
            'log_interval': 5,
            'num_eval_episodes': 30,
            'eval_interval': 10,
            'fc_layer_params': (64,32)
        },
   ]
   dqn_hyperparams_sb3 = {
        'batch_size': 128,
        'buffer_size': 50000,
        'exploration_final_eps': 0.1,
        'exploration_fraction': 0.12,
        'gamma': 0.99,
        'gradient_steps': -1,
        'learning_rate': 0.00063,
        'learning_starts': 0,
        'policy_kwargs': dict(net_arch=[256, 256]),
        'target_update_interval': 250,
        'train_freq': 4,
   }
   ppo_hyperparams_sb3 = {
      'batch_size': 64,
      'ent_coef': 0.01,
      'gae_lambda': 0.98,
      'gamma': 0.999,
      'n_epochs': 4,
      'n_steps': 1024,
      'normalize_advantage': False
   }
   
   a2c_hyperparams_sb3 = {
    'ent_coef': 1e-5,
    'gamma': 0.995,
    'learning_rate': 0.00083,
    'n_steps': 5
    }
   trpo_hyperparams_sb3  = {
    'cg_damping': 0.01,
    'gae_lambda': 0.98,
    'gamma': 0.99,
    'learning_rate': 0.001,
    'n_critic_updates': 15,
    'n_steps': 512,
   }
   dqn_config_rrlib = {
    "gamma": 0.99,
    "lr": 0.001,
    "buffer_size": 10000, 
    "learning_starts": 1000, 
    "timesteps_per_iteration": 1000,
    "train_batch_size": 32,
    "exploration_config": {
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.0,
        "final_epsilon": 0.01,
        "epsilon_timesteps": 10000,
    },
    "model": {
        "fcnet_hiddens": [64, 64],
    },
    "framework": "torch"
   }
   
   


   algoritmosTitulos,sb3TrainRewards,sb3RewardsAveragen = train_and_evaluate_sb3_agents(env_name,200,a2c_hyperparams_sb3,8,dqn_hyperparams_sb3,4,ppo_hyperparams_sb3,16,trpo_hyperparams_sb3,2)
   #plot_train_results(sb3TrainRewards,env_name + "_rewards.png",algoritmosTitulos,env_name)
   #tfa_results = train_and_evaluate_tf_agents(env_name,hyperparameters_sets_dqn)
   


def taxi_v3_experiment():
   # https://www.gymlibrary.dev/environments/toy_text/taxi/
   env_name = 'Taxi-v3'
   classic_rl_results = train_and_evaluate_classic_rl_agents(env_name)
   results = classic_rl_results
   #train_and_evaluate_rllib_agents(env_name,7)

   dqn_hyperparams_sb3 = {
        'batch_size': 128,
        'buffer_size': 50000,
        'exploration_final_eps': 0.1,
        'exploration_fraction': 0.12,
        'gamma': 0.99,
        'gradient_steps': -1,
        'learning_rate': 0.00063,
        'learning_starts': 0,
        'policy_kwargs': dict(net_arch=[256, 256]),
        'target_update_interval': 250,
        'train_freq': 4,
   }
   ppo_hyperparams_sb3 = {
      'batch_size': 64,
      'ent_coef': 0.01,
      'gae_lambda': 0.98,
      'gamma': 0.999,
      'n_epochs': 4,
      'n_steps': 1024,
      'normalize_advantage': False
   }
   a2c_hyperparams_sb3 = {
    'ent_coef': 1e-5,
    'gamma': 0.995,
    'learning_rate': 0.00083,
    'n_steps': 5
    }
   trpo_hyperparams_sb3  = {
    'cg_damping': 0.01,
    'gae_lambda': 0.98,
    'gamma': 0.99,
    'learning_rate': 0.001,
    'n_critic_updates': 15,
    'n_steps': 512,
   }
   dqn_config_rrlib = {
    "gamma": 0.99,
    "lr": 0.001,
    "buffer_size": 10000, 
    "learning_starts": 1000, 
    "timesteps_per_iteration": 1000,
    "train_batch_size": 32,
    "exploration_config": {
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.0,
        "final_epsilon": 0.01,
        "epsilon_timesteps": 10000,
    },
    "model": {
        "fcnet_hiddens": [64, 64],
    },
    "framework": "torch"
   }
   algoritmosTitulos,sb3TrainRewards,sb3RewardsAveragen = train_and_evaluate_sb3_agents(env_name,20,a2c_hyperparams_sb3,8,dqn_hyperparams_sb3,4,ppo_hyperparams_sb3,16,trpo_hyperparams_sb3,2)
   
def humanoid_experiment():
   env_name = 'HumanoidPyBulletEnv-v0'
   eval_env = gym.make(env_name)
   ppo_hyperparams_sb3 = {
    "n_steps": 2048,
    "batch_size": 64,
    "gae_lambda": 0.95,
    "gamma": 0.99,
    "n_epochs": 10,
    "ent_coef": 0.0,
    "learning_rate": 2.5e-4,
    "clip_range": 0.2
   }

   trpo_hyperparams_sb3  = {
    'cg_damping': 0.01,
    'gae_lambda': 0.98,
    'gamma': 0.99,
    'learning_rate': 0.001,
    'n_critic_updates': 15,
    'n_steps': 512,
   }
   trpo_hyperparams_sb3 = {
    'batch_size': 128,
    'n_steps': 1024,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'sub_sampling_factor': 1,
    'cg_max_steps': 25,
    'cg_damping': 0.1,
    'n_critic_updates': 20,
    'learning_rate': 0.001
   }
   algoritmosTitulos,sb3TrainRewards,sb3RewardsAveragen = train_and_evaluate_sb3_agents(env_name,1000,None,8,None,4,ppo_hyperparams_sb3,16,trpo_hyperparams_sb3,2)

def evaluate_trained_models(env_name,reward_threshold):
   results: List[Tuple[str, float]] = []
   path = 'results/' +env_name+'_average_rewards_sb3.png'
   savepath = Path(path)
   dqn_path = "algorithms/sb3/trained_models/"+ env_name + "_trained_dqn.zip"
   ppo_path = "algorithms/sb3/trained_models/"+ env_name + "_trained_ppo.zip"
   trpo_path = "algorithms/sb3/trained_models/"+ env_name + "_trained_trpo.zip"
   a2c_path = "algorithms/sb3/trained_models/"+ env_name + "_trained_a2c.zip"
   trained = False
   if os.path.exists(dqn_path):
      print("DQN_"+env_name+"_EVALUATE")
      trained = True
      sb3_dqn = get_trained_model_dqn(dqn_path)
      sb3_dqn_mean_reward = evaluate_agent(env_name, agent=sb3_dqn,num_episodes=100)
      results.append(("DQN", sb3_dqn_mean_reward))
   
   if os.path.exists(ppo_path):
      trained = True
      print("PPO_"+env_name+"_EVALUATE")
      sb3_ppo = get_trained_model_ppo(ppo_path)
      sb3_ppo_mean_reward = evaluate_agent(env_name, agent=sb3_ppo,num_episodes=100)
      results.append(("PPO", sb3_ppo_mean_reward))
   
   if os.path.exists(trpo_path):
      trained = True
      print("TRPO_"+env_name+"_EVALUATE")
      sb3_trpo = get_trained_model_trpo(trpo_path)
      sb3_trpo_mean_reward = evaluate_agent(env_name, agent=sb3_trpo,num_episodes=100)
      results.append(("TRPO", sb3_trpo_mean_reward))
   
   if os.path.exists(a2c_path):
      trained = True
      print("A2C_"+env_name+"_EVALUATE")
      sb3_a2c = get_trained_model_a2c(a2c_path)
      sb3_a2c_mean_reward = evaluate_agent(env_name, agent=sb3_a2c,num_episodes=100)
      results.append(("A2C", sb3_a2c_mean_reward))
   if(env_name == "Taxi-v3"):
      trained = True
      q_learning, sarsa = train_and_evaluate_classic_rl_agents(env_name)
      results.append(("Q-Learning", q_learning))
      results.append(("SARSA", sarsa))
   if(trained):
      plot_mean_reward_per_algorithm(results, env_name, savepath, reward_threshold)
      print(f"Se ha generado un png con el gráfico de las recompensas medias para cada algoritmo para el entorno {env_name} en la ruta {path}")
   else: 
      print(f"No hay modelos entrenados para {env_name}")


   
def train_custom_env(env_name):
    try:
        algorithms = ["A2C","DQN","PPO","TRPO"]
        hiperparametros = []
        discrete = False
        if(env_name == 'KukaDiverseObjectEnv'):
            env = KukaDiverseObjectEnv()
        else:        
            env = gym.make(env_name)
            
        recompensa = input("Introduce el umbral de recompensa para dar por finalizado el entrenamiento: ")
        try:
            recompensa = float(recompensa)
        except ValueError:
            raise ValueError("Introduce un umbral de recompensa válido")
        if isinstance(env.action_space, gym.spaces.Discrete):
            discrete = True
        for alg in algorithms:
            if(alg == "DQN" and not discrete):
                hiperparametros.append(None)
            else:
                input_dict_str = input(f"Introduce hiperparametros para el algoritmo {alg} \nSi no quiere entrenar este modelo déjalo en blanco: ")
                
                if(input_dict_str == ""):
                    hiperparametros.append(None)
                else:
                    input_dict = ast.literal_eval(input_dict_str)
                    print(isinstance(input_dict, dict))
                    if not isinstance(input_dict, dict):
                        raise ValueError(f"Los parametros introducidos para {alg} no son un diccionario bien formado")
                    hiperparametros.append(input_dict)
        train_and_evaluate_sb3_agents(env_name,recompensa,hiperparametros[0],8,hiperparametros[1],4,hiperparametros[2],16,hiperparametros[3],2)
                

    except ValueError as e:
        print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
        print(f"Se produjo un error: {e}")
        print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")

    except Exception as e:
        print(f"Se produjo un error al crear el entorno:  {e}")
        
def render_env_trained(custom_env):
    env = gym.make("LunarLander-v2")
    env.reset()
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample())  # toma una acción aleatoria
    env.close()
def main():
    while True:
        print("\nSeleccione la opción que desea ejecutar que desea ejecutar: ")
        print("1. Entrenar modelos para Taxi-v3")
        print("2. Entrenar modelos paraLunarLander-v2")
        print("3. Entrenar modelos para HumanoidPyBulletEnv-v0")
        print("4. Evaluar modelos entrenados para LunarLander-v2")
        print("5. Evaluar modelos entrenados para Taxi-v3")
        print("6. Evaluar modelos entrenados para HumanoidPyBulletEnv-v0")
        print("7. Evaluar modelos entrenados para Taxi-v3, LunarLander, HumanoidPyBulletEnv-v0")
        print("8. Entrenar modelos para un entorno personalizado")
        print("9. Evaluar modelos para un entorno personalizado")

        print("Introduzca 'q' para salir.")

        choice = input("Introduce el número de la opción: ")

        if choice.lower() == 'q':
            break

        elif choice == "1":
            taxi_v3_experiment()
        elif choice == "2":
            lunarlander_v2_experiment()
        elif choice == "3":
            humanoid_experiment()
        elif choice == "4":
            evaluate_trained_models("LunarLander-v2", 200)
        elif choice == "5":
            evaluate_trained_models("Taxi-v3", 7)
        elif choice == "6":
            evaluate_trained_models("HumanoidPyBulletEnv-v0", 200)
        elif choice == "7":
            taxi_v3_experiment()
            lunarlander_v2_experiment()
            humanoid_experiment()
        elif choice == "8":
            envs = gym.envs.registry.all()
            for env in envs:
                print(env)
            custom_env = input("Introduce el nombre del entorno personalizado: ")
            train_custom_env(custom_env)  # Asume que el número de iteraciones para entornos personalizados es 200
        elif choice == "9":
            custom_env = input("Introduce el nombre del entorno personalizado: ")
            umbral = input("Introduce el umbral de recompensa del entorno: ")
            evaluate_trained_models(custom_env,umbral)  # Asume que el número de iteraciones para entornos personalizados es 200
        
        elif choice == "11":
            #path = "algorithms/sb3/trained_models/LunarLander-v2_trained_ppo.zip"
            custom_env = input("Introduce el nombre del entorno que quieres renderizar: ")
            render_env_trained(custom_env)
            #transferLearning(path,custom_env)
        else:
            print("Opción no válida. Por favor, introduce un número del 1 al 9 o 'q' para salir.")




if __name__ == '__main__':
   main()

