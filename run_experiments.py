

from algorithms.sarsa import SARSAAgent

from algorithms.sb3.dqn_sb3 import train_sb3_dqn
from algorithms.sb3.dqn_sb3 import get_trained_model_dqn

from algorithms.sb3.ppo_sb3 import train_sb3_ppo
from algorithms.sb3.ppo_sb3 import get_trained_model_ppo

from algorithms.sb3.trpo_sb3 import train_sb3_trpo
from algorithms.sb3.trpo_sb3 import get_trained_model_trpo

from algorithms.sb3.a2c_sb3 import train_sb3_a2c
from algorithms.sb3.a2c_sb3 import get_trained_model_a2c

from algorithms.rllib.dqn_rllib import DQNrllib
from algorithms.sarsa import SARSAAgent
from algorithms.q_learning import QAgent

from pathlib import Path
import pybulletgym  # register PyBullet enviroments with open ai gym
import gym
from tqdm import tqdm

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

from utils import plot_train_results
from utils import plot_mean_reward_per_algorithm


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
           action = int(agent.predict(state, deterministic=True)[0])
           state, reward, done, _ = env.step(action)
           total_reward += reward
       total_rewards.append(total_reward)
   return sum(total_rewards) / len(total_rewards)

def train_and_evaluate_classic_rl_agents(env_name):
   env = gym.make(env_name)


   # ------------------------ Q LEARNING ------------------------ #
   q_agent = QAgent(env=env)
   q_agent.learn(total_episodes=10_000)
   q_agent_mean_reward = evaluate_agent(env_name, agent=q_agent)


   # ------------------------ SARSA ------------------------ #
   sarsa_agent = SARSAAgent(env=env)
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


   # ------------------------ stable-baselines3 A2C ------------------------ #
   vec_env_dqn = make_vec_env(env_name, n_envs=n_envs_a2c)
   eval_env = gym.make(env_name)
   sb3_a2c,rewards_a2c = train_sb3_a2c(env=vec_env_dqn,eval_env=eval_env,reward_threshold = reward_threshold,**{})
   sb3_a2c_mean_reward = evaluate_agent(env_name, agent=sb3_a2c)
   print(f'stable-baselines3 A2C mean reward: {sb3_a2c_mean_reward}')
   path = "algorithms/sb3/trained_models/"+ env_name + "trained_a2c.zip"
   sb3_a2c.save(path)
   
   # ------------------------ stable-baselines3 DQN ------------------------ #
   vec_env_dqn = make_vec_env(env_name, n_envs=n_envs_dqn)
   eval_env = gym.make(env_name)
   sb3_dqn,rewards_dqn = train_sb3_dqn(env=vec_env_dqn,eval_env=eval_env,reward_threshold = reward_threshold,**dqn_hyperparams)
   sb3_dqn_mean_reward = evaluate_agent(env_name, agent=sb3_dqn)
   print(f'stable-baselines3 DQN mean reward: {sb3_dqn_mean_reward}')
   path = "algorithms/sb3/trained_models/"+ env_name + "trained_dqn.zip"
   sb3_dqn.save(path)
   # ------------------------ stable-baselines3 PPO ------------------------ #
   vec_env_ppo = make_vec_env(env_name, n_envs=n_envs_dqn)
   eval_env = gym.make(env_name)
   sb3_ppo,rewards_ppo = train_sb3_ppo(env=vec_env_ppo,eval_env=eval_env,reward_threshold = reward_threshold,**ppo_hyperparams)
   sb3_ppo_mean_reward = evaluate_agent(env_name, agent=sb3_ppo)
   print(f'stable-baselines3 PPO mean reward: {sb3_ppo_mean_reward}')
   path = "algorithms/sb3/trained_models/"+ env_name + "trained_ppo.zip"
   sb3_ppo.save(path)
   
   # ------------------------ stable-baselines3 TRPO ------------------------ #
   vec_env_trpo = make_vec_env(env_name, n_envs=n_envs_trpo)
   eval_env = gym.make(env_name)
   sb3_trpo,rewards_trpo = train_sb3_trpo(env=vec_env_trpo,eval_env=eval_env,reward_threshold = reward_threshold,**trpo_hyperparams)
   sb3_trpo_mean_reward = evaluate_agent(env_name, agent=sb3_trpo)
   print(f'stable-baselines3 TRPO mean reward: {sb3_trpo_mean_reward}')
   path = "algorithms/sb3/trained_models/"+ env_name + "trained_trpo.zip"
   sb3_trpo.save(path)
   
   titulos = ["A2C-SB3","DQN-SB3","PPO-SB3","TRPO-SB3"]
   rewardsAverageArray = [sb3_a2c_mean_reward,sb3_dqn_mean_reward,sb3_ppo_mean_reward,sb3_ppo_mean_reward]
   print(rewardsAverageArray)
   rewardsArray = [rewards_a2c,rewards_dqn,rewards_ppo,rewards_trpo]
   plot_train_results(rewardsArray,env_name + "_rewards.png",titulos,env_name)

   return titulos,rewardsArray,rewardsAverageArray


def train_and_evaluate_rllib_agents(env_name,dqn_hyperparams):

   # ------------------------ RLLIB DQN ------------------------ #
   rllib_dqn = DQNrllib(env_name,dqn_hyperparams)
   rllib_dqn.learn()
   rllib_dqn_mean_reward = evaluate_agent(env_name, agent=rllib_dqn)

   
def lunarlander_v2_experiment():
   # https://www.gymlibrary.dev/environments/classic_control/acrobot/
   env_name = 'LunarLander-v2'
   #sb3_results = train_and_evaluate_sb3_agents(env_name)
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
   #train_and_evaluate_rllib_agents("LunarLander-v2",dqn_config_rrlib)
   
   


   algoritmosTitulos,sb3TrainRewards,sb3RewardsAveragen = train_and_evaluate_sb3_agents(env_name,200,a2c_hyperparams_sb3,8,dqn_hyperparams_sb3,4,ppo_hyperparams_sb3,16,trpo_hyperparams_sb3,2)
   #plot_train_results(sb3TrainRewards,env_name + "_rewards.png",algoritmosTitulos,env_name)

   #rllib_results = train_and_evaluate_rllib_agents(env_name)
   #tfa_results = train_and_evaluate_tf_agents(env_name,hyperparameters_sets_dqn)
   


def taxi_v3_experiment():
   # https://www.gymlibrary.dev/environments/toy_text/taxi/
   env_name = 'Taxi-v3'
   classic_rl_results = train_and_evaluate_classic_rl_agents(env_name)
   results = classic_rl_results 
   savepath = Path('Taxi-v3-results.png')
   plot_mean_reward_per_algorithm(results, env_name, savepath)
   yperparameters_sets_dqn = [
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
   algoritmosTitulos,sb3TrainRewards,sb3RewardsAveragen = train_and_evaluate_sb3_agents(env_name,20,a2c_hyperparams_sb3,8,dqn_hyperparams_sb3,4,ppo_hyperparams_sb3,16,trpo_hyperparams_sb3,2)

   
def humanoid_experiment():
   env_name = 'HumanoidPyBulletEnv-v0'
   vec_env_ppo = make_vec_env(env_name, n_envs=16)
   eval_env = gym.make(env_name)
   ppo_hyperparams_sb3 = {
      'batch_size': 256,
      'ent_coef': 0.01,
      'gae_lambda': 0.98,
      'gamma': 0.999,
      'n_epochs': 4,
      'n_steps': 1024,
      'policy_kwargs': dict(net_arch=[1024,516,256,128]),

      'normalize_advantage': False
   }
   algoritmosTitulos,sb3TrainRewards,sb3RewardsAveragen = train_and_evaluate_sb3_agents(env_name,200,{},8,{},4,ppo_hyperparams_sb3,16,{},2)

   sb3_ppo,rewards_ppo = train_sb3_ppo(env=vec_env_ppo,eval_env=eval_env,**{})
   sb3_ppo_mean_reward = evaluate_agent(env_name, agent=sb3_ppo)
   print(f'stable-baselines3 PPO mean reward: {sb3_ppo_mean_reward}')
   path = "algorithms/sb3/trained_models/"+ env_name + "trained_ppo.zip"
   sb3_ppo.save(path)

def evaluate_trained_models(env_name,reward_threshold):
   results: List[Tuple[str, float]] = []
   savepath = Path(env_name+'_average_rewards_sb3.png')
   print("DQN_"+env_name+"_EVALUATE")
   sb3_dqn = get_trained_model_dqn("algorithms/sb3/trained_models/"+ env_name + "trained_dqn.zip")
   sb3_dqn_mean_reward = evaluate_agent(env_name, agent=sb3_dqn,num_episodes=100)
   results.append(("DQN", sb3_dqn_mean_reward))
   print("PPO_"+env_name+"_EVALUATE")
   sb3_ppo = get_trained_model_ppo("algorithms/sb3/trained_models/"+ env_name + "trained_ppo.zip")
   sb3_ppo_mean_reward = evaluate_agent(env_name, agent=sb3_ppo,num_episodes=100)
   results.append(("PPO", sb3_ppo_mean_reward))
   print("TRPO_"+env_name+"_EVALUATE")
   sb3_trpo = get_trained_model_trpo("algorithms/sb3/trained_models/"+ env_name + "trained_trpo.zip")
   sb3_trpo_mean_reward = evaluate_agent(env_name, agent=sb3_trpo,num_episodes=100)
   results.append(("TRPO", sb3_trpo_mean_reward))
   print("A2C_"+env_name+"_EVALUATE")

   sb3_a2c = get_trained_model_a2c("algorithms/sb3/trained_models/"+ env_name + "trained_a2c.zip")
   sb3_a2c_mean_reward = evaluate_agent(env_name, agent=sb3_a2c,num_episodes=100)
   results.append(("A2C", sb3_a2c_mean_reward))
   if(env_name == "Taxi-v3"):
      q_learning, sarsa = train_and_evaluate_classic_rl_agents(env_name)
      results.append(("Q-Learning", q_learning))
      results.append(("SARSA", sarsa))

   plot_mean_reward_per_algorithm(results, env_name, savepath, reward_threshold)


   


def main():
   #taxi_v3_experiment()
   #cartpole_v1_experiment()
   #lunarlander_v2_experiment()
   #evaluate_trained_models("LunarLander-v2")
   evaluate_trained_models("Taxi-v3",7)

   #humanoid_experiment()



if __name__ == '__main__':
   main()

