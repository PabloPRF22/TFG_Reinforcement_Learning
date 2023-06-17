import reverb
import tensorflow as tf

from tqdm import tqdm
from tf_agents.utils import common
from tf_agents.specs import tensor_spec
from tf_agents.drivers import py_driver
from tf_agents.networks import sequential
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_utils
from tf_agents.policies import epsilon_greedy_policy, py_tf_eager_policy
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import reverb_replay_buffer
from algorithms.tf_agents.networkDQN import DQNNetwork
from utils import plot_train_results
from pathlib import Path

class DQNTFA:

    def __init__(self, env_name: str, hparams: dict):
        # Set hyperparameters
        self.num_iterations = hparams.get('num_iterations', 20_000)
        self.initial_collect_steps = hparams.get('initial_collect_steps', 100)
        self.collect_steps_per_iteration = hparams.get('collect_steps_per_iteration', 1)
        self.replay_buffer_max_length = hparams.get('replay_buffer_max_length', 100_000)
        self.batch_size = hparams.get('batch_size', 64)
        self.learning_rate = hparams.get('learning_rate', 0.005)
        self.log_interval = hparams.get('log_interval', 200)
        self.num_eval_episodes = hparams.get('num_eval_episodes', 10)
        self.eval_interval = hparams.get('eval_interval', 1_000)
        self.fc_layer_params = hparams.get('fc_layer_params', (256,128))
        self.epsilon_decay_duration = hparams.get('epsilon_decay_duration',500) # For epsilon decay
        self.initial_epsilon = hparams.get('initial_epsilon', 1.0) # For epsilon decay
        self.final_epsilon = hparams.get('final_epsilon', 0.01) # For epsilon decay
        self._initialize_envs(env_name)
        self.create_networks()
        self._create_optimizer()
        self._create_agent()
        self._create_replay_buffer_and_observer()
        self._create_driver_and_dataset()
    def learn(self):
        self.agent.train = common.function(self.agent.train)
        self.agent.train_step_counter.assign(0)
        lostArray = []
        rewardArray = []
        episilonArray = []
        time_step = self.train_py_env.reset()
        collect_driver = py_driver.PyDriver(
            self.env,
            py_tf_eager_policy.PyTFEagerPolicy(
                self.agent.collect_policy, use_tf_function=True),
            [self.rb_observer],
            max_steps=self.collect_steps_per_iteration)

        for i in range(self.num_iterations):
            time_step, _ = collect_driver.run(time_step)


            experience, unused_info = next(self.iterator)
            train_loss = self.agent.train(experience).loss

            step = self.agent.train_step_counter.numpy()

            if step % self.log_interval == 0:
                epsilon_value = self.agent.collect_policy._epsilon
                episilonArray.append(epsilon_value)
                print('step = {0}: loss = {1}, epsilon = {2}'.format(step, train_loss, epsilon_value))
            if step % self.eval_interval == 0:
                reward = self.evaluate()                
                rewardArray.append(reward)
                print("step = {0}: loss = {1}, Average reward = {2}".format(step, train_loss,reward))
                if(reward>=200.0):
                    break
                
        savepath = Path('Taxi-2v3-results.png')

        plot_train_results(rewardArray,episilonArray,savepath,"LOL")
        
                



    def evaluate(self) -> float:
        return self.compute_avg_return(self.eval_env, self.agent.policy, self.num_eval_episodes)

    def _initialize_envs(self, env_name: str):
        self.env = suite_gym.load(env_name)
        self.train_py_env = suite_gym.load(env_name)
        self.eval_py_env = suite_gym.load(env_name)
        self.train_env = tf_py_environment.TFPyEnvironment(self.train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(self.eval_py_env)

        self.action_tensor_spec = tensor_spec.from_spec(self.env.action_spec())
        self.num_actions = self.action_tensor_spec.maximum - \
            self.action_tensor_spec.minimum + 1

    @staticmethod
    def dense_layer(num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))

    def create_networks(self):
        self.q_net =  DQNNetwork(
            input_tensor_spec=self.train_env.observation_spec(),
            fc1_dims=self.fc_layer_params[0],
            fc2_dims=self.fc_layer_params[1],
            n_actions=self.num_actions)
        self.target_net =  DQNNetwork(
            input_tensor_spec=self.train_env.observation_spec(),
            fc1_dims=self.fc_layer_params[0],
            fc2_dims=self.fc_layer_params[1],
            n_actions=self.num_actions)
        

    def _create_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _create_agent(self):
        self.train_step_counter = tf.Variable(0)
        self.agent = dqn_agent.DqnAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            q_network=self.q_net,
            optimizer=self.optimizer,
            target_update_tau = 1e-3,
            target_update_period = 2,
            gamma = 0.99,
            target_q_network = self.target_net,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.train_step_counter)
        self.agent.initialize()
        self.eval_policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy
        self.random_policy = random_tf_policy.RandomTFPolicy(
            self.train_env.time_step_spec(), self.train_env.action_spec())
        

    @staticmethod
    def compute_avg_return(environment, policy, num_episodes=10):
        total_return = 0.0
        for _ in tqdm(range(num_episodes), desc='Evaluating agent'):
            time_step, episode_return = environment.reset(), 0.0
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    def _create_replay_buffer_and_observer(self):
        self.table_name = 'uniform_table'
        self.replay_buffer_signature = tensor_spec.from_spec(self.agent.collect_data_spec)
        self.replay_buffer_signature = tensor_spec.add_outer_dim(self.replay_buffer_signature)

        self.table = reverb.Table(
            self.table_name,
            max_size=self.replay_buffer_max_length,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=self.replay_buffer_signature)
        self.reverb_server = reverb.Server([self.table])

        self.replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            self.agent.collect_data_spec,
            table_name=self.table_name,
            sequence_length=2,
            local_server=self.reverb_server)

        self.rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            self.replay_buffer.py_client,
            self.table_name,
            sequence_length=2)

    def _create_driver_and_dataset(self):
        py_driver.PyDriver(
            self.env,
            py_tf_eager_policy.PyTFEagerPolicy(
                self.random_policy, use_tf_function=True),
            [self.rb_observer],
            max_steps=self.initial_collect_steps).run(self.train_py_env.reset())

        # Dataset generates trajectories with shape [Bx2x...]
        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.batch_size,
            num_steps=2).prefetch(3)

        self.iterator = iter(self.dataset)
