import reverb
import tensorflow as tf

from tqdm import tqdm
from tf_agents.utils import common
from tf_agents.specs import tensor_spec
from tf_agents.drivers import py_driver
from tf_agents.agents.ppo import ppo_agent
from tf_agents.environments import suite_gym
from tf_agents.replay_buffers import reverb_utils
from tf_agents.policies import py_tf_eager_policy
from tf_agents.environments import tf_py_environment
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.networks import actor_distribution_network


class PPOTFA:

    num_iterations = 750
    collect_episodes_per_iteration = 2
    replay_buffer_capacity = 2000
    fc_layer_params = (100,)
    learning_rate = 1e-3
    log_interval = 25
    num_eval_episodes = 10
    eval_interval = 50

    def __init__(self, env_name: str):
        self._create_envs(env_name)
        self._create_actor()
        self._create_optimizer()
        self._create_agent()
        self._create_replay_buffer_and_observer()

    def learn(self):
        self.agent.train = common.function(self.agent.train)
        self.agent.train_step_counter.assign(0)

        for _ in range(self.num_iterations):
            self._collect_episode(
                self.train_py_env, self.agent.collect_policy,
                self.collect_episodes_per_iteration)

            iterator = iter(self.replay_buffer.as_dataset(sample_batch_size=1))
            trajectories, _ = next(iterator)
            train_loss = self.agent.train(experience=trajectories)

            self.replay_buffer.clear()
            step = self.agent.train_step_counter.numpy()

            if step % self.log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    def evaluate(self) -> float:
        return self.compute_avg_return(self.eval_env, self.agent.policy, 100)

    def _create_envs(self, env_name: str):
        self.env = suite_gym.load(env_name)
        self.train_py_env = suite_gym.load(env_name)
        self.eval_py_env = suite_gym.load(env_name)
        self.train_env = tf_py_environment.TFPyEnvironment(self.train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(self.eval_py_env)

    def _create_actor(self):
        self.actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            fc_layer_params=self.fc_layer_params)

    def _create_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _create_agent(self):
        self.train_step_counter = tf.Variable(0)
        self.agent = reinforce_agent.ReinforceAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            actor_network=self.actor_net,
            optimizer=self.optimizer,
            normalize_returns=True,
            train_step_counter=self.train_step_counter)
        self.agent.initialize()
        self.method = ppo_agent
        self.eval_policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy

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
            max_size=self.replay_buffer_capacity,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=self.replay_buffer_signature)
        self.reverb_server = reverb.Server([self.table])

        self.replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            self.agent.collect_data_spec,
            table_name=self.table_name,
            sequence_length=None,
            local_server=self.reverb_server)

        self.rb_observer = reverb_utils.ReverbAddEpisodeObserver(
            self.replay_buffer.py_client,
            self.table_name,
            self.replay_buffer_capacity
        )

    def _collect_episode(self, environment, policy, num_episodes):
        self.driver = py_driver.PyDriver(
            environment,
            py_tf_eager_policy.PyTFEagerPolicy(
                policy, use_tf_function=True),
            [self.rb_observer],
            max_episodes=num_episodes)
        self.initial_time_step = environment.reset()
        self.driver.run(self.initial_time_step)
