import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.agents.ppo import ppo_agent
from tf_agents.replay_buffers import TFUniformReplayBuffer
from tf_agents.drivers import dynamic_step_driver

class PPOAgentTFA:

    def __init__(self, env_name: str, hparams: dict):
        # Set hyperparameters
        self.num_iterations = hparams.get('num_iterations', 20_000) #Episodios
        self.batch_size = hparams.get('batch_size', 64) #Buscar
        self.lr = hparams.get('lr', 0.01) #Buscar
        self.ent_coef = hparams.get('ent_coef', 0.01) #entropy_regularization
        self.importance_ratio_clipping = hparams.get('importance_ratio_clipping', 0.98) #lambda_value
        self.use_gae = hparams.get('use_gae', True) #lambda_value
        self.gamma = hparams.get('gamma', 0.999) #discount_factor
        self.n_envs = hparams.get('n_envs', 16)
        self.n_epochs = hparams.get('n_epochs', 16) #num_epochs
        self.n_steps = hparams.get('n_steps', 1024)
        self.n_timesteps = hparams.get('n_timesteps', 1000000)
        self.fc_layer_params = hparams.get('fc_layer_params', (64,64))
        self.initialize_envs(env_name)
        self.create_actor_network()
        self.create_value_network()
        self.create_optimizer()
        self.create_agent()


    def create_actor_network(self):
        self.actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            fc_layer_params=self.fc_layer_params,
            activation_fn=tf.keras.activations.relu
        )

    def create_value_network(self):
        self.value_net = value_network.ValueNetwork(
            self.train_env.observation_spec(),
            fc_layer_params=self.fc_layer_params,
            activation_fn=tf.keras.activations.relu
        )

    def create_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def create_agent(self):
        train_step_counter = tf.compat.v2.Variable(0)
        vf_coef = 0.5
        max_grad_norm = None
        self.agent = ppo_agent.PPOAgent(
            time_step_spec=self.train_env.time_step_spec(),
            action_spec=self.train_env.action_spec(),
            optimizer=self.optimizer,
            actor_net=self.actor_net,
            value_net=self.value_net,
            use_gae=self.use_gae,
            importance_ratio_clipping=self.importance_ratio_clipping,
            discount_factor=self.gamma,
            num_epochs=self.n_epochs,
            use_td_lambda_return=True,
            train_step_counter=train_step_counter,
            value_pred_loss_coef=vf_coef,
            gradient_clipping=max_grad_norm,
            debug_summaries=False,
            summarize_grads_and_vars=False
        )
        self.agent.initialize()
        self.replay_buffer = TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.n_timesteps
        )

        self.driver = dynamic_step_driver.DynamicStepDriver(
            self.train_env,
            self.agent.collect_policy,
            observers=[self.replay_buffer.add_batch],
            num_steps=self.n_steps
        )
    def initialize_envs(self, env_name: str):
        self.env = suite_gym.load(env_name)
        self.train_py_env = suite_gym.load(env_name)
        self.eval_py_env = suite_gym.load(env_name)
        self.train_env = tf_py_environment.TFPyEnvironment(self.train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(self.eval_py_env)


