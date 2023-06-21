from tf_agents.networks import network
from tf_agents.networks import utils
import tensorflow as tf

class DQNNetwork(network.Network):
    def __init__(self, input_tensor_spec, fc1_dims, fc2_dims, n_actions, **kwargs):
        super(DQNNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            **kwargs)

        self._layer1 = tf.keras.layers.Dense(fc1_dims, activation='relu')
        self._layer2 = tf.keras.layers.Dense(fc2_dims, activation='relu')
        self._layer3 = tf.keras.layers.Dense(n_actions)

    def call(self, observations, step_type=None, network_state=()):
        output = self._layer1(observations)
        output = self._layer2(output)
        output = self._layer3(output)

        return output, network_state

