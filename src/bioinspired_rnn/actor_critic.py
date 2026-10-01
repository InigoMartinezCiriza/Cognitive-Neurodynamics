"""Actor (policy) and critic (value) networks and the agent that holds them.

Both networks share the same layout::

    Dense (ReLU)  ->  hidden layer(s)  ->  Dense output

where the hidden layer is one of

* ``"Dense"``: feedforward ReLU layer (FFNN agent),
* ``"GRU_standard"``: ``tf.keras.layers.GRU`` (standard RNN agent),
* ``"GRU_modified"``: :class:`~bioinspired_rnn.modified_gru.ModifiedGRU`
  (bioinspired RNN agent).

When ``prob_connection < 1`` the input and recurrent weights of the hidden
layer are pruned with a fixed random mask (:class:`SparseConstraint`). In the
paper only the actor is sparse (10 % of the connections); the critic is dense.

The actor maps observations to action probabilities (softmax output). As in
Song et al. (2017), the critic receives the actor's hidden state concatenated
with the one-hot action taken, and outputs a scalar value estimate.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers

from .modified_gru import ModifiedGRU
from .sparse_constraint import SparseConstraint

LAYER_TYPES = ("Dense", "GRU_standard", "GRU_modified")


def _sparse_constraint(prob_connection):
    return SparseConstraint(prob_connection) if prob_connection < 1.0 else None


def _build_hidden_layers(prefix, num_layers, hidden_size, layer_type, prob_connection, alpha):
    """Hidden layers shared by the actor and the critic."""
    if layer_type not in LAYER_TYPES:
        raise ValueError(f"Unknown layer type: {layer_type}. Use one of {LAYER_TYPES}.")

    hidden_layers = []
    for i in range(num_layers):
        name = f'{prefix}_{layer_type.lower().replace("_", "")}_{i}'
        kernel_constraint = _sparse_constraint(prob_connection)
        recurrent_constraint = _sparse_constraint(prob_connection) if "GRU" in layer_type else None

        if layer_type == "GRU_modified":
            layer = ModifiedGRU(hidden_size, return_sequences=True, return_state=True,
                                kernel_constraint=kernel_constraint,
                                recurrent_constraint=recurrent_constraint,
                                name=name, alpha=alpha)
        elif layer_type == "GRU_standard":
            layer = layers.GRU(hidden_size, activation="tanh", recurrent_activation="sigmoid",
                               return_sequences=True, return_state=True,
                               kernel_constraint=kernel_constraint,
                               recurrent_constraint=recurrent_constraint,
                               name=name)
        else:
            layer = layers.Dense(hidden_size, activation="relu",
                                 kernel_constraint=kernel_constraint, name=name)
        hidden_layers.append(layer)
    return hidden_layers


def _forward_hidden(model, inputs, hidden_states, training):
    """Input dense layer followed by the hidden layers.

    Returns the output sequence of the last hidden layer and, for recurrent
    layers, the list of final hidden states.
    """
    output = model.input_fc(inputs, training=training)
    recurrent = "GRU" in model.layer_type
    initial_states = hidden_states if (recurrent and hidden_states is not None) \
        else [None] * model.num_layers

    new_states = []
    for hidden_layer, initial_state in zip(model.hidden_layers, initial_states):
        if recurrent:
            output, state = hidden_layer(output, initial_state=initial_state, training=training)
            new_states.append(state)
        else:
            output = hidden_layer(output, training=training)
    return output, new_states


@tf.keras.utils.register_keras_serializable(package="bioinspired_rnn")
class ActorModel(Model):
    """Policy network: observations -> action probabilities.

    Args:
        input_size: Dimension of the observations.
        hidden_size: Number of units of each hidden layer.
        output_size: Number of actions.
        num_layers: Number of hidden layers.
        prob_connection: Fraction of hidden-layer connections kept.
        layer_type: ``"Dense"``, ``"GRU_standard"`` or ``"GRU_modified"``.
        alpha: :math:`\\Delta t/\\tau` of the modified GRU (ignored otherwise).
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 prob_connection=1.0, layer_type="GRU_standard", alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.prob_connection = prob_connection
        self.alpha = alpha

        self.input_fc = layers.Dense(input_size, activation="relu", name="actor_input_dense")
        self.hidden_layers = _build_hidden_layers("actor", num_layers, hidden_size, layer_type,
                                                  prob_connection, alpha)
        self.fc_out = layers.Dense(output_size, activation="softmax", name="actor_output_dense")

    def call(self, inputs, hidden_states=None, training=None):
        """Args: ``inputs`` of shape (batch, time, input_size).

        Returns:
            probs: Action probabilities, shape (batch, time, output_size).
            new_states: Final hidden state of each recurrent layer (empty for Dense).
        """
        output, new_states = _forward_hidden(self, inputs, hidden_states, training)
        probs = self.fc_out(output, training=training)
        return probs, new_states

    def get_hidden_dense(self, inputs):
        """Hidden representation of a Dense actor (used as critic input)."""
        output, _ = _forward_hidden(self, inputs, None, training=False)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "num_layers": self.num_layers,
            "prob_connection": self.prob_connection,
            "layer_type": self.layer_type,
            "alpha": self.alpha,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="bioinspired_rnn")
class CriticModel(Model):
    """Value network: [actor hidden state, one-hot action] -> value estimate.

    Args:
        actor_hidden_size: Dimension of the actor hidden representation.
        act_size: Number of actions.
        hidden_size: Number of units of each hidden layer.
        num_layers: Number of hidden layers.
        prob_connection: Fraction of hidden-layer connections kept.
        layer_type: ``"Dense"``, ``"GRU_standard"`` or ``"GRU_modified"``.
        alpha: :math:`\\Delta t/\\tau` of the modified GRU (ignored otherwise).
    """

    def __init__(self, actor_hidden_size, act_size, hidden_size, num_layers=1,
                 prob_connection=1.0, layer_type="GRU_standard", alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.input_size = actor_hidden_size + act_size
        self.actor_hidden_size = actor_hidden_size
        self.act_size = act_size
        self.hidden_size = hidden_size
        self.prob_connection = prob_connection
        self.alpha = alpha

        self.input_fc = layers.Dense(self.input_size, activation="relu", name="critic_input_dense")
        self.hidden_layers = _build_hidden_layers("critic", num_layers, hidden_size, layer_type,
                                                  prob_connection, alpha)
        self.fc_out = layers.Dense(1, name="critic_output_dense")

    def call(self, inputs, hidden_states=None, training=None):
        """Args: ``inputs`` of shape (batch, time, actor_hidden_size + act_size).

        Returns:
            value: Value estimates, shape (batch, time, 1).
            new_states: Final hidden state of each recurrent layer (empty for Dense).
        """
        output, new_states = _forward_hidden(self, inputs, hidden_states, training)
        value = self.fc_out(output, training=training)
        return value, new_states

    def get_config(self):
        config = super().get_config()
        config.update({
            "actor_hidden_size": self.actor_hidden_size,
            "act_size": self.act_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "prob_connection": self.prob_connection,
            "layer_type": self.layer_type,
            "alpha": self.alpha,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(config.pop("actor_hidden_size"), config.pop("act_size"), **config)


class ActorCriticAgent:
    """Actor-critic agent: the two networks plus their Adam optimizers.

    Args:
        obs_size: Dimension of the observations.
        act_size: Number of actions.
        actor_hidden_size, critic_hidden_size: Hidden units of each network.
        actor_layers, critic_layers: Hidden layers of each network.
        actor_lr, critic_lr: Adam learning rates.
        noise_std: Standard deviation of the Gaussian noise added to the
            observations before the actor selects an action (0 disables it).
        actor_prob_connection, critic_prob_connection: Fraction of hidden-layer
            connections kept in each network.
        layer_type: ``"Dense"``, ``"GRU_standard"`` or ``"GRU_modified"``.
        alpha: :math:`\\Delta t/\\tau` of the modified GRU.
    """

    def __init__(self, obs_size, act_size, actor_hidden_size=128, critic_hidden_size=128,
                 actor_layers=1, critic_layers=1, actor_lr=1e-3, critic_lr=1e-3, noise_std=0.0,
                 actor_prob_connection=1.0, critic_prob_connection=1.0,
                 layer_type="GRU_standard", alpha=0.1):
        self.actor = ActorModel(input_size=obs_size, hidden_size=actor_hidden_size,
                                output_size=act_size, num_layers=actor_layers,
                                prob_connection=actor_prob_connection, layer_type=layer_type,
                                alpha=alpha)
        self.critic = CriticModel(actor_hidden_size=actor_hidden_size, act_size=act_size,
                                  hidden_size=critic_hidden_size, num_layers=critic_layers,
                                  prob_connection=critic_prob_connection, layer_type=layer_type,
                                  alpha=alpha)
        self.actor_optimizer = optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = optimizers.Adam(learning_rate=critic_lr)
        self.noise_std = noise_std
        self.act_size = act_size

    def add_noise(self, state):
        """Adds zero-mean Gaussian noise of std ``noise_std`` to ``state``."""
        if self.noise_std != 0.0:
            noise = np.random.normal(0, self.noise_std, size=state.shape).astype(state.dtype)
            return state + noise
        return state

    def select_action(self, state, actor_hidden_states=None, training=True):
        """Samples an action from the policy for a single time step.

        Returns:
            action: Index of the sampled action.
            log_prob: Log-probability of that action.
            new_actor_hidden_states: Updated actor hidden states.
        """
        state_tensor = tf.convert_to_tensor(self.add_noise(state), dtype=tf.float32)
        state_tensor = state_tensor[None, None, :]

        probs, new_actor_hidden_states = self.actor(
            state_tensor, hidden_states=actor_hidden_states, training=training)

        probs_t = probs[:, 0, :]
        action = tf.random.categorical(tf.math.log(probs_t + 1e-10), num_samples=1)
        action = tf.squeeze(action, axis=-1)
        action_one_hot = tf.one_hot(tf.cast(action[0], tf.int32), depth=self.act_size)[None, :]
        log_prob = tf.math.log(tf.reduce_sum(probs_t * action_one_hot, axis=1) + 1e-10)
        return int(action.numpy()[0]), log_prob[0], new_actor_hidden_states

    def evaluate_critic_step(self, r_t, action_one_hot, critic_hidden_states=None, training=False):
        """Runs the critic for a single time step.

        Args:
            r_t: Actor hidden representation, shape (actor_hidden_size,).
            action_one_hot: One-hot encoding of the action taken, shape (act_size,).
            critic_hidden_states: Previous critic hidden states.

        Returns:
            value: Scalar value estimate.
            new_critic_hidden_states: Updated critic hidden states.
        """
        if r_t is None:
            raise ValueError("r_t is required; for a Dense actor compute it with "
                             "actor.get_hidden_dense.")
        inp = np.concatenate([r_t, action_one_hot], axis=0)[None, None, :]
        inp = tf.convert_to_tensor(inp, dtype=tf.float32)
        value, new_critic_hidden_states = self.critic(inp, hidden_states=critic_hidden_states,
                                                      training=training)
        return value[0, 0, 0], new_critic_hidden_states
