r"""Bioinspired (modified) GRU unit, implemented as a Keras cell and layer.

For an input :math:`x_t` and previous hidden state :math:`h_{t-1}` the cell
computes (Fig. 1b of the paper)

.. math::

    \lambda_t &= \sigma(W_{\lambda,\mathrm{rec}} h_{t-1} + W_{\lambda,\mathrm{in}} x_t + b_\lambda) \\
    \gamma_t  &= \sigma(W_{\gamma,\mathrm{rec}} h_{t-1} + W_{\gamma,\mathrm{in}} x_t + b_\gamma) \\
    \tilde h_t &= W_\mathrm{rec} (\gamma_t \odot h_{t-1}) + W_\mathrm{in} x_t + b \\
    h_t &= \left[ h_{t-1} + \alpha\, \lambda_t \odot (\tilde h_t - h_{t-1}) \right]_+

with three differences with respect to a standard GRU:

1. *Continuous-time dynamics*: the state is updated incrementally, as the Euler
   discretisation of a leaky neuron with :math:`\alpha = \Delta t / \tau`.
2. *Leak gate* :math:`\lambda_t`: modulates the effective time constant.
3. *Threshold-linear output*: the state is rectified, so it can be read as a
   firing rate.

In the code the gate :math:`\lambda_t` is called ``l`` and :math:`\gamma_t` is
called ``g``; the weight names (``W_in``, ``Wl_rec``, ...) are kept as in the
trained checkpoints.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="bioinspired_rnn")
class ModifiedGRUCell(tf.keras.layers.Layer):
    """Single step of the modified GRU.

    Args:
        units: Number of units.
        alpha: Discretisation step over time constant, :math:`\\Delta t/\\tau`.
        kernel_constraint: Constraint applied to the input weights
            (``W_in``, ``Wl_in``, ``Wg_in``).
        recurrent_constraint: Constraint applied to the recurrent weights
            (``W_rec``, ``Wl_rec``, ``Wg_rec``).
    """

    def __init__(self, units, alpha, kernel_constraint=None, recurrent_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.alpha = alpha
        self.state_size = units
        self.output_size = units
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.recurrent_constraint = tf.keras.constraints.get(recurrent_constraint)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        def input_weight(name):
            return self.add_weight(shape=(input_dim, self.units), initializer="glorot_uniform",
                                   name=name, constraint=self.kernel_constraint)

        def recurrent_weight(name):
            return self.add_weight(shape=(self.units, self.units), initializer="orthogonal",
                                   name=name, constraint=self.recurrent_constraint)

        def bias(name):
            return self.add_weight(shape=(self.units,), initializer="zeros", name=name)

        # Candidate state
        self.W_in = input_weight("W_in")
        self.W_rec = recurrent_weight("W_rec")
        self.b = bias("b")
        # Leak gate (lambda)
        self.Wl_in = input_weight("Wl_in")
        self.Wl_rec = recurrent_weight("Wl_rec")
        self.bl = bias("bl")
        # Recurrent gate (gamma)
        self.Wg_in = input_weight("Wg_in")
        self.Wg_rec = recurrent_weight("Wg_rec")
        self.bg = bias("bg")

        super().build(input_shape)

    def call(self, inputs, states):
        h_prev = states[0]

        l = tf.sigmoid(tf.matmul(h_prev, self.Wl_rec) + tf.matmul(inputs, self.Wl_in) + self.bl)
        g = tf.sigmoid(tf.matmul(h_prev, self.Wg_rec) + tf.matmul(inputs, self.Wg_in) + self.bg)
        candidate = tf.matmul(g * h_prev, self.W_rec) + tf.matmul(inputs, self.W_in) + self.b

        h_new = h_prev + self.alpha * l * (candidate - h_prev)
        h_new = tf.maximum(0.0, h_new)
        return h_new, [h_new]

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "alpha": self.alpha,
            "kernel_constraint": tf.keras.constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": tf.keras.constraints.serialize(self.recurrent_constraint),
        })
        return config


@tf.keras.utils.register_keras_serializable(package="bioinspired_rnn")
class ModifiedGRU(tf.keras.layers.Layer):
    """Recurrent layer built on :class:`ModifiedGRUCell`, with the API of
    ``tf.keras.layers.GRU`` (``return_sequences``, ``return_state``,
    ``initial_state``, constraints, ``stateful``).
    """

    def __init__(self, units, return_sequences=False, return_state=False, alpha=0.1,
                 kernel_constraint=None, recurrent_constraint=None, **kwargs):
        stateful = kwargs.pop("stateful", False)
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.alpha = alpha
        self.stateful = stateful
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.recurrent_constraint = tf.keras.constraints.get(recurrent_constraint)

        self.cell = ModifiedGRUCell(units, alpha=alpha,
                                    kernel_constraint=self.kernel_constraint,
                                    recurrent_constraint=self.recurrent_constraint,
                                    name=self.name + "_cell")
        self.rnn_layer = tf.keras.layers.RNN(self.cell,
                                             return_sequences=return_sequences,
                                             return_state=return_state,
                                             stateful=stateful,
                                             name=self.name + "_rnn")

    def build(self, input_shape):
        if not self.rnn_layer.built:
            self.rnn_layer.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, initial_state=None, training=None, mask=None):
        return self.rnn_layer(inputs, initial_state=initial_state, training=training, mask=mask)

    @property
    def states(self):
        return self.rnn_layer.states if self.stateful else None

    def get_initial_state(self, inputs):
        return self.rnn_layer.get_initial_state(inputs)

    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError("Layer must be stateful.")
        self.rnn_layer.reset_states(states)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
            "alpha": self.alpha,
            "stateful": self.stateful,
            "kernel_constraint": tf.keras.constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": tf.keras.constraints.serialize(self.recurrent_constraint),
        })
        return config
