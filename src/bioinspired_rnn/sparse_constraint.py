"""Fixed random sparsity mask applied to a weight matrix."""

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="bioinspired_rnn")
class SparseConstraint(tf.keras.constraints.Constraint):
    """Keras constraint that keeps a fixed random subset of the connections.

    The binary mask is drawn lazily the first time the constraint is applied,
    with each connection kept independently with probability
    ``prob_connection``. Afterwards the mask never changes, so the same
    connections stay pruned for the whole training. Keras applies constraints
    after every optimizer step; the mask can be saved and restored through the
    ``mask`` attribute (see :mod:`bioinspired_rnn.checkpoints`).

    Args:
        prob_connection: Probability of keeping each connection, in (0, 1].
    """

    def __init__(self, prob_connection):
        if not 0.0 < prob_connection <= 1.0:
            raise ValueError("prob_connection must be in (0, 1].")
        self.prob_connection = prob_connection
        self.mask = None

    def __call__(self, w):
        if self.mask is None:
            mask = np.random.rand(*w.shape) < self.prob_connection
            self.mask = tf.constant(mask, dtype=w.dtype)
        if self.mask.shape != w.shape:
            raise ValueError(
                f"Mask shape {self.mask.shape} does not match weight shape {w.shape}."
            )
        return w * self.mask

    def get_config(self):
        return {"prob_connection": self.prob_connection}
