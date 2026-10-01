"""Saving and restoring an agent between curriculum stages.

Stage ``n`` of a run with prefix ``P`` is stored in the directory ``P_n``:

* a ``tf.train.Checkpoint`` with the actor, the critic and both optimizers;
* the sparsity masks of the hidden layers, one ``.npy`` file per weight matrix
  (``stage{n}_{actor|critic}_layer{i}_{kernel|recur}.npy``), since the masks
  are not TensorFlow variables and are not part of the checkpoint.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf


def _mask_path(ckpt_dir, stage, network, layer_index, kind):
    return os.path.join(ckpt_dir, f"stage{stage}_{network}_layer{layer_index}_{kind}.npy")


def _weight_dtype(layer, cell_attr, keras_attr):
    """dtype of a hidden-layer weight (modified GRU, Keras GRU or Dense)."""
    if not hasattr(layer, "cell"):
        return layer.kernel.dtype
    name = cell_attr if hasattr(layer.cell, cell_attr) else keras_attr
    return getattr(layer.cell, name).dtype


def _checkpoint(agent):
    return tf.train.Checkpoint(actor=agent.actor, critic=agent.critic,
                               actor_optimizer=agent.actor_optimizer,
                               critic_optimizer=agent.critic_optimizer)


def build_agent(agent, obs_size, act_size):
    """Creates all the weights of the agent with a dummy forward pass."""
    agent.actor.build((None, None, obs_size))
    agent.critic.build((None, None, agent.actor.hidden_size + act_size))
    agent.actor(tf.zeros((1, 1, obs_size), dtype=tf.float32), training=False)
    agent.critic(tf.zeros((1, 1, agent.critic.input_size), dtype=tf.float32), training=False)


def _load_masks(agent, stage, ckpt_dir):
    for network, model in (("actor", agent.actor), ("critic", agent.critic)):
        for i, layer in enumerate(model.hidden_layers):
            kernel_path = _mask_path(ckpt_dir, stage, network, i, "kernel")
            if os.path.exists(kernel_path) and layer.kernel_constraint is not None:
                layer.kernel_constraint.mask = tf.constant(
                    np.load(kernel_path), dtype=_weight_dtype(layer, "W_in", "kernel"))

            recur_path = _mask_path(ckpt_dir, stage, network, i, "recur")
            if (os.path.exists(recur_path) and hasattr(layer, "cell")
                    and getattr(layer, "recurrent_constraint", None) is not None):
                layer.recurrent_constraint.mask = tf.constant(
                    np.load(recur_path), dtype=_weight_dtype(layer, "W_rec", "recurrent_kernel"))


def _initialise_optimizers(agent, obs_size, act_size):
    """One dummy update so that the optimizer slots exist before restoring."""
    dummy_obs = np.zeros((1, 1, obs_size), dtype=np.float32)
    hidden = agent.actor.input_fc(dummy_obs, training=False)
    for layer in agent.actor.hidden_layers:
        hidden = layer(hidden, training=False)[0] if "GRU" in agent.actor.layer_type \
            else layer(hidden, training=False)
    action = np.zeros((act_size,), dtype=np.float32)
    action[0] = 1.0
    dummy_critic_in = np.concatenate([hidden.numpy()[0, 0, :], action])[None, None, :]

    with tf.GradientTape(persistent=True) as tape:
        actor_out, _ = agent.actor(dummy_obs, training=True)
        critic_out, _ = agent.critic(dummy_critic_in, training=True)
        loss_actor = tf.reduce_mean(tf.square(actor_out))
        loss_critic = tf.reduce_mean(tf.square(critic_out))
    agent.actor_optimizer.apply_gradients(
        zip(tape.gradient(loss_actor, agent.actor.trainable_variables),
            agent.actor.trainable_variables))
    agent.critic_optimizer.apply_gradients(
        zip(tape.gradient(loss_critic, agent.critic.trainable_variables),
            agent.critic.trainable_variables))
    del tape


def load_model(agent, obs_size, act_size, stage, ckpt_prefix):
    """Restores into ``agent`` the networks, optimizers and masks saved at the end
    of stage ``stage - 1``, so that stage ``stage`` can continue the training.

    Returns:
        The directory of the current stage, ``f"{ckpt_prefix}_{stage}"``.
    """
    prev_ckpt_dir = f"{ckpt_prefix}_{stage - 1}"
    this_ckpt_dir = f"{ckpt_prefix}_{stage}"
    os.makedirs(this_ckpt_dir, exist_ok=True)

    build_agent(agent, obs_size, act_size)
    _load_masks(agent, stage - 1, prev_ckpt_dir)
    _initialise_optimizers(agent, obs_size, act_size)

    manager = tf.train.CheckpointManager(_checkpoint(agent), prev_ckpt_dir, max_to_keep=3)
    status = _checkpoint_restore(agent, manager.latest_checkpoint)
    status.assert_existing_objects_matched()
    print(f"Restored stage {stage - 1} from {manager.latest_checkpoint}")
    return this_ckpt_dir


def _checkpoint_restore(agent, path):
    if path is None:
        raise FileNotFoundError("No checkpoint found to restore.")
    return _checkpoint(agent).restore(path)


def _readable_by_tensorflow(path):
    """TensorFlow cannot open files under non-ASCII paths on Windows; in that
    case returns a temporary ASCII-path copy of the directory."""
    path = Path(path)
    if sys.platform != "win32" or str(path.resolve()).isascii():
        return str(path)
    copy = Path(tempfile.mkdtemp()) / path.name
    shutil.copytree(path, copy)
    return str(copy)


def restore_weights(agent, obs_size, act_size, ckpt_dir, stage):
    """Loads a trained agent (e.g. to evaluate it) from the stage directory
    ``ckpt_dir``, whose masks are named after ``stage``."""
    ckpt_dir = _readable_by_tensorflow(ckpt_dir)
    build_agent(agent, obs_size, act_size)
    _load_masks(agent, stage, ckpt_dir)
    _initialise_optimizers(agent, obs_size, act_size)
    path = tf.train.latest_checkpoint(ckpt_dir)
    status = _checkpoint_restore(agent, path)
    status.assert_existing_objects_matched()
    return agent


def save_model(agent, stage, ckpt_prefix):
    """Saves the networks, optimizers and sparsity masks at the end of a stage.

    Returns:
        Path of the checkpoint written by ``tf.train.CheckpointManager``.
    """
    this_ckpt_dir = f"{ckpt_prefix}_{stage}"
    os.makedirs(this_ckpt_dir, exist_ok=True)

    manager = tf.train.CheckpointManager(_checkpoint(agent), this_ckpt_dir, max_to_keep=3)
    path = manager.save()

    for network, model in (("actor", agent.actor), ("critic", agent.critic)):
        for i, layer in enumerate(model.hidden_layers):
            if layer.kernel_constraint is not None and layer.kernel_constraint.mask is not None:
                np.save(_mask_path(this_ckpt_dir, stage, network, i, "kernel"),
                        layer.kernel_constraint.mask.numpy())
            recurrent = getattr(layer, "recurrent_constraint", None)
            if recurrent is not None and recurrent.mask is not None:
                np.save(_mask_path(this_ckpt_dir, stage, network, i, "recur"),
                        recurrent.mask.numpy())
    print(f"Stage {stage} saved at {path}")
    return path
