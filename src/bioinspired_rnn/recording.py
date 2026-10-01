"""Recording the hidden-state activity of a trained agent.

The agent plays trials without learning. The actor is run step by step,
carrying its recurrent state; once the trial is over, the critic is run over
the whole sequence of [actor hidden state, one-hot action] inputs from its
initial state, exactly as during the training updates, so both networks are
recorded with their full recurrent dynamics.
"""

import numpy as np
import tensorflow as tf

from .reinforce import trial_record


def critic_hidden_sequence(agent, actor_states, actions):
    """Hidden states of the critic's last hidden layer at every step.

    Args:
        actor_states: Array (steps, actor units).
        actions: Sequence of the actions taken.

    Returns:
        Array (steps, critic units).
    """
    one_hot = np.eye(agent.act_size, dtype=np.float32)[np.asarray(actions)]
    inputs = tf.convert_to_tensor(np.concatenate([actor_states, one_hot], axis=1)[None],
                                  dtype=tf.float32)
    output = agent.critic.input_fc(inputs, training=False)
    for layer in agent.critic.hidden_layers:
        output, _ = layer(output, training=False)
    return output[0].numpy()


def play_trial(agent, env):
    """Plays one trial with a recurrent agent.

    Returns:
        actor_states: Array (steps, units).
        critic_states: Array (steps, units).
        total_reward: Sum of the rewards.
        record: :func:`bioinspired_rnn.reinforce.trial_record` of the trial.
    """
    state, _ = env.reset()
    hidden, done, total = None, False, 0.0
    actor_states, actions = [], []
    while not done:
        action, _, hidden = agent.select_action(state, actor_hidden_states=hidden, training=False)
        actor_states.append(hidden[0].numpy().flatten())
        actions.append(action)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += reward
    actor_states = np.array(actor_states)
    critic_states = critic_hidden_sequence(agent, actor_states, actions)
    return actor_states, critic_states, total, trial_record(env, done)
