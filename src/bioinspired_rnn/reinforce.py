"""REINFORCE with a learned baseline (actor-critic), one episode per update.

For each episode the agent interacts with the environment until termination,
then

* the actor minimises :math:`-\\frac{1}{T}\\sum_t \\log\\pi(a_t|s_t)\\,(G_t - \\hat v_t)`
  (advantage with stopped gradient),
* the critic minimises :math:`\\frac{1}{T}\\sum_t (G_t - \\hat v_t)^2`,

each with an L2 penalty and gradient clipping (global norm 1). During the
interaction the hidden states of both networks are recorded; they are the
"firing rates" analysed in the paper.
"""

import numpy as np
import tensorflow as tf

from .envs.economic_choice import EconomicChoiceEnv


def discount_rewards(rewards, gamma):
    """Discounted returns :math:`G_t = \\sum_{k \\ge t} \\gamma^{k-t} r_k`."""
    discounted = np.zeros_like(rewards, dtype=np.float32)
    cumulative = 0.0
    for i in reversed(range(len(rewards))):
        cumulative = rewards[i] + gamma * cumulative
        discounted[i] = cumulative
    return discounted


def pad_hidden_states(hidden_states, max_steps, units):
    """Right-pads a (units, steps) array with NaN up to ``max_steps`` steps."""
    padded = np.full((units, max_steps), np.nan)
    padded[:, :hidden_states.shape[1]] = hidden_states
    return padded


def trial_record(env, done):
    """Summary of a finished Economic Choice trial.

    Returns ``[[juice_left, juice_right], (n_B, n_A), chosen_juice]`` when the
    trial ended with a left/right choice, e.g. ``[['A', 'B'], (3, 1), 'A']``
    (A on the left, offer 3B:1A, A chosen), and ``None`` otherwise (aborted
    trials, or environments other than the Economic Choice task).
    """
    env = getattr(env, "unwrapped", env)
    if not isinstance(env, EconomicChoiceEnv):
        return None
    if not done or env.chosen_action not in (1, 2):
        return None
    juice_pair = env.trial_juice_LR
    chosen_juice = juice_pair[0] if env.chosen_action == 1 else juice_pair[1]
    return [list(juice_pair), env.trial_offer_BA, chosen_juice]


def _l2_penalty(model):
    # Penalises the variables whose name contains "kernel".
    return tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_weights
                     if "kernel" in v.name or "recurrent_kernel" in v.name])


def train_agent(env, agent, num_episodes=500, gamma=0.99, print_interval=10,
                l2_actor=1e-4, l2_critic=1e-4):
    """Trains an :class:`~bioinspired_rnn.actor_critic.ActorCriticAgent`.

    Args:
        env: Gymnasium environment.
        agent: The agent; its networks must have a ``layer_type`` attribute.
        num_episodes: Number of episodes (one update per episode).
        gamma: Discount factor.
        print_interval: Print progress every this many episodes.
        l2_actor, l2_critic: L2 regularisation strengths.

    Returns:
        A 6-tuple, in this order (it is also the content of the saved ``.pkl``
        output files):

        * ``total_rewards``: total reward per episode (list).
        * ``actor_losses``, ``critic_losses``: loss per episode (lists).
        * ``actor_states``, ``critic_states``: recorded hidden states, arrays of
          shape (units, max_steps, episodes), NaN-padded; ``None`` for Dense
          networks.
        * ``trial_records``: one :func:`trial_record` per episode.
    """
    total_rewards_history = []
    actor_loss_history = []
    critic_loss_history = []
    trial_records = []

    actor_is_recurrent = "GRU" in getattr(agent.actor, "layer_type", "")
    critic_is_recurrent = "GRU" in getattr(agent.critic, "layer_type", "")
    act_size = env.action_space.n
    one_hot = np.eye(act_size, dtype=np.float32)

    actor_states_all = []
    critic_states_all = []

    for episode in range(1, num_episodes + 1):

        # --- Interaction ---
        state, _ = env.reset()
        done = False
        states, actions, rewards = [], [], []
        actor_hidden_ep, critic_hidden_ep = [], []
        current_actor_hidden = None
        current_critic_hidden = None

        while not done:
            action, _, actor_hidden = agent.select_action(
                state, actor_hidden_states=current_actor_hidden, training=True)
            if actor_is_recurrent:
                current_actor_hidden = actor_hidden
                actor_hidden_ep.append(current_actor_hidden[0].numpy().flatten())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            if actor_is_recurrent:
                r_t = actor_hidden_ep[-1]
            else:
                state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)[None, None, :]
                r_t = agent.actor.get_hidden_dense(state_tensor)[0, 0, :].numpy()

            # Critic activity at this step, carrying the critic's recurrent state.
            _, critic_hidden = agent.evaluate_critic_step(
                r_t, one_hot[actions[-1]], critic_hidden_states=current_critic_hidden,
                training=False)
            if critic_is_recurrent:
                current_critic_hidden = critic_hidden
                critic_hidden_ep.append(critic_hidden[0].numpy().flatten())

            state = next_state

        # --- Update ---
        returns = discount_rewards(rewards, gamma)
        states_seq = tf.convert_to_tensor(states, dtype=tf.float32)[None]
        actions_seq = tf.convert_to_tensor(actions, dtype=tf.int32)[None]
        returns_seq = tf.convert_to_tensor(returns, dtype=tf.float32)[None]

        if actor_is_recurrent:
            actor_features = actor_hidden_ep
        else:
            dense_hidden = agent.actor.get_hidden_dense(
                tf.convert_to_tensor(np.stack(states, axis=0), dtype=tf.float32)[None])
            actor_features = dense_hidden[0].numpy()
        critic_inputs = [np.concatenate([actor_features[t], one_hot[actions[t]]])
                         for t in range(len(actions))]
        critic_inputs = tf.convert_to_tensor([critic_inputs], dtype=tf.float32)

        with tf.GradientTape() as tape_actor:
            all_probs, _ = agent.actor(states_seq, hidden_states=None, training=True)
            probs_taken = tf.reduce_sum(all_probs * tf.one_hot(actions_seq, depth=act_size), axis=-1)
            log_probs = tf.math.log(probs_taken + 1e-10)
            values, _ = agent.critic(critic_inputs, hidden_states=None, training=True)
            advantage = returns_seq - tf.squeeze(values, axis=-1)
            actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantage))
            actor_loss += l2_actor * _l2_penalty(agent.actor)
        actor_grads = tape_actor.gradient(actor_loss, agent.actor.trainable_variables)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, clip_norm=1.0)
        agent.actor_optimizer.apply_gradients(zip(actor_grads, agent.actor.trainable_variables))

        with tf.GradientTape() as tape_critic:
            values, _ = agent.critic(critic_inputs, hidden_states=None, training=True)
            critic_loss = tf.reduce_mean(tf.square(returns_seq - tf.squeeze(values, axis=-1)))
            critic_loss += l2_critic * _l2_penalty(agent.critic)
        critic_grads = tape_critic.gradient(critic_loss, agent.critic.trainable_variables)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, clip_norm=1.0)
        agent.critic_optimizer.apply_gradients(zip(critic_grads, agent.critic.trainable_variables))

        # --- Recording ---
        if actor_is_recurrent:
            actor_states_all.append(np.array(actor_hidden_ep).T)    # units x steps
        if critic_is_recurrent:
            critic_states_all.append(np.array(critic_hidden_ep).T)
        trial_records.append(trial_record(env, done))
        total_rewards_history.append(sum(rewards))
        actor_loss_history.append(actor_loss.numpy())
        critic_loss_history.append(critic_loss.numpy())

        if episode % print_interval == 0:
            print(f"Episode {episode}\tTotal Reward: {sum(rewards):.2f}\t"
                  f"Actor Loss: {actor_loss.numpy():.4f}\tCritic Loss: {critic_loss.numpy():.4f}")

    actor_states = _stack_episodes(actor_states_all) if actor_is_recurrent else None
    critic_states = _stack_episodes(critic_states_all) if critic_is_recurrent else None

    return (total_rewards_history, actor_loss_history, critic_loss_history,
            actor_states, critic_states, trial_records)


def _stack_episodes(states_per_episode):
    """List of (units, steps_i) arrays -> (units, max_steps, episodes), NaN-padded."""
    units = states_per_episode[0].shape[0]
    max_steps = max(s.shape[1] for s in states_per_episode)
    padded = np.array([pad_hidden_states(s, max_steps, units) for s in states_per_episode])
    return np.transpose(padded, (1, 2, 0))
