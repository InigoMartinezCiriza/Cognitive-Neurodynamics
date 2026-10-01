"""Runs the curriculum stages described by a :class:`~bioinspired_rnn.config.RunConfig`.

Outputs follow the layout of the published results::

    <output_dir>/Outputs/<run>_<stage>.pkl       training records of the stage
    <output_dir>/Checkpoints/<run>_<stage>/      agent at the end of the stage

The ``.pkl`` file holds the 6-tuple returned by
:func:`bioinspired_rnn.reinforce.train_agent`.
"""

import pickle
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from .actor_critic import ActorCriticAgent
from .checkpoints import load_model, save_model
from .envs import EconomicChoiceEnv, make_cartpole
from .reinforce import train_agent


def set_seeds(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


def make_env(config, stage):
    """Environment of stage ``stage`` (1-based) of a run."""
    if config.task == "cartpole":
        env = make_cartpole(partial=config.partial, **config.stage_env_params(stage))
        env.reset(seed=config.seed)
        return env
    if config.task == "economic_choice":
        return EconomicChoiceEnv(partial=config.partial, **config.stage_env_params(stage))
    raise ValueError(f"Unknown task: {config.task}")


def make_agent(config, env):
    a = config.agent
    return ActorCriticAgent(
        obs_size=env.observation_space.shape[0],
        act_size=env.action_space.n,
        actor_hidden_size=a["actor_hidden_size"],
        critic_hidden_size=a["critic_hidden_size"],
        actor_layers=a.get("actor_layers", 1),
        critic_layers=a.get("critic_layers", 1),
        actor_lr=a["learning_rate"],
        critic_lr=a["learning_rate"],
        noise_std=a.get("noise_std", 0.0),
        actor_prob_connection=a["actor_prob_connection"],
        critic_prob_connection=a["critic_prob_connection"],
        layer_type=a["layer_type"],
        alpha=a.get("alpha", 0.1),
    )


def run_stage(config, stage, output_dir, max_episodes=None, print_interval=100):
    """Trains stage ``stage`` (1-based) of a run and saves its outputs.

    Stage 1 starts from a freshly initialised agent; later stages restore the
    checkpoint of the previous stage from ``output_dir``, including the
    optimizer state, so by default the learning rate of the whole run is
    ``agent.learning_rate``. A stage can set its own ``learning_rate``, which
    is then assigned after the restore.

    Args:
        max_episodes: Cap on the number of episodes (for quick tests).
    """
    output_dir = Path(output_dir)
    ckpt_prefix = output_dir / "Checkpoints" / config.name
    (output_dir / "Outputs").mkdir(parents=True, exist_ok=True)

    env = make_env(config, stage)
    agent = make_agent(config, env)
    obs_size, act_size = env.observation_space.shape[0], env.action_space.n
    if stage > 1:
        load_model(agent, obs_size, act_size, stage, ckpt_prefix)

    stage_cfg = config.stages[stage - 1]
    if "learning_rate" in stage_cfg:
        agent.actor_optimizer.learning_rate.assign(stage_cfg["learning_rate"])
        agent.critic_optimizer.learning_rate.assign(stage_cfg["learning_rate"])

    episodes = stage_cfg["episodes"] if max_episodes is None else min(max_episodes, stage_cfg["episodes"])
    print(f"[{config.name}] stage {stage}/{config.n_stages}: {episodes} episodes")
    results = train_agent(env=env, agent=agent, num_episodes=episodes,
                          gamma=config.training["gamma"], print_interval=print_interval,
                          l2_actor=config.training["l2_actor"],
                          l2_critic=config.training["l2_critic"])

    save_model(agent, stage, ckpt_prefix)
    with open(output_dir / "Outputs" / f"{config.name}_{stage}.pkl", "wb") as f:
        pickle.dump(results, f)
    env.close()
    return results


def run(config, output_dir, stages=None, max_episodes=None, print_interval=100):
    """Runs several stages (all by default) of a run, in order."""
    set_seeds(config.seed)
    for stage in stages or range(1, config.n_stages + 1):
        run_stage(config, stage, output_dir, max_episodes=max_episodes,
                  print_interval=print_interval)
