"""Runs a trained agent (no learning) on the environment of its last stage.

Examples::

    python scripts/evaluate.py EC_F_rnn --episodes 500
    python scripts/evaluate.py CP_F_ffnn --episodes 20

The agents of the paper are in ``checkpoints/`` (last stage of each run). For
the Economic Choice task the script prints the psychometric table of the
evaluation trials.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bioinspired_rnn.analysis.behaviour import psychometric  # noqa: E402
from bioinspired_rnn.checkpoints import restore_weights  # noqa: E402
from bioinspired_rnn.config import load_config  # noqa: E402
from bioinspired_rnn.reinforce import trial_record  # noqa: E402
from bioinspired_rnn.training import make_agent, make_env, set_seeds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def evaluate(config, ckpt_dir, episodes):
    stage = config.n_stages
    env = make_env(config, stage)
    agent = make_agent(config, env)
    restore_weights(agent, env.observation_space.shape[0], env.action_space.n,
                    ckpt_dir, stage)

    rewards, records = [], []
    for _ in range(episodes):
        state, _ = env.reset()
        hidden, done, total = None, False, 0.0
        while not done:
            action, _, hidden = agent.select_action(state, actor_hidden_states=hidden,
                                                    training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        rewards.append(total)
        records.append(trial_record(env, done))
    return np.array(rewards), records


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run name, e.g. EC_F_rnn")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--checkpoint", help="stage directory (default: checkpoints/<run>_<last stage>)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.run)
    ckpt_dir = args.checkpoint or REPO_ROOT / "checkpoints" / f"{config.name}_{config.n_stages}"
    set_seeds(args.seed)
    rewards, records = evaluate(config, ckpt_dir, args.episodes)

    print(f"{config.description}: {args.episodes} episodes, mean reward {rewards.mean():.3f}")
    if config.task == "economic_choice":
        trials = pd.DataFrame([
            {"completed": r is not None,
             "n_B": r[1][0] if r else None, "n_A": r[1][1] if r else None,
             "chosen_juice": r[2] if r else None} for r in records])
        print(f"Completed trials: {trials['completed'].mean():.1%}")
        print(psychometric(trials)[["offer", "trials", "percent_B"]].to_string(index=False))


if __name__ == "__main__":
    main()
