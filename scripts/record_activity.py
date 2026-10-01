"""Records the offer-period activity of the trained Economic Choice RNNs.

Each trained agent (``checkpoints/``) plays ``--trials`` trials of the original
task without learning. For every completed trial the hidden state of each unit
of the critic and of the actor is averaged over the last 100 steps of the
trial, and saved to ``data/neural/<run>_{critic,actor}_offer_window.csv``.

Usage::

    python scripts/record_activity.py                 # the four RNN runs
    python scripts/record_activity.py EC_P_rnn --trials 100
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd  # noqa: E402

from bioinspired_rnn.analysis.data import (DATA_DIR, NEURAL_RUNS,  # noqa: E402
                                           OFFER_WINDOW_STEPS, neural_path)
from bioinspired_rnn.checkpoints import restore_weights  # noqa: E402
from bioinspired_rnn.config import load_config  # noqa: E402
from bioinspired_rnn.recording import play_trial  # noqa: E402
from bioinspired_rnn.training import make_agent, make_env, set_seeds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def record(run, trials, seed):
    config = load_config(run)
    stage = config.n_stages
    env = make_env(config, stage)
    agent = make_agent(config, env)
    restore_weights(agent, env.observation_space.shape[0], env.action_space.n,
                    REPO_ROOT / "checkpoints" / f"{run}_{stage}", stage)
    set_seeds(seed)
    env.reset(seed=seed)

    rows = {"critic": [], "actor": []}
    for trial in range(1, trials + 1):
        actor_states, critic_states, _, record = play_trial(agent, env)
        if record is None:              # aborted trial
            continue
        (left, _right), (n_b, n_a), chosen = record
        info = {"trial": trial, "n_B": n_b, "n_A": n_a, "juice_left": left,
                "chosen_juice": chosen, "n_steps": len(actor_states)}
        for network, states in (("critic", critic_states), ("actor", actor_states)):
            window = states[-OFFER_WINDOW_STEPS:].mean(axis=0)
            rows[network].append({**info, **{f"unit_{u:02d}": v for u, v in enumerate(window)}})
        if trial % 100 == 0:
            print(f"[{run}] {trial}/{trials} trials")
    return {network: pd.DataFrame(r) for network, r in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", default=list(NEURAL_RUNS.values()))
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    for run in args.runs:
        tables = record(run, args.trials, args.seed)
        for network, table in tables.items():
            path = neural_path(run, args.data_dir, network)
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(path, index=False, lineterminator="\n")
        n = len(tables["critic"])
        print(f"[{run}] {n} completed trials ({n / args.trials:.0%}) -> {path.parent}")


if __name__ == "__main__":
    main()
