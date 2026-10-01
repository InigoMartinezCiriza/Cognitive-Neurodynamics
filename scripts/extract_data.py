"""Builds the processed data in ``data/`` from the raw training outputs.

The raw outputs are the ``<run>_<stage>.pkl`` files written by
``scripts/train.py`` (about 6.5 GB for all the runs of the paper, because they
contain the hidden states of every episode; see the README for where to get
them). This script keeps only what the figures and tables need:

* ``data/training_curves/<run>.csv``: reward and losses per episode;
* ``data/trials/<run>.csv``: offer and choice of every Economic Choice trial;
* ``data/sources.csv``: the raw files used.

The neural activity analysed in the paper is recorded from the trained agents
with ``scripts/record_activity.py``.

Usage::

    python scripts/extract_data.py --raw-dir path/to/Saved/Outputs
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from bioinspired_rnn.analysis.data import DATA_DIR, curves_path, trials_path
from bioinspired_rnn.config import all_configs


def load_stage(raw_dir, run, stage):
    path = Path(raw_dir) / f"{run}_{stage}.pkl"
    with open(path, "rb") as f:
        rewards, actor_losses, critic_losses, actor_states, critic_states, records = pickle.load(f)
    return path, rewards, actor_losses, critic_losses, actor_states, critic_states, records


def trial_rows(records, stage, first_episode):
    rows = []
    for i, record in enumerate(records):
        row = {"stage": stage, "episode": first_episode + i, "completed": record is not None,
               "juice_left": None, "juice_right": None, "n_B": None, "n_A": None,
               "chosen_juice": None}
        if record is not None:
            (left, right), (n_b, n_a), chosen = record
            row.update({"juice_left": left, "juice_right": right, "n_B": n_b, "n_A": n_a,
                        "chosen_juice": chosen})
        rows.append(row)
    return rows


def extract_run(config, raw_dir, data_dir):
    curves, trials, sources = [], [], []
    episode = 1
    for stage in range(1, config.n_stages + 1):
        path, rewards, actor_losses, critic_losses, actor_states, critic_states, records = \
            load_stage(raw_dir, config.name, stage)

        expected = config.stages[stage - 1]["episodes"]
        if len(rewards) != expected:
            raise ValueError(f"{path.name}: {len(rewards)} episodes, config says {expected}")
        if actor_states is not None and actor_states.shape[0] != config.agent["actor_hidden_size"]:
            raise ValueError(f"{path.name}: {actor_states.shape[0]} hidden units, config says "
                             f"{config.agent['actor_hidden_size']}")

        curves.append(pd.DataFrame({
            "stage": stage,
            "episode": np.arange(episode, episode + len(rewards)),
            "total_reward": np.asarray(rewards, dtype=float),
            "actor_loss": np.asarray(actor_losses, dtype=float),
            "critic_loss": np.asarray(critic_losses, dtype=float),
        }))
        if config.task == "economic_choice":
            trials.extend(trial_rows(records, stage, episode))

        sources.append({"run": config.name, "stage": stage, "file": path.name,
                        "bytes": path.stat().st_size, "episodes": len(rewards)})
        episode += len(rewards)
        del actor_states, critic_states

    out = curves_path(config.name, data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(curves, ignore_index=True).to_csv(out, index=False, lineterminator="\n")
    if trials:
        out = trials_path(config.name, data_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(trials)
        table[["n_B", "n_A"]] = table[["n_B", "n_A"]].astype("Int64")
        table.to_csv(out, index=False, lineterminator="\n")
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", required=True, help="directory with the <run>_<stage>.pkl files")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="output directory (default: data/)")
    parser.add_argument("--runs", nargs="*", help="runs to extract (default: all configs)")
    args = parser.parse_args()

    configs = all_configs()
    sources = []
    for name in args.runs or configs:
        print(f"Extracting {name} ...")
        sources.extend(extract_run(configs[name], args.raw_dir, args.data_dir))

    sources_path = Path(args.data_dir) / "sources.csv"
    if sources_path.exists() and args.runs:
        old = pd.read_csv(sources_path)
        sources = old[~old["run"].isin(args.runs)].to_dict("records") + sources
    pd.DataFrame(sources).sort_values(["run", "stage"]).to_csv(sources_path, index=False,
                                                               lineterminator="\n")


if __name__ == "__main__":
    main()
