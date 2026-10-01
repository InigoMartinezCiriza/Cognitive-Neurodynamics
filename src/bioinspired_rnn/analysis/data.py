"""Processed data shipped in ``data/`` and the choices of which trials are analysed.

The learning curves and trial records are extracted from the raw training
outputs by ``scripts/extract_data.py``; the neural activity is recorded from
the trained agents by ``scripts/record_activity.py``. See ``data/README.md``
for the formats.
"""

from pathlib import Path

import pandas as pd

from ..config import all_configs, load_config

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

#: Number of steps (10 ms each) averaged to obtain the offer-period activity of a
#: trial: the last 100 steps, i.e. the last second before the choice.
OFFER_WINDOW_STEPS = 100

#: Durations of the original task, [fixation, offer_min, offer_max, decision] in ms.
ORIGINAL_DURATIONS = [1500, 1000, 2000, 2000]

#: Economic Choice runs whose hidden-unit activity is analysed (Fig. 5, Table 1).
NEURAL_RUNS = {
    ("Full", "Standard RNN"): "EC_F_rnn_std",
    ("Full", "Modified RNN"): "EC_F_rnn",
    ("Partial", "Standard RNN"): "EC_P_rnn_std",
    ("Partial", "Modified RNN"): "EC_P_rnn",
}


def psychometric_stages(config):
    """Stages whose trials enter the psychometric curves (Fig. 4).

    The stages run with the original task durations; the partial-observables
    FFNN never reached them, so all its stages are used.
    """
    stages = [i + 1 for i, stage in enumerate(config.stages)
              if list(stage.get("duration_params", [])) == ORIGINAL_DURATIONS]
    return stages or list(range(1, config.n_stages + 1))


def curves_path(run, data_dir=DATA_DIR):
    return Path(data_dir) / "training_curves" / f"{run}.csv"


def trials_path(run, data_dir=DATA_DIR):
    return Path(data_dir) / "trials" / f"{run}.csv"


def neural_path(run, data_dir=DATA_DIR, network="critic"):
    return Path(data_dir) / "neural" / f"{run}_{network}_offer_window.csv"


def load_curves(run, data_dir=DATA_DIR):
    """Per-episode total reward and losses of a run (all stages)."""
    return pd.read_csv(curves_path(run, data_dir))


def load_trials(run, data_dir=DATA_DIR, stages=None):
    """Per-episode trial records of an Economic Choice run.

    Args:
        stages: Stages to keep; ``"psychometric"`` selects
            :func:`psychometric_stages`; ``None`` keeps all.
    """
    trials = pd.read_csv(trials_path(run, data_dir))
    if stages == "psychometric":
        stages = psychometric_stages(load_config(run))
    if stages is not None:
        trials = trials[trials["stage"].isin(stages)]
    return trials.reset_index(drop=True)


def load_neural(run, data_dir=DATA_DIR, network="critic"):
    """Offer-window activity of the trained agent in its completed trials.

    Args:
        network: ``"critic"`` (value network, analysed in the paper) or ``"actor"``.

    Returns:
        trials: DataFrame with one row per completed trial (offer, choice and
            trial length).
        activity: Array (trials, units) of mean hidden-state values.
    """
    # round_trip: read back exactly the float64 values that were written
    table = pd.read_csv(neural_path(run, data_dir, network), float_precision="round_trip")
    unit_cols = [c for c in table.columns if c.startswith("unit_")]
    return table.drop(columns=unit_cols), table[unit_cols].to_numpy(dtype=float)


__all__ = ["DATA_DIR", "OFFER_WINDOW_STEPS", "NEURAL_RUNS", "all_configs", "load_config",
           "psychometric_stages", "load_curves", "load_trials", "load_neural"]
