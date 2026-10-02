"""The data in data/ reproduce the numbers reported in the paper."""

import pandas as pd
import pytest

from bioinspired_rnn.analysis import neurons
from bioinspired_rnn.analysis.data import NEURAL_RUNS, all_configs, load_curves, load_neural


@pytest.fixture(scope="module")
def analyses():
    out = {}
    for key, run in NEURAL_RUNS.items():
        trials, activity = load_neural(run)
        out[key] = neurons.analyse_units(activity, trials)
    return out


def test_table1(analyses):
    counts = neurons.type_counts(analyses)
    expected = {  # Table 1 of the paper
        ("Full", "Standard RNN"): [0, 14, 20, 14, 2],
        ("Full", "Modified RNN"): [2, 10, 1, 5, 32],
        ("Partial", "Standard RNN"): [0, 18, 28, 3, 1],
        ("Partial", "Modified RNN"): [1, 5, 10, 0, 34],
    }
    for column, values in expected.items():
        assert counts[column].loc[list(neurons.TYPES)].tolist() == values


def test_figure5_exemplars_are_of_their_type(analyses):
    expected = {("Full", "Standard RNN"): (29, 10), ("Full", "Modified RNN"): (46, 5),
                ("Partial", "Standard RNN"): (16, 42), ("Partial", "Modified RNN"): (26, 39)}
    for key, units in expected.items():
        for variable, unit in zip(("Offer B", "Chosen value"), units):
            exemplar = neurons.exemplar(analyses[key], variable)
            assert exemplar["unit"] == unit
            assert analyses[key].fits.loc[unit, "type"] == variable


def test_offer_slopes_rank_sum(analyses):
    # Section 4.2: Offer A |slopes| all above Offer B |slopes| in the modified RNNs
    full = neurons.offer_slopes_rank_sum(analyses[("Full", "Modified RNN")].fits)
    partial = neurons.offer_slopes_rank_sum(analyses[("Partial", "Modified RNN")].fits)
    pooled = neurons.offer_slopes_rank_sum(pd.concat(
        [a.fits for (_, net), a in analyses.items() if net == "Modified RNN"], ignore_index=True))
    for result, sizes, p in ((full, (2, 10), 0.03), (partial, (1, 5), 0.33),
                             (pooled, (3, 15), 0.002)):
        assert (result["n_a"], result["n_b"]) == sizes
        assert result["separated"] and result["median_abs_slope_a"] > result["median_abs_slope_b"]
        assert result["p_value_exact"] == pytest.approx(result["p_value_min"])
        assert round(result["p_value_exact"], 3 if p < 0.01 else 2) == p


def test_episodes_per_run():
    # Section 4
    expected = {"CP_F_ffnn": 700, "CP_F_rnn": 700, "CP_F_rnn_std": 700,
                "CP_P_ffnn": 8000, "CP_P_rnn": 8000, "CP_P_rnn_std": 1000,
                "EC_F_ffnn": 12450, "EC_F_rnn": 44200, "EC_F_rnn_std": 44200,
                "EC_P_ffnn": 43550, "EC_P_rnn": 43550, "EC_P_rnn_std": 48550}
    configs = all_configs()
    for run, episodes in expected.items():
        assert configs[run].total_episodes == episodes
        assert len(load_curves(run)) == episodes
