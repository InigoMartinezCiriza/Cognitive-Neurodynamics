"""Unit tests of the analysis functions on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from bioinspired_rnn.analysis import behaviour, neurons
from bioinspired_rnn.task import OFFER_SETS


def synthetic_activity(tuning_fns, trials_per_offer=6, noise=0.0, seed=0):
    """Activity (trials, units) whose mean is ``fn(n_B, n_A, chosen)`` for each unit.

    The agent chooses the better juice, except in half of the 2B:1A trials.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n_b, n_a in OFFER_SETS:
        for i in range(trials_per_offer):
            chosen = "B" if n_b > 2.2 * n_a or ((n_b, n_a) == (2, 1) and i % 2) else "A"
            rows.append({"n_B": n_b, "n_A": n_a, "chosen_juice": chosen})
    trials = pd.DataFrame(rows)
    activity = np.array([[fn(r.n_B, r.n_A, r.chosen_juice) for fn in tuning_fns]
                         for r in trials.itertuples()])
    return activity + noise * rng.standard_normal(activity.shape), trials


def test_offer_window_means_uses_last_valid_steps():
    states = np.full((2, 10, 2), np.nan)
    states[:, :6, 0] = np.arange(6)          # trial of 6 steps
    states[:, :, 1] = 1.0                     # trial of 10 steps
    means = neurons.offer_window_means(states, last_steps=3)
    np.testing.assert_allclose(means[0], [4.0, 4.0])   # mean of steps 3, 4, 5
    np.testing.assert_allclose(means[1], [1.0, 1.0])


def test_trial_types_split_by_choice():
    _, trials = synthetic_activity([lambda b, a, c: 0.0])
    types = neurons.trial_types(trials)
    split = types[(types["n_B"] == 2) & (types["n_A"] == 1)]
    assert split["chosen_juice"].tolist() == ["A", "B"]
    assert split["Chosen value"].tolist() == pytest.approx([2.2, 2.0])


def test_classification_of_ideal_units():
    analysis = neurons.analyse_units(*synthetic_activity([
        lambda b, a, c: b,                                # offer value B
        lambda b, a, c: b if c == "B" else 2.2 * a,       # chosen value
        lambda b, a, c: 3.0,                              # constant
    ], noise=1e-3), min_modulation=0.0)
    fits = analysis.fits
    assert fits["type"].tolist() == ["Offer B", "Chosen value", "Unclassified"]
    assert fits.loc[0, "slope_Offer B"] > 0


def test_inactive_units():
    activity, trials = synthetic_activity([lambda b, a, c: 1e-6 * b, lambda b, a, c: 0.1 * b])

    def types(min_modulation):
        return neurons.analyse_units(activity, trials, min_modulation=min_modulation) \
            .fits["type"].tolist()

    assert types(0.0) == ["Offer B", "Offer B"]
    assert types(1e-2) == ["Inactive", "Offer B"]


def test_type_counts_has_all_types():
    analysis = neurons.UnitAnalysis(types=None, tuning=None, fits=pd.DataFrame(
        {"type": ["Offer B", "Offer B", "Unclassified", "Inactive"]}))
    counts = neurons.type_counts({("Full", "Net"): analysis})
    assert counts[("Full", "Net")].to_dict() == {
        "Offer A": 0, "Offer B": 2, "Chosen value": 0, "Unclassified": 1, "Inactive": 1,
        "Total": 4}


def test_exemplar_is_of_its_type():
    analysis = neurons.analyse_units(*synthetic_activity([
        lambda b, a, c: b + (b if c == "B" else 2.2 * a),   # mixed, classified by best R²
        lambda b, a, c: b,
    ], noise=1e-3), min_modulation=0.0)
    ex = neurons.exemplar(analysis, "Offer B")
    assert analysis.fits.loc[ex["unit"], "type"] == "Offer B"
    assert neurons.exemplar(analysis, "Offer A") is None


def test_exact_rank_sum_complete_separation():
    # 3 vs 7 completely separated samples: two-sided p = 2 / C(10, 3)
    p = neurons.exact_rank_sum_pvalue(np.array([5.0, 6.0, 7.0]), np.arange(7.0) - 10)
    assert p == pytest.approx(2 / 120)


def test_psychometric_order_and_percentages():
    trials = pd.DataFrame({
        "completed": [True, True, True, False],
        "n_B": [10, 10, 0, 10],
        "n_A": [1, 1, 1, 1],
        "chosen_juice": ["B", "A", "A", None],
    })
    table = behaviour.psychometric(trials)
    assert table["offer"].tolist() == ["0B:1A", "10B:1A"]
    assert table["percent_B"].tolist() == [0.0, 50.0]


def test_first_episode_above():
    rewards = [0, 0, 10, 10, 10]
    assert behaviour.first_episode_above(rewards, threshold=10, window=2) == 4
    assert behaviour.first_episode_above(rewards, threshold=11, window=2) is None
