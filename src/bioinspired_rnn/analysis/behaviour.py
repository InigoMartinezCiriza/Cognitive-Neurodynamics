"""Behavioural results: learning curves (Fig. 3) and psychometric curves (Fig. 4)."""

import numpy as np
import pandas as pd

from ..task import A_TO_B_RATIO, OFFER_SETS


def rolling_stats(rewards, window):
    """Moving average and moving median of the total reward per episode."""
    rewards = pd.Series(np.asarray(rewards, dtype=float))
    return rewards.rolling(window).mean(), rewards.rolling(window).median()


def first_episode_above(rewards, threshold, window):
    """First episode (1-based) at which the moving average reaches ``threshold``
    (``None`` if it never does)."""
    mean, _ = rolling_stats(rewards, window)
    hits = np.flatnonzero(mean.to_numpy() >= threshold)
    return int(hits[0]) + 1 if hits.size else None


def psychometric(trials, rel_value_A=A_TO_B_RATIO):
    """Percentage of trials in which juice B was chosen, per offer type.

    Args:
        trials: Trial records (columns ``completed``, ``n_B``, ``n_A``,
            ``chosen_juice``); aborted trials are ignored.

    Returns:
        DataFrame with one row per offer type, sorted by decreasing relative
        value of A (``rel_value_A * #A / #B``), as in Fig. 4.
    """
    done = trials[trials["completed"]]
    rows = []
    for n_b, n_a in OFFER_SETS:
        sel = done[(done["n_B"] == n_b) & (done["n_A"] == n_a)]
        if sel.empty:
            continue
        rows.append({
            "offer": f"{n_b}B:{n_a}A",
            "n_B": n_b,
            "n_A": n_a,
            "relative_value_A": rel_value_A * n_a / n_b if n_b else np.inf,
            "trials": len(sel),
            "percent_B": 100.0 * (sel["chosen_juice"] == "B").mean(),
        })
    table = pd.DataFrame(rows)
    return table.sort_values("relative_value_A", ascending=False, kind="stable").reset_index(drop=True)
