"""Classification of hidden units by the economic variable they encode.

Procedure (Section 4.2 of the paper, following Padoa-Schioppa & Assad 2006 and
Padoa-Schioppa 2009):

1. **Offer-period activity.** For every completed trial, the hidden state of
   each unit is averaged over the last ``OFFER_WINDOW_STEPS`` steps of the
   trial (:func:`offer_window_means`).
2. **Trial types.** Trials are grouped by *trial type*: offer (``#B:#A``) and
   juice chosen. Trial types with fewer than ``min_trials`` trials are
   discarded (:func:`trial_types`).
3. **Tuning curve.** The activity of each unit is averaged over the trials of
   each trial type and Min-Max normalised per unit.
4. **Regressions.** The tuning curve is regressed (OLS) on three variables:

   * *Offer value A*: drops of A offered,
   * *Offer value B*: drops of B offered,
   * *Chosen value*: value of the juice chosen, in units of B
     (``#B`` if B is chosen, ``2.2 #A`` if A is chosen).

5. **Classification.** Units whose tuning curve spans less than
   ``min_modulation`` (in hidden-state units, before normalisation) are
   considered inactive. Among the variables whose slope is significant
   (p < ``p_threshold``), an active unit is assigned to the one with the
   highest R²; otherwise it is *Unclassified*.

Table 1 counts the units of each type (:func:`type_counts`), and the slopes of
the Offer A and Offer B units are compared with a Wilcoxon rank-sum test
(:func:`offer_slopes_rank_sum`). With few units the exact test cannot reach
small p-values: when the two groups are completely separated, the p-value is
the minimum attainable for their sizes (:func:`min_rank_sum_pvalue`).
"""

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy.stats import linregress, mannwhitneyu

from ..task import A_TO_B_RATIO, OFFER_SETS

VARIABLES = ("Offer A", "Offer B", "Chosen value")
TYPES = VARIABLES + ("Unclassified", "Inactive")
OFFER_LABELS = [f"{n_b}:{n_a}" for n_b, n_a in OFFER_SETS]

#: Default minimum span of a unit's tuning curve (hidden-state units).
MIN_MODULATION = 1e-3


# ----------------------------------------------------------------- activity
def offer_window_means(hidden_states, last_steps=100):
    """Mean activity of each unit over the last ``last_steps`` steps of each trial.

    Args:
        hidden_states: Recorded states, shape (units, max_steps, trials),
            NaN-padded after the end of each trial.

    Returns:
        Array (trials, units); rows of trials without valid steps are NaN.
    """
    n_units, _, n_trials = hidden_states.shape
    means = np.full((n_trials, n_units), np.nan)
    for i in range(n_trials):
        trial = hidden_states[:, :, i]
        valid = np.flatnonzero(~np.isnan(trial[0]))
        if valid.size == 0:
            continue
        last = valid.max()
        window = trial[:, max(0, last - last_steps + 1):last + 1]
        if window.shape[1] and not np.all(np.isnan(window)):
            means[i] = np.nanmean(window, axis=1)
    return means


def trial_types(trials, min_trials=2, rel_value_A=A_TO_B_RATIO):
    """Trial types (offer and choice) present in ``trials``.

    Args:
        trials: DataFrame with columns ``n_B``, ``n_A`` and ``chosen_juice``.
        min_trials: Minimum number of trials of a trial type.

    Returns:
        DataFrame with one row per trial type, ordered by offer (as in
        ``OFFER_SETS``) and choice (A first), with columns ``n_B``, ``n_A``,
        ``chosen_juice``, ``trials``, ``offer_index`` and the value of each
        variable.
    """
    counts = trials.groupby(["n_B", "n_A", "chosen_juice"]).size().rename("trials").reset_index()
    order = {offer: i for i, offer in enumerate(OFFER_SETS)}
    counts["offer_index"] = [order[(b, a)] for b, a in zip(counts["n_B"], counts["n_A"])]
    counts = counts[counts["trials"] >= min_trials]
    counts = counts.sort_values(["offer_index", "chosen_juice"]).reset_index(drop=True)
    counts["Offer A"] = counts["n_A"].astype(float)
    counts["Offer B"] = counts["n_B"].astype(float)
    counts["Chosen value"] = np.where(counts["chosen_juice"] == "B", counts["n_B"],
                                      rel_value_A * counts["n_A"]).astype(float)
    return counts


# ------------------------------------------------------------ classification
@dataclass
class UnitAnalysis:
    """Tuning and classification of the units of one network.

    Attributes:
        types: Trial types (see :func:`trial_types`), the columns of ``tuning``.
        tuning: Normalised tuning curves, array (units, trial types).
        fits: One row per unit: ``slope_``, ``intercept_``, ``r2_`` and ``p_``
            of each variable, ``modulation`` (span of the raw tuning curve)
            and ``type``.
    """

    types: pd.DataFrame
    tuning: np.ndarray
    fits: pd.DataFrame


def _normalise(means):
    out = np.full_like(means, np.nan)
    for u, y in enumerate(means):
        lo, hi = np.nanmin(y), np.nanmax(y)
        out[u] = 0.5 if np.isclose(lo, hi) else (y - lo) / (hi - lo)
    return out


def _fit(x, y):
    if len(x) < 3 or np.isclose(np.std(x), 0.0):
        return dict(slope=np.nan, intercept=np.nan, r2=np.nan, p=np.nan)
    fit = linregress(x, y)
    return dict(slope=fit.slope, intercept=fit.intercept, r2=fit.rvalue ** 2, p=fit.pvalue)


def classify(fits, p_threshold=0.05, min_modulation=MIN_MODULATION):
    """Type of each unit: the significant variable with the highest R²."""
    types = []
    for _, row in fits.iterrows():
        if row["modulation"] < min_modulation:
            types.append("Inactive")
            continue
        significant = [v for v in VARIABLES
                       if not np.isnan(row[f"p_{v}"]) and row[f"p_{v}"] < p_threshold]
        types.append(max(significant, key=lambda v: row[f"r2_{v}"]) if significant
                     else "Unclassified")
    return pd.Series(types, index=fits.index, name="type")


def analyse_units(activity, trials, p_threshold=0.05, min_modulation=MIN_MODULATION,
                  min_trials=2, rel_value_A=A_TO_B_RATIO):
    """Tuning, regressions and type of every unit of a network.

    Args:
        activity: Array (trials, units) of offer-window activity.
        trials: DataFrame with ``n_B``, ``n_A`` and ``chosen_juice`` per trial.

    Returns:
        :class:`UnitAnalysis`.
    """
    types = trial_types(trials, min_trials, rel_value_A)
    keys = list(zip(trials["n_B"], trials["n_A"], trials["chosen_juice"]))
    raw = np.array([activity[[k == key for k in keys]].mean(axis=0)
                    for key in zip(types["n_B"], types["n_A"], types["chosen_juice"])]).T
    tuning = _normalise(raw)

    rows = []
    for u, y in enumerate(tuning):
        row = {"unit": u, "modulation": np.ptp(raw[u])}
        for var in VARIABLES:
            row.update({f"{k}_{var}": v for k, v in _fit(types[var].to_numpy(), y).items()})
        rows.append(row)
    fits = pd.DataFrame(rows).set_index("unit")
    fits["type"] = classify(fits, p_threshold, min_modulation)
    return UnitAnalysis(types=types, tuning=tuning, fits=fits)


def type_counts(analyses):
    """Table 1: number of units of each type.

    Args:
        analyses: ``{(environment, network): UnitAnalysis}``.

    Returns:
        DataFrame with one row per type and one column per (environment, network).
    """
    counts = {key: a.fits["type"].value_counts().reindex(TYPES, fill_value=0)
              for key, a in analyses.items()}
    counts = pd.DataFrame(counts)
    counts.columns = pd.MultiIndex.from_tuples(counts.columns, names=["environment", "network"])
    counts.loc["Total"] = counts.sum()
    counts.index.name = "type"
    return counts


def exemplar(analysis, variable):
    """Unit classified as ``variable`` with the highest R² for it (Fig. 5), or
    ``None`` if the network has no unit of that type."""
    fits = analysis.fits[analysis.fits["type"] == variable]
    if fits.empty:
        return None
    unit = int(fits[f"r2_{variable}"].idxmax())
    return {"unit": unit, "r2": float(fits.loc[unit, f"r2_{variable}"]),
            "p": float(fits.loc[unit, f"p_{variable}"]),
            "slope": float(fits.loc[unit, f"slope_{variable}"])}


# ----------------------------------------------------------- statistical test
def exact_rank_sum_pvalue(x, y):
    """Exact two-sided permutation p-value of the rank-sum (Mann-Whitney U)
    statistic, valid with ties (enumerates every split of the pooled sample)."""
    pooled = np.concatenate([x, y])
    ranks = pd.Series(pooled).rank().to_numpy()
    n, n_x = pooled.size, len(x)
    if comb(n, n_x) > 2_000_000:
        raise ValueError("Too many permutations for an exact test.")
    expected = n_x * (n + 1) / 2
    observed = abs(ranks[:n_x].sum() - expected)
    sums = np.array([ranks[list(idx)].sum() for idx in combinations(range(n), n_x)])
    return float(np.mean(np.abs(sums - expected) >= observed - 1e-12))


def min_rank_sum_pvalue(n_x, n_y):
    """Smallest two-sided p-value of the exact rank-sum test for samples of
    sizes ``n_x`` and ``n_y`` (without ties), reached when the two samples are
    completely separated: ``2 / C(n_x + n_y, n_x)``."""
    return min(1.0, 2 / comb(n_x + n_y, n_x))


def offer_slopes_rank_sum(fits, group_a="Offer A", group_b="Offer B"):
    """Wilcoxon rank-sum test between the absolute slopes of two unit types.

    Each unit contributes the absolute slope of the variable it was assigned
    to. Returns the normal approximation with continuity correction, the
    exact permutation p-value (when feasible), the minimum p-value attainable
    with these sample sizes and whether the two groups are completely
    separated (every slope of one group larger than every slope of the other).
    """
    x = fits.loc[fits["type"] == group_a, f"slope_{group_a}"].abs().dropna().to_numpy()
    y = fits.loc[fits["type"] == group_b, f"slope_{group_b}"].abs().dropna().to_numpy()
    result = {"group_a": group_a, "n_a": x.size, "median_abs_slope_a": np.nan,
              "group_b": group_b, "n_b": y.size, "median_abs_slope_b": np.nan,
              "separated": np.nan, "U": np.nan, "p_value": np.nan, "p_value_exact": np.nan,
              "p_value_min": np.nan}
    if x.size and y.size:
        test = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        exact = (exact_rank_sum_pvalue(x, y) if comb(x.size + y.size, x.size) <= 2_000_000
                 else np.nan)
        result.update({"median_abs_slope_a": float(np.median(x)),
                       "median_abs_slope_b": float(np.median(y)),
                       "separated": bool(x.min() > y.max() or x.max() < y.min()),
                       "U": float(test.statistic), "p_value": float(test.pvalue),
                       "p_value_exact": exact,
                       "p_value_min": min_rank_sum_pvalue(x.size, y.size)})
    return result
