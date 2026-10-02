"""Figures 3, 4 and 5 of the paper."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import linregress  # noqa: E402

from .behaviour import rolling_stats  # noqa: E402
from .neurons import OFFER_LABELS  # noqa: E402

#: Colours of the learning and psychometric curves (Figs. 3 and 4).
NETWORK_COLORS = {"FFNN": "C0", "Modified RNN": "C1", "Standard RNN": "C2"}
NETWORK_MARKERS = {"FFNN": "D", "Modified RNN": "o", "Standard RNN": "s"}
#: Colours of the neural-activity figure (Fig. 5).
NEURON_COLORS = {"Standard RNN": "#CC0000", "Modified RNN": "#0055CC"}


def _save(fig, filename):
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight",
                    metadata={"CreationDate": None} if str(filename).endswith(".pdf") else None)
        plt.close(fig)
    return fig


def learning_curves(rewards, windows, scale=None, filename=None):
    """Fig. 3: total reward per episode with its moving average (solid) and
    moving median (dotted).

    Args:
        rewards: ``{network: rewards per episode}`` in plotting order.
        windows: ``{network: moving-window length}``.
        scale: ``{network: factor}`` to stretch the episode axis of a network
            whose run is shorter; its true episode count is then shown on a
            secondary top axis (Fig. 3b). At most one network may be scaled.
        filename: Output file (``None`` returns the figure).
    """
    scale = scale or {}
    fig, ax = plt.subplots(figsize=(6, 3))
    curves = {}
    for net, r in rewards.items():
        x = np.arange(len(r)) * scale.get(net, 1)
        curves[net] = (x, *rolling_stats(r, windows[net]))
        ax.plot(x, r, color=NETWORK_COLORS[net], alpha=0.2, linewidth=1)
    for net, (x, mean, _) in curves.items():
        ax.plot(x, mean, color=NETWORK_COLORS[net], linewidth=1, label=net)
    for net, (x, _, median) in curves.items():
        ax.plot(x, median, color=NETWORK_COLORS[net], linewidth=1, linestyle="dotted")

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Total reward", fontsize=11)
    if scale:
        (factor,) = set(scale.values())
        top = ax.secondary_xaxis("top", functions=(lambda x: x / factor, lambda x: x * factor))
        top.tick_params(direction="in", pad=-15, labelsize=9)
    ax.grid(linestyle="--", alpha=0.7)
    fig.tight_layout()
    return _save(fig, filename)


def psychometric_curves(tables, filename=None):
    """Fig. 4: percentage of B choices per offer type.

    Args:
        tables: ``{network: behaviour.psychometric(...)}`` in plotting order.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    order = next(iter(tables.values()))["offer"].tolist()
    for net, table in tables.items():
        table = table.set_index("offer").reindex(order)
        ax.scatter(order, table["percent_B"], color=NETWORK_COLORS[net],
                   marker=NETWORK_MARKERS[net], s=80, alpha=0.5, label=net)
    ax.set_xlabel("Offer (#B:#A)", fontsize=11)
    ax.set_ylabel("Percentage of B choice [%]", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()
    return _save(fig, filename)


def representative_neurons(analyses, exemplars, variables=("Offer B", "Chosen value"),
                           environments=("Full", "Partial"), annotate=True, filename=None):
    """Fig. 5: tuning of representative units.

    One row per variable; for each environment, the left panel shows the
    normalised activity of the unit in each trial type against the offer, and
    the right panel the same values against the variable, with the regression
    line. Circles: juice B chosen; diamonds: juice A chosen. In the left
    panel, a solid line joins the circles and a dashed line the diamonds, in
    offer order, of the offers in which both juices are present.

    Args:
        analyses: ``{(environment, network): neurons.UnitAnalysis}``.
        exemplars: ``{(environment, network, variable): unit or None}``.
    """
    ticks = {"Offer A": [0, 1, 2], "Offer B": [0, 2, 4, 6, 8, 10],
             "Chosen value": [0, 2, 4, 6, 8, 10]}
    x_cat = np.arange(len(OFFER_LABELS))

    fig, axes = plt.subplots(len(variables), 2 * len(environments),
                             figsize=(16, 3.4 * len(variables)), squeeze=False)
    for r, var in enumerate(variables):
        for e, env in enumerate(environments):
            ax_offer, ax_value = axes[r, 2 * e], axes[r, 2 * e + 1]
            for k, net in enumerate(NEURON_COLORS):
                unit = exemplars.get((env, net, var))
                if unit is None:
                    continue
                analysis = analyses[(env, net)]
                types, y = analysis.types, analysis.tuning[unit]
                x = types[var].to_numpy()
                color = NEURON_COLORS[net]
                both_juices = ((types["n_B"] > 0) & (types["n_A"] > 0)).to_numpy()
                for juice, marker, line in (("B", "o", "-"), ("A", "D", "--")):
                    sel = (types["chosen_juice"] == juice).to_numpy()
                    # Trial types are ordered by offer; forced choices are not joined
                    joined = sel & both_juices
                    ax_offer.plot(types["offer_index"][joined], y[joined], line, color=color,
                                  alpha=0.8, linewidth=1.2, zorder=2)
                    ax_offer.plot(types["offer_index"][sel], y[sel], marker, color=color,
                                  alpha=0.85, markersize=5, linestyle="none", zorder=3)
                    ax_value.plot(x[sel], y[sel], marker, color=color, alpha=0.85,
                                  markersize=5, linestyle="none", zorder=3)

                fit = linregress(x, y)
                xs = np.array([x.min(), x.max()])
                ax_value.plot(xs, fit.slope * xs + fit.intercept, "-", color=color, alpha=0.8,
                              linewidth=1.2, zorder=2)
                if annotate:
                    ax_value.annotate(f"R²={fit.rvalue ** 2:.2f}", xy=(0.03, 0.97 - 0.09 * k),
                                      xycoords="axes fraction", ha="left", va="top", fontsize=7,
                                      color=color)

            ax_offer.set_xticks(x_cat)
            ax_offer.set_xticklabels(OFFER_LABELS, rotation=45, ha="right", fontsize=6.5)
            ax_offer.grid(True, linestyle="--", alpha=0.6)
            ax_offer.set_xlabel("Offer (#B:#A)", fontsize=9)
            if e == 0:
                ax_offer.set_ylabel("Normalized hidden state value", fontsize=10)

            ax_value.set_xticks(ticks[var])
            ax_value.set_xticklabels([str(t) for t in ticks[var]], fontsize=8)
            ax_value.set_xlim(min(ticks[var]) - 0.5, max(ticks[var]) + 0.5)
            ax_value.grid(True, linestyle="--", alpha=0.6)
            ax_value.set_xlabel(var, fontsize=9)

    fig.tight_layout()
    return _save(fig, filename)
