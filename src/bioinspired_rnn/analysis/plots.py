"""Figures 3, 4 and 5 of the paper.

The figures are drawn at their printed size (one column, 84 mm, or the full
page, 174 mm) with 8-9 pt lettering, and the networks are told apart by their
markers as well as by their colours, so that they remain distinguishable in
greyscale. No transparency is used, so that the EPS versions stay vector.
"""

import functools

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import linregress  # noqa: E402

from .behaviour import rolling_stats  # noqa: E402
from .neurons import OFFER_LABELS  # noqa: E402

#: Colour, marker and fill of each network, in all the figures: FFNN, blue
#: squares; standard RNN, filled green circles; modified RNN, open orange
#: circles. In Fig. 5 the marker shape shows the juice chosen instead.
NETWORK_COLORS = {"FFNN": "C0", "Modified RNN": "C1", "Standard RNN": "C2"}
NETWORK_MARKERS = {"FFNN": "s", "Modified RNN": "o", "Standard RNN": "o"}
NETWORK_FILLED = {"FFNN": True, "Modified RNN": False, "Standard RNN": True}
#: Networks of Fig. 5, in plotting order.
NEURON_NETWORKS = ("Standard RNN", "Modified RNN")

#: Width of a one-column and of a full-width figure of the journal, in inches.
COLUMN_WIDTH = 84 / 25.4
PAGE_WIDTH = 174 / 25.4

#: Computer Modern, the font of the paper (cmr10 ships with Matplotlib), with
#: fonts embedded as TrueType instead of Type 3, and lines of at least 0.5 pt.
PAPER_STYLE = {"font.family": "cmr10", "mathtext.fontset": "cm",
               "axes.formatter.use_mathtext": True, "pdf.fonttype": 42, "ps.fonttype": 42,
               "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
               "legend.fontsize": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
               "ytick.major.width": 0.6, "grid.color": "0.85", "grid.linewidth": 0.5,
               "grid.linestyle": "--"}


def _paper_style(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with plt.rc_context(PAPER_STYLE):
            return func(*args, **kwargs)
    return wrapper


def _save(fig, filename):
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight", pad_inches=0.02,
                    metadata={"CreationDate": None} if str(filename).endswith(".pdf") else None)
        plt.close(fig)
    return fig


def _marker(net, size, marker=None):
    """Line2D keywords of the markers of network ``net`` (filled or open)."""
    color = NETWORK_COLORS[net]
    return dict(marker=marker or NETWORK_MARKERS[net], markersize=size, markeredgewidth=0.9,
                markeredgecolor=color, markerfacecolor=color if NETWORK_FILLED[net] else "none")


def _light(color, amount=0.35):
    """``color`` mixed with white (opaque, unlike a transparent line)."""
    return tuple(amount * c + (1 - amount) for c in to_rgb(color))


@_paper_style
def learning_curves(rewards, windows, scale=None, filename=None):
    """Fig. 3: total reward per episode (light line) with its moving average
    (solid, with markers) and moving median (dotted).

    Args:
        rewards: ``{network: rewards per episode}`` in plotting order.
        windows: ``{network: moving-window length}``.
        scale: ``{network: factor}`` to stretch the episode axis of a network
            whose run is shorter; its true episode count is then shown on a
            secondary top axis (Fig. 3b). At most one network may be scaled.
        filename: Output file (``None`` returns the figure).
    """
    scale = scale or {}
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.3))
    curves = {}
    for net, r in rewards.items():
        x = np.arange(len(r)) * scale.get(net, 1)
        curves[net] = (x, *rolling_stats(r, windows[net]))
        ax.plot(x, r, color=_light(NETWORK_COLORS[net]), linewidth=0.5)
    for k, (net, (x, mean, _)) in enumerate(curves.items()):
        # About eight markers per curve, staggered between the networks
        step = len(x) // 8
        ax.plot(x, mean, color=NETWORK_COLORS[net], linewidth=1, **_marker(net, 3.5),
                markevery=(windows[net] + k * step // 3, step), label=net)
    for net, (x, _, median) in curves.items():
        ax.plot(x, median, color=NETWORK_COLORS[net], linewidth=1, linestyle="dotted")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    # Same limits in both panels, with room above 500 for the top-axis labels of Fig. 3b
    ax.set_ylim(-20, 590)
    ax.set_yticks(range(0, 501, 100))
    if scale:
        (factor,) = set(scale.values())
        top = ax.secondary_xaxis("top", functions=(lambda x: x / factor, lambda x: x * factor))
        top.tick_params(direction="in", pad=-11)   # labels inside the box
    ax.grid(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              handlelength=2.4, columnspacing=1.2, borderaxespad=0.2)
    fig.tight_layout()
    return _save(fig, filename)


@_paper_style
def psychometric_curves(tables, filename=None):
    """Fig. 4: percentage of B choices per offer type, with the networks side
    by side in each offer.

    Args:
        tables: ``{network: behaviour.psychometric(...)}`` in plotting order.
    """
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.3))
    order = next(iter(tables.values()))["offer"].tolist()
    x = np.arange(len(order))
    for (net, table), dx in zip(tables.items(), np.linspace(-0.2, 0.2, len(tables))):
        table = table.set_index("offer").reindex(order)
        ax.plot(x + dx, table["percent_B"], linestyle="none", **_marker(net, 4.5), label=net)
    ax.set_xticks(x, [o.replace("B", "").replace("A", "") for o in order])   # "6B:1A" -> "6:1"
    ax.set_xlabel("Offer (#B:#A)")
    ax.set_ylabel("Juice B chosen [%]")
    ax.set_ylim(-5, 105)
    ax.grid(True, axis="y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              columnspacing=1.2, borderaxespad=0.2)
    fig.tight_layout()
    return _save(fig, filename)


@_paper_style
def representative_neurons(analyses, exemplars, variables=("Offer B", "Chosen value"),
                           environments=("Full", "Partial"), annotate=True, errors=None,
                           filename=None):
    """Fig. 5: tuning of representative units.

    One row per variable; for each environment, the left panel shows the
    normalised activity of the unit in each trial type against the offer, and
    the right panel the same values against the variable, with the regression
    line. Circles: juice B chosen; diamonds: juice A chosen; filled markers:
    standard RNN; open markers: modified RNN. In the left panel, one line
    joins the circles and another the diamonds, in offer order, of the offers
    in which both juices are present. Optional vertical bars show the
    variability over the trials of each trial type.

    Args:
        analyses: ``{(environment, network): neurons.UnitAnalysis}``.
        exemplars: ``{(environment, network, variable): unit or None}``.
        errors: ``None`` (no bars, as in the paper), ``"sem"`` (standard
            error of the mean, as in Padoa-Schioppa 2009) or ``"sd"``
            (standard deviation).
    """
    ticks = {"Offer A": [0, 1, 2], "Offer B": [0, 2, 4, 6, 8, 10],
             "Chosen value": [0, 2, 4, 6, 8, 10]}
    x_cat = np.arange(len(OFFER_LABELS))

    fig, axes = plt.subplots(len(variables), 2 * len(environments), layout="constrained",
                             figsize=(PAGE_WIDTH, 1.75 * len(variables) + 0.35), squeeze=False)
    for r, var in enumerate(variables):
        for e, env in enumerate(environments):
            ax_offer, ax_value = axes[r, 2 * e], axes[r, 2 * e + 1]
            for k, net in enumerate(NEURON_NETWORKS):
                unit = exemplars.get((env, net, var))
                if unit is None:
                    continue
                analysis = analyses[(env, net)]
                types, y = analysis.types, analysis.tuning[unit]
                x = types[var].to_numpy()
                color = NETWORK_COLORS[net]
                spread = None
                if errors is not None and analysis.tuning_sd is not None:
                    spread = analysis.tuning_sd[unit]
                    if errors == "sem":
                        spread = spread / np.sqrt(types["trials"].to_numpy())
                both_juices = ((types["n_B"] > 0) & (types["n_A"] > 0)).to_numpy()
                for juice, marker in (("B", "o"), ("A", "D")):
                    sel = (types["chosen_juice"] == juice).to_numpy()
                    # Trial types are ordered by offer; forced choices are not joined
                    joined = sel & both_juices
                    ax_offer.plot(types["offer_index"][joined], y[joined], "-", color=color,
                                  linewidth=1, zorder=2)
                    if spread is not None:
                        for ax, xs in ((ax_offer, types["offer_index"][sel]), (ax_value, x[sel])):
                            ax.errorbar(xs, y[sel], yerr=spread[sel], fmt="none", ecolor=color,
                                        elinewidth=0.8, zorder=2.5)
                    points = dict(linestyle="none", zorder=3, **_marker(net, 4, marker))
                    ax_offer.plot(types["offer_index"][sel], y[sel], **points)
                    ax_value.plot(x[sel], y[sel], **points)

                fit = linregress(x, y)
                xs = np.array([x.min(), x.max()])
                ax_value.plot(xs, fit.slope * xs + fit.intercept, "-", color=color,
                              linewidth=1, zorder=2)
                if annotate:
                    ax_value.annotate(f"{net.split()[0]}: $R^2$ = {fit.rvalue ** 2:.2f}",
                                      xy=(0.04, 0.96 - 0.13 * k), xycoords="axes fraction",
                                      ha="left", va="top")

            ax_offer.set_xticks(x_cat, OFFER_LABELS, rotation=90)
            ax_offer.grid(True)
            ax_offer.set_xlabel("Offer (#B:#A)")

            # Headroom so that the R² labels do not overlap the points
            lo, hi = ax_value.get_ylim()
            ax_value.set_ylim(lo, hi + 0.38 * (hi - lo))
            ax_value.set_xticks(ticks[var])
            ax_value.set_xlim(min(ticks[var]) - 0.5, max(ticks[var]) + 0.5)
            ax_value.grid(True)
            ax_value.set_xlabel(var)

    fig.supylabel("Normalized hidden state value", fontsize=9)
    handles = [Line2D([], [], color=NETWORK_COLORS[net], **_marker(net, 4, marker),
                      label=f"{net}, juice {juice} chosen")
               for net in NEURON_NETWORKS for juice, marker in (("B", "o"), ("A", "D"))]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False,
               columnspacing=1.5, handlelength=2)
    return _save(fig, filename)
