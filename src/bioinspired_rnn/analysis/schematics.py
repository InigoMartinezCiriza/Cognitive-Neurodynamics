"""Figures 1 and 2 of the paper: the standard and modified GRU cells, and the
observables of an economic choice trial.

Both are vector drawings at their printed size, in the style of the rest of
the figures (:data:`plots.PAPER_STYLE`). The GRU diagrams are laid out on a
grid of points with ``y`` growing downwards, those of the original diagrams.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from .plots import COLUMN_WIDTH, _paper_style, _save  # noqa: E402

#: Inches per unit of the GRU layout (the modified cell, 1324 units wide, is
#: 6.3 in wide).
_UNIT = 6.3 / 1324
_LW = 0.9


class _Diagram:
    """Boxes, labels and lines with arrowheads on a ``width`` x ``height`` grid."""

    def __init__(self, width, height):
        self.fig = plt.figure(figsize=((width + 12) * _UNIT, (height + 14) * _UNIT))
        ax = self.fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(-6, width + 6)
        ax.set_ylim(height + 6, -8)
        ax.axis("off")
        self.ax = ax

    def box(self, x, y, w, h, text, size=11, rounding=14):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
                                         facecolor="white", edgecolor="black", linewidth=_LW,
                                         zorder=3))
        self.ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, zorder=4)

    def small_box(self, cx, cy, text):
        self.box(cx - 25, cy - 20, 50, 40, text, size=10, rounding=8)

    def line(self, *points, arrow=True):
        """Polyline through ``points``, with an arrowhead at the last one."""
        xs, ys = zip(*points)
        end = len(points) - 1 if arrow else len(points)
        if end > 1:
            self.ax.plot(xs[:end], ys[:end], color="black", linewidth=_LW,
                         solid_joinstyle="miter", solid_capstyle="projecting", zorder=2)
        if arrow:
            self.ax.add_patch(FancyArrowPatch(points[-2], points[-1], color="black",
                                              arrowstyle="-|>,head_length=5,head_width=2.2",
                                              linewidth=_LW, shrinkA=0, shrinkB=0, zorder=2))

    def label(self, x, y, text, ha="left", va="center", size=10):
        self.ax.text(x, y, text, ha=ha, va=va, fontsize=size, zorder=4)


@_paper_style
def gru_cell(modified=False, filename=None):
    """Fig. 1: diagram of the standard (``modified=False``) or modified GRU cell.

    Standard: update gate ``z``, reset gate ``r`` and tanh candidate state.
    Modified: leak gate ``lambda`` (scaled by ``alpha``), gate ``gamma``,
    linear candidate state and ReLU output.
    """
    # Columns that differ between the two cells: candidate state, (−) and (⊙)
    # boxes, update (+) box and output
    c = 933 if modified else 939
    minus, odot, plus = c - 30, c + 170, c + 200
    d = _Diagram(1324 if modified else 1255, 443)
    gate1, gate2 = (r"\lambda", r"\gamma") if modified else ("z", "r")

    # Inputs: h_{t-1} bus and x_t
    d.line((0, 291), (149, 291), arrow=False)
    d.line((149, 1), (149, 401), arrow=False)
    d.label(6, 280, r"$h_{t-1}$", va="bottom")
    d.line((10, 421), (c - 25, 421), arrow=False)
    d.label(16, 413, r"$x_t$", va="bottom")
    d.line((149, 401), (719, 401), arrow=False)

    # Gates
    d.box(349, 241, 120, 80, r"$\sigma$", size=13)
    d.box(629, 241, 120, 80, r"$\sigma$", size=13)
    for x_in, x_rec, gate in ((379, 439, gate1), (659, 719, gate2)):
        d.line((x_in, 421), (x_in, 321))
        d.line((x_rec, 401), (x_rec, 321))
        d.label(x_in - 5, 376, rf"$W_{{{gate},\mathrm{{input}}}}$", ha="right", size=9)
        d.label(x_rec + 5, 376, rf"$W_{{{gate},\mathrm{{rec}}}}$", size=9)
    d.label(420, 236, rf"${gate1}_t$", va="bottom")
    d.label(700, 236, rf"${gate2}_t$", va="bottom")

    # Second gate times h_{t-1}, into the candidate state
    d.line((689, 241), (689, 191), (149, 191), arrow=False)
    d.line((240, 191), (240, 276))
    d.small_box(240, 296, r"$\odot$")
    d.line((240, 316), (240, 441), (c + 25, 441), (c + 25, 321))

    # Candidate state
    d.box(c - 60, 241, 120, 80, "$+$" if modified else "tanh", size=13 if modified else 11)
    d.line((c - 25, 421), (c - 25, 321))
    d.label(c - 30, 376, r"$W_{\mathrm{input}}$", ha="right", size=9)
    d.label(c + 30, 376, r"$W_{\mathrm{rec}}$", size=9)
    d.line((c, 241), (c, 131))
    d.label(c + 10, 212, r"$\tilde{h}_t$")

    # Increment: first gate (times alpha) times (candidate - h_{t-1})
    d.box(minus - 60, 51, 120, 80, "$-$", size=13)
    d.line((149, 161), (minus - 30, 161), (minus - 30, 131))
    d.line((minus + 60, 91), (odot - 25, 91))
    d.line((409, 241), (409, 31), (odot, 31), (odot, 71))
    if modified:
        d.small_box(604, 31, r"$\alpha$")
    d.small_box(odot, 91, r"$\odot$")
    d.line((odot, 111), (odot, 241))
    d.label(odot - 8, 150, r"$\Delta h_t$", ha="right")

    # Update: h_{t-1} + increment
    d.box(plus - 60, 241, 120, 80, "$+$", size=13)
    d.line((149, 1), (plus + 30, 1), (plus + 30, 241))
    if modified:
        d.line((plus, 321), (plus, 401), (1262, 401), (1262, 321))
        d.label(plus + 10, 355, r"$h_t^{\prime}$")
        d.box(1202, 241, 120, 80, "ReLU")
        d.line((1262, 241), (1262, 0))
        d.label(1272, 212, r"$h_t$")
    else:
        d.line((plus, 321), (plus, 401), (plus + 75, 401), (plus + 75, 0))
        d.label(plus + 85, 212, r"$h_t$")
    return _save(d.fig, filename)


@_paper_style
def observables(partial=False, filename=None):
    """Fig. 2: observables of an economic choice trial with the offer 6B:1A and
    juice A on the right, and the reward of each action in each epoch.

    With ``partial=True`` the offer is hidden during the decision epoch.
    """
    shown = 0 if partial else 1
    rows = [("Fixation\ncue", [1, 1, 0], (0, 1), [0, 1], "C2"),
            ("Juice\ntype", [0, 1, shown], (-1, 1), [-1, 0, 1], "C1"),
            ("Amount\nof juice A", [0, 1, shown], (0, 2), [0, 2], "C3"),
            ("Amount\nof juice B", [0, 6, 6 * shown], (0, 10), [0, 10], "C0")]
    epochs = [("Fixation", "1500 ms", ["$-1$", "0", "$-1$"], 1),
              ("Offer", "1000-2000 ms", ["$-1$", "0", "$-1$"], 1),
              ("Decision", "2000 ms", ["6", "0", "2.2"], 0)]

    fig = plt.figure(figsize=(COLUMN_WIDTH, 2.9))
    grid = fig.add_gridspec(len(rows) + 1, 1, height_ratios=[1] * len(rows) + [2.3],
                            left=0.26, right=0.97, top=0.98, bottom=0.02, hspace=0.45)
    for i, (name, values, (lo, hi), ticks, color) in enumerate(rows):
        ax = fig.add_subplot(grid[i])
        ax.step(range(4), values + values[-1:], where="post", color=color, linewidth=1.2)
        margin = 0.1 * (hi - lo)
        ax.set_xlim(0, 3)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_yticks(ticks, [f"${t}$" for t in ticks])
        ax.set_xticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8, labelpad=4)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)

    ax = fig.add_subplot(grid[-1])
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for j, (name, duration, rewards, chosen) in enumerate(epochs):
        for i, (side, reward) in enumerate(zip("LFR", rewards)):
            x = j + 0.5 + 0.2 * (i - 1)
            ax.plot(x, 0.72, "o", markersize=8, markeredgewidth=0.8, markeredgecolor="black",
                    markerfacecolor="black" if i == chosen else "white")
            ax.text(x, 0.88, side, ha="center", va="bottom", fontsize=8)
            ax.text(x, 0.58, reward, ha="center", va="top", fontsize=8)
        ax.text(j + 0.5, 0.25, f"{name}\n{duration}", ha="center", va="top", fontsize=8,
                linespacing=1.1)
    ax.plot([0, 3], [0.33, 0.33], color="black", linewidth=0.8)
    for x in range(4):
        ax.plot([x, x], [0.29, 0.37], color="black", linewidth=0.8)
    return _save(fig, filename)
