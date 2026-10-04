"""The figures that carry the paper's three claims.

One figure per claim, and nothing drawn that the text does not argue:

- `displacement`, that swapping the generator moves the decision much further
  than reseeding it;
- `gaps`, that the metaheuristics land close enough to a proven optimum to be
  trusted, and that the discrete constraints are worth solving;
- `transmission`, that one diagnostic out of six survives both tests and the
  loss of any single date.

The older `fig_transmission` in `scripts/make_figures.py` plotted the eleven
generator averages against their realised Sharpe ratios. That is the
cross-section, and Section 7 of the paper says plainly that it could not have
found anything at n = 11. Giving it a figure argued the opposite of the text,
so it is gone and this is what replaced it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scenport.plots import style


def _jitter(n: int, seed: int, spread: float = 0.13) -> np.ndarray:
    """Vertical scatter so overlapping points stay countable."""
    return np.random.default_rng(seed).uniform(-spread, spread, n)


def fig_displacement(artifacts: Path):
    """How far the chosen portfolio moves: reseeding against swapping.

    Both rows are mean L1 distance between pairs of weight vectors, same dates
    and same allocator. The top row holds one point per generator, pairing its
    own seeds against each other. The bottom holds one point per pair of
    generators. The gap between the two is the whole sensitivity argument, so
    the figure is just the two clouds and the distance between their means.
    """
    import matplotlib.pyplot as plt

    within = pd.read_csv(artifacts / "displacement_within.csv")
    between = pd.read_csv(artifacts / "displacement_between.csv")

    floor = float(within["l1_mean"].mean())
    moved = float(between["l1_mean"].mean())

    fig, ax = plt.subplots(figsize=(style.TWO_COL, 2.5))

    rows = [
        (1.0, within["l1_mean"].to_numpy(), style.REFERENCE, 11,
         "Same generator,\ndifferent seed"),
        (0.0, between["l1_mean"].to_numpy(), style.PALETTE[0], 12,
         "Different\ngenerator"),
    ]
    for y, values, colour, seed, _ in rows:
        ax.scatter(values, y + _jitter(len(values), seed), s=26, color=colour,
                   alpha=0.75, edgecolor="white", linewidth=0.5, zorder=3)

    for y, values, colour, _, _ in rows:
        mean = float(values.mean())
        ax.vlines(mean, y - 0.28, y + 0.28, color=colour, linewidth=1.8, zorder=4)

    # The ratio is the headline number, so draw it rather than leave it to the
    # caption. The arrow sits between the two means at mid height.
    ax.annotate(
        "", xy=(moved, 0.5), xytext=(floor, 0.5),
        arrowprops={"arrowstyle": "<->", "color": style.INK, "linewidth": 0.9},
    )
    ax.text(np.sqrt(floor * moved), 0.58, f"{moved / floor:.1f}x", ha="center",
            fontsize=9, color=style.INK)

    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels([r[4] for r in rows], fontsize=8)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(0.0, None)
    ax.set_xlabel("Mean L1 distance between portfolios")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return style.save(fig, artifacts, "displacement")


def fig_gaps(artifacts: Path):
    """Distance from the proven optimum, per allocator.

    The dot is the median over the dates the exact solver closed; the line runs
    out to the worst single date. Log axis because the span is five orders of
    magnitude, which is itself the point: the metaheuristics and the naive
    weightings are not the same kind of answer to the same question.

    MILP is the reference and sits at zero by construction, so it has no row.
    """
    import matplotlib.pyplot as plt

    frame = pd.read_csv(artifacts / "optimality_gaps.csv")
    frame = frame[frame["allocator"] != "milp"].sort_values("median_gap")

    searched = {"ga", "de", "pso", "cmaes"}
    relaxed = {"cvar_lp", "cvar_lp_bound"}

    fig, ax = plt.subplots(figsize=(style.TWO_COL, 3.4))

    for y, (_, row) in enumerate(frame.iterrows()):
        name = row["allocator"]
        if name in searched:
            colour = style.PALETTE[0]
        elif name in relaxed:
            colour = style.PALETTE[1]
        else:
            colour = style.REFERENCE

        ax.hlines(y, row["median_gap"], row["worst_gap"], color=colour,
                  linewidth=1.2, alpha=0.5, zorder=2)
        ax.scatter(row["median_gap"], y, s=38, color=colour, zorder=3,
                   edgecolor="white", linewidth=0.6)
        ax.scatter(row["worst_gap"], y, s=22, facecolor="white", zorder=3,
                   edgecolor=colour, linewidth=1.0)

    ax.set_yticks(range(len(frame)))
    ax.set_yticklabels([style.label(n) for n in frame["allocator"]])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Objective gap above the proven optimum (log scale)")
    ax.grid(axis="y", visible=False)

    # Three families, and the legend says which is which rather than repeating
    # the row labels.
    handles = [
        plt.Line2D([], [], color=style.PALETTE[0], marker="o", linestyle="",
                   markersize=6, label="Searched the constrained problem"),
        plt.Line2D([], [], color=style.PALETTE[1], marker="o", linestyle="",
                   markersize=6, label="Relaxation, trimmed and untrimmed"),
        plt.Line2D([], [], color=style.REFERENCE, marker="o", linestyle="",
                   markersize=6, label="No optimisation"),
        plt.Line2D([], [], color=style.MUTED, marker="o", linestyle="",
                   markersize=5, markerfacecolor="white", label="Worst single date"),
    ]
    # Below the axes. Inside, it covered the three rows at the bottom, which
    # are the ones the figure exists to contrast against.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, fontsize=7.5)
    fig.tight_layout()
    return style.save(fig, artifacts, "gaps")


def fig_transmission(artifacts: Path):
    """Which diagnostic predicts the decision, and which only looks like it does.

    Left: the two tests of the same null, which disagree. Right: how much is
    left of each coefficient once the single worst date is removed. A result
    that needs one month out of eighty-five is not a result, and reading the
    panels together is what separates the one finding from the four that look
    like findings on the left.
    """
    import matplotlib.pyplot as plt

    frame = pd.read_csv(artifacts / "transmission_panel.csv")
    frame = frame.sort_values("t_min_drop_one")

    survivors = frame["survives_dropping_any_date"].to_numpy()
    colours = [style.PALETTE[0] if s else style.REFERENCE for s in survivors]
    y = np.arange(len(frame))

    fig, axes = plt.subplots(1, 2, figsize=(style.TWO_COL, 2.9), sharey=True)

    left, right = axes
    left.scatter(frame["p_clustered"], y, s=40, color=colours, marker="o",
                 zorder=3, edgecolor="white", linewidth=0.6)
    left.scatter(frame["p_permutation"], y, s=40, color=colours, marker="D",
                 zorder=3, facecolor="white", linewidth=1.1,
                 edgecolor=colours)
    for yi, row in zip(y, frame.itertuples(), strict=True):
        left.hlines(yi, min(row.p_clustered, row.p_permutation),
                    max(row.p_clustered, row.p_permutation),
                    color=style.MUTED, linewidth=0.8, alpha=0.5, zorder=2)
    left.axvline(0.05, color=style.INK, linewidth=0.9, linestyle="--", zorder=1)
    left.set_xscale("log")
    left.set_xlabel("$p$ value (log scale)")
    left.text(0.05, len(frame) - 0.35, " 0.05", fontsize=7.5, color=style.INK,
              ha="left", va="top")

    right.barh(y, np.abs(frame["t_min_drop_one"]), height=0.55, color=colours,
               alpha=0.85, zorder=3)
    right.axvline(1.96, color=style.INK, linewidth=0.9, linestyle="--", zorder=4)
    right.set_xlabel("Smallest $|t|$ over all single-date deletions")
    right.text(1.96, len(frame) - 0.35, " 1.96", fontsize=7.5, color=style.INK,
               ha="left", va="top")
    right.grid(axis="y", visible=False)

    left.set_yticks(y)
    left.set_yticklabels([style.label(s) for s in frame["score"]])
    left.grid(axis="y", visible=False)

    handles = [
        plt.Line2D([], [], color=style.MUTED, marker="o", linestyle="",
                   markersize=6, label="Cluster-robust"),
        plt.Line2D([], [], color=style.MUTED, marker="D", linestyle="",
                   markersize=6, markerfacecolor="white", label="Permutation"),
    ]
    # Under both panels. Anywhere inside the left one lands on a row, and the
    # row it landed on was the only one that matters.
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=2, fontsize=7.5)
    fig.tight_layout()
    return style.save(fig, artifacts, "transmission")
