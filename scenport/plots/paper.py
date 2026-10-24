"""The figures that carry the paper's three claims, one figure per claim.

- `displacement`: swapping the generator moves the decision much further than
  reseeding it.
- `gaps`: the metaheuristics land close to a proven optimum, and the discrete
  constraints are worth solving.
- `transmission`: which diagnostics are associated with same-month return, under
  which test, and whether the association survives losing a date.

The cross-section of generator averages against realised Sharpe ratios has no
figure. With eleven points it has almost no power, and the paper gives it as a
table for completeness.
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
    out to the worst single date. Distances are in objective units, not relative
    to the optimum, because one date has an optimum close to zero and would
    decide every relative worst case. Log axis because the span is several orders
    of magnitude, which is itself the point: the metaheuristics and the naive
    weightings are not the same kind of answer to the same question.

    MILP is the reference and sits at zero by construction, so it has no row.
    """
    import matplotlib.pyplot as plt

    frame = pd.read_csv(artifacts / "optimality_gaps.csv")
    frame = frame[frame["allocator"] != "milp"].sort_values("median_absolute_gap")

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

        ax.hlines(y, row["median_absolute_gap"], row["worst_absolute_gap"], color=colour,
                  linewidth=1.2, alpha=0.5, zorder=2)
        ax.scatter(row["median_absolute_gap"], y, s=38, color=colour, zorder=3,
                   edgecolor="white", linewidth=0.6)
        ax.scatter(row["worst_absolute_gap"], y, s=22, facecolor="white", zorder=3,
                   edgecolor=colour, linewidth=1.0)

    ax.set_yticks(range(len(frame)))
    ax.set_yticklabels([style.label(n) for n in frame["allocator"]])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Distance from the proven optimum, objective units (log scale)")
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
    # Below the axes, so it cannot cover the bottom rows, which are the ones the
    # figure contrasts against.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, fontsize=7.5)
    fig.tight_layout()
    return style.save(fig, artifacts, "gaps")


def fig_transmission(artifacts: Path):
    """How robust each association between diagnostic and return is.

    Left: three tests of the same null on the full panel, each Holm-adjusted over
    the six diagnostics: cluster-robust errors, the within-date permutation and
    the permutation that relabels whole generator series. Right: the largest
    within-date permutation p value over the 85 refits that each drop one date,
    and the p value with the whole February to April 2021 episode removed.
    """
    import matplotlib.pyplot as plt

    frame = pd.read_csv(artifacts / "inference_family.csv")
    drop = pd.read_csv(artifacts / "inference_drop_permutation.csv").set_index("score")
    frame = frame.join(drop, on="score").sort_values("max_p_drop_one", ascending=False)

    robust = (frame["max_p_drop_one"] < 0.05).to_numpy()
    colours = [style.PALETTE[0] if r else style.REFERENCE for r in robust]
    y = np.arange(len(frame))

    fig, axes = plt.subplots(1, 2, figsize=(style.TWO_COL, 3.0), sharey=True)
    left, right = axes

    tests = [("p_clustered_holm", "o", True, "Cluster-robust"),
             ("p_permutation_holm", "D", False, "Within-date permutation"),
             ("p_block_holm", "s", False, "Generator-series permutation")]
    for column, marker, filled, _ in tests:
        if filled:
            left.scatter(frame[column], y, s=36, color=colours, marker=marker, zorder=3,
                         edgecolor="white", linewidth=0.6)
        else:
            left.scatter(frame[column], y, s=36, marker=marker, zorder=3,
                         facecolor="white", linewidth=1.1, edgecolor=colours)
    values = frame[[t[0] for t in tests]].to_numpy()
    left.hlines(y, values.min(axis=1), values.max(axis=1), color=style.MUTED,
                linewidth=0.8, alpha=0.5, zorder=2)
    left.set_xlabel("Holm-adjusted $p$ value, full panel")

    right.scatter(frame["max_p_drop_one"], y, s=36, color=colours, marker="D", zorder=3,
                  facecolor="white", linewidth=1.1, edgecolor=colours)
    right.scatter(frame["p_without_feb_to_apr_2021"], y, s=30, marker="x", zorder=3,
                  color=colours, linewidth=1.1)
    right.hlines(y, frame[["max_p_drop_one", "p_without_feb_to_apr_2021"]].min(axis=1),
                 frame[["max_p_drop_one", "p_without_feb_to_apr_2021"]].max(axis=1),
                 color=style.MUTED, linewidth=0.8, alpha=0.5, zorder=2)
    right.set_xlabel("Within-date permutation $p$ after deleting dates")

    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(0.05, color=style.INK, linewidth=0.9, linestyle="--", zorder=1)
        ax.text(0.05, len(frame) - 0.35, " 0.05", fontsize=7.5, color=style.INK,
                ha="left", va="top")
        ax.grid(axis="y", visible=False)

    left.set_yticks(y)
    left.set_yticklabels([style.label(s) for s in frame["score"]])

    handles = [
        plt.Line2D([], [], color=style.MUTED, marker=marker, linestyle="",
                   markersize=6, markerfacecolor=style.MUTED if filled else "white",
                   label=name)
        for _, marker, filled, name in tests
    ]
    handles.append(plt.Line2D([], [], color=style.MUTED, marker="x", linestyle="",
                              markersize=6, label="Feb to Apr 2021 removed"))
    # Under both panels. Inside the left one it would sit on a row.
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06),
               ncol=4, fontsize=7.2)
    fig.tight_layout()
    return style.save(fig, artifacts, "transmission")
