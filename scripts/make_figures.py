#!/usr/bin/env python
"""Build the figures from the aggregated output. Vector PDF plus a PNG to eyeball.

    python scripts/make_figures.py --only nav transmission
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenport import config as cfgmod  # noqa: E402
from scenport.backtest import metrics  # noqa: E402
from scenport.plots import style  # noqa: E402

log = logging.getLogger("figures")

FIGURES = ("nav", "transmission", "calibration", "scaling", "convergence", "regimes")


def _daily(runs: Path) -> pd.DataFrame:
    series = {}
    for path in sorted((runs / "backtests").rglob("daily.parquet")):
        allocator, _, seed = path.parent.name.rpartition("_seed")
        forecaster = path.parent.parent.name
        series[f"{forecaster}|{allocator}|{seed}"] = pd.read_parquet(path).iloc[:, 0]
    return pd.DataFrame(series).sort_index()


def _mean_over_seeds(daily: pd.DataFrame, forecaster: str, allocator: str) -> pd.Series:
    """Average the daily returns across seeds -- the portfolio a user would get
    if they had no way to pick the lucky seed in advance."""
    cols = [c for c in daily.columns if c.startswith(f"{forecaster}|{allocator}|")]
    return daily[cols].mean(axis=1) if cols else pd.Series(dtype=float)


# ---------------------------------------------------------------------------


def fig_nav(runs: Path, artifacts: Path, cfg, headline: str, allocator: str):
    import matplotlib.pyplot as plt

    daily = _daily(runs)
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.5, 5.0), sharex=True, height_ratios=[2.2, 1.0]
    )

    tracks = [
        (headline, allocator, f"{style.label(headline)} + {style.label(allocator)}"),
        ("rw", allocator, f"{style.label('rw')} + {style.label(allocator)}"),
        ("rw", "equal", style.label("equal")),
        ("rw", "mcap", style.label("mcap")),
    ]

    for forecaster, alloc_name, text in tracks:
        series = _mean_over_seeds(daily, forecaster, alloc_name)
        if series.empty:
            continue
        nav = (1.0 + series.fillna(0.0)).cumprod()
        spec = style.style_for(
            alloc_name if alloc_name in ("equal", "mcap") else forecaster,
            style.FORECASTER_ORDER,
        )
        top.plot(nav.index, nav.to_numpy(), label=text, **spec)
        style.annotate_last(top, nav.index[-1], nav.iloc[-1], text, spec["color"])

        drop = metrics.drawdown_curve(series)
        bottom.plot(drop.index, 100 * drop.to_numpy(), **{**spec, "marker": ""})

    top.set_yscale("log")
    top.set_ylabel("Growth of 1 unit (log)")
    top.legend(loc="upper left", ncol=1)
    bottom.set_ylabel("Drawdown (%)")
    bottom.set_xlabel("")
    fig.align_ylabels()
    return style.save(fig, artifacts, "nav")


def fig_transmission(artifacts: Path):
    """The headline: forecast quality on the x axis, portfolio Sharpe on the y.

    Two panels, one for point accuracy and one for distributional accuracy, on
    identical y scales so the two slopes can be compared by eye. Every point is
    labelled -- with eleven forecasters a legend would be unreadable and the
    identity of the outliers is the interesting part.
    """
    import matplotlib.pyplot as plt

    quality = pd.read_csv(artifacts / "forecast_quality.csv")
    cross = pd.read_csv(artifacts / "transmission_cross.csv")
    main = pd.read_csv(artifacts / "main.csv")

    reference = main[main["allocator"] == "ga"].set_index("forecaster")["sharpe"]
    merged = quality.set_index("forecaster").join(reference.rename("sharpe")).dropna(
        subset=["sharpe"]
    )

    panels = [("mase", "Point accuracy (MASE, lower is better)"),
              ("crps", "Distributional accuracy (CRPS, lower is better)")]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2), sharey=True)

    for ax, (column, xlabel) in zip(axes, panels, strict=True):
        if column not in merged:
            continue
        for name, row in merged.iterrows():
            spec = style.style_for(name, style.FORECASTER_ORDER)
            ax.scatter(row[column], row["sharpe"], s=34, color=spec["color"],
                       marker=spec["marker"], zorder=3, edgecolor="white", linewidth=0.6)
            ax.annotate(style.label(name), (row[column], row["sharpe"]),
                        xytext=(4, 3), textcoords="offset points", fontsize=6.8,
                        color=style.MUTED)

        good = merged[[column, "sharpe"]].dropna()
        if len(good) > 2:
            slope, intercept = np.polyfit(good[column], good["sharpe"], 1)
            xs = np.linspace(good[column].min(), good[column].max(), 20)
            ax.plot(xs, slope * xs + intercept, color=style.REFERENCE, linewidth=1.0,
                    linestyle="--", zorder=1)
            row = cross[cross["score"] == column]
            if len(row):
                ax.set_title(
                    rf"Spearman $\rho$ = {row['spearman_vs_sharpe'].iloc[0]:.2f}"
                    rf"  ($p$ = {row['p_value'].iloc[0]:.2f})",
                    color=style.MUTED,
                )
        ax.set_xlabel(xlabel)

    axes[0].set_ylabel("Out-of-sample Sharpe ratio")
    fig.tight_layout()
    return style.save(fig, artifacts, "transmission")


def fig_calibration(runs: Path, artifacts: Path):
    """PIT histograms. Flat means honest; a U shape means the intervals are too
    narrow, a hump in the middle means they are too wide."""
    import matplotlib.pyplot as plt

    frames = [pd.read_parquet(p) for p in sorted((runs / "scores").glob("*.parquet"))]
    scores = pd.concat(frames, ignore_index=True)
    scores = scores[(scores["asset"] != "__joint__") & scores["pit"].notna()]

    names = [f for f in style.FORECASTER_ORDER if f in set(scores["forecaster"])]
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.5, 1.5 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    bins = np.linspace(0, 1, 11)
    for ax, name in zip(axes, names, strict=False):
        values = scores.loc[scores["forecaster"] == name, "pit"]
        spec = style.style_for(name, style.FORECASTER_ORDER)
        ax.hist(values, bins=bins, density=True, color=spec["color"], alpha=0.85,
                edgecolor="white", linewidth=0.5)
        ax.axhline(1.0, color=style.REFERENCE, linewidth=1.0, linestyle="--")
        ax.set_title(style.label(name), fontsize=8)
        ax.set_ylim(0, 3.0)
        ax.grid(axis="x", visible=False)
    for ax in axes[len(names):]:
        ax.set_visible(False)

    fig.supxlabel("Probability integral transform", fontsize=9)
    fig.supylabel("Density", fontsize=9)
    fig.tight_layout()
    return style.save(fig, artifacts, "calibration")


def fig_scaling(runs: Path, artifacts: Path):
    """Solve time against universe size: where branch and bound stops finishing."""
    import matplotlib.pyplot as plt

    path = runs / "exact" / "scaling.parquet"
    if not path.exists():
        log.warning("no scaling results; skipping")
        return None
    frame = pd.read_parquet(path)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for name, group in frame.groupby("allocator"):
        group = group.sort_values("n_assets")
        spec = style.style_for(name, style.ALLOCATOR_ORDER)
        if name == "milp":
            spec = {"color": style.INK, "linestyle": "-", "marker": "o",
                    "linewidth": 2.0, "markersize": 5}
        ax.plot(group["n_assets"], group["seconds"], label=style.label(name), **spec)
        style.annotate_last(ax, group["n_assets"].iloc[-1], group["seconds"].iloc[-1],
                            style.label(name), spec["color"])

    ax.set_yscale("log")
    ax.set_xlabel("Universe size (number of assets)")
    ax.set_ylabel("Wall-clock time per instance (s)")
    ax.legend(loc="upper left", ncol=2)
    return style.save(fig, artifacts, "scaling")


def fig_convergence(runs: Path, artifacts: Path):
    """Incumbent objective against generation, averaged over seeds and dates."""
    import matplotlib.pyplot as plt

    curves: dict[str, list[np.ndarray]] = {}
    for path in sorted((runs / "backtests").rglob("periods.parquet")):
        allocator = path.parent.name.rpartition("_seed")[0]
        if allocator not in ("ga", "de", "pso", "cmaes"):
            continue
        frame = pd.read_parquet(path)
        if "exante_generations" in frame:
            curves.setdefault(allocator, []).append(frame["exante_generations"].to_numpy())

    if not curves:
        log.warning("no convergence traces recorded; skipping")
        return None

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for name, runs_list in curves.items():
        values = np.concatenate(runs_list)
        spec = style.style_for(name, style.ALLOCATOR_ORDER)
        ax.hist(values, bins=20, histtype="step", label=style.label(name),
                color=spec["color"], linestyle=spec["linestyle"], linewidth=1.6, density=True)
    ax.set_xlabel("Generations used before the incumbent stopped improving")
    ax.set_ylabel("Density")
    ax.legend()
    return style.save(fig, artifacts, "convergence")


def fig_regimes(artifacts: Path):
    """Sharpe by market regime, grouped bars, one group per regime."""
    import matplotlib.pyplot as plt

    path = artifacts / "regimes.csv"
    if not path.exists():
        log.warning("no regime table; skipping")
        return None
    frame = pd.read_csv(path)
    frame[["forecaster", "allocator", "seed"]] = frame["strategy"].str.split("|", expand=True)
    keep = frame[frame["allocator"].isin(["ga", "equal", "mcap", "cvar_lp"])]
    pivot = keep.pivot_table(index="regime", columns="allocator", values="sharpe", aggfunc="mean")
    pivot = pivot.reindex(["bull", "sideways", "bear"]).dropna(how="all")

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    width = 0.8 / max(len(pivot.columns), 1)
    positions = np.arange(len(pivot))
    for i, column in enumerate(pivot.columns):
        spec = style.style_for(column, style.ALLOCATOR_ORDER)
        ax.bar(positions + i * width, pivot[column].to_numpy(), width * 0.92,
               label=style.label(column), color=spec["color"], edgecolor="white", linewidth=0.8)
    ax.axhline(0.0, color=style.MUTED, linewidth=0.8)
    ax.set_xticks(positions + width * (len(pivot.columns) - 1) / 2)
    ax.set_xticklabels([r.capitalize() for r in pivot.index])
    ax.set_ylabel("Sharpe ratio")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(pivot.columns))
    return style.save(fig, artifacts, "regimes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--only", nargs="+", default=list(FIGURES), choices=FIGURES)
    parser.add_argument("--headline-forecaster", default="garch")
    parser.add_argument("--headline-allocator", default="ga")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = cfgmod.repo_root()
    cfg = cfgmod.load(args.config, args.overrides)
    runs = cfg.resolve("paths.runs", root)
    artifacts = cfg.resolve("paths.artifacts", root)

    style.use_paper_style()

    builders = {
        "nav": lambda: fig_nav(runs, artifacts, cfg, args.headline_forecaster,
                               args.headline_allocator),
        "transmission": lambda: fig_transmission(artifacts),
        "calibration": lambda: fig_calibration(runs, artifacts),
        "scaling": lambda: fig_scaling(runs, artifacts),
        "convergence": lambda: fig_convergence(runs, artifacts),
        "regimes": lambda: fig_regimes(artifacts),
    }

    for name in args.only:
        try:
            written = builders[name]()
        except FileNotFoundError as exc:
            log.warning("skipping %s: %s", name, exc)
            continue
        if written:
            log.info("wrote %s", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
