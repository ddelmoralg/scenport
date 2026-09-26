#!/usr/bin/env python
"""Collect the runs into the tables the paper reports, as CSV and LaTeX.

    python scripts/aggregate.py --only main transmission tests

Nothing here re-runs an experiment. A table with missing rows means a job has not
finished.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenport import config as cfgmod  # noqa: E402
from scenport import stats as tests
from scenport.backtest import metrics  # noqa: E402

log = logging.getLogger("aggregate")

TABLES = ("main", "forecasts", "transmission", "displacement", "tests", "regimes", "exact")

QUALITY_COLUMNS = {
    "mase": "point accuracy",
    "crps": "distributional accuracy",
    "pinball": "quantile loss",
    "coverage_error": "interval calibration",
    "pit_ks": "PIT uniformity",
    # The energy score is known to discriminate weakly against a misspecified
    # dependence structure (Scheuerer and Hamill, 2015), so anything said about
    # the coupling stage leans on the variogram and the energy score is reported
    # beside it rather than instead of it.
    "energy_score": "joint accuracy (weak against misspecified dependence)",
    "variogram_score": "dependence accuracy (primary for the coupling stage)",
}


# ---------------------------------------------------------------------------
# loading


def load_summaries(runs: Path) -> pd.DataFrame:
    files = sorted((runs / "summary").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no backtest summaries under {runs / 'summary'}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_scores(runs: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted((runs / "scores").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no forecast scores under {runs / 'scores'}")
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    joint = frame[frame["asset"] == "__joint__"].drop(columns=["asset"])
    per_asset = frame[frame["asset"] != "__joint__"].copy()
    per_asset["mase"] = per_asset["abs_error"] / per_asset["mase_scale"].replace(0, np.nan)
    per_asset["coverage_error"] = (per_asset["cover_80"] - 0.80).abs()
    return per_asset, joint


def load_daily(runs: Path) -> pd.DataFrame:
    """Daily net returns of every run, one column per label."""
    series = {}
    for meta in sorted((runs / "backtests").rglob("daily.parquet")):
        parts = meta.parent.name.rsplit("_seed", 1)
        forecaster = meta.parent.parent.name
        label = f"{forecaster}|{parts[0]}|{parts[1] if len(parts) > 1 else '00'}"
        series[label] = pd.read_parquet(meta).iloc[:, 0]
    if not series:
        raise FileNotFoundError(f"no daily series under {runs / 'backtests'}")
    return pd.DataFrame(series).sort_index()


# ---------------------------------------------------------------------------
# tables


def table_main(summaries: pd.DataFrame) -> pd.DataFrame:
    """Forecaster x allocator, averaged over seeds, with dispersion."""
    grouped = summaries.groupby(["forecaster", "allocator"])
    out = grouped.agg(
        n_seeds=("seed", "nunique"),
        sharpe=("sharpe", "mean"),
        sharpe_sd=("sharpe", "std"),
        ann_return=("ann_return", "mean"),
        ann_vol=("ann_vol", "mean"),
        sortino=("sortino", "mean"),
        max_drawdown=("max_drawdown", "mean"),
        calmar=("calmar", "mean"),
        cvar_95=("cvar_95", "mean"),
        turnover=("mean_turnover", "mean"),
        cost=("total_cost", "mean"),
        seconds=("mean_seconds", "mean"),
    ).reset_index()
    return out.sort_values(["allocator", "sharpe"], ascending=[True, False], ignore_index=True)


def table_forecasts(per_asset: pd.DataFrame, joint: pd.DataFrame) -> pd.DataFrame:
    """One row per forecaster: how good the scenario sets themselves were."""
    rows = []
    for name, group in per_asset.groupby("forecaster"):
        pit_values = group["pit"].dropna()
        ks = sps.kstest(pit_values, "uniform").statistic if len(pit_values) > 20 else np.nan
        joint_group = joint[joint["forecaster"] == name]
        rows.append(
            {
                "forecaster": name,
                "n_observations": len(group),
                "mase": group["mase"].mean(),
                "crps": group["crps"].mean(),
                "pinball": group["pinball"].mean(),
                "cover_80": group["cover_80"].mean(),
                "coverage_error": group["coverage_error"].mean(),
                "pit_ks": ks,
                "energy_score": joint_group["energy_score"].mean(),
                "variogram_score": joint_group["variogram_score"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("crps", ignore_index=True)


def table_transmission(
    summaries: pd.DataFrame,
    per_asset: pd.DataFrame,
    forecast_quality: pd.DataFrame,
    joint: pd.DataFrame,
    runs: Path,
    reference_allocator: str = "ga",
    panel_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Does forecast quality carry through to portfolio performance?

    Two views. Across forecasters, a rank correlation between each quality score and
    the Sharpe of the portfolio built on it. Within the panel, a regression of the
    realised period return on the same scores with a fixed effect per rebalancing
    date, so the coefficient is identified off the *cross-section* of forecasters in
    the same month rather than off the market itself.
    """
    reference = summaries[summaries["allocator"] == reference_allocator]
    by_forecaster = reference.groupby("forecaster")["sharpe"].mean().rename("sharpe")
    merged = forecast_quality.set_index("forecaster").join(by_forecaster, how="inner")

    cross_rows = []
    for column, description in QUALITY_COLUMNS.items():
        if column not in merged or merged[column].notna().sum() < 4:
            continue
        rho, p = sps.spearmanr(merged[column], merged["sharpe"], nan_policy="omit")
        n = int(merged[column].notna().sum())
        cross_rows.append(
            {
                "score": column,
                "measures": description,
                "spearman_vs_sharpe": rho,
                "p_value": p,
                "n_forecasters": n,
                # Declared in advance, and printed next to the correlation so no
                # reader has to work out whether this column could have found
                # anything. With eleven forecasters it is around 0.76.
                #
                # This is the power of the *between-generator ranking*, and the
                # panel does not rescue it: however many forecaster-date rows the
                # panel has, a pure ordering of the generators still rests on
                # eleven units. What the panel buys power for is the part of the
                # relationship that varies by date, which is a different question
                # and is reported in the panel table.
                "min_detectable_rho": tests.minimum_detectable_correlation(n),
                "n_for_rho_0.3": tests.required_sample_for_correlation(0.3),
            }
        )
    cross = pd.DataFrame(cross_rows)

    panel = _transmission_panel(per_asset, joint, runs, reference_allocator, panel_dir)
    return cross, panel


def _market_betas(panel_dir: Path, dates, window_days: int = 365) -> pd.DataFrame:
    """Trailing beta of every asset against the cap-weighted index, one row per date.

    Estimated strictly on the window ending the day before the rebalancing date, so
    nothing here can see the period it is used to control.
    """
    from scenport.data.panel import Panel

    try:
        panel = Panel.read(panel_dir)
    except FileNotFoundError:
        return pd.DataFrame()

    returns = panel.returns()
    caps = panel.mcap.reindex(returns.index).shift(1)
    market_weights = caps.div(caps.sum(axis=1), axis=0)
    market = (returns * market_weights).sum(axis=1)

    rows = {}
    for date in pd.DatetimeIndex(dates):
        start = date - pd.Timedelta(days=window_days)
        window = returns.loc[start : date - pd.Timedelta(days=1)]
        bench = market.loc[start : date - pd.Timedelta(days=1)]
        if len(bench) < 60:
            continue
        variance = float(np.nanvar(bench.to_numpy()))
        if variance <= 0:
            continue
        aligned = window.sub(window.mean())
        centred_bench = bench - bench.mean()
        cov = aligned.mul(centred_bench, axis=0).mean()
        rows[date] = cov / variance
    return pd.DataFrame(rows).T if rows else pd.DataFrame()


def _portfolio_exposure(runs: Path, panel_dir: Path, allocator: str) -> pd.DataFrame:
    """Ex-ante market exposure of each forecaster's portfolio at each rebalancing.

    A forecaster that happened to hold less of the market in 2022 looks good on
    Sharpe for a reason that has nothing to do with how well it forecast. This is
    the control for that, and it is the weights times the trailing asset betas, not
    anything measured over the period being explained.
    """
    frames = []
    for path in sorted((runs / "backtests").rglob("weights.parquet")):
        if not path.parent.name.startswith(f"{allocator}_seed"):
            continue
        weights = pd.read_parquet(path)
        weights.index = pd.DatetimeIndex(weights.index)
        frames.append((path.parent.parent.name, weights))
    if not frames:
        return pd.DataFrame()

    all_dates = sorted({d for _, w in frames for d in w.index})
    betas = _market_betas(panel_dir, all_dates)
    if betas.empty:
        return pd.DataFrame()

    rows = []
    for forecaster, weights in frames:
        for date, w in weights.iterrows():
            if date not in betas.index:
                continue
            b = betas.loc[date].reindex(w.index)
            usable = w.notna() & b.notna()
            if not usable.any():
                continue
            rows.append(
                {
                    "rebalance": date,
                    "forecaster": forecaster,
                    "exposure": float((w[usable] * b[usable]).sum()),
                }
            )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .groupby(["rebalance", "forecaster"], as_index=False)["exposure"]
        .mean()
    )


def _transmission_panel(
    per_asset: pd.DataFrame,
    joint: pd.DataFrame,
    runs: Path,
    allocator: str,
    panel_dir: Path | None = None,
) -> pd.DataFrame:
    """Within-date regression of period return on forecast quality.

    The unit is the forecaster-date pair, not the forecaster. Eleven forecasters
    give eleven points and a correlation on eleven points cannot resolve anything
    below about 0.76, which is reported in the cross-sectional table. Here the same
    eleven give several hundred observations across the rebalancing calendar.

    Three things are done to keep that from being a cheat. The date carries a fixed
    effect, so the coefficient is identified off the cross-section of forecasters
    inside one month and not off the market. Standard errors are clustered on the
    date and the interval comes from resampling whole dates, because the months
    overlap and the forecasters inside a month share a market. And the p value is a
    permutation over which forecaster got which score inside each date, which
    assumes nothing about the errors.

    Market exposure enters as a control where the weights are available: a
    forecaster that held less of the market through 2022 scores well for a reason
    that is not forecast quality.
    """
    periods = []
    for path in sorted((runs / "backtests").rglob("periods.parquet")):
        name = path.parent.name
        if not name.startswith(f"{allocator}_seed"):
            continue
        frame = pd.read_parquet(path)
        frame["forecaster"] = path.parent.parent.name
        periods.append(frame.reset_index()[["rebalance", "forecaster", "net_return"]])
    if not periods:
        return pd.DataFrame()

    performance = (
        pd.concat(periods).groupby(["rebalance", "forecaster"], as_index=False)["net_return"].mean()
    )
    quality = (
        per_asset.groupby(["rebalance", "forecaster"], as_index=False)[
            ["mase", "crps", "pinball", "coverage_error"]
        ].mean()
    )
    panel = performance.merge(quality, on=["rebalance", "forecaster"], how="inner")

    # The joint scores live on their own rows because they are not per asset. They
    # are also the only two that say anything about the coupling stage, so leaving
    # them out of the panel would mean the dependence half of the scenario
    # generator never gets tested at all.
    joint_columns = [c for c in ("energy_score", "variogram_score") if c in joint]
    if joint_columns:
        joint_quality = (
            joint.groupby(["rebalance", "forecaster"], as_index=False)[joint_columns].mean()
        )
        panel = panel.merge(joint_quality, on=["rebalance", "forecaster"], how="left")

    exposure = (
        _portfolio_exposure(runs, panel_dir, allocator)
        if panel_dir is not None
        else pd.DataFrame()
    )
    controlled = not exposure.empty
    if controlled:
        panel = panel.merge(exposure, on=["rebalance", "forecaster"], how="left")
        if panel["exposure"].isna().mean() > 0.2:
            log.warning("market exposure missing for most rows; reporting without the control")
            panel = panel.drop(columns="exposure")
            controlled = False

    required = ["net_return", "mase", "crps", "pinball", "coverage_error"]
    if controlled:
        required.append("exposure")
    panel = panel.dropna(subset=[c for c in required if c in panel])
    if panel.empty:
        return panel

    scores = [
        c
        for c in ("mase", "crps", "pinball", "coverage_error", "energy_score", "variogram_score")
        if c in panel
    ]

    rows = []
    for column in scores:
        regressors = [column] + (["exposure"] if controlled else [])
        demeaned = tests.within_transform(panel, ["net_return", *regressors], "rebalance")
        x = demeaned[regressors].to_numpy(float)
        y = demeaned["net_return"].to_numpy(float)
        if np.allclose(x[:, 0], 0.0):
            continue

        clusters = panel["rebalance"].to_numpy()
        fit = tests.cluster_robust_ols(y, x, clusters)
        ci = tests.cluster_bootstrap_ci(y, x, clusters, column=0, reps=2000, seed=0)
        perm = tests.permutation_within_clusters(y, x, clusters, column=0, reps=5000, seed=0)

        rows.append(
            {
                "score": column,
                "beta": float(fit["beta"][0]),
                "std_error": float(fit["std_error"][0]),
                "t_stat": float(fit["t_stat"][0]),
                "p_clustered": float(fit["p_value"][0]),
                "p_permutation": float(perm.p_value),
                "ci_lower": ci["lower"],
                "ci_upper": ci["upper"],
                "exposure_controlled": controlled,
                "beta_exposure": float(fit["beta"][1]) if controlled else np.nan,
                "n_obs": int(fit["n_obs"]),
                "n_dates": int(fit["n_clusters"]),
            }
        )
    return pd.DataFrame(rows)


def _load_weight_runs(runs: Path, allocator: str) -> dict:
    """Every stored weight path for one allocator, keyed by (forecaster, seed)."""
    found = {}
    for path in sorted((runs / "backtests").rglob("weights.parquet")):
        name = path.parent.name
        if not name.startswith(f"{allocator}_seed"):
            continue
        seed = name.rsplit("_seed", 1)[-1]
        frame = pd.read_parquet(path)
        frame.index = pd.DatetimeIndex(frame.index)
        found[(path.parent.parent.name, seed)] = frame.sort_index()
    return found


def _pair_distance(a: pd.DataFrame, b: pd.DataFrame, tol: float) -> pd.DataFrame:
    """L1 distance and support overlap between two weight histories, per date."""
    dates = a.index.intersection(b.index)
    if len(dates) == 0:
        return pd.DataFrame()
    assets = a.columns.union(b.columns)
    wa = a.reindex(index=dates, columns=assets).fillna(0.0)
    wb = b.reindex(index=dates, columns=assets).fillna(0.0)

    l1 = (wa - wb).abs().sum(axis=1)

    held_a, held_b = wa > tol, wb > tol
    intersection = (held_a & held_b).sum(axis=1)
    union = (held_a | held_b).sum(axis=1)
    jaccard = intersection.divide(union.replace(0, np.nan))

    return pd.DataFrame({"rebalance": dates, "l1": l1.to_numpy(), "jaccard": jaccard.to_numpy()})


def table_displacement(
    runs: Path, allocator: str = "ga", tol: float = 1e-6
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How far the chosen portfolio moves when the scenario generator is swapped.

    This is the dependent quantity the sensitivity question actually asks about, and
    it is not the same thing as realised performance. It needs no outcome data at
    all: it is measured entirely from the decisions, which is what keeps it from
    collapsing into "forecast quality in disguise".

    Two frames come back. The first is every pair of generators. The second is the
    same estimator applied to one generator rerun under a different seed, which is
    the noise floor: before claiming the decision moves because the generator
    changed, we have to know how far it moves when nothing changes but the random
    draw. Both use cross-seed pairs, so they are measured the same way and the ratio
    between them means something.

    ``l1`` runs from 0 to 2 for two portfolios on the simplex. ``jaccard`` is the
    overlap of the chosen assets, 1 when both hold exactly the same names.
    """
    stored = _load_weight_runs(runs, allocator)
    if not stored:
        return pd.DataFrame(), pd.DataFrame()

    by_forecaster: dict[str, list] = {}
    for (forecaster, seed), frame in stored.items():
        by_forecaster.setdefault(forecaster, []).append((seed, frame))

    within_rows = []
    for forecaster, runs_for_one in sorted(by_forecaster.items()):
        pieces = [
            _pair_distance(a, b, tol)
            for (_, a), (_, b) in combinations(sorted(runs_for_one, key=lambda x: x[0]), 2)
        ]
        pieces = [p for p in pieces if not p.empty]
        if not pieces:
            continue
        joined = pd.concat(pieces)
        within_rows.append(
            {
                "forecaster": forecaster,
                "l1_mean": float(joined["l1"].mean()),
                "l1_median": float(joined["l1"].median()),
                "jaccard_mean": float(joined["jaccard"].mean()),
                "n_pairs": len(pieces),
                "n_comparisons": int(len(joined)),
            }
        )
    within = pd.DataFrame(within_rows)

    noise_floor = float(within["l1_mean"].mean()) if not within.empty else np.nan

    between_rows = []
    for left, right in combinations(sorted(by_forecaster), 2):
        pieces = [
            _pair_distance(a, b, tol)
            for _, a in by_forecaster[left]
            for _, b in by_forecaster[right]
        ]
        pieces = [p for p in pieces if not p.empty]
        if not pieces:
            continue
        joined = pd.concat(pieces)
        l1_mean = float(joined["l1"].mean())
        between_rows.append(
            {
                "forecaster_a": left,
                "forecaster_b": right,
                "l1_mean": l1_mean,
                "l1_median": float(joined["l1"].median()),
                "jaccard_mean": float(joined["jaccard"].mean()),
                # Everything the paper says about between-generator displacement is
                # said relative to this. Below 1 means swapping the generator moves
                # the portfolio less than rerunning the same one with a new seed.
                "over_noise_floor": l1_mean / noise_floor if noise_floor else np.nan,
                "n_comparisons": int(len(joined)),
            }
        )

    between = pd.DataFrame(between_rows)
    if not between.empty:
        between = between.sort_values("l1_mean", ascending=False, ignore_index=True)
        between.attrs["noise_floor_l1"] = noise_floor
    return between, within


def displacement_by_date(runs: Path, allocator: str = "ga", tol: float = 1e-6) -> pd.DataFrame:
    """Mean pairwise displacement per rebalancing date, for the panel.

    One row per date per generator: how far that generator's portfolio sits from the
    portfolios every other generator chose on the same date. This is the left-hand
    side of the clean half of the study, the half that needs no outcome data.
    """
    stored = _load_weight_runs(runs, allocator)
    if not stored:
        return pd.DataFrame()

    seed_mean: dict[str, pd.DataFrame] = {}
    grouped: dict[str, list] = {}
    for (forecaster, _), frame in stored.items():
        grouped.setdefault(forecaster, []).append(frame)
    for forecaster, frames in grouped.items():
        assets = frames[0].columns
        for f in frames[1:]:
            assets = assets.union(f.columns)
        aligned = [f.reindex(columns=assets).fillna(0.0) for f in frames]
        seed_mean[forecaster] = sum(aligned) / len(aligned)

    rows = []
    names = sorted(seed_mean)
    for left, right in combinations(names, 2):
        pair = _pair_distance(seed_mean[left], seed_mean[right], tol)
        if pair.empty:
            continue
        for name in (left, right):
            block = pair.copy()
            block["forecaster"] = name
            rows.append(block)
    if not rows:
        return pd.DataFrame()

    return (
        pd.concat(rows)
        .groupby(["rebalance", "forecaster"], as_index=False)[["l1", "jaccard"]]
        .mean()
        .rename(columns={"l1": "displacement_l1", "jaccard": "support_overlap"})
    )


def _seed_mean_weights(runs: Path, allocator: str) -> dict:
    """One weight history per generator, averaged over its seeds."""
    stored = _load_weight_runs(runs, allocator)
    grouped: dict[str, list] = {}
    for (forecaster, _), frame in stored.items():
        grouped.setdefault(forecaster, []).append(frame)

    averaged = {}
    for forecaster, frames in grouped.items():
        assets = frames[0].columns
        for f in frames[1:]:
            assets = assets.union(f.columns)
        aligned = [f.reindex(columns=assets).fillna(0.0) for f in frames]
        averaged[forecaster] = sum(aligned) / len(aligned)
    return averaged


def table_objective_difference(
    runs: Path, panel_dir: Path, cfg, allocator: str = "ga", every: int = 1
) -> pd.DataFrame:
    """What each generator's portfolio is worth under a common reference set.

    The third displacement quantity. L1 and support overlap say how far apart two
    decisions are; this says how much that distance is worth, in the units of the
    objective the allocator was optimising.

    The reference set is a stationary block bootstrap of realised returns over the
    window ending the day before the date, so it uses nothing the decision maker did
    not have. That is what keeps this separate from realised performance: no outcome
    data enters at any point. See ``scenport.forecast.reference`` for why it is built
    the way it is.
    """
    from scenport.alloc.objective import Problem, score
    from scenport.data.panel import Panel
    from scenport.forecast import reference as ref

    weights = _seed_mean_weights(runs, allocator)
    if len(weights) < 2:
        return pd.DataFrame()

    try:
        panel = Panel.read(panel_dir)
    except FileNotFoundError:
        log.warning("no price panel beside the runs; skipping the objective difference")
        return pd.DataFrame()

    returns = panel.returns()
    horizon = int(cfg.get("backtest.horizon_days", 30))
    common = sorted(set.intersection(*(set(w.index) for w in weights.values())))[::every]

    rows = []
    for date in common:
        assets = sorted({a for w in weights.values() for a in w.columns})
        scenarios = ref.reference_scenarios(
            returns,
            date,
            assets,
            horizon,
            window_days=int(cfg.get("backtest.train_days", 730)),
            seed=ref.reference_seed(date),
        )
        if scenarios.shape[0] == 0:
            continue

        problem = Problem(
            scenarios=scenarios,
            assets=assets,
            w_max=float(cfg.get("alloc.w_max", 0.20)),
            w_min=float(cfg.get("alloc.w_min", 0.02)),
            cardinality=cfg.get("alloc.cardinality", 8),
            cvar_alpha=float(cfg.get("alloc.cvar_alpha", 0.95)),
            gamma=float(cfg.get("alloc.gamma", 0.5)),
            # Turnover is a property of the path the portfolio took, not of the
            # reference distribution, so it is left out here. Including it would
            # make the yardstick depend on which generator ran last.
            cost_rate=0.0,
            objective=cfg.get("alloc.objective", "mean_cvar"),
        )

        for forecaster, history in weights.items():
            if date not in history.index:
                continue
            w = history.loc[date].reindex(assets).fillna(0.0).to_numpy(float)
            total = w.sum()
            if total <= 0:
                continue
            rows.append(
                {
                    "rebalance": date,
                    "forecaster": forecaster,
                    "reference_objective": float(score(w / total, problem)[0]),
                }
            )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    # Centred inside the date, because the level of the objective on any given date
    # is a property of that month and not of the generator.
    frame["objective_gap"] = frame["reference_objective"] - frame.groupby("rebalance")[
        "reference_objective"
    ].transform("mean")
    return frame


def table_sensitivity(
    runs: Path,
    panel_dir: Path,
    cfg,
    summaries: pd.DataFrame,
    allocator: str = "ga",
) -> tuple[pd.DataFrame, dict]:
    """Decision movement and realised performance side by side, per generator.

    The two can come apart, and every paper in this area measures only the second
    one, so the case where they do would go unnoticed. Three things can happen and
    the table has to distinguish them:

    - the decision moves and performance follows: the generator matters, and the
      question becomes which diagnostic predicts it;
    - neither moves: the program is insensitive to its generator inside this class,
      which is a statement about the formulation;
    - the decision sits at the seed noise floor and performance still varies: the
      variation came from the realised path, not from the generator. That is a
      decomposition this literature leaves mixed, and it is the most interesting of
      the three.

    ``displacement_over_noise`` below one means swapping the generator moves the
    portfolio less than rerunning one generator with a different seed.
    """
    between, within = table_displacement(runs, allocator)
    by_date = displacement_by_date(runs, allocator)
    if between.empty or by_date.empty:
        return pd.DataFrame(), {}

    noise_floor = float(between.attrs.get("noise_floor_l1", np.nan))

    # Both sides of this ratio have to be measured the same way. `between` and
    # `within` are both averages over raw seed pairs, so they are comparable.
    # `displacement_by_date` averages the weights over seeds before comparing, which
    # cancels most of the Monte Carlo noise, and dividing that by a raw-pair floor
    # would compare a denoised numerator against a noisy denominator and make
    # everything look insensitive. It is kept for the panel, where it is the right
    # quantity, and reported here as its own column.
    stacked = pd.concat(
        [
            between.rename(columns={"forecaster_a": "forecaster"})[
                ["forecaster", "l1_mean", "jaccard_mean"]
            ],
            between.rename(columns={"forecaster_b": "forecaster"})[
                ["forecaster", "l1_mean", "jaccard_mean"]
            ],
        ]
    )
    per_generator = (
        stacked.groupby("forecaster", as_index=False)[["l1_mean", "jaccard_mean"]]
        .mean()
        .rename(columns={"l1_mean": "displacement", "jaccard_mean": "overlap"})
    )
    per_generator["displacement_over_noise"] = per_generator["displacement"] / noise_floor

    seed_averaged = (
        by_date.groupby("forecaster", as_index=False)["displacement_l1"]
        .mean()
        .rename(columns={"displacement_l1": "displacement_seed_averaged"})
    )
    per_generator = per_generator.merge(seed_averaged, on="forecaster", how="left")

    reference = summaries[summaries["allocator"] == allocator]
    performance = (
        reference.groupby("forecaster", as_index=False)[["sharpe", "ann_return"]].mean()
        if not reference.empty
        else pd.DataFrame(columns=["forecaster", "sharpe", "ann_return"])
    )
    table = per_generator.merge(performance, on="forecaster", how="left")

    gaps = table_objective_difference(runs, panel_dir, cfg, allocator)
    if not gaps.empty:
        table = table.merge(
            gaps.groupby("forecaster", as_index=False)["objective_gap"].mean(),
            on="forecaster",
            how="left",
        )

    headline = {
        "noise_floor_l1": noise_floor,
        "mean_displacement": float(per_generator["displacement"].mean()),
        "mean_displacement_over_noise": float(per_generator["displacement_over_noise"].mean()),
        "displacement_spread": float(per_generator["displacement"].std()),
        "sharpe_spread": float(table["sharpe"].std()) if "sharpe" in table else np.nan,
        "n_generators": int(len(table)),
    }
    # The reading of the table in one field, so a null does not get written up as a
    # failure by whoever opens the csv first.
    moves = headline["mean_displacement_over_noise"] > 1.5
    performs = bool(np.isfinite(headline["sharpe_spread"]) and headline["sharpe_spread"] > 0.05)
    headline["reading"] = {
        (True, True): "decision moves and performance follows",
        (True, False): "decision moves, performance does not separate",
        (False, True): "decision at the noise floor, performance still varies: "
                       "the variation is in the realised path, not the generator",
        (False, False): "insensitive to the generator within this class",
    }[(moves, performs)]

    table = table.sort_values("displacement_over_noise", ascending=False, ignore_index=True)
    table.attrs.update(headline)
    return table, headline


def table_tests(daily: pd.DataFrame, summaries: pd.DataFrame, cfg) -> pd.DataFrame:
    """Sharpe comparisons against the two benchmarks, plus deflation and PBO."""
    benchmarks = [c for c in daily.columns if c.split("|")[1] in ("equal", "mcap")]
    if not benchmarks:
        log.warning("no equal-weight or market-cap column found; skipping the comparisons")
        return pd.DataFrame()

    reference = sorted(benchmarks, key=lambda c: c.split("|")[1])[0]
    block = int(cfg["statistics.block_length"])
    reps = int(cfg["statistics.bootstrap_reps"])

    rows = []
    for column in daily.columns:
        if column == reference:
            continue
        result = tests.sharpe_difference(
            daily[column], daily[reference], bootstrap=reps // 5, block=block
        )
        rows.append(
            {
                "strategy": column,
                "benchmark": reference,
                **result.detail,
                "z": result.statistic,
                "p_value": result.p_value,
            }
        )
    frame = pd.DataFrame(rows)

    # Deflate the best Sharpe by the number of configurations that were compared.
    n_trials = summaries.groupby(["forecaster", "allocator"]).ngroups
    best_label = daily.apply(lambda s: metrics.sharpe(s)).idxmax()
    best = daily[best_label].dropna()
    daily_sharpe = metrics.sharpe(best) / np.sqrt(cfg["backtest.annualisation"])
    deflated = tests.deflated_sharpe(
        daily_sharpe,
        n_trials=n_trials,
        n_observations=len(best),
        skew=float(best.skew()),
        excess_kurtosis=float(best.kurtosis()),
    )
    frame.attrs["deflated_sharpe"] = {
        "best": best_label, "n_trials": n_trials, **deflated.detail,
        "z": deflated.statistic, "p_value": deflated.p_value,
    }
    return frame


def table_regimes(daily: pd.DataFrame, runs: Path, cfg) -> pd.DataFrame:
    """Performance split into bull, bear and sideways stretches."""
    market = [c for c in daily.columns if c.split("|")[1] == "mcap"]
    if not market:
        log.warning("no market-cap benchmark; regimes cannot be labelled")
        return pd.DataFrame()
    nav = (1.0 + daily[market[0]].fillna(0.0)).cumprod()
    labels = metrics.regime_labels(nav)

    rows = []
    for column in daily.columns:
        series = daily[column].dropna()
        for regime, chunk in series.groupby(labels.reindex(series.index)):
            if regime == "warmup" or len(chunk) < 30:
                continue
            rows.append(
                {
                    "strategy": column,
                    "regime": regime,
                    "n_days": len(chunk),
                    **{k: v for k, v in metrics.summary(chunk).items()
                       if k in ("ann_return", "ann_vol", "sharpe", "max_drawdown")},
                }
            )
    return pd.DataFrame(rows)


def table_exact(runs: Path) -> pd.DataFrame:
    path = runs / "exact" / "gaps.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    return (
        frame.groupby("allocator")
        .agg(
            mean_gap=("relative_gap", "mean"),
            median_gap=("relative_gap", "median"),
            worst_gap=("relative_gap", "max"),
            mean_seconds=("seconds", "mean"),
            n=("relative_gap", "size"),
        )
        .reset_index()
        .sort_values("mean_gap", ignore_index=True)
    )


# ---------------------------------------------------------------------------


def write(frame: pd.DataFrame, directory: Path, name: str, caption: str = "") -> None:
    if frame is None or frame.empty:
        log.warning("table %s is empty, not written", name)
        return
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / f"{name}.csv", index=False)
    numeric = frame.select_dtypes(include="number").columns
    frame.to_latex(
        directory / f"{name}.tex",
        index=False,
        escape=True,
        float_format="%.3f",
        caption=caption or name.replace("_", " "),
        label=f"tab:{name}",
        column_format="l" * (len(frame.columns) - len(numeric)) + "r" * len(numeric),
    )
    log.info("wrote %s (%d rows)", name, len(frame))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--only", nargs="+", default=list(TABLES), choices=TABLES)
    parser.add_argument("--reference-allocator", default="ga")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = cfgmod.repo_root()
    cfg = cfgmod.load(args.config, args.overrides)
    runs = cfg.resolve("paths.runs", root)
    artifacts = cfg.resolve("paths.artifacts", root)

    summaries = load_summaries(runs)
    per_asset, joint = load_scores(runs)

    # Consolidated copies of the two inputs, so re-analysis elsewhere does not need
    # the run tree. These are what travel back over git; the tree itself does not.
    artifacts.mkdir(parents=True, exist_ok=True)
    pd.concat([per_asset, joint]).to_parquet(artifacts / "scores.parquet", index=False)
    try:
        load_daily(runs).to_parquet(artifacts / "daily_returns.parquet")
    except FileNotFoundError:
        log.warning("no daily series to bundle")

    if "main" in args.only:
        write(table_main(summaries), artifacts, "main",
              "Walk-forward performance by forecaster and allocator.")

    quality = table_forecasts(per_asset, joint)
    if "forecasts" in args.only:
        write(quality, artifacts, "forecast_quality",
              "Accuracy and calibration of the scenario sets.")

    if "transmission" in args.only:
        cross, panel = table_transmission(
            summaries, per_asset, quality, joint, runs, args.reference_allocator,
            cfg.resolve("paths.panel", root),
        )
        write(cross, artifacts, "transmission_cross",
              "Rank correlation between forecast quality and portfolio Sharpe, across "
              "generators. min_detectable_rho is the smallest correlation this "
              "many generators could resolve at 80% power; the panel below does "
              "not improve it, because ranking generators rests on the number of "
              "generators however many rows are stacked underneath.")
        write(panel, artifacts, "transmission_panel",
              "Period return on forecast quality, with a fixed effect per rebalancing "
              "date, errors clustered on the date, interval from resampling whole "
              "dates and p value from permuting scores within a date. This is powered "
              "for the date-varying part of the relationship, not for a ranking of "
              "the generators.")

    if "displacement" in args.only:
        between, within = table_displacement(runs, args.reference_allocator)
        if between.empty:
            log.warning("no stored weights; skipping the displacement tables")
        else:
            floor = between.attrs.get("noise_floor_l1", float("nan"))
            write(between, artifacts, "displacement_between",
                  "Pairwise decision displacement between scenario generators. "
                  f"Mean L1 within one generator across seeds: {floor:.4f}.")
            write(within, artifacts, "displacement_within",
                  "Decision displacement from rerunning one generator with a new seed. "
                  "This is the floor every between-generator figure is quoted against.")
            by_date = displacement_by_date(runs, args.reference_allocator)
            if not by_date.empty:
                by_date.to_parquet(artifacts / "displacement_panel.parquet", index=False)

            sensitivity, headline = table_sensitivity(
                runs, cfg.resolve("paths.panel", root), cfg, summaries,
                args.reference_allocator,
            )
            if not sensitivity.empty:
                write(sensitivity, artifacts, "sensitivity",
                      "Decision displacement and realised performance per generator. "
                      f"Reading: {headline['reading']}.")
                (artifacts / "sensitivity_headline.json").write_text(
                    json.dumps(headline, indent=2), encoding="utf-8"
                )
                log.info("sensitivity: %s", headline["reading"])

            gaps = table_objective_difference(
                runs, cfg.resolve("paths.panel", root), cfg, args.reference_allocator
            )
            if not gaps.empty:
                gaps.to_parquet(artifacts / "objective_gap_panel.parquet", index=False)
                summary = (
                    gaps.groupby("forecaster", as_index=False)["objective_gap"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )
                write(summary, artifacts, "objective_gap",
                      "Objective value of each generator's portfolio under a common "
                      "reference set built only from data available at the date, "
                      "centred within each date.")

    if {"tests", "regimes"} & set(args.only):
        daily = load_daily(runs)
        if "tests" in args.only:
            frame = table_tests(daily, summaries, cfg)
            write(frame, artifacts, "sharpe_tests",
                  "Sharpe differences against the equal-weight benchmark.")
            if frame is not None and frame.attrs.get("deflated_sharpe"):
                pd.Series(frame.attrs["deflated_sharpe"]).to_csv(
                    artifacts / "deflated_sharpe.csv"
                )
            pbo = tests.probability_of_backtest_overfitting(
                daily.dropna(axis=1, how="any").resample("ME").sum()
            )
            pd.Series({"pbo": pbo.statistic, **pbo.detail}).to_csv(artifacts / "pbo.csv")
            log.info("probability of backtest overfitting: %.3f", pbo.statistic)
        if "regimes" in args.only:
            write(table_regimes(daily, runs, cfg), artifacts, "regimes",
                  "Performance by market regime.")

    if "exact" in args.only:
        write(table_exact(runs), artifacts, "optimality_gaps",
              "Distance to the mixed-integer optimum on identical instances.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
