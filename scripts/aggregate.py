#!/usr/bin/env python
"""Collect the runs into the tables the paper reports, as CSV and LaTeX.

    python scripts/aggregate.py --only main transmission tests

Nothing here re-runs an experiment. A table with missing rows means a job has not
finished.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenport import config as cfgmod  # noqa: E402
from scenport import stats as tests
from scenport.backtest import metrics  # noqa: E402

log = logging.getLogger("aggregate")

TABLES = ("main", "forecasts", "transmission", "tests", "regimes", "exact")

QUALITY_COLUMNS = {
    "mase": "point accuracy",
    "crps": "distributional accuracy",
    "pinball": "quantile loss",
    "coverage_error": "interval calibration",
    "pit_ks": "PIT uniformity",
    "energy_score": "joint accuracy",
    "variogram_score": "dependence accuracy",
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
    runs: Path,
    reference_allocator: str = "ga",
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
        cross_rows.append(
            {
                "score": column,
                "measures": description,
                "spearman_vs_sharpe": rho,
                "p_value": p,
                "n_forecasters": int(merged[column].notna().sum()),
            }
        )
    cross = pd.DataFrame(cross_rows)

    panel = _transmission_panel(per_asset, runs, reference_allocator)
    return cross, panel


def _transmission_panel(per_asset: pd.DataFrame, runs: Path, allocator: str) -> pd.DataFrame:
    """Within-date regression of period return on forecast quality."""
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
    panel = performance.merge(quality, on=["rebalance", "forecaster"], how="inner").dropna()
    if panel.empty:
        return panel

    rows = []
    for column in ("mase", "crps", "pinball", "coverage_error"):
        # Within transform: subtract the date mean from both sides. Algebraically
        # identical to a full set of date dummies and far cheaper.
        demeaned = panel.copy()
        for col in ("net_return", column):
            demeaned[col] = demeaned[col] - demeaned.groupby("rebalance")[col].transform("mean")
        x, y = demeaned[column].to_numpy(), demeaned["net_return"].to_numpy()
        if np.std(x) == 0:
            continue
        beta = float(np.dot(x, y) / np.dot(x, x))
        resid = y - beta * x

        # Cluster-robust by date: the same market shock hits every forecaster in a
        # given month, so independent errors would understate the standard error.
        bread = 1.0 / np.dot(x, x)
        meat = 0.0
        for _, idx in demeaned.groupby("rebalance").indices.items():
            score_c = float(np.dot(x[idx], resid[idx]))
            meat += score_c**2
        se = float(np.sqrt(bread * meat * bread))
        t = beta / se if se > 0 else np.nan
        rows.append(
            {
                "score": column,
                "beta": beta,
                "std_error": se,
                "t_stat": t,
                "p_value": float(2 * (1 - sps.norm.cdf(abs(t)))) if np.isfinite(t) else np.nan,
                "n_obs": len(panel),
                "n_dates": panel["rebalance"].nunique(),
            }
        )
    return pd.DataFrame(rows)


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
            summaries, per_asset, quality, runs, args.reference_allocator
        )
        write(cross, artifacts, "transmission_cross",
              "Rank correlation between forecast quality and portfolio Sharpe.")
        write(panel, artifacts, "transmission_panel",
              "Period return on forecast quality, with a fixed effect per rebalancing date.")

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
