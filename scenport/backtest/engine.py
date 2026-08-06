"""Walk-forward backtest.

One pass over the rebalancing calendar. The forecaster only sees prices strictly
before the decision date, the universe is the one investable on that date, and the
portfolio is held unchanged until the next rebalancing so weights drift with prices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..alloc.base import Allocator
from ..alloc.objective import Problem
from ..data.panel import Panel
from ..data.universe import Universe
from ..forecast.base import ScenarioForecaster
from ..forecast.coupling import apply as couple

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    daily: pd.Series                      # net daily returns of the strategy
    periods: pd.DataFrame                 # one row per rebalancing date
    weights: pd.DataFrame                 # target weights, dates x assets
    label: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def nav(self) -> pd.Series:
        return (1.0 + self.daily.fillna(0.0)).cumprod()

    def write(self, directory) -> None:
        from pathlib import Path

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.daily.to_frame("ret").to_parquet(directory / "daily.parquet")
        self.periods.to_parquet(directory / "periods.parquet")
        self.weights.to_parquet(directory / "weights.parquet")


def _hold(prices: pd.DataFrame, weights: pd.Series, start, end) -> pd.Series:
    """Daily returns of a buy-and-hold portfolio between two dates.

    The value index is built from price *relatives*, so the weights drift with the
    market inside the period instead of being silently rebalanced every day. Gaps
    are carried forward, which is the same thing as assuming a position that stops
    printing a price was liquidated at its last one and held as cash.
    """
    window = prices.loc[start:end, list(weights.index)].ffill()
    if len(window) < 2:
        return pd.Series(dtype=float)
    relative = window.divide(window.iloc[0], axis=1)
    value = relative.mul(weights.reindex(relative.columns).to_numpy(), axis=1).sum(axis=1)
    return value.pct_change().iloc[1:]


def _forced_turnover(w_prev: pd.Series, universe: list[str]) -> float:
    """Weight held in assets that are no longer investable and must be sold."""
    leaving = w_prev.drop(labels=[a for a in universe if a in w_prev.index], errors="ignore")
    return float(leaving.abs().sum())


def run(
    panel: Panel,
    universe: Universe,
    forecaster: ScenarioForecaster,
    allocator: Allocator,
    *,
    horizon: int,
    train_days: int,
    n_paths: int,
    coupling: str = "gaussian_copula",
    coupling_window: int | None = 365,
    cost_bps: float = 20.0,
    rf_annual: pd.Series | None = None,
    seed: int = 0,
    problem_kwargs: dict | None = None,
    scenario_cache: dict | None = None,
    label: str = "",
) -> BacktestResult:
    """Run one (forecaster, allocator) combination over the whole calendar.

    ``scenario_cache`` lets several allocators reuse the scenario sets produced for
    the same forecaster and seed, which is where almost all the compute goes.
    """
    rng = np.random.default_rng(seed)
    cost_rate = cost_bps / 10_000.0
    problem_kwargs = dict(problem_kwargs or {})

    dates = universe.dates
    prices = panel.price
    w_prev = pd.Series(dtype=float)              # drifted weights carried in
    daily_chunks: list[pd.Series] = []
    rows: list[dict] = []
    weight_rows: dict[pd.Timestamp, pd.Series] = {}

    for i, date in enumerate(dates):
        assets = universe.at(date)
        history = prices.loc[:date, assets].iloc[-(train_days + 1) :]
        if history.shape[0] < 60:
            log.warning("%s: only %d days of history, skipping", date.date(), history.shape[0])
            continue
        # Strictly before the decision date: the price at `date` is the one we
        # trade at, not one the forecaster is allowed to have seen.
        history = history.iloc[:-1]

        key = (forecaster.name, date, seed)
        if scenario_cache is not None and key in scenario_cache:
            scenarios = scenario_cache[key]
        else:
            marginals = forecaster.simulate(history, horizon, n_paths, rng, origin=date)
            scenarios = couple(
                marginals, history, method=coupling, rng=rng, window=coupling_window
            )
            if scenario_cache is not None:
                scenario_cache[key] = scenarios

        rf_h = 0.0
        if rf_annual is not None:
            from ..data.benchmarks import horizon_rate

            rf_h = horizon_rate(rf_annual, date, horizon)

        forced = _forced_turnover(w_prev, assets)
        problem = Problem(
            scenarios=scenarios.reorder(assets).terminal(),
            assets=list(assets),
            w_prev=w_prev.reindex(assets).fillna(0.0).to_numpy(),
            cost_rate=cost_rate,
            rf=rf_h,
            meta={"mcap": panel.mcap.loc[date, assets].to_numpy(dtype=float)},
            **problem_kwargs,
        )

        allocation = allocator(problem, np.random.default_rng(rng.integers(0, 2**63 - 1)))
        w = pd.Series(allocation.weights, index=assets)
        weight_rows[date] = w

        turnover = float(np.abs(w - w_prev.reindex(assets).fillna(0.0)).sum()) + forced
        cost = cost_rate * turnover

        end = dates[i + 1] if i + 1 < len(dates) else prices.index.max()
        period_returns = _hold(prices, w, date, end)
        if period_returns.empty:
            log.warning("%s: no price history to hold through, stopping", date.date())
            break

        # The rebalancing cost lands on the first day of the new period.
        net = period_returns.copy()
        net.iloc[0] = (1.0 + net.iloc[0]) * (1.0 - cost) - 1.0
        daily_chunks.append(net)

        drifted = _drift(prices, w, date, end)
        w_prev = drifted

        rows.append(
            {
                "rebalance": date,
                "period_end": end,
                "n_assets": len(assets),
                "turnover": turnover,
                "forced_turnover": forced,
                "cost": cost,
                "gross_return": float(np.prod(1.0 + period_returns) - 1.0),
                "net_return": float(np.prod(1.0 + net) - 1.0),
                "rf_horizon": rf_h,
                "seconds": allocation.seconds,
                **{f"exante_{k}": v for k, v in allocation.diagnostics.items()
                   if isinstance(v, (int, float, bool))},
            }
        )

    daily = pd.concat(daily_chunks).sort_index() if daily_chunks else pd.Series(dtype=float)
    daily = daily[~daily.index.duplicated(keep="first")]
    weights = pd.DataFrame(weight_rows).T.reindex(columns=sorted({a for w in weight_rows.values()
                                                                  for a in w.index})).fillna(0.0)

    return BacktestResult(
        daily=daily.rename(label or f"{forecaster.name}+{allocator.name}"),
        periods=pd.DataFrame(rows).set_index("rebalance") if rows else pd.DataFrame(),
        weights=weights,
        label=label or f"{forecaster.name}+{allocator.name}",
        meta={
            "forecaster": forecaster.name,
            "allocator": allocator.name,
            "seed": seed,
            "horizon": horizon,
            "train_days": train_days,
            "n_paths": n_paths,
            "coupling": coupling,
            "cost_bps": cost_bps,
            **problem_kwargs,
        },
    )


def _drift(prices: pd.DataFrame, weights: pd.Series, start, end) -> pd.Series:
    """Weights at the end of a holding period after price drift."""
    window = prices.loc[start:end, list(weights.index)].ffill()
    if len(window) < 2:
        return weights.copy()
    growth = window.iloc[-1] / window.iloc[0]
    value = weights * growth
    total = value.sum()
    return value / total if total > 0 else weights.copy()


def run_benchmark(
    panel: Panel,
    universe: Universe,
    weight_fn,
    *,
    cost_bps: float = 20.0,
    label: str = "benchmark",
) -> BacktestResult:
    """Same loop for a rule that needs no forecast, so the calendars line up exactly."""
    cost_rate = cost_bps / 10_000.0
    dates = universe.dates
    prices = panel.price
    w_prev = pd.Series(dtype=float)
    chunks, rows, weight_rows = [], [], {}

    for i, date in enumerate(dates):
        assets = universe.at(date)
        w = weight_fn(date, assets)
        weight_rows[date] = w

        turnover = float(np.abs(w - w_prev.reindex(assets).fillna(0.0)).sum())
        turnover += _forced_turnover(w_prev, assets)
        cost = cost_rate * turnover

        end = dates[i + 1] if i + 1 < len(dates) else prices.index.max()
        period = _hold(prices, w, date, end)
        if period.empty:
            break
        net = period.copy()
        net.iloc[0] = (1.0 + net.iloc[0]) * (1.0 - cost) - 1.0
        chunks.append(net)
        w_prev = _drift(prices, w, date, end)
        rows.append(
            {
                "rebalance": date,
                "period_end": end,
                "turnover": turnover,
                "cost": cost,
                "net_return": float(np.prod(1.0 + net) - 1.0),
            }
        )

    daily = pd.concat(chunks).sort_index() if chunks else pd.Series(dtype=float)
    daily = daily[~daily.index.duplicated(keep="first")]
    return BacktestResult(
        daily=daily.rename(label),
        periods=pd.DataFrame(rows).set_index("rebalance") if rows else pd.DataFrame(),
        weights=pd.DataFrame(weight_rows).T.fillna(0.0),
        label=label,
    )
