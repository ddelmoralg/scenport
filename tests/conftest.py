"""Synthetic fixtures. Nothing in the test suite touches the network or the panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scenport.data.panel import Panel
from scenport.data.universe import Universe


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def scenarios(rng):
    """(S, n) horizon returns with a realistic amount of correlation."""
    n_scen, n_assets = 500, 8
    loadings = rng.uniform(0.5, 1.2, size=n_assets)
    common = rng.standard_normal(n_scen) * 0.18
    idio = rng.standard_normal((n_scen, n_assets)) * 0.10
    drift = rng.normal(0.01, 0.03, size=n_assets)
    return drift + common[:, None] * loadings[None, :] + idio


@pytest.fixture
def problem(scenarios):
    from scenport.alloc.objective import Problem

    return Problem(
        scenarios=scenarios,
        assets=[f"a{i}" for i in range(scenarios.shape[1])],
        w_max=0.30,
        w_min=0.05,
        cardinality=5,
        cvar_alpha=0.90,
        gamma=0.5,
        cost_rate=0.0,
    )


@pytest.fixture
def synthetic_panel(rng):
    """Ten correlated geometric random walks over six years of daily data."""
    dates = pd.date_range("2019-01-01", "2024-12-31", freq="D")
    n_assets = 10
    # Loadings and shock sizes chosen so pairwise correlations land in the 0.5-0.8
    # range, which is what the real cryptoasset panel looks like.
    common = rng.standard_normal(len(dates)) * 0.035
    shocks = rng.standard_normal((len(dates), n_assets)) * 0.020
    logret = 0.0004 + common[:, None] * rng.uniform(0.7, 1.2, n_assets)[None, :] + shocks
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(logret, axis=0)),
        index=dates,
        columns=[f"tok{i:02d}" for i in range(n_assets)],
    )
    # A market cap that moves with price but is not proportional to it, so the
    # universe ranking is not a trivial function of the price level.
    supply = pd.Series(rng.uniform(1e6, 5e8, n_assets), index=prices.columns)
    mcap = prices.mul(supply, axis=1)
    return Panel(price=prices, mcap=mcap)


@pytest.fixture
def synthetic_universe(synthetic_panel):
    from scenport.data import universe as uni

    dates = uni.rebalance_dates(synthetic_panel, "2021-01-01", "2024-11-01", "MS")
    return uni.build(synthetic_panel, dates, size=6, min_history_days=540, rank_window=30)


@pytest.fixture
def scenario_universe(synthetic_universe) -> Universe:
    return synthetic_universe
