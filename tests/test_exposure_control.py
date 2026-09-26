"""The market exposure control, exercised on synthetic data.

This code only runs when there is a full run tree to aggregate, which on the cluster
means it would go first through its paces at the end of a multi-day pipeline. These
tests give it a dry run instead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scenport.data.panel import Panel

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def aggregate():
    spec = importlib.util.spec_from_file_location("aggregate_under_test", SCRIPTS / "aggregate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def market(tmp_path):
    """Three assets. One tracks the market, one is twice as jumpy, one is flat.

    Built from a common factor so the true betas are known: roughly 1, 2 and 0.
    """
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2021-01-01", periods=600)
    factor = rng.normal(0.0, 0.02, len(dates))

    returns = pd.DataFrame(
        {
            "AAA": factor + rng.normal(0, 0.002, len(dates)),
            "BBB": 2.0 * factor + rng.normal(0, 0.002, len(dates)),
            "CCC": rng.normal(0, 0.002, len(dates)),
        },
        index=dates,
    )
    price = (1.0 + returns).cumprod() * 100.0
    # Equal caps, so the cap-weighted index is the equal-weighted average and the
    # implied market beta of the three assets is about 1, 2 and 0.
    mcap = pd.DataFrame(1e9, index=dates, columns=returns.columns)

    directory = tmp_path / "panel"
    Panel(price, mcap).write(directory)
    return directory, dates


def test_betas_come_out_near_the_truth(aggregate, market):
    directory, dates = market
    asked = pd.DatetimeIndex([dates[400], dates[500]])
    betas = aggregate._market_betas(directory, asked)

    assert not betas.empty
    assert list(betas.columns) == ["AAA", "BBB", "CCC"]
    row = betas.iloc[-1]
    # The index is the average of the three, so AAA and BBB load on it in a 1:2
    # ratio and CCC barely at all.
    assert row["BBB"] > row["AAA"] > row["CCC"]
    assert row["CCC"] == pytest.approx(0.0, abs=0.25)
    assert row["BBB"] / row["AAA"] == pytest.approx(2.0, rel=0.25)


def test_betas_use_only_the_past(aggregate, market):
    """Scramble everything after the date and the beta must not move."""
    directory, dates = market
    cut = dates[400]

    before = aggregate._market_betas(directory, pd.DatetimeIndex([cut]))

    panel = Panel.read(directory)
    rng = np.random.default_rng(99)
    future = panel.price.index > cut
    scrambled = panel.price.copy()
    noise = rng.uniform(0.2, 5.0, scrambled.loc[future].shape)
    scrambled.loc[future] = scrambled.loc[future] * noise
    Panel(scrambled, panel.mcap).write(directory)

    after = aggregate._market_betas(directory, pd.DatetimeIndex([cut]))
    pd.testing.assert_frame_equal(before, after)


def test_missing_panel_is_not_fatal(aggregate, tmp_path):
    """Aggregation has to survive a run tree with no panel beside it."""
    assert aggregate._market_betas(tmp_path / "nope", pd.DatetimeIndex(["2024-01-01"])).empty


def test_exposure_is_the_weighted_beta(aggregate, market, tmp_path):
    directory, dates = market
    rebalances = pd.DatetimeIndex([dates[400], dates[500]])

    runs = tmp_path / "runs"
    for forecaster, weights in (
        ("rw", {"AAA": 0.0, "BBB": 0.0, "CCC": 1.0}),
        ("garch", {"AAA": 0.0, "BBB": 1.0, "CCC": 0.0}),
    ):
        target = runs / "backtests" / forecaster / "ga_seed00"
        target.mkdir(parents=True)
        frame = pd.DataFrame([weights] * len(rebalances), index=rebalances)
        frame.to_parquet(target / "weights.parquet")

    exposure = aggregate._portfolio_exposure(runs, directory, "ga")

    assert set(exposure["forecaster"]) == {"rw", "garch"}
    by_forecaster = exposure.groupby("forecaster")["exposure"].mean()
    assert by_forecaster["garch"] > by_forecaster["rw"]
    assert by_forecaster["rw"] == pytest.approx(0.0, abs=0.25)
    assert by_forecaster["garch"] == pytest.approx(2.0, rel=0.3)


def test_exposure_ignores_other_allocators(aggregate, market, tmp_path):
    directory, dates = market
    runs = tmp_path / "runs"
    target = runs / "backtests" / "garch" / "hrp_seed00"
    target.mkdir(parents=True)
    pd.DataFrame(
        [{"AAA": 1.0, "BBB": 0.0, "CCC": 0.0}], index=pd.DatetimeIndex([dates[400]])
    ).to_parquet(target / "weights.parquet")

    assert aggregate._portfolio_exposure(runs, directory, "ga").empty
