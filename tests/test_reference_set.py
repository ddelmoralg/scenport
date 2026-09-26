"""The reference set two decisions get priced against.

Its job is to be neutral and to be available at the decision date. If either fails,
the objective difference stops being a statement about the decision and turns into
realised performance under a different name, which is the whole thing it exists to
avoid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scenport.forecast import naive, reference

ASSETS = ["AAA", "BBB", "CCC"]


@pytest.fixture
def returns():
    """Two assets driven by a common factor, one independent of it."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-03", periods=800)
    factor = rng.normal(0.0, 0.03, len(dates))
    return pd.DataFrame(
        {
            "AAA": factor + rng.normal(0, 0.004, len(dates)),
            "BBB": factor + rng.normal(0, 0.004, len(dates)),
            "CCC": rng.normal(0, 0.03, len(dates)),
        },
        index=dates,
    )


def _draw(returns, date, **kwargs):
    kwargs.setdefault("n_paths", 3000)
    kwargs.setdefault("seed", reference.reference_seed(date))
    return reference.reference_scenarios(returns, date, ASSETS, horizon=30, **kwargs)


def test_shape_and_orientation(returns):
    cut = returns.index[700]
    out = _draw(returns, cut)
    assert out.shape == (3000, len(ASSETS))
    assert np.isfinite(out).all()


def test_it_cannot_see_past_the_date(returns):
    """The property the objective difference depends on entirely."""
    cut = returns.index[700]
    before = _draw(returns, cut)

    scrambled = returns.copy()
    scrambled.loc[cut:] = scrambled.loc[cut:] * 7.0
    after = _draw(scrambled, cut)

    np.testing.assert_array_equal(before, after)


def test_same_seed_gives_the_same_set(returns):
    cut = returns.index[700]
    np.testing.assert_array_equal(_draw(returns, cut), _draw(returns, cut))


def test_different_dates_get_different_seeds(returns):
    a, b = returns.index[700], returns.index[701]
    assert reference.reference_seed(a) != reference.reference_seed(b)


def test_the_seed_stream_is_its_own(returns):
    """Sharing a stream with a forecaster would tie the yardstick to a competitor."""
    from scenport.config import stable_seed

    date = returns.index[700]
    mine = reference.reference_seed(date)
    for forecaster in ("rw", "garch", "prophet", "lstm"):
        assert mine != stable_seed(forecaster, date.strftime("%Y-%m-%d"))


def test_cross_asset_dependence_survives(returns):
    """Whole rows are resampled, so the empirical dependence comes along untouched.

    This is what lets the reference stay neutral about the coupling model, which
    matters because the coupling model is one of the things under study.
    """
    cut = returns.index[700]
    out = _draw(returns, cut)
    simulated = np.corrcoef(out.T)
    empirical = returns.loc[: cut - pd.Timedelta(days=1)].corr().to_numpy()

    assert simulated[0, 1] == pytest.approx(empirical[0, 1], abs=0.08)
    assert abs(simulated[0, 2]) < 0.2


def test_it_is_not_the_historical_simulation_forecaster(returns):
    """The reference must not be the same object as one of the eleven it judges."""
    assert reference.REFERENCE_BLOCK != naive.HistoricalSimulation().params.get("block", 5)
    assert reference.REFERENCE_BLOCK > 5
    assert reference.REFERENCE_PATHS >= 10_000


def test_it_is_not_registered_as_a_forecaster():
    """It must be impossible to pick it as a competitor by accident."""
    from scenport.forecast import base

    assert "reference" not in base.available()


def test_too_little_history_comes_back_empty(returns):
    early = returns.index[5]
    assert _draw(returns, early).shape[0] == 0


def test_a_longer_block_keeps_more_autocorrelation():
    """Sanity check on the block machinery, on a series built to be persistent."""
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2022-01-03", periods=800)
    series = pd.Series(rng.normal(0, 0.02, len(dates))).rolling(20, min_periods=1).mean()
    frame = pd.DataFrame({name: series.to_numpy() for name in ASSETS}, index=dates)

    cut = dates[700]
    short = reference.reference_scenarios(
        frame, cut, ASSETS, horizon=40, n_paths=2000, block=2, seed=1
    )
    long = reference.reference_scenarios(
        frame, cut, ASSETS, horizon=40, n_paths=2000, block=60, seed=1
    )
    assert long[:, 0].std() > short[:, 0].std()
