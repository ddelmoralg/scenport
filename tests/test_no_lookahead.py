"""No decision may depend on a price that had not been printed yet.

Blunt on purpose: scramble the future, check nothing about the past moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scenport import alloc
from scenport.backtest import engine
from scenport.data.panel import Panel
from scenport.forecast import get


def _run(panel, universe, seed=0, cost_bps=20.0):
    return engine.run(
        panel,
        universe,
        get("rw"),
        alloc.get("ga", pop_size=30, generations=15, patience=5),
        horizon=30,
        train_days=540,
        n_paths=120,
        coupling="gaussian_copula",
        cost_bps=cost_bps,
        seed=seed,
        problem_kwargs={"w_max": 0.35, "w_min": 0.05, "cardinality": 4},
    )


def test_scrambling_the_future_leaves_past_weights_untouched(
    synthetic_panel, synthetic_universe, rng
):
    cut = synthetic_universe.dates[len(synthetic_universe.dates) // 2]

    baseline = _run(synthetic_panel, synthetic_universe)

    scrambled = synthetic_panel.price.copy()
    tail = scrambled.index > cut
    noise = rng.uniform(0.3, 3.0, size=scrambled.loc[tail].shape)
    scrambled.loc[tail] = scrambled.loc[tail] * noise
    perturbed = _run(Panel(price=scrambled, mcap=synthetic_panel.mcap), synthetic_universe)

    before = baseline.weights.loc[baseline.weights.index <= cut]
    after = perturbed.weights.loc[perturbed.weights.index <= cut]
    pd.testing.assert_frame_equal(before, after, atol=1e-12)


def test_realised_returns_do_change_when_the_future_changes(
    synthetic_panel, synthetic_universe, rng
):
    """The mirror image of the test above: if nothing changed, it would prove
    the backtest is not reading prices at all."""
    cut = synthetic_universe.dates[len(synthetic_universe.dates) // 2]
    baseline = _run(synthetic_panel, synthetic_universe)

    scrambled = synthetic_panel.price.copy()
    tail = scrambled.index > cut
    noise = rng.uniform(0.3, 3.0, size=scrambled.loc[tail].shape)
    scrambled.loc[tail] = scrambled.loc[tail] * noise
    perturbed = _run(Panel(price=scrambled, mcap=synthetic_panel.mcap), synthetic_universe)

    late = baseline.daily.index > cut
    aligned = perturbed.daily.reindex(baseline.daily.index)
    assert not np.allclose(baseline.daily[late], aligned[late])


def test_the_forecaster_never_sees_the_decision_day(synthetic_panel, synthetic_universe):
    """The price at the rebalancing date is the one we trade at, not an input."""
    seen: list[pd.Timestamp] = []

    class Spy(type(get("rw"))):
        name = "spy"

        def simulate(self, history, horizon, n_paths, rng, origin=None, progress=None):
            seen.append(history.index.max())
            return super().simulate(history, horizon, n_paths, rng, origin=origin)

    engine.run(
        synthetic_panel,
        synthetic_universe,
        Spy(),
        alloc.get("equal"),
        horizon=30,
        train_days=540,
        n_paths=50,
        cost_bps=0.0,
        seed=0,
    )
    assert len(seen) == len(synthetic_universe.dates)
    for last_seen, decision_date in zip(seen, synthetic_universe.dates, strict=True):
        assert last_seen < decision_date


def test_universe_membership_uses_no_future_information(synthetic_panel):
    """Truncating the panel after a date must not change the universe before it."""
    from scenport.data import universe as uni

    dates = uni.rebalance_dates(synthetic_panel, "2021-01-01", "2024-11-01", "MS")
    full = uni.build(synthetic_panel, dates, size=6, min_history_days=540, rank_window=30)

    cut = dates[len(dates) // 2]
    truncated_panel = synthetic_panel.slice(end=cut)
    early = [d for d in dates if d <= cut]
    truncated = uni.build(
        truncated_panel, early, size=6, min_history_days=540, rank_window=30
    )

    for date in early:
        assert full.at(date) == truncated.at(date)


def test_evaluation_windows_are_shared_across_strategies(
    synthetic_panel, synthetic_universe
):
    """Every strategy has to be scored over exactly the same days."""
    def equal(date, assets):
        return pd.Series(np.full(len(assets), 1.0 / len(assets)), index=assets)

    def concentrated(date, assets):
        weights = np.zeros(len(assets))
        weights[0] = 1.0
        return pd.Series(weights, index=assets)

    windows = [
        engine.run_benchmark(synthetic_panel, synthetic_universe, fn, cost_bps=0.0,
                             label=name).daily.index
        for name, fn in (("equal", equal), ("concentrated", concentrated))
    ]
    windows.append(_run(synthetic_panel, synthetic_universe).daily.index)

    for other in windows[1:]:
        assert windows[0].equals(other)


@pytest.mark.parametrize("cost_bps", [0.0, 20.0, 100.0])
def test_cost_only_ever_reduces_the_return(synthetic_panel, synthetic_universe, cost_bps):
    result = _run(synthetic_panel, synthetic_universe, cost_bps=cost_bps)
    assert (result.periods["net_return"] <= result.periods["gross_return"] + 1e-12).all()
    assert (result.periods["cost"] >= -1e-12).all()
