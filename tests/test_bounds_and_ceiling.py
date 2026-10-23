"""Two things the first full run got wrong, pinned so they cannot come back.

The scenario ceiling exists because the generators compounded daily draws into
30-day returns of 1e14, which then dominated the mean in a mean-CVaR objective and
chose the portfolio by themselves. The relaxation bound exists because the CVaR
linear programme was competing in the backtest while holding more names than the
cardinality allows, which made it beat the exact optimum for a reason that had
nothing to do with being a better allocator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scenport import alloc
from scenport.alloc.objective import Problem, is_feasible, score
from scenport.forecast import base, naive

# ---------------------------------------------------------------------------
# the ceiling


def _prices(n=800, vol=0.05, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n)
    return pd.DataFrame(
        {a: 100 * np.cumprod(1 + rng.normal(0, vol, n)) for a in ("AAA", "BBB")}, index=dates
    )


def test_the_default_ceiling_has_room_for_anything_ever_observed():
    """45x is the largest 30-day return in the panel, so the bound must clear it."""
    assert base.MAX_CUMULATIVE_RETURN > 45.53
    assert base.MAX_CUMULATIVE_RETURN < 1000, "a ceiling this high would not bind at all"


def test_paths_are_clipped_on_both_sides():
    forecaster = naive.HistoricalSimulation(max_cumulative_return=0.25)
    out = forecaster.simulate(_prices(), horizon=30, n_paths=400, rng=np.random.default_rng(1))

    assert out.paths.max() <= 0.25 + 1e-12
    assert out.paths.min() >= -1.0 - 1e-12, "a long-only position cannot lose more than it cost"


def test_the_clipped_fraction_is_recorded():
    """A generator that hits the ceiling often is saying something about itself."""
    loose = naive.HistoricalSimulation(max_cumulative_return=1e9)
    tight = naive.HistoricalSimulation(max_cumulative_return=0.05)
    kwargs = dict(horizon=30, n_paths=400, rng=np.random.default_rng(1))

    assert loose.simulate(_prices(), **kwargs).meta["clipped_share"] == 0.0
    assert tight.simulate(_prices(), **kwargs).meta["clipped_share"] > 0.0


def test_a_wild_generator_is_brought_back_to_the_bound():
    """The real failure: compounding fat-tailed daily draws over the horizon."""

    class Explosive(base.ScenarioForecaster):
        name = "explosive_for_test"

        def simulate_one(self, history, horizon, n_paths, rng):
            daily = rng.normal(0.5, 0.5, (n_paths, horizon))
            return np.cumprod(1.0 + daily, axis=1) - 1.0

    raw = Explosive().simulate_one(None, 30, 200, np.random.default_rng(0))
    assert raw.max() > 1e4, "the fixture has to actually explode"

    tamed = Explosive().simulate(_prices(), horizon=30, n_paths=200, rng=np.random.default_rng(0))
    assert tamed.paths.max() <= base.MAX_CUMULATIVE_RETURN + 1e-9
    assert tamed.meta["clipped_share"] > 0


def test_a_forecaster_that_replaces_simulate_cannot_step_around_the_ceiling():
    """The ceiling holds for forecasters that override simulate().

    The neural forecasters batch every asset through a single network, so they
    override simulate() wholesale instead of implementing simulate_one. A ceiling
    applied inside simulate() would never run for them. The bound is enforced in
    ScenarioSet, which every path into a scenario set goes through.
    """

    class Overriding(base.ScenarioForecaster):
        name = "overriding_for_test"

        def simulate_one(self, history, horizon, n_paths, rng):  # pragma: no cover
            raise NotImplementedError("this one builds the set itself")

        def simulate(self, history, horizon, n_paths, rng, origin=None, progress=None):
            return base.ScenarioSet(
                paths=np.full((6, 4, 2), 1e15),
                assets=["AAA", "BBB"],
                origin=pd.Timestamp("2024-01-01"),
                forecaster=self.name,
            )

    out = Overriding().simulate(_prices(), horizon=4, n_paths=6, rng=np.random.default_rng(0))
    assert out.paths.max() <= base.MAX_CUMULATIVE_RETURN + 1e-9
    assert out.meta["clipped_share"] > 0


def test_loading_an_unbounded_set_bounds_it():
    """Sets written before the ceiling existed are bounded when they are read."""
    import tempfile

    raw = base.ScenarioSet(
        paths=np.full((4, 3, 2), 5.0),
        assets=["AAA", "BBB"],
        origin=pd.Timestamp("2024-01-01"),
        forecaster="legacy_for_test",
    )
    raw.paths[0, 0, 0] = 1e12  # sneak past the constructor, as an old file would
    target = Path(tempfile.mkdtemp()) / "legacy.npz"
    raw.to_npz(target)

    back = base.ScenarioSet.from_npz(target)
    assert back.paths.max() <= base.MAX_CUMULATIVE_RETURN + 1e-9


# ---------------------------------------------------------------------------
# the relaxation bound


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    return Problem(
        scenarios=rng.normal(0.02, 0.30, (400, 15)),
        assets=[f"a{i}" for i in range(15)],
        w_max=0.20,
        w_min=0.02,
        cardinality=8,
        cvar_alpha=0.95,
        gamma=0.5,
        cost_rate=0.002,
    )


@pytest.mark.slow
def test_the_bound_is_above_the_exact_optimum(problem):
    """Which is exactly why it cannot be scored against allocators as a peer."""
    pytest.importorskip("cvxpy")
    bound = alloc.get("cvar_lp_bound")(problem, np.random.default_rng(0)).weights
    exact = alloc.get("milp")(problem, np.random.default_rng(0)).weights

    assert score(bound, problem)[0] > score(exact, problem)[0]
    assert not is_feasible(bound, problem)
    assert is_feasible(exact, problem)


@pytest.mark.slow
def test_the_feasible_lp_is_a_portfolio_somebody_could_hold(problem):
    pytest.importorskip("cvxpy")
    w = alloc.get("cvar_lp")(problem, np.random.default_rng(0)).weights

    assert is_feasible(w, problem)
    assert int((w > 1e-9).sum()) <= problem.cardinality
    assert w[w > 1e-9].min() >= problem.w_min - 1e-9


@pytest.mark.slow
def test_the_ordering_theory_requires(problem):
    """bound >= exact >= heuristic. Any other order means something is wrong."""
    pytest.importorskip("cvxpy")
    values = {
        name: score(alloc.get(name)(problem, np.random.default_rng(0)).weights, problem)[0]
        for name in ("cvar_lp_bound", "milp", "ga", "cvar_lp")
    }
    assert values["cvar_lp_bound"] >= values["milp"] >= values["ga"] - 1e-9
    assert values["milp"] >= values["cvar_lp"] - 1e-9


def test_the_bound_is_not_in_the_allocator_lists():
    """It must be impossible to put it in a backtest by picking 'all'."""
    assert "cvar_lp_bound" in alloc.BOUNDS
    for group in (alloc.HEURISTIC, alloc.CONVEX, alloc.EVOLUTIONARY, alloc.EXACT):
        assert "cvar_lp_bound" not in group
