"""The decoder has to produce a feasible portfolio for any input at all.

If it does not, the search algorithms are optimising over a space that includes
portfolios nobody could hold, and every number downstream is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from scenport.alloc.objective import (
    cvar,
    decode,
    feasible_cardinality,
    is_feasible,
    project_simplex_box,
    report,
    score,
)


def test_projection_hits_the_simplex_and_the_box(rng):
    for _ in range(200):
        k = rng.integers(4, 12)
        lo, hi = 0.02, max(0.15, 1.5 / k)
        v = rng.normal(0, 1, size=k)
        w = project_simplex_box(v, lo, hi)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert (w >= lo - 1e-9).all()
        assert (w <= hi + 1e-9).all()


def test_projection_is_the_nearest_point(rng):
    """Compare against a brute-force search on a small instance."""
    v = np.array([0.9, 0.05, 0.02, 0.6])
    lo, hi = 0.05, 0.5
    w = project_simplex_box(v, lo, hi)

    best = np.inf
    for _ in range(20000):
        candidate = rng.dirichlet(np.ones(4))
        if (candidate < lo - 1e-12).any() or (candidate > hi + 1e-12).any():
            continue
        best = min(best, float(np.sum((candidate - v) ** 2)))
    assert float(np.sum((w - v) ** 2)) <= best + 1e-6


def test_projection_rejects_impossible_bounds():
    with pytest.raises(ValueError):
        project_simplex_box(np.ones(3), lo=0.0, hi=0.2)      # 3 * 0.2 < 1
    with pytest.raises(ValueError):
        project_simplex_box(np.ones(3), lo=0.5, hi=1.0)      # 3 * 0.5 > 1


def test_decode_is_always_feasible(problem, rng):
    for scale in (0.01, 1.0, 100.0):
        for _ in range(300):
            x = rng.normal(0, scale, size=problem.n)
            w = decode(x, problem)
            assert is_feasible(w, problem), (x, w)


def test_decode_handles_degenerate_input(problem):
    for x in (np.zeros(problem.n), -np.ones(problem.n), np.full(problem.n, 1e-12)):
        assert is_feasible(decode(x, problem), problem)


def test_cardinality_is_respected(problem, rng):
    for _ in range(200):
        w = decode(rng.normal(0, 1, size=problem.n), problem)
        assert int((w > 1e-9).sum()) <= problem.cardinality


def test_cvar_matches_the_naive_computation(rng):
    returns = rng.standard_normal((3, 1000))
    tail = 50
    fast = cvar(returns, tail)
    slow = np.array([-np.sort(row)[:tail].mean() for row in returns])
    assert fast == pytest.approx(slow)


def test_cvar_is_at_least_var(problem, rng):
    w = decode(rng.normal(size=problem.n), problem)
    stats = report(w, problem)
    assert stats["cvar"] >= stats["var"] - 1e-9


def test_gamma_zero_maximises_expected_return(scenarios):
    from scenport.alloc.objective import Problem

    kwargs = dict(
        scenarios=scenarios,
        assets=[f"a{i}" for i in range(scenarios.shape[1])],
        w_max=1.0,
        w_min=0.0,
        cardinality=None,
        cost_rate=0.0,
    )
    mean_only = Problem(**kwargs, gamma=0.0)
    w = np.zeros(mean_only.n)
    w[int(np.argmax(scenarios.mean(axis=0)))] = 1.0
    everything_else = np.full(mean_only.n, 1.0 / mean_only.n)
    assert score(w, mean_only)[0] >= score(everything_else, mean_only)[0]


def test_cost_reduces_the_objective(scenarios):
    from scenport.alloc.objective import Problem

    assets = [f"a{i}" for i in range(scenarios.shape[1])]
    previous = np.zeros(len(assets))
    free = Problem(scenarios=scenarios, assets=assets, w_prev=previous, cost_rate=0.0)
    costly = Problem(scenarios=scenarios, assets=assets, w_prev=previous, cost_rate=0.01)
    w = np.full(len(assets), 1.0 / len(assets))
    assert score(w, costly)[0] < score(w, free)[0]


# The cardinality has to agree with the weight bounds. K names need K * w_min <= 1
# to fit the budget and K * w_max >= 1 to fill it, and feasible_cardinality is the
# one place that does the arithmetic.


def test_cardinality_is_raised_when_the_cap_cannot_fill_the_budget():
    # Eight names at ten per cent each reach 0.8. Ten is the smallest that works.
    assert feasible_cardinality(8, 70, 0.02, 0.10) == 10


def test_cardinality_is_capped_when_the_floor_will_not_fit():
    # Sixty-six names at two per cent each need 1.32 of a budget of 1.
    assert feasible_cardinality(66, 70, 0.02, 0.20) == 50


def test_cardinality_never_exceeds_the_universe():
    assert feasible_cardinality(100, 15, 0.02, 0.20) == 15


def test_the_paper_configuration_is_left_alone():
    # K = 8 with weights in [0.02, 0.2] is feasible, so nothing should move.
    assert feasible_cardinality(8, 70, 0.02, 0.20) == 8


def test_impossible_bounds_raise_rather_than_round():
    with pytest.raises(ValueError, match="no cardinality works"):
        feasible_cardinality(8, 5, 0.02, 0.10)


def test_every_returned_cardinality_admits_a_portfolio(rng):
    for _ in range(200):
        n = int(rng.integers(5, 80))
        w_min = float(rng.uniform(0.005, 0.05))
        w_max = float(rng.uniform(max(w_min * 2, 1.0 / n), 0.5))
        try:
            k = feasible_cardinality(int(rng.integers(1, n + 1)), n, w_min, w_max)
        except ValueError:
            continue
        assert 1 <= k <= n
        # Feasibility of a k-name portfolio is exactly these two inequalities.
        assert k * w_min <= 1.0 + 1e-12
        assert k * w_max >= 1.0 - 1e-12
        project_simplex_box(np.full(k, 1.0 / k), w_min, w_max)
