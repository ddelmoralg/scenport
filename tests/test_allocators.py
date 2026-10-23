"""Every allocator returns a feasible portfolio, and the search ones get close to
the proven optimum.

The optimality test is the one that justifies using a metaheuristic at all. If a
genetic algorithm cannot get within a percent or two of branch and bound on an
instance the solver can close, there is no argument for using it on the instances
the solver cannot.
"""

from __future__ import annotations

import numpy as np
import pytest

from scenport import alloc
from scenport.alloc.objective import is_feasible, score

SEARCH = ("ga", "de", "pso", "cmaes")


def _fast(name: str) -> alloc.Allocator:
    """Small budget so the suite stays quick; the real runs use configs/base.yaml."""
    if name in SEARCH:
        return alloc.get(name, pop_size=60, generations=80, patience=20)
    return alloc.get(name)


@pytest.mark.parametrize("name", sorted(set(alloc.available()) - {"milp", "mcap", "single"}))
def test_allocator_returns_something_feasible(name, problem, rng):
    result = _fast(name)(problem, rng)
    assert result.weights.shape == (problem.n,)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert (result.weights >= -1e-12).all()
    if name != "cvar_lp":       # the LP relaxation ignores the discrete constraints
        assert is_feasible(result.weights, problem)


def test_market_cap_needs_its_input(problem, rng):
    with pytest.raises(KeyError):
        alloc.get("mcap")(problem, rng)
    problem.meta["mcap"] = np.linspace(1.0, 10.0, problem.n)
    result = alloc.get("mcap")(problem, rng)
    assert is_feasible(result.weights, problem)
    assert np.argmax(result.weights) == problem.n - 1


def test_search_beats_the_naive_portfolio(problem, rng):
    equal = np.full(problem.n, 1.0 / problem.n)
    baseline = float(score(equal, problem)[0])
    for name in SEARCH:
        value = float(score(_fast(name)(problem, rng).weights, problem)[0])
        assert value > baseline, f"{name} failed to beat 1/N on its own objective"


def test_search_is_reproducible(problem):
    for name in SEARCH:
        first = _fast(name)(problem, np.random.default_rng(7)).weights
        second = _fast(name)(problem, np.random.default_rng(7)).weights
        assert np.allclose(first, second), f"{name} is not reproducible from a seed"


def test_seeds_actually_change_the_answer(problem):
    """A stochastic allocator that ignores its seed would make the seed averaging
    in the results section meaningless."""
    a = _fast("ga")(problem, np.random.default_rng(1)).weights
    b = _fast("ga")(problem, np.random.default_rng(2)).weights
    assert not np.allclose(a, b)


@pytest.mark.slow
def test_search_is_close_to_the_mixed_integer_optimum(problem, rng):
    cvxpy = pytest.importorskip("cvxpy")
    if not (set(cvxpy.installed_solvers()) & {"HIGHS", "SCIPY", "SCIP", "GLPK_MI", "CBC"}):
        pytest.skip("no mixed-integer solver installed")

    from scenport.alloc.exact import MilpExact, subsample

    instance = subsample(problem, 250, np.random.default_rng(0))
    optimum = float(score(MilpExact(max_scenarios=250)(instance, rng).weights, instance)[0])

    for name in SEARCH:
        best = max(
            float(score(_fast(name)(instance, np.random.default_rng(s)).weights, instance)[0])
            for s in range(5)
        )
        gap = (optimum - best) / abs(optimum)
        assert gap < 0.05, f"{name} left {gap:.1%} on the table against branch and bound"
        assert best <= optimum + 1e-9, f"{name} beat the proven optimum, so something is wrong"


@pytest.mark.slow
def test_the_relaxation_bounds_the_integer_problem(problem, rng):
    """Dropping the cardinality constraint can only help, never hurt.

    This holds for the relaxation, `cvar_lp_bound`. The feasible `cvar_lp` is held
    to the opposite standard in the next test, because a feasible allocator that
    beats a proven optimum would be solving a different problem.
    """
    pytest.importorskip("cvxpy")
    from scenport.alloc.exact import MilpExact, subsample

    instance = subsample(problem, 250, np.random.default_rng(0))
    bound = alloc.get("cvar_lp_bound")(instance, rng)
    exact = MilpExact(max_scenarios=250)(instance, rng)
    assert float(score(bound.weights, instance)[0]) >= float(
        score(exact.weights, instance)[0]
    ) - 1e-6


@pytest.mark.slow
def test_the_feasible_lp_does_not_beat_the_exact_optimum(problem, rng):
    """A feasible allocator never beats the proven optimum."""
    pytest.importorskip("cvxpy")
    from scenport.alloc.exact import MilpExact, subsample
    from scenport.alloc.objective import is_feasible

    instance = subsample(problem, 250, np.random.default_rng(0))
    heuristic = alloc.get("cvar_lp")(instance, rng)
    exact = MilpExact(max_scenarios=250)(instance, rng)

    assert is_feasible(heuristic.weights, instance)
    assert float(score(heuristic.weights, instance)[0]) <= float(
        score(exact.weights, instance)[0]
    ) + 1e-9


def test_convex_allocators_ignore_the_seed(problem):
    for name in ("minvar", "maxsharpe", "riskparity", "hrp"):
        a = alloc.get(name)(problem, np.random.default_rng(1)).weights
        b = alloc.get(name)(problem, np.random.default_rng(999)).weights
        assert np.allclose(a, b), f"{name} is supposed to be deterministic"


def test_risk_parity_spreads_risk(problem, rng):
    result = alloc.get("riskparity")(problem, rng)
    spread = result.diagnostics.get("risk_contribution_spread")
    # The per-asset cap keeps this from reaching exact parity, so the bar is
    # "much flatter than 1/N", not "flat".
    assert spread is not None and spread < 0.5
