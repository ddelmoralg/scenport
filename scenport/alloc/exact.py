"""Mixed-integer optimum of the constrained problem.

Rockafellar-Uryasev plus a binary support variable. Slow, but it is what makes the
optimality gaps measurable instead of assumed.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import Allocation, Allocator, register
from .objective import Problem

log = logging.getLogger(__name__)

SOLVER_ORDER = ("HIGHS", "SCIPY", "SCIP", "GLPK_MI", "CBC")


def _pick_solver(cp):
    installed = set(cp.installed_solvers())
    for name in SOLVER_ORDER:
        if name in installed:
            return getattr(cp, name)
    raise RuntimeError(
        "no mixed-integer solver available; install highspy or one of "
        + ", ".join(SOLVER_ORDER[1:])
    )


def subsample(problem: Problem, limit: int, rng: np.random.Generator) -> Problem:
    """Copy of ``problem`` with at most ``limit`` scenarios, drawn without replacement.

    The branch-and-bound tree grows fast in the scenario count, so large instances
    have to be thinned before they are solvable. Reported gaps are always computed
    against the same thinned instance.
    """
    if problem.n_scenarios <= limit:
        return problem
    idx = np.sort(rng.choice(problem.n_scenarios, size=limit, replace=False))
    reduced = Problem(**{**problem.__dict__, "scenarios": problem.scenarios[idx]})
    reduced.meta = dict(problem.meta, subsampled_from=problem.n_scenarios)
    return reduced


@register
class MilpExact(Allocator):
    """Branch and bound on the full constrained problem."""

    name = "milp"

    def solve(self, problem, rng):
        import cvxpy as cp

        limit = int(self.params.get("max_scenarios", 400))
        reduced = subsample(problem, limit, rng)

        R = reduced.scenarios
        n_scen = reduced.n_scenarios
        alpha = reduced.cvar_alpha

        w = cp.Variable(reduced.n, nonneg=True)
        b = cp.Variable(reduced.n, boolean=True)
        zeta = cp.Variable()
        u = cp.Variable(n_scen, nonneg=True)
        buy = cp.Variable(reduced.n, nonneg=True)
        sell = cp.Variable(reduced.n, nonneg=True)

        cvar = zeta + cp.sum(u) / ((1.0 - alpha) * n_scen)
        expected = cp.sum(R @ w) / n_scen
        cost = reduced.cost_rate * cp.sum(buy + sell)

        constraints = [
            cp.sum(w) == 1,
            u >= -(R @ w) - zeta,
            w <= reduced.w_max * b,
            w >= reduced.w_min * b,
            cp.sum(b) <= reduced.max_positions,
            w - reduced.w_prev == buy - sell,
        ]

        if reduced.objective != "mean_cvar":
            raise NotImplementedError(
                f"the MILP is only formulated for mean_cvar, not {reduced.objective!r}"
            )

        prob = cp.Problem(
            cp.Maximize((1.0 - reduced.gamma) * expected - reduced.gamma * cvar - cost),
            constraints,
        )

        solver = _pick_solver(cp)
        options = {}
        time_limit = self.params.get("time_limit")
        if time_limit is not None:
            options["time_limit"] = float(time_limit)
        try:
            prob.solve(solver=solver, **options)
        except (cp.error.SolverError, TypeError) as exc:
            log.warning("MILP solve failed with %s (%s); retrying without options", solver, exc)
            prob.solve(solver=solver)

        if w.value is None:
            raise RuntimeError(f"MILP returned no solution (status {prob.status})")

        weights = np.clip(np.asarray(w.value, dtype=float), 0.0, None)
        weights[weights < 1e-9] = 0.0
        weights = weights / weights.sum()

        return Allocation(
            weights,
            self.name,
            0.0,
            diagnostics={
                "status": prob.status,
                "milp_bound": float(prob.value),
                "n_scenarios_solved": n_scen,
                "solver": str(solver),
            },
        )


def optimality_gap(candidate: np.ndarray, problem: Problem, rng: np.random.Generator,
                   max_scenarios: int = 400) -> dict:
    """How far ``candidate`` sits below the proven optimum on the same instance.

    Both values are computed on the identical (possibly subsampled) scenario set,
    so the gap reflects the search and not the sampling.
    """
    from .objective import score

    reduced = subsample(problem, max_scenarios, rng)
    best = MilpExact(max_scenarios=max_scenarios)(reduced, rng)
    theirs = float(score(candidate, reduced)[0])
    mine = float(score(best.weights, reduced)[0])
    denom = abs(mine) if abs(mine) > 1e-12 else 1.0
    return {
        "milp_objective": mine,
        "candidate_objective": theirs,
        "absolute_gap": mine - theirs,
        "relative_gap": (mine - theirs) / denom,
        "milp_seconds": best.seconds,
        "n_scenarios_solved": reduced.n_scenarios,
    }
