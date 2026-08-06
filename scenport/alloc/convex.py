"""Allocators that solve a convex relaxation to global optimality.

``cvar_lp`` optimises the same objective as the evolutionary allocators minus the
discrete constraints, so the gap between them prices those constraints. All
deterministic; none of them look at the seed.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from .base import Allocation, Allocator, register
from .objective import Problem, project_simplex_box

log = logging.getLogger(__name__)


def covariance(problem: Problem, shrink: bool = True) -> np.ndarray:
    """Covariance of the horizon returns implied by the scenario set.

    Ledoit-Wolf shrinkage by default. A thousand scenarios against fifteen assets
    is a comfortable ratio, but the same code runs on the wider universes of the
    scalability experiment, where the sample estimate starts to be ill conditioned.
    """
    if not shrink:
        return np.cov(problem.scenarios, rowvar=False, ddof=1)
    from sklearn.covariance import ledoit_wolf

    return ledoit_wolf(problem.scenarios)[0]


def _box_bounds(problem: Problem) -> list[tuple[float, float]]:
    return [(0.0, problem.w_max)] * problem.n


@register
class MinimumVariance(Allocator):
    """Smallest scenario variance subject to the box. Ignores expected returns."""

    name = "minvar"

    def solve(self, problem, rng):
        import cvxpy as cp

        sigma = covariance(problem, shrink=self.params.get("shrink", True))
        w = cp.Variable(problem.n, nonneg=True)
        constraints = [cp.sum(w) == 1, w <= problem.w_max]
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma))), constraints)
        prob.solve(solver=cp.CLARABEL)
        return Allocation(_clean(w.value, problem), self.name, 0.0,
                          diagnostics={"status": prob.status})


@register
class MaximumSharpe(Allocator):
    """Tangency portfolio under the box, by the standard convex reformulation.

    Maximising ``(mu'w - rf) / sqrt(w' S w)`` is not concave in ``w``, but with
    ``y = kappa w`` and ``(mu - rf)'y = 1`` it becomes a QP (Cornuejols and
    Tutuncu). Worth solving exactly: without the discrete constraints there is
    nothing here a metaheuristic can find that a QP solver cannot.
    """

    name = "maxsharpe"

    def solve(self, problem, rng):
        import cvxpy as cp

        sigma = covariance(problem, shrink=self.params.get("shrink", True))
        excess = problem.scenarios.mean(axis=0) - problem.rf

        if excess.max() <= 0:
            log.info("no asset has a positive expected excess return; falling back to minvar")
            return MinimumVariance().solve(problem, rng)

        y = cp.Variable(problem.n, nonneg=True)
        kappa = cp.Variable(nonneg=True)
        constraints = [excess @ y == 1, cp.sum(y) == kappa, y <= problem.w_max * kappa]
        prob = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(sigma))), constraints)
        prob.solve(solver=cp.CLARABEL)

        if y.value is None or kappa.value is None or kappa.value <= 1e-12:
            log.warning("max-Sharpe QP returned %s; falling back to minvar", prob.status)
            return MinimumVariance().solve(problem, rng)
        return Allocation(_clean(y.value / kappa.value, problem), self.name, 0.0,
                          diagnostics={"status": prob.status})


@register
class RiskParity(Allocator):
    """Equal risk contribution, via the convex log-barrier formulation.

    Minimising ``0.5 w'Sw - (1/n) sum log w_i`` over the positive orthant has the
    equal-risk-contribution portfolio as its unique solution up to scale (Spinu).
    The per-asset cap is applied afterwards by projection, so the cap can break
    exact parity -- that is reported rather than hidden.
    """

    name = "riskparity"

    def solve(self, problem, rng):
        sigma = covariance(problem, shrink=self.params.get("shrink", True))
        n = problem.n

        def objective(w):
            return 0.5 * w @ sigma @ w - np.mean(np.log(w))

        def gradient(w):
            return sigma @ w - 1.0 / (n * w)

        x0 = np.full(n, 1.0 / n)
        res = minimize(
            objective, x0, jac=gradient, method="L-BFGS-B",
            bounds=[(1e-8, None)] * n, options={"maxiter": 500},
        )
        raw = res.x / res.x.sum()
        contributions = raw * (sigma @ raw)
        return Allocation(
            _clean(raw, problem),
            self.name,
            0.0,
            diagnostics={
                "converged": bool(res.success),
                "risk_contribution_spread": float(contributions.std() / contributions.mean()),
            },
        )


@register
class HierarchicalRiskParity(Allocator):
    """Lopez de Prado's HRP: cluster, quasi-diagonalise, bisect recursively.

    Never inverts the covariance matrix, which is why it holds up when the
    correlation structure is close to singular -- a routine situation in crypto,
    where most tokens track bitcoin.
    """

    name = "hrp"

    def solve(self, problem, rng):
        sigma = covariance(problem, shrink=False)
        sd = np.sqrt(np.diag(sigma))
        corr = sigma / np.outer(sd, sd)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(np.nan_to_num(corr, nan=0.0), -1.0, 1.0)

        distance = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
        np.fill_diagonal(distance, 0.0)
        link = linkage(squareform(distance, checks=False), method="single")
        order = _leaf_order(link, problem.n)

        w = np.ones(problem.n)
        _bisect(w, order, sigma)
        return Allocation(_clean(w, problem), self.name, 0.0)


@register
class CvarLinearProgram(Allocator):
    """Mean-CVaR at global optimality, without the discrete constraints.

    Rockafellar and Uryasev's linear formulation of CVaR, with the transaction cost
    linearised through the usual pair of non-negative auxiliaries. This is the upper
    bound on what any allocator can achieve on the relaxed problem, so it is the
    reference the mixed-integer and evolutionary solutions are measured against.
    """

    name = "cvar_lp"

    def solve(self, problem, rng):
        import cvxpy as cp

        R = problem.scenarios
        n_scen = problem.n_scenarios
        alpha = problem.cvar_alpha

        w = cp.Variable(problem.n, nonneg=True)
        zeta = cp.Variable()
        u = cp.Variable(n_scen, nonneg=True)
        buy = cp.Variable(problem.n, nonneg=True)
        sell = cp.Variable(problem.n, nonneg=True)

        cvar = zeta + cp.sum(u) / ((1.0 - alpha) * n_scen)
        expected = cp.sum(R @ w) / n_scen
        cost = problem.cost_rate * cp.sum(buy + sell)

        constraints = [
            cp.sum(w) == 1,
            w <= problem.w_max,
            u >= -(R @ w) - zeta,
            w - problem.w_prev == buy - sell,
        ]
        objective = cp.Maximize((1.0 - problem.gamma) * expected - problem.gamma * cvar - cost)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL)

        if w.value is None:
            raise RuntimeError(f"CVaR LP failed with status {prob.status}")
        return Allocation(
            _clean(w.value, problem, enforce_support=False),
            self.name,
            0.0,
            diagnostics={"status": prob.status, "lp_bound": float(prob.value)},
        )


# ---------------------------------------------------------------------------


def _clean(raw, problem: Problem, enforce_support: bool = True) -> np.ndarray:
    """Tidy a solver answer into a portfolio that satisfies every constraint.

    Solvers return weights of order 1e-12 rather than exact zeros, and none of
    these formulations knows about the cardinality or minimum-holding constraints,
    so the support is trimmed here and the remainder reprojected.
    """
    w = np.clip(np.nan_to_num(np.asarray(raw, dtype=float), nan=0.0), 0.0, None)
    if w.sum() <= 0:
        w = np.ones(problem.n)
    if not enforce_support:
        return w / w.sum()

    keep = min(problem.max_positions, problem.n)
    chosen = np.argsort(-w, kind="stable")[:keep]
    out = np.zeros(problem.n)
    out[chosen] = project_simplex_box(
        w[chosen] / w[chosen].sum(), problem.w_min, problem.w_max
    )
    return out


def _leaf_order(link: np.ndarray, n: int) -> list[int]:
    tree, _ = to_tree(link, rd=True)
    return [int(i) for i in tree.pre_order(lambda x: x.id) if int(i) < n]


def _cluster_variance(sigma: np.ndarray, idx: list[int]) -> float:
    sub = sigma[np.ix_(idx, idx)]
    inv_diag = 1.0 / np.maximum(np.diag(sub), 1e-16)
    w = inv_diag / inv_diag.sum()
    return float(w @ sub @ w)


def _bisect(w: np.ndarray, order: list[int], sigma: np.ndarray) -> None:
    """Recursive bisection, in place on ``w``."""
    clusters = [order]
    while clusters:
        clusters = [
            part
            for cluster in clusters
            for part in (cluster[: len(cluster) // 2], cluster[len(cluster) // 2 :])
            if len(part) > 0
        ]
        for i in range(0, len(clusters) - 1, 2):
            left, right = clusters[i], clusters[i + 1]
            var_l, var_r = _cluster_variance(sigma, left), _cluster_variance(sigma, right)
            factor = 1.0 - var_l / (var_l + var_r)
            w[left] *= factor
            w[right] *= 1.0 - factor
        clusters = [c for c in clusters if len(c) > 1]
