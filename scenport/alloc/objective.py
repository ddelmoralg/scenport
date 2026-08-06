"""The allocation problem.

    max  (1-g) E[r] - g CVaR_a(r) - c ||w - w_prev||_1
    s.t. sum(w) = 1,  0 <= w_i <= w_max,  w_i in {0} u [w_min, w_max],  ||w||_0 <= K

evaluated directly on the scenario matrix -- no expected-return vector, no
covariance. Feasibility lives in the decoder, so every candidate the search sees is
a portfolio somebody could hold. notes/objective.md has the reasoning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

OBJECTIVES = ("mean_cvar", "sharpe", "mean_variance")


@dataclass
class Problem:
    """Everything the allocator needs at one rebalancing date."""

    scenarios: np.ndarray                 # (S, n) horizon simple returns
    assets: list[str]
    w_prev: np.ndarray | None = None      # weights carried in from the previous period
    w_max: float = 0.20
    w_min: float = 0.02
    cardinality: int | None = 8
    cvar_alpha: float = 0.95
    gamma: float = 0.5                    # 0 -> pure mean, 1 -> pure -CVaR
    cost_rate: float = 0.0020             # one way, as a fraction of turnover
    rf: float = 0.0                       # horizon-matched risk-free rate
    objective: str = "mean_cvar"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"unknown objective {self.objective!r}; expected {OBJECTIVES}")
        self.scenarios = np.ascontiguousarray(self.scenarios, dtype=float)
        if self.scenarios.shape[1] != len(self.assets):
            raise ValueError("scenario matrix and asset list disagree on n")
        if self.w_prev is None:
            self.w_prev = np.zeros(self.n)
        self.w_prev = np.asarray(self.w_prev, dtype=float)

        if self.cardinality is not None and self.cardinality < self.min_positions:
            raise ValueError(
                f"cardinality {self.cardinality} cannot reach a fully invested portfolio "
                f"with w_max={self.w_max}; at least {self.min_positions} positions are needed"
            )
        if self.w_min * self.min_positions > 1.0:
            raise ValueError("w_min is too large to fill the portfolio")

    @property
    def n(self) -> int:
        return self.scenarios.shape[1]

    @property
    def n_scenarios(self) -> int:
        return self.scenarios.shape[0]

    @property
    def min_positions(self) -> int:
        """Fewest positions that can still add up to one under the per-asset cap."""
        return max(int(math.ceil(1.0 / self.w_max)), 1)

    @property
    def max_positions(self) -> int:
        return min(self.cardinality or self.n, self.n)

    @property
    def tail_size(self) -> int:
        """Number of scenarios averaged into the CVaR estimate."""
        return max(int(round((1.0 - self.cvar_alpha) * self.n_scenarios)), 1)


# ---------------------------------------------------------------------------
# decoding


def project_simplex_box(v: np.ndarray, lo: float, hi: float, tol: float = 1e-10) -> np.ndarray:
    """Closest point to ``v`` in ``{w : sum w = 1, lo <= w_i <= hi}``.

    The solution has the form ``clip(v + lam, lo, hi)`` for a scalar ``lam``, and the
    sum is non-decreasing in ``lam``, so a bisection finds it exactly. Bounded at 100
    iterations, which takes the residual well below anything that matters here.
    """
    k = len(v)
    if k * lo > 1.0 + tol or k * hi < 1.0 - tol:
        raise ValueError(f"no point in the simplex satisfies [{lo}, {hi}] with {k} assets")

    left, right = lo - v.max(), hi - v.min()
    for _ in range(100):
        mid = 0.5 * (left + right)
        total = np.clip(v + mid, lo, hi).sum()
        if abs(total - 1.0) < tol:
            break
        if total < 1.0:
            left = mid
        else:
            right = mid
    w = np.clip(v + 0.5 * (left + right), lo, hi)
    return w / w.sum()


def decode(x: np.ndarray, problem: Problem) -> np.ndarray:
    """Map an unconstrained real vector to a feasible portfolio.

    The sign of a gene selects the asset and its magnitude proposes a weight. That
    keeps the search space continuous while still letting the population explore
    different supports: a gene crossing zero adds or removes a position.

    The support is trimmed to at most ``cardinality`` entries and padded up to
    ``min_positions`` when too few genes are positive, then the proposal is
    projected onto the simplex subject to the box.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(-x, kind="stable")

    chosen = np.flatnonzero(x > 0)
    if len(chosen) > problem.max_positions:
        chosen = order[: problem.max_positions]
    elif len(chosen) < problem.min_positions:
        chosen = order[: problem.min_positions]

    proposal = np.clip(x[chosen], 0.0, None)
    if proposal.sum() <= 0:
        proposal = np.ones(len(chosen))
    proposal = proposal / proposal.sum()

    w = np.zeros(problem.n)
    w[chosen] = project_simplex_box(proposal, problem.w_min, problem.w_max)
    return w


def decode_population(X: np.ndarray, problem: Problem) -> np.ndarray:
    """``decode`` applied row-wise to a (P, n) population."""
    return np.vstack([decode(row, problem) for row in np.atleast_2d(X)])


# ---------------------------------------------------------------------------
# evaluation


def portfolio_returns(W: np.ndarray, problem: Problem) -> np.ndarray:
    """(P, S) scenario returns of each portfolio in ``W``."""
    return np.atleast_2d(W) @ problem.scenarios.T


def cvar(returns: np.ndarray, tail: int) -> np.ndarray:
    """Conditional value at risk as a positive loss, row-wise over (P, S).

    ``np.partition`` puts the ``tail`` worst scenarios at the front without paying
    for a full sort. With a thousand scenarios inside a population loop that is the
    difference between the objective being cheap and being the bottleneck.
    """
    worst = np.partition(returns, tail - 1, axis=1)[:, :tail]
    return -worst.mean(axis=1)


def turnover(W: np.ndarray, w_prev: np.ndarray) -> np.ndarray:
    return np.abs(np.atleast_2d(W) - w_prev[None, :]).sum(axis=1)


def score(W: np.ndarray, problem: Problem) -> np.ndarray:
    """Objective value of each portfolio in ``W``. Higher is better."""
    W = np.atleast_2d(W)
    r = portfolio_returns(W, problem)
    cost = problem.cost_rate * turnover(W, problem.w_prev)

    if problem.objective == "mean_cvar":
        base = (1.0 - problem.gamma) * r.mean(axis=1) - problem.gamma * cvar(r, problem.tail_size)
    elif problem.objective == "mean_variance":
        base = r.mean(axis=1) - problem.gamma * r.var(axis=1, ddof=1)
    else:  # sharpe
        sd = r.std(axis=1, ddof=1)
        base = np.where(sd > 0, (r.mean(axis=1) - problem.rf) / np.maximum(sd, 1e-12), -np.inf)

    return base - cost


def score_encoded(X: np.ndarray, problem: Problem) -> np.ndarray:
    """Decode then score. This is what the search algorithms call."""
    return score(decode_population(X, problem), problem)


def report(w: np.ndarray, problem: Problem) -> dict:
    """Everything about one portfolio, for logging and for the results tables."""
    w = np.asarray(w, dtype=float)
    r = (problem.scenarios @ w).ravel()
    tail = problem.tail_size
    worst = np.sort(r)[:tail]
    active = np.flatnonzero(w > 1e-9)
    return {
        "objective": float(score(w, problem)[0]),
        "expected_return": float(r.mean()),
        "median_return": float(np.median(r)),
        "volatility": float(r.std(ddof=1)),
        "var": float(-np.quantile(r, 1.0 - problem.cvar_alpha)),
        "cvar": float(-worst.mean()),
        "downside_prob": float((r < 0).mean()),
        "turnover": float(np.abs(w - problem.w_prev).sum()),
        "cost": float(problem.cost_rate * np.abs(w - problem.w_prev).sum()),
        "n_positions": int(active.size),
        "max_weight": float(w.max()) if w.size else 0.0,
        "herfindahl": float(np.sum(w**2)),
    }


def is_feasible(w: np.ndarray, problem: Problem, tol: float = 1e-6) -> bool:
    w = np.asarray(w, dtype=float)
    active = w[w > tol]
    return bool(
        abs(w.sum() - 1.0) < 1e-6
        and (w >= -tol).all()
        and (w <= problem.w_max + tol).all()
        and (active >= problem.w_min - tol).all()
        and len(active) <= problem.max_positions
    )
