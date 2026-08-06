"""Rule-based allocators. No optimiser, no scenarios."""

from __future__ import annotations

import numpy as np

from .base import Allocation, Allocator, register
from .objective import Problem, project_simplex_box


def _spread(problem: Problem, raw: np.ndarray) -> np.ndarray:
    """Turn a non-negative score vector into a feasible portfolio.

    The heuristics face the same cap and the same minimum position size as the
    optimisers. Letting 1/N run unconstrained while the optimisers are boxed in
    would not be a fair comparison.
    """
    raw = np.clip(np.nan_to_num(raw, nan=0.0), 0.0, None)
    if raw.sum() <= 0:
        raw = np.ones(problem.n)
    keep = min(problem.max_positions, problem.n)
    chosen = np.argsort(-raw, kind="stable")[:keep]
    w = np.zeros(problem.n)
    w[chosen] = project_simplex_box(raw[chosen] / raw[chosen].sum(), problem.w_min, problem.w_max)
    return w


@register
class EqualWeight(Allocator):
    name = "equal"

    def solve(self, problem, rng):
        return Allocation(_spread(problem, np.ones(problem.n)), self.name, 0.0)


@register
class MarketCap(Allocator):
    """Capitalisation weighted: the asset-class beta.

    Needs ``problem.meta['mcap']``. This, not an equity index, is what a crypto
    strategy has to beat.
    """

    name = "mcap"

    def solve(self, problem, rng):
        caps = problem.meta.get("mcap")
        if caps is None:
            raise KeyError("the mcap allocator needs problem.meta['mcap']")
        return Allocation(_spread(problem, np.asarray(caps, dtype=float)), self.name, 0.0)


@register
class Random(Allocator):
    """Uniform random weights -- the "monkey" benchmark.

    Stochastic, so it gets the same treatment as the search allocators: many
    draws, and the proposal is reported as a percentile of the distribution
    rather than against a single lucky monkey.
    """

    name = "random"
    stochastic = True

    def solve(self, problem, rng):
        return Allocation(_spread(problem, rng.random(problem.n)), self.name, 0.0)


@register
class InverseVolatility(Allocator):
    """Weights proportional to 1/sigma_i, with sigma taken from the scenarios.

    Uses only the diagonal of the scenario covariance, so it is the cheapest way
    to react to a forecast at all -- useful for separating "the forecast helped"
    from "the optimiser helped".
    """

    name = "invvol"

    def solve(self, problem, rng):
        sigma = problem.scenarios.std(axis=0, ddof=1)
        return Allocation(_spread(problem, 1.0 / np.maximum(sigma, 1e-8)), self.name, 0.0)


@register
class SingleAsset(Allocator):
    """Everything in one asset, by default the largest by capitalisation.

    Buy-and-hold bitcoin is the allocation most retail investors actually run, and
    over several of the periods in this backtest it is very hard to beat.
    """

    name = "single"

    def solve(self, problem, rng):
        target = self.params.get("asset")
        if target is not None:
            idx = problem.assets.index(target)
        else:
            caps = problem.meta.get("mcap")
            idx = int(np.argmax(caps)) if caps is not None else 0
        w = np.zeros(problem.n)
        w[idx] = 1.0
        # Deliberately left unprojected: it breaks w_max on purpose, and the
        # backtest flags it as an unconstrained reference rather than a candidate.
        return Allocation(w, self.name, 0.0, diagnostics={"unconstrained": True})
