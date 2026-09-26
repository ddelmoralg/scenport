"""The neutral yardstick two decisions are compared on.

Not one of the eleven forecasters and deliberately kept out of the registry, so it
can never be selected as a competitor by accident.

Comparing the portfolios two generators chose needs a common distribution to price
them against. Three properties decide what it can be.

It has to be available at the rebalancing date. If the yardstick saw anything after
that date the objective difference stops being a statement about the decision and
becomes realised performance wearing a different name, which is the one thing this
measurement exists to avoid.

It cannot depend on which generators are in the study. A mixture of the eleven would
be neutral, but every published number would move if a twelfth were added and nobody
outside could reproduce it against their own method.

And it should impose no dependence model, because the dependence model is one of the
things under study. So whole cross-sections are resampled in blocks rather than each
asset separately: the empirical cross-asset dependence comes along untouched and no
copula is fitted at any point.

What is left is a stationary block bootstrap of realised daily returns over the
window ending the day before the date. Block length and path count are fixed in the
config and are not tuned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Deliberately unlike the historical-simulation forecaster's default of 5. The
#: reference must not be the same object as one of the things it is judging.
REFERENCE_BLOCK = 21
REFERENCE_PATHS = 20_000


def _block_indices(
    n: int, horizon: int, n_paths: int, rng: np.random.Generator, block: int
) -> np.ndarray:
    """Stationary block bootstrap row indices, shape (n_paths, horizon).

    Geometric block lengths with mean ``block``, as in Politis and Romano. Written
    against whole rows so every asset in a path comes from the same historical day.
    """
    p = 1.0 / max(block, 1)
    starts = rng.integers(0, n, size=(n_paths, horizon))
    fresh = rng.random((n_paths, horizon)) < p
    fresh[:, 0] = True

    position = np.arange(horizon)[None, :].repeat(n_paths, axis=0)
    block_start = np.maximum.accumulate(np.where(fresh, position, -1), axis=1)
    offset = position - block_start
    base = np.take_along_axis(starts, block_start, axis=1)
    return (base + offset) % n


def reference_scenarios(
    returns: pd.DataFrame,
    date: pd.Timestamp,
    assets: list[str],
    horizon: int,
    *,
    window_days: int = 730,
    n_paths: int = REFERENCE_PATHS,
    block: int = REFERENCE_BLOCK,
    seed: int = 0,
) -> np.ndarray:
    """Reference scenario set for one date, as (n_paths, len(assets)).

    ``returns`` is a wide frame of daily simple returns. Only rows strictly before
    ``date`` are used. The result is cumulative simple return over the horizon, the
    same shape the allocators already consume.
    """
    date = pd.Timestamp(date)
    window = returns.loc[date - pd.Timedelta(days=window_days) : date - pd.Timedelta(days=1)]
    window = window.reindex(columns=assets).dropna(how="all")
    if len(window) < max(horizon, 60):
        return np.empty((0, len(assets)))

    values = window.fillna(0.0).to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = _block_indices(len(values), horizon, n_paths, rng, block)

    # (n_paths, horizon, n_assets): one historical day per step, all assets together.
    drawn = values[idx]
    cumulative = np.cumprod(1.0 + drawn, axis=1)[:, -1, :] - 1.0
    return np.ascontiguousarray(cumulative)


def reference_seed(date: pd.Timestamp, tag: str = "reference") -> int:
    """Seed stream for the reference sets, separate from every decision generator.

    Sharing a stream with a forecaster would correlate the yardstick with one of the
    things it measures.
    """
    from ..config import stable_seed

    return stable_seed(tag, pd.Timestamp(date).strftime("%Y-%m-%d"))


__all__ = [
    "REFERENCE_BLOCK",
    "REFERENCE_PATHS",
    "reference_scenarios",
    "reference_seed",
]
