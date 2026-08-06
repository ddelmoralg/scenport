"""Scores for sample-based forecasts.

``rmse``/``mae``/``mase`` grade the scenario median as a point forecast; ``crps`` and
``pinball`` grade the univariate distribution; ``pit`` and ``interval_coverage`` say
whether the stated uncertainty is honest; ``energy_score`` and ``variogram_score``
grade the joint forecast, which is the only way to see whether the coupling worked.

All negatively oriented except coverage.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "rmse",
    "mae",
    "mase",
    "crps",
    "pinball",
    "pit",
    "interval_coverage",
    "energy_score",
    "variogram_score",
]


def _clean(samples: np.ndarray, obs: float) -> np.ndarray | None:
    if not np.isfinite(obs):
        return None
    good = samples[np.isfinite(samples)]
    return good if good.size else None


def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    d = np.asarray(pred, float) - np.asarray(obs, float)
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d**2))) if d.size else float("nan")


def mae(pred: np.ndarray, obs: np.ndarray) -> float:
    d = np.abs(np.asarray(pred, float) - np.asarray(obs, float))
    d = d[np.isfinite(d)]
    return float(np.mean(d)) if d.size else float("nan")


def mase(pred: np.ndarray, obs: np.ndarray, in_sample: np.ndarray) -> float:
    """Mean absolute scaled error against a one-step naive forecast.

    Scale free, so it can be averaged across assets whose prices differ by four
    orders of magnitude. Plain MAPE cannot: it explodes on the cheap tokens.
    """
    naive = np.abs(np.diff(np.asarray(in_sample, float)))
    naive = naive[np.isfinite(naive)]
    denom = np.mean(naive) if naive.size else np.nan
    if not np.isfinite(denom) or denom == 0:
        return float("nan")
    return mae(pred, obs) / float(denom)


def crps(samples: np.ndarray, obs: float, fair: bool = True) -> float:
    """Continuous ranked probability score from an ensemble.

    Energy form: ``E|X - y| - 0.5 E|X - X'|``. The second term is evaluated in
    O(S log S) from the order statistics rather than the O(S^2) pairwise sum.

    ``fair=True`` uses the 1/(S(S-1)) normalisation, which is unbiased for the
    ensemble size; ``fair=False`` gives the classical 1/S^2 estimator, which
    rewards small ensembles. Ensembles here are all the same size, so the choice
    only shifts the level, but the unbiased one is the defensible default.
    """
    x = _clean(np.asarray(samples, float).ravel(), obs)
    if x is None:
        return float("nan")
    n = x.size
    if n == 1:
        return float(abs(x[0] - obs))
    x = np.sort(x)
    first = np.mean(np.abs(x - obs))
    # sum_{i<j} (x_j - x_i) = sum_i (2i - n + 1) x_i  for 0-based i
    weights = 2.0 * np.arange(n) - n + 1.0
    pair_sum = float(np.dot(weights, x))
    spread = pair_sum / (n * (n - 1)) if fair else pair_sum / (n * n)
    return float(first - spread)


def pinball(samples: np.ndarray, obs: float, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)) -> float:
    """Mean quantile (pinball) loss over ``quantiles``."""
    x = _clean(np.asarray(samples, float).ravel(), obs)
    if x is None:
        return float("nan")
    qs = np.asarray(quantiles, float)
    preds = np.quantile(x, qs)
    diff = obs - preds
    loss = np.where(diff >= 0, qs * diff, (qs - 1.0) * diff)
    return float(np.mean(loss))


def pit(samples: np.ndarray, obs: float, rng: np.random.Generator | None = None) -> float:
    """Probability integral transform value of ``obs`` under the ensemble.

    Randomised within the tie interval so that, under a correctly specified
    forecast, the values are uniform on [0, 1] rather than lumpy on a grid of
    1/S. That matters when a histogram of these values is used as the calibration
    diagnostic.
    """
    x = _clean(np.asarray(samples, float).ravel(), obs)
    if x is None:
        return float("nan")
    below = float(np.sum(x < obs))
    equal = float(np.sum(x == obs))
    u = (rng.random() if rng is not None else 0.5) if equal else 0.0
    return float((below + u * equal) / x.size)


def interval_coverage(samples: np.ndarray, obs: float, level: float = 0.80) -> float:
    """1.0 if ``obs`` falls inside the central interval, else 0.0."""
    x = _clean(np.asarray(samples, float).ravel(), obs)
    if x is None:
        return float("nan")
    lo, hi = np.quantile(x, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo <= obs <= hi)


def energy_score(samples: np.ndarray, obs: np.ndarray, beta: float = 1.0) -> float:
    """Multivariate proper score. ``samples`` is (S, d), ``obs`` is (d,).

    Strictly proper for the joint distribution, so unlike a per-asset average of
    CRPS it penalises a scenario set that gets every marginal right and the
    dependence wrong. This is the score that judges the coupling step.
    """
    x = np.asarray(samples, float)
    y = np.asarray(obs, float)
    keep = np.isfinite(y)
    if keep.sum() == 0:
        return float("nan")
    x, y = x[:, keep], y[keep]
    x = x[np.isfinite(x).all(axis=1)]
    if x.shape[0] < 2:
        return float("nan")

    first = np.mean(np.linalg.norm(x - y, axis=1) ** beta)
    # Pairwise distances without materialising the full (S, S) matrix twice.
    sq = np.sum(x**2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (x @ x.T), 0.0)
    n = x.shape[0]
    spread = np.sum(np.sqrt(d2) ** beta) / (n * (n - 1))
    return float(first - 0.5 * spread)


def variogram_score(samples: np.ndarray, obs: np.ndarray, p: float = 0.5) -> float:
    """Score built on pairwise differences, ``samples`` (S, d), ``obs`` (d,).

    Insensitive to a common bias in the marginals and sensitive to the correlation
    structure, which makes it the sharper of the two multivariate diagnostics for
    the question "is the dependence right".
    """
    x = np.asarray(samples, float)
    y = np.asarray(obs, float)
    keep = np.isfinite(y)
    if keep.sum() < 2:
        return float("nan")
    x, y = x[:, keep], y[keep]
    x = x[np.isfinite(x).all(axis=1)]
    if x.shape[0] < 2:
        return float("nan")

    obs_vg = np.abs(y[:, None] - y[None, :]) ** p
    ens_vg = np.mean(np.abs(x[:, :, None] - x[:, None, :]) ** p, axis=0)
    iu = np.triu_indices(len(y), k=1)
    return float(np.sum((obs_vg[iu] - ens_vg[iu]) ** 2))
