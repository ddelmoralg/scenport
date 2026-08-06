"""Cross-asset dependence, imposed after the fact by rank reordering.

Independent marginals would be badly wrong here -- pairwise correlations across the
large cryptoassets sit well above 0.7, so an uncoupled scenario set makes
diversification look free. Iman and Conover (1982) permute each asset's paths to
match a target rank structure while leaving every marginal exactly as it was.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .base import ScenarioSet

log = logging.getLogger(__name__)

METHODS = ("gaussian_copula", "empirical", "independent")


def horizon_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Overlapping ``horizon``-day simple returns.

    Overlapping windows are autocorrelated, which would matter for a mean test but
    not for estimating a rank correlation, and they buy roughly ``horizon`` times
    more observations than non-overlapping ones. With a two-year training window
    and a one-month horizon that is the difference between 24 and 700 rows.
    """
    return prices.pct_change(horizon).dropna(how="all")


def spearman_target(prices: pd.DataFrame, horizon: int, assets: list[str]) -> np.ndarray:
    """Spearman correlation of horizon returns, restricted to ``assets``."""
    hr = horizon_returns(prices[assets], horizon)
    corr = hr.corr(method="spearman").reindex(index=assets, columns=assets)
    # A brand-new asset can have too few overlapping windows for a pair estimate.
    return corr.fillna(0.0).to_numpy()


def spearman_to_pearson(rho_s: np.ndarray) -> np.ndarray:
    """Gaussian correlation reproducing a given Spearman correlation.

    Exact for the Gaussian copula: rho = 2 sin(pi rho_s / 6).
    """
    out = 2.0 * np.sin(np.pi * np.clip(rho_s, -1.0, 1.0) / 6.0)
    np.fill_diagonal(out, 1.0)
    return out


def nearest_psd(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Closest positive semi-definite correlation matrix by eigenvalue clipping.

    Needed because the pairwise Spearman-to-Pearson transform is applied element
    by element and can push a valid correlation matrix outside the PSD cone.
    """
    sym = 0.5 * (matrix + matrix.T)
    vals, vecs = np.linalg.eigh(sym)
    if vals.min() >= eps:
        return sym
    vals = np.clip(vals, eps, None)
    rebuilt = vecs @ np.diag(vals) @ vecs.T
    scale = np.sqrt(np.diag(rebuilt))
    rebuilt = rebuilt / np.outer(scale, scale)
    np.fill_diagonal(rebuilt, 1.0)
    return rebuilt


def _ranks(x: np.ndarray) -> np.ndarray:
    """Zero-based integer ranks along axis 0, ties broken by position."""
    return np.argsort(np.argsort(x, axis=0, kind="stable"), axis=0, kind="stable")


def _reorder_to(paths: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Permute each asset's paths so their terminal ranks follow ``template``.

    ``paths`` is (S, H, n), ``template`` is (S, n) of arbitrary continuous values.
    Marginals are untouched: the multiset of paths per asset is unchanged.
    """
    n_paths, _, n_assets = paths.shape
    terminal = paths[:, -1, :]
    out = np.empty_like(paths)
    target_ranks = _ranks(template)
    for j in range(n_assets):
        ascending = np.argsort(terminal[:, j], kind="stable")   # path idx, worst to best
        out[:, :, j] = paths[ascending[target_ranks[:, j]], :, j]
    return out


def apply(
    scen: ScenarioSet,
    prices: pd.DataFrame,
    *,
    method: str = "gaussian_copula",
    rng: np.random.Generator,
    window: int | None = 365,
) -> ScenarioSet:
    """Return a copy of ``scen`` with cross-asset dependence imposed.

    ``prices`` is the training-window price panel used to estimate the target
    dependence; only history up to the scenario origin may appear in it.
    """
    if method not in METHODS:
        raise ValueError(f"unknown coupling method {method!r}; expected one of {METHODS}")
    if method == "independent":
        return scen

    if prices.index.max() > scen.origin:
        raise ValueError("coupling was handed prices past the scenario origin")

    history = prices[scen.assets]
    if window is not None:
        history = history.iloc[-window:]

    n_paths = scen.n_paths
    if method == "gaussian_copula":
        rho = nearest_psd(spearman_to_pearson(spearman_target(history, scen.horizon, scen.assets)))
        chol = np.linalg.cholesky(rho)
        template = rng.standard_normal((n_paths, len(scen.assets))) @ chol.T
    else:  # empirical
        hr = horizon_returns(history, scen.horizon).dropna()
        if len(hr) < 30:
            log.warning(
                "%s: only %d overlapping windows for the empirical copula, "
                "falling back to the Gaussian one",
                scen.origin.date(),
                len(hr),
            )
            return apply(scen, prices, method="gaussian_copula", rng=rng, window=window)
        rows = rng.integers(0, len(hr), size=n_paths)      # joint resample keeps the copula
        template = rankdata(hr.to_numpy()[rows], axis=0).astype(float)

    coupled = _reorder_to(scen.paths, template)
    meta = dict(scen.meta)
    meta["coupling"] = method
    return ScenarioSet(
        paths=coupled,
        assets=list(scen.assets),
        origin=scen.origin,
        forecaster=scen.forecaster,
        meta=meta,
    )


def realised_spearman(scen: ScenarioSet) -> np.ndarray:
    """Rank correlation of terminal returns actually present in a scenario set.

    Used in the tests and in the diagnostics table: the reordering should recover
    the target to within Monte Carlo error.
    """
    term = scen.terminal()
    return np.corrcoef(rankdata(term, axis=0), rowvar=False)
