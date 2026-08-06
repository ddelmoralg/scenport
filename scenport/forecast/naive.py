"""Reference forecasters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ScenarioForecaster, block_bootstrap, register


@register
class HistoricalSimulation(ScenarioForecaster):
    """Block bootstrap of realised daily log returns.

    No view on the future beyond "next month looks like a reshuffled recent
    month". Inherits the drift and the volatility clustering of the training
    window and nothing else. Hard to beat.
    """

    name = "rw"

    def simulate_one(self, history, horizon, n_paths, rng):
        logret = np.diff(np.log(history.to_numpy(dtype=float)))
        logret = logret[np.isfinite(logret)]
        sim = block_bootstrap(logret, horizon, n_paths, rng, block=self.params.get("block", 5))
        return np.exp(np.cumsum(sim, axis=1)) - 1.0


@register
class DriftlessRandomWalk(HistoricalSimulation):
    """Same, with the training-window mean removed.

    A martingale in log space, so a portfolio built on it is driven purely by the
    dependence structure. Separates "the optimiser is doing the work" from "the
    forecaster is doing the work".
    """

    name = "rw_nodrift"

    def simulate_one(self, history, horizon, n_paths, rng):
        logret = np.diff(np.log(history.to_numpy(dtype=float)))
        logret = logret[np.isfinite(logret)]
        centred = logret - logret.mean()
        sim = block_bootstrap(centred, horizon, n_paths, rng, block=self.params.get("block", 5))
        return np.exp(np.cumsum(sim, axis=1)) - 1.0


@register
class LastValue(ScenarioForecaster):
    """Degenerate point forecast: no dispersion at all.

    A diagnostic, not a competitor. Any objective that reads a quantile of the
    scenario set sees zero risk here and goes to a corner.
    """

    name = "flat"

    def simulate_one(self, history, horizon, n_paths, rng):
        return np.zeros((n_paths, horizon))


def realised_path(prices: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> np.ndarray:
    """What actually happened: (horizon, n) cumulative returns after ``origin``.

    Same convention as the scenario sets, so the two are directly comparable.
    """
    idx = prices.index
    start = idx.get_indexer([origin], method="pad")[0]
    window = prices.iloc[start : start + horizon + 1]
    base = window.iloc[0].to_numpy(dtype=float)
    future = window.iloc[1:].to_numpy(dtype=float)
    if len(future) < horizon:                            # truncated at the panel edge
        pad = np.full((horizon - len(future), future.shape[1]), np.nan)
        future = np.vstack([future, pad])
    return future / base - 1.0
