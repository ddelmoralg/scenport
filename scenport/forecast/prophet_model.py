"""Prophet as a scenario forecaster.

Fitted on log prices with a linear trend (no logistic cap), no daily seasonality,
and hyperparameters chosen on rolling-origin inner folds. ``prophet`` selects on
CRPS, ``prophet_rmse`` on the error of the median; the gap between them is one of
the results.
"""

from __future__ import annotations

import logging
import os
import warnings

import numpy as np
import pandas as pd

from ..scoring import crps, rmse
from .base import ScenarioForecaster, register

log = logging.getLogger(__name__)

# changepoint_prior_scale / weekly seasonality / seasonality_prior_scale
GRID = [
    {"changepoint_prior_scale": 0.01, "weekly_seasonality": False},
    {"changepoint_prior_scale": 0.05, "weekly_seasonality": False},
    {"changepoint_prior_scale": 0.25, "weekly_seasonality": False},
    {"changepoint_prior_scale": 0.50, "weekly_seasonality": False},
    {"changepoint_prior_scale": 0.01, "weekly_seasonality": True, "seasonality_prior_scale": 0.1},
    {"changepoint_prior_scale": 0.05, "weekly_seasonality": True, "seasonality_prior_scale": 0.1},
    {"changepoint_prior_scale": 0.05, "weekly_seasonality": True, "seasonality_prior_scale": 10.0},
    {"changepoint_prior_scale": 0.25, "weekly_seasonality": True, "seasonality_prior_scale": 10.0},
]

_QUIETED = False


def _quiet_stan() -> None:
    """Silence cmdstanpy, which is extremely chatty inside a 30k-fit sweep."""
    global _QUIETED
    if _QUIETED:
        return
    for name in ("cmdstanpy", "prophet", "prophet.models"):
        logger = logging.getLogger(name)
        logger.handlers = [logging.NullHandler()]
        logger.propagate = False
        logger.setLevel(logging.CRITICAL)
    os.environ.setdefault("CMDSTANPY_LOG_LEVEL", "CRITICAL")
    _QUIETED = True


def _fit_and_sample(
    series: pd.Series, horizon: int, n_paths: int, params: dict, seed: int
) -> np.ndarray:
    """(n_paths, horizon) cumulative simple returns from a single Prophet fit."""
    from prophet import Prophet

    _quiet_stan()
    frame = pd.DataFrame({"ds": pd.to_datetime(series.index), "y": np.log(series.to_numpy(float))})

    kwargs = {
        "growth": "linear",
        "daily_seasonality": False,          # daily data: nothing to estimate
        "yearly_seasonality": False,         # two years of history cannot identify one
        "weekly_seasonality": False,
        "seasonality_mode": "additive",      # already multiplicative through the log
        "uncertainty_samples": int(n_paths),
        "interval_width": 0.80,
    }
    kwargs.update(params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Prophet(**kwargs)
        np.random.seed(seed)                 # Prophet samples through the global RNG
        model.fit(frame)
        future = model.make_future_dataframe(periods=horizon, freq="D", include_history=False)
        samples = model.predictive_samples(future)["yhat"]      # (horizon, n_paths)

    log_paths = np.asarray(samples, dtype=float).T             # (n_paths, horizon)
    if log_paths.shape[0] != n_paths:                          # Prophet can return fewer
        idx = np.arange(n_paths) % log_paths.shape[0]
        log_paths = log_paths[idx]
    return np.exp(log_paths - np.log(float(series.iloc[-1]))) - 1.0


def _inner_folds(series: pd.Series, horizon: int, n_folds: int, min_train: int):
    """Rolling-origin folds taken from the end of the training window."""
    n = len(series)
    for f in range(n_folds):
        cut = n - (f + 1) * horizon
        if cut < min_train:
            return
        train = series.iloc[:cut]
        truth = series.iloc[cut : cut + horizon]
        if len(truth) < horizon:
            continue
        actual = truth.to_numpy(float) / float(train.iloc[-1]) - 1.0
        yield train, actual


@register
class ProphetScenarios(ScenarioForecaster):
    """Prophet with hyperparameters selected on inner folds by CRPS."""

    name = "prophet"
    tunable = True
    selection = "crps"

    def _score(self, paths: np.ndarray, actual: np.ndarray) -> float:
        if self.selection == "rmse":
            return rmse(np.median(paths, axis=0), actual)
        return float(np.nanmean([crps(paths[:, h], actual[h]) for h in range(len(actual))]))

    def _select(self, series: pd.Series, horizon: int, rng: np.random.Generator) -> dict:
        grid = self.params.get("grid", GRID)
        if len(grid) == 1:
            return dict(grid[0])

        n_folds = int(self.params.get("n_folds", 3))
        n_paths = int(self.params.get("select_paths", 200))
        min_train = int(self.params.get("min_train", 365))
        folds = list(_inner_folds(series, horizon, n_folds, min_train))
        if not folds:
            log.warning("series of length %d is too short to tune; using the grid median",
                        len(series))
            return dict(grid[len(grid) // 2])

        seed = int(rng.integers(0, 2**31 - 1))
        scores = []
        for params in grid:
            fold_scores = []
            for train, actual in folds:
                try:
                    paths = _fit_and_sample(train, horizon, n_paths, params, seed)
                except Exception as exc:                     # noqa: BLE001 - Stan raises broadly
                    log.warning("prophet fit failed for %s: %s", params, exc)
                    fold_scores.append(np.inf)
                    continue
                fold_scores.append(self._score(paths, actual))
            scores.append(np.nanmean(fold_scores))

        best = int(np.nanargmin(scores))
        return dict(grid[best])

    def simulate_one(self, history, horizon, n_paths, rng):
        params = self._select(history, horizon, rng)
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            return _fit_and_sample(history, horizon, n_paths, params, seed)
        except Exception as exc:                             # noqa: BLE001
            log.error("prophet final fit failed (%s); returning a zero path", exc)
            return np.zeros((n_paths, horizon))


@register
class ProphetRmse(ProphetScenarios):
    """Same model, hyperparameters chosen by the error of the point forecast."""

    name = "prophet_rmse"
    selection = "rmse"
