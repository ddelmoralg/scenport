"""LightGBM iterated one day at a time.

The model predicts the conditional mean; the spread comes from block-bootstrapped
residuals taken from a validation split the final model never trained on.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import ScenarioForecaster, block_bootstrap, register

log = logging.getLogger(__name__)

LAGS = (1, 2, 3, 5, 10)
WINDOWS = (5, 21, 63)
LOOKBACK = 63

FEATURE_NAMES = [f"lag_{l}" for l in LAGS] + [
    f"{stat}_{w}" for w in WINDOWS for stat in ("mean", "std", "sum")
]


def _features(recent: np.ndarray) -> np.ndarray:
    """(m, LOOKBACK) windows of log returns -> (m, n_features).

    The most recent return sits in the last column. Both the training design and
    the simulation step go through this function, which is the only way to be sure
    they agree.
    """
    cols = [recent[:, -lag] for lag in LAGS]
    for w in WINDOWS:
        block = recent[:, -w:]
        cols.append(block.mean(axis=1))
        cols.append(block.std(axis=1))
        cols.append(block.sum(axis=1))
    return np.column_stack(cols)


def _design(logret: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(logret)
    if n <= LOOKBACK + 1:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0)
    windows = np.lib.stride_tricks.sliding_window_view(logret, LOOKBACK)[:-1]
    # Both arrays are copied rather than handed over as views. LightGBM reads the
    # label through a raw pointer and segfaults on a non-owning array.
    return (
        np.ascontiguousarray(_features(windows)),
        np.ascontiguousarray(logret[LOOKBACK:]),
    )


@register
class LightGbm(ScenarioForecaster):
    """Recursive one-step LightGBM with bootstrapped residual innovations."""

    name = "lgbm"
    tunable = True

    def _train(self, logret: np.ndarray, seed: int):
        import lightgbm as lgb

        X, y = _design(logret)
        if len(y) < 120:
            return None, None

        holdout = max(int(self.params.get("holdout", 120)), 60)
        holdout = min(holdout, len(y) // 3)
        X_fit = np.ascontiguousarray(X[:-holdout])
        y_fit = np.ascontiguousarray(y[:-holdout])
        X_val = np.ascontiguousarray(X[-holdout:])
        y_val = np.ascontiguousarray(y[-holdout:])

        params = {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": self.params.get("learning_rate", 0.03),
            "num_leaves": self.params.get("num_leaves", 15),
            "min_data_in_leaf": self.params.get("min_data_in_leaf", 30),
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": seed,
            "num_threads": int(self.params.get("num_threads", 1)),
        }
        train_set = lgb.Dataset(X_fit, label=y_fit, feature_name=FEATURE_NAMES)
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=self.params.get("num_boost_round", 800),
            valid_sets=[
                lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES,
                            reference=train_set)
            ],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        residuals = y_val - booster.predict(X_val, num_iteration=booster.best_iteration)

        # Refit on everything at the iteration count the validation split picked.
        final = lgb.train(
            params,
            lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES),
            num_boost_round=max(booster.best_iteration, 20),
        )
        return final, np.asarray(residuals, float)

    def simulate_one(self, history, horizon, n_paths, rng):
        logret = np.diff(np.log(history.to_numpy(dtype=float)))
        logret = logret[np.isfinite(logret)]

        booster, residuals = self._train(logret, seed=int(rng.integers(0, 2**31 - 1)))
        if booster is None:
            log.info("series too short for lgbm (%d returns); using a block bootstrap", len(logret))
            sim = block_bootstrap(logret, horizon, n_paths, rng, block=5)
            return np.exp(np.cumsum(sim, axis=1)) - 1.0

        shocks = block_bootstrap(residuals, horizon, n_paths, rng, block=5)
        recent = np.tile(logret[-LOOKBACK:], (n_paths, 1))
        steps = np.empty((n_paths, horizon))
        for h in range(horizon):
            mean = booster.predict(_features(recent))
            steps[:, h] = mean + shocks[:, h]
            recent = np.hstack([recent[:, 1:], steps[:, h : h + 1]])
        return np.exp(np.cumsum(steps, axis=1)) - 1.0
