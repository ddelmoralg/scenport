"""ARIMA, exponential smoothing and AR-GARCH, simulated forward from the fitted state.

Parameter uncertainty is not propagated: the paths condition on the fitted
coefficients. Every model here gets the same treatment, and the calibration
diagnostics will show which ones suffer for it.
"""

from __future__ import annotations

import inspect
import logging
import warnings

import numpy as np

from .base import ScenarioForecaster, register

log = logging.getLogger(__name__)


def _as_paths(raw, horizon: int, n_paths: int) -> np.ndarray:
    """Coerce a statsmodels simulate() result into (n_paths, horizon).

    The shape depends on the statsmodels version and on whether ``repetitions``
    was passed, so the axes are identified by length rather than by position.
    """
    arr = np.asarray(raw, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape == (horizon, n_paths):
        return arr.T
    if arr.shape == (n_paths, horizon):
        return arr
    raise ValueError(f"cannot interpret simulate() output of shape {np.asarray(raw).shape}")


def _seed_kwargs(simulate, rng: np.random.Generator) -> dict:
    """However this statsmodels wants its simulate() draw seeded.

    0.14 took ``random_state`` and a legacy RandomState; 0.15 replaced it with
    ``rng`` and a Generator. Passing the wrong one raises, and passing neither
    would fall back to the global state, which looks fine until two array tasks
    started in the same second produce the same scenarios.
    """
    params = inspect.signature(simulate).parameters
    if "rng" in params:
        return {"rng": rng}
    if "random_state" in params:
        return {"random_state": np.random.RandomState(rng.integers(0, 2**31 - 1))}
    raise TypeError(
        "statsmodels simulate() takes neither 'rng' nor 'random_state'; "
        "seeding it is required, so check what this version expects"
    )


def _log_returns(history) -> np.ndarray:
    logret = np.diff(np.log(history.to_numpy(dtype=float)))
    return logret[np.isfinite(logret)]


@register
class Arima(ScenarioForecaster):
    """ARMA on log returns, i.e. ARIMA(p, 1, q) on log prices.

    Fitting the differenced series directly and integrating afterwards avoids the
    ambiguity of what ``simulate`` does with an integrated state space across
    statsmodels versions, and it is how this model is normally specified in
    finance anyway.
    """

    name = "arima"
    tunable = True

    def _select(self, logret: np.ndarray):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        orders = self.params.get("orders", [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 2)])
        best, best_aic = None, np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p, q in orders:
                try:
                    res = SARIMAX(logret, order=(p, 0, q), trend="c").fit(disp=False)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if np.isfinite(res.aic) and res.aic < best_aic:
                    best, best_aic = res, res.aic
        return best

    def simulate_one(self, history, horizon, n_paths, rng):
        logret = _log_returns(history)
        fitted = self._select(logret)
        if fitted is None:
            log.warning("ARIMA did not converge for a series of length %d", len(logret))
            return np.zeros((n_paths, horizon))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = fitted.simulate(
                nsimulations=horizon,
                anchor="end",
                repetitions=n_paths,
                **_seed_kwargs(fitted.simulate, rng),
            )
        sim = _as_paths(raw, horizon, n_paths)
        return np.exp(np.cumsum(sim, axis=1)) - 1.0


@register
class Ets(ScenarioForecaster):
    """Damped-trend exponential smoothing on log prices (ETS AAdN)."""

    name = "ets"

    def simulate_one(self, history, horizon, n_paths, rng):
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        logp = np.log(history.to_numpy(dtype=float))
        logp = logp[np.isfinite(logp)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fitted = ETSModel(
                    logp, error="add", trend="add", damped_trend=True, seasonal=None
                ).fit(disp=False)
            except (ValueError, np.linalg.LinAlgError):
                log.warning("ETS did not converge for a series of length %d", len(logp))
                return np.zeros((n_paths, horizon))
            raw = fitted.simulate(
                nsimulations=horizon,
                anchor="end",
                repetitions=n_paths,
                **_seed_kwargs(fitted.simulate, rng),
            )
        sim = _as_paths(raw, horizon, n_paths)
        return np.exp(sim - logp[-1]) - 1.0


@register
class ArGarch(ScenarioForecaster):
    """AR(1)-GARCH(1,1) with skewed-t innovations.

    The only forecaster here whose conditional variance reacts to the current
    volatility regime, which should show up in the tail calibration.
    """

    name = "garch"

    #: Distribution name -> arch class. Instantiated with our own seed rather than
    #: named by string, because ``forecast(random_state=...)`` is ignored for the
    #: simulation method: the draws come from the distribution object's own
    #: generator, so that is the one that has to be seeded.
    DISTRIBUTIONS = {"skewt": "SkewStudent", "t": "StudentsT", "normal": "Normal",
                     "ged": "GeneralizedError"}

    def simulate_one(self, history, horizon, n_paths, rng):
        import arch.univariate as au
        from arch.univariate.base import DataScaleWarning

        logret = _log_returns(history)
        scaled = 100.0 * logret            # arch is happier away from machine epsilon
        name = self.params.get("dist", "skewt")
        distribution = getattr(au, self.DISTRIBUTIONS[name])(
            seed=np.random.default_rng(rng.integers(0, 2**63 - 1))
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataScaleWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                # The class API rather than arch_model(), which only takes a
                # distribution by name and so gives no way to seed it.
                model = au.ARX(
                    scaled,
                    lags=self.params.get("lags", 1),
                    volatility=au.GARCH(p=1, q=1),
                    distribution=distribution,
                    rescale=False,
                )
                fitted = model.fit(disp="off", show_warning=False)
                forecast = fitted.forecast(
                    horizon=horizon,
                    method="simulation",
                    simulations=n_paths,
                    reindex=False,
                )
            except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
                log.warning("GARCH failed (%s); falling back to a zero path", exc)
                return np.zeros((n_paths, horizon))

        sim = np.asarray(forecast.simulations.values)[0] / 100.0   # (n_paths, horizon)
        if sim.shape != (n_paths, horizon):
            sim = _as_paths(sim, horizon, n_paths)
        return np.exp(np.cumsum(sim, axis=1)) - 1.0
