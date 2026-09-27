"""The scenario interface.

A forecaster produces simulated forward paths, not a point. Two conventions:

* paths are cumulative simple returns from the origin price, so
  ``paths[s, h, i] == P_i(t0 + h + 1) / P_i(t0) - 1``;
* each asset is simulated on its own. Cross-asset dependence is applied afterwards
  by ``coupling.apply``, which is what lets the forecaster and the dependence model
  be varied independently.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Ceiling on a simulated cumulative return, and a floor of -1 because a long-only
#: position cannot lose more than it cost.
#:
#: Every forecaster here simulates daily returns and compounds them over the
#: horizon, and a fat-tailed model can put out a handful of draws whose product is
#: astronomical. Left alone they produce 30-day returns of 1e10 and worse, which
#: then dominate the mean in a mean-CVaR objective and decide the portfolio on
#: their own.
#:
#: The bound is twice the largest 30-day return anywhere in the panel over ten
#: years and seventy-six assets, which is 45x, held by XVG. The 99.99th percentile
#: is 22x. Anything above this has no precedent in the data the models were fitted
#: to, so clipping it removes arithmetic rather than removing tail risk, and the
#: fraction clipped is recorded per scenario set because a generator that hits the
#: bound often is telling you something about itself.
MAX_CUMULATIVE_RETURN = 90.0


@dataclass
class ScenarioSet:
    """``n_paths`` simulated forward paths for ``len(assets)`` assets."""

    paths: np.ndarray                       # (S, H, n) cumulative simple returns
    assets: list[str]
    origin: pd.Timestamp
    forecaster: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.paths.ndim != 3:
            raise ValueError(f"paths must be (S, H, n), got shape {self.paths.shape}")
        if self.paths.shape[2] != len(self.assets):
            raise ValueError(
                f"paths has {self.paths.shape[2]} assets but {len(self.assets)} names were given"
            )

        # The bound lives here, not in simulate(), because simulate() can be
        # overridden and was: the neural forecasters batch every asset through one
        # network and replace it wholesale, so they sailed past a ceiling applied
        # there and produced 30-day returns of 3.98e15. Every scenario set is
        # constructed through this, however it was built or loaded, so this is the
        # one place that cannot be stepped around.
        over = int((self.paths > MAX_CUMULATIVE_RETURN).sum() + (self.paths < -1.0).sum())
        if over:
            self.paths = np.clip(self.paths, -1.0, MAX_CUMULATIVE_RETURN)
            self.meta.setdefault("clipped_points", 0)
            self.meta["clipped_points"] += over
            self.meta["clipped_share"] = self.meta["clipped_points"] / self.paths.size
            self.meta.setdefault("max_cumulative_return", MAX_CUMULATIVE_RETURN)

    @property
    def n_paths(self) -> int:
        return self.paths.shape[0]

    @property
    def horizon(self) -> int:
        return self.paths.shape[1]

    def terminal(self) -> np.ndarray:
        """(S, n) return over the full holding period."""
        return self.paths[:, -1, :]

    def expected_return(self) -> np.ndarray:
        return self.terminal().mean(axis=0)

    def median_return(self) -> np.ndarray:
        return np.median(self.terminal(), axis=0)

    def covariance(self) -> np.ndarray:
        """(n, n) covariance of horizon returns.

        Of the *horizon* return, not of daily returns -- everything in this package
        stays on the holding-period scale.
        """
        return np.cov(self.terminal(), rowvar=False, ddof=1)

    def quantiles(self, qs: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
        term = self.terminal()
        return pd.DataFrame(
            np.quantile(term, qs, axis=0), index=[f"q{int(q * 100):02d}" for q in qs],
            columns=self.assets,
        )

    def reorder(self, assets: list[str]) -> ScenarioSet:
        """Same scenarios with the asset axis permuted to match ``assets``."""
        idx = [self.assets.index(a) for a in assets]
        return ScenarioSet(
            paths=self.paths[:, :, idx],
            assets=list(assets),
            origin=self.origin,
            forecaster=self.forecaster,
            meta=dict(self.meta),
        )

    def to_npz(self, path) -> None:
        # meta travels as JSON. It carries how much of this set hit the ceiling,
        # which is a property of the generator worth reporting and was being
        # dropped here, so nothing downstream could see it.
        np.savez_compressed(
            path,
            paths=self.paths.astype(np.float32),
            assets=np.array(self.assets, dtype=object),
            origin=str(self.origin),
            forecaster=self.forecaster,
            meta=json.dumps(self.meta, default=str),
        )

    @classmethod
    def from_npz(cls, path) -> ScenarioSet:
        with np.load(path, allow_pickle=True) as z:
            try:
                meta = json.loads(str(z["meta"])) if "meta" in z else {}
            except (ValueError, KeyError):
                meta = {}
            return cls(
                paths=z["paths"].astype(np.float64),
                assets=[str(a) for a in z["assets"]],
                origin=pd.Timestamp(str(z["origin"])),
                forecaster=str(z["forecaster"]),
                meta=meta,
            )


class ScenarioForecaster(ABC):
    """Base class. Subclasses only implement ``simulate_one``."""

    name: str = "base"
    requires_gpu: bool = False
    # Set by subclasses that pick hyperparameters on an inner validation split.
    tunable: bool = False

    def __init__(self, **params) -> None:
        self.params = params

    @abstractmethod
    def simulate_one(
        self,
        history: pd.Series,
        horizon: int,
        n_paths: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Simulate one asset. Returns (n_paths, horizon) cumulative simple returns."""

    def simulate(
        self,
        history: pd.DataFrame,
        horizon: int,
        n_paths: int,
        rng: np.random.Generator,
        origin: pd.Timestamp | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> ScenarioSet:
        """Simulate every column of ``history`` independently."""
        assets = list(history.columns)
        out = np.empty((n_paths, horizon, len(assets)), dtype=np.float64)
        for j, asset in enumerate(assets):
            if progress is not None:
                progress(asset)
            series = history[asset].dropna()
            child = np.random.default_rng(rng.integers(0, 2**63 - 1))
            out[:, :, j] = self.simulate_one(series, horizon, n_paths, child)

        # A forecaster may ask for a tighter bound than the global one. The loose
        # direction is not available: ScenarioSet enforces the global ceiling
        # whatever arrives.
        cap = min(
            float(self.params.get("max_cumulative_return", MAX_CUMULATIVE_RETURN)),
            MAX_CUMULATIVE_RETURN,
        )
        clipped = int((out > cap).sum() + (out < -1.0).sum())
        if clipped:
            out = np.clip(out, -1.0, cap)

        return ScenarioSet(
            paths=out,
            assets=assets,
            origin=pd.Timestamp(origin if origin is not None else history.index[-1]),
            forecaster=self.name,
            meta={
                "params": dict(self.params),
                "max_cumulative_return": cap,
                "clipped_points": clipped,
                "clipped_share": clipped / out.size if out.size else 0.0,
            },
        )


# ---------------------------------------------------------------------------
# registry

_REGISTRY: dict[str, type[ScenarioForecaster]] = {}


def register(cls: type[ScenarioForecaster]) -> type[ScenarioForecaster]:
    if cls.name in _REGISTRY:
        raise ValueError(f"forecaster name {cls.name!r} is already taken")
    _REGISTRY[cls.name] = cls
    return cls


def get(name: str, **params) -> ScenarioForecaster:
    if name not in _REGISTRY:
        raise KeyError(f"unknown forecaster {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**params)


def available() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# helpers shared by several forecasters


def prices_to_paths(price_paths: np.ndarray, last_price: float) -> np.ndarray:
    """Convert simulated *price* paths into cumulative returns from the origin."""
    return price_paths / last_price - 1.0


def compound(daily: np.ndarray) -> np.ndarray:
    """Cumulative simple return from simulated daily simple returns.

    ``daily`` is (S, H); the result is (S, H).
    """
    return np.cumprod(1.0 + daily, axis=1) - 1.0


def block_bootstrap(
    residuals: np.ndarray, horizon: int, n_paths: int, rng: np.random.Generator, block: int = 5
) -> np.ndarray:
    """Stationary block bootstrap of a residual vector into (n_paths, horizon).

    Geometric block lengths with mean ``block`` (Politis and Romano). Blocks keep
    the short-range dependence that iid resampling destroys, which matters here
    because crypto returns cluster in volatility.
    """
    n = len(residuals)
    if n == 0:
        return np.zeros((n_paths, horizon))
    out = np.empty((n_paths, horizon))
    p = 1.0 / max(block, 1)
    for s in range(n_paths):
        filled = 0
        idx = rng.integers(0, n)
        while filled < horizon:
            out[s, filled] = residuals[idx % n]
            filled += 1
            idx = rng.integers(0, n) if rng.random() < p else idx + 1
    return out
