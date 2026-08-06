"""On-disk store for simulated scenario sets.

Marginals only, before coupling: reordering costs a few argsorts, so applying it at
load time keeps the coupling method a free experimental knob.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .base import ScenarioForecaster, ScenarioSet


@dataclass
class ScenarioStore:
    root: Path
    forecaster: str
    seed: int = 0

    @property
    def directory(self) -> Path:
        return self.root / self.forecaster / f"seed{self.seed:02d}"

    def path(self, date) -> Path:
        return self.directory / f"{pd.Timestamp(date).date()}.npz"

    def exists(self, date) -> bool:
        return self.path(date).exists()

    def save(self, scenarios: ScenarioSet) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.path(scenarios.origin)
        scenarios.to_npz(target)
        return target

    def load(self, date) -> ScenarioSet:
        target = self.path(date)
        if not target.exists():
            raise FileNotFoundError(
                f"no scenarios for {self.forecaster} on {pd.Timestamp(date).date()} "
                f"(expected {target}); run scripts/run_forecasts.py first"
            )
        return ScenarioSet.from_npz(target)

    def dates(self) -> list[pd.Timestamp]:
        if not self.directory.exists():
            return []
        return sorted(pd.Timestamp(p.stem) for p in self.directory.glob("*.npz"))


class Replay(ScenarioForecaster):
    """A forecaster that serves scenario sets computed by an earlier job."""

    def __init__(self, store: ScenarioStore) -> None:
        super().__init__(store=str(store.directory))
        self.store = store
        self.name = store.forecaster

    def simulate_one(self, history, horizon, n_paths, rng):     # pragma: no cover
        raise NotImplementedError("Replay serves whole scenario sets")

    def simulate(self, history, horizon, n_paths, rng, origin=None, progress=None):
        date = origin if origin is not None else history.index[-1]
        scenarios = self.store.load(date)
        if scenarios.horizon != horizon:
            raise ValueError(
                f"stored scenarios for {pd.Timestamp(date).date()} have horizon "
                f"{scenarios.horizon}, the backtest asked for {horizon}"
            )
        if scenarios.n_paths < n_paths:
            raise ValueError(
                f"stored scenarios have {scenarios.n_paths} paths, {n_paths} were requested"
            )
        if scenarios.n_paths > n_paths:
            return ScenarioSet(
                paths=scenarios.paths[:n_paths],
                assets=scenarios.assets,
                origin=scenarios.origin,
                forecaster=scenarios.forecaster,
                meta=scenarios.meta,
            )
        return scenarios


def realised(prices: pd.DataFrame, origin, assets: list[str], horizon: int) -> np.ndarray:
    """Observed cumulative returns over the horizon, aligned with a scenario set."""
    idx = prices.index
    pos = idx.get_indexer([pd.Timestamp(origin)], method="pad")[0]
    window = prices.iloc[pos : pos + horizon + 1][assets].ffill()
    base = window.iloc[0].to_numpy(dtype=float)
    future = window.iloc[1:].to_numpy(dtype=float)
    out = np.full((horizon, len(assets)), np.nan)
    out[: len(future)] = future / base - 1.0
    return out
