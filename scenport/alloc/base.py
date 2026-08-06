"""Allocator interface and registry."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .objective import Problem, is_feasible, report


@dataclass
class Allocation:
    """A portfolio plus enough context to audit how it was reached."""

    weights: np.ndarray
    allocator: str
    seconds: float
    history: list[float] = field(default_factory=list)   # incumbent objective per iteration
    diagnostics: dict = field(default_factory=dict)

    def as_series(self, assets: list[str]):
        import pandas as pd

        return pd.Series(self.weights, index=assets, name=self.allocator)


class Allocator(ABC):
    name: str = "base"
    stochastic: bool = False        # True => results depend on the seed, run it many times

    def __init__(self, **params) -> None:
        self.params = params

    @abstractmethod
    def solve(self, problem: Problem, rng: np.random.Generator) -> Allocation:
        ...

    def __call__(self, problem: Problem, rng: np.random.Generator | None = None) -> Allocation:
        rng = rng if rng is not None else np.random.default_rng(0)
        started = time.perf_counter()
        result = self.solve(problem, rng)
        result.seconds = time.perf_counter() - started
        result.diagnostics.setdefault("feasible", is_feasible(result.weights, problem))
        result.diagnostics.update(report(result.weights, problem))
        return result


_REGISTRY: dict[str, type[Allocator]] = {}


def register(cls: type[Allocator]) -> type[Allocator]:
    if cls.name in _REGISTRY:
        raise ValueError(f"allocator name {cls.name!r} is already taken")
    _REGISTRY[cls.name] = cls
    return cls


def get(name: str, **params) -> Allocator:
    if name not in _REGISTRY:
        raise KeyError(f"unknown allocator {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**params)


def available() -> list[str]:
    return sorted(_REGISTRY)
