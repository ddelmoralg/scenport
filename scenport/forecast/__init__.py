"""Scenario forecasters.

Importing this package registers every CPU forecaster. The deep models live in
``scenport.forecast.deep`` and are registered lazily by ``load_deep()`` so that
CPU-only jobs never import torch.
"""

from . import classical, gbm, naive, prophet_model  # noqa: F401  (import registers)
from .base import ScenarioForecaster, ScenarioSet, available, get, register

CPU_FORECASTERS = ("rw", "rw_nodrift", "arima", "ets", "garch", "prophet", "prophet_rmse", "lgbm")
GPU_FORECASTERS = ("lstm", "deepar", "transformer")


def load_deep() -> None:
    """Register the torch-backed forecasters. Safe to call more than once."""
    if "lstm" in available():
        return
    from . import deep  # noqa: F401


def get_any(name: str, **params) -> ScenarioForecaster:
    """Look up a forecaster by name, pulling in torch only if it is a deep one."""
    if name in GPU_FORECASTERS:
        load_deep()
    return get(name, **params)


__all__ = [
    "CPU_FORECASTERS",
    "GPU_FORECASTERS",
    "ScenarioForecaster",
    "ScenarioSet",
    "available",
    "get",
    "get_any",
    "load_deep",
    "register",
]
