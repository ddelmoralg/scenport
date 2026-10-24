"""The two diagnostics that were once computed wrong stay computed right."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import aggregate  # noqa: E402


def _prices() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=400, freq="D")
    walk = np.exp(np.cumsum(rng.normal(0.0, 0.03, (400, 2)), axis=0))
    return pd.DataFrame(walk, index=dates, columns=["cheap", "dear"])


def test_mase_scale_does_not_depend_on_the_price_level():
    price = _prices()
    price["dear"] = price["cheap"] * 1000.0          # same path, a thousand times dearer
    rows = pd.DataFrame({"rebalance": [price.index[-1]] * 2, "asset": ["cheap", "dear"]})
    scale = aggregate.return_scales(rows, price, train_days=300)
    assert scale.iloc[0] == pytest.approx(scale.iloc[1])
    assert 0.01 < scale.iloc[0] < 0.05               # a daily return, not a dollar amount


def test_mase_scale_uses_only_the_training_window_before_the_date():
    price = _prices()
    date = price.index[200]
    rows = pd.DataFrame({"rebalance": [date], "asset": ["cheap"]})
    before = aggregate.return_scales(rows, price, train_days=100).iloc[0]
    later = price.copy()
    later.loc[date:, "cheap"] *= np.linspace(1.0, 50.0, len(later.loc[date:]))
    after = aggregate.return_scales(rows, later, train_days=100).iloc[0]
    assert before == pytest.approx(after)


def test_coverage_error_penalises_wide_intervals_as_much_as_narrow_ones():
    frame = pd.DataFrame({"cover_50": [0.60, 0.40], "cover_80": [0.90, 0.70]})
    error = aggregate.coverage_error(frame)
    assert error.iloc[0] == pytest.approx(error.iloc[1]) == pytest.approx(0.10)


def test_perfect_calibration_scores_zero_and_full_coverage_does_not():
    frame = pd.DataFrame({"cover_50": [0.50, 1.00], "cover_80": [0.80, 1.00]})
    error = aggregate.coverage_error(frame)
    assert error.iloc[0] == pytest.approx(0.0)
    assert error.iloc[1] == pytest.approx((0.5 + 0.2) / 2)
