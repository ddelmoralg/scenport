"""The three outcomes the sensitivity table has to be able to tell apart.

Every paper in this area measures realised cost and nothing else, so the case
where the decision does not move but performance still varies has no name and no
table. It is also the most interesting of the three, and it would be a poor way to
find out that the pipeline cannot show it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scenport.data.panel import Panel

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
ASSETS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


@pytest.fixture(scope="module")
def aggregate():
    spec = importlib.util.spec_from_file_location(
        "aggregate_sensitivity", SCRIPTS / "aggregate.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cfg():
    from scenport import config as cfgmod

    return cfgmod.load(str(Path(__file__).resolve().parents[1] / "configs" / "base.yaml"), [])


def _runs(tmp_path, gap: float, jitter: float = 0.01, seeds=("00", "01", "02")):
    """Two generators `gap` apart, each wobbling by `jitter` between seeds."""
    runs = tmp_path / "runs"
    dates = pd.bdate_range("2023-01-31", periods=24, freq="ME")
    rng = np.random.default_rng(6)
    base = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
    other = np.roll(base, 1) * gap + base * (1 - gap)

    for name, centre in (("garch", base), ("prophet", other)):
        for seed in seeds:
            target = runs / "backtests" / name / f"ga_seed{seed}"
            target.mkdir(parents=True, exist_ok=True)
            rows = []
            for _ in dates:
                w = np.clip(centre + rng.normal(0, jitter, centre.size), 0, None)
                rows.append(w / w.sum())
            pd.DataFrame(rows, index=dates, columns=ASSETS).to_parquet(
                target / "weights.parquet"
            )
    return runs


def _summaries(sharpe_alpha: float, sharpe_beta: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"forecaster": "garch", "allocator": "ga", "sharpe": sharpe_alpha, "ann_return": 0.2},
            {"forecaster": "prophet", "allocator": "ga", "sharpe": sharpe_beta, "ann_return": 0.3},
        ]
    )


@pytest.fixture
def panel_dir(tmp_path):
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2021-01-01", periods=700)
    price = pd.DataFrame(
        {a: 100 * np.cumprod(1 + rng.normal(0, 0.02, len(dates))) for a in ASSETS}, index=dates
    )
    mcap = pd.DataFrame(1e9, index=dates, columns=ASSETS)
    directory = tmp_path / "panel"
    Panel(price, mcap).write(directory)
    return directory


def test_decision_moves_and_performance_follows(aggregate, tmp_path, panel_dir):
    runs = _runs(tmp_path, gap=0.9)
    table, headline = aggregate.table_sensitivity(
        runs, panel_dir, _cfg(), _summaries(0.4, 1.3), "ga"
    )
    assert not table.empty
    assert headline["mean_displacement_over_noise"] > 1.5
    assert headline["reading"] == "decision moves and performance follows"


def test_the_case_nobody_measures(aggregate, tmp_path, panel_dir):
    """Decision at the noise floor, performance still varying.

    This is the one the literature cannot see, because it only looks at cost.
    """
    runs = _runs(tmp_path, gap=0.0, jitter=0.02)
    table, headline = aggregate.table_sensitivity(
        runs, panel_dir, _cfg(), _summaries(0.4, 1.3), "ga"
    )
    assert not table.empty
    assert headline["mean_displacement_over_noise"] < 1.5
    assert "realised path" in headline["reading"]


def test_insensitive_to_the_generator(aggregate, tmp_path, panel_dir):
    runs = _runs(tmp_path, gap=0.0, jitter=0.02)
    table, headline = aggregate.table_sensitivity(
        runs, panel_dir, _cfg(), _summaries(0.80, 0.81), "ga"
    )
    assert headline["reading"] == "insensitive to the generator within this class"


def test_decision_moves_but_performance_does_not(aggregate, tmp_path, panel_dir):
    runs = _runs(tmp_path, gap=0.9)
    table, headline = aggregate.table_sensitivity(
        runs, panel_dir, _cfg(), _summaries(0.80, 0.81), "ga"
    )
    assert headline["reading"] == "decision moves, performance does not separate"


def test_the_table_carries_what_the_reading_rests_on(aggregate, tmp_path, panel_dir):
    runs = _runs(tmp_path, gap=0.7)
    table, headline = aggregate.table_sensitivity(
        runs, panel_dir, _cfg(), _summaries(0.4, 1.3), "ga"
    )
    for column in ("forecaster", "displacement", "displacement_over_noise", "overlap", "sharpe"):
        assert column in table, column
    assert set(table["forecaster"]) == {"garch", "prophet"}
    assert headline["n_generators"] == 2
    assert np.isfinite(headline["noise_floor_l1"])


def test_no_weights_is_not_fatal(aggregate, tmp_path, panel_dir):
    table, headline = aggregate.table_sensitivity(
        tmp_path / "nothing", panel_dir, _cfg(), _summaries(0.4, 1.3), "ga"
    )
    assert table.empty and headline == {}
