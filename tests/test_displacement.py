"""Decision displacement, and the seed noise floor it is quoted against.

The sensitivity question is about how far the chosen portfolio moves when the
scenario generator is swapped. That is a property of the decisions alone, with no
outcome data anywhere in it, and these tests pin that: the numbers must come out the
same whatever the market did afterwards.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
ASSETS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


@pytest.fixture(scope="module")
def aggregate():
    spec = importlib.util.spec_from_file_location(
        "aggregate_displacement", SCRIPTS / "aggregate.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(runs: Path, forecaster: str, seed: str, frame: pd.DataFrame, allocator="ga"):
    target = runs / "backtests" / forecaster / f"{allocator}_seed{seed}"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target / "weights.parquet")


def _tree(tmp_path, jitter: float, gap: float, n_dates: int = 20, seeds=("00", "01", "02")):
    """Two generators a fixed distance apart, each wobbling by `jitter` across seeds."""
    runs = tmp_path / "runs"
    dates = pd.bdate_range("2023-01-31", periods=n_dates, freq="ME")
    rng = np.random.default_rng(4)

    base = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
    other = np.roll(base, 1) * gap + base * (1 - gap)

    for forecaster, centre in (("alpha", base), ("beta", other)):
        for seed in seeds:
            rows = []
            for _ in dates:
                w = centre + rng.normal(0.0, jitter, centre.size)
                w = np.clip(w, 0.0, None)
                rows.append(w / w.sum())
            _write(runs, forecaster, seed, pd.DataFrame(rows, index=dates, columns=ASSETS))
    return runs


def test_between_exceeds_the_seed_noise_floor(aggregate, tmp_path):
    runs = _tree(tmp_path, jitter=0.005, gap=0.9)
    between, within = aggregate.table_displacement(runs, "ga")

    assert len(between) == 1
    assert len(within) == 2
    floor = between.attrs["noise_floor_l1"]
    assert floor > 0
    assert between.loc[0, "l1_mean"] > floor
    assert between.loc[0, "over_noise_floor"] > 3


def test_two_identical_generators_sit_at_the_noise_floor(aggregate, tmp_path):
    """Same centre, so all that is left between them is the seed wobble."""
    runs = _tree(tmp_path, jitter=0.02, gap=0.0)
    between, _ = aggregate.table_displacement(runs, "ga")
    assert between.loc[0, "over_noise_floor"] == pytest.approx(1.0, abs=0.25)


def test_l1_stays_inside_its_range(aggregate, tmp_path):
    runs = _tree(tmp_path, jitter=0.01, gap=1.0)
    between, within = aggregate.table_displacement(runs, "ga")
    assert 0 <= between["l1_mean"].min() and between["l1_mean"].max() <= 2
    assert 0 <= within["l1_mean"].min() and within["l1_mean"].max() <= 2
    assert 0 <= between["jaccard_mean"].min() and between["jaccard_mean"].max() <= 1


def test_support_overlap_counts_names_not_sizes(aggregate, tmp_path):
    """Same five names in wildly different sizes is still a perfect overlap."""
    runs = tmp_path / "runs"
    dates = pd.bdate_range("2023-01-31", periods=6, freq="ME")
    sizes = [[0.9, 0.025, 0.025, 0.025, 0.025]] * len(dates)
    heavy = pd.DataFrame(sizes, index=dates, columns=ASSETS)
    flat = pd.DataFrame([[0.2] * 5] * len(dates), index=dates, columns=ASSETS)
    _write(runs, "heavy", "00", heavy)
    _write(runs, "flat", "00", flat)

    between, _ = aggregate.table_displacement(runs, "ga")
    assert between.loc[0, "jaccard_mean"] == pytest.approx(1.0)
    assert between.loc[0, "l1_mean"] > 1.0


def test_disjoint_portfolios_have_no_overlap(aggregate, tmp_path):
    runs = tmp_path / "runs"
    dates = pd.bdate_range("2023-01-31", periods=6, freq="ME")
    left = pd.DataFrame([[0.5, 0.5, 0.0, 0.0, 0.0]] * len(dates), index=dates, columns=ASSETS)
    right = pd.DataFrame([[0.0, 0.0, 0.5, 0.5, 0.0]] * len(dates), index=dates, columns=ASSETS)
    _write(runs, "left", "00", left)
    _write(runs, "right", "00", right)

    between, _ = aggregate.table_displacement(runs, "ga")
    assert between.loc[0, "jaccard_mean"] == pytest.approx(0.0)
    assert between.loc[0, "l1_mean"] == pytest.approx(2.0)


def test_displacement_never_looks_at_returns(aggregate, tmp_path):
    """The whole point of this quantity. It is computed from decisions only.

    Nothing in the run tree except the weights can change it, so there is no path by
    which realised performance could leak in and turn this into forecast quality
    wearing a different name.
    """
    runs = _tree(tmp_path, jitter=0.01, gap=0.7)
    before, _ = aggregate.table_displacement(runs, "ga")

    rng = np.random.default_rng(1)
    for forecaster in ("alpha", "beta"):
        for seed in ("00", "01", "02"):
            target = runs / "backtests" / forecaster / f"ga_seed{seed}"
            dates = pd.read_parquet(target / "weights.parquet").index
            pd.DataFrame(
                {"net_return": rng.normal(0, 0.3, len(dates))}, index=dates
            ).to_parquet(target / "periods.parquet")

    after, _ = aggregate.table_displacement(runs, "ga")
    pd.testing.assert_frame_equal(before, after)


def test_panel_has_one_row_per_generator_and_date(aggregate, tmp_path):
    runs = _tree(tmp_path, jitter=0.01, gap=0.8, n_dates=12)
    panel = aggregate.displacement_by_date(runs, "ga")

    assert set(panel.columns) == {"rebalance", "forecaster", "displacement_l1", "support_overlap"}
    assert set(panel["forecaster"]) == {"alpha", "beta"}
    assert len(panel) == 12 * 2
    assert (panel["displacement_l1"] >= 0).all()


def test_empty_tree_is_not_fatal(aggregate, tmp_path):
    between, within = aggregate.table_displacement(tmp_path / "nothing", "ga")
    assert between.empty and within.empty
    assert aggregate.displacement_by_date(tmp_path / "nothing", "ga").empty


def test_other_allocators_are_ignored(aggregate, tmp_path):
    runs = _tree(tmp_path, jitter=0.01, gap=0.5)
    dates = pd.bdate_range("2023-01-31", periods=6, freq="ME")
    _write(runs, "alpha", "00",
           pd.DataFrame([[0.2] * 5] * len(dates), index=dates, columns=ASSETS), allocator="hrp")

    between, _ = aggregate.table_displacement(runs, "hrp")
    assert between.empty, "one generator cannot make a pair"
