"""Rebuild the two failures of the first full run and confirm the check catches them.

A guard nobody has seen fire is a guess. Both faults below are reconstructions of
what was actually on disk: scenario sets holding a 30-day return of 1e14, and a
gap study in which the relaxation outscored branch and bound on every date.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
ASSETS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("check_under_test", SCRIPTS / "check_results.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _gaps(tmp_path, rows) -> Path:
    runs = tmp_path / "runs"
    (runs / "exact").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(runs / "exact" / "gaps.parquet")
    return runs


CEILING = 90.0


def _scenario_set(path: Path, forecaster: str, clipped_share: float, size: int = 200_000):
    """A set in which `clipped_share` of the points sit exactly on the ceiling.

    Written as real paths rather than as a metadata field, because that is how the
    check reads it: a clipped point lands on the bound, so the data says how often
    that happened whether or not anything recorded it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = np.full(size, 0.1, dtype=np.float32)
    flat[: int(round(clipped_share * size))] = CEILING
    np.savez_compressed(
        path,
        paths=flat.reshape(size // 100, 50, 2),
        assets=np.array(["AAA", "BBB"], dtype=object),
        origin="2024-01-01",
        forecaster=forecaster,
        meta=json.dumps({"max_cumulative_return": CEILING}),
    )


# ---------------------------------------------------------------------------
# fault 1: the objective was 38,054


def test_an_impossible_objective_is_caught(checker, tmp_path):
    runs = _gaps(
        tmp_path,
        [
            {"rebalance": "2024-07-01", "allocator": "milp", "objective": 6403531.23,
             "relative_gap": 0.0, "seconds": 1.0, "kind": "exact", "status": "optimal"},
        ],
    )
    report = checker.Report()
    checker.check_objective_scale(runs, report)
    assert report.failures, "an objective of 6.4 million has to fail"
    assert "not a portfolio" in report.failures[0]


def test_a_normal_objective_passes(checker, tmp_path):
    runs = _gaps(
        tmp_path,
        [
            {"rebalance": "2024-07-01", "allocator": "milp", "objective": -0.0138,
             "relative_gap": 0.0, "seconds": 1.0, "kind": "exact", "status": "optimal"},
        ],
    )
    report = checker.Report()
    checker.check_objective_scale(runs, report)
    assert not report.failures


# ---------------------------------------------------------------------------
# fault 2: a heuristic beat branch and bound


def test_a_candidate_beating_the_optimum_is_caught(checker, tmp_path):
    runs = _gaps(
        tmp_path,
        [
            {"rebalance": "2019-08-01", "allocator": "milp", "objective": -0.018286,
             "relative_gap": 0.0, "seconds": 1.0, "kind": "exact", "status": "optimal"},
            {"rebalance": "2019-08-01", "allocator": "cvar_lp", "objective": -0.016822,
             "relative_gap": -0.08, "seconds": 0.1, "kind": "candidate"},
        ],
    )
    report = checker.Report()
    checker.check_nothing_beats_the_optimum(runs, report)
    assert report.failures
    assert "cvar_lp" in report.failures[0]


def test_the_relaxation_is_allowed_to_beat_it(checker, tmp_path):
    """Which is the whole reason it was given a name of its own."""
    runs = _gaps(
        tmp_path,
        [
            {"rebalance": "2019-08-01", "allocator": "milp", "objective": -0.018286,
             "relative_gap": 0.0, "seconds": 1.0, "kind": "exact", "status": "optimal"},
            {"rebalance": "2019-08-01", "allocator": "cvar_lp_bound", "objective": -0.016822,
             "relative_gap": 0.08, "seconds": 0.1, "kind": "relaxation"},
            {"rebalance": "2019-08-01", "allocator": "ga", "objective": -0.018300,
             "relative_gap": 0.0008, "seconds": 3.0, "kind": "candidate"},
        ],
    )
    report = checker.Report()
    checker.check_nothing_beats_the_optimum(runs, report)
    assert not report.failures


# ---------------------------------------------------------------------------
# the ceiling, and who leaned on it


def test_heavy_clipping_fails(checker, tmp_path):
    runs = tmp_path / "runs"
    _scenario_set(runs / "scenarios" / "wild" / "seed00" / "2024-01-01.npz", "wild", 0.35)
    report = checker.Report()
    checker.check_scenarios(runs, report)
    assert report.failures
    assert "reshaping" in report.failures[0]


def test_light_clipping_only_warns(checker, tmp_path):
    runs = tmp_path / "runs"
    # The share AR-GARCH actually clipped in the run that made this necessary.
    _scenario_set(runs / "scenarios" / "garch" / "seed00" / "2024-01-01.npz", "garch", 3.3e-4)
    report = checker.Report()
    checker.check_scenarios(runs, report)
    assert not report.failures
    assert report.warnings, "it should still say which generator leaned on the ceiling"


# ---------------------------------------------------------------------------
# feasibility, which is what the backtest was not enforcing


class _Cfg(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


def _weights(runs: Path, label: str, rows):
    target = runs / "backtests" / label
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, index=pd.bdate_range("2024-01-31", periods=len(rows), freq="ME"),
                 columns=ASSETS).to_parquet(target / "weights.parquet")


def test_too_many_names_is_caught(checker, tmp_path):
    runs = tmp_path / "runs"
    _weights(runs, "garch/cvar_lp_seed00", [[0.2, 0.2, 0.2, 0.2, 0.2]])
    cfg = _Cfg({"alloc.cardinality": 3, "alloc.w_min": 0.02, "alloc.w_max": 0.5})
    report = checker.Report()
    checker.check_feasibility(runs, cfg, report)
    assert report.failures
    assert "holds 5" in report.failures[0]


def test_a_feasible_history_passes(checker, tmp_path):
    runs = tmp_path / "runs"
    _weights(runs, "garch/ga_seed00", [[0.4, 0.3, 0.3, 0.0, 0.0]])
    cfg = _Cfg({"alloc.cardinality": 3, "alloc.w_min": 0.02, "alloc.w_max": 0.5})
    report = checker.Report()
    checker.check_feasibility(runs, cfg, report)
    assert not report.failures


# ---------------------------------------------------------------------------
# the ablations counted as generators


def test_too_many_generators_is_caught(checker, tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    pd.DataFrame({
        "forecaster": [f"g{i}" for i in range(28)],
        "displacement": [0.5] * 28,
        "overlap": [0.6] * 28,
    }).to_csv(artifacts / "sensitivity.csv", index=False)

    report = checker.Report()
    checker.check_sensitivity(artifacts, report)
    assert report.failures
    assert "eleven forecasters" in report.failures[0]


def test_eleven_generators_pass(checker, tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    pd.DataFrame({
        "forecaster": [f"g{i}" for i in range(11)],
        "displacement": [0.79] * 11,
        "overlap": [0.55] * 11,
    }).to_csv(artifacts / "sensitivity.csv", index=False)

    report = checker.Report()
    checker.check_sensitivity(artifacts, report)
    assert not report.failures
