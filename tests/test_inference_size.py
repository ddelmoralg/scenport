"""The simulation that decides which inference the panel regression uses.

Two tests of one null disagreed on the real data, and picking by preference is
not an argument. These pin the properties the choice rests on, so a later edit
cannot quietly move it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("size_study", SCRIPTS / "inference_size_study.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _skew_of(sigma: float) -> float:
    u = np.exp(sigma * sigma)
    return (u + 2.0) * np.sqrt(max(u - 1.0, 0.0))


def test_the_skew_inversion_round_trips(study):
    """Matching the measured skew is what lets the grid speak about real metrics."""
    for target in (0.5, 0.9, 3.17, 3.50, 8.17):
        sigma = study.sigma_for_skew(target)
        assert _skew_of(sigma) == pytest.approx(target, rel=1e-3)


def test_a_symmetric_diagnostic_is_not_skewed(study):
    assert study.sigma_for_skew(0.0) == 0.0


def test_the_grid_brackets_what_was_measured(study):
    """MASE is the extreme case and the grid has to reach it."""
    assert max(s for s, _ in study.REAL.values()) == pytest.approx(8.17)
    assert study.sigma_for_skew(8.17) < 3.0, "the simulated grid tops out at sigma 3"


def test_the_panel_has_the_shape_of_the_real_one(study):
    y, X, date = study.synthetic_panel(0.0, np.random.default_rng(0), 1.0, 0.4)
    assert y.shape == (study.N_DATES * study.N_GENERATORS,)
    assert X.shape == (study.N_DATES * study.N_GENERATORS, 2)
    assert len(np.unique(date)) == study.N_DATES


def test_a_skewed_diagnostic_really_is_skewed(study):
    """If the fixture is not skewed the whole study says nothing."""
    import pandas as pd

    _, X, _ = study.synthetic_panel(0.0, np.random.default_rng(0), 2.0, 0.0)
    assert pd.Series(X[:, 0]).skew() > 3.0


@pytest.mark.slow
def test_skew_breaks_the_clustered_test_and_not_the_permutation(study):
    """The finding the choice of inference rests on.

    Few replications, so this checks the direction and the size of the effect
    rather than the exact rates. The published numbers come from the script.
    """
    clustered_flat, permutation_flat = study.rejection_rates(
        0.0, 0.0, 0.0, reps=60, seed=5, permutations=200
    )
    clustered_skewed, permutation_skewed = study.rejection_rates(
        0.0, 3.0, 0.0, reps=60, seed=5, permutations=200
    )

    assert clustered_skewed > clustered_flat + 0.08, "skew has to break the clustered test"
    assert permutation_skewed < 0.12, "the permutation test has to survive it"
    assert clustered_skewed > permutation_skewed * 2
