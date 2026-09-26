"""Panel inference for the transmission question.

Eleven forecasters are eleven points, and the whole reason the panel exists is that
eleven points cannot answer the question. These tests pin the properties that claim
rests on: the power arithmetic, that clustering on the date widens the interval the
way it should, that resampling covers the truth, and that the permutation test finds
a real effect and stays quiet when there is none.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scenport.stats import tests as T


def _panel(beta: float, n_dates: int = 80, n_forecasters: int = 11, seed: int = 0):
    """Forecaster-date panel with a shared monthly shock on top of the effect."""
    rng = np.random.default_rng(seed)
    date = np.repeat(np.arange(n_dates), n_forecasters)
    shock = np.repeat(rng.normal(0.0, 0.05, n_dates), n_forecasters)
    x = rng.normal(0.0, 1.0, date.size)
    y = shock + beta * x + rng.normal(0.0, 0.02, date.size)
    frame = pd.DataFrame({"date": date, "x": x, "y": y})
    demeaned = T.within_transform(frame, ["x", "y"], "date")
    return frame, demeaned["x"].to_numpy().reshape(-1, 1), demeaned["y"].to_numpy(), date


# ---------------------------------------------------------------------------
# power


def test_eleven_points_cannot_resolve_a_moderate_correlation():
    """The number that justifies the whole design."""
    assert T.minimum_detectable_correlation(11) == pytest.approx(0.758, abs=0.01)
    assert T.minimum_detectable_correlation(11) > 0.7


def test_power_improves_with_sample_size():
    sizes = [11, 20, 50, 138, 500]
    detectable = [T.minimum_detectable_correlation(n) for n in sizes]
    assert detectable == sorted(detectable, reverse=True)


def test_minimum_detectable_is_undefined_for_tiny_samples():
    for n in (0, 1, 3):
        assert np.isnan(T.minimum_detectable_correlation(n))


def test_required_sample_round_trips_against_detectable():
    """The two functions are inverses, so neither can drift from the other."""
    for n in (20, 50, 138):
        rho = T.minimum_detectable_correlation(n)
        assert T.required_sample_for_correlation(rho) == pytest.approx(n, abs=1)


def test_the_m6_correlation_is_out_of_reach_here():
    """M6 reported r = 0.04. Nothing this study could run would resolve that."""
    assert T.required_sample_for_correlation(0.04) > 4000


# ---------------------------------------------------------------------------
# estimation and inference


def test_within_transform_removes_the_group_mean():
    frame = pd.DataFrame({"g": [0, 0, 1, 1], "v": [1.0, 3.0, 10.0, 20.0]})
    out = T.within_transform(frame, ["v"], "g")
    assert out["v"].tolist() == [-1.0, 1.0, -5.0, 5.0]
    assert frame["v"].tolist() == [1.0, 3.0, 10.0, 20.0], "must not mutate the input"


def test_slope_is_recovered():
    _, x, y, date = _panel(beta=0.01)
    fit = T.cluster_robust_ols(y, x, date)
    assert fit["beta"][0] == pytest.approx(0.01, abs=0.002)
    assert fit["n_obs"] == 80 * 11
    assert fit["n_clusters"] == 80


def test_clustering_on_the_date_widens_the_standard_error():
    """A common monthly shock makes the naive error too small. That is the point."""
    _, x, y, date = _panel(beta=0.01)
    clustered = T.cluster_robust_ols(y, x, date)["std_error"][0]
    independent = T.cluster_robust_ols(y, x, np.arange(y.size))["std_error"][0]
    assert clustered > independent


def test_bootstrap_interval_covers_the_truth():
    _, x, y, date = _panel(beta=0.01)
    ci = T.cluster_bootstrap_ci(y, x, date, reps=400, seed=1)
    assert ci["lower"] < 0.01 < ci["upper"]
    assert ci["reps"] > 0


def test_bootstrap_interval_excludes_zero_when_the_effect_is_strong():
    _, x, y, date = _panel(beta=0.05)
    ci = T.cluster_bootstrap_ci(y, x, date, reps=400, seed=1)
    assert ci["lower"] > 0


# ---------------------------------------------------------------------------
# permutation


def test_permutation_finds_a_real_effect():
    _, x, y, date = _panel(beta=0.02)
    result = T.permutation_within_clusters(y, x, date, reps=500, seed=2)
    assert result.p_value < 0.01


def test_permutation_stays_quiet_under_the_null():
    _, x, y, date = _panel(beta=0.0)
    result = T.permutation_within_clusters(y, x, date, reps=500, seed=2)
    assert result.p_value > 0.05


def test_permutation_p_value_is_never_zero():
    """Add-one on both sides. Reporting p = 0 from a finite permutation is wrong."""
    _, x, y, date = _panel(beta=0.5)
    result = T.permutation_within_clusters(y, x, date, reps=100, seed=3)
    assert result.p_value > 0
    assert result.p_value == pytest.approx(1 / 101, abs=1e-9)


# ---------------------------------------------------------------------------
# the control


def test_a_confound_is_absorbed_by_the_control():
    """Net market exposure is the confound the beta control exists for.

    The market is given a positive drift here, which is not a convenience: it is the
    sample. Crypto from 2019 to 2026 is one big cycle that ends higher than it
    started, so a forecaster that simply held more of the market earns more, and any
    forecast diagnostic that happens to correlate with exposure inherits the credit.

    With a zero-mean market the date fixed effect would handle this on its own, since
    the confound would pay in up months and cost the same in down months. It is the
    drift that makes the control necessary.
    """
    rng = np.random.default_rng(5)
    n_dates, n_fc = 120, 11
    date = np.repeat(np.arange(n_dates), n_fc)
    market = np.repeat(rng.normal(0.04, 0.05, n_dates), n_fc)

    exposure = rng.normal(1.0, 0.3, date.size)
    score = exposure + rng.normal(0.0, 0.1, date.size)
    y = exposure * market + rng.normal(0.0, 0.005, date.size)

    frame = pd.DataFrame({"date": date, "score": score, "exposure": exposure, "y": y})
    dm = T.within_transform(frame, ["score", "exposure", "y"], "date")

    naive = T.cluster_robust_ols(
        dm["y"].to_numpy(), dm[["score"]].to_numpy(), date
    )
    controlled = T.cluster_robust_ols(
        dm["y"].to_numpy(), dm[["score", "exposure"]].to_numpy(), date
    )

    assert abs(naive["t_stat"][0]) > 2, "the confound should show up without the control"
    assert abs(controlled["t_stat"][0]) < abs(naive["t_stat"][0])
    assert abs(controlled["t_stat"][1]) > abs(controlled["t_stat"][0])
