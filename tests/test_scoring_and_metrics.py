"""Scoring rules and performance statistics, checked against closed forms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scenport import scoring
from scenport.backtest import metrics


def test_crps_of_a_point_mass_is_the_absolute_error():
    samples = np.full(500, 2.0)
    assert scoring.crps(samples, 3.5) == pytest.approx(1.5, abs=1e-9)


def test_crps_rewards_the_better_forecast(rng):
    truth = 0.05
    sharp = rng.normal(truth, 0.02, 4000)
    vague = rng.normal(truth, 0.20, 4000)
    biased = rng.normal(truth + 0.30, 0.02, 4000)
    assert scoring.crps(sharp, truth) < scoring.crps(vague, truth)
    assert scoring.crps(sharp, truth) < scoring.crps(biased, truth)


def test_crps_matches_the_pairwise_definition(rng):
    x = rng.standard_normal(200)
    y = 0.3
    naive = np.mean(np.abs(x - y)) - 0.5 * np.sum(np.abs(x[:, None] - x[None, :])) / (
        len(x) * (len(x) - 1)
    )
    assert scoring.crps(x, y, fair=True) == pytest.approx(naive, rel=1e-10)


def test_pit_is_uniform_when_the_forecast_is_right(rng):
    from scipy import stats

    values = [
        scoring.pit(rng.standard_normal(500), rng.standard_normal(), rng) for _ in range(3000)
    ]
    assert stats.kstest(values, "uniform").pvalue > 0.01


def test_pit_is_u_shaped_when_the_intervals_are_too_narrow(rng):
    values = np.array(
        [scoring.pit(rng.normal(0, 0.2, 500), rng.standard_normal(), rng) for _ in range(3000)]
    )
    extreme = np.mean((values < 0.1) | (values > 0.9))
    assert extreme > 0.4          # 0.2 would be the correctly specified case


def test_coverage_is_close_to_nominal_for_a_correct_forecast(rng):
    hits = [
        scoring.interval_coverage(rng.standard_normal(400), rng.standard_normal(), 0.80)
        for _ in range(2000)
    ]
    assert 0.77 < np.mean(hits) < 0.83


def test_energy_score_sees_dependence_that_crps_cannot(rng):
    """Same marginals, different correlation. Per-asset CRPS is blind to this;
    the energy score is not, which is why the coupling step is scored with it."""
    n = 3000
    truth = np.array([0.10, 0.10])

    common = rng.standard_normal(n)
    correlated = np.column_stack([0.1 + 0.1 * common, 0.1 + 0.1 * common])
    scrambled = correlated.copy()
    rng.shuffle(scrambled[:, 1])

    crps_a = np.mean([scoring.crps(correlated[:, j], truth[j]) for j in range(2)])
    crps_b = np.mean([scoring.crps(scrambled[:, j], truth[j]) for j in range(2)])
    assert crps_a == pytest.approx(crps_b, rel=1e-9)      # identical marginals

    assert scoring.energy_score(correlated, truth) < scoring.energy_score(scrambled, truth)


def test_scores_survive_missing_observations():
    assert np.isnan(scoring.crps(np.array([1.0, 2.0]), np.nan))
    assert np.isnan(scoring.energy_score(np.zeros((10, 2)), np.array([np.nan, np.nan])))


def test_sharpe_of_a_constant_series_is_not_a_number():
    assert np.isnan(metrics.sharpe(pd.Series(np.full(100, 0.001))))


def test_sharpe_annualises_correctly(rng):
    daily = pd.Series(rng.normal(0.0005, 0.01, 3650))
    expected = daily.mean() / daily.std(ddof=1) * np.sqrt(365)
    assert metrics.sharpe(daily) == pytest.approx(expected, rel=1e-9)


def test_max_drawdown_on_a_known_path():
    # 1.00 -> 1.20 -> 0.60 -> 0.90, so the worst peak-to-trough is -50 %.
    returns = pd.Series([0.20, -0.50, 0.50])
    assert metrics.max_drawdown(returns) == pytest.approx(-0.50, abs=1e-12)


def test_annualised_return_recovers_a_constant_growth_rate():
    daily = pd.Series(np.full(365, 1.10 ** (1 / 365) - 1.0))
    assert metrics.annualised_return(daily) == pytest.approx(0.10, rel=1e-6)


def test_sortino_ignores_upside_volatility():
    calm = pd.Series([0.01, 0.01, -0.01, 0.01] * 50)
    spiky = pd.Series([0.20, 0.01, -0.01, 0.01] * 50)
    assert metrics.sortino(spiky) > metrics.sortino(calm)


def test_cvar_is_never_below_var(rng):
    returns = pd.Series(rng.standard_t(4, 5000) * 0.02)
    assert metrics.conditional_var(returns, 0.95) >= metrics.value_at_risk(returns, 0.95)


def test_regime_labels_follow_the_benchmark():
    dates = pd.date_range("2020-01-01", periods=400, freq="D")
    rising = pd.Series(np.exp(np.linspace(0, 1.5, 400)), index=dates)
    labels = metrics.regime_labels(rising, window=90, threshold=0.10)
    assert (labels.iloc[90:] == "bull").all()
    assert (labels.iloc[:90] == "warmup").all()
