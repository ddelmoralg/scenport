"""Contract tests every forecaster has to pass.

Parametrised over the registry, so a new forecaster is covered the moment it is
registered and cannot quietly return the wrong shape or a path that starts
somewhere other than zero.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from scenport.forecast import CPU_FORECASTERS, get

HORIZON = 10
N_PATHS = 60


@functools.lru_cache(maxsize=1)
def _lightgbm_usable() -> bool:
    """Probe LightGBM before letting a test depend on it.

    On some Windows installations LightGBM's OpenMP runtime clashes with the one
    already loaded by pandas, and the first Dataset construction dies with an
    access violation. It has nothing to do with this package, and the same failure
    reproduces in four lines of plain LightGBM, but it should be reported as an
    environment problem rather than as a broken forecaster.
    """
    try:
        import lightgbm as lgb

        lgb.train(
            {"objective": "regression", "verbosity": -1, "num_threads": 1},
            lgb.Dataset(np.zeros((40, 2)), label=np.arange(40, dtype=float)),
            num_boost_round=1,
        )
        return True
    except (OSError, ImportError):
        return False


@pytest.fixture
def history(synthetic_panel):
    return synthetic_panel.price.iloc[-500:, :4]


def _build(name):
    if name in ("prophet", "prophet_rmse"):
        pytest.importorskip("prophet")
        # One grid point and no inner folds: the tuning path is exercised in the
        # dedicated test below, and eight Stan fits per asset is too slow here.
        return get(name, grid=[{"changepoint_prior_scale": 0.05}])
    if name == "lgbm":
        pytest.importorskip("lightgbm")
        if not _lightgbm_usable():
            pytest.skip("LightGBM's OpenMP runtime is broken in this environment")
    if name == "garch":
        pytest.importorskip("arch")
    return get(name)


@pytest.mark.parametrize("name", CPU_FORECASTERS)
def test_shape_and_convention(name, history, rng):
    scenarios = _build(name).simulate(history, HORIZON, N_PATHS, rng)

    assert scenarios.paths.shape == (N_PATHS, HORIZON, history.shape[1])
    assert scenarios.assets == list(history.columns)
    assert scenarios.origin == history.index[-1]
    assert np.isfinite(scenarios.paths).all(), f"{name} produced non-finite paths"
    # Cumulative returns from the origin: a total loss is -1, nothing below it.
    assert (scenarios.paths >= -1.0).all()


@pytest.mark.parametrize("name", CPU_FORECASTERS)
def test_reproducible_from_a_seed(name, history):
    a = _build(name).simulate(history, HORIZON, N_PATHS, np.random.default_rng(3))
    b = _build(name).simulate(history, HORIZON, N_PATHS, np.random.default_rng(3))
    assert np.allclose(a.paths, b.paths), f"{name} is not reproducible"


@pytest.mark.parametrize("name", [n for n in CPU_FORECASTERS if n != "flat"])
def test_dispersion_grows_with_the_horizon(name, history, rng):
    """Uncertainty about a month out has to exceed uncertainty about tomorrow."""
    if name in ("prophet", "prophet_rmse"):
        pytest.importorskip("prophet")
    scenarios = _build(name).simulate(history, HORIZON, 200, rng)
    spread = scenarios.paths.std(axis=0).mean(axis=1)     # per horizon step
    assert spread[-1] > spread[0]


def test_flat_forecaster_has_no_dispersion(history, rng):
    scenarios = get("flat").simulate(history, HORIZON, N_PATHS, rng)
    assert np.allclose(scenarios.paths, 0.0)
    assert np.allclose(scenarios.covariance(), 0.0)


def test_scenario_set_round_trips_through_disk(history, rng, tmp_path):
    from scenport.forecast.base import ScenarioSet

    original = get("rw").simulate(history, HORIZON, N_PATHS, rng)
    original.to_npz(tmp_path / "s.npz")
    restored = ScenarioSet.from_npz(tmp_path / "s.npz")

    assert restored.assets == original.assets
    assert restored.origin == original.origin
    assert np.allclose(restored.paths, original.paths, atol=1e-6)   # stored as float32


def test_reorder_permutes_without_losing_anything(history, rng):
    scenarios = get("rw").simulate(history, HORIZON, N_PATHS, rng)
    flipped = scenarios.reorder(list(reversed(scenarios.assets)))
    assert flipped.assets == list(reversed(scenarios.assets))
    assert np.allclose(flipped.paths[:, :, -1], scenarios.paths[:, :, 0])


def _mean_daily_log_return(scenarios) -> np.ndarray:
    """Average simulated log return per day, per asset."""
    return np.log1p(scenarios.terminal()).mean(axis=0) / scenarios.horizon


def test_the_two_random_walks_differ_exactly_by_the_drift(history, rng):
    """Centring happens on log returns, so the driftless walk has none there.

    Its *mean simple* return stays slightly positive by Jensen's inequality, which
    is correct behaviour and not something to test away, hence the check on log
    returns.
    """
    historical = np.diff(np.log(history.to_numpy(float)), axis=0).mean(axis=0)

    plain = _mean_daily_log_return(get("rw").simulate(history, HORIZON, 4000, rng))
    centred = _mean_daily_log_return(get("rw_nodrift").simulate(history, HORIZON, 4000, rng))

    # The bootstrap is unbiased, so both should land within Monte Carlo error of
    # their targets: the historical drift, and zero.
    tolerance = 3e-4
    assert np.abs(plain - historical).max() < tolerance
    assert np.abs(centred).max() < tolerance


@pytest.mark.slow
def test_prophet_tuning_picks_a_grid_point(history, rng):
    pytest.importorskip("prophet")
    model = get(
        "prophet",
        grid=[
            {"changepoint_prior_scale": 0.001},
            {"changepoint_prior_scale": 0.50},
        ],
        n_folds=2,
        select_paths=60,
    )
    chosen = model._select(history.iloc[:, 0].dropna(), HORIZON, rng)
    assert chosen["changepoint_prior_scale"] in (0.001, 0.50)


# ---------------------------------------------------------------------------
# statsmodels moved the simulate() seeding keyword in 0.15


def test_seed_kwargs_matches_what_the_installed_statsmodels_takes():
    """Both spellings have to work, because the cluster and the laptop differ.

    0.14 takes `random_state` and a legacy RandomState, 0.15 takes `rng` and a
    Generator. This was found by running the suite on the cluster: six tests
    failed there and none here.
    """
    import inspect

    import numpy as np

    from scenport.forecast.classical import _seed_kwargs

    rng = np.random.default_rng(0)

    def new_style(nsimulations, anchor=None, repetitions=1, random_errors=None, rng=None):
        pass

    def old_style(nsimulations, anchor=None, repetitions=1, random_state=None):
        pass

    assert set(_seed_kwargs(new_style, rng)) == {"rng"}
    assert set(_seed_kwargs(old_style, rng)) == {"random_state"}

    for fn in (new_style, old_style):
        kwargs = _seed_kwargs(fn, rng)
        inspect.signature(fn).bind(nsimulations=1, **kwargs)


def test_unseedable_simulate_is_refused():
    """Falling back to the global state would make two array tasks agree by accident."""
    import numpy as np
    import pytest as _pytest

    from scenport.forecast.classical import _seed_kwargs

    def unseedable(nsimulations, anchor=None, repetitions=1):
        pass

    with _pytest.raises(TypeError, match="neither"):
        _seed_kwargs(unseedable, np.random.default_rng(0))
