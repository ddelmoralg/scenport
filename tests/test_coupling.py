"""Rank reordering must change the dependence and nothing else.

The whole separation of forecaster from dependence model rests on this: if the
coupling step altered the marginals, then a forecaster's scenario set would no
longer be its own, and the factorial design would be comparing something else.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import rankdata

from scenport.forecast.base import ScenarioSet
from scenport.forecast.coupling import (
    apply,
    nearest_psd,
    realised_spearman,
    spearman_to_pearson,
)


@pytest.fixture
def independent_set(rng, synthetic_panel):
    """Marginals simulated with no cross-asset dependence at all."""
    assets = list(synthetic_panel.price.columns[:5])
    n_paths, horizon = 2000, 20
    daily = rng.standard_normal((n_paths, horizon, len(assets))) * 0.03
    daily += np.array([0.001, 0.0, -0.001, 0.002, 0.0])
    paths = np.cumprod(1.0 + daily, axis=1) - 1.0
    # Deliberately not the last date in the panel, so that handing the coupling
    # step the whole panel is genuinely a look-ahead and the guard can be tested.
    origin = synthetic_panel.dates[-200]
    return ScenarioSet(paths=paths, assets=assets, origin=origin, forecaster="test")


def _history(panel, scen):
    return panel.price.loc[: scen.origin, scen.assets]


def test_marginals_survive_the_reordering(independent_set, synthetic_panel, rng):
    coupled = apply(
        independent_set, _history(synthetic_panel, independent_set),
        method="gaussian_copula", rng=rng,
    )
    for j in range(len(independent_set.assets)):
        before = np.sort(independent_set.paths[:, -1, j])
        after = np.sort(coupled.paths[:, -1, j])
        assert np.allclose(before, after)


def test_whole_paths_move_together(independent_set, synthetic_panel, rng):
    """A path is permuted as a unit, not resampled step by step."""
    coupled = apply(
        independent_set, _history(synthetic_panel, independent_set),
        method="gaussian_copula", rng=rng,
    )
    original = {tuple(np.round(independent_set.paths[s, :, 0], 12))
                for s in range(independent_set.n_paths)}
    for s in range(0, coupled.n_paths, 37):
        assert tuple(np.round(coupled.paths[s, :, 0], 12)) in original


def test_coupling_creates_the_dependence_that_was_missing(
    independent_set, synthetic_panel, rng
):
    before = realised_spearman(independent_set)
    coupled = apply(
        independent_set, _history(synthetic_panel, independent_set),
        method="gaussian_copula", rng=rng,
    )
    after = realised_spearman(coupled)
    off = ~np.eye(len(independent_set.assets), dtype=bool)

    assert abs(before[off]).mean() < 0.05          # independent to begin with
    assert after[off].mean() > 0.3                 # the panel is strongly co-moving


def test_recovers_a_known_target(rng):
    """Feed in a target directly and check the reordering reproduces it."""
    from scenport.forecast.coupling import _reorder_to

    n_paths, n_assets = 4000, 4
    paths = rng.standard_normal((n_paths, 1, n_assets))
    target_corr = np.array(
        [
            [1.0, 0.8, 0.4, 0.0],
            [0.8, 1.0, 0.3, 0.0],
            [0.4, 0.3, 1.0, 0.2],
            [0.0, 0.0, 0.2, 1.0],
        ]
    )
    template = rng.standard_normal((n_paths, n_assets)) @ np.linalg.cholesky(target_corr).T
    reordered = _reorder_to(paths, template)
    achieved = np.corrcoef(rankdata(reordered[:, -1, :], axis=0), rowvar=False)

    iu = np.triu_indices(n_assets, k=1)
    # Rank correlation is slightly below the Pearson target by construction; the
    # tolerance covers that plus Monte Carlo error at 4000 draws.
    assert np.abs(achieved[iu] - target_corr[iu]).max() < 0.07


def test_independent_is_a_no_op(independent_set, synthetic_panel, rng):
    same = apply(
        independent_set, _history(synthetic_panel, independent_set),
        method="independent", rng=rng,
    )
    assert same is independent_set


def test_refuses_to_look_at_the_future(independent_set, synthetic_panel, rng):
    """Coupling handed prices past the origin would be leakage."""
    with pytest.raises(ValueError, match="past the scenario origin"):
        apply(independent_set, synthetic_panel.price[independent_set.assets],
              method="gaussian_copula", rng=rng)


def test_spearman_to_pearson_is_monotone_and_bounded():
    rho_s = np.linspace(-0.99, 0.99, 41)
    grid = np.diag(np.ones(2))
    values = [spearman_to_pearson(np.array([[1.0, r], [r, 1.0]]))[0, 1] for r in rho_s]
    assert np.all(np.diff(values) > 0)
    assert max(abs(v) for v in values) < 1.0
    assert spearman_to_pearson(grid)[0, 1] == pytest.approx(0.0)


def test_nearest_psd_repairs_an_indefinite_matrix():
    broken = np.array([[1.0, 0.95, -0.95], [0.95, 1.0, 0.95], [-0.95, 0.95, 1.0]])
    assert np.linalg.eigvalsh(broken).min() < 0
    fixed = nearest_psd(broken)
    assert np.linalg.eigvalsh(fixed).min() >= -1e-10
    assert np.allclose(np.diag(fixed), 1.0)
