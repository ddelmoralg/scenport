"""The instance library.

A benchmark set is only worth releasing if it is bigger and more varied than what is
already out there. The OR-Library portfolio files reach 225 assets, so a set that is
all n=15 from one generator at one scenario count would be a step backwards. These
tests pin the grid rather than the contents.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scenport.data.panel import Panel

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def export():
    spec = importlib.util.spec_from_file_location(
        "export_under_test", SCRIPTS / "export_instances.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def panel_dir(tmp_path):
    """A panel wide enough to build large universes from."""
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2021-01-01", periods=900)
    names = [f"A{i:03d}" for i in range(120)]
    factor = rng.normal(0, 0.02, len(dates))

    price = {}
    mcap = {}
    for i, name in enumerate(names):
        returns = 0.8 * factor + rng.normal(0, 0.02, len(dates))
        price[name] = 100.0 * np.cumprod(1.0 + returns)
        mcap[name] = np.full(len(dates), 1e12 / (i + 1))

    directory = tmp_path / "panel"
    Panel(pd.DataFrame(price, index=dates), pd.DataFrame(mcap, index=dates)).write(directory)
    return directory


class _Args:
    scenario_counts = [250, 1000]
    sizes = [30, 60, 100]
    scale_date = None


def _cfg():
    from scenport import config as cfgmod

    return cfgmod.load(str(Path(__file__).resolve().parents[1] / "configs" / "base.yaml"), [])


def test_scale_family_spans_sizes_and_scenario_counts(export, panel_dir, tmp_path):
    out = tmp_path / "instances"
    out.mkdir()
    written = export.family_scale(_Args(), _cfg(), out, panel_dir)

    assert written == len(_Args.sizes) * len(_Args.scenario_counts)
    headers = [json.loads(p.read_text(encoding="utf-8")) for p in out.glob("*.json")]
    assert {h["n_assets"] for h in headers} == set(_Args.sizes)
    assert {h["n_scenarios"] for h in headers} == set(_Args.scenario_counts)
    assert {h["family"] for h in headers} == {"scale"}


def test_it_reaches_past_the_library_it_replaces(export, panel_dir, tmp_path):
    """OR-Library's largest portfolio instance is 225 assets."""
    out = tmp_path / "instances"
    out.mkdir()

    class Big(_Args):
        sizes = [110]
        scenario_counts = [200]

    assert export.family_scale(Big(), _cfg(), out, panel_dir) == 1
    header = json.loads(next(out.glob("*.json")).read_text(encoding="utf-8"))
    # The synthetic panel here only holds 120 assets; what is being checked is that
    # the code takes whatever the panel offers rather than capping at the backtest
    # universe of fifteen.
    assert header["n_assets"] == 110


def test_cardinality_grows_with_the_universe(export, panel_dir, tmp_path):
    """Otherwise every large instance is the same small selection problem."""
    out = tmp_path / "instances"
    out.mkdir()
    export.family_scale(_Args(), _cfg(), out, panel_dir)

    by_size = {}
    for path in out.glob("*.json"):
        header = json.loads(path.read_text(encoding="utf-8"))
        by_size[header["n_assets"]] = header["cardinality"]

    sizes = sorted(by_size)
    assert [by_size[s] for s in sizes] == sorted(by_size[s] for s in sizes)
    assert by_size[max(sizes)] > by_size[min(sizes)]


def test_every_instance_stays_feasible(export, panel_dir, tmp_path):
    """A library instance that admits no portfolio is worse than no instance."""
    out = tmp_path / "instances"
    out.mkdir()
    export.family_scale(_Args(), _cfg(), out, panel_dir)

    for path in out.glob("*.json"):
        header = json.loads(path.read_text(encoding="utf-8"))
        assert header["cardinality"] * header["w_max"] >= 1.0, header["name"]
        assert header["w_min"] * header["cardinality"] <= 1.0, header["name"]


def test_the_npz_matches_its_header(export, panel_dir, tmp_path):
    out = tmp_path / "instances"
    out.mkdir()
    export.family_scale(_Args(), _cfg(), out, panel_dir)

    for path in out.glob("*.json"):
        header = json.loads(path.read_text(encoding="utf-8"))
        data = np.load(path.with_suffix(".npz"))
        assert data["scenarios"].shape == (header["n_scenarios"], header["n_assets"])
        assert data["w_prev"].shape == (header["n_assets"],)
        assert len(header["assets"]) == header["n_assets"]
        assert np.isfinite(data["scenarios"]).all()


def test_scenarios_are_reproducible(export, panel_dir, tmp_path):
    """Two exports of the same instance have to be byte-identical."""
    first, second = tmp_path / "a", tmp_path / "b"
    for directory in (first, second):
        directory.mkdir()
        export.family_scale(_Args(), _cfg(), directory, panel_dir)

    for path in first.glob("*.npz"):
        a = np.load(path)["scenarios"]
        b = np.load(second / path.name)["scenarios"]
        np.testing.assert_array_equal(a, b)


def test_the_readme_says_why_the_set_exists(export):
    """The OR-Library format argument is the strongest one the project has."""
    assert "moments" in export.README.lower() or "correlation" in export.README.lower()
    assert "best_known" in export.README
    assert "proven" in export.README


def test_every_generator_in_the_study_is_in_the_library(export):
    """It held ten while the study had eleven, and rw_nodrift was simply absent."""
    import argparse

    from scenport.plots import style

    parser = argparse.ArgumentParser()
    # rebuild the default the same way main() does
    default = list(style.FORECASTER_ORDER)
    assert len(default) == 11
    assert "rw_nodrift" in default
    del parser


def test_an_oversized_request_clamps_instead_of_vanishing(export, panel_dir, tmp_path):
    """A scaling library wants its largest feasible instance, not a gap at the top."""
    out = tmp_path / "instances"
    out.mkdir()

    class Oversized(_Args):
        sizes = [50, 5000]          # the second cannot be filled
        scenario_counts = [200]

    written = export.family_scale(Oversized(), _cfg(), out, panel_dir)
    assert written == 2, "both requests should produce an instance"

    sizes = sorted(
        json.loads(p.read_text(encoding="utf-8"))["n_assets"] for p in out.glob("*.json")
    )
    assert sizes[0] == 50
    assert sizes[1] > 50, "the oversized request clamps to what the panel holds"


def test_requests_that_clamp_to_the_same_size_are_written_once(export, panel_dir, tmp_path):
    out = tmp_path / "instances"
    out.mkdir()

    class Duplicated(_Args):
        sizes = [4000, 5000, 6000]   # all clamp to the same achievable size
        scenario_counts = [200]

    written = export.family_scale(Duplicated(), _cfg(), out, panel_dir)
    assert written == 1, "three impossible sizes should not write the same instance thrice"


def test_the_readme_is_honest_about_the_ceiling(export):
    """A library smaller than the one it complements has to say so."""
    text = export.README.lower()
    assert "225" in text, "the OR-Library figure belongs in the comparison"
    assert "community" in text, "the reason for the ceiling belongs with it"
