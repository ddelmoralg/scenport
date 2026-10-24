"""Every bound in the released instance library belongs to its own instance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import solve_instances  # noqa: E402

LIBRARY = Path(__file__).resolve().parent.parent / "artifacts" / "instances"
HEADERS = sorted(p for p in LIBRARY.glob("*.json") if p.name != "index.json")


def test_rockafellar_uryasev_matches_the_tail_mean_when_the_tail_is_whole():
    r = np.linspace(-1.0, 1.0, 1000)
    assert solve_instances.cvar_ru(r, 0.95) == pytest.approx(-np.sort(r)[:50].mean())


def test_a_fractional_tail_weights_its_last_scenario():
    r = -np.arange(1.0, 251.0)                 # losses 1..250, worst is 250
    expected = (np.arange(250, 238, -1).sum() + 0.5 * 238) / 12.5
    assert solve_instances.cvar_ru(r, 0.95) == pytest.approx(expected)


@pytest.mark.parametrize("path", HEADERS, ids=[p.stem for p in HEADERS])
def test_best_known_is_the_objective_of_the_stored_weights(path):
    head = json.loads(path.read_text(encoding="utf-8"))
    if head.get("best_known") is None:
        assert not head.get("proven")
        return
    data = np.load(path.with_suffix(".npz"))
    w = np.asarray(head["best_weights"], dtype=float)
    # HiGHS holds bounds to its primal feasibility tolerance, 1e-7.
    assert solve_instances.feasible(w, head, tol=1e-7)
    value = solve_instances.objective(w, head, data["scenarios"].astype(float),
                                      data["w_prev"].astype(float))
    assert value == pytest.approx(head["best_known"], abs=1e-9)
