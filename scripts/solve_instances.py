#!/usr/bin/env python
"""Solve every released instance exactly, on its own scenarios, and record the answer.

    python scripts/solve_instances.py                    # all of artifacts/instances
    python scripts/solve_instances.py --only gen_rw_2019-08-01_n15_S1000

Each instance is solved as released: every scenario, no thinning. The conditional
value at risk is the Rockafellar-Uryasev form, which is also how the README defines
it, so for a scenario count where (1 - alpha) S is not whole the tail carries a
fractional weight on its last scenario.

Written into each JSON header:

  best_known       objective of best_weights, recomputed from the scenarios
  best_weights     the portfolio that attains it
  proven           the solver reported optimality within mip_rel_gap
  solver, mip_rel_gap, solve_seconds, status

Before this script the headers borrowed a bound from the gap study, which solved one
generator on a 400-scenario subsample of each date. Every instance on that date
carried that number, whatever its generator and scenario count, so the bound
belonged to a different problem. Nothing here is borrowed: the number in a header is
the objective of the weights in the same header, on the scenarios in the same npz.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
MIP_REL_GAP = 1e-6
log = logging.getLogger("solve_instances")


def cvar_ru(returns: np.ndarray, alpha: float) -> float:
    """Rockafellar-Uryasev CVaR of the loss -returns, as a positive number.

    Equal to the mean of the worst (1 - alpha) S scenarios when that count is
    whole, and to the weighted version with a fractional last scenario when not.
    """
    losses = np.sort(-np.asarray(returns, dtype=float))[::-1]
    m = (1.0 - alpha) * losses.size
    k = int(np.floor(m))
    total = losses[:k].sum() + (m - k) * (losses[k] if k < losses.size else 0.0)
    return float(total / m)


def objective(w: np.ndarray, head: dict, R: np.ndarray, w_prev: np.ndarray) -> float:
    r = R @ w
    gamma = head["gamma"]
    return float((1 - gamma) * r.mean() - gamma * cvar_ru(r, head["cvar_alpha"])
                 - head["cost_rate"] * np.abs(w - w_prev).sum())


def feasible(w: np.ndarray, head: dict, tol: float = 1e-7) -> bool:
    held = w > tol
    return bool(abs(w.sum() - 1.0) < 1e-7 and (w >= -tol).all()
                 and (w <= head["w_max"] + tol).all()
                 and (w[held] >= head["w_min"] - tol).all()
                 and held.sum() <= head["cardinality"])


def solve(head: dict, R: np.ndarray, w_prev: np.ndarray, time_limit: float):
    import cvxpy as cp

    S, n = R.shape
    alpha, gamma = head["cvar_alpha"], head["gamma"]
    w = cp.Variable(n, nonneg=True)
    b = cp.Variable(n, boolean=True)
    zeta = cp.Variable()
    u = cp.Variable(S, nonneg=True)
    buy = cp.Variable(n, nonneg=True)
    sell = cp.Variable(n, nonneg=True)
    cvar = zeta + cp.sum(u) / ((1.0 - alpha) * S)
    constraints = [
        cp.sum(w) == 1,
        u >= -(R @ w) - zeta,
        w <= head["w_max"] * b,
        w >= head["w_min"] * b,
        cp.sum(b) <= head["cardinality"],
        w - w_prev == buy - sell,
    ]
    prob = cp.Problem(cp.Maximize((1 - gamma) * cp.sum(R @ w) / S - gamma * cvar
                                  - head["cost_rate"] * cp.sum(buy + sell)), constraints)
    start = time.perf_counter()
    prob.solve(solver=cp.HIGHS, time_limit=time_limit, mip_rel_gap=MIP_REL_GAP)
    seconds = time.perf_counter() - start
    if w.value is None:
        return None, prob.status, seconds
    # Positions the solver left at round-off are zero; the rest are renormalised so
    # the budget holds exactly. The objective is then recomputed on these weights,
    # so whatever the cleaning changes is in the reported number.
    weights = np.clip(np.asarray(w.value, dtype=float), 0.0, None)
    weights[weights < 1e-9] = 0.0
    weights /= weights.sum()
    return weights, prob.status, seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="artifacts/instances")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import cvxpy as cp
    import highspy

    directory = ROOT / args.dir
    paths = sorted(p for p in directory.glob("*.json") if p.name != "index.json")
    if args.only:
        paths = [p for p in paths if p.stem in set(args.only)]

    failures = 0
    for path in paths:
        head = json.loads(path.read_text(encoding="utf-8"))
        data = np.load(path.with_suffix(".npz"))
        R = data["scenarios"].astype(float)
        w_prev = data["w_prev"].astype(float)
        weights, status, seconds = solve(head, R, w_prev, args.time_limit)

        for key in ("best_known", "best_weights", "proven", "status", "solver",
                    "mip_rel_gap", "solve_seconds"):
            head.pop(key, None)
        if weights is None or not feasible(weights, head):
            failures += 1
            log.warning("%s: no feasible solution (%s)", path.stem, status)
            head.update(best_known=None, proven=False, status=str(status))
        else:
            head.update(
                best_known=objective(weights, head, R, w_prev),
                best_weights=[round(float(x), 12) for x in weights],
                proven=status == "optimal",
                status=str(status),
            )
        head.update(
            solver=f"HiGHS {highspy.__version__ if hasattr(highspy, '__version__') else ''} "
                   f"via CVXPY {cp.__version__}".replace("  ", " "),
            mip_rel_gap=MIP_REL_GAP,
            solve_seconds=round(seconds, 3),
        )
        path.write_text(json.dumps(head, indent=2), encoding="utf-8")
        log.info("%-40s %-10s %8.2fs  %s", path.stem, status, seconds, head.get("best_known"))

    manifest = [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(directory.glob("*.json")) if p.name != "index.json"]
    (directory / "index.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    proven = sum(1 for m in manifest if m.get("proven"))
    log.info("%d instances, %d proven, %d without a feasible solution",
             len(manifest), proven, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
