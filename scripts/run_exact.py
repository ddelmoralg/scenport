#!/usr/bin/env python
"""Optimality gaps against the mixed-integer optimum, and how far it scales.

``gaps`` solves the same instance by branch and bound and by each metaheuristic.
``scaling`` grows the universe until the solver stops finishing.

    python scripts/run_exact.py gaps --forecaster garch --every 6
    python scripts/run_exact.py scaling --sizes 10 15 25 50 100 150
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenport import alloc  # noqa: E402
from scenport import config as cfgmod
from scenport.alloc.exact import MilpExact, subsample  # noqa: E402
from scenport.alloc.objective import Problem, feasible_cardinality, score  # noqa: E402
from scenport.data.panel import Panel  # noqa: E402
from scenport.data.universe import read as read_universe  # noqa: E402
from scenport.forecast.store import ScenarioStore  # noqa: E402

log = logging.getLogger("exact")

CANDIDATES = ("ga", "de", "pso", "cmaes", "cvar_lp", "hrp", "minvar", "equal")


def problem_kwargs(cfg) -> dict:
    return {
        "w_max": cfg["alloc.w_max"],
        "w_min": cfg["alloc.w_min"],
        "cardinality": cfg["alloc.cardinality"],
        "cvar_alpha": cfg["alloc.cvar_alpha"],
        "gamma": cfg["alloc.gamma"],
        "objective": "mean_cvar",
    }


def gaps(args, cfg, root) -> pd.DataFrame:
    panel_dir = cfg.resolve("paths.panel", root)
    runs_dir = cfg.resolve("paths.runs", root)
    universe = read_universe(panel_dir / "universe.parquet")
    store = ScenarioStore(runs_dir / "scenarios", args.forecaster, args.forecast_seed)

    dates = universe.dates[:: args.every]
    rows = []

    for date in dates:
        if not store.exists(date):
            continue
        assets = universe.at(date)
        scenarios = store.load(date).reorder(assets)
        base = Problem(
            scenarios=scenarios.terminal(),
            assets=list(assets),
            cost_rate=cfg["backtest.cost_bps"] / 10_000.0,
            **problem_kwargs(cfg),
        )
        instance = subsample(base, args.max_scenarios, np.random.default_rng(0))

        started = time.perf_counter()
        best = MilpExact(max_scenarios=args.max_scenarios, time_limit=args.time_limit)(
            instance, np.random.default_rng(0)
        )
        milp_seconds = time.perf_counter() - started
        milp_value = float(score(best.weights, instance)[0])

        rows.append(
            {
                "rebalance": date, "allocator": "milp", "seed": -1,
                "objective": milp_value, "seconds": milp_seconds,
                "relative_gap": 0.0, "status": best.diagnostics.get("status"),
                "kind": "exact",
            }
        )

        # The relaxation, recorded as what it is. Its answer holds more names than
        # the cardinality allows, so it is not a competitor and its distance above
        # the exact optimum is not a gap in the usual sense: it is the integrality
        # gap, which is what the discrete constraints cost.
        try:
            relaxed = alloc.get("cvar_lp_bound")(instance, np.random.default_rng(0))
            bound_value = float(score(relaxed.weights, instance)[0])
            denom = abs(milp_value) if abs(milp_value) > 1e-12 else 1.0
            rows.append(
                {
                    "rebalance": date, "allocator": "cvar_lp_bound", "seed": -1,
                    "objective": bound_value, "seconds": relaxed.seconds,
                    "relative_gap": (bound_value - milp_value) / denom,
                    "status": relaxed.diagnostics.get("status"),
                    "kind": "relaxation",
                }
            )
        except Exception as exc:  # the LP is optional, the gap study is not
            log.warning("%s: relaxation bound failed (%s)", date.date(), exc)

        for name in args.allocator:
            allocator = alloc.get(name) if name not in ("ga", "de", "pso", "cmaes") else alloc.get(
                name,
                pop_size=cfg["ea.pop_size"],
                generations=cfg["ea.generations"],
                patience=cfg["ea.patience"],
                **cfg.get(f"ea.{name}", {}),
            )
            seeds = range(args.seeds) if allocator.stochastic else [0]
            for seed in seeds:
                result = allocator(instance, np.random.default_rng(seed))
                value = float(score(result.weights, instance)[0])
                denom = abs(milp_value) if abs(milp_value) > 1e-12 else 1.0
                rows.append(
                    {
                        "rebalance": date, "allocator": name, "seed": seed,
                        "objective": value, "seconds": result.seconds,
                        "relative_gap": (milp_value - value) / denom,
                        "feasible": result.diagnostics.get("feasible"),
                        "kind": "candidate",
                    }
                )
        log.info("%s: milp %.6f in %.1fs", date.date(), milp_value, milp_seconds)

    return pd.DataFrame(rows)


def scaling(args, cfg, root) -> pd.DataFrame:
    """Grow the universe on a fixed date and watch the solver give up.

    The scenario set is widened by drawing extra assets from the panel rather than
    by simulating new ones, so the dependence structure stays realistic: a
    synthetic block of independent assets would make the problem artificially easy.
    """
    panel_dir = cfg.resolve("paths.panel", root)
    panel = Panel.read(panel_dir)
    date = pd.Timestamp(args.date) if args.date else panel.dates[-400]

    history = panel.price.loc[:date].iloc[-(cfg["backtest.train_days"] + 1) : -1]
    liquid = panel.mcap.loc[date].dropna().sort_values(ascending=False)
    rng = np.random.default_rng(args.seed)
    rows = []

    # One untimed solve first. The first call in a process pays for importing the
    # solver and building its data structures, and without this the smallest
    # universe came out an order of magnitude slower than the next one up, which
    # reads as a scaling result and is not one.
    warm = Problem(
        scenarios=rng.normal(0.0, 0.05, (64, 8)),
        assets=[f"warmup{i}" for i in range(8)],
        cost_rate=0.0,
        **problem_kwargs(cfg),
    )
    for name in ["milp", *args.allocator]:
        warm_alloc = (
            MilpExact(max_scenarios=args.max_scenarios, time_limit=args.time_limit)
            if name == "milp"
            else alloc.get(name)
        )
        try:
            warm_alloc(warm, np.random.default_rng(0))
        except Exception as exc:                                  # noqa: BLE001
            log.warning("warm-up for %s failed: %s", name, exc)

    seen: set[int] = set()
    for size in args.sizes:
        assets = [a for a in liquid.index[: size * 3] if history[a].notna().sum() > 400][:size]
        if len(assets) < size:
            # The panel runs out at about seventy names. Asking for more used to
            # run the sweep again on the same seventy and label the row with the
            # achieved size, so the file held four different experiments all
            # called n=70 and the figure drew a vertical line through them.
            log.warning("only %d assets available for a universe of %d; skipping",
                        len(assets), size)
            continue
        if len(assets) in seen:
            continue
        seen.add(len(assets))

        # Historical horizon returns stand in for a forecast here: the point of the
        # experiment is solver behaviour as n grows, not forecast quality.
        horizon = cfg["backtest.horizon_days"]
        block = history[assets].pct_change(horizon).dropna().to_numpy()
        idx = rng.integers(0, len(block), size=cfg["scenarios.n_paths"])
        instance = Problem(
            scenarios=block[idx],
            assets=assets,
            cost_rate=0.0,
            **{
                **problem_kwargs(cfg),
                # A third of the universe, but the weight bounds have the last
                # word: sixty-six names at a two per cent floor needs 132% of the
                # capital, which is how this sweep used to produce eight rows of
                # ValueError that the figure then plotted as solve times.
                "cardinality": feasible_cardinality(
                    max(cfg["alloc.cardinality"], size // 3),
                    len(assets),
                    problem_kwargs(cfg)["w_min"],
                    problem_kwargs(cfg)["w_max"],
                ),
            },
        )

        for name in ["milp", *args.allocator]:
            allocator = (
                MilpExact(max_scenarios=args.max_scenarios, time_limit=args.time_limit)
                if name == "milp"
                else alloc.get(name)
            )
            started = time.perf_counter()
            try:
                result = allocator(instance, np.random.default_rng(args.seed))
                value = float(score(result.weights, instance)[0])
                status = result.diagnostics.get("status", "ok")
            except Exception as exc:                              # noqa: BLE001
                log.warning("n=%d %s failed: %s", len(assets), name, exc)
                value, status = float("nan"), f"failed: {type(exc).__name__}"
            rows.append(
                {
                    "n_assets": len(assets), "allocator": name, "objective": value,
                    "seconds": time.perf_counter() - started, "status": status,
                }
            )
            log.info("n=%3d %-8s %8.2fs  obj %s", len(assets), name, rows[-1]["seconds"], value)

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["gaps", "scaling"])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--forecaster", default="rw")
    parser.add_argument("--forecast-seed", type=int, default=0)
    parser.add_argument("--allocator", nargs="+", default=list(CANDIDATES))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--every", type=int, default=6, help="use every k-th rebalancing date")
    parser.add_argument("--max-scenarios", type=int, default=400)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 15, 25, 50, 100])
    parser.add_argument("--date", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = cfgmod.repo_root()
    cfg = cfgmod.load(args.config, args.overrides)

    frame = gaps(args, cfg, root) if args.mode == "gaps" else scaling(args, cfg, root)

    out = cfg.resolve("paths.runs", root) / "exact"
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{args.mode}.parquet"
    frame.to_parquet(target, index=False)
    log.info("wrote %d rows to %s", len(frame), target)

    if args.mode == "gaps" and len(frame):
        pivot = frame.groupby("allocator")["relative_gap"].agg(["mean", "median", "max"])
        log.info("relative gap to the MILP optimum:\n%s", pivot.round(5).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
