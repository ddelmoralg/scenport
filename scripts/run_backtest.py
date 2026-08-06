#!/usr/bin/env python
"""Walk-forward backtest over a grid of allocators, replaying stored scenarios.

Allocation is cheap next to simulation, so one job covers every allocator and seed
for a single forecaster.

    python scripts/run_backtest.py --forecaster prophet --allocator all --seeds 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenport import alloc  # noqa: E402
from scenport import config as cfgmod
from scenport.backtest import engine, metrics  # noqa: E402
from scenport.data.panel import Panel  # noqa: E402
from scenport.data.universe import read as read_universe  # noqa: E402
from scenport.forecast.store import Replay, ScenarioStore  # noqa: E402

log = logging.getLogger("backtest")

DEFAULT_ALLOCATORS = (
    *alloc.HEURISTIC,
    *alloc.CONVEX,
    *alloc.EVOLUTIONARY,
)


def build_allocator(name: str, cfg) -> alloc.Allocator:
    """Instantiate an allocator with its slice of the config."""
    shared = {
        "pop_size": cfg["ea.pop_size"],
        "generations": cfg["ea.generations"],
        "patience": cfg["ea.patience"],
        "tol": cfg["ea.tol"],
    }
    per_algorithm = {
        "ga": cfg.get("ea.ga", {}),
        "de": cfg.get("ea.de", {}),
        "pso": cfg.get("ea.pso", {}),
        "cmaes": cfg.get("ea.cmaes", {}),
    }
    if name in per_algorithm:
        return alloc.get(name, **shared, **per_algorithm[name])
    return alloc.get(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--forecaster", required=True)
    parser.add_argument("--forecast-seed", type=int, default=0)
    parser.add_argument("--allocator", nargs="+", default=["all"])
    parser.add_argument("--seeds", type=int, default=None,
                        help="number of seeds for stochastic allocators (default: ea.n_seeds)")
    parser.add_argument("--tag", default="", help="suffix for the output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = cfgmod.repo_root()
    cfg = cfgmod.load(args.config, args.overrides)

    panel_dir = cfg.resolve("paths.panel", root)
    runs_dir = cfg.resolve("paths.runs", root)
    panel = Panel.read(panel_dir)
    universe = read_universe(panel_dir / "universe.parquet")

    rf_path = panel_dir / "risk_free.parquet"
    rf_annual = pd.read_parquet(rf_path).iloc[:, 0] if rf_path.exists() else None
    rf_daily = (1.0 + rf_annual) ** (1 / 365) - 1.0 if rf_annual is not None else 0.0

    store = ScenarioStore(runs_dir / "scenarios", args.forecaster, args.forecast_seed)
    if not store.dates():
        log.error("no scenarios stored for %s seed %d", args.forecaster, args.forecast_seed)
        return 1
    replay = Replay(store)

    names = list(DEFAULT_ALLOCATORS) if args.allocator == ["all"] else args.allocator
    n_seeds = args.seeds if args.seeds is not None else int(cfg["ea.n_seeds"])

    problem_kwargs = {
        "w_max": cfg["alloc.w_max"],
        "w_min": cfg["alloc.w_min"],
        "cardinality": cfg["alloc.cardinality"],
        "cvar_alpha": cfg["alloc.cvar_alpha"],
        "gamma": cfg["alloc.gamma"],
        "objective": cfg.get("alloc.objective", "mean_cvar"),
    }

    out_root = runs_dir / "backtests" / (args.forecaster + (f"_{args.tag}" if args.tag else ""))
    summary_rows = []

    for name in names:
        allocator = build_allocator(name, cfg)
        seeds = range(n_seeds) if allocator.stochastic else [0]
        for seed in seeds:
            label = f"{args.forecaster}+{name}" + (f"+s{seed}" if allocator.stochastic else "")
            log.info("running %s", label)
            result = engine.run(
                panel,
                universe,
                replay,
                allocator,
                horizon=cfg["backtest.horizon_days"],
                train_days=cfg["backtest.train_days"],
                n_paths=cfg["scenarios.n_paths"],
                coupling=cfg["scenarios.coupling"],
                coupling_window=cfg["scenarios.coupling_window"],
                cost_bps=cfg["backtest.cost_bps"],
                rf_annual=rf_annual,
                seed=seed,
                problem_kwargs=problem_kwargs,
                label=label,
            )
            directory = out_root / f"{name}_seed{seed:02d}"
            result.write(directory)
            (directory / "meta.json").write_text(json.dumps(result.meta, default=str, indent=2))

            stats = metrics.summary(
                result.daily, rf_daily, periods_per_year=cfg["backtest.annualisation"]
            )
            summary_rows.append(
                {
                    "forecaster": args.forecaster,
                    "allocator": name,
                    "seed": seed,
                    "label": label,
                    "mean_turnover": float(result.periods["turnover"].mean())
                    if len(result.periods) else float("nan"),
                    "total_cost": float(result.periods["cost"].sum())
                    if len(result.periods) else float("nan"),
                    "mean_seconds": float(result.periods["seconds"].mean())
                    if "seconds" in result.periods else float("nan"),
                    **stats,
                }
            )
            log.info(
                "  sharpe %.3f  ann.ret %.1f%%  mdd %.1f%%  turnover %.2f",
                stats["sharpe"], 100 * stats["ann_return"], 100 * stats["max_drawdown"],
                summary_rows[-1]["mean_turnover"],
            )

    summary_dir = runs_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    target = summary_dir / f"{args.forecaster}{('_' + args.tag) if args.tag else ''}.parquet"
    pd.DataFrame(summary_rows).to_parquet(target, index=False)
    log.info("wrote %d rows to %s", len(summary_rows), target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
