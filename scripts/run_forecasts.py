#!/usr/bin/env python
"""Simulate and score the scenario sets for one forecaster.

Shard across a SLURM array with --shard/--n-shards; shards are interleaved so
every task gets a mix of early and late dates and they finish together.

    python scripts/run_forecasts.py --forecaster prophet --shard 0 --n-shards 16
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenport import config as cfgmod  # noqa: E402
from scenport import scoring
from scenport.data.panel import Panel  # noqa: E402
from scenport.data.universe import read as read_universe  # noqa: E402
from scenport.forecast import get_any  # noqa: E402
from scenport.forecast.coupling import apply as couple  # noqa: E402
from scenport.forecast.store import ScenarioStore, realised  # noqa: E402

log = logging.getLogger("forecast")

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def one_date(args_tuple) -> list[dict]:
    """Simulate, store and score a single rebalancing date. Runs in a worker."""
    (date, cfg_path, overrides, forecaster_name, seed, panel_dir, store_root,
     forecaster_params) = args_tuple

    cfg = cfgmod.load(cfg_path, overrides)
    panel = Panel.read(Path(panel_dir))
    universe = read_universe(Path(panel_dir) / "universe.parquet")

    horizon = int(cfg["backtest.horizon_days"])
    train_days = int(cfg["backtest.train_days"])
    n_paths = int(cfg["scenarios.n_paths"])
    coupling = cfg["scenarios.coupling"]
    coupling_window = cfg["scenarios.coupling_window"]

    assets = universe.at(date)
    history = panel.price.loc[:date, assets].iloc[-(train_days + 1) : -1]

    rng = np.random.default_rng(cfgmod.stable_seed(forecaster_name, date.date(), seed))
    model = get_any(forecaster_name, **forecaster_params)
    marginals = model.simulate(history, horizon, n_paths, rng, origin=date)

    store = ScenarioStore(Path(store_root), forecaster_name, seed)
    store.save(marginals)

    coupled = couple(marginals, history, method=coupling, rng=rng, window=coupling_window)
    truth = realised(panel.price, date, assets, horizon)

    rows = []
    terminal_scen = coupled.terminal()
    terminal_truth = truth[-1]
    in_sample = history.to_numpy(dtype=float)

    for j, asset in enumerate(assets):
        samples = terminal_scen[:, j]
        observed = terminal_truth[j]
        rows.append(
            {
                "rebalance": date,
                "forecaster": forecaster_name,
                "seed": seed,
                "asset": asset,
                "realised": float(observed),
                "median": float(np.median(samples)),
                "mean": float(samples.mean()),
                "sd": float(samples.std(ddof=1)),
                "abs_error": abs(float(np.median(samples)) - float(observed)),
                "crps": scoring.crps(samples, observed),
                "pinball": scoring.pinball(samples, observed, QUANTILES),
                "pit": scoring.pit(samples, observed, rng),
                "cover_80": scoring.interval_coverage(samples, observed, 0.80),
                "cover_50": scoring.interval_coverage(samples, observed, 0.50),
                "mase_scale": float(np.nanmean(np.abs(np.diff(in_sample[:, j])))),
            }
        )

    joint = {
        "rebalance": date,
        "forecaster": forecaster_name,
        "seed": seed,
        "asset": "__joint__",
        "energy_score": scoring.energy_score(terminal_scen, terminal_truth),
        "variogram_score": scoring.variogram_score(terminal_scen, terminal_truth),
        "coupling": coupling,
    }
    rows.append(joint)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--forecaster", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--param", action="append", default=[],
                        help="extra forecaster kwarg, key=value")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = cfgmod.repo_root()
    cfg = cfgmod.load(args.config, args.overrides)

    panel_dir = cfg.resolve("paths.panel", root)
    runs_dir = cfg.resolve("paths.runs", root)
    store_root = runs_dir / "scenarios"
    store = ScenarioStore(store_root, args.forecaster, args.seed)

    universe = read_universe(panel_dir / "universe.parquet")
    dates = universe.dates[args.shard :: args.n_shards]
    if not args.overwrite:
        pending = [d for d in dates if not store.exists(d)]
        if len(pending) < len(dates):
            log.info("skipping %d dates already on disk", len(dates) - len(pending))
        dates = pending
    if not dates:
        log.info("nothing to do")
        return 0

    forecaster_params = {
        key: cfgmod.coerce(value)
        for key, value in (item.split("=", 1) for item in args.param)
    }

    payload = [
        (d, str(Path(args.config).resolve()), args.overrides, args.forecaster, args.seed,
         str(panel_dir), str(store_root), forecaster_params)
        for d in dates
    ]

    log.info(
        "%s: %d dates on shard %d/%d, %d worker(s)",
        args.forecaster, len(dates), args.shard, args.n_shards, args.workers,
    )

    rows: list[dict] = []
    if args.workers <= 1:
        for i, item in enumerate(payload, 1):
            rows.extend(one_date(item))
            log.info("[%d/%d] %s done", i, len(payload), item[0].date())
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one_date, item): item[0] for item in payload}
            for i, future in enumerate(as_completed(futures), 1):
                rows.extend(future.result())
                log.info("[%d/%d] %s done", i, len(payload), futures[future].date())

    scores_dir = runs_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    target = scores_dir / f"{args.forecaster}_seed{args.seed:02d}_shard{args.shard:03d}.parquet"
    pd.DataFrame(rows).to_parquet(target, index=False)
    log.info("wrote %d score rows to %s", len(rows), target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
