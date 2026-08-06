#!/usr/bin/env python
"""Export the allocation instances as a standalone benchmark set.

    python scripts/export_instances.py --forecaster garch --every 3

Each instance is the scenario matrix plus the constraint parameters, which is
everything an optimiser needs. No forecaster, no price panel, no dependency on this
package -- an npz and a JSON header, readable from anything.

The point is that somebody comparing a new solver against ours should not have to
rerun a Prophet sweep first.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenport import config as cfgmod  # noqa: E402
from scenport.alloc.exact import subsample  # noqa: E402
from scenport.alloc.objective import Problem  # noqa: E402
from scenport.data.universe import read as read_universe  # noqa: E402
from scenport.forecast.store import ScenarioStore  # noqa: E402

log = logging.getLogger("export")

README = """\
# scenport allocation instances

Each instance is a pair of files:

    <name>.npz     scenarios: (S, n) float32, horizon simple returns
                   w_prev:    (n,)  float32, weights carried in
    <name>.json    constraint parameters and provenance

The problem is

    max  (1-gamma) mean(R w) - gamma CVaR_alpha(R w) - cost_rate ||w - w_prev||_1
    s.t. sum(w) = 1
         0 <= w_i <= w_max
         w_i = 0 or w_i >= w_min
         ||w||_0 <= cardinality

with CVaR written as a positive loss: the mean of the worst (1 - alpha) fraction of
the S scenario returns.

`best_known` in the JSON is the best objective value we have found, and `proven`
says whether a mixed-integer solver closed it. Improvements welcome.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--forecaster", default="garch")
    parser.add_argument("--forecast-seed", type=int, default=0)
    parser.add_argument("--every", type=int, default=3)
    parser.add_argument("--max-scenarios", type=int, default=400)
    parser.add_argument("--out", default="artifacts/instances")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = cfgmod.repo_root()
    cfg = cfgmod.load(args.config, args.overrides)

    panel_dir = cfg.resolve("paths.panel", root)
    runs_dir = cfg.resolve("paths.runs", root)
    universe = read_universe(panel_dir / "universe.parquet")
    store = ScenarioStore(runs_dir / "scenarios", args.forecaster, args.forecast_seed)

    out = (root / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(README, encoding="utf-8")

    # Whatever the gap study already solved, so the headers can carry a bound.
    solved = {}
    gaps_path = runs_dir / "exact" / "gaps.parquet"
    if gaps_path.exists():
        import pandas as pd

        gaps = pd.read_parquet(gaps_path)
        best = gaps.groupby(["rebalance", "allocator"])["objective"].max().unstack()
        for date, row in best.iterrows():
            solved[date] = {
                "best_known": float(row.max()),
                "proven": bool("milp" in row.index and row.idxmax() == "milp"),
            }

    written = 0
    for date in universe.dates[:: args.every]:
        if not store.exists(date):
            continue
        assets = universe.at(date)
        scenarios = store.load(date).reorder(assets)
        instance = subsample(
            Problem(
                scenarios=scenarios.terminal(),
                assets=list(assets),
                w_max=cfg["alloc.w_max"],
                w_min=cfg["alloc.w_min"],
                cardinality=cfg["alloc.cardinality"],
                cvar_alpha=cfg["alloc.cvar_alpha"],
                gamma=cfg["alloc.gamma"],
                cost_rate=cfg["backtest.cost_bps"] / 10_000.0,
            ),
            args.max_scenarios,
            np.random.default_rng(0),
        )

        name = f"{args.forecaster}_{date.date()}_n{instance.n}_S{instance.n_scenarios}"
        np.savez_compressed(
            out / f"{name}.npz",
            scenarios=instance.scenarios.astype(np.float32),
            w_prev=instance.w_prev.astype(np.float32),
        )
        header = {
            "name": name,
            "n_assets": instance.n,
            "n_scenarios": instance.n_scenarios,
            "assets": instance.assets,
            "w_max": instance.w_max,
            "w_min": instance.w_min,
            "cardinality": instance.cardinality,
            "cvar_alpha": instance.cvar_alpha,
            "gamma": instance.gamma,
            "cost_rate": instance.cost_rate,
            "rebalance_date": str(date.date()),
            "forecaster": args.forecaster,
        }
        header.update(solved.get(date, {"best_known": None, "proven": False}))
        (out / f"{name}.json").write_text(json.dumps(header, indent=2), encoding="utf-8")
        written += 1

    log.info("wrote %d instances to %s", written, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
