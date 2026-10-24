#!/usr/bin/env python
"""Export the allocation instances as a standalone benchmark set.

    python scripts/export_instances.py --families generators scale constraints
    python scripts/solve_instances.py

Each instance is the scenario matrix plus the constraint parameters, which is
everything an optimiser needs. No forecaster, no price panel, no dependency on this
package: an npz and a JSON header, readable from anything.

The set is a deliberate grid, not whatever the backtest happened to produce. Three
families:

  generators   every fitted generator, on shared dates, at the point-in-time
               universe size. Same assets and same dates throughout, so the only
               thing that differs between instances is the tail and dependence
               structure the generator produced. This is the diversity that
               existing portfolio benchmarks do not have.
  scale        larger universes drawn from the panel by market capitalisation, with
               scenarios resampled from realised horizon returns. This is the size
               range, and it goes past what is already out there.
  constraints  one base instance per family with the cardinality, the cap, the risk
               aversion and the tail fraction moved one at a time.

Why the set exists at all: the OR-Library portfolio files give per-asset means and
standard deviations and a correlation matrix. Moments, not scenarios. Running a CVaR
objective on them means inventing a distribution consistent with those moments, and
that choice is usually not reported, so gaps from two papers are not comparable.
SIPLIB's portfolio entry is chance-constrained and carries no tail risk measure,
cardinality or semicontinuous variables.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenport import config as cfgmod  # noqa: E402
from scenport.alloc.exact import subsample  # noqa: E402
from scenport.alloc.objective import Problem  # noqa: E402
from scenport.data.panel import Panel  # noqa: E402
from scenport.data.universe import read as read_universe  # noqa: E402
from scenport.forecast.store import ScenarioStore  # noqa: E402
from scenport.plots import style  # noqa: E402

log = logging.getLogger("export")

FAMILIES = ("generators", "scale", "constraints")

README = """\
# scenport allocation instances

Cardinality-constrained mean-CVaR portfolio selection over simulated scenarios.

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

with CVaR written as a positive loss in the Rockafellar-Uryasev form,

    CVaR_alpha = min_z  z + sum_s max(0, -(R w)_s - z) / ((1 - alpha) S),

which is the mean of the worst (1 - alpha) S scenario returns when that number is
whole and gives the last tail scenario a fractional weight when it is not (S = 250
at alpha = 0.95, for example).

Each header carries the answer we have for its own scenarios, with no thinning:

    best_known     objective of best_weights, recomputed from the npz
    best_weights   the portfolio that attains it, in the order of `assets`
    proven         HiGHS reported optimality within mip_rel_gap
    solver, mip_rel_gap, solve_seconds, status

so a result can be checked by evaluating best_weights, without trusting the number.
Improvements welcome; open an issue with the instance name, the weights and how you
got them.

## Families

`family` in the header says how the scenarios were produced.

- `generators`: simulated by one of the fitted forecasters named in `generator`, on
  the point-in-time universe for `rebalance_date`. Instances sharing a date share
  their assets and their carried-in weights, so they differ only in the scenario
  distribution.
- `scale`: assets drawn from the panel by market capitalisation on `rebalance_date`,
  with scenarios resampled in blocks from realised horizon returns. Use these for
  solver scaling. They are not forecasts and should not be read as any.
- `constraints`: a base instance with one constraint parameter moved. `varied` names
  which one.

## Reading one

    import json, numpy as np
    head = json.load(open("name.json"))
    data = np.load("name.npz")
    R, w_prev = data["scenarios"], data["w_prev"]

## What it covers, and what it does not

The `generators` family is at the point-in-time universe size, which is fifteen.
The `scale` family goes as far as the price panel allows, and that is the binding
limit: the panel holds 76 assets, of which about 71 carry a market capitalisation
on any given date, so instances stop around seventy. The OR-Library reaches 225.

That ceiling is the data source rather than the code. Price and market
capitalisation come from the Coin Metrics Community tier, which serves both
metrics for 87 assets; a licensed tier would lift it and the export takes the
larger universe without changes.

So this set is more varied than what came before it and not yet larger. Every
instance carries the scenario matrix rather than moments, which is the part that
makes gaps comparable at all, and eleven fitted generators give eleven different
tail and dependence structures over the same assets and the same dates.

## Why this set exists

The OR-Library portfolio files, which is what this problem class has been
benchmarked on since 2000, give per-asset means and standard deviations and a
correlation matrix. Moments, not scenarios. A CVaR objective cannot be evaluated on
moments, so anyone using them for this problem has to invent a distribution
consistent with them, and that choice is usually not reported. Two papers quoting
optimality gaps on port1 are therefore not necessarily quoting them on the same
instance.

SIPLIB does not fill the gap either. Its portfolio entry, PROBPORT, is
chance-constrained, and nothing in the collection carries a tail risk measure with
cardinality and semicontinuous variables.

These instances ship the scenario matrix itself, so the objective is unambiguous and
gaps are comparable.
"""


def _write(out: Path, name: str, instance: Problem, header: dict) -> None:
    np.savez_compressed(
        out / f"{name}.npz",
        scenarios=instance.scenarios.astype(np.float32),
        w_prev=instance.w_prev.astype(np.float32),
    )
    full = {
        "name": name,
        "n_assets": instance.n,
        "n_scenarios": instance.n_scenarios,
        "assets": list(instance.assets),
        "w_max": instance.w_max,
        "w_min": instance.w_min,
        "cardinality": instance.cardinality,
        "cvar_alpha": instance.cvar_alpha,
        "gamma": instance.gamma,
        "cost_rate": instance.cost_rate,
        "objective": instance.objective,
    }
    full.update(header)
    full.setdefault("best_known", None)
    full.setdefault("proven", False)
    (out / f"{name}.json").write_text(json.dumps(full, indent=2), encoding="utf-8")


def _base_kwargs(cfg) -> dict:
    return {
        "w_max": cfg["alloc.w_max"],
        "w_min": cfg["alloc.w_min"],
        "cardinality": cfg["alloc.cardinality"],
        "cvar_alpha": cfg["alloc.cvar_alpha"],
        "gamma": cfg["alloc.gamma"],
        "cost_rate": cfg["backtest.cost_bps"] / 10_000.0,
    }


def family_generators(args, cfg, out, panel_dir, runs_dir) -> int:
    """Every generator on the same dates and the same assets."""
    universe = read_universe(panel_dir / "universe.parquet")
    dates = list(universe.dates[:: args.every])[: args.n_dates]
    base = _base_kwargs(cfg)
    written = 0

    for forecaster in args.generators:
        store = ScenarioStore(runs_dir / "scenarios", forecaster, args.forecast_seed)
        for date in dates:
            if not store.exists(date):
                continue
            assets = universe.at(date)
            terminal = store.load(date).reorder(assets).terminal()
            for n_scenarios in args.scenario_counts:
                instance = subsample(
                    Problem(scenarios=terminal, assets=list(assets), **base),
                    n_scenarios,
                    np.random.default_rng(cfgmod.stable_seed("export", forecaster, str(date))),
                )
                name = (
                    f"gen_{forecaster}_{date.date()}"
                    f"_n{instance.n}_S{instance.n_scenarios}"
                )
                _write(out, name, instance, {
                    "family": "generators",
                    "generator": forecaster,
                    "rebalance_date": str(date.date()),
                })
                written += 1
    return written


def _historical_instance(panel: Panel, cfg, date, size, n_scenarios, rng, base) -> Problem | None:
    """Larger universe with scenarios resampled from realised horizon returns.

    Extra assets come off the panel by market capitalisation rather than being
    simulated, because a synthetic block of independent assets would make the
    problem artificially easy and the scaling numbers meaningless.
    """
    history = panel.price.loc[:date].iloc[-(cfg["backtest.train_days"] + 1) : -1]
    liquid = panel.mcap.loc[date].dropna().sort_values(ascending=False)
    assets = [a for a in liquid.index[: size * 3] if history[a].notna().sum() > 400][:size]
    if len(assets) < size:
        # Clamp rather than skip. Skipping left the library with a gap at the top
        # instead of its largest feasible instance, which is the one a scaling
        # study most wants. The panel here holds 76 assets and 71 carry a market
        # capitalisation on a given date, so anything past about 70 is the data
        # talking, not the code.
        log.warning(
            "universe of %d requested, %d assets available; exporting at %d",
            size, len(assets), len(assets),
        )
        if len(assets) < 10:
            return None

    horizon = cfg["backtest.horizon_days"]
    block = history[assets].pct_change(horizon).dropna().to_numpy()
    if len(block) < 30:
        return None
    idx = rng.integers(0, len(block), size=n_scenarios)
    return Problem(scenarios=block[idx], assets=list(assets), **base)


def family_scale(args, cfg, out, panel_dir) -> int:
    panel = Panel.read(panel_dir)
    date = pd.Timestamp(args.scale_date) if args.scale_date else panel.dates[-400]
    base = _base_kwargs(cfg)
    written = 0

    seen: set[tuple[int, int]] = set()
    for size in args.sizes:
        for n_scenarios in args.scenario_counts:
            rng = np.random.default_rng(cfgmod.stable_seed("scale", size, n_scenarios))
            instance = _historical_instance(panel, cfg, date, size, n_scenarios, rng, base)
            if instance is None:
                continue
            # Several requested sizes clamp to the same achievable one, so only
            # the first of them is written.
            if (instance.n, n_scenarios) in seen:
                continue
            seen.add((instance.n, n_scenarios))
            # The cardinality has to grow with the universe or every large instance
            # is the same tiny selection problem wearing a bigger hat.
            instance.cardinality = max(cfg["alloc.cardinality"], int(round(size * 0.2)))
            instance.w_max = max(cfg["alloc.w_max"], 1.5 / instance.cardinality)
            name = f"scale_{date.date()}_n{instance.n}_S{instance.n_scenarios}"
            _write(out, name, instance, {
                "family": "scale",
                "generator": "historical_bootstrap",
                "rebalance_date": str(date.date()),
            })
            written += 1
    return written


def family_constraints(args, cfg, out, panel_dir, runs_dir) -> int:
    """One base instance with each constraint parameter moved on its own."""
    universe = read_universe(panel_dir / "universe.parquet")
    store = ScenarioStore(runs_dir / "scenarios", args.generators[0], args.forecast_seed)
    date = next((d for d in universe.dates[:: args.every] if store.exists(d)), None)
    if date is None:
        log.warning("no stored scenarios for the constraint family")
        return 0

    assets = universe.at(date)
    terminal = store.load(date).reorder(assets).terminal()
    base = _base_kwargs(cfg)
    written = 0

    variations = (
        [("cardinality", k) for k in args.cardinalities]
        + [("w_max", w) for w in args.caps]
        + [("gamma", g) for g in args.gammas]
        + [("cvar_alpha", a) for a in args.alphas]
    )

    for key, value in variations:
        kwargs = dict(base)
        kwargs[key] = value
        # The cap and the cardinality have to multiply to at least one or the
        # portfolio cannot be fully invested and the instance has no feasible
        # point. Whichever of the two is being varied, the other follows.
        if key == "cardinality":
            kwargs["w_max"] = max(base["w_max"], 1.05 / value)
        elif key == "w_max" and base["cardinality"] is not None:
            needed = int(np.ceil(1.0 / value))
            if needed > base["cardinality"]:
                kwargs["cardinality"] = needed + 2
        try:
            problem = Problem(scenarios=terminal, assets=list(assets), **kwargs)
        except ValueError as exc:
            log.warning("skipping %s=%s: %s", key, value, exc)
            continue
        instance = subsample(
            problem,
            args.scenario_counts[-1],
            np.random.default_rng(cfgmod.stable_seed("constraints", key, value)),
        )
        tag = str(value).replace(".", "p")
        name = f"cons_{key}_{tag}_n{instance.n}_S{instance.n_scenarios}"
        _write(out, name, instance, {
            "family": "constraints",
            "generator": args.generators[0],
            "rebalance_date": str(date.date()),
            "varied": key,
        })
        written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    # All eleven generators, so the library covers every one the paper reports.
    parser.add_argument("--generators", nargs="+", default=list(style.FORECASTER_ORDER))
    parser.add_argument("--forecast-seed", type=int, default=0)
    parser.add_argument("--every", type=int, default=6)
    parser.add_argument("--n-dates", type=int, default=4)
    parser.add_argument("--scenario-counts", type=int, nargs="+", default=[250, 1000])
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[30, 50, 75, 100, 150, 225])
    parser.add_argument("--scale-date", default=None)
    parser.add_argument("--cardinalities", type=int, nargs="+", default=[5, 12, 20])
    parser.add_argument("--caps", type=float, nargs="+", default=[0.10, 0.33, 1.0])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.25, 1.0])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.90, 0.99])
    parser.add_argument("--out", default="artifacts/instances")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = cfgmod.repo_root()
    cfg = cfgmod.load(args.config, args.overrides)
    panel_dir = cfg.resolve("paths.panel", root)
    runs_dir = cfg.resolve("paths.runs", root)

    out = (root / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(README, encoding="utf-8")

    # Bounds are not filled in here. They used to be borrowed from the gap study,
    # which solved one generator on a 400-scenario subsample of each date, and that
    # number then labelled every instance on the date. solve_instances.py solves
    # each instance on its own scenarios instead; run it after this.
    counts = {}
    if "generators" in args.families:
        counts["generators"] = family_generators(args, cfg, out, panel_dir, runs_dir)
    if "scale" in args.families:
        counts["scale"] = family_scale(args, cfg, out, panel_dir)
    if "constraints" in args.families:
        counts["constraints"] = family_constraints(args, cfg, out, panel_dir, runs_dir)

    total = sum(counts.values())
    for family, n in counts.items():
        log.info("%-12s %3d instances", family, n)
    log.info("wrote %d instances to %s", total, out)

    manifest = []
    for path in sorted(out.glob("*.json")):
        # Skip the index itself, which the glob would otherwise read back in as
        # one more entry.
        if path.name == "index.json":
            continue
        manifest.append(json.loads(path.read_text(encoding="utf-8")))
    (out / "index.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
