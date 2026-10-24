#!/usr/bin/env python
"""How far the portfolio moves when the whole pipeline is rerun with a new seed.

    python scripts/noise_floor.py

The noise floor that aggregate.py reports reruns the backtest with a new seed on
the stored scenario set of forecast seed 0. That redraws the coupling and the
search but not the marginal paths, so it leaves out the sampling noise of the
generator itself, which is part of what a between-generator comparison carries.

Here every replicate redraws everything. Replicate 0 is the main run: forecast
seed 0, backtest seed 0. Replicate r, for r from 1 to 4, refits and resimulates
the generator with forecast seed r and runs the backtest with seed r, which
redraws the coupling template and the search. The ten pairs of replicates per
generator and date give a floor measured the same way as the between-generator
displacement, so the ratio of the two is the quantity the paper reports.

Writes artifacts/noise_floor.csv (one row per generator) and
artifacts/noise_floor.json (the headline numbers, with a bootstrap interval over
dates for the ratio).
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import _known_forecasters, _pair_distance  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-6


def weights(directory: Path) -> pd.DataFrame | None:
    path = directory / "weights.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index)
    return frame.sort_index()


def pairs_by_date(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Every pair of weight histories, one row per pair and date."""
    pieces = [_pair_distance(a, b, TOL) for a, b in combinations(frames, 2)]
    pieces = [p for p in pieces if not p.empty]
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--replicates", default="runs/noise_floor")
    parser.add_argument("--allocator", default="ga")
    parser.add_argument("--n-replicates", type=int, default=5)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()

    runs = ROOT / args.runs
    reps = ROOT / args.replicates
    out = ROOT / args.out
    names = sorted(n for n in _known_forecasters()
                   if (runs / "backtests" / n / f"{args.allocator}_seed00").exists())

    search_only: dict[str, pd.DataFrame] = {}    # five backtest seeds, scenarios of seed 0
    full: dict[str, pd.DataFrame] = {}           # five replicates of the whole pipeline
    marginal: dict[str, pd.DataFrame] = {}       # same backtest seed, new marginal paths
    seed_runs: dict[str, list[pd.DataFrame]] = {}
    missing = []
    for name in names:
        same = [weights(runs / "backtests" / name / f"{args.allocator}_seed{s:02d}")
                for s in range(args.n_replicates)]
        seed_runs[name] = [w for w in same if w is not None]
        search_only[name] = pairs_by_date(seed_runs[name])

        replicate = [same[0]] + [
            weights(reps / "backtests" / name / f"{args.allocator}_seed{r:02d}")
            for r in range(1, args.n_replicates)
        ]
        if any(w is None for w in replicate):
            missing.append(name)
            continue
        full[name] = pairs_by_date(replicate)
        # Replicate r and main-run seed r share the backtest seed, hence the
        # coupling template and the search, and differ only in the marginal paths.
        marginal[name] = pd.concat(
            [_pair_distance(same[r], replicate[r], TOL) for r in range(1, args.n_replicates)],
            ignore_index=True,
        )

    if missing:
        print("replicates missing for:", ", ".join(missing))
        return 1

    between = []
    for left, right in combinations(names, 2):
        pieces = [_pair_distance(a, b, TOL) for a in seed_runs[left] for b in seed_runs[right]]
        joined = pd.concat([p for p in pieces if not p.empty], ignore_index=True)
        joined["a"], joined["b"] = left, right
        between.append(joined)
    between = pd.concat(between, ignore_index=True)

    rows = []
    for name in names:
        b = between[(between["a"] == name) | (between["b"] == name)]
        rows.append({
            "forecaster": name,
            "floor_search_l1": float(search_only[name]["l1"].mean()),
            "floor_search_jaccard": float(search_only[name]["jaccard"].mean()),
            "floor_full_l1": float(full[name]["l1"].mean()),
            "floor_full_jaccard": float(full[name]["jaccard"].mean()),
            "floor_marginal_l1": float(marginal[name]["l1"].mean()),
            "between_l1": float(b["l1"].mean()),
            "between_jaccard": float(b["jaccard"].mean()),
            "n_full_pairs": int(len(full[name])),
        })
    table = pd.DataFrame(rows)
    table["between_over_full"] = table["between_l1"] / table["floor_full_l1"]

    # Per-date means, so the interval resamples dates and keeps every pair and
    # generator of a resampled date together.
    def by_date(frames: dict[str, pd.DataFrame]) -> pd.Series:
        stacked = pd.concat([f.assign(g=g) for g, f in frames.items()], ignore_index=True)
        # Mean over pairs within a generator first, then over generators, which is
        # how the headline floor is averaged.
        return stacked.groupby(["rebalance", "g"])["l1"].mean().groupby("rebalance").mean()

    floor_full_d = by_date(full)
    floor_search_d = by_date(search_only)
    between_d = between.groupby("rebalance")["l1"].mean()
    dates = floor_full_d.index.intersection(between_d.index)
    floor_full_d, floor_search_d, between_d = (
        floor_full_d[dates], floor_search_d[dates], between_d[dates])

    rng = np.random.default_rng(0)
    ratios_full, ratios_search = [], []
    for _ in range(args.boot):
        pick = rng.integers(0, len(dates), len(dates))
        ratios_full.append(between_d.iloc[pick].mean() / floor_full_d.iloc[pick].mean())
        ratios_search.append(between_d.iloc[pick].mean() / floor_search_d.iloc[pick].mean())

    floor_full = float(table["floor_full_l1"].mean())
    floor_search = float(table["floor_search_l1"].mean())
    mean_between = float(between["l1"].mean())
    head = {
        "allocator": args.allocator,
        "n_generators": len(names),
        "n_dates": int(len(dates)),
        "replicates": args.n_replicates,
        "floor_search_l1": floor_search,
        "floor_full_l1": floor_full,
        "floor_marginal_l1": float(table["floor_marginal_l1"].mean()),
        "floor_full_jaccard": float(table["floor_full_jaccard"].mean()),
        "floor_search_jaccard": float(table["floor_search_jaccard"].mean()),
        "between_l1": mean_between,
        "between_jaccard": float(between["jaccard"].mean()),
        "ratio_search": mean_between / floor_search,
        "ratio_full": mean_between / floor_full,
        "ratio_full_ci95": [float(np.quantile(ratios_full, 0.025)),
                            float(np.quantile(ratios_full, 0.975))],
        "ratio_search_ci95": [float(np.quantile(ratios_search, 0.025)),
                              float(np.quantile(ratios_search, 0.975))],
        "share_dates_between_above_full": float((between_d > floor_full_d).mean()),
        "min_generator_ratio_full": float(table["between_over_full"].min()),
        "max_generator_ratio_full": float(table["between_over_full"].max()),
    }

    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "noise_floor.csv", index=False)
    (out / "noise_floor.json").write_text(json.dumps(head, indent=2))
    print(table.round(3).to_string(index=False))
    print(json.dumps(head, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
