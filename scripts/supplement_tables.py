#!/usr/bin/env python
"""Tables for the supplementary material, from the per-run summaries.

    python scripts/supplement_tables.py

Writes two files to artifacts/:

    supplement_performance.csv   realised performance of the reference allocator
                                 for each generator, with 1/N alongside
    supplement_ablations.csv     the one-at-a-time sensitivity runs for AR-GARCH,
                                 against the base configuration

Every number is a mean over the seeds of the run, which is five for the
stochastic allocators and one for the deterministic ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenport import config as cfgmod  # noqa: E402
from scenport.plots import style  # noqa: E402

REFERENCE = "ga"
METRICS = ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "cvar_95",
           "mean_turnover"]

# What each ablation changes, as set in slurm/31_ablations.sh.
ABLATIONS = {
    "gamma00": "risk weight gamma = 0",
    "gamma25": "risk weight gamma = 0.25",
    "gamma75": "risk weight gamma = 0.75",
    "gamma100": "risk weight gamma = 1",
    "card05": "cardinality K = 5",
    "card10": "cardinality K = 10",
    "card15": "cardinality K = 15",
    "wmax10": "cap w_max = 0.10, with K = 12",
    "wmax33": "cap w_max = 0.33",
    "wmax100": "no cap, no minimum position",
    "cost00": "cost 0 bp",
    "cost50": "cost 50 bp",
    "cost100": "cost 100 bp",
    "alpha90": "tail level alpha = 0.90",
    "alpha99": "tail level alpha = 0.99",
    "nocouple": "no coupling",
    "empirical": "empirical copula coupling",
    "sharpeobj": "Sharpe objective",
}


def load(summary_dir: Path, forecaster: str, allocator: str) -> pd.Series | None:
    path = summary_dir / f"{forecaster}__{allocator}.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    row = frame[METRICS].mean()
    row["n_seeds"] = len(frame)
    return row


def main() -> int:
    root = cfgmod.repo_root()
    cfg = cfgmod.load("configs/base.yaml", [])
    runs = cfg.resolve("paths.runs", root)
    artifacts = cfg.resolve("paths.artifacts", root)
    summary = runs / "summary"

    rows = []
    for name in style.FORECASTER_ORDER:
        row = load(summary, name, REFERENCE)
        if row is not None:
            rows.append({"strategy": name, **row.to_dict()})
    # 1/N does not read the scenarios, so any generator's run gives the same answer.
    equal = load(summary, "garch", "equal")
    if equal is not None:
        rows.append({"strategy": "equal", **equal.to_dict()})
    perf = pd.DataFrame(rows)
    perf.to_csv(artifacts / "supplement_performance.csv", index=False)

    base = load(summary, "garch", REFERENCE)
    abl = [{"variant": "base", "change": "base configuration", **base.to_dict()}]
    for key, change in ABLATIONS.items():
        row = load(summary, f"garch_{key}", REFERENCE)
        if row is not None:
            abl.append({"variant": key, "change": change, **row.to_dict()})
    abl = pd.DataFrame(abl)
    abl["sharpe_change"] = abl["sharpe"] - float(base["sharpe"])
    abl.to_csv(artifacts / "supplement_ablations.csv", index=False)

    print("supplement_performance.csv: %d rows" % len(perf))
    print("supplement_ablations.csv:   %d rows" % len(abl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
