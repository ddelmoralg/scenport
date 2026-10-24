#!/usr/bin/env python
"""Skew and within-date variance share of each diagnostic in the transmission panel.

    python scripts/panel_diagnostics.py

The paper explains which diagnostic survives the leave-one-date-out check by two
properties: how skewed it is, and how much of its variation lies within dates,
which is the variation the fixed-effects specification uses. This prints both for
all six diagnostics from the same panel the regression runs on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenport import config as cfgmod  # noqa: E402

import aggregate  # noqa: E402

DIAGNOSTICS = ("mase", "crps", "pinball", "coverage_error", "energy_score", "variogram_score")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    root = cfgmod.repo_root()
    cfg = cfgmod.load("configs/base.yaml", args.overrides)
    runs = cfg.resolve("paths.runs", root)
    panel_dir = cfg.resolve("paths.panel", root)

    per_asset, joint = aggregate.load_scores(runs, panel_dir, int(cfg["backtest.train_days"]))
    panel, _ = aggregate.build_transmission_panel(per_asset, joint, runs, "ga", panel_dir)
    print("panel rows:", len(panel), " dates:", panel["rebalance"].nunique())
    print()
    print("%-18s %8s %14s" % ("diagnostic", "skew", "within-date %"))
    rows = []
    for name in DIAGNOSTICS:
        if name not in panel:
            continue
        x = panel[name].astype(float)
        within = x - panel.groupby("rebalance")[name].transform("mean")
        share = within.var() / x.var()
        rows.append({"diagnostic": name, "skew": x.skew(), "within_date_share": share})
        print("%-18s %8.2f %13.1f%%" % (name, x.skew(), 100 * share))

    out = cfg.resolve("paths.artifacts", root) / "panel_diagnostics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print()
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
