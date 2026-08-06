#!/usr/bin/env python
"""Download the raw data and build the price panel and the point-in-time universe.

Needs network access. Everything downstream reads the parquet files this writes.

    python scripts/fetch_data.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenport import config as cfgmod  # noqa: E402
from scenport.data import coinmetrics, panel, universe  # noqa: E402

log = logging.getLogger("fetch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--assets", nargs="*", help="restrict the download to these tickers")
    parser.add_argument("--skip-rf", action="store_true", help="do not fetch the T-bill series")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = cfgmod.repo_root()
    cfg = cfgmod.load(args.config, args.overrides)

    raw_dir = cfg.resolve("paths.raw", root)
    panel_dir = cfg.resolve("paths.panel", root)
    metrics = list(cfg["data.metrics"])

    session = coinmetrics._Session()
    assets = args.assets or coinmetrics.available_assets(metrics, session)
    log.info("%d assets expose %s at daily frequency", len(assets), " and ".join(metrics))

    raw = coinmetrics.fetch(
        assets,
        metrics,
        start=cfg["data.start"],
        end=cfg["data.end"],
        cache_dir=raw_dir / "coinmetrics",
        session=session,
    )
    log.info("downloaded %d rows", len(raw))

    built = panel.build(
        raw,
        price_col=metrics[0],
        mcap_col=metrics[1],
        exclude=cfg["data.exclude"],
        peg_window=cfg["data.peg_detection.window"],
        peg_max_cv=cfg["data.peg_detection.max_cv"],
    )
    built.write(panel_dir)
    log.info("panel: %d dates x %d assets -> %s", *built.price.shape, panel_dir)

    dates = universe.rebalance_dates(
        built,
        first=cfg["backtest.first_rebalance"],
        last=cfg["backtest.last_rebalance"],
        freq=cfg["backtest.rebalance"],
    )
    selected = universe.build(
        built,
        dates,
        size=cfg["universe.size"],
        min_history_days=cfg["universe.min_history_days"],
        rank_window=cfg["universe.rank_window"],
    )
    selected.as_frame().to_parquet(panel_dir / "universe.parquet")

    panel.summarise(built).to_csv(panel_dir / "asset_summary.csv", index=False)
    universe.coverage_report(selected).to_csv(panel_dir / "universe_coverage.csv")
    selected.turnover().to_csv(panel_dir / "universe_turnover.csv")

    log.info(
        "universe: %d rebalancing dates, %d distinct assets ever selected, mean turnover %.1f%%",
        len(dates),
        selected.as_frame()["asset"].nunique(),
        100 * selected.turnover().mean(),
    )

    if not args.skip_rf:
        from scenport.data.benchmarks import risk_free

        rf = risk_free(cfg["data.start"], cfg["data.end"], cache=raw_dir / "fred_dtb3.csv")
        rf.to_frame().to_parquet(panel_dir / "risk_free.parquet")
        log.info("risk-free series: %d days, last value %.4f", len(rf), rf.iloc[-1])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
