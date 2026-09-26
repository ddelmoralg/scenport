"""Turn the raw Coin Metrics download into aligned price and market-cap panels."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MAX_GAP_DAYS = 3      # short holes get carried forward; longer ones stay NaN


@dataclass
class Panel:
    """Wide daily panels sharing one calendar and one column order."""

    price: pd.DataFrame       # index: date, columns: asset
    mcap: pd.DataFrame

    @property
    def assets(self) -> list[str]:
        return list(self.price.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.price.index)

    def returns(self, log_returns: bool = False) -> pd.DataFrame:
        if log_returns:
            return np.log(self.price).diff()
        # fill_method=None: leading and interior gaps stay NaN instead of being
        # forward filled into a fake zero return. pandas deprecated the padding
        # default, and padding was never what this wanted.
        return self.price.pct_change(fill_method=None)

    def slice(self, start=None, end=None) -> Panel:
        return Panel(self.price.loc[start:end], self.mcap.loc[start:end])

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.price.to_parquet(directory / "price.parquet")
        self.mcap.to_parquet(directory / "mcap.parquet")

    @classmethod
    def read(cls, directory: Path) -> Panel:
        price = pd.read_parquet(directory / "price.parquet")
        mcap = pd.read_parquet(directory / "mcap.parquet")
        return cls(price.sort_index(), mcap.sort_index().reindex(columns=price.columns))


def build(
    raw: pd.DataFrame,
    *,
    price_col: str = "PriceUSD",
    mcap_col: str = "CapMrktEstUSD",
    exclude: list[str] | None = None,
    peg_window: int = 180,
    peg_max_cv: float = 0.02,
) -> Panel:
    """Pivot the long download, drop pegged assets, align on a daily calendar."""
    price = raw.pivot(index="time", columns="asset", values=price_col).sort_index()
    mcap = raw.pivot(index="time", columns="asset", values=mcap_col).sort_index()

    calendar = pd.date_range(price.index.min(), price.index.max(), freq="D")
    price = price.reindex(calendar).ffill(limit=MAX_GAP_DAYS)
    mcap = mcap.reindex(calendar).ffill(limit=MAX_GAP_DAYS)
    price.index.name = mcap.index.name = "date"

    dropped_manual = sorted(set(exclude or []) & set(price.columns))
    if dropped_manual:
        price = price.drop(columns=dropped_manual)
        log.info("excluded %d assets by name: %s", len(dropped_manual), ", ".join(dropped_manual))

    pegged = detect_pegs(price, window=peg_window, max_cv=peg_max_cv)
    if pegged:
        price = price.drop(columns=pegged)
        log.info("excluded %d assets by peg detection: %s", len(pegged), ", ".join(pegged))

    # Anything with no price at all is useless even if the catalog listed it.
    empty = [c for c in price.columns if price[c].notna().sum() == 0]
    if empty:
        price = price.drop(columns=empty)

    mcap = mcap.reindex(columns=price.columns)
    return Panel(price=price, mcap=mcap)


def detect_pegs(price: pd.DataFrame, *, window: int, max_cv: float) -> list[str]:
    """Assets whose price barely moves relative to its level.

    A rolling coefficient of variation is used rather than a global one so that a
    token that was pegged for part of its life is still caught. The median across
    windows keeps a single volatile episode from rescuing an obvious stablecoin.
    """
    rolling_cv = price.rolling(window, min_periods=window // 2).std() / price.rolling(
        window, min_periods=window // 2
    ).mean().abs()
    median_cv = rolling_cv.median(skipna=True)
    return sorted(median_cv.index[median_cv < max_cv].tolist())


def continuous_history(price: pd.DataFrame) -> pd.DataFrame:
    """Length of the unbroken run of observations ending at each date, per asset.

    Not the same as "days since first observation": Coin Metrics occasionally
    back-publishes an isolated early observation for an asset long before
    continuous coverage begins, and a naive age test would let it through.
    """
    observed = price.notna()
    runs = {}
    for asset in observed.columns:
        seen = observed[asset]
        blocks = (~seen).cumsum()          # a new block starts after every gap
        runs[asset] = seen.groupby(blocks).cumsum()
    return pd.DataFrame(runs, index=price.index)


def summarise(panel: Panel) -> pd.DataFrame:
    """One row per asset: coverage and basic scale, for the data appendix."""
    price, mcap = panel.price, panel.mcap
    rows = []
    for asset in panel.assets:
        series = price[asset].dropna()
        if series.empty:
            continue
        daily = series.pct_change().dropna()
        rows.append(
            {
                "asset": asset,
                "first": series.index[0].date(),
                "last": series.index[-1].date(),
                "n_days": len(series),
                "median_mcap_usd": float(mcap[asset].median(skipna=True)),
                "ann_vol": float(daily.std() * np.sqrt(365)),
                "max_drawdown": float((series / series.cummax() - 1.0).min()),
            }
        )
    return pd.DataFrame(rows).sort_values("median_mcap_usd", ascending=False, ignore_index=True)
