"""Benchmark weights, the risk-free series and a few return helpers."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .panel import Panel
from .universe import Universe

log = logging.getLogger(__name__)

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"


def buy_and_hold(panel: Panel, asset: str, dates: pd.DatetimeIndex) -> pd.Series:
    """Daily simple returns of a single asset over ``dates``."""
    return panel.price[asset].reindex(dates).pct_change().rename(f"bh_{asset}")


def index_weights(
    panel: Panel, universe: Universe, kind: str = "mcap"
) -> dict[pd.Timestamp, pd.Series]:
    """Benchmark weights at each rebalancing date, on that date's universe."""
    out = {}
    for date in universe.dates:
        assets = universe.at(date)
        if kind == "mcap":
            caps = panel.mcap.loc[date, assets].astype(float)
            weights = caps / caps.sum()
        elif kind == "equal":
            weights = pd.Series(1.0 / len(assets), index=assets)
        else:
            raise ValueError(f"unknown benchmark kind {kind!r}")
        out[date] = weights
    return out


def risk_free(start: str, end: str, cache: Path | None = None) -> pd.Series:
    """Daily 3-month T-bill discount rate as a decimal, forward filled.

    FRED publishes DTB3 on business days only and marks holidays with ".", so the
    series is reindexed onto a full calendar before use.
    """
    if cache is not None and cache.exists():
        raw = cache.read_text(encoding="utf-8")
    else:
        resp = requests.get(FRED_CSV, timeout=60)
        resp.raise_for_status()
        raw = resp.text
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(raw, encoding="utf-8")

    df = pd.read_csv(io.StringIO(raw))
    date_col, value_col = df.columns[0], df.columns[-1]
    df[date_col] = pd.to_datetime(df[date_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    series = df.set_index(date_col)[value_col].dropna() / 100.0
    calendar = pd.date_range(start, end, freq="D")
    return series.reindex(calendar).ffill().bfill().rename("rf_annual")


def horizon_rate(rf_annual: pd.Series, date: pd.Timestamp, horizon_days: int) -> float:
    """Convert the annual rate quoted on ``date`` into a horizon-matched rate.

    Compounding rather than scaling, so the number stays comparable with the
    compounded asset returns used everywhere else.
    """
    annual = float(rf_annual.loc[:date].iloc[-1])
    return float((1.0 + annual) ** (horizon_days / 365.0) - 1.0)


def load_sp500(path: Path) -> pd.Series:
    """Read a local S&P 500 close series.

    Accepts either an Investing.com export (``Fecha``/``Último``, comma decimals,
    dots as thousands separators) or a plain two-column ``date,close`` file.
    """
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    date_col = next((cols[k] for k in ("fecha", "date") if k in cols), None)
    close_col = next(
        (cols[k] for k in ("último", "ultimo", "close", "adj close", "cierre") if k in cols), None
    )
    if date_col is None or close_col is None:
        raise ValueError(f"could not find date/close columns in {path.name}: {list(df.columns)}")

    dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    values = df[close_col]
    if values.dtype == object:
        values = values.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    values = pd.to_numeric(values, errors="coerce")

    series = pd.Series(values.to_numpy(), index=dates, name="sp500").dropna().sort_index()
    return series[~series.index.duplicated(keep="last")]


def period_return(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Return between the last observation at or before each endpoint.

    The S&P 500 does not trade every day, so an exact date match silently drops
    periods. Both endpoints go through here so nothing ends up measured over a
    different window from its benchmark.
    """
    left = series.loc[:start]
    right = series.loc[:end]
    if left.empty or right.empty:
        return float("nan")
    return float(right.iloc[-1] / left.iloc[-1] - 1.0)


def annualise(returns: pd.Series, periods_per_year: int = 365) -> tuple[float, float]:
    """(annualised geometric return, annualised volatility) of a return series."""
    clean = returns.dropna()
    if clean.empty:
        return float("nan"), float("nan")
    growth = float((1.0 + clean).prod())
    years = len(clean) / periods_per_year
    ann_ret = growth ** (1.0 / years) - 1.0 if years > 0 and growth > 0 else float("nan")
    ann_vol = float(clean.std(ddof=1) * np.sqrt(periods_per_year))
    return ann_ret, ann_vol
