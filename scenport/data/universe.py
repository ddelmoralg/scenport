"""Investable universe as it stood at each rebalancing date.

See notes/universe.md for why this is rebuilt rather than fixed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .panel import Panel, continuous_history

log = logging.getLogger(__name__)


@dataclass
class Universe:
    """Membership of the investable set at each rebalancing date."""

    members: dict[pd.Timestamp, list[str]]

    @property
    def dates(self) -> list[pd.Timestamp]:
        return sorted(self.members)

    def at(self, date: pd.Timestamp) -> list[str]:
        return self.members[pd.Timestamp(date)]

    def as_frame(self) -> pd.DataFrame:
        """Long form, convenient for the data appendix table."""
        rows = [
            {"rebalance": d, "rank": i + 1, "asset": a}
            for d, assets in sorted(self.members.items())
            for i, a in enumerate(assets)
        ]
        return pd.DataFrame(rows)

    def turnover(self) -> pd.Series:
        """Fraction of the universe replaced at each rebalancing date."""
        dates = self.dates
        out = {}
        for prev, curr in zip(dates, dates[1:], strict=True):
            before, after = set(self.members[prev]), set(self.members[curr])
            out[curr] = 1.0 - len(before & after) / max(len(after), 1)
        return pd.Series(out, name="universe_turnover")


def rebalance_dates(panel: Panel, first: str, last: str, freq: str = "MS") -> list[pd.Timestamp]:
    """Calendar of rebalancing dates that actually exist in the panel."""
    wanted = pd.date_range(first, last, freq=freq)
    available = panel.dates
    out = []
    for date in wanted:
        # Snap forward to the first day with data, which matters for month starts
        # that fall on a hole in the source series.
        candidates = available[available >= date]
        if len(candidates):
            out.append(candidates[0])
    return sorted(set(out))


def build(
    panel: Panel,
    dates: list[pd.Timestamp],
    *,
    size: int = 15,
    min_history_days: int = 540,
    rank_window: int = 30,
) -> Universe:
    """Select the ``size`` largest eligible assets at each date in ``dates``."""
    history = continuous_history(panel.price)
    mcap_rank = panel.mcap.rolling(rank_window, min_periods=max(rank_window // 2, 1)).median()

    members: dict[pd.Timestamp, list[str]] = {}
    for date in dates:
        if date not in panel.price.index:
            raise KeyError(f"{date.date()} is not in the panel calendar")

        long_enough = history.loc[date] >= min_history_days
        priced = panel.price.loc[date].notna()
        caps = mcap_rank.loc[date]
        eligible = caps[long_enough & priced].dropna()

        if len(eligible) < size:
            log.warning(
                "%s: only %d eligible assets for a universe of %d",
                date.date(),
                len(eligible),
                size,
            )
        members[date] = eligible.sort_values(ascending=False).head(size).index.tolist()

    return Universe(members=members)


def read(path) -> Universe:
    """Load a universe written by ``Universe.as_frame().to_parquet``."""
    frame = pd.read_parquet(path)
    members = {
        pd.Timestamp(date): group.sort_values("rank")["asset"].tolist()
        for date, group in frame.groupby("rebalance")
    }
    return Universe(members=members)


def coverage_report(universe: Universe) -> pd.DataFrame:
    """How often each asset appears, and over which span."""
    frame = universe.as_frame()
    grouped = frame.groupby("asset")
    report = pd.DataFrame(
        {
            "n_periods": grouped.size(),
            "first": grouped["rebalance"].min().dt.date,
            "last": grouped["rebalance"].max().dt.date,
            "mean_rank": grouped["rank"].mean().round(1),
        }
    )
    return report.sort_values("n_periods", ascending=False)
