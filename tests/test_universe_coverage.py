"""Universe construction against partial data coverage.

All three things pinned here were live bugs found while fetching the real panel.
Market capitalisation on the free tier starts months after price does, and the
universe builder answered that by handing back empty universes with a warning
nobody was going to read in a cluster log.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scenport.data.panel import Panel
from scenport.data.universe import Universe, build, first_covered, rebalance_dates


def _panel(n_assets=20, start="2016-01-01", periods=2200, mcap_from="2019-06-22"):
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=periods, freq="D")
    names = [f"A{i:02d}" for i in range(n_assets)]

    price = pd.DataFrame(
        {n: 100 * np.cumprod(1 + rng.normal(0, 0.02, periods)) for n in names}, index=dates
    )
    mcap = pd.DataFrame(
        {n: np.linspace(1e9 * (i + 1), 2e9 * (i + 1), periods) for i, n in enumerate(names)},
        index=dates,
    )
    # Price reaches back further than capitalisation, which is the real shape.
    mcap.loc[mcap.index < pd.Timestamp(mcap_from)] = np.nan
    return Panel(price, mcap)


def test_turnover_does_not_raise():
    """zip(dates, dates[1:], strict=True) raised for every input it was ever given."""
    members = {
        pd.Timestamp("2020-01-01"): ["a", "b", "c"],
        pd.Timestamp("2020-02-01"): ["a", "b", "d"],
        pd.Timestamp("2020-03-01"): ["a", "b", "d"],
    }
    turnover = Universe(members).turnover()

    assert list(turnover.index) == [pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-01")]
    assert turnover.iloc[0] == pytest.approx(1 / 3)
    assert turnover.iloc[1] == pytest.approx(0.0)


def test_turnover_of_a_single_date_is_empty():
    assert Universe({pd.Timestamp("2020-01-01"): ["a"]}).turnover().empty


def test_dates_without_market_cap_are_dropped_not_returned_empty():
    panel = _panel()
    dates = rebalance_dates(panel, "2017-01-01", "2021-01-01")
    universe = build(panel, dates, size=10, min_history_days=180)

    assert universe.dates, "something should survive"
    assert min(universe.dates) >= pd.Timestamp("2019-06-22")
    for date in universe.dates:
        assert len(universe.at(date)) >= 5, f"{date} came back short"


def test_no_usable_date_raises_with_the_coverage_in_the_message():
    """A silent empty universe is how this got as far as the backtest."""
    panel = _panel(mcap_from="2030-01-01")
    dates = rebalance_dates(panel, "2017-01-01", "2019-01-01")

    with pytest.raises(ValueError, match="market"):
        build(panel, dates, size=10, min_history_days=180)


def test_the_error_says_where_coverage_starts():
    panel = _panel(mcap_from="2030-01-01")
    dates = rebalance_dates(panel, "2017-01-01", "2019-01-01")
    with pytest.raises(ValueError) as caught:
        build(panel, dates, size=10, min_history_days=180)
    assert "never" in str(caught.value) or "2030" in str(caught.value)


def test_first_covered_reports_the_first_date_with_anything():
    panel = _panel()
    assert first_covered(panel.mcap) == "2019-06-22"
    assert first_covered(panel.mcap.where(lambda f: f.isna())) == "never"


def test_short_but_usable_dates_are_kept():
    """Fewer than requested is a warning. Fewer than half is a drop."""
    panel = _panel(n_assets=8)
    dates = rebalance_dates(panel, "2020-01-01", "2020-06-01")
    universe = build(panel, dates, size=15, min_history_days=180)

    assert universe.dates
    for date in universe.dates:
        assert len(universe.at(date)) == 8


def test_selection_is_by_capitalisation():
    panel = _panel(n_assets=12)
    dates = rebalance_dates(panel, "2021-01-01", "2021-03-01")
    universe = build(panel, dates, size=3, min_history_days=180)

    # mcap was built increasing in the asset index, so the last three win.
    assert universe.at(universe.dates[0]) == ["A11", "A10", "A09"]
