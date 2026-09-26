# Why the universe is rebuilt at every rebalancing date

Pick the fifteen largest cryptoassets today, download their history, backtest on it,
and you have measured the performance of a portfolio that could only have been
assembled by someone who already knew which tokens would still exist and still be
large in 2026. The assets that collapsed are not in the sample. Neither are the ones
that were large in 2019 and are now irrelevant.

This matters more in crypto than in equities. The turnover in the top fifteen by
capitalisation is brutal: over 2019-2026 well over thirty distinct assets pass through
it, and several of the exits were not gentle declines but collapses. A fixed
present-day universe removes exactly the observations a risk-aware allocator most
needs to see.

## What the code does

At each rebalancing date `t`:

1. an asset is **eligible** if it has an unbroken price record of at least
   `universe.min_history_days` days ending at `t`, and a price on `t` itself;
2. eligible assets are ranked by the **median market capitalisation over the previous
   `universe.rank_window` days**, not the spot value;
3. the top `universe.size` are the universe for the period starting at `t`.

Nothing in that uses information from after `t`. `tests/test_no_lookahead.py`
truncates the panel at a cut date, rebuilds the universe from the truncated panel, and
asserts membership before the cut is unchanged.

## Three details that are not obvious

**Continuous history, not age.** Coin Metrics occasionally publishes an isolated early
observation for an asset long before its coverage becomes continuous. "Days since the
first observation" would let such an asset in far too early, with a history full of
holes. `panel.continuous_history` counts the length of the unbroken run ending at each
date instead.

**Median capitalisation, not spot.** Small-cap market capitalisation is noisy and, on
thin order books, cheap to move. Ranking on a thirty-day median stops a single-day
spike from buying a token a month in the portfolio.

**Assets leave, and leaving costs money.** When an asset drops out of the universe the
position has to be sold whether the optimiser wants to or not. The backtest charges
that forced turnover even though it never appears in the optimisation problem, which
only ever sees the new universe. Not charging it would understate costs precisely in
the periods where the universe is churning most.

## Pegs and wrapped tokens

Stablecoins carry no forecasting signal and would dominate any minimum-variance
allocation for the wrong reason. Wrapped assets are duplicates of something already in
the universe and would let the allocator take a double position while appearing
diversified.

Both are excluded by name in `configs/base.yaml`. The name list is a maintenance
liability, so there is a backstop: an asset whose median rolling coefficient of
variation over a 180-day window is below 2 % is treated as pegged and dropped. The
rolling form matters, because a token that was pegged for part of its life and then broke
still gets caught, where a single global statistic would be rescued by the break.

Wrapped tokens are not detected automatically. A ratio test against the underlying
would work, but the name list covers the ones large enough to enter a top-fifteen
universe and the added machinery did not seem worth it.

## What is left

The universe is reconstructed from capitalisation, not from what was actually
*tradeable*. An asset can be large and still be hard to buy in size on a regulated
venue, and exchange listings and delistings are not modelled at all. Liquidity screens
based on traded volume would be the obvious next step; the transaction cost assumption
is doing that work at the moment, and doing it bluntly.
