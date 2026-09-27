# Data source and licence

The parquet files in this directory are derived from **Coin Metrics Community
Data**, retrieved through the Community API with no key, and rebuilt by
`scripts/fetch_data.py`.

Coin Metrics Community Data is published under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). That licence
permits redistribution, including the redistribution of this panel, on two
conditions:

- **Attribution.** Coin Metrics is credited as the source, here, in the README
  and in the paper.
- **Non-commercial use only.** This is academic research and is not used for any
  commercial purpose.

If you build on this panel, carry both conditions forward.

## What is here

| File | Contents |
|---|---|
| `price.parquet` | Daily `PriceUSD`, wide, dates by asset |
| `mcap.parquet` | Daily `CapMrktEstUSD`, same shape |
| `universe.parquet` | Investable set at each rebalancing date |
| `risk_free.parquet` | `DTB3` from FRED, which is public domain |
| `asset_summary.csv`, `universe_coverage.csv`, `universe_turnover.csv` | Descriptives |

## Coverage, and what the free tier costs

The Community tier serves daily `PriceUSD` for 138 assets and `CapMrktEstUSD`
for 839. Only 87 carry both, and 76 survive the exclusion of stablecoins,
wrapped tokens and commodity-backed tokens.

Solana is the notable absence: it ranks seventh by market capitalisation over
most of the sample and the Community tier does not serve its price, so it never
enters the universe. NEAR, LEO and HYPE are missing for the same reason.

Market capitalisation coverage starts 2019-06-22, months after price does, which
is why the backtest begins in August 2019 rather than January.

Anyone rerunning this with a licensed Coin Metrics tier should check whether the
licence still permits republishing the panel. Under CC BY-NC it does.
