"""Coin Metrics Community API.

No key needed, rate limited to roughly ten requests per six seconds, so every page
is cached on disk as JSON Lines.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE = "https://community-api.coinmetrics.io/v4"
PAGE_SIZE = 10_000
ASSETS_PER_REQUEST = 20          # keeps the query string comfortably short
MIN_INTERVAL = 0.7               # seconds between requests; community tier is ~10/6s
MAX_RETRIES = 6


class CoinMetricsError(RuntimeError):
    pass


class _Session:
    """requests session with throttling and retry on 429/5xx."""

    def __init__(self) -> None:
        self.http = requests.Session()
        self.http.headers["User-Agent"] = "scenport/0.4 (research; academic use)"
        self._last_call = 0.0

    def get(self, url: str, params: dict | None = None) -> dict:
        for attempt in range(MAX_RETRIES):
            gap = MIN_INTERVAL - (time.monotonic() - self._last_call)
            if gap > 0:
                time.sleep(gap)
            self._last_call = time.monotonic()

            resp = self.http.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                backoff = 2.0 * (2**attempt)
                log.warning(
                    "HTTP %s from Coin Metrics, retrying in %.0fs", resp.status_code, backoff
                )
                time.sleep(backoff)
                continue
            raise CoinMetricsError(f"HTTP {resp.status_code} for {resp.url}: {resp.text[:400]}")
        raise CoinMetricsError(f"gave up after {MAX_RETRIES} attempts: {url}")


def _batched(items: list[str], n: int) -> Iterator[list[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _catalog_pages(session: _Session, url: str, params: dict) -> Iterator[dict]:
    """Every page of a catalog response.

    The catalog paginates like the timeseries endpoints do. Reading only the first
    page silently truncates the asset list, which would then quietly shrink the
    investable universe without anything failing.
    """
    payload = session.get(url, params=params)
    while True:
        yield payload
        following = payload.get("next_page_url")
        if not following:
            return
        payload = session.get(following)


def _daily_on_community(frequencies: list[dict]) -> bool:
    """Daily coverage that this API tier will actually serve.

    An asset can carry a metric that the community tier does not expose. Without
    this check it lands in the universe and then comes back as an empty series.
    """
    for entry in frequencies:
        if entry.get("frequency") != "1d":
            continue
        if entry.get("community", True):
            return True
    return False


def available_assets(metrics: Iterable[str], session: _Session | None = None) -> list[str]:
    """Assets that expose every requested metric at daily frequency.

    The catalog response shape has changed across v4 revisions. Current asset-major
    responses carry ``metrics`` as a list of ``{"metric", "frequencies"}``; older
    ones used a mapping keyed by metric name, and there is a metric-major layout as
    well. All three are handled.
    """
    session = session or _Session()
    wanted = list(metrics)
    per_metric: list[set[str]] = []

    for metric in wanted:
        found: set[str] = set()
        for payload in _catalog_pages(
            session, f"{BASE}/catalog-v2/asset-metrics", {"metrics": metric}
        ):
            for entry in payload.get("data", []):
                asset = entry.get("asset")
                if asset is None:                                   # metric-major
                    for freq in entry.get("frequencies", []):
                        if freq.get("frequency") == "1d":
                            found.update(freq.get("assets", []))
                    continue

                listed = entry.get("metrics", [])
                if isinstance(listed, dict):                        # older asset-major
                    if _daily_on_community(listed.get(metric, [])):
                        found.add(asset)
                    continue

                for item in listed:                                 # current asset-major
                    if item.get("metric") != metric:
                        continue
                    if _daily_on_community(item.get("frequencies", [])):
                        found.add(asset)

        if not found:
            raise CoinMetricsError(f"catalog returned no daily coverage for {metric!r}")
        log.info("%s: %d assets with daily community coverage", metric, len(found))
        per_metric.append(found)

    both = sorted(set.intersection(*per_metric))
    log.info("%d assets carry all of %s", len(both), ", ".join(wanted))
    return both


def fetch(
    assets: list[str],
    metrics: list[str],
    start: str,
    end: str,
    cache_dir: Path,
    session: _Session | None = None,
) -> pd.DataFrame:
    """Return a long dataframe with columns ``asset``, ``time`` and one per metric."""
    session = session or _Session()
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    for batch in _batched(sorted(assets), ASSETS_PER_REQUEST):
        tag = f"{batch[0]}-{batch[-1]}_{start}_{end}_{'-'.join(metrics)}"
        cached = cache_dir / f"{tag}.jsonl"
        if cached.exists():
            rows = [json.loads(line) for line in cached.read_text(encoding="utf-8").splitlines()]
            log.info("cache hit for %d assets (%s)", len(batch), batch[0])
        else:
            rows = list(
                _paginate(
                    session,
                    assets=batch,
                    metrics=metrics,
                    start=start,
                    end=end,
                )
            )
            with cached.open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
            log.info("fetched %d rows for %d assets (%s...)", len(rows), len(batch), batch[0])
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        raise CoinMetricsError("no data returned for any asset")

    df = pd.concat(frames, ignore_index=True)
    stamps = pd.to_datetime(df["time"], format="ISO8601", utc=True)
    df["time"] = stamps.dt.tz_localize(None).dt.normalize()
    for metric in metrics:
        df[metric] = pd.to_numeric(df.get(metric), errors="coerce")
    return df[["asset", "time", *metrics]].sort_values(["asset", "time"], ignore_index=True)


def _paginate(
    session: _Session, *, assets: list[str], metrics: list[str], start: str, end: str
) -> Iterator[dict]:
    params = {
        "assets": ",".join(assets),
        "metrics": ",".join(metrics),
        "frequency": "1d",
        "start_time": start,
        "end_time": end,
        "page_size": PAGE_SIZE,
        "paging_from": "start",
    }
    url = f"{BASE}/timeseries/asset-metrics"
    while True:
        payload = session.get(url, params=params)
        yield from payload.get("data", [])
        nxt = payload.get("next_page_url")
        if not nxt:
            return
        url, params = nxt, None      # next_page_url already carries the query string
