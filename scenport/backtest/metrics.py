"""Performance and risk statistics for a realised return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365          # cryptoassets trade every day of the week


def _clean(returns) -> np.ndarray:
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    return r[np.isfinite(r)]


def total_return(returns) -> float:
    r = _clean(returns)
    return float(np.prod(1.0 + r) - 1.0) if r.size else float("nan")


def annualised_return(returns, periods_per_year: int = DAYS_PER_YEAR) -> float:
    r = _clean(returns)
    if r.size == 0:
        return float("nan")
    growth = float(np.prod(1.0 + r))
    if growth <= 0:
        return -1.0
    return float(growth ** (periods_per_year / r.size) - 1.0)


def annualised_volatility(returns, periods_per_year: int = DAYS_PER_YEAR) -> float:
    r = _clean(returns)
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year)) if r.size > 1 else float("nan")


def sharpe(
    returns, rf_daily: float | pd.Series = 0.0, periods_per_year: int = DAYS_PER_YEAR
) -> float:
    """Annualised Sharpe ratio of the excess return series.

    ``rf_daily`` is a *daily* rate, constant or a series aligned with the returns,
    so numerator and denominator stay on the same time scale.
    """
    r = pd.Series(returns).dropna()
    if isinstance(rf_daily, pd.Series):
        excess = r - rf_daily.reindex(r.index).fillna(0.0)
    else:
        excess = r - rf_daily
    e = _clean(excess)
    if e.size < 2:
        return float("nan")
    sd = float(np.std(e, ddof=1))
    # An exactly constant series comes out of numpy with a standard deviation of
    # order 1e-19 rather than zero, which would report a Sharpe of 1e16.
    if sd <= 1e-12 * max(abs(float(np.mean(e))), 1.0):
        return float("nan")
    return float(np.mean(e) / sd * np.sqrt(periods_per_year))


def sortino(returns, target: float = 0.0, periods_per_year: int = DAYS_PER_YEAR) -> float:
    r = _clean(returns)
    if r.size < 2:
        return float("nan")
    downside = np.minimum(r - target, 0.0)
    dd = np.sqrt(np.mean(downside**2))
    if dd == 0:
        return float("inf") if np.mean(r) > target else float("nan")
    return float((np.mean(r) - target) / dd * np.sqrt(periods_per_year))


def drawdown_curve(returns) -> pd.Series:
    r = pd.Series(returns).dropna()
    nav = (1.0 + r).cumprod()
    return nav / nav.cummax() - 1.0


def max_drawdown(returns) -> float:
    curve = drawdown_curve(returns)
    return float(curve.min()) if len(curve) else float("nan")


def calmar(returns, periods_per_year: int = DAYS_PER_YEAR) -> float:
    mdd = abs(max_drawdown(returns))
    if not np.isfinite(mdd) or mdd == 0:
        return float("nan")
    return annualised_return(returns, periods_per_year) / mdd


def value_at_risk(returns, alpha: float = 0.95) -> float:
    """Historical VaR as a positive loss."""
    r = _clean(returns)
    return float(-np.quantile(r, 1.0 - alpha)) if r.size else float("nan")


def conditional_var(returns, alpha: float = 0.95) -> float:
    """Historical CVaR as a positive loss."""
    r = _clean(returns)
    if r.size == 0:
        return float("nan")
    threshold = np.quantile(r, 1.0 - alpha)
    tail = r[r <= threshold]
    return float(-tail.mean()) if tail.size else float("nan")


def ulcer_index(returns) -> float:
    """Root mean square drawdown. Penalises long shallow declines that MDD misses."""
    curve = drawdown_curve(returns)
    return float(np.sqrt(np.mean(np.square(curve)))) if len(curve) else float("nan")


def hit_rate(returns) -> float:
    r = _clean(returns)
    return float((r > 0).mean()) if r.size else float("nan")


def tail_ratio(returns, q: float = 0.05) -> float:
    """Right tail over left tail. Above one means the upside is fatter."""
    r = _clean(returns)
    if r.size < 20:
        return float("nan")
    left = abs(np.quantile(r, q))
    return float(np.quantile(r, 1 - q) / left) if left > 0 else float("nan")


def summary(
    returns,
    rf_daily: float | pd.Series = 0.0,
    periods_per_year: int = DAYS_PER_YEAR,
    alpha: float = 0.95,
) -> dict:
    """Everything at once, in the order the results tables use."""
    r = pd.Series(returns).dropna()
    return {
        "n_days": int(len(r)),
        "total_return": total_return(r),
        "ann_return": annualised_return(r, periods_per_year),
        "ann_vol": annualised_volatility(r, periods_per_year),
        "sharpe": sharpe(r, rf_daily, periods_per_year),
        "sortino": sortino(r, 0.0, periods_per_year),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar(r, periods_per_year),
        "ulcer": ulcer_index(r),
        f"var_{int(alpha * 100)}": value_at_risk(r, alpha),
        f"cvar_{int(alpha * 100)}": conditional_var(r, alpha),
        "skew": float(r.skew()) if len(r) > 2 else float("nan"),
        "excess_kurtosis": float(r.kurtosis()) if len(r) > 3 else float("nan"),
        "hit_rate": hit_rate(r),
        "tail_ratio": tail_ratio(r),
    }


def regime_labels(benchmark: pd.Series, window: int = 90, threshold: float = 0.10) -> pd.Series:
    """Split the sample into bull, bear and sideways stretches.

    Labels come from the trailing ``window``-day return of a *benchmark* series,
    the capitalisation-weighted index. Defining the regimes from the returns of the
    strategy being evaluated would be circular.
    """
    trailing = benchmark.pct_change(window)
    labels = pd.Series("sideways", index=benchmark.index, dtype=object)
    labels[trailing > threshold] = "bull"
    labels[trailing < -threshold] = "bear"
    labels[trailing.isna()] = "warmup"
    return labels.rename("regime")
