"""Inference for backtests.

Ledoit-Wolf for two dependent Sharpe ratios, Diebold-Mariano for equal predictive
accuracy, the deflated Sharpe ratio for the number of configurations tried, CSCV for
the probability of backtest overfitting. ``spa_test`` and ``model_confidence_set``
wrap the arch implementations so every test in the paper is called from one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


@dataclass
class TestResult:
    statistic: float
    p_value: float
    detail: dict

    def __repr__(self) -> str:  # keeps log lines readable
        return f"TestResult(stat={self.statistic:.4f}, p={self.p_value:.4f})"


# ---------------------------------------------------------------------------
# HAC machinery


def _parzen(x: np.ndarray) -> np.ndarray:
    a = np.abs(x)
    out = np.zeros_like(a)
    small = a <= 0.5
    mid = (a > 0.5) & (a <= 1.0)
    out[small] = 1.0 - 6.0 * a[small] ** 2 + 6.0 * a[small] ** 3
    out[mid] = 2.0 * (1.0 - a[mid]) ** 3
    return out


def andrews_bandwidth(y: np.ndarray) -> float:
    """Andrews' plug-in bandwidth for the Parzen kernel, from AR(1) fits."""
    T, k = y.shape
    num = den = 0.0
    for j in range(k):
        series = y[:, j]
        lagged, current = series[:-1], series[1:]
        denom = float(lagged @ lagged)
        rho = float(lagged @ current) / denom if denom > 0 else 0.0
        rho = np.clip(rho, -0.97, 0.97)
        resid = current - rho * lagged
        sigma2 = float(resid @ resid) / max(len(resid), 1)
        num += 4.0 * rho**2 * sigma2**2 / (1.0 - rho) ** 8
        den += sigma2**2 / (1.0 - rho) ** 4
    alpha2 = num / den if den > 0 else 0.0
    return float(2.6614 * (alpha2 * T) ** 0.2) if alpha2 > 0 else 1.0


def hac_covariance(y: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    """Long-run covariance of the mean of ``y`` (T, k), Parzen kernel."""
    y = np.asarray(y, dtype=float)
    T = y.shape[0]
    y = y - y.mean(axis=0, keepdims=True)
    if bandwidth is None:
        bandwidth = andrews_bandwidth(y)
    bandwidth = max(bandwidth, 1e-8)

    psi = (y.T @ y) / T
    max_lag = min(T - 1, int(np.ceil(4.0 * bandwidth)))
    for lag in range(1, max_lag + 1):
        weight = float(_parzen(np.array([lag / bandwidth]))[0])
        if weight == 0.0:
            continue
        gamma = (y[lag:].T @ y[:-lag]) / T
        psi += weight * (gamma + gamma.T)
    return psi


def sharpe_difference(
    returns_a, returns_b, bootstrap: int = 0, block: int = 21, seed: int = 0
) -> TestResult:
    """Test H0: the two return series have the same Sharpe ratio.

    Both series must be aligned and observed over the same dates -- they are two
    strategies run on the same data, so their errors are strongly dependent and a
    two-sample t-test on the Sharpe ratios would be badly oversized.

    With ``bootstrap > 0`` a studentised circular block bootstrap p-value is
    reported alongside the asymptotic one, which is what Ledoit and Wolf recommend
    for samples of the size a monthly-rebalanced backtest produces.
    """
    frame = pd.concat(
        [pd.Series(returns_a).rename("a"), pd.Series(returns_b).rename("b")], axis=1
    ).dropna()
    a, b = frame["a"].to_numpy(float), frame["b"].to_numpy(float)
    T = len(a)
    if T < 20:
        return TestResult(float("nan"), float("nan"), {"reason": "fewer than 20 observations"})

    diff, se = _sharpe_delta(a, b)
    z = diff / se if se > 0 else np.nan
    p_asymptotic = float(2.0 * (1.0 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else float("nan")

    detail = {
        "sharpe_a": float(a.mean() / a.std(ddof=0)),
        "sharpe_b": float(b.mean() / b.std(ddof=0)),
        "difference": float(diff),
        "std_error": float(se),
        "n": T,
        "p_asymptotic": p_asymptotic,
    }

    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        centred = []
        n_blocks = int(np.ceil(T / block))
        for _ in range(bootstrap):
            starts = rng.integers(0, T, size=n_blocks)
            idx = np.concatenate([(np.arange(block) + s) % T for s in starts])[:T]
            d_star, se_star = _sharpe_delta(a[idx], b[idx])
            if se_star > 0:
                centred.append(abs((d_star - diff) / se_star))
        if centred:
            detail["p_bootstrap"] = float(np.mean(np.array(centred) >= abs(z)))
            detail["bootstrap_reps"] = len(centred)

    return TestResult(float(z), detail.get("p_bootstrap", p_asymptotic), detail)


def _sharpe_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Sharpe difference and its HAC standard error (Ledoit and Wolf, 2008)."""
    mu_a, mu_b = a.mean(), b.mean()
    g_a, g_b = (a**2).mean(), (b**2).mean()
    var_a, var_b = g_a - mu_a**2, g_b - mu_b**2
    if var_a <= 0 or var_b <= 0:
        return float("nan"), float("nan")

    diff = mu_a / np.sqrt(var_a) - mu_b / np.sqrt(var_b)
    grad = np.array(
        [
            g_a / var_a**1.5,
            -g_b / var_b**1.5,
            -0.5 * mu_a / var_a**1.5,
            0.5 * mu_b / var_b**1.5,
        ]
    )
    moments = np.column_stack([a, b, a**2, b**2])
    psi = hac_covariance(moments)
    variance = float(grad @ psi @ grad) / len(a)
    return float(diff), float(np.sqrt(variance)) if variance > 0 else float("nan")


def diebold_mariano(loss_a, loss_b, horizon: int = 1) -> TestResult:
    """Equal predictive accuracy of two forecasters, on their loss differentials.

    Uses the Harvey, Leybourne and Newbold small-sample correction and a t
    reference distribution, which is the version that behaves at the sample sizes
    a monthly walk-forward produces.
    """
    frame = pd.concat(
        [pd.Series(loss_a).rename("a"), pd.Series(loss_b).rename("b")], axis=1
    ).dropna()
    d = (frame["a"] - frame["b"]).to_numpy(float)
    T = len(d)
    if T < 10:
        return TestResult(float("nan"), float("nan"), {"reason": "fewer than 10 pairs"})

    d_bar = d.mean()
    gamma0 = np.var(d, ddof=0)
    acov = sum(2.0 * np.cov(d[k:], d[:-k], ddof=0)[0, 1] for k in range(1, horizon))
    long_run = (gamma0 + acov) / T
    if long_run <= 0:
        return TestResult(float("nan"), float("nan"), {"reason": "non-positive long-run variance"})

    stat = d_bar / np.sqrt(long_run)
    correction = np.sqrt(
        (T + 1 - 2 * horizon + horizon * (horizon - 1) / T) / T
    )
    stat *= correction
    p = float(2.0 * (1.0 - stats.t.cdf(abs(stat), df=T - 1)))
    return TestResult(float(stat), p, {"mean_loss_difference": float(d_bar), "n": T})


def deflated_sharpe(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
    sharpe_variance: float | None = None,
) -> TestResult:
    """Bailey and Lopez de Prado's deflated Sharpe ratio.

    ``observed_sharpe`` must be on the *observation* frequency, not annualised.
    ``n_trials`` is the number of configurations that were compared before this one
    was selected -- for this study, the size of the factorial grid.
    """
    if sharpe_variance is None:
        # Under the null of no skill, Var(SR) across trials is 1/n_observations.
        sharpe_variance = 1.0 / max(n_observations - 1, 1)

    n = max(int(n_trials), 2)
    expected_max = np.sqrt(sharpe_variance) * (
        (1.0 - EULER_MASCHERONI) * stats.norm.ppf(1.0 - 1.0 / n)
        + EULER_MASCHERONI * stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    )

    denominator = 1.0 - skew * observed_sharpe + 0.25 * (excess_kurtosis) * observed_sharpe**2
    if denominator <= 0:
        return TestResult(float("nan"), float("nan"), {"reason": "negative variance adjustment"})

    z = (observed_sharpe - expected_max) * np.sqrt(n_observations - 1) / np.sqrt(denominator)
    return TestResult(
        float(z),
        float(1.0 - stats.norm.cdf(z)),
        {
            "expected_max_sharpe": float(expected_max),
            "deflated_probability": float(stats.norm.cdf(z)),
            "n_trials": n,
        },
    )


def probability_of_backtest_overfitting(
    performance: pd.DataFrame, n_splits: int = 12, metric=None
) -> TestResult:
    """CSCV probability that the in-sample winner underperforms out of sample.

    ``performance`` is (observations x configurations) of period returns. The sample
    is cut into ``n_splits`` contiguous blocks; every way of splitting those blocks
    into equal in-sample and out-of-sample halves is enumerated, the best
    configuration in sample is found, and its out-of-sample rank recorded. The
    reported probability is the share of splits where that rank lands below median.

    Above roughly 0.5 the backtest carries no information about future performance.
    """
    matrix = performance.dropna(axis=0, how="any")
    if matrix.shape[1] < 2:
        return TestResult(float("nan"), float("nan"), {"reason": "need at least 2 configurations"})
    if n_splits % 2:
        n_splits += 1

    blocks = np.array_split(np.arange(len(matrix)), n_splits)
    if any(len(b) == 0 for b in blocks):
        return TestResult(float("nan"), float("nan"), {"reason": "too few observations to split"})

    scorer = metric or (lambda x: x.mean(axis=0) / x.std(axis=0, ddof=1))
    values = matrix.to_numpy(float)

    logits = []
    for chosen in combinations(range(n_splits), n_splits // 2):
        rest = [b for b in range(n_splits) if b not in chosen]
        is_rows = np.concatenate([blocks[b] for b in chosen])
        oos_rows = np.concatenate([blocks[b] for b in rest])

        with np.errstate(invalid="ignore", divide="ignore"):
            is_score = np.nan_to_num(scorer(values[is_rows]), nan=-np.inf)
            oos_score = np.nan_to_num(scorer(values[oos_rows]), nan=-np.inf)

        winner = int(np.argmax(is_score))
        rank = float(stats.rankdata(oos_score)[winner])
        omega = rank / (len(oos_score) + 1.0)
        logits.append(np.log(omega / (1.0 - omega)))

    logits = np.asarray(logits)
    pbo = float((logits <= 0).mean())
    return TestResult(
        pbo,
        pbo,
        {
            "n_combinations": len(logits),
            "median_logit": float(np.median(logits)),
            "n_configurations": matrix.shape[1],
        },
    )


def spa_test(benchmark, candidates: pd.DataFrame, reps: int = 5000, block: int = 21, seed: int = 0):
    """Hansen's Superior Predictive Ability test. Returns the arch result object.

    Answers "does *any* of these configurations beat the benchmark once the whole
    search is accounted for", which is the question a reader should be asking after
    seeing a table with eighty rows in it.
    """
    from arch.bootstrap import SPA

    losses = -candidates.dropna()
    base = -pd.Series(benchmark).reindex(losses.index).dropna()
    losses = losses.loc[base.index]
    spa = SPA(base.to_numpy(float), losses.to_numpy(float), reps=reps, block_size=block,
              seed=seed)
    spa.compute()
    return spa


def model_confidence_set(losses: pd.DataFrame, alpha: float = 0.10, reps: int = 5000,
                         block: int = 21, seed: int = 0):
    """Hansen, Lunde and Nason's Model Confidence Set. Returns the arch result object.

    Reports the set of configurations that cannot be distinguished from the best at
    level ``alpha``, which is a far more honest summary than a single winner.
    """
    from arch.bootstrap import MCS

    mcs = MCS(losses.dropna().to_numpy(float), size=alpha, reps=reps, block_size=block, seed=seed)
    mcs.compute()
    return mcs


def stationary_bootstrap_ci(
    series, statistic, reps: int = 5000, block: int = 21, confidence: float = 0.95, seed: int = 0
) -> dict:
    """Percentile confidence interval for any statistic of a return series."""
    from arch.bootstrap import StationaryBootstrap

    values = pd.Series(series).dropna()
    boot = StationaryBootstrap(block, values.to_numpy(float), seed=seed)
    draws = np.array([statistic(sample[0][0]) for sample, _ in boot.bootstrap(reps)])
    draws = draws[np.isfinite(draws)]
    lo, hi = np.quantile(draws, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    return {
        "point": float(statistic(values.to_numpy(float))),
        "lower": float(lo),
        "upper": float(hi),
        "reps": int(draws.size),
    }
