#!/usr/bin/env python
"""Which inference the panel regression should use, decided by simulation.

    python scripts/inference_size_study.py --reps 500

The transmission panel can be tested two ways and on the real data they disagree.
Cluster-robust standard errors call MASE significant at p = 0.002 where a
permutation test reads 0.12; for CRPS and three others it goes the other way. Two
tests of one null cannot both be right, and picking by preference is not an
argument.

So: generate panels with a known null and count how often each test rejects. A
test at five per cent must reject five per cent of the time when there is nothing
to find. The panels are built to match the real one where it matters, and the two
properties that turn out to matter are how skewed the diagnostic is and how
strongly it correlates with market exposure. Measured on the real panel:

    diagnostic        skew    rho with exposure (within date)
    mase              2.95    0.44
    crps              3.17    0.36
    pinball           3.50    0.34
    coverage_error    0.73    0.28
    energy_score      3.32    0.34
    variogram_score   3.60    0.29

Earlier versions measured a MASE scaled by price changes in dollars (skew 8.17)
and a per-asset coverage error that was really a miss rate (skew 0.90). Both were
corrected in aggregate.py and the values above are the corrected ones. The return
noise and the spread of the market return are also set to what the panel shows,
0.08 within a date and 0.19 across dates, instead of 0.02 and 0.06.

Loss metrics are skewed by construction, which is the whole point: this is not an
accident of one dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenport.stats import tests as T  # noqa: E402

log = logging.getLogger("size-study")

N_DATES = 85
N_GENERATORS = 11
ALPHA = 0.05

#: Skew and confound as measured on the real panel, so the grid brackets it.
REAL = {
    "mase": (2.95, 0.44),
    "crps": (3.17, 0.36),
    "pinball": (3.50, 0.34),
    "coverage_error": (0.73, 0.28),
    "energy_score": (3.32, 0.34),
    "variogram_score": (3.60, 0.29),
}
#: Within-date standard deviation of the realised return and cross-date standard
#: deviation of the market return over the holding period, from the panel.
NOISE_SD = 0.08
MARKET_SD = 0.19


def sigma_for_skew(target: float) -> float:
    """The lognormal sigma whose skew matches a measured one.

    A lognormal with parameter s has skew (e^{s^2} + 2) sqrt(e^{s^2} - 1), which
    has no closed-form inverse, so this bisects. Matching the measured skew is what
    lets the grid say something about the real diagnostics.
    """
    if target <= 0:
        return 0.0

    def skew_of(sigma: float) -> float:
        u = np.exp(sigma * sigma)
        return (u + 2.0) * np.sqrt(max(u - 1.0, 0.0))

    low, high = 0.0, 3.0
    if skew_of(high) < target:
        return high
    for _ in range(60):
        mid = 0.5 * (low + high)
        if skew_of(mid) < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def synthetic_panel(beta: float, rng: np.random.Generator, log_sigma: float, rho: float):
    """One study: 85 dates, 11 generators, a shared market, an exposure control.

    ``log_sigma`` sets the skew of the diagnostic, zero meaning symmetric. ``rho``
    correlates it with exposure, which is the case a permutation on the diagnostic
    alone has to survive: it shuffles the score and leaves the control in place.
    """
    date = np.repeat(np.arange(N_DATES), N_GENERATORS)
    generator = np.tile(np.arange(N_GENERATORS), N_DATES)

    market = np.repeat(rng.normal(0.02, MARKET_SD, N_DATES), N_GENERATORS)
    shared = rng.normal(0.0, 1.0, N_GENERATORS)[generator]
    independent = np.sqrt(max(1.0 - rho * rho, 0.0))

    raw = rho * shared + independent * rng.normal(0.0, 1.0, date.size)
    score = np.exp(log_sigma * raw) if log_sigma > 0 else raw
    exposure = 1.0 + 0.2 * (rho * shared + independent * rng.normal(0.0, 1.0, date.size))

    ret = exposure * market + beta * score + rng.normal(0.0, NOISE_SD, date.size)

    frame = pd.DataFrame({"date": date, "score": score, "exposure": exposure, "ret": ret})
    demeaned = T.within_transform(frame, ["score", "exposure", "ret"], "date")
    return (
        demeaned["ret"].to_numpy(),
        demeaned[["score", "exposure"]].to_numpy(),
        date,
    )


def fast_permutation_p(y, X, clusters, reps: int, seed: int) -> float:
    """T.permutation_within_clusters with one lexsort per draw, same null."""
    observed = abs(np.linalg.lstsq(X, y, rcond=None)[0][0])
    codes = pd.factorize(clusters)[0]
    sorted_idx = np.argsort(codes, kind="stable")
    rng = np.random.default_rng(seed)
    shuffled = X.copy()
    extreme = 0
    for _ in range(reps):
        order = np.lexsort((rng.random(len(codes)), codes))
        shuffled[sorted_idx, 0] = X[order, 0]
        extreme += abs(np.linalg.lstsq(shuffled, y, rcond=None)[0][0]) >= observed
    return (extreme + 1) / (reps + 1)


def rejection_rates(beta, log_sigma, rho, reps, seed, permutations):
    rng = np.random.default_rng(seed)
    clustered = permutation = 0
    for _ in range(reps):
        y, X, date = synthetic_panel(beta, rng, log_sigma, rho)
        if T.cluster_robust_ols(y, X, date)["p_value"][0] < ALPHA:
            clustered += 1
        if fast_permutation_p(y, X, date, permutations,
                              int(rng.integers(1 << 30))) < ALPHA:
            permutation += 1
    return clustered / reps, permutation / reps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=400)
    parser.add_argument("--permutations", type=int, default=400)
    parser.add_argument("--out", default="artifacts/inference_size.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = []

    log.info(
        "Size at alpha = %.2f, %d replications each. Both columns should read 5%%.",
        ALPHA, args.reps,
    )
    log.info("")
    log.info("%-34s %11s %13s", "", "clustered", "permutation")

    grid = [("symmetric, independent", 0.0, 0.0)]
    grid += [(f"log-sigma {s}, rho {r}", s, r) for s in (1.0, 2.0, 3.0) for r in (0.0, 0.4)]
    grid += [
        (f"as measured for {name}", sigma_for_skew(skew), rho)
        for name, (skew, rho) in REAL.items()
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        # Written after every cell. A full run takes about an hour, and a job cut
        # short by the wall clock should leave the cells it finished on disk.
        pd.DataFrame(rows).to_csv(out, index=False)

    for label, sigma, rho in grid:
        c, p = rejection_rates(0.0, sigma, rho, args.reps, 11, args.permutations)
        log.info("%-34s %10.1f%% %12.1f%%", label, 100 * c, 100 * p)
        rows.append(
            {
                "case": label,
                "log_sigma": sigma,
                "rho": rho,
                "clustered_size": c,
                "permutation_size": p,
                "replications": args.reps,
                "skew": float((np.exp(sigma**2) + 2) * np.sqrt(max(np.exp(sigma**2) - 1, 0.0))),
            }
        )
        flush()

    log.info("")
    log.info("Power, symmetric and independent, so size is not the binding issue:")
    for beta in (0.005, 0.01):
        c, p = rejection_rates(beta, 0.0, 0.0, args.reps // 2, 21, args.permutations)
        log.info("  beta = %-8s clustered %5.1f%%   permutation %5.1f%%", beta, 100 * c, 100 * p)
        rows.append(
            {
                "case": f"power, beta = {beta}",
                "log_sigma": 0.0,
                "rho": 0.0,
                "clustered_size": c,
                "permutation_size": p,
                "replications": args.reps // 2,
                "skew": 0.0,
            }
        )
        flush()

    flush()
    (out.with_suffix(".json")).write_text(
        json.dumps(
            {
                "alpha": ALPHA,
                "replications": args.reps,
                "n_dates": N_DATES,
                "n_generators": N_GENERATORS,
                "measured_on_the_real_panel": {
                    k: {"skew": v[0], "rho": v[1]} for k, v in REAL.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("")
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
