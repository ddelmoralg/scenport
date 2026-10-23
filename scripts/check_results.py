#!/usr/bin/env python
"""Refuse to hand over results that cannot be true.

    python scripts/check_results.py --config configs/base.yaml

The first full run of this pipeline finished with every SLURM task green and 166
tests passing, and produced numbers that were measuring the wrong thing. Scenario
sets held 30-day returns of 1e14, which decided the portfolio on their own, and
the CVaR relaxation was scored against allocators as though it were one, so it
beat branch and bound on every date.

Neither was a programming error, so no unit test was ever going to catch them.
Both were visible in the first table anyone printed, if anyone had asked whether
the numbers could be true. That is what this does, on the outputs rather than on
the code.

Exit code 1 if any check fails, so the job chain stops instead of publishing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenport import config as cfgmod  # noqa: E402
from scenport.forecast.base import MAX_CUMULATIVE_RETURN  # noqa: E402

log = logging.getLogger("check")

#: A mean-CVaR objective is a return over one horizon. Anything past this is
#: arithmetic, not a portfolio.
PLAUSIBLE_OBJECTIVE = 10.0

#: Report any generator that touches the ceiling at all, and refuse one that is
#: being reshaped by it rather than trimmed.
#:
#: The warning threshold is set from the run that made this necessary. AR-GARCH
#: clipped 3.3e-4 of its points, LightGBM 4.9e-4, DeepAR 2.8e-4 and the
#: transformer 2.1e-4, while ARIMA clipped 7.8e-7. An order of magnitude below the
#: first group separates the heavy-tailed generators from the well behaved ones,
#: which is the distinction worth surfacing: a generator leaning on the ceiling is
#: telling you something about its tails.
CLIP_SHARE_WARN = 1e-5
CLIP_SHARE_FAIL = 0.02


def _known_forecasters() -> set[str]:
    """Forecaster names, so ablation directories can be told apart from runs.

    The ablations write to <forecaster>_<tag>, and they change the constraint set
    on purpose, so checking them against the base config reports violations that
    are the point of the experiment.
    """
    from scenport.forecast import base as forecast_base
    from scenport.plots import style

    try:
        import scenport.forecast.deep  # noqa: F401
    except Exception:  # torch is optional
        pass

    return set(forecast_base.available()) | set(style.FORECASTER_ORDER)


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def check(self, ok: bool, message: str) -> None:
        if ok:
            log.info("  ok    %s", message)
        else:
            log.error("  FAIL  %s", message)
            self.failures.append(message)

    def warn(self, ok: bool, message: str) -> None:
        if ok:
            log.info("  ok    %s", message)
        else:
            log.warning("  warn  %s", message)
            self.warnings.append(message)


def check_scenarios(runs: Path, report: Report) -> None:
    """The ceiling held, and who leaned on it."""
    log.info("scenario sets")
    store = runs / "scenarios"
    if not store.exists():
        report.warn(False, "no scenario sets on disk, skipping")
        return

    import json

    # Counted from the paths rather than read from the metadata. A clipped point
    # sits exactly on the ceiling, so the data carries the answer whether or not
    # anything wrote it down, and sets produced before the metadata existed still
    # get measured. Every tenth file, because this is 1.5 GB of scenarios.
    files = sorted(store.rglob("*.npz"))[::10]
    rows = []
    for npz in files:
        try:
            with np.load(npz, allow_pickle=True) as z:
                paths = z["paths"]
                name = str(z["forecaster"]) if "forecaster" in z else npz.parent.parent.name
                ceiling = MAX_CUMULATIVE_RETURN
                if "meta" in z:
                    try:
                        ceiling = float(
                            json.loads(str(z["meta"])).get("max_cumulative_return", ceiling)
                        )
                    except ValueError:
                        pass
                touching = int(np.isclose(paths, ceiling).sum() + np.isclose(paths, -1.0).sum())
                rows.append(
                    {
                        "forecaster": name,
                        "clipped_share": touching / paths.size,
                        # The largest value as well as the share at the bound:
                        # a generator that bypasses the ceiling leaves nothing
                        # on it to count.
                        "beyond": float(np.nanmax(paths)),
                        "below": float(np.nanmin(paths)),
                    }
                )
        except Exception:  # a truncated set is a different problem
            continue
    if not rows:
        report.warn(False, "no readable scenario sets")
        return

    frame = pd.DataFrame(rows)
    worst = frame.groupby("forecaster")[["beyond", "below"]].agg({"beyond": "max", "below": "min"})
    for name, row in worst.iterrows():
        report.check(
            row["beyond"] <= MAX_CUMULATIVE_RETURN * (1 + 1e-6),
            f"{name}: largest simulated return is {row['beyond']:.4g}, the ceiling is "
            f"{MAX_CUMULATIVE_RETURN:g}",
        )
        report.check(
            row["below"] >= -1.0 - 1e-6,
            f"{name}: smallest simulated return is {row['below']:.4g}; a long-only "
            "position cannot lose more than it cost",
        )

    by_forecaster = frame.groupby("forecaster")["clipped_share"].mean()
    for name, share in by_forecaster.sort_values(ascending=False).items():
        report.warn(
            share < CLIP_SHARE_WARN,
            f"{name}: {share:.2%} of simulated points hit the {MAX_CUMULATIVE_RETURN:g}x ceiling",
        )
        report.check(
            share < CLIP_SHARE_FAIL,
            f"{name}: clipping {share:.2%} is reshaping the distribution, not trimming it",
        )


def check_objective_scale(runs: Path, report: Report) -> None:
    """The number that gave the first run away: an objective of 38,054."""
    log.info("objective scale")
    path = runs / "exact" / "gaps.parquet"
    if not path.exists():
        report.warn(False, "no gap study on disk, skipping")
        return

    frame = pd.read_parquet(path)
    worst = float(frame["objective"].abs().max())
    report.check(
        worst < PLAUSIBLE_OBJECTIVE,
        f"largest objective seen is {worst:.4g}; a horizon return above "
        f"{PLAUSIBLE_OBJECTIVE:g} is not a portfolio",
    )


def check_nothing_beats_the_optimum(runs: Path, report: Report) -> None:
    """The other thing that gave it away: a heuristic beating branch and bound."""
    log.info("optimality gaps")
    path = runs / "exact" / "gaps.parquet"
    if not path.exists():
        report.warn(False, "no gap study on disk, skipping")
        return

    frame = pd.read_parquet(path)
    kind = frame["kind"] if "kind" in frame else pd.Series("candidate", index=frame.index)
    candidates = frame[kind == "candidate"]
    if candidates.empty:
        report.warn(False, "no candidate rows in the gap study")
        return

    closed = frame[(kind == "exact") & (frame.get("status", "optimal") == "optimal")]
    if closed.empty:
        report.warn(False, "branch and bound never closed, so gaps are against an incumbent")

    beaten = candidates[candidates["relative_gap"] < -1e-6]
    report.check(
        beaten.empty,
        "no candidate beats the exact optimum"
        + (
            ""
            if beaten.empty
            else f"; {sorted(set(beaten['allocator']))} did, so they are solving "
            "a different problem"
        ),
    )

    relaxation = frame[kind == "relaxation"]
    if not relaxation.empty:
        report.check(
            bool((relaxation["relative_gap"] >= -1e-6).all()),
            "the relaxation bound sits above the exact optimum, as it must",
        )


def check_feasibility(runs: Path, cfg, report: Report) -> None:
    """Every portfolio in the backtest is one somebody could actually hold."""
    log.info("portfolio feasibility")
    cardinality = cfg.get("alloc.cardinality")
    w_min = float(cfg.get("alloc.w_min", 0.0))
    w_max = float(cfg.get("alloc.w_max", 1.0))

    from scenport import alloc

    known = _known_forecasters()
    checked = 0
    skipped_ablations = 0
    offenders: dict[str, str] = {}
    for path in sorted((runs / "backtests").rglob("weights.parquet")):
        # Ablation runs deliberately move the constraints, so judging them against
        # the base config says nothing. They live in <forecaster>_<tag> directories.
        if path.parent.parent.name not in known:
            skipped_ablations += 1
            continue
        if path.parent.name.rsplit("_seed", 1)[0] in alloc.UNCONSTRAINED:
            continue

        weights = pd.read_parquet(path)
        if weights.empty:
            continue
        label = f"{path.parent.parent.name}/{path.parent.name}"
        values = weights.to_numpy(float)
        values = np.nan_to_num(values)
        held = values > 1e-9

        if cardinality is not None and held.sum(axis=1).max() > int(cardinality):
            offenders[label] = f"holds {int(held.sum(axis=1).max())} names, limit is {cardinality}"
        elif values.max() > w_max + 1e-6:
            offenders[label] = f"max weight {values.max():.4f} over the cap {w_max}"
        elif held.any() and values[held].min() < w_min - 1e-6:
            offenders[label] = f"min held weight {values[held].min():.4f} under {w_min}"
        checked += 1

    if not checked:
        report.warn(False, "no stored weights to check")
        return
    report.check(
        not offenders,
        f"all {checked} weight histories satisfy the constraints"
        f" ({skipped_ablations} ablation runs skipped, they carry their own)"
        + ("" if not offenders else f"; {list(offenders.items())[:3]}"),
    )


def check_sensitivity(artifacts: Path, report: Report) -> None:
    """The headline, sanity checked rather than trusted."""
    log.info("sensitivity")
    path = artifacts / "sensitivity.csv"
    if not path.exists():
        report.warn(False, "no sensitivity table, skipping")
        return

    table = pd.read_csv(path)
    report.check(
        len(table) <= 12,
        f"{len(table)} generators in the sensitivity table; there are eleven forecasters, "
        "so anything more means ablation runs are being counted as generators",
    )
    if "displacement" in table:
        report.check(
            bool((table["displacement"] <= 2.0 + 1e-9).all()),
            "L1 displacement stays inside its range of 0 to 2",
        )
    if "overlap" in table:
        report.check(
            bool(((table["overlap"] >= 0) & (table["overlap"] <= 1)).all()),
            "support overlap stays inside 0 to 1",
        )



def check_no_silent_failures(runs: Path, report: Report) -> None:
    """Flag runs that raised.

    A failed run still writes a row, with a plausible wall-clock time and no
    objective, and anything that plots or averages timings will read it as a
    solve. Reporting failures here keeps them out of the figures.
    """
    log.info("failed runs")
    found = False
    for path in sorted(runs.rglob("*.parquet")):
        frame = pd.read_parquet(path)
        if "status" not in frame:
            continue
        bad = frame["status"].astype(str).str.startswith("failed")
        if bad.any():
            found = True
            reasons = sorted(set(frame.loc[bad, "status"]))
            report.check(
                False,
                f"{path.name}: {int(bad.sum())} of {len(frame)} rows failed "
                f"({', '.join(reasons)})",
            )
    if not found:
        report.check(True, "no run recorded a failure")


def check_cvar_forms_agree(cfg, report: Report) -> None:
    """The two CVaR implementations have to coincide, or the gaps mean nothing.

    The metaheuristics take the mean of the m = round((1-alpha)S) worst scenarios
    with a partial sort, because the objective is evaluated once per candidate per
    generation and a full sort is the bottleneck. The linear programme and the
    mixed-integer model use the Rockafellar-Uryasev form. The two agree exactly
    when (1-alpha)S is a whole number and only approximately otherwise, so a gap
    between a metaheuristic and the exact optimum would then be partly a
    difference of definition rather than of search.
    """
    log.info("CVaR consistency")
    alpha = float(cfg.get("alloc.cvar_alpha", 0.95))
    for label, n_scenarios in (
        ("backtest", int(cfg.get("scenarios.n_paths", 1000))),
        ("gap study", 400),
    ):
        tail = (1.0 - alpha) * n_scenarios
        report.check(
            abs(tail - round(tail)) < 1e-9,
            f"{label}: (1-{alpha}) x {n_scenarios} = {tail:g} is a whole number, so the "
            "empirical and Rockafellar-Uryasev forms coincide",
        )


def check_universe(panel_dir: Path, cfg, report: Report) -> None:
    log.info("universe")
    path = panel_dir / "universe.parquet"
    if not path.exists():
        report.warn(False, "no universe on disk, skipping")
        return

    from scenport.data.universe import read as read_universe

    universe = read_universe(path)
    size = int(cfg.get("universe.size", 15))
    short = [d for d in universe.dates if len(universe.at(d)) < size]
    report.warn(
        not short,
        f"{len(short)} of {len(universe.dates)} dates have fewer than {size} assets",
    )
    report.check(
        bool(universe.dates), "the universe has at least one usable rebalancing date"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument(
        "--warn-only", action="store_true", help="report but always exit 0"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = cfgmod.repo_root()
    cfg = cfgmod.load(args.config, args.overrides)
    runs = cfg.resolve("paths.runs", root)
    artifacts = cfg.resolve("paths.artifacts", root)
    panel_dir = cfg.resolve("paths.panel", root)

    report = Report()
    check_cvar_forms_agree(cfg, report)
    check_universe(panel_dir, cfg, report)
    check_scenarios(runs, report)
    check_objective_scale(runs, report)
    check_nothing_beats_the_optimum(runs, report)
    check_feasibility(runs, cfg, report)
    check_sensitivity(artifacts, report)
    check_no_silent_failures(runs, report)

    log.info("")
    if report.warnings:
        log.warning("%d warning(s)", len(report.warnings))
    if report.failures:
        log.error("%d check(s) failed:", len(report.failures))
        for line in report.failures:
            log.error("  - %s", line)
        if not args.warn_only:
            log.error("")
            log.error("These results are not fit to publish. Fix the cause, do not")
            log.error("relax the check.")
            return 1
    else:
        log.info("everything checks out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
