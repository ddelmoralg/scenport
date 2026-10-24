#!/usr/bin/env python
"""The inference added in the final revision, on results that already exist.

    python scripts/inference_audit.py --build    # on the cluster, needs runs/backtests
    python scripts/inference_audit.py            # anywhere, from the saved panel

The plan is in docs/plan_inferencia_revision_2026-10-24.md and was written before
this was run, the addendum included. Nothing here reruns a forecast or an
allocation. --build saves the generator-date panel the transmission regression
uses, so everything after it can be reproduced from the repository alone.

Writes, under artifacts/:
  transmission_panel_data.parquet  the 935-row panel (with --build)
  inference_family.csv             six diagnostics, three tests, Holm, drop-one-date
  inference_noexposure.csv         the same without the exposure control
  inference_beta_market.csv        the same with exposure times the market return
  inference_lagged.csv             the same with each diagnostic averaged over the
                                   twelve previous dates, which a user could know
  inference_drop_permutation.csv   the drop-one check under the permutation test
  inference_compare.csv            coverage error against each other diagnostic
  inference_miss80.csv             what earlier versions called coverage error
  dispersion_by_date.csv           how far apart the generators' returns were each date
  solver_precision.csv             intervals for the solver gaps, absolute and relative
  solver_pairs.csv                 direct comparisons between the three best searches
  inference_audit.json             the numbers the paper quotes
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenport import config as cfgmod  # noqa: E402
from scenport.stats import tests  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
SCORES = ["mase", "crps", "pinball", "coverage_error", "energy_score", "variogram_score"]
BONFERRONI_T = float(stats.norm.ppf(1 - 0.05 / (2 * len(SCORES))))
EPISODE = [pd.Timestamp("2021-02-01"), pd.Timestamp("2021-03-01"), pd.Timestamp("2021-04-01")]
CONTROLS = {
    "exposure": ["exposure"],
    "none": [],
    # Within a date the market return is fixed, so exposure acts through
    # exposure x market return, whose sign changes from month to month. A single
    # slope on exposure alone cannot follow that.
    "exposure_market": ["exposure", "exposure_market"],
}


def holm(p: pd.Series) -> pd.Series:
    """Holm step-down adjusted p values."""
    order = p.sort_values()
    m = len(order)
    adjusted = np.maximum.accumulate([(m - i) * v for i, v in enumerate(order.to_numpy())])
    return pd.Series(np.minimum(adjusted, 1.0), index=order.index).reindex(p.index)


def design(panel: pd.DataFrame, score: str, control: str = "exposure"):
    regressors = [score] + CONTROLS[control]
    demeaned = tests.within_transform(panel, ["net_return", *regressors], "rebalance")
    return (demeaned["net_return"].to_numpy(float), demeaned[regressors].to_numpy(float),
            panel["rebalance"].to_numpy())


def beta0(y, x):
    return float(np.linalg.lstsq(x, y, rcond=None)[0][0])


def within_permutation(y, x, clusters, reps: int, seed: int) -> float:
    """Within-date permutation of the first regressor, vectorised.

    Same null as tests.permutation_within_clusters; this version draws a whole
    permutation with one lexsort so the drop-one loop stays affordable.
    """
    observed = abs(beta0(y, x))
    codes = pd.factorize(clusters)[0]
    sorted_idx = np.argsort(codes, kind="stable")
    rng = np.random.default_rng(seed)
    shuffled = x.copy()
    extreme = 0
    for _ in range(reps):
        order = np.lexsort((rng.random(len(codes)), codes))
        shuffled[sorted_idx, 0] = x[order, 0]
        extreme += abs(beta0(y, shuffled)) >= observed
    return (extreme + 1) / (reps + 1)


def block_permutation(panel: pd.DataFrame, score: str, reps: int, seed: int,
                      control: str = "exposure") -> float:
    """Relabel generators with one permutation for every date.

    The whole series of a diagnostic travels with its generator, so the
    persistence of each generator's score and return over the dates is kept.
    That is the dependence the within-date permutation ignores.
    """
    y, x, _ = design(panel, score, control)
    observed = abs(beta0(y, x))
    dates = panel["rebalance"].to_numpy()
    names = panel["forecaster"].to_numpy()
    generators = list(np.unique(names))
    wide = pd.DataFrame({"d": dates, "g": names, "v": x[:, 0]}).pivot(index="d", columns="g",
                                                                      values="v")
    r_idx = np.array([wide.index.get_loc(d) for d in dates])
    c_idx = np.array([generators.index(g) for g in names])
    matrix = wide[generators].to_numpy()

    rng = np.random.default_rng(seed)
    extreme = 0
    shuffled = x.copy()
    for _ in range(reps):
        perm = rng.permutation(len(generators))
        shuffled[:, 0] = matrix[r_idx, perm[c_idx]]
        extreme += abs(beta0(y, shuffled)) >= observed
    return (extreme + 1) / (reps + 1)


def drop_one_date(y, x, clusters) -> np.ndarray:
    out = []
    for date in np.unique(clusters):
        keep = clusters != date
        out.append(tests.cluster_robust_ols(y[keep], x[keep], clusters[keep])["t_stat"][0])
    return np.asarray(out)


def family(panel: pd.DataFrame, control: str, block_reps: int,
           scores: list[str] = SCORES) -> pd.DataFrame:
    rows = []
    for score in scores:
        y, x, clusters = design(panel, score, control)
        fit = tests.cluster_robust_ols(y, x, clusters)
        ci = tests.cluster_bootstrap_ci(y, x, clusters, column=0, reps=2000, seed=0)
        t_drop = drop_one_date(y, x, clusters)
        row = {
            "score": score,
            "beta": float(fit["beta"][0]),
            "ci_lower": ci["lower"],
            "ci_upper": ci["upper"],
            "t_stat": float(fit["t_stat"][0]),
            "p_clustered": float(fit["p_value"][0]),
            "p_permutation": within_permutation(y, x, clusters, 5000, seed=0),
            "min_abs_t_drop_one": float(np.min(np.abs(t_drop))),
            "date_min_t": str(pd.Timestamp(np.unique(clusters)[np.argmin(np.abs(t_drop))]).date()),
            "n_obs": int(len(y)),
            "n_dates": int(len(np.unique(clusters))),
        }
        if block_reps:
            row["p_block"] = block_permutation(panel, score, block_reps, seed=0, control=control)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("score")
    if len(scores) > 1:
        table["p_clustered_holm"] = holm(table["p_clustered"])
        table["p_permutation_holm"] = holm(table["p_permutation"])
        if block_reps:
            table["p_block_holm"] = holm(table["p_block"])
        table["drop_one_196"] = table["min_abs_t_drop_one"] > 1.96
        table["drop_one_bonferroni"] = table["min_abs_t_drop_one"] > BONFERRONI_T
        if block_reps:
            table["passes_all_adjusted"] = ((table["p_clustered_holm"] < 0.05)
                                            & (table["p_block_holm"] < 0.05)
                                            & table["drop_one_bonferroni"])
    return table.reset_index()


def drop_permutation(panel: pd.DataFrame, reps: int) -> pd.DataFrame:
    """The drop-one check repeated under the within-date permutation test."""
    rows = []
    for score in SCORES:
        y, x, clusters = design(panel, score)
        p_drop = {}
        for date in np.unique(clusters):
            keep = clusters != date
            p_drop[pd.Timestamp(date)] = within_permutation(y[keep], x[keep], clusters[keep],
                                                            reps, seed=0)
        worst = max(p_drop, key=p_drop.get)
        keep = ~pd.Series(clusters).isin(EPISODE).to_numpy()
        rows.append({
            "score": score,
            "max_p_drop_one": p_drop[worst],
            "date_max_p": str(worst.date()),
            "p_without_april_2021": p_drop[pd.Timestamp("2021-04-01")],
            "p_without_feb_to_apr_2021": within_permutation(y[keep], x[keep], clusters[keep],
                                                            reps, seed=0),
        })
    return pd.DataFrame(rows)


def lagged(panel: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Each diagnostic replaced by its mean over the generator's previous dates.

    The contemporaneous diagnostic is scored against the same holding period
    whose return it is regressed on, so nobody choosing a generator could have
    computed it. The lagged one only uses outcomes already observed.
    """
    out = panel.sort_values(["forecaster", "rebalance"]).copy()
    for score in SCORES:
        out[score] = (out.groupby("forecaster")[score]
                      .transform(lambda s: s.shift(1).rolling(window, min_periods=window).mean()))
    out = out.dropna(subset=SCORES)
    complete = out.groupby("rebalance")["forecaster"].transform("size") == panel["forecaster"].nunique()
    return out[complete].reset_index(drop=True)


def standardised_betas(panel: pd.DataFrame) -> dict:
    """Coefficient per within-date standard deviation of each diagnostic."""
    out = {}
    for score in SCORES:
        y, x, _ = design(panel, score)
        x = x.copy()
        x[:, 0] = x[:, 0] / x[:, 0].std()
        out[score] = beta0(y, x)
    return out


def compare(panel: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    observed = standardised_betas(panel)
    dates = np.unique(panel["rebalance"].to_numpy())
    groups = {d: panel[panel["rebalance"] == d] for d in dates}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        pick = rng.choice(dates, size=len(dates), replace=True)
        boot = pd.concat([groups[d].assign(rebalance=i) for i, d in enumerate(pick)],
                         ignore_index=True)
        draws.append(standardised_betas(boot))
    draws = pd.DataFrame(draws)
    rows = []
    for other in [s for s in SCORES if s != "coverage_error"]:
        diff = draws["coverage_error"] - draws[other]
        p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
        rows.append({
            "other": other,
            "std_beta_coverage": observed["coverage_error"],
            "std_beta_other": observed[other],
            "difference": observed["coverage_error"] - observed[other],
            "ci_lower": float(diff.quantile(0.025)),
            "ci_upper": float(diff.quantile(0.975)),
            "p_boot": float(max(p, 1 / reps)),
        })
    table = pd.DataFrame(rows).set_index("other")
    table["p_boot_holm"] = holm(table["p_boot"])
    return table.reset_index()


def dispersion(panel: pd.DataFrame) -> pd.DataFrame:
    by_date = panel.groupby("rebalance")["net_return"].agg(["std", "min", "max"])
    by_date["range"] = by_date["max"] - by_date["min"]
    by_date["rank_std"] = by_date["std"].rank(ascending=False).astype(int)
    return by_date.reset_index()


def solver_precision(gaps: pd.DataFrame, reps: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gaps in objective units, with the relative gap as a median.

    The relative gap divides by |f*|, which is close to zero on one instance and
    then dominates every mean. The absolute gap does not have that problem.
    """
    gaps = gaps.copy()
    optimum = gaps[gaps["allocator"] == "milp"].set_index("rebalance")["objective"]
    gaps["absolute_gap"] = optimum.reindex(gaps["rebalance"]).to_numpy() - gaps["objective"]
    # One value per allocator and instance: the stochastic searches are averaged
    # over their ten seeds, so the bootstrap resamples instances and not seeds.
    absolute = gaps.groupby(["allocator", "rebalance"])["absolute_gap"].mean().unstack(0)
    relative = gaps.groupby(["allocator", "rebalance"])["relative_gap"].mean().unstack(0)
    rng = np.random.default_rng(seed)
    n = len(absolute)
    picks = [rng.integers(0, n, n) for _ in range(reps)]
    boots = pd.DataFrame([absolute.iloc[p].mean().to_numpy() for p in picks],
                         columns=absolute.columns)
    rows = []
    for allocator in absolute.columns:
        rows.append({
            "allocator": allocator,
            "mean_absolute_gap": float(absolute[allocator].mean()),
            "ci_lower": float(boots[allocator].quantile(0.025)),
            "ci_upper": float(boots[allocator].quantile(0.975)),
            "median_absolute_gap": float(gaps.loc[gaps["allocator"] == allocator,
                                                  "absolute_gap"].median()),
            "worst_absolute_gap": float(gaps.loc[gaps["allocator"] == allocator,
                                                 "absolute_gap"].abs().max()),
            "median_relative_gap": float(gaps.loc[gaps["allocator"] == allocator,
                                                  "relative_gap"].median()),
            "mean_relative_gap": float(relative[allocator].mean()),
        })
    for search in ("cmaes", "ga", "de", "pso"):
        diff = boots["cvar_lp"] - boots[search]
        rows.append({"allocator": f"cvar_lp minus {search}",
                     "mean_absolute_gap": float((absolute["cvar_lp"] - absolute[search]).mean()),
                     "ci_lower": float(diff.quantile(0.025)),
                     "ci_upper": float(diff.quantile(0.975))})
    precision = pd.DataFrame(rows)
    precision.attrs["smallest_optimum"] = float(optimum.abs().min())
    precision.attrs["smallest_optimum_date"] = str(optimum.abs().idxmin().date())
    precision.attrs["median_optimum"] = float(optimum.median())

    pairs = []
    for a, b in combinations(["cmaes", "ga", "de"], 2):
        d = absolute[a] - absolute[b]
        res = stats.wilcoxon(absolute[a], absolute[b], zero_method="zsplit")
        pairs.append({"pair": f"{a} vs {b}", "mean_difference": float(d.mean()),
                      "median_difference": float(d.median()), "p_wilcoxon": float(res.pvalue)})
    pairs = pd.DataFrame(pairs).set_index("pair")
    pairs["p_holm"] = holm(pairs["p_wilcoxon"])
    return precision, pairs.reset_index()


def market_period_returns(panel_dir: Path, dates) -> pd.Series:
    """Return of the cap-weighted index over each holding period.

    The same index the exposure betas are estimated against, compounded from the
    day after a rebalancing date to the next one.
    """
    from scenport.data.panel import Panel

    panel = Panel.read(panel_dir)
    returns = panel.returns()
    caps = panel.mcap.reindex(returns.index).shift(1)
    market = (returns * caps.div(caps.sum(axis=1), axis=0)).sum(axis=1)
    dates = sorted(pd.DatetimeIndex(dates))
    out = {}
    for start, end in zip(dates, dates[1:] + [market.index.max()]):
        window = market.loc[start + pd.Timedelta(days=1): end]
        out[start] = float(np.prod(1.0 + window.to_numpy()) - 1.0)
    return pd.Series(out, name="market_return")


def build(args) -> None:
    from aggregate import build_transmission_panel, load_scores

    cfg = cfgmod.load(args.config)
    runs = cfg.resolve("paths.runs", ROOT)
    panel_dir = cfg.resolve("paths.panel", ROOT)
    per_asset, joint = load_scores(runs, panel_dir, int(cfg["backtest.train_days"]))
    panel, controlled = build_transmission_panel(per_asset, joint, runs, "ga", panel_dir)
    if not controlled:
        raise SystemExit("exposure control missing; the panel would not match the paper")
    market = market_period_returns(panel_dir, panel["rebalance"].unique())
    panel["market_return"] = market.reindex(pd.DatetimeIndex(panel["rebalance"])).to_numpy()
    panel.to_parquet(ART / "transmission_panel_data.parquet", index=False)
    print("panel:", panel.shape)


def as_plain(value):
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--block-reps", type=int, default=9999)
    parser.add_argument("--drop-reps", type=int, default=2000)
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()

    if args.build:
        build(args)

    panel = pd.read_parquet(ART / "transmission_panel_data.parquet")
    panel["rebalance"] = pd.to_datetime(panel["rebalance"])
    panel["exposure_market"] = panel["exposure"] * panel["market_return"]
    assert panel.groupby("rebalance").size().eq(panel["forecaster"].nunique()).all(), \
        "the block permutation needs every generator on every date"

    fam = family(panel, "exposure", args.block_reps)
    noexp = family(panel, "none", args.block_reps)
    beta_market = family(panel, "exposure_market", args.block_reps)
    lag_panel = lagged(panel)
    lag = family(lag_panel, "exposure", args.block_reps)
    miss = family(panel, "exposure", args.block_reps, scores=["miss_80"])
    drop_perm = drop_permutation(panel, args.drop_reps)
    comp = compare(panel, args.boot, seed=0)
    disp = dispersion(panel)
    precision, pairs = solver_precision(pd.read_parquet(ROOT / "runs" / "exact" / "gaps.parquet"),
                                        args.boot, seed=0)

    fam.to_csv(ART / "inference_family.csv", index=False)
    noexp.to_csv(ART / "inference_noexposure.csv", index=False)
    beta_market.to_csv(ART / "inference_beta_market.csv", index=False)
    lag.to_csv(ART / "inference_lagged.csv", index=False)
    miss.to_csv(ART / "inference_miss80.csv", index=False)
    drop_perm.to_csv(ART / "inference_drop_permutation.csv", index=False)
    comp.to_csv(ART / "inference_compare.csv", index=False)
    disp.to_csv(ART / "dispersion_by_date.csv", index=False)
    precision.to_csv(ART / "solver_precision.csv", index=False)
    pairs.to_csv(ART / "solver_pairs.csv", index=False)

    april = disp[disp["rebalance"] == pd.Timestamp("2021-04-01")].iloc[0]
    head = {
        "bonferroni_t": BONFERRONI_T,
        "block_permutations": args.block_reps,
        "possible_relabelings": factorial(panel["forecaster"].nunique()),
        "passing_all_adjusted": {
            "exposure": fam.loc[fam["passes_all_adjusted"], "score"].tolist(),
            "none": noexp.loc[noexp["passes_all_adjusted"], "score"].tolist(),
            "exposure_market": beta_market.loc[beta_market["passes_all_adjusted"],
                                               "score"].tolist(),
            "lagged": lag.loc[lag["passes_all_adjusted"], "score"].tolist(),
        },
        "lagged_dates": int(lag_panel["rebalance"].nunique()),
        "april_2021_dispersion": float(april["std"]),
        "median_dispersion": float(disp["std"].median()),
        "april_2021_rank_by_generator_dispersion": int(april["rank_std"]),
        "smallest_optimum": precision.attrs["smallest_optimum"],
        "smallest_optimum_date": precision.attrs["smallest_optimum_date"],
        "median_optimum": precision.attrs["median_optimum"],
        "n_dates": int(len(disp)),
    }
    (ART / "inference_audit.json").write_text(json.dumps(head, indent=2, default=as_plain))

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    for title, frame in [("exposure", fam), ("no control", noexp),
                         ("exposure x market", beta_market), ("lagged 12", lag),
                         ("miss rate 80", miss), ("drop one, permutation", drop_perm),
                         ("compare", comp), ("solver", precision), ("pairs", pairs)]:
        print(f"\n== {title}")
        print(frame.round(4).to_string(index=False))
    print(disp.sort_values("rank_std").head(4).to_string(index=False))
    print(json.dumps(head, indent=2, default=as_plain))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
