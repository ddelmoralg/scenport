#!/usr/bin/env python
"""The inference added in the final revision, on results that already exist.

    python scripts/inference_audit.py --build    # on the cluster, needs runs/backtests
    python scripts/inference_audit.py            # anywhere, from the saved panel

The plan is in docs/plan_inferencia_revision_2026-10-24.md and was written before
this was run. Nothing here reruns a forecast or an allocation. --build saves the
generator-date panel the transmission regression uses, so everything after it can
be reproduced from the repository alone.

Writes, under artifacts/:
  transmission_panel_data.parquet  the 935-row panel (with --build)
  inference_family.csv             six diagnostics, three tests, Holm, drop-one-date
  inference_compare.csv            coverage error against each other diagnostic
  inference_noexposure.csv         the same regressions without the exposure control
  dispersion_by_date.csv           how far apart the generators' returns were each date
  solver_precision.csv             intervals for the solver gaps
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


def holm(p: pd.Series) -> pd.Series:
    """Holm step-down adjusted p values."""
    order = p.sort_values()
    m = len(order)
    adjusted = np.maximum.accumulate([(m - i) * v for i, v in enumerate(order.to_numpy())])
    return pd.Series(np.minimum(adjusted, 1.0), index=order.index).reindex(p.index)


def design(panel: pd.DataFrame, score: str, exposure: bool):
    regressors = [score] + (["exposure"] if exposure else [])
    demeaned = tests.within_transform(panel, ["net_return", *regressors], "rebalance")
    return (demeaned["net_return"].to_numpy(float), demeaned[regressors].to_numpy(float),
            panel["rebalance"].to_numpy())


def beta0(y, x):
    return float((np.linalg.pinv(x.T @ x) @ (x.T @ y))[0])


def block_permutation(panel: pd.DataFrame, score: str, reps: int, seed: int) -> float:
    """Relabel generators with one permutation for every date.

    The whole series of a diagnostic travels with its generator, so the
    persistence of each generator's score and return over the 85 dates is kept.
    That is the dependence the within-date permutation ignores.
    """
    y, x, _ = design(panel, score, exposure=True)
    observed = abs(beta0(y, x))
    dates = panel["rebalance"].to_numpy()
    names = panel["forecaster"].to_numpy()
    generators = np.unique(names)
    wide = pd.DataFrame({"d": dates, "g": names, "v": x[:, 0]}).pivot(index="d", columns="g",
                                                                      values="v")
    rows = [(wide.index.get_loc(d), list(generators).index(g)) for d, g in zip(dates, names)]
    r_idx = np.array([r for r, _ in rows])
    c_idx = np.array([c for _, c in rows])
    matrix = wide[generators].to_numpy()

    rng = np.random.default_rng(seed)
    extreme, drawn = 0, 0
    shuffled = x.copy()
    for _ in range(reps):
        perm = rng.permutation(len(generators))
        shuffled[:, 0] = matrix[r_idx, perm[c_idx]]
        drawn += 1
        extreme += abs(beta0(y, shuffled)) >= observed
    return (extreme + 1) / (drawn + 1)


def drop_one_date(y, x, clusters) -> np.ndarray:
    out = []
    for date in np.unique(clusters):
        keep = clusters != date
        out.append(tests.cluster_robust_ols(y[keep], x[keep], clusters[keep])["t_stat"][0])
    return np.asarray(out)


def family(panel: pd.DataFrame, exposure: bool, block_reps: int) -> pd.DataFrame:
    rows = []
    for score in SCORES:
        y, x, clusters = design(panel, score, exposure)
        fit = tests.cluster_robust_ols(y, x, clusters)
        perm = tests.permutation_within_clusters(y, x, clusters, column=0, reps=5000, seed=0)
        t_drop = drop_one_date(y, x, clusters)
        row = {
            "score": score,
            "beta": float(fit["beta"][0]),
            "t_stat": float(fit["t_stat"][0]),
            "p_clustered": float(fit["p_value"][0]),
            "p_permutation": float(perm.p_value),
            "min_abs_t_drop_one": float(np.min(np.abs(t_drop))),
            "date_min_t": str(pd.Timestamp(np.unique(clusters)[np.argmin(np.abs(t_drop))]).date()),
        }
        if block_reps:
            row["p_block"] = block_permutation(panel, score, block_reps, seed=0)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("score")
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


def standardised_betas(panel: pd.DataFrame) -> dict:
    """Coefficient per within-date standard deviation of each diagnostic."""
    out = {}
    for score in SCORES:
        y, x, _ = design(panel, score, exposure=True)
        sd = x[:, 0].std()
        x = x.copy()
        x[:, 0] = x[:, 0] / sd
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
    # One value per allocator and instance: the stochastic searches are averaged
    # over their ten seeds, so the bootstrap resamples instances and not seeds.
    per_instance = gaps.groupby(["allocator", "rebalance"])["relative_gap"].mean().unstack(0)
    rng = np.random.default_rng(seed)
    n = len(per_instance)
    boots = np.array([per_instance.iloc[rng.integers(0, n, n)].mean().to_numpy()
                      for _ in range(reps)])
    boots = pd.DataFrame(boots, columns=per_instance.columns)
    rows = []
    for allocator in per_instance.columns:
        rows.append({"allocator": allocator, "mean_gap": float(per_instance[allocator].mean()),
                     "ci_lower": float(boots[allocator].quantile(0.025)),
                     "ci_upper": float(boots[allocator].quantile(0.975))})
    for search in ("cmaes", "ga", "de", "pso"):
        diff = boots["cvar_lp"] - boots[search]
        rows.append({"allocator": f"cvar_lp minus {search}",
                     "mean_gap": float((per_instance["cvar_lp"] - per_instance[search]).mean()),
                     "ci_lower": float(diff.quantile(0.025)),
                     "ci_upper": float(diff.quantile(0.975))})
    precision = pd.DataFrame(rows)

    pairs = []
    for a, b in combinations(["cmaes", "ga", "de"], 2):
        d = per_instance[a] - per_instance[b]
        res = stats.wilcoxon(per_instance[a], per_instance[b], zero_method="zsplit")
        pairs.append({"pair": f"{a} vs {b}", "mean_difference": float(d.mean()),
                      "median_difference": float(d.median()), "p_wilcoxon": float(res.pvalue)})
    pairs = pd.DataFrame(pairs).set_index("pair")
    pairs["p_holm"] = holm(pairs["p_wilcoxon"])
    return precision, pairs.reset_index()


def build(args) -> None:
    from aggregate import build_transmission_panel, load_scores

    cfg = cfgmod.load(args.config)
    runs = cfg.resolve("paths.runs", ROOT)
    panel_dir = cfg.resolve("paths.panel", ROOT)
    per_asset, joint = load_scores(runs)
    panel, controlled = build_transmission_panel(per_asset, joint, runs, "ga", panel_dir)
    if not controlled:
        raise SystemExit("exposure control missing; the panel would not match the paper")
    panel.to_parquet(ART / "transmission_panel_data.parquet", index=False)
    print("panel:", panel.shape)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--block-reps", type=int, default=9999)
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()

    if args.build:
        build(args)

    panel = pd.read_parquet(ART / "transmission_panel_data.parquet")
    panel["rebalance"] = pd.to_datetime(panel["rebalance"])
    assert panel.groupby("rebalance").size().eq(panel["forecaster"].nunique()).all(), \
        "the block permutation needs every generator on every date"

    fam = family(panel, exposure=True, block_reps=args.block_reps)
    noexp = family(panel, exposure=False, block_reps=0)
    comp = compare(panel, args.boot, seed=0)
    disp = dispersion(panel)
    precision, pairs = solver_precision(pd.read_parquet(ROOT / "runs" / "exact" / "gaps.parquet"),
                                        args.boot, seed=0)

    fam.to_csv(ART / "inference_family.csv", index=False)
    noexp.to_csv(ART / "inference_noexposure.csv", index=False)
    comp.to_csv(ART / "inference_compare.csv", index=False)
    disp.to_csv(ART / "dispersion_by_date.csv", index=False)
    precision.to_csv(ART / "solver_precision.csv", index=False)
    pairs.to_csv(ART / "solver_pairs.csv", index=False)

    april = disp[disp["rebalance"] == pd.Timestamp("2021-04-01")].iloc[0]
    cov = fam.set_index("score").loc["coverage_error"]
    head = {
        "bonferroni_t": BONFERRONI_T,
        "block_permutations": args.block_reps,
        "possible_relabelings": factorial(panel["forecaster"].nunique()),
        "passing_all_adjusted": fam.loc[fam["passes_all_adjusted"], "score"].tolist(),
        "coverage": {k: (float(v) if isinstance(v, (int, float, np.floating, np.bool_)) else v)
                     for k, v in cov.items()},
        "april_2021_rank_by_generator_dispersion": int(april["rank_std"]),
        "n_dates": int(len(disp)),
    }
    (ART / "inference_audit.json").write_text(json.dumps(head, indent=2, default=str))

    pd.set_option("display.width", 200)
    print(fam.round(4).to_string(index=False))
    print()
    print(noexp.round(4).to_string(index=False))
    print()
    print(comp.round(4).to_string(index=False))
    print()
    print(disp.sort_values("rank_std").head(8).round(4).to_string(index=False))
    print()
    print(precision.round(4).to_string(index=False))
    print(pairs.round(5).to_string(index=False))
    print(json.dumps(head, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
