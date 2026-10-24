# scenport allocation instances

Cardinality-constrained mean-CVaR portfolio selection over simulated scenarios.

Each instance is a pair of files:

    <name>.npz     scenarios: (S, n) float32, horizon simple returns
                   w_prev:    (n,)  float32, weights carried in
    <name>.json    constraint parameters and provenance

The problem is

    max  (1-gamma) mean(R w) - gamma CVaR_alpha(R w) - cost_rate ||w - w_prev||_1
    s.t. sum(w) = 1
         0 <= w_i <= w_max
         w_i = 0 or w_i >= w_min
         ||w||_0 <= cardinality

with CVaR written as a positive loss in the Rockafellar-Uryasev form,

    CVaR_alpha = min_z  z + sum_s max(0, -(R w)_s - z) / ((1 - alpha) S),

which is the mean of the worst (1 - alpha) S scenario returns when that number is
whole and gives the last tail scenario a fractional weight when it is not (S = 250
at alpha = 0.95, for example).

Each header carries the answer we have for its own scenarios, with no thinning:

    best_known     objective of best_weights, recomputed from the npz
    best_weights   the portfolio that attains it, in the order of `assets`
    proven         HiGHS reported optimality within mip_rel_gap
    solver, mip_rel_gap, solve_seconds, status

so a result can be checked by evaluating best_weights, without trusting the number.
Improvements welcome; open an issue with the instance name, the weights and how you
got them.

## Families

`family` in the header says how the scenarios were produced.

- `generators`: simulated by one of the fitted forecasters named in `generator`, on
  the point-in-time universe for `rebalance_date`. Instances sharing a date share
  their assets and their carried-in weights, so they differ only in the scenario
  distribution.
- `scale`: assets drawn from the panel by market capitalisation on `rebalance_date`,
  with scenarios resampled in blocks from realised horizon returns. Use these for
  solver scaling. They are not forecasts and should not be read as any.
- `constraints`: a base instance with one constraint parameter moved. `varied` names
  which one.

## Reading one

    import json, numpy as np
    head = json.load(open("name.json"))
    data = np.load("name.npz")
    R, w_prev = data["scenarios"], data["w_prev"]

## What it covers, and what it does not

The `generators` family is at the point-in-time universe size, which is fifteen.
The `scale` family goes as far as the price panel allows, and that is the binding
limit: the panel holds 76 assets, of which about 71 carry a market capitalisation
on any given date, so instances stop around seventy. The OR-Library reaches 225.

That ceiling is the data source rather than the code. Price and market
capitalisation come from the Coin Metrics Community tier, which serves both
metrics for 87 assets; a licensed tier would lift it and the export takes the
larger universe without changes.

So this set is more varied than what came before it and not yet larger. Every
instance carries the scenario matrix rather than moments, which is the part that
makes gaps comparable at all, and eleven fitted generators give eleven different
tail and dependence structures over the same assets and the same dates.

## Why this set exists

The OR-Library portfolio files, which is what this problem class has been
benchmarked on since 2000, give per-asset means and standard deviations and a
correlation matrix. Moments, not scenarios. A CVaR objective cannot be evaluated on
moments, so anyone using them for this problem has to invent a distribution
consistent with them, and that choice is usually not reported. Two papers quoting
optimality gaps on port1 are therefore not necessarily quoting them on the same
instance.

SIPLIB does not fill the gap either. Its portfolio entry, PROBPORT, is
chance-constrained, and nothing in the collection carries a tail risk measure with
cardinality and semicontinuous variables.

These instances ship the scenario matrix itself, so the objective is unambiguous and
gaps are comparable.
