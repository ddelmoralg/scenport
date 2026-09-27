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

with CVaR written as a positive loss: the mean of the worst (1 - alpha) fraction of
the S scenario returns.

`best_known` in the JSON is the best objective value we have found and `proven` says
whether a mixed-integer solver closed it. Improvements welcome; open an issue with
the instance name, the objective value and how you got it.

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
