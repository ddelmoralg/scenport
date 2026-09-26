# The allocation problem

## What is optimised

Let `R` be the `(S, n)` matrix of scenario returns over the holding period, `w` the
weight vector and `w_prev` the weights carried in from the previous period. Write
`r = R w` for the `S` scenario returns of the portfolio. The objective is

```
max  (1 - g) * mean(r)  -  g * CVaR_a(r)  -  c * ||w - w_prev||_1
```

subject to

```
sum(w) = 1
0 <= w_i <= w_max
w_i = 0  or  w_i >= w_min
||w||_0 <= K
```

`CVaR_a(r)` is the mean of the worst `(1 - a)` fraction of scenario returns, written as
a positive loss. `g` is the only risk-aversion knob: `g = 0` maximises expected return,
`g = 1` minimises tail loss.

Three things follow from evaluating this on the scenarios rather than on summary
statistics.

**The tail enters the decision.** A forecaster that gets the centre of the
distribution right and the tails wrong is penalised, because CVaR is computed from the
worst scenarios and nothing else. Mean-variance cannot see this; it only ever looks at
the second moment.

**Everything is on one time scale.** The scenarios are horizon returns, so the mean,
the CVaR and the risk-free rate are all horizon quantities. Sounds trivial, is not:
dividing a four-month expected return by a *daily* standard deviation gives a ratio
whose level means nothing, and once a risk-free rate is subtracted from the numerator
the ranking is wrong too, because the denominator varies with `w`.

**Any scenario set works.** The objective never asks where the paths came from, which
is what lets eleven very different forecasters be compared on equal terms.

## Why a metaheuristic

Mean-CVaR on its own is a **linear program**. Rockafellar and Uryasev's formulation
introduces one auxiliary variable per scenario and a scalar threshold, and the whole
thing solves to global optimality in milliseconds. So does maximum Sharpe under box
constraints, through the standard `y = kappa w` reformulation. If the problem stopped
there, using a genetic algorithm would be indefensible, and saying so is one of the
things the paper has to do.

What makes it hard is the discrete part. `||w||_0 <= K` and the semicontinuous
condition `w_i = 0 or w_i >= w_min` turn the LP into a mixed-integer program with `n`
binaries. Branch and bound still closes it at `n = 15`, and `scenport/alloc/exact.py`
does exactly that, which is how the optimality gaps in the results table are measured.
It stops closing it somewhere above that, and the scaling experiment in
`scripts/run_exact.py` finds where.

So the argument is empirical, not assumed:

- on instances the solver can close, the metaheuristics are within a small percentage
  of proven optimality (`tests/test_allocators.py` asserts under 5 %);
- on instances it cannot, the metaheuristics still return an answer in seconds.

Neither half is interesting without the other.

## The decoder

Search happens in unconstrained `R^n`. A real vector becomes a portfolio like this:

1. **sign selects.** Genes above zero are candidate positions. The magnitude proposes
   a weight. A gene crossing zero adds or removes a position, so the population
   explores different supports without a separate binary chromosome.
2. **trim and pad.** If more than `K` genes are positive, keep the `K` largest. If
   fewer than `ceil(1 / w_max)` are positive, too few to add up to one under the cap,
   take that many largest instead.
3. **project.** The proposal is projected onto `{w : sum(w) = 1, w_min <= w_i <= w_max}`
   over the chosen support. The Euclidean projection has the closed form
   `clip(v + lam, w_min, w_max)` for a scalar `lam`, and the sum is monotone in `lam`,
   so a bisection finds it exactly.

Every candidate is therefore feasible by construction. The alternative, a penalty
term on the cap, sounds equivalent and is not. A hard rejection turns the constraint
boundary into a cliff and throws away most of the population's gradient information; a
soft penalty leaves the search wandering in infeasible territory and needs its weight
tuned. Repairing avoids both, and it makes the reported evaluation budget mean
something, which matters because the budget is one of the axes the algorithms are
compared on.

## Choosing gamma, K and w_max

They are not tuned. `configs/base.yaml` fixes `g = 0.5`, `K = 8`, `w_max = 0.20` and
`w_min = 0.02`, and `slurm/31_ablations.sh` sweeps each of them one at a time. Tuning
them on the backtest and then reporting the backtest would be the single most effective
way to produce a result that does not survive contact with the future, which is what
the probability-of-backtest-overfitting figure in the results is there to quantify.

`w_max = 0.20` matches the concentration limit UCITS imposes on retail funds, so it is
at least a cap somebody has to live with rather than one picked to look good.
