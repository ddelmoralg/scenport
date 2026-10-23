# scenport

Scenario-based portfolio construction for cryptoassets. Code and data for the
paper *Cardinality-constrained mean-CVaR portfolio selection over simulated
scenarios: exact bounds, heuristic gaps, and sensitivity to the scenario
generator*.

The repository holds a mean-CVaR allocator that optimises directly over simulated
scenarios under a cardinality limit and a minimum position size, checked against
the exact mixed-integer optimum. On top of it sits a benchmark that asks which
property of a forecast shows up in the portfolio built from it. Forecasting papers
report accuracy, while a risk-aware objective consumes calibration, and the two
need not agree.

The second question needs eleven forecasters and fifteen allocators to be
interchangeable parts, which is what most of this repository is.

## Layout

```
scenport/data       prices, market caps, point-in-time universe, benchmarks
scenport/forecast   scenario interface, coupling, the eleven models, on-disk store
scenport/alloc      objective, decoder, heuristics, convex, MILP, GA/DE/PSO/CMA-ES
scenport/backtest   walk-forward engine and performance metrics
scenport/stats      Ledoit-Wolf, Diebold-Mariano, deflated Sharpe, CSCV, SPA, MCS
scripts/            entry points
slurm/              cluster jobs
notes/              design notes
```

Two interfaces carry the design.

A **forecaster** returns simulated paths, not a point: `(n_paths, horizon)`
cumulative simple returns per asset. Assets are simulated independently and the
cross-sectional dependence is imposed afterwards by a separate step, so the
forecaster and the dependence model vary one at a time.

An **allocator** sees only the scenario matrix. No expected-return vector, no
covariance. The objective is evaluated on the scenarios, so the tail of the
predictive distribution enters the decision and everything stays on one time scale.

## What is implemented

Forecasters: historical simulation, driftless random walk, ARIMA, ETS, AR-GARCH
with skew-*t* innovations, Prophet (tuned by CRPS and, separately, by RMSE),
LightGBM, LSTM, DeepAR, transformer encoder.

Coupling: Gaussian copula by rank reordering, empirical copula, or none.

Allocators: 1/N, cap-weighted, random, inverse volatility, single asset, minimum
variance, maximum Sharpe, risk parity, HRP, mean-CVaR LP, GA, DE, PSO, CMA-ES, and
the exact mixed-integer optimum.

Objective: mean-CVaR with a per-asset cap, a cardinality limit, a minimum position
size and turnover cost. Mean-variance and Sharpe are special cases.

## Install

```bash
conda env create -f environment.yml
conda activate scenport
pip install -e ".[dev]"
```

torch is deliberately not in `environment.yml`. Install the build that matches the
cluster driver, which you read off a compute node rather than the login node:

```bash
nvidia-smi                 # on a compute node, to read the driver version
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## The loop

Code goes out over git, results come back over git. Nothing is copied by hand.

```
laptop                          github                    cluster
  edit, commit, push  ------------->  ------ git pull, submit_all.sh
                                                              |
                                                        jobs run
                                                              |
  git pull  <-------------  <------------  publish.sh on the login node
```

**Once, on the laptop:**

```bash
python scripts/fetch_data.py --config configs/base.yaml   # needs network
git add data/panel && git commit -m "panel" && git push
```

The panel is committed on purpose. It is a few megabytes, it pins the exact data
every result was computed from, and it means the compute nodes never need network
access.

**Every run, on the login node:**

```bash
git pull
bash slurm/submit_all.sh
```

`submit_all.sh` refuses to run on a dirty tree, pulls, checks the queues it is
about to use still exist, submits the seven jobs with the right dependencies, and
prints the ids.

The compute nodes on our cluster cannot resolve github.com, so publishing is a
separate step. Once the aggregation job finishes, on the login node:

```bash
bash scripts/publish.sh
```

It commits `artifacts/` and pushes, rebasing and retrying if the laptop pushed in
the meantime.

**Back on the laptop:** `git pull`, and the tables and figures are in `artifacts/`.

What travels back: `artifacts/` (tables, figures, consolidated scores and daily
returns) and the small summary parquets. What does not: the scenario sets, which
run to a couple of gigabytes, and the per-run backtest tree.

### Letting the cluster push

The cluster needs a deploy key with write access. Set it up from here, once:

```bash
bash scripts/setup_cluster_from_laptop.sh user@your-login-node
```

On Windows, where bash is usually not on the path, there is a PowerShell version
that does the same thing with the OpenSSH client Windows ships:

```powershell
.\scripts\setup_cluster_from_laptop.ps1 user@your-login-node
```

It copies `scripts/bootstrap_cluster.sh` over, runs it so the private key is born
on the cluster and stays there, registers the public half through the GitHub API,
and clones. Nothing to paste, and the only credential that reaches the cluster is a
key scoped to this one repository.

If you would rather do it by hand, run `bash scripts/bootstrap_cluster.sh` on the
login node. It prints the public key and stops; add it under Settings > Deploy keys
with write access ticked, then run it again.

Both scripts are safe to rerun.

## Jobs

The queue names below are the ones on our cluster: `short` caps at 4 hours,
`medium` at a day, `long` at a week. Rename them in the job headers if yours
differ.

| job | queue | shape | roughly |
|---|---|---|---|
| `10_forecast_cpu` | `medium` | 8 forecasters x 16 shards, 8 cores, 20 at a time | 4-10 h |
| `20_forecast_gpu` | `medium` | 3 forecasters x 2 shards, 1 GPU, 3 at a time | 2-5 h |
| `30_backtest` | `medium` | one task per forecaster, cheap allocators | under 1 h |
| `32_backtest_ea` | `medium` | 11 forecasters x 4 metaheuristics, 5 seeds | 2-3 h |
| `31_ablations` | `medium` | 18 sensitivity runs | 2-6 h |
| `40_exact` | `long` | gaps + scaling sweep, 24 cores | a few hours |
| `50_aggregate` | `medium` | one task | under 1 h |

Everything fits in a day. `40_exact` sits on the long queue for headroom: each
exact solve is allowed fifteen minutes, although on the thinned instances in the
paper none needed more than a few seconds.

The two forecasting jobs run at the same time and share one node, so their
concurrency limits are set together: 20 x 8 cores plus 3 x 8 leaves headroom on a
224-core node. Retune the `%N` on each array if your node is a different size.

Edit `slurm/env.sh` first; it is the only place the environment is named.

Simulation and allocation are separate stages with the scenario sets stored in
between. That is why the eighteen-variant ablation sweep costs hours instead of
another full forecasting pass.

## Reproducibility

Every stochastic component takes an explicit `numpy.random.Generator`, seeded
through `scenport.config.stable_seed` rather than `hash()`, which is salted per
interpreter and would differ between array tasks.

`tests/test_no_lookahead.py` scrambles all prices after a cut date and asserts that
no weight chosen before it moves; it does the same for universe selection, the
coupling step and the evaluation windows. `tests/test_allocators.py` checks the
metaheuristics against branch and bound on instances the solver can close.

```bash
pytest -q                 # a few minutes
pytest -q -m "not slow"   # skips the mixed-integer comparisons
```

## Configuration

Everything lives in `configs/base.yaml`. Override single keys instead of editing it:

```bash
python scripts/run_backtest.py --forecaster garch --set alloc.cardinality=5 \
                               --set backtest.cost_bps=50 --tag card05
```

`scenport check configs/base.yaml` prints what a config resolves to.

## Data

Prices and market capitalisations: [Coin Metrics
Community](https://docs.coinmetrics.io/api/v4), no key required, published under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). That licence is
why the panel can be committed here: it permits redistribution with attribution
for non-commercial use. See [`data/panel/SOURCE.md`](data/panel/SOURCE.md). Chosen over the
alternatives because it publishes a long history of market capitalisation, without
which the universe cannot be rebuilt as it stood at each rebalancing date. Risk-free
rate: `DTB3` from FRED.

Stablecoins and wrapped tokens are dropped by name, with a rolling
coefficient-of-variation test as a backstop for pegs the list misses.

## Instance library

`scripts/export_instances.py` dumps the allocation problems as an npz plus a JSON
header: scenario matrix, previous weights, constraint parameters, and the best
objective we know of with a flag for whether a solver proved it. The released set
in `artifacts/instances/` has 105 instances, 88 of them with a proven optimum; the
other 17 vary a constraint or a universe size that the exact run did not cover and
carry no bound.

Reading an instance needs numpy and nothing from this package, so anyone
benchmarking a different solver does not have to run a Prophet sweep first.

## Notes

- [`notes/universe.md`](notes/universe.md), why the universe is rebuilt at every
  date.
- [`notes/objective.md`](notes/objective.md), the mean-CVaR formulation, the
  decoder, and why the discrete constraints are what justify a metaheuristic.

## Troubleshooting

*`OSError: access violation` the first time LightGBM builds a `Dataset`, on
Windows.* LightGBM's OpenMP runtime clashing with the one pandas already loaded.
Reproduces in four lines of plain LightGBM. Install both from conda-forge; the test
suite detects it and skips.

*`no mixed-integer solver available`.* `pip install highspy`.

*Prophet compiles cmdstan on first import.* Install it from conda-forge.

## Licence

The code is MIT. The derived data panel in `data/panel/` keeps the CC BY-NC 4.0
licence of its source, so anything built on the panel is for non-commercial use
and has to credit Coin Metrics.
