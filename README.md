# scenport

Scenario-based portfolio construction for cryptoassets. Code for the paper (title
pending).

Two things. A mean-CVaR allocator that optimises directly over simulated scenarios
under a cardinality limit and a minimum position size, validated against the exact
mixed-integer optimum wherever branch and bound can close it. And, on top of that,
a benchmark that asks which property of a forecast actually shows up in the
portfolio built from it — forecasting papers report accuracy, a risk-aware
objective consumes calibration, and they are not the same thing.

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

torch is deliberately not in `environment.yml` — install the build that matches the
cluster driver:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## The loop

Code goes out over git, results come back over git. Nothing is copied by hand.

```
laptop                          github                    cluster
  edit, commit, push  ------------->  ------ git pull, submit_all.sh
                                                              |
                                                        jobs run
                                                              |
  git pull  <-------------  <------------  60_publish.sh pushes artifacts/
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

`submit_all.sh` refuses to run on a dirty tree, pulls, submits the six jobs with
the right dependencies, and prints the ids. The last job commits `artifacts/` and
pushes; if the laptop pushed in the meantime it rebases and retries.

**Back on the laptop:** `git pull`, and the tables and figures are in `artifacts/`.

What travels back: `artifacts/` (tables, figures, consolidated scores and daily
returns) and the small summary parquets. What does not: the scenario sets, which
run to a couple of gigabytes, and the per-run backtest tree.

### Letting the cluster push

A deploy key with write access is the least painful option.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/scenport_deploy -N ""
cat ~/.ssh/scenport_deploy.pub     # paste into Settings > Deploy keys, tick "Allow write access"

cat >> ~/.ssh/config <<'EOF'
Host github-scenport
    HostName github.com
    User git
    IdentityFile ~/.ssh/scenport_deploy
    IdentitiesOnly yes
EOF

git remote set-url origin git@github-scenport:<user>/scenport.git
```

If the compute nodes are firewalled off from github.com, drop `60_publish.sh` from
`submit_all.sh` and run `bash scripts/publish.sh` on the login node once the
aggregation finishes.

## Jobs

| job | partition | shape | roughly |
|---|---|---|---|
| `10_forecast_cpu` | `cpu` | 8 forecasters x 16 shards, 8 cores | 4-10 h |
| `20_forecast_gpu` | `gpu-small` | 3 forecasters x 2 shards, 1 GPU | 2-5 h |
| `30_backtest` | `cpu` | one task per forecaster | 1-3 h |
| `31_ablations` | `cpu` | 18 sensitivity runs | 2-6 h |
| `40_exact` | `cpu` | gaps + scaling sweep | up to 2 days |
| `50_aggregate` | `cpu` | one task | under 1 h |

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
Community](https://docs.coinmetrics.io/api/v4), no key required. Chosen over the
alternatives because it publishes a long history of market capitalisation, without
which the universe cannot be rebuilt as it stood at each rebalancing date. Risk-free
rate: `DTB3` from FRED.

Stablecoins and wrapped tokens are dropped by name, with a rolling
coefficient-of-variation test as a backstop for pegs the list misses.

## Notes

- [`notes/universe.md`](notes/universe.md) — why the universe is rebuilt at every
  date.
- [`notes/objective.md`](notes/objective.md) — the mean-CVaR formulation, the
  decoder, and why the discrete constraints are what justify a metaheuristic.

## Troubleshooting

*`OSError: access violation` the first time LightGBM builds a `Dataset`, on
Windows.* LightGBM's OpenMP runtime clashing with the one pandas already loaded.
Reproduces in four lines of plain LightGBM. Install both from conda-forge; the test
suite detects it and skips.

*`no mixed-integer solver available`.* `pip install highspy`.

*Prophet compiles cmdstan on first import.* Install it from conda-forge.

## Licence

MIT.
