# Energy UC + ML: Forecast-Driven Unit Commitment

Day-ahead unit commitment for a diversified generation fleet (coal, CCGT, gas
peaker, wind, solar, battery storage), where the renewable input comes from a
machine-learned quantile forecaster rather than a naive assumption. The
project's core finding is less "look, I made a forecast and an optimizer"
and more a specific, defensible insight about how they interact:

> **A forecast that minimizes average error is not the same as a forecast
> that produces good commitment decisions.** For unit commitment, the cost
> of *overestimating* available renewables (leaving demand unserved) is far
> higher than the cost of underestimating it (running a bit more thermal
> capacity than strictly needed). A conservative (P10) forecast produced a
> realized cost of $486,003 — essentially matching the perfect-foresight
> lower bound of $485,996 — while the "more accurate" median (P50) forecast
> led to a realized cost of $2,744,174 due to 133.5 MW of unserved demand
> in one hour. Naive persistence (yesterday's actuals) landed at $545,803 —
> worse than the conservative ML forecast despite having lower MAE.

This is the standard cost-asymmetry argument for why decision-quality
metrics (not just forecast accuracy) matter in operations research — and
it's a good jumping-off point for the Phase 2 extensions below.

## Project structure

```
energy-uc-ml/
├── src/
│   ├── data_gen.py         # synthetic demand/wind/solar + thermal fleet spec
│   ├── forecasting.py      # quantile (P10/P50/P90) GBM forecaster for wind & solar
│   ├── unit_commitment.py  # MILP unit commitment model (scipy.optimize.milp / HiGHS)
│   ├── pipeline.py         # end-to-end run: forecast -> UC -> scenario comparison
│   └── app.py               # Streamlit dashboard (interactive quantile/battery sliders)
├── outputs/                 # charts + CSV summaries land here
├── data/                    # generated CSVs land here
├── pyproject.toml           # uv / pip project metadata + dependencies
├── environment.yml          # conda alternative
└── requirements.txt         # plain pip alternative
```

## Quickstart

**Option A — uv (recommended, faster, reproducible lockfile):**

```bash
uv sync                              # creates .venv, installs from pyproject.toml
uv run python src/pipeline.py        # full comparison, saves charts to outputs/
uv run streamlit run src/app.py      # interactive dashboard
```

`uv sync` will also produce a `uv.lock` file on first run — commit that to
the repo so anyone cloning it gets the exact versions this was built
against. For the Phase 2 extras (Pyomo/HiGHS): `uv sync --extra phase2`.

**Option B — conda:**

```bash
conda env create -f environment.yml
conda activate energy-uc-ml
python src/pipeline.py
streamlit run src/app.py
```

**Option C — plain pip / venv:**

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/pipeline.py
streamlit run src/app.py
```

No external MILP solver install or license needed for any of the above —
`unit_commitment.py` uses `scipy.optimize.milp`, which ships with SciPy and
uses the HiGHS solver under the hood.

## Methodology

**Data.** Synthetic hourly demand/wind/solar with realistic diurnal and
seasonal shape (see `data_gen.py` docstring for pointers to swap in real
data: EIA/ENTSO-E for demand, NREL Wind/Solar Integration Toolkit for
renewables, EIA-860/923 for real generator fleet specs). Wind is generated
as a mean-reverting stochastic process with weak seasonality — deliberately
hard to forecast a day ahead, which is realistic and is what drives the
core finding above.

**Forecasting** (`forecasting.py`). A `GradientBoostingRegressor` with
quantile loss, trained separately for P10/P50/P90, using calendar features
(hour, day-of-week, seasonal sin/cos) plus autoregressive lags (24h, 48h,
168h) and a rolling mean — the lag features are what actually let the
model beat a flat seasonal average on wind. Solar forecasts far better
(MAE ~5 MW / 250 MW capacity) than wind (MAE ~77 MW / 300 MW capacity)
because solar is almost entirely diurnal/seasonal while wind has a large
stochastic component — this gap is itself a realistic and useful result to
discuss.

**Optimization** (`unit_commitment.py`). A MILP with binary commitment and
startup variables per generator per hour, continuous power output linked
to commitment via `pmin·u ≤ p ≤ pmax·u`, ramp limits, a simplified min-up-time
constraint (min-down-time is a noted simplification — see Limitations),
battery state-of-charge dynamics with round-trip efficiency, and a
curtailment variable so renewable oversupply doesn't break feasibility.
Solved via `scipy.optimize.milp`.

**The planned-vs-realized comparison** (`pipeline.py`). This is the
important design choice: comparing the "cost" of each forecast's UC
solution *as computed under that forecast* is misleading, because an
optimistic forecast lets the solver under-commit thermal capacity and
looks artificially cheap. Instead, each scenario's commitment schedule is
*fixed* and re-settled against the actual realized renewable output (via
`fixed_commitment` in `UnitCommitmentModel.build_and_solve`), which is what
exposes the P50 forecast's real cost.

## Limitations (honest, and worth stating if asked in an interview)

- Min-down-time constraint is simplified/omitted relative to a textbook UC
  formulation — a straightforward addition if you want full rigor.
- Battery charge/discharge doesn't have a binary mutual-exclusivity
  constraint (relies on cost structure discouraging simultaneous
  charge+discharge); fine for a demo, worth tightening for production use.
- Single fixed thermal fleet and single representative day — no network
  constraints (transmission), no reserve margin requirements.
- Realized-cost re-dispatch approximates a real-time settlement; a fully
  rigorous version would model reserve/ramp feasibility more carefully.

## Phase 2 roadmap (scaling this up)

1. **Stochastic / chance-constrained UC.** Use the full P10/P50/P90 (or a
   richer scenario set) instead of a single quantile, and solve a
   scenario-based stochastic program that hedges across the distribution
   rather than committing to one point forecast. Natural point to migrate
   from `scipy.optimize.milp` to **Pyomo + HiGHS**, since indexed variables
   over (scenario, generator, hour) get unwieldy as hand-built sparse
   matrices.
2. **Rolling-horizon (MPC-style) simulation.** Re-solve daily over a
   multi-week simulation as new forecasts arrive, rather than a single
   static day — makes for a much better demo (a dashboard that evolves
   day by day) and is closer to how this is actually operated.
3. **EV charging as flexible demand.** Add a controllable EV charging load
   the optimizer can shift within a time window, extending the battery
   storage story.
4. **Decision-focused learning.** The most differentiated extension: train
   the forecasting model's loss function on downstream dispatch cost
   rather than pure forecast accuracy (e.g. via `cvxpylayers` or a
   differentiable optimization layer), so the ML model directly learns to
   produce forecasts that lead to good commitment decisions — closing the
   loop between the two halves of this project rather than pipelining them.

## Results snapshot

See `outputs/dispatch_comparison.png` (forecast vs. actual, and the
resulting dispatch stack) and `outputs/planned_vs_realized_cost.png` (the
core finding, visualized) after running `pipeline.py`.
