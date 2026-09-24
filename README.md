# Energy UC + ML: Forecast-Driven Unit Commitment

Day-ahead unit commitment for a diversified generation fleet (coal, CCGT, gas
peaker, wind, solar, battery storage), where the renewable input comes from a
machine-learned quantile forecaster rather than a rated output.

> **A forecast that minimizes average error is not the same as a forecast
> that produces good commitment decisions.** For unit commitment, the cost
> of *overestimating* available renewables (leaving demand unserved) is far
> higher than the cost of underestimating it (running a bit more thermal
> capacity than strictly needed). The median (P50) wind/solar forecast has
> a *one-directional* bias — it overestimates renewable output by 77.3 MW
> on average, never underestimates — which leaves the resulting commitment
> schedule short of thermal capacity and produces a realized cost of
> $3,113,130 once its unserved-demand hours are priced in. Switching to
> the conservative P10 quantile (which underestimates by 44.6 MW on
> average) produces a realized cost of $446,523 — essentially matching the
> perfect-foresight lower bound of $446,174 — despite P10 *not* being the
> most accurate forecast on offer: naive persistence (yesterday's actuals)
> has the lowest MAE of the three (40.8 MW vs. P10's 47.1 MW) and lands at
> a similarly good $446,309. The ML model doesn't beat naive persistence on
> raw accuracy here — the win comes entirely from picking the quantile
> whose *bias direction* matches what the optimization needs, not from the
> forecast being "better."

## Project structure

```
energy-uc-ml/
├── src/
│   ├── data_gen.py         # synthetic demand/wind/solar + thermal fleet spec
│   ├── forecasting.py      # quantile (P10/P50/P90) GBM forecaster for wind & solar
│   ├── unit_commitment.py  # MILP unit commitment model (scipy.optimize.milp / HiGHS)
│   ├── analysis.py         # residual-load, ramp-headroom & Gantt-chart helpers (Streamlit-free, testable standalone)
│   ├── pipeline.py         # end-to-end run: forecast -> UC -> scenario comparison
│   └── app.py               # Streamlit dashboard (primary way to explore this project)
├── outputs/                 # charts + CSV summaries land here
├── data/                    # generated CSVs land here
├── pyproject.toml           # uv / pip project metadata + dependencies
├── environment.yml          # conda alternative
└── requirements.txt         # plain pip alternative
```

## Quickstart

The Streamlit app (`src/app.py`) is the primary way to explore this
project — interactive fuel prices, forecast quantile, battery sizing, and
the residual-load / commitment / ramp visualizations. `pipeline.py` is the
non-interactive scenario comparison that produces the headline numbers
above.

**Option A — uv (recommended, faster, reproducible lockfile):**

```bash
uv sync                              # creates .venv, installs from pyproject.toml
uv run streamlit run src/app.py      # interactive dashboard (primary)
uv run python src/pipeline.py        # scenario comparison, saves charts to outputs/
```

`uv sync` will also produce a `uv.lock` file on first run — commit that to
the repo so anyone cloning it gets the exact versions this was built
against. For the Phase 2 extras (Pyomo/HiGHS): `uv sync --extra phase2`.

**Option B — conda:**

```bash
conda env create -f environment.yml
conda activate energy-uc-ml
streamlit run src/app.py
python src/pipeline.py
```

**Option C — plain pip / venv:**

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/app.py
python src/pipeline.py
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
168h) and a rolling mean. Solar forecasts far better (MAE ~5 MW / 250 MW
capacity) than wind (MAE ~77 MW at P50 / 300 MW capacity) because solar is
almost entirely diurnal/seasonal while wind has a large stochastic
component with weak 24h autocorrelation in this synthetic data. Worth
saying plainly: the wind forecaster does **not** beat naive persistence on
raw MAE (77.3 MW at P50 vs. 40.8 MW for persistence) — what makes it
useful isn't superior accuracy, it's that its quantiles give you a *choice
of bias direction*. In this run P50 isn't neutral at all — its bias
(+77.3 MW) exactly equals its MAE, meaning it overestimates renewable
output in every single hour of the test day — while P10 pulls that back to
a -44.6 MW underestimate. For unit commitment, which direction you're
biased toward matters more than the average error. See the headline
finding above.

**Thermal fleet cost model** (`data_gen.py`). Cost is decomposed as
`heat_rate (MMBtu/MWh) x fuel_cost ($/MMBtu) + var_om ($/MWh)` rather than
an opaque per-unit `marginal_cost`, so fuel price is a real, tunable input
(exposed as sliders in the app) instead of a fixed number. Generator
capacities are deliberately sized so the gas peaker's 100 MW sits outside
what the baseload + mid-merit units (Coal 350 + CCGT 200 + CCGT 150 = 700
MW) plus the battery (60 MW) can cover — the peaker engages for a few
hours on roughly 40% of days, entirely from capacity being tight at the
evening peak, with **no need to distort fuel prices** to make that happen
(merit order stays coal-cheapest under normal coal/gas pricing throughout).

**Optimization** (`unit_commitment.py`). A MILP with binary commitment,
startup, and shutdown-indicator variables per generator per hour,
continuous power output linked to commitment via `pmin·u ≤ p ≤ pmax·u`,
ramp limits, both min-up-time *and* min-down-time constraints (the
shutdown-indicator variable that enforces min-down-time is only loosely
constrained on already-off hours since nothing in the objective penalizes
it — harmless for the MILP's correctness, but not meaningful for display;
see Limitations), battery state-of-charge dynamics with round-trip
efficiency, and a curtailment variable so renewable oversupply doesn't
break feasibility. Solved via `scipy.optimize.milp`.

**Making the residual load visible** (`analysis.py`, `app.py`). The
app is built around one framing: demand minus renewable output leaves a
**residual load** that thermal generation and the battery are the only
levers to cover. `analysis.py` holds that logic (residual load, thermal +
battery coverage, ramp headroom, commitment Gantt chart) as plain
functions with no Streamlit dependency, so it's testable standalone;
`app.py` renders it interactively, including a per-generator commitment
Gantt chart and a fleet table that updates live as you move the fuel-price
sliders.

**The planned-vs-realized comparison** (`pipeline.py`). This is the
important design choice: comparing the "cost" of each forecast's UC
solution *as computed under that forecast* is misleading, because an
optimistic forecast lets the solver under-commit thermal capacity and
looks artificially cheap. Instead, each scenario's commitment schedule is
*fixed* and re-settled against the actual realized renewable output (via
`fixed_commitment` in `UnitCommitmentModel.build_and_solve`), which is what
exposes the P50 forecast's real cost.

## Limitations

- The shutdown-indicator variable (`v` in `unit_commitment.py`) that
  enforces min-down-time is correctly constrained for MILP feasibility,
  but is not itself a meaningful "did this unit shut down" signal on hours
  where the unit was already off — `analysis.py`'s Gantt chart derives
  actual on/off transitions from the commitment variable directly rather
  than trusting it, and anything else built on `dispatch` should do the
  same rather than reading the raw `shutdown` column.
- Battery charge/discharge doesn't have a binary mutual-exclusivity
  constraint (relies on cost structure discouraging simultaneous
  charge+discharge); fine for a demo, worth tightening for production use.
- Fleet capacities were hand-tuned to make the peaker's role visible in a
  single representative day, not derived from a real reserve-margin or
  capacity-planning study — swap in real EIA-860/923 data before treating
  the specific MW/cost numbers as meaningful.
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
