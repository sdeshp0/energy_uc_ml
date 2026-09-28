# Energy UC + ML: Forecast-Driven Unit Commitment

Day-ahead unit commitment for a diversified generation fleet (coal, CCGT,
gas peaker, wind, solar, battery storage), with renewable and demand input
from machine-learned quantile forecasters. The project's central finding
concerns how forecast quality and decision quality relate.

## Headline finding

A forecast that minimizes average error is not the same as a forecast that
produces good commitment decisions. In unit commitment, the cost of
*overestimating* available renewables (unserved demand) is far higher than
the cost of underestimating it (extra thermal capacity committed). The
median (P50) wind/solar forecast has a one-directional bias: it
overestimates renewable output by 79.5 MW on average and never
underestimates. That bias leaves the resulting commitment schedule short
of thermal capacity, producing a realized cost of $6,682,672 once
unserved-demand hours are priced in. The conservative P10 quantile
(average underestimate of 59.2 MW) produces a realized cost of $359,999 —
close to the perfect-foresight lower bound of $358,274 — despite not being
the most accurate forecast available: naive persistence (yesterday's
actuals) has the lowest MAE of the three (40.8 MW vs. P10's 60.1 MW) and
lands at a comparable $503,223.

The ML forecaster does not beat naive persistence on raw accuracy here.
Its value comes from letting the commitment decision pick a quantile whose
bias direction matches what the optimization needs, not from the forecast
being more accurate.

## Project structure

```
energy-uc-ml/
├── src/
│   ├── data_gen.py           synthetic demand/wind/solar, thermal fleet, price simulation
│   ├── forecasting.py        quantile (P10/P50/P90) forecasters; single-fit and range-fit variants
│   ├── unit_commitment.py    single-scenario MILP and two-stage stochastic MILP
│   ├── scenario.py           shared single-day scenario builder (used by app.py and pages/)
│   ├── cached_forecasts.py   shared Streamlit cache for the demand/wind/solar forecast fit
│   ├── scenarios.py          empirical joint (demand x renewable) scenario probabilities
│   ├── rolling_horizon.py    multi-day walk-forward simulation engine
│   ├── analysis.py           residual load, sweeps, economics, all plotting
│   ├── pipeline.py           CLI: forecast-quality comparison (P10 vs P50 vs persistence)
│   ├── app.py                main Streamlit dashboard
│   └── pages/
│       ├── 1_Market_And_Battery_Arbitrage.py   simulated prices, settlement, battery arbitrage
│       ├── 2_Stochastic_Unit_Commitment.py     the two-stage stochastic hedge
│       ├── 3_Rolling_Horizon_Simulation.py     multi-day comparison of both approaches
│       └── 4_Sensitivity_Analysis.py           parameter sweeps, optional stochastic overlay
├── docs/
│   └── CODE_WALKTHROUGH.md   full technical walkthrough of the codebase
├── pyproject.toml            uv / pip project metadata and dependencies
├── environment.yml           conda alternative
└── requirements.txt          plain pip alternative
```

## Quickstart

The Streamlit app is the primary entry point. `pipeline.py` is a
non-interactive CLI that reproduces the headline numbers above.

**uv (recommended):**

```bash
uv sync
uv run streamlit run src/app.py
uv run python src/pipeline.py
```

`uv sync` produces a `uv.lock` on first run; commit it. For the Phase 2
extras (Pyomo/HiGHS): `uv sync --extra phase2`.

**conda:**

```bash
conda env create -f environment.yml
conda activate energy-uc-ml
streamlit run src/app.py
```

**pip / venv:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/app.py
```

No external MILP solver install is required. `unit_commitment.py` uses
`scipy.optimize.milp` (HiGHS backend), included with SciPy.

**Imports and IDE setup.** Modules under `src/` import each other by bare
name (`from data_gen import ...`); `src/` is the import root at runtime,
since Streamlit puts the entrypoint's directory on `sys.path`. IDEs need
to be told the same thing, or they flag these imports as unresolved:

- **PyCharm:** right-click `src/` → Mark Directory as → Sources Root
  (stored per machine in `.idea/`, so it is not part of the repo).
- **VS Code (Pylance/pyright):** `pyproject.toml` sets
  `[tool.pyright] extraPaths = ["src"]`.

Do not work around IDE warnings by switching to `src.`-prefixed imports:
the same file would load twice as two separate modules, splitting the
shared Streamlit cache and duplicating module-level state.

## What's implemented

1. **Single-scenario MILP** (`unit_commitment.py`). Binary commitment,
   startup, and shutdown variables per generator per hour; ramp limits;
   min-up and min-down time; battery state-of-charge dynamics; renewable
   curtailment. Solved via `scipy.optimize.milp`.
2. **Quantile forecasting** (`forecasting.py`). Gradient-boosted quantile
   regression for wind, solar, and demand, using calendar and
   autoregressive-lag features.
3. **Planned-vs-realized cost comparison** (`pipeline.py`). Each
   forecast's commitment schedule is fixed and re-settled against actual
   demand/renewable output, since comparing planned cost under a forecast
   is misleading — an optimistic forecast under-commits capacity and looks
   cheap on paper. This is what produces the headline finding above.
4. **Fuel-cost decomposition and price simulation** (`data_gen.py`).
   Generator cost as heat rate × fuel price + variable O&M; an exogenous
   merit-order-derived electricity price for settlement reporting.
5. **Battery arbitrage** (`unit_commitment.py`, optional `price` argument).
   A price term layered on top of production-cost minimization, evaluated
   independently of the forecast-uncertainty story.
6. **Two-stage stochastic unit commitment** (`unit_commitment.py`,
   `StochasticUnitCommitmentModel`). One shared commitment schedule across
   nine (demand × renewable) scenarios, with scenario-specific recourse
   dispatch. Scenario probabilities are estimated empirically from paired
   historical forecast errors (`scenarios.py`), not assumed independent.
7. **Rolling-horizon simulation** (`rolling_horizon.py`). Multi-day
   walk-forward comparison of committing on the P50 forecast alone versus
   the stochastic hedge, with battery state of charge and generator status
   carried forward across day boundaries.
8. **Sensitivity analysis** (`analysis.py`, page 4). Parameter sweeps over
   fuel prices and battery sizing, with an optional stochastic-hedge
   overlay.

Full detail on each component, including the MILP formulations, is in
`docs/CODE_WALKTHROUGH.md`.

## Selected results

**Stochastic hedge vs. committing on P50 alone.** Committing based on the
P50 scenario, then facing a high-demand/low-renewable scenario: 186.8 MW
unserved demand, $8,932,848 cost. The same bad scenario, same fleet, with
the stochastic hedge's commitment instead: 0 MW unserved, $360,967. The
hedge costs more in expectation across all nine scenarios (an insurance
premium — more thermal capacity committed than P50 alone would choose) in
exchange for eliminating the worst-case failure mode.

**Empirical vs. independence-assumed scenario probabilities.**
Demand and renewable forecast errors have measured correlation −0.13 in
this dataset (via a deliberate correlated "cold snap" shock in
`data_gen.py`; without it the two are independent by construction and
there is nothing for this comparison to find). The high-demand/
low-renewable joint probability is 0.124 empirically vs. 0.111 assuming
independence — the independence assumption understates exactly the
scenario a hedge is meant to protect against.

**Battery arbitrage.** Adding a price incentive to the battery's objective
changed production cost by $0 while increasing battery P&L from $2,207 to
$2,616 (+18.5%) on one representative day — a contained shift, not a
change in overall system behavior.

**Sensitivity ranges.** Coal price swept $1–8/MMBtu changes total cost by
147%. Battery power swept 0–150 MW shows diminishing returns past
approximately 60–65 MW for a 200 MWh battery — a property of the
power-to-energy ratio, not a tuning artifact.

**Rolling horizon, 14-day default window.** P50-only: 8 of 14 days with
unserved demand, 458.7 MWh total unserved, $9,651,519 total cost.
Stochastic hedge: 1 of 14 days with unserved demand, 0.8 MWh total
unserved, $4,986,721 total cost — lower on both reliability and aggregate
cost over the window, despite costing a premium on any single day
analyzed in isolation.

## Performance

The demand/wind/solar quantile forecast fit is the dominant cost on every
page that needs one (roughly 20s at the original setting). Two changes
address this:

- **`n_estimators` tuning** (`forecasting.py`). The original 300
  estimators per GBM were overfitting on ~4,800 training rows: reducing to
  100 cut fit time roughly 3x (20s → 7s for all three targets) while
  *improving* out-of-sample MAE on every target tested. This is not a
  speed/accuracy tradeoff.
- **Shared caching across pages** (`cached_forecasts.py`). `app.py`, page
  1's fallback, and page 2 previously each fit their own copy of the same
  models independently. A single `@st.cache_data`-decorated function,
  imported by all three, means the first page visited in a session pays
  the fit cost once; subsequent pages reuse the cached result for the same
  training-window setting.

Page 4 (no forecasting; MILP solves only) and page 3 (fits once across a
larger multi-day window, not shareable with the single-next-day cache)
were not restructured, but page 3 benefits automatically from the
`n_estimators` change — its default 14-day window dropped from ~47s to
~32s.

## Limitations

- The shutdown-indicator variable (`v` in `unit_commitment.py`) is
  correctly constrained for MILP feasibility but is not a reliable
  standalone signal on hours where the unit was already off.
  `analysis.py`'s Gantt chart derives on/off transitions from the
  commitment variable directly.
- Battery charge/discharge has no mutual-exclusivity constraint; relies on
  cost structure to discourage simultaneous charge and discharge.
- Fleet capacities are hand-tuned so the peaker has a visible, occasional
  role (engages on 32% of days); not derived from a capacity-planning
  study.
- The simulated electricity price (`simulate_price_series`) is an
  exogenous merit-order proxy, not the shadow price of any UC solve —
  deliberate, so the battery has a genuine external signal to react to.
- Joint scenario probabilities are estimated from in-sample (not
  walk-forward) historical forecast errors. Error magnitudes are
  optimistic; the correlation between demand and renewable errors is
  preserved correctly.
- Combined wind+solar quantile forecasts are computed by summing matching
  quantile levels, which carries its own independence assumption between
  wind and solar specifically — separate from the demand-vs-renewable
  correlation fix in `scenarios.py`.
- No transmission constraints, no reserve-margin requirements, single bus.
- The sensitivity page's stochastic overlay uses a simplified,
  independence-weighted scenario set (page 4) rather than the empirically
  estimated probabilities used elsewhere (`scenarios.py`, pages 2–3),
  since its representative day is not anchored to a specific forecast
  window.

## Remaining roadmap

- **EV charging as flexible demand.** A controllable load the optimizer
  can shift within a charging window, alongside the battery.
- **Decision-focused learning.** Train the forecast model's loss on
  downstream dispatch cost rather than quantile accuracy.
- **Walk-forward backtesting.** Replace the in-sample error estimates
  used for joint scenario probabilities with true rolling retraining.
- **Real data.** Swap the synthetic generator for NREL/EIA data (the
  swap-in path is documented in `data_gen.py`).
- **Disk-based precomputation.** The shared cache (`cached_forecasts.py`)
  eliminates redundant fits within a running session but still pays the
  full cost on a fresh process start. Precomputing and serializing the
  default scenario's forecast fit to disk would remove that cold-start
  cost too; not implemented, since the underlying data generation's
  determinism (fixed-seed RNG at module import) would need care to keep
  a disk cache from silently going stale against code changes.
