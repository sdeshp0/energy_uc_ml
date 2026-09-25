# Code Walkthrough: ML-Forecast-Driven Unit Commitment

Technical reference for the codebase: data model, optimization
formulations, ML pipeline, and how a request flows from a Streamlit page
to a MILP solve and back.

---

## 1. Repository layout

```
src/
├── data_gen.py              synthetic demand/wind/solar, thermal fleet, price simulation
├── forecasting.py           quantile (P10/P50/P90) forecasters
├── unit_commitment.py       single-scenario MILP and two-stage stochastic MILP
├── scenario.py              shared single-day scenario builder for app.py and pages/
├── scenarios.py             empirical joint (demand x renewable) scenario probabilities
├── rolling_horizon.py       multi-day walk-forward simulation engine
├── analysis.py              residual load, sweeps, economics, all plotting
├── pipeline.py              CLI: forecast-quality comparison (P10 vs P50 vs persistence)
├── app.py                   main Streamlit dashboard
└── pages/
    ├── 1_Sensitivity_Analysis.py            parameter sweeps, optional stochastic overlay
    ├── 2_Market_And_Battery_Arbitrage.py    simulated prices, settlement, battery arbitrage
    ├── 3_Stochastic_Unit_Commitment.py      the two-stage stochastic hedge
    └── 4_Rolling_Horizon_Simulation.py      multi-day comparison of both approaches
```

Modules that don't need Streamlit don't import it (`data_gen`,
`forecasting`, `unit_commitment`, `scenarios`, `analysis`, `scenario`,
`rolling_horizon` are plain Python, directly unit-testable). Logic used by
more than one page lives outside `app.py` rather than being duplicated or
imported from a script with top-level UI side effects.

---

## 2. Data generation (`data_gen.py`)

### Synthetic hourly series

`generate_hourly_dataset(n_days)` builds demand, wind, and solar as
separate stochastic processes, then correlates them:

- **Demand**: two-peak (morning + evening) diurnal shape — base 0.50,
  morning amplitude 0.18, evening amplitude 0.38 — plus weekend derate and
  Gaussian noise. The peak-to-trough ratio is deliberately pronounced
  (roughly 1.9x) so battery and peaker flexibility have a genuine daily
  swing to respond to; an earlier, flatter version (base 0.65, smaller
  amplitudes) left battery sizing sliders showing too little effect over
  most of their range.
- **Solar**: seasonal-amplitude bell curve over daylight hours.
- **Wind**: mean-reverting process, `noise[i] = 0.9*noise[i-1] +
  N(0, 0.12)`, weakly seasonal — hard to forecast a day ahead by
  construction, which drives several of the project's findings (§3).
- **Cold-snap shock**: 5% of days apply `demand *= 1.16` and
  `wind_cf *= 0.4` simultaneously, representing a correlated weather
  event. Without it, the three series are statistically independent and
  there is nothing for the joint-probability estimation in §6 to find.

Resulting demand ranges roughly 400–1,109 MW across a generated year.

### Thermal fleet (`thermal_fleet_spec`)

Four units — `Coal_1` (350 MW, $26.10/MWh), `CCGT_1` (200 MW, $34.50/MWh),
`CCGT_2` (150 MW, $37.20/MWh), `GasPeaker_1` (100 MW, $56.75/MWh) at
default fuel prices — each with `pmin/pmax`, ramp rate, min-up/min-down
time, startup cost, and cost decomposed as `heat_rate (MMBtu/MWh) ×
fuel_price ($/MMBtu) + var_om`. `fuel_type` (`coal`/`gas`) determines
which slider moves a unit's cost. Capacities are sized so
`Coal + CCGT1 + CCGT2 + battery` (700 + 60 = 760 MW) sits below the daily
peak residual load on 32% of days — enough that the peaker has a real,
occasional role without dominating.

### Price simulation (`simulate_price_series`)

An exogenous hourly price: the marginal cost of the last unit needed to
cover that hour's residual load, read from a merit-order stack of the
fleet's own marginal costs, with noise, floored during renewable
oversupply and capped at a scarcity ceiling. Deliberately not the shadow
price of a UC solve — that would make it circular for the battery to
react to (§9).

---

## 3. Forecasting (`forecasting.py`)

One `GradientBoostingRegressor` per quantile (P10/P50/P90), per target,
using `loss="quantile"`.

- **`fit_forecast_and_history`**: fits once, returns both the next-day
  forecast and in-sample predictions on the training rows. `forecast_
  next_day` wraps this and discards the second output.
- **`fit_and_predict_range`**: fits once on an initial training window,
  then predicts across every subsequent row from that single fit, used by
  the rolling-horizon simulation (§8) so a multi-day run doesn't refit the
  model per day.
- **`historical_quantile_predictions`**: standalone in-sample predictions
  when no next-day forecast is also needed.

**Features**: calendar (hour, day-of-week, sin/cos of hour and
day-of-year) plus lags of the target at 24h, 48h, and 168h, and a rolling
mean. Measured accuracy: solar P50 MAE ≈ 5 MW / 250 MW capacity; demand
P50 MAE ≈ 21.7 MW; wind P50 MAE ≈ 84.5 MW / 300 MW capacity, worse than
naive persistence (40.8 MW). The wind forecaster's bias is entirely
one-directional — +84.5 MW, equal to its MAE — meaning it overestimates
in every hour of the test day. The P10 quantile pulls that back to a
−42.0 MW average underestimate. This bias direction, not raw accuracy, is
what the unit commitment result in §5 depends on.

`QuantileForecaster.clip_range` bounds predictions: `(0.0, 1.0)` for a
capacity factor, `(0.0, None)` for demand (non-negative, unbounded above).
Quantiles are sorted per row after prediction to guarantee P10 ≤ P50 ≤
P90, since each quantile is fit independently and can otherwise cross.

---

## 4. The core optimization model (`unit_commitment.py`, `UnitCommitmentModel`)

A mixed-integer program solved via `scipy.optimize.milp` (HiGHS backend).
One instance covers one 24-hour horizon.

### Decision variables (per generator *g*, hour *t*)

| Variable | Type | Meaning |
|---|---|---|
| `u[g,t]` | binary | committed (on/off) |
| `p[g,t]` | continuous | power output (MW) |
| `s[g,t]` | binary | startup indicator |
| `v[g,t]` | binary | shutdown indicator |

Plus, per hour: battery `charge[t]`, `discharge[t]`, `soc[t]`
(continuous), renewable `curtailment[t]`, and `unserved[t]` (a penalized
slack variable, 5000 $/MWh by default, so the model stays feasible rather
than reporting infeasible when it cannot physically meet demand).

All variables are packed into one flat vector; each family gets a fixed
offset and an index function (`iu(g,t)`, `ip(g,t)`, …) mapping `(g,t)` to
a position. Constraints are `LinearConstraint` rows built by setting a few
entries of a zero vector.

### Constraints

- **Power balance**: `Σ_g p[g,t] + (renewable[t] − curt[t]) +
  discharge[t] − charge[t] + unserved[t] = demand[t]`.
- **Curtailment bound**: `curt[t] ≤ renewable[t]`.
- **Commitment linking**: `pmin·u[g,t] ≤ p[g,t] ≤ pmax·u[g,t]`.
- **Startup/shutdown linking**: `s[g,t] ≥ u[g,t] − u[g,t−1]`,
  `v[g,t] ≥ u[g,t−1] − u[g,t]`. Nothing in the objective penalizes `v`, so
  it is under-constrained on already-off hours and the solver can set
  spurious values there. Code reading `dispatch` for shutdown timing
  should derive it from `u`'s transitions directly, as `analysis.py`'s
  Gantt chart does, rather than trust the `shutdown` column.
- **Min-up/min-down time**: the sum of startups (or shutdowns) in the
  preceding `min_up` (or `min_down`) window bounds `u[g,t]`.
- **Ramp limits**: `|p[g,t] − p[g,t−1]| ≤ ramp[g]`.
- **Battery dynamics**: `soc[t] = soc[t−1] + η·charge[t] −
  discharge[t]/η`, bounded within `[soc_min_frac, soc_max_frac] ×
  capacity`. No mutual-exclusivity constraint between charging and
  discharging in the same hour.

### Optional arguments

- **`fixed_commitment`**: an optional `(G, T)` 0/1 array pinning every
  `u[g,t]`, turning the MILP into an LP over dispatch alone. Used
  throughout the project to compute realized cost: solve once under a
  forecast, then re-solve with that commitment fixed and actual
  demand/renewable substituted.
- **`price`**: an optional length-T array. Adds `price[t]·charge[t]` as a
  cost and `price[t]·discharge[t]` as a revenue credit — a battery
  arbitrage incentive on top of production-cost minimization, not a
  market reformulation. `total_cost` from a price-aware solve mixes
  production cost with arbitrage revenue and is not directly comparable
  to a `price=None` solve; recompute true production cost from the
  dispatch instead (`analysis.generator_economics`, §9).
- **`soc_init_mwh`**: overrides the default starting state of charge
  (`soc_init_frac × capacity`) with an explicit value — the previous
  day's ending SoC, when chaining solves across days (§8).
- **`soc_terminal_min_mwh`**: a floor on the final hour's state of charge.
  Without it, a finite-horizon solve has no incentive to end with any
  charge above the physical minimum, since stored energy has no value in
  the objective once the horizon ends. Harmless for a single isolated day;
  fatal for a rolling multi-day simulation, where every day would start
  from the previous day's floor. §8 covers the effect of omitting this.

### Objective

`Σ marginal_cost[g]·p[g,t] + Σ startup_cost[g]·s[g,t] + Σ
unserved_penalty·unserved[t]`, plus the optional price term above.

### Result: planned vs. realized cost

Comparing each forecast scenario's cost as computed under that forecast
is misleading: an optimistic forecast lets the solver under-commit
capacity and appears cheap. Fixing the commitment and re-solving against
actual conditions (`fixed_commitment`) is what exposes the true cost.
Measured: perfect foresight $358,274; P50 forecast realized $7,427,092;
P10 (conservative) realized $359,608; naive persistence realized
$503,223. P10's realized cost nearly matches perfect foresight despite
P10 not being the most accurate forecast on offer (§3).

---

## 5. Codeflow: a single-scenario run

`pipeline.py` and `app.py` follow this sequence:

```
generate_hourly_dataset()
        │
        ├─► history (all but last 24h), next_day (last 24h, actuals held out)
        ▼
fit_forecast_and_history()  for wind_cf, solar_cf (and demand_mw, for §6)
        ▼
chosen_renewable = wind_pXX + solar_pXX   (XX = selected quantile)
        ▼
UnitCommitmentModel.build_and_solve(demand, chosen_renewable)  →  "planned"
        ▼
fix commitment, re-solve against ACTUAL renewable  →  "realized"
        ▼
analysis.py: residual load / Gantt / ramp headroom / battery SoC
```

`scenario.py`'s `run_pipeline()` implements this sequence once, so
`app.py` and any page can build an identical scenario without importing
`app.py` itself (which has top-level Streamlit calls that would
re-execute its sidebar).

---

## 6. Joint scenario probabilities (`scenarios.py`)

Replaces an independence assumption when combining demand and renewable
uncertainty into a discrete scenario set.

1. **`paired_errors_from_predictions`**: takes in-sample P50 predictions
   for demand, wind, and solar (already computed by
   `fit_forecast_and_history` for the next-day forecast — no redundant
   refit) and computes `error = actual − P50` for demand and for combined
   renewable (`wind_error + solar_error`, exact regardless of correlation
   between wind and solar, since error of a sum is the sum of errors).
   This is in-sample, not a walk-forward backtest: error magnitudes are
   optimistic, but the co-movement between demand and renewable errors at
   the same hour is preserved, which is the only property this step needs.
2. **`tercile_labels`**: equal-count low/mid/high split at the 1/3 and 2/3
   quantiles of the error distribution.
3. **`joint_scenario_probabilities`**: the empirical joint frequency of
   the two tercile labelings via `pd.crosstab`.
   **`independence_baseline_probabilities`** computes the alternative
   (outer product of the marginals) from the same data for comparison.
   Measured: correlation −0.112; high-demand/low-renewable probability
   0.123 empirically vs. 0.111 under independence.
4. **`build_nine_scenarios`**: maps `low/mid/high → P10/P50/P90` for both
   variables, pairing each of the nine combinations with its joint
   probability.

---

## 7. The stochastic MILP (`StochasticUnitCommitmentModel`)

A two-stage stochastic program, not nine independent solves averaged
afterward — several different discrete commitment schedules cannot be
blended into one implementable plan.

### Shared vs. scenario-specific

| | Shared (1st stage) | Per-scenario (2nd stage / recourse) |
|---|---|---|
| Variables | `u[g,t]`, `s[g,t]`, `v[g,t]` | `p[g,t,w]`, `charge[t,w]`, `discharge[t,w]`, `soc[t,w]`, `curt[t,w]`, `unserved[t,w]` |
| Rationale | Commitment must be decided before any scenario is known | Dispatch adjusts once the scenario resolves |

Index functions extend the single-scenario pattern with a scenario axis:
`ip(g,t,w) = off_p + (g·T + t)·W + w`. Constraints mirror §4's, looped
over `w`; startup/shutdown linking and min-up/min-down time stay
first-stage-only, since they describe the commitment decision itself.

### Objective

`Σ startup_cost[g]·s[g,t]` (shared, paid once) `+ Σ_w probability[w] ·
(Σ marginal_cost[g]·p[g,t,w] + unserved_penalty·Σ unserved[t,w])`
(expected recourse cost). With exactly one scenario (`probability=1`),
this reduces to `UnitCommitmentModel`'s result exactly, confirmed by
direct comparison — the two-stage formulation is a strict generalization.

### Result and measured effect

`StochasticUCResult` carries the shared commitment once, plus one
`ScenarioOutcome` per scenario with its own dispatch, battery trajectory,
and `cost = shared_startup_cost + that_scenario's_recourse_cost`.

Committing on the P50 scenario alone, then facing a high-demand/
low-renewable scenario: 186.8 MW unserved, $8,932,848. The same scenario,
same fleet, with the stochastic hedge's commitment instead: 0 MW
unserved, $360,967. The hedge costs more in expectation across all nine
scenarios — more thermal capacity committed than P50 alone would choose —
in exchange for eliminating the worst-case failure.

---

## 8. Rolling-horizon simulation (`rolling_horizon.py`)

Every model above solves one isolated 24-hour day. `rolling_horizon.py`
walks forward day by day over a multi-day window, carrying battery state
of charge and generator on/off status from each day's end into the next
day's start, and runs both commitment approaches side by side against
identical actual conditions each day.

### Fitting once, simulating many days

Each target's quantile models are fit once, at the start of the window
(`fit_and_predict_range`), and applied across every simulated day.
Refitting daily would cost several seconds × 9 models × N days for
limited benefit over a few weeks. Joint scenario probabilities (§6) are
likewise computed once, from the training window's in-sample errors.

### State continuity and the terminal SoC constraint

`soc_init_mwh` carries each approach's battery state forward
independently (the two approaches generally commit differently, so each
needs its own trajectory). Without `soc_terminal_min_mwh`, this loop
surfaced a real bug during testing: with no value placed on ending
charge, the optimizer drained the battery to its floor by the last hour
of every single day, crippling the next day's starting flexibility. The
terminal constraint is applied at both the planning solve (so the
day-ahead plan accounts for it) and the settlement solve (`fixed_
commitment` still leaves charge/discharge timing free, so the constraint
remains satisfiable under actual conditions even though the commitment
itself was chosen under the forecast).

### Measured effect, 14-day default window

P50-only: 6 of 14 days with unserved demand, 457.4 MWh total unserved,
$10,013,505 total cost. Stochastic hedge: 1 of 14 days with unserved
demand, 0.8 MWh total unserved, $4,972,930 total cost — lower on both
reliability and aggregate cost over this window, despite costing a
premium on any single day analyzed in isolation (§7). The one day both
approaches fail on is informative: a nine-scenario hedge reduces risk, it
does not eliminate outcomes beyond its scenario coverage.

Default window runtime is approximately 40–50 seconds, dominated by the
one-time model fit rather than the per-day solves.

---

## 9. Market economics layer (`analysis.py` + page 2)

Kept separate from the cost-minimization objective except for the
optional battery arbitrage term.

- **`generator_economics(dispatch, fleet, price)`**: revenue minus
  fuel/var-O&M cost minus startup cost, per generator, applied after
  solving. The marginal (most expensive committed) unit should show
  near-zero margin; cheaper units a positive one — standard merit-order
  economics, useful as a sanity check on the price series.
- **`battery_economics(battery, price)`**: per-hour charge cost,
  discharge revenue, net and cumulative P&L.
- **Battery arbitrage**: page 2 solves the same demand/renewable/fleet/
  battery twice, with and without the price term, and reports true
  production cost (recomputed from dispatch, not the solver's raw
  `total_cost`) alongside battery P&L. Measured on one representative
  day: production cost unchanged ($0 delta), battery P&L $2,207 → $2,616
  (+18.5%) — a contained shift, not a change in overall system behavior.

---

## 10. Sensitivity analysis (`analysis.py` + page 1)

`run_sweep` is a generic engine: `apply_fn(fleet, battery, v) → (fleet,
battery)` is the only thing that varies between a fuel-price sweep and a
battery-size sweep; solving and metric extraction are shared. Each
single-scenario solve is roughly 0.25s, so an 8–10 point sweep costs a
few seconds.

**Stochastic overlay**: `illustrative_nine_scenarios` builds a simplified
9-scenario set — symmetric ±5% demand / ±30% renewable perturbations,
independence-weighted — around the page's representative day, and
`sweep_fuel_price_stochastic` / `sweep_battery_param_stochastic` sweep
the two-stage MILP's expected and worst-case cost alongside the
single-scenario cost. This deliberately does not reuse §6's empirically
estimated probabilities: page 1's representative day is chosen by
peak-residual percentile from a freshly generated dataset, not anchored
to a specific forecast/history window, so there is no paired historical
error data available without fitting fresh models (an added ~18s the
page is built to avoid). Each stochastic solve costs roughly 1s, so the
full overlay is an order of magnitude slower than the single-scenario
sweeps alone; a sidebar checkbox makes it optional.

Measured ranges: coal price swept $1–8/MMBtu changes total cost by 147%.
Battery power swept 0–150 MW shows diminishing returns past roughly
60–65 MW for a 200 MWh battery — the energy, not the power rating,
becomes the binding constraint past that point, a property of the
power-to-energy ratio rather than a tuning artifact.

---

## 11. Analysis & reporting utilities (`analysis.py`)

Plain functions operating on the dataclasses the two UC models return, no
Streamlit dependency:

- **`residual_load` / `thermal_and_battery_coverage`**: demand minus
  renewables is what thermal and battery must jointly cover.
- **`commitment_matrix` / `plot_commitment_gantt`**: on/off blocks per
  generator, derived from `u`'s transitions (not the `shutdown` variable;
  §4).
- **`ramp_headroom`**: flags hours where a generator is within 95% of its
  ramp limit.
- **`plot_battery_soc`**: state-of-charge trajectory with min/max bounds
  and charge/discharge bars.

---

## 12. Application layer

- **`app.py`**: sidebar sliders (explicit `key=`s so other pages can read
  `st.session_state["coal_price"]` etc.) drive `scenario.run_pipeline()`,
  rendering the forecast chart, residual load, fleet table, commitment
  Gantt, dispatch stack, ramp detail, and battery SoC.
- **Page 1 (Sensitivity Analysis)**: reads the main page's slider values
  as sweep baselines, falling back to defaults if unvisited this session;
  runs four sweeps, optionally with the stochastic overlay (§10).
- **Page 2 (Market & Battery Arbitrage)**: settlement reporting on the
  realized dispatch, and an independent arbitrage-on/off comparison (§9).
- **Page 3 (Stochastic Unit Commitment)**: builds the nine scenarios (§6),
  solves the two-stage MILP (§7), and compares against P50-only.
- **Page 4 (Rolling Horizon Simulation)**: the multi-day walk-forward
  comparison (§8).

All pages fall back to `scenario.DEFAULTS` when `st.session_state` lacks
a prior scenario, so each also works as a standalone entry point.

---

## 13. Known simplifications

1. `v` (shutdown indicator) is loosely constrained on already-off hours —
   correct for MILP feasibility, not meaningful for direct display.
2. Battery has no charge/discharge mutual-exclusivity constraint.
3. Fleet capacities are hand-tuned for the peaker's role to be visible,
   not derived from a capacity-planning study.
4. Wind and solar quantiles are combined by simple addition
   (`wind_p10 + solar_p10 = renewable_p10`), carrying its own implicit
   independence assumption between wind and solar specifically — separate
   from the demand-vs-renewable correlation fix in §6.
5. Joint probability estimation uses in-sample, not walk-forward,
   historical errors (§6, §8).
6. The simulated price (§2) is an exogenous merit-order proxy, not the
   shadow price of any UC solve.
7. No transmission or reserve-margin constraints; single bus.
8. The sensitivity page's stochastic overlay (§10) uses a simplified,
   independence-weighted scenario set rather than §6's empirically
   estimated probabilities, since its representative day has no
   associated forecast/history window.
