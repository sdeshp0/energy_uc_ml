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
├── cached_forecasts.py      shared Streamlit cache for the demand/wind/solar forecast fit
├── scenarios.py             empirical joint (demand x renewable) scenario probabilities
├── rolling_horizon.py       multi-day walk-forward simulation engine
├── analysis.py              residual load, sweeps, economics, all plotting
├── pipeline.py              CLI: forecast-quality comparison (P10 vs P50 vs persistence)
├── app.py                   main Streamlit dashboard
└── pages/
    ├── 1_Market_And_Battery_Arbitrage.py    simulated prices, settlement, battery arbitrage
    ├── 2_Stochastic_Unit_Commitment.py      the two-stage stochastic hedge
    ├── 3_Rolling_Horizon_Simulation.py      multi-day comparison of both approaches
    └── 4_Sensitivity_Analysis.py            parameter sweeps, optional stochastic overlay
```

Modules that don't need Streamlit don't import it (`data_gen`,
`forecasting`, `unit_commitment`, `scenarios`, `analysis`, `scenario`,
`rolling_horizon` are plain Python, directly unit-testable). Logic used by
more than one page lives outside `app.py` rather than being duplicated or
imported from a script with top-level UI side effects.

**Import convention.** All project modules import each other by bare name
(`from analysis import ...`), with `src/` as the import root. Mixing in
`src.`-prefixed imports loads the same file twice under two names, which
has three concrete effects: `cached_forecasts.get_day_and_forecasts`
becomes two distinct functions with separate `st.cache_data` entries
(defeating the shared cache, §14); module-level state such as
`data_gen.RNG` is duplicated; and classes such as `UCResult` are defined
twice. IDE import warnings are handled in the IDE, not the code: PyCharm
via Sources Root on `src/`, Pylance/pyright via `[tool.pyright]
extraPaths` in `pyproject.toml`.

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
P50 MAE ≈ 21.6 MW; wind P50 MAE ≈ 79.5 MW / 300 MW capacity, worse than
naive persistence (40.8 MW). The wind forecaster's bias is entirely
one-directional — +79.5 MW, equal to its MAE — meaning it overestimates
in every hour of the test day. The P10 quantile pulls that back to a
−59.2 MW average underestimate. This bias direction, not raw accuracy, is
what the unit commitment result in §5 depends on.

`QuantileForecaster.clip_range` bounds predictions: `(0.0, 1.0)` for a
capacity factor, `(0.0, None)` for demand (non-negative, unbounded above).
Quantiles are sorted per row after prediction to guarantee P10 ≤ P50 ≤
P90, since each quantile is fit independently and can otherwise cross.

`n_estimators=100` (previously 300) is a measured, not assumed, choice:
at 300 estimators the model was overfitting on ~4,800 training rows, and
100 was both faster and more accurate on every target tested (§14).

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
- **Reserve** (optional, §4a): total and spinning headroom requirements
  against residual load, with shortfalls as penalized slack.

### 4a. Reserve requirements

Two optional margins, both disabled by default (`reserve_margin=0.0`,
`spin_reserve_margin=0.0`), sized as a fraction of **residual load**
(`demand − renewable`), not raw demand. This fleet's total thermal
capacity (800 MW) is often below raw demand (peaks above 1,100 MW) by
design — the system is meant to rely on renewables — so a margin sized
against raw demand would be infeasible on ordinary days, consistent with
the project's residual-load framing (§9's `residual_load` function is the
same quantity).

**Total reserve**: `Σ_g (pmax[g]·u[g,t] − p[g,t]) + reserve_short[t] ≥
reserve_margin · residual[t]` — the sum of every committed unit's unused
headroom must cover the margin, or the gap shows up as `reserve_short`.

**Spinning reserve** is stricter: a unit's contribution isn't just its
headroom but is additionally capped by how much it can ramp up within a
response window (`spin_response_hours`, default 1/6 = 10 minutes, a
standard convention). This needs a new per-generator variable
`spin[g,t]`, bounded by two linear constraints —
`spin[g,t] ≤ pmax[g]·u[g,t] − p[g,t]` (headroom) and
`spin[g,t] ≤ ramp[g]·spin_response_hours` (deliverability) — since MILP
constraints can't express `min(a, b)` directly. Then
`Σ_g spin[g,t] + spin_short[t] ≥ spin_reserve_margin · residual[t]`.
Verified directly: a slow-ramping unit (Coal_1, ramp 70 MW/hr) with ~210
MW of idle headroom contributed almost nothing to spinning reserve, while
a faster unit (CCGT_2, ramp 90 MW/hr) was capped exactly at
`90 × 1/6 = 15 MW` — the ramp bound binding exactly as intended, not the
much larger headroom bound.

Both shortfalls are **penalized slack**, not hard constraints
(`reserve_penalty`, default 1000 $/MW — below `unserved_penalty`'s 5000,
since a reserve shortfall is a reliability-standard violation, not actual
unserved demand). A hard constraint would have made the MILP infeasible
on tight days rather than degrading gracefully; measuring the actual
shortfall is also a more useful output than a solve failure. Measured: a
15% total margin raised planned cost 1.8% with zero shortfall (the fleet
could hold it); adding an 8% spinning margin on top raised cost further
and produced a small (0.4 MW) spinning shortfall, confirming spinning
reserve is the tighter of the two constraints.

The stochastic model (§7) applies both margins **per scenario**, since
each scenario has its own residual load and its own dispatch, but shares
one commitment across all of them. Verified: in a 3-scenario hedge, a
15%/8% requirement produced shortfall only in the worst (high-demand/
low-renewable) scenario — the other two held full margin, since the
shared commitment was already sized generously by the hedge itself.

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
- **`reserve_margin`, `spin_reserve_margin`, `spin_response_hours`,
  `reserve_penalty`**: see §4a. Applied by `scenario.py` only to the
  PLANNED (day-ahead) solve, not the perfect-foresight benchmark (no
  uncertainty to hedge against) or the realized/settlement solve
  (commitment is already fixed by then).

### Objective

`Σ marginal_cost[g]·p[g,t] + Σ startup_cost[g]·s[g,t] + Σ
unserved_penalty·unserved[t] + Σ reserve_penalty·(reserve_short[t] +
spin_short[t])`, plus the optional price term above.

### Result: planned vs. realized cost

Comparing each forecast scenario's cost as computed under that forecast
is misleading: an optimistic forecast lets the solver under-commit
capacity and appears cheap. Fixing the commitment and re-solving against
actual conditions (`fixed_commitment`) is what exposes the true cost.
Measured: perfect foresight $358,274; P50 forecast realized $6,682,672;
P10 (conservative) realized $359,999; naive persistence realized
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
re-execute its sidebar). `run_pipeline()` accepts an optional
`prefetched` dict so the forecast-fitting step can be supplied by the
shared cache (§14) instead of fitting locally, while `scenario.py` itself
stays Streamlit-free.

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
   Measured: correlation −0.13; high-demand/low-renewable probability
   0.124 empirically vs. 0.111 under independence.
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
| Variables | `u[g,t]`, `s[g,t]`, `v[g,t]` | `p[g,t,w]`, `charge[t,w]`, `discharge[t,w]`, `soc[t,w]`, `curt[t,w]`, `unserved[t,w]`, `spin[g,t,w]`, `reserve_short[t,w]`, `spin_short[t,w]` |
| Rationale | Commitment must be decided before any scenario is known | Dispatch -- and therefore headroom, and therefore reserve adequacy -- adjusts once the scenario resolves |

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

P50-only: 8 of 14 days with unserved demand, 458.7 MWh total unserved,
$9,651,519 total cost. Stochastic hedge: 1 of 14 days with unserved
demand, 0.8 MWh total unserved, $4,986,721 total cost — lower on both
reliability and aggregate cost over this window, despite costing a
premium on any single day analyzed in isolation (§7). The one day both
approaches fail on is informative: a nine-scenario hedge reduces risk, it
does not eliminate outcomes beyond its scenario coverage.

Default window runtime is approximately 32 seconds (down from ~47s
before the `n_estimators` tuning in §14), dominated by the one-time model
fit rather than the per-day solves.

---

## 9. Market economics layer (`analysis.py` + page 1)

Kept separate from the cost-minimization objective except for the
optional battery arbitrage term.

- **`generator_economics(dispatch, fleet, price)`**: revenue minus
  fuel/var-O&M cost minus startup cost, per generator, applied after
  solving. The marginal (most expensive committed) unit should show
  near-zero margin; cheaper units a positive one — standard merit-order
  economics, useful as a sanity check on the price series.
- **`battery_economics(battery, price)`**: per-hour charge cost,
  discharge revenue, net and cumulative P&L.
- **Battery arbitrage**: page 1 solves the same demand/renewable/fleet/
  battery twice, with and without the price term, and reports true
  production cost (recomputed from dispatch, not the solver's raw
  `total_cost`) alongside battery P&L. Measured on one representative
  day: production cost unchanged ($0 delta), battery P&L $2,207 → $2,616
  (+18.5%) — a contained shift, not a change in overall system behavior.

---

## 10. Sensitivity analysis (`analysis.py` + page 4)

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
estimated probabilities: page 4's representative day is chosen by
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
  fetching the forecast fit through the shared cache
  (`cached_forecasts.get_day_and_forecasts`, §14) rather than fitting
  locally, and rendering the forecast chart, residual load, fleet table,
  commitment Gantt, dispatch stack, ramp detail, and battery SoC. Two
  reserve sliders (total, spinning; both 0% by default) apply to the
  planned solve (§4a); their metrics only render when either is nonzero.
- **Page 1 (Market & Battery Arbitrage)**: settlement reporting on the
  realized dispatch, and an independent arbitrage-on/off comparison (§9).
  Its fallback scenario (when the main page hasn't been visited) also
  goes through the shared cache.
- **Page 2 (Stochastic Unit Commitment)**: builds the nine scenarios (§6),
  solves the two-stage MILP (§7), and compares against P50-only. Uses the
  shared cache directly rather than its own local fit.
- **Page 3 (Rolling Horizon Simulation)**: the multi-day walk-forward
  comparison (§8). Fits its own models across a larger window (training +
  simulation days), which the single-next-day shared cache doesn't cover,
  so it keeps its own `@st.cache_data` entry point.
- **Page 4 (Sensitivity Analysis)**: reads the main page's slider values
  as sweep baselines, falling back to defaults if unvisited this session;
  runs four sweeps, optionally with the stochastic overlay (§10). Does
  not fit any forecast model, so is unaffected by §14.

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

---

## 14. Performance

The demand/wind/solar quantile forecast fit dominates runtime on every
page that needs one — roughly 20s combined at the original settings,
against MILP solves that run in well under a second each. Two changes
address this.

**`n_estimators` tuning** (`forecasting.py`). Each `GradientBoostingRegressor`
used 300 estimators. Benchmarked against 150/100/75/50 on the project's
actual training data (~4,800 rows), 300 was consistently worse than 100
on both fit time and out-of-sample MAE, across all three targets:

| n_estimators | fit time (single quantile) | wind P50 MAE | demand P50 MAE |
|---|---|---|---|
| 300 | 7.65s | 48.0 | 21.8 |
| 150 | 3.78s | 47.3 | 21.7 |
| 100 | 2.50s | 44.2 | 21.3 |
| 75  | 1.88s | 45.1 | 20.7 |

300 estimators was overfitting on this data size; 100 is a measured
sweet spot rather than a speed/accuracy tradeoff (75 starts trading real
solar accuracy for speed — see the fuller table in the commit history).
Net effect: fitting all three targets dropped from ~20s to ~7s.

**Shared caching across pages** (`cached_forecasts.py`). Before this
change, `app.py`, page 1's fallback path, and page 2 each independently
fit their own copies of the same three models for the same
`n_days_history` — Streamlit's `@st.cache_data` is keyed per decorated
function object, so three separately-defined local wrapper functions
produce three separate cache entries even when the underlying computation
is identical. `cached_forecasts.get_day_and_forecasts(n_days_history)` is
now the single decorated function all three import and call; the first
page visited in a session pays the fit cost, and every subsequent page
using the same `n_days_history` hits the cache instead of refitting.
`scenario.run_pipeline()` accepts the result as an optional `prefetched`
argument so it stays Streamlit-free itself — the caller is responsible
for going through the shared cache.

Page 4 does no forecasting (its sweeps operate on actual historical data
via `representative_day`, not a forecast) and is unaffected by either
change. Page 3 fits its own models across a training-plus-simulation
window that the single-next-day shared cache doesn't cover, so it keeps
its own cache entry point, but benefits automatically from the
`n_estimators` change: its default 14-day window dropped from ~47s to
~32s.

**Not implemented**: disk-based precomputation of the default scenario.
The shared cache removes redundant fits within a running session but
still pays full cost on a fresh process start. A disk cache would remove
that too, at the cost of needing an invalidation strategy so it doesn't
silently serve stale results after a code change — judged not worth the
complexity given the shared in-session cache already addresses the more
common case (navigating between pages during one sitting).
