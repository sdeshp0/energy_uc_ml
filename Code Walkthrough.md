# Code Walkthrough: ML-Forecast-Driven Unit Commitment

A technical companion to the README, for understanding how the code actually
works: the data model, the optimization formulations, the ML pipeline, and
how a request flows from a Streamlit page down to a MILP solve and back.

---

## 1. Repository layout

```
src/
├── data_gen.py              synthetic demand/wind/solar, thermal fleet spec, price simulation
├── forecasting.py           quantile (P10/P50/P90) ML forecasters
├── unit_commitment.py       the two MILP models (single-scenario + two-stage stochastic)
├── scenario.py              shared scenario-building logic for the main app
├── scenarios.py             joint (demand x renewable) probability estimation for stochastic UC
├── analysis.py              residual load, sweeps, economics, all plotting
├── pipeline.py              CLI: forecast-quality comparison (P10 vs P50 vs persistence)
├── app.py                   main Streamlit dashboard
└── pages/
    ├── 1_Sensitivity_Analysis.py       parameter sweeps (fuel prices, battery sizing)
    ├── 2_Market_And_Battery_Arbitrage.py  simulated prices, revenue, battery arbitrage
    └── 3_Stochastic_Unit_Commitment.py    the two-stage stochastic hedge
```

Two design rules hold throughout: **modules that don't need Streamlit don't
import it** (`data_gen`, `forecasting`, `unit_commitment`, `scenarios`,
`analysis`, `scenario` are all plain Python, directly unit-testable), and
**business logic used by more than one page lives outside `app.py`**
(`scenario.py`) rather than being duplicated or imported from a script with
top-level UI side effects.

---

## 2. Data generation (`data_gen.py`)

### Synthetic hourly series
`generate_hourly_dataset(n_days)` builds demand, wind, and solar as three
separate stochastic processes, then correlates them:

- **Demand**: a two-peak (morning + evening) diurnal shape, weekend derate,
  Gaussian noise.
- **Solar**: a seasonal-amplitude bell curve over daylight hours (zero at
  night, by construction).
- **Wind**: a mean-reverting (AR(1)-like) process with weak seasonality —
  `noise[i] = 0.9*noise[i-1] + N(0, 0.12)` — deliberately hard to forecast
  a day ahead, which is realistic and drives several of the project's
  findings (see §3).
- **Cold-snap shock**: ~5% of days get `demand *= 1.16` and `wind_cf *= 0.4`
  *simultaneously* — a stand-in for a real correlated weather event (cold
  front → high heating load + still air). Without this, the three series are
  statistically independent and there is nothing for the joint-probability
  machinery in §6 to find.

### Thermal fleet (`thermal_fleet_spec`)
Four units — `Coal_1`, `CCGT_1`, `CCGT_2`, `GasPeaker_1` — each with
`pmin/pmax`, ramp rate, min-up/min-down time, startup cost, and **cost
decomposed as `heat_rate (MMBtu/MWh) × fuel_price ($/MMBtu) + var_om`**
rather than a single opaque number. `fuel_type` (`coal`/`gas`) is what a
fuel-price slider actually moves. Capacities are deliberately sized so
`Coal + CCGT1 + CCGT2 + battery` (700 + 60 = 760 MW) sits below the daily
peak residual load on ~40% of days — enough that the peaker has a real,
occasional role without dominating or being unreachable.

### Price simulation (`simulate_price_series`)
An **exogenous** hourly price: the marginal cost of the last unit needed to
cover that hour's residual load (demand − renewable), read off a
merit-order stack built from the fleet's own marginal costs, plus noise,
floored during renewable oversupply and capped at a scarcity ceiling. It's
intentionally *not* the literal shadow price of a UC solve — that would
make it circular for the battery to react to (see §7).

---

## 3. Forecasting (`forecasting.py`)

One `GradientBoostingRegressor` per quantile (P10/P50/P90), per target,
using `loss="quantile"`. Two function families:

- **`fit_forecast_and_history(history, target_col, next_day_calendar, ...)`**
  — fits once, returns *both* the next-day forecast and in-sample
  predictions on the training rows themselves. `forecast_next_day()` is a
  thin wrapper that discards the second output; call the combined function
  directly when you need both (§6 does), since fitting the same model twice
  roughly doubles wall-clock time for no benefit.
- **`historical_quantile_predictions`** — standalone in-sample predictions,
  used when you don't also need a next-day forecast.

**Features**: calendar (hour, day-of-week, sin/cos of hour and
day-of-year) plus autoregressive lags of the target itself (24h, 48h, 168h)
and a rolling mean. The lags matter enormously for wind (weak calendar
signal) and barely for solar (almost pure calendar signal) — reflected in
the measured accuracy: solar P50 MAE ≈ 5 MW / 250 MW capacity; wind P50 MAE
≈ 77–84 MW / 300 MW capacity, and notably **not better than naive
persistence** (yesterday's actuals), which lands around 40 MW MAE. The
wind forecaster's value isn't raw accuracy — it's that its quantiles let
you choose which *direction* to be biased in, which turns out to matter
more than average error for unit commitment (§5's "planned vs. realized"
comparison).

**`QuantileForecaster.clip_range`** bounds predictions: `(0.0, 1.0)` for a
capacity factor (wind/solar), `(0.0, None)` for a raw MW quantity that
can't go negative but has no fixed ceiling (demand). Quantiles are sorted
post-hoc per row to guarantee P10 ≤ P50 ≤ P90 even though each is fit as an
independent model (which can otherwise "cross").

---

## 4. The core optimization model (`unit_commitment.py`, `UnitCommitmentModel`)

A mixed-integer program solved via `scipy.optimize.milp` (HiGHS backend,
no external solver install needed). One instance = one 24-hour horizon.

### Decision variables (per generator *g*, hour *t*)
| Variable | Type | Meaning |
|---|---|---|
| `u[g,t]` | binary | committed (on/off) |
| `p[g,t]` | continuous | power output (MW) |
| `s[g,t]` | binary | startup indicator |
| `v[g,t]` | binary | shutdown indicator |

Plus, per hour: battery `charge[t]`, `discharge[t]`, `soc[t]`
(continuous), renewable `curtailment[t]`, and `unserved[t]` (a slack
variable with a heavy penalty — 5000 $/MWh by default — so the model stays
feasible even when physically unable to meet demand, rather than reporting
infeasible).

All variables are packed into one flat vector `x`; each variable family
gets a fixed offset and a small index function (`iu(g,t)`, `ip(g,t)`, …)
that maps `(g,t)` to a position in `x`. This is the pattern the whole file
is built on — constraints are just `LinearConstraint` rows built by setting
a few entries of a zero vector.

### Constraints
- **Power balance** (per hour): `Σ_g p[g,t] + (renewable[t] − curt[t]) + discharge[t] − charge[t] + unserved[t] = demand[t]`
- **Curtailment bound**: `curt[t] ≤ renewable[t]`
- **Commitment linking**: `pmin·u[g,t] ≤ p[g,t] ≤ pmax·u[g,t]` — this is
  *the* constraint that makes commitment matter: a unit can only produce
  power in the hours it's on, and only within its physical range.
- **Startup/shutdown linking**: `s[g,t] ≥ u[g,t] − u[g,t−1]`,
  `v[g,t] ≥ u[g,t−1] − u[g,t]` — standard "≥" linking. Because nothing in
  the objective penalizes `v`, it's under-constrained on already-off
  hours (the solver can set spurious `v=1`); anything reading `dispatch`
  for a "did it shut down" signal should derive it from `u`'s transitions
  directly (`analysis.py`'s Gantt chart does this) rather than trust the
  `shutdown` column.
- **Min-up/min-down time**: for each hour *t*, the sum of startups (or
  shutdowns) in the preceding `min_up` (or `min_down`) window bounds what
  `u[g,t]` is allowed to be — the textbook symmetric formulation.
- **Ramp limits**: `|p[g,t] − p[g,t−1]| ≤ ramp[g]`.
- **Battery dynamics**: `soc[t] = soc[t−1] + η·charge[t] − discharge[t]/η`,
  bounded within `[soc_min_frac, soc_max_frac] × capacity`. No mutual-
  exclusivity constraint between charging and discharging in the same
  hour — relies on cost structure discouraging it; fine for this scale,
  worth tightening for production use.

### Objective
`Σ marginal_cost[g]·p[g,t] + Σ startup_cost[g]·s[g,t] + Σ unserved_penalty·unserved[t]`
— plus, **optionally**, `Σ price[t]·charge[t] − Σ price[t]·discharge[t]`
when a `price` array is passed (§7's battery arbitrage term). This is a
blended objective (production cost *minus* arbitrage revenue against an
exogenous signal), not a market reformulation — `total_cost` from a
price-aware solve is not directly comparable to one without; true
production cost should be recomputed from the dispatch instead
(`analysis.generator_economics`).

### `fixed_commitment`
An optional `(G, T)` 0/1 array that pins every `u[g,t]` via `lb=ub=value`,
turning the MILP into effectively an LP over dispatch alone. This is how
**realized cost** is computed throughout the project: solve once under a
forecast to get a commitment schedule, then re-solve with that commitment
fixed and *actual* demand/renewable plugged in. Comparing the "planned"
cost of each forecast scenario directly would be misleading — an
optimistic forecast lets the solver under-commit capacity and look cheap on
paper. Only the realized (settled-against-reality) cost is a fair
comparison, and it's what exposes the project's central finding: the
median (P50) forecast's realized cost is roughly **6× the P10 (conservative)
forecast's**, despite P10 not even being the most accurate forecast
available (persistence is) — because P50's error is a *one-directional*
overestimate (bias = MAE exactly), which is precisely the direction that
causes unserved demand.

---

## 5. Codeflow: a single-scenario run

This is what `pipeline.py` and the main `app.py` do, end to end:

```
generate_hourly_dataset()
        │
        ├─► history (all but last 24h), next_day (last 24h, held out with known actuals)
        │
        ▼
fit_forecast_and_history() × {wind_cf, solar_cf}   (and demand_mw, for §6)
        │
        ▼
chosen_renewable = wind_pXX + solar_pXX   (XX = 10/50/90, whichever quantile is selected)
        │
        ▼
UnitCommitmentModel(fleet, battery).build_and_solve(demand, chosen_renewable)
        │                                    ──► "planned" schedule (commitment + dispatch)
        ▼
fix commitment from "planned", re-solve against ACTUAL renewable
        │                                    ──► "realized" schedule (true settlement cost)
        ▼
analysis.py: residual load / commitment Gantt / ramp headroom / battery SoC / (optionally) economics
```

`scenario.py`'s `run_pipeline()` is exactly this sequence, factored out so
both `app.py` and any `pages/` script can build an identical scenario
without re-running `app.py`'s top-level Streamlit sidebar code (which would
happen if a page tried to `import app`).

---

## 6. Joint scenario probabilities (`scenarios.py`)

Built to replace a naive independence assumption when combining demand and
renewable uncertainty into a discrete scenario set.

1. **`paired_errors_from_predictions`**: takes in-sample P50 predictions for
   demand, wind, and solar (from `fit_forecast_and_history`, already
   computed once for the next-day forecast — no redundant refit), computes
   `error = actual − P50` for demand and for combined renewable
   (`wind_error + solar_error`, which is exact regardless of correlation
   between wind and solar specifically — error of a sum is always the sum
   of errors). This is **in-sample**, not a true walk-forward backtest — a
   documented, deliberate cost/accuracy tradeoff: the error *magnitudes*
   are optimistic, but the *co-movement* between demand and renewable
   errors at the same hour is preserved correctly, which is the only
   property this step actually needs.
2. **`tercile_labels`**: an equal-*count* split into low/mid/high at the
   1/3 and 2/3 quantiles of the error distribution — standard, unlike an
   equal-error-*mass* split (which doesn't correspond to a fixed
   probability and is sensitive to outliers).
3. **`joint_scenario_probabilities`**: `pd.crosstab` of the two tercile
   labelings, normalized — the empirical joint frequency of each
   (demand-level, renewable-level) pair, directly from the data.
   **`independence_baseline_probabilities`** computes the alternative
   (outer product of the two marginals) from the same data, purely so the
   two can be compared. On this project's data the difference is modest
   (correlation ≈ −0.15) but real, and lands where it matters: the
   high-demand/low-renewable cell is empirically ~12% more likely than
   independence would assume.
4. **`build_nine_scenarios`**: maps `low/mid/high → P10/P50/P90` for both
   variables and pairs each of the 9 combinations with its joint
   probability, producing the `[{probability, demand, renewable, ...}, ...]`
   list that `StochasticUnitCommitmentModel` consumes directly.

---

## 7. The stochastic MILP (`StochasticUnitCommitmentModel`)

A genuine **two-stage stochastic program**, not 9 independent solves
averaged after the fact (which wouldn't be meaningful — you cannot
fractionally blend several different discrete commitment schedules into
one implementable plan).

### What's shared vs. scenario-specific
| | Shared (1st stage) | Per-scenario (2nd stage / recourse) |
|---|---|---|
| Variables | `u[g,t]`, `s[g,t]`, `v[g,t]` | `p[g,t,w]`, `charge[t,w]`, `discharge[t,w]`, `soc[t,w]`, `curt[t,w]`, `unserved[t,w]` |
| Why | Commitment must be decided before any scenario is known (startup lead time) | Dispatch/battery/curtailment can adjust in real time once the scenario resolves |

Index functions extend the single-scenario pattern with a scenario axis:
`ip(g,t,w) = off_p + (g·T + t)·W + w`, etc. Constraints mirror §4's, each
now looped over `w`, with two exceptions that stay first-stage-only (no `w`
dependence) because they're properties of the commitment decision itself,
not of any one scenario: **startup/shutdown linking** and **min-up/min-down
time**.

### Objective
`Σ_g,t startup_cost[g]·s[g,t]` (shared, paid once, unweighted) `+
Σ_w probability[w] · (Σ_g,t marginal_cost[g]·p[g,t,w] + unserved_penalty·Σ_t unserved[t,w])`
(expected recourse cost). **Correctness check performed**: with exactly one
scenario (`probability=1`), this reduces to `UnitCommitmentModel`'s result
exactly — confirms the two-stage formulation is a strict generalization,
not a different model that happens to look similar.

### Result
`StochasticUCResult` carries the shared `commitment` DataFrame once, plus a
list of `ScenarioOutcome` (one per scenario) each with its own dispatch,
battery trajectory, curtailment, unserved energy, and
`cost = shared_startup_cost + this_scenario's_recourse_cost` — "what it
would actually cost if this particular scenario happens, given the
commitment we already locked in."

### Why bother — the actual measured effect
Committing based on the P50 scenario alone and then having a bad
(high-demand/low-renewable) scenario materialize: **187 MW unserved
demand, $8.3M cost**. The same bad scenario, same fleet, but with the
stochastic hedge's commitment instead: **0 MW unserved, $461K**. The hedge
costs a little more in expectation across all 9 scenarios (an "insurance
premium" — slightly more thermal capacity committed than P50 alone would
choose) in exchange for eliminating the worst-case failure mode.

---

## 8. Market economics layer (`analysis.py` + page 2)

Deliberately kept **separate** from the cost-minimization objective (except
for the optional battery arbitrage term):

- **`generator_economics(dispatch, fleet, price)`**: revenue (`power × price`)
  minus fuel/var-O&M cost minus startup cost, per generator — a reporting
  lens applied *after* solving. The marginal (most expensive committed)
  unit should show ~zero margin and cheaper units a healthy one — standard
  merit-order economics, useful as a sanity check that the price series
  behaves sensibly.
- **`battery_economics(battery, price)`**: per-hour charge cost / discharge
  revenue / net P&L / cumulative P&L.
- **Battery arbitrage** (§4's optional `price` argument): a controlled
  comparison in page 2 solves the *same* demand/renewable/fleet/battery
  twice — with and without the price term — and reports true production
  cost (recomputed from dispatch, not the solver's raw `total_cost`, which
  mixes in the arbitrage term) alongside battery P&L. Measured effect on
  one run: **+13% battery P&L for +0.03% production cost** — a small,
  well-contained shift, not a wholesale change in behavior.

---

## 9. Analysis & reporting utilities (`analysis.py`)

Everything here is plain functions operating on the dataclasses
`UnitCommitmentModel` and `StochasticUnitCommitmentModel` return —
no Streamlit dependency, so each is independently testable (and was,
standalone, before being wired into any page):

- **`residual_load` / `thermal_and_battery_coverage`**: the project's
  central framing — demand minus renewables is what thermal + battery must
  jointly cover.
- **`commitment_matrix` / `plot_commitment_gantt`**: on/off blocks per
  generator, derived from `u`'s transitions directly (not the `shutdown`
  variable — see §4's caveat).
- **`ramp_headroom`**: flags hours where a generator is within 95% of its
  ramp limit — i.e. the constraint is actually binding, not merely present.
- **`plot_battery_soc`**: state-of-charge trajectory with min/max bounds
  drawn in, plus charge/discharge bars.
- **Sensitivity sweeps** (`run_sweep`, `sweep_fuel_price`,
  `sweep_battery_param`): a generic engine — `apply_fn(fleet, battery, v) →
  (fleet, battery)` is the only thing that varies between a fuel-price
  sweep and a battery-size sweep; everything else (solving, metric
  extraction) is shared. Each solve is ~0.25s, so an 8–10 point sweep costs
  a couple of seconds — cheap enough to run interactively.

---

## 10. Application layer

- **`app.py`**: the main dashboard. Sidebar sliders (`key=`'d explicitly so
  other pages can read `st.session_state["coal_price"]` etc. as their
  baseline) drive `scenario.run_pipeline()`, then render the demand/forecast
  chart, residual load, thermal fleet table, commitment Gantt, dispatch
  stack, ramp detail, and battery SoC.
- **Page 1 (Sensitivity Analysis)**: reads the main page's slider values as
  sweep baselines (via `st.session_state`, falling back to defaults if the
  main page hasn't been visited this session), runs 4 sweeps, plots cost +
  generation-mix (fuel price sweeps) or cost + throughput (battery sweeps).
- **Page 2 (Market & Battery Arbitrage)**: two independent sections —
  settlement reporting on the already-computed *realized* dispatch (zero
  effect on dispatch decisions), and a controlled arbitrage-on/off
  comparison (does change dispatch, isolated from the forecast-uncertainty
  story).
- **Page 3 (Stochastic Unit Commitment)**: builds the 9 scenarios (§6),
  solves the two-stage MILP (§7), and runs the P50-only-vs-hedge comparison.
  Caches the expensive step (`load_and_forecast`, ~18s — three quantile
  models × three targets, fit once each) keyed on `n_days_history`.

All three pages follow the same fallback pattern: if `st.session_state`
lacks a prior scenario, build one from `scenario.DEFAULTS` rather than
erroring, so each page also works as a standalone entry point.

---

## 11. Known simplifications (consolidated)

Scattered through the code as inline comments; collected here for a single
reference:

1. **`v` (shutdown indicator) is loosely constrained** on already-off
   hours — fine for MILP correctness, not meaningful for direct display.
2. **Battery has no charge/discharge mutual-exclusivity constraint** —
   relies on cost structure to discourage it.
3. **Fleet capacities are hand-tuned** for the peaker to have a visible,
   occasional role — not derived from a real capacity-planning study.
4. **Wind+solar quantiles are combined by simple addition**
   (`wind_p10 + solar_p10 = "renewable_p10"`), which carries its own
   implicit independence assumption between wind and solar specifically —
   a separate, pre-existing simplification not addressed by §6's fix
   (which is about demand-vs-renewable, not wind-vs-solar).
5. **In-sample (not walk-forward) historical errors** drive the joint
   probability estimate (§6) — optimistic error magnitudes, correct
   co-movement.
6. **The simulated price (§2) is an exogenous merit-order proxy**, not the
   literal shadow price of any UC solve — deliberate, so the battery has a
   genuine external signal rather than one circularly derived from its own
   schedule.
7. **No transmission/network constraints, no reserve margin requirements**
   — single-bus, single-day-ahead-horizon model throughout.