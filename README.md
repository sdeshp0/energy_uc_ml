# ucsim: unit commitment under configurable scenarios and time frames

A small simulator for thermal unit commitment with renewables and battery
storage. You describe a system and a time frame in a YAML file, pick how
commitment decisions are made, and the simulator walks forward through
time: plan, lock in, settle against what actually happened, repeat.

- **One model.** A Pyomo MILP (solved with HiGHS) covers deterministic and
  two-stage stochastic commitment; a deterministic solve is one scenario
  with probability 1.
- **One engine for every time frame.** A single rolling-horizon loop
  handles a one-day day-ahead study, two weeks of daily re-commitment, a
  48-hour look-ahead, or 15-minute intraday re-commitment every few hours.
- **Scenarios as config.** Fleet, fuel prices, battery, reserves, penalties,
  time resolution, horizon, re-commit interval, date window, forecast
  settings and strategies all live in one file.

## Quickstart

```bash
uv sync                                        # or: pip install -e .
uv run streamlit run app/Run_Scenario.py       # interactive app
uv run python -m ucsim scenarios/base.yaml     # command line comparison
uv run pytest                                  # tests
```

Command-line overrides use dotted keys, and `--out` writes CSVs:

```bash
python -m ucsim scenarios/base.yaml --set time.sim_days=3 fuel_prices.gas=9 --strategies p50 stochastic
python -m ucsim scenarios/intraday_15min.yaml --out outputs/intraday
```

## How a simulation runs

For each decision point, every `commit_hr` hours starting at `sim_start`:

1. Build the planning input for the strategy over the next `horizon_hr`.
2. Solve unit commitment from the current system state (which units are on
   and for how long, their output, battery charge).
3. Lock in the first `commit_hr` of commitment. Settle those hours against
   actual demand and renewables: dispatch is re-optimized, commitment is
   fixed except for `fast_start` units, which can be called in real time.
4. Carry the realized state forward.

Costs are always reported from the settlement step, so a strategy that
plans optimistically can't look cheap on paper.

## Strategies

| Strategy | Plans on |
|---|---|
| `perfect` | Actual demand and renewables (lower bound) |
| `persistence` | Values from the most recent day already known at decision time |
| `p50` | Median forecast |
| `conservative` | Demand P90 with renewable P10 |
| `stochastic` | `n_scenarios` sampled trajectories, one shared commitment |

Forecasts are gradient-boosted quantile models of demand and total
renewable output, trained once on `train_days` of history before
`sim_start`. Lag features only use values that are already known when the
decision is made, so a 48-hour horizon uses lags of at least 2 days.

Stochastic scenarios are the P50 forecast plus forecast-error paths copied
from a held-out calibration window (the last `calib_frac` of the training
period, so errors are measured out of sample). Each path is a contiguous
stretch starting at the same time of day, with demand and renewable errors
taken together, which keeps autocorrelation and the cold-snap correlation
between high demand and low wind.

## Model

Per generator: on/off, startup and shutdown binaries; min and max output;
minimum up and down time, including time already spent in the current
state when the solve starts; ramp limits with startup/shutdown allowances
(a unit can jump from 0 to `max(pmin, ramp)` when it starts); spinning
reserve capped by what the unit can ramp in `spin_response_min`.

System: power balance with curtailment, unserved energy and over-generation
slacks; battery state of charge with round-trip losses and a per-MWh
cycling cost; optional total and spinning reserve as a share of net load,
soft-penalized; a soft end-of-horizon state-of-charge target so the
battery isn't drained just because a solve ends.

Weather is synthetic: seasonal demand with two daily peaks, AR(1) noise,
clear-sky solar with clouds, mean-reverting wind, and occasional cold-snap
days (+16% demand, wind drought). Each series has its own seeded stream
from a fixed epoch, so a given timestamp always has the same weather no
matter the window or training length.

## Scenario presets

| File | What it shows |
|---|---|
| `base.yaml` | Two winter weeks, hourly, day-ahead. Every field is listed with its default. |
| `rigid_fleet.yaml` | Same system with no fast-start units, so the day-ahead plan has to cover forecast misses. |
| `tight_system.yaml` | The original 800 MW fleet against a ~1,100 MW peak. |
| `gas_price_spike.yaml` | Gas at $10/MMBtu, which reorders the merit order. |
| `reserves.yaml` | 15% total and 5% spinning reserve. |
| `summer_week.yaml` | One summer week, strong solar. |
| `lookahead_48h.yaml` | Solve 48 hours ahead, lock in 24. |
| `intraday_15min.yaml` | 15-minute steps, re-committed every 6 hours. |

A preset lists only what differs from `base.yaml`. Unknown keys are
rejected, so typos fail loudly.

## App

- **Run Scenario**: pick a preset, adjust it in the sidebar (time frame,
  prices, battery, reserves, fleet table, seed), choose a strategy, and see
  plan vs. actual, the realized dispatch stack, commitment, battery state
  of charge and a per-day table.
- **Compare**: run several strategies on the same data, or sweep one
  parameter (fuel price, battery size, reserve level, look-ahead, value of
  lost load) for one strategy.

## Results

Four winter weeks from 2024-01-08 (seed 42), hourly, day-ahead commitment
re-solved daily, the first day being a cold snap. Reproduce with
`python -m ucsim scenarios/<preset>.yaml --set time.sim_days=28`.

| Strategy | base: total cost | base: unserved MWh | rigid_fleet: total cost | rigid_fleet: unserved MWh | tight_system: total cost | tight_system: unserved MWh |
|---|---:|---:|---:|---:|---:|---:|
| Perfect foresight | $9.82M | 0 | $9.82M | 0 | $18.38M | 1,715 |
| Persistence | $14.95M | 965 | $80.05M | 14,138 | $44.14M | 6,857 |
| Forecast P50 | $9.98M | 0 | $38.52M | 5,780 | $24.73M | 2,972 |
| Conservative P90/P10 | $9.99M | 0 | $28.78M | 3,802 | $18.51M | 1,715 |
| Stochastic (10 scenarios) | $9.95M | 0 | $25.65M | 3,163 | $18.60M | 1,733 |

What this shows:

- **When peakers can start in real time (base), the commitment strategy
  barely matters.** Every forecast-based strategy is within 2% of perfect
  foresight; only persistence, which misses the cold snap, sheds load.
- **When the day-ahead plan has to cover everything (rigid_fleet), it
  matters a lot.** The stochastic plan cuts unserved energy by 45% versus
  P50 and total cost by a third. The conservative plan gets most of the way
  there with far fewer starts, but commits more capacity on every day.
- **In a short system (tight_system), unserved energy is set by the
  weather,** and the best a plan can do is not add to it: conservative and
  stochastic land at the perfect-foresight floor, P50 adds 1,257 MWh.
- Forecast accuracy and decision quality are different things. Over this
  window the P50 demand forecast is much better than persistence (MAE 37 vs
  55 MW) while the P50 renewable forecast is slightly worse (78 vs 73 MW),
  yet P50 commitment beats persistence everywhere because persistence
  misses the cold snap entirely.

Penalties use a value of lost load of $5,000/MWh, so total cost is very
sensitive to it; the Compare page can sweep it.

## Layout

```
src/ucsim/
  config.py       Scenario dataclasses, YAML loading, presets
  data.py         synthetic demand / wind / solar at any resolution
  forecast.py     quantile forecasts and persistence
  uncertainty.py  out-of-sample error pool and scenario sampling
  model.py        the UC MILP (Pyomo + HiGHS)
  simulate.py     rolling-horizon engine, costs, summaries
  __main__.py     command line
app/              Streamlit app (Run_Scenario.py, pages/2_Compare.py)
scenarios/        YAML presets
tests/
```

## Limitations

- Single bus, no transmission.
- Fast-start units are planned as first-stage decisions in the stochastic
  model but are free in settlement, so the hedge is slightly conservative.
- Forecast models are trained once per run, not retrained as the
  simulation walks forward.
- Data is synthetic. `data.generate` returns a plain time-indexed frame
  (`demand_mw`, `wind_mw`, `solar_mw`, `renewable_mw`), so real series
  (EIA, ENTSO-E, renewables.ninja) can replace it.
