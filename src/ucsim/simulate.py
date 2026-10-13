"""Rolling-horizon simulation: the single engine behind every time frame.

At each decision point (every `commit_hr`):
  1. build the planning input for the strategy over the next `horizon_hr`
     (actuals, persistence, a forecast quantile, or sampled scenarios),
  2. solve unit commitment from the current system state,
  3. lock in the first `commit_hr` of commitment and settle it against
     actual demand and renewables (dispatch re-optimized, commitment fixed),
  4. carry the realized state (on/off, output, time in state, battery
     charge) into the next decision.

A one-day day-ahead study is the special case horizon = commit = 24h,
sim_days = 1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data, forecast, uncertainty
from .config import Scenario
from .model import State, solve, to_frame

STRATEGY_LABELS = {
    "perfect": "Perfect foresight",
    "persistence": "Persistence",
    "p50": "Forecast P50",
    "conservative": "Conservative P90/P10",
    "stochastic": "Stochastic",
}


@dataclass
class Prepared:
    """Data and forecasts shared by every strategy of one scenario."""
    df: pd.DataFrame
    forecasts: dict[str, pd.DataFrame]
    error_pool: pd.DataFrame
    sim_index: pd.DatetimeIndex


@dataclass
class SimResult:
    scenario: Scenario
    strategy: str
    timeseries: pd.DataFrame
    summary: dict = field(default_factory=dict)

    def daily(self) -> pd.DataFrame:
        ts = self.timeseries
        dt = self.scenario.time.dt
        g = ts.groupby(ts.index.normalize())
        return pd.DataFrame({
            "total_cost": g["total_cost"].sum(),
            "unserved_mwh": g["unserved"].sum() * dt,
            "curtailed_mwh": g["curtail"].sum() * dt,
            "starts": g[[c for c in ts if c.startswith("start_")]].sum().sum(axis=1),
        })


def prepare(scenario: Scenario) -> Prepared:
    t, d = scenario.time, scenario.data
    sim_start = pd.Timestamp(t.sim_start)
    sim_end = sim_start + pd.Timedelta(days=t.sim_days)
    train_start = sim_start - pd.Timedelta(days=d.train_days)
    calib_days = max(1, round(d.train_days * scenario.uncertainty.calib_frac))
    fit_end = sim_start - pd.Timedelta(days=calib_days)
    warmup = pd.Timedelta(days=forecast.lead_days(t.horizon_hr) + 9)
    if train_start - warmup < data.EPOCH:
        raise ValueError(f"sim_start - train_days must be after {(data.EPOCH + warmup).date()}")

    df = data.generate(d, t.resolution_min, end=sim_end + pd.Timedelta(hours=t.horizon_hr))
    df = df[df.index >= train_start - warmup]
    caps = {"demand_mw": None, "renewable_mw": d.wind_mw + d.solar_mw}
    fc = forecast.fit_predict(df, train_start, fit_end, t.steps_per_day, t.horizon_hr, caps)
    pool = uncertainty.error_pool(fc, fit_end, sim_start)
    sim_index = df.index[(df.index >= sim_start) & (df.index < sim_end)]
    return Prepared(df, fc, pool, sim_index)


def _planning_scenarios(strategy: str, scenario: Scenario, prep: Prepared,
                        window: pd.DatetimeIndex, step_no: int) -> list[dict]:
    dem, ren = prep.forecasts["demand_mw"].loc[window], prep.forecasts["renewable_mw"].loc[window]
    one = lambda d, r: [{"prob": 1.0, "demand": d.to_numpy(), "renewable": r.to_numpy()}]
    if strategy == "perfect":
        return one(dem["actual"], ren["actual"])
    if strategy == "persistence":
        return one(dem["persistence"], ren["persistence"])
    if strategy == "p50":
        return one(dem["p50"], ren["p50"])
    if strategy == "conservative":
        return one(dem["p90"], ren["p10"])
    if strategy == "stochastic":
        rng = np.random.default_rng([scenario.data.seed, step_no])
        return uncertainty.sample(
            prep.error_pool, dem["p50"].to_numpy(), ren["p50"].to_numpy(), window[0],
            scenario.uncertainty.n_scenarios, scenario.time.steps_per_day,
            scenario.data.wind_mw + scenario.data.solar_mw, rng)
    raise ValueError(strategy)


def run(scenario: Scenario, strategy: str, prepared: Prepared | None = None) -> SimResult:
    prep = prepared or prepare(scenario)
    t, b = scenario.time, scenario.battery
    fleet = scenario.fleet_frame()
    names = list(fleet["name"])
    H, S = t.horizon_steps, t.commit_steps
    all_idx = prep.df.index
    first = all_idx.get_loc(prep.sim_index[0])
    n_sim = len(prep.sim_index)

    state = State.initial(scenario)
    soc_goal = b.soc_init_frac * b.energy_mwh
    frames, started = [], time.perf_counter()
    for step_no, offset in enumerate(range(0, n_sim, S)):
        i0 = first + offset
        window = all_idx[i0:i0 + H]
        commit = min(S, n_sim - offset)
        plan_in = _planning_scenarios(strategy, scenario, prep, window, step_no)
        plan = solve(scenario, plan_in, state, soc_target=soc_goal)

        actual_idx = window[:commit]
        actual = [{"prob": 1.0,
                   "demand": prep.df.loc[actual_idx, "demand_mw"].to_numpy(),
                   "renewable": prep.df.loc[actual_idx, "renewable_mw"].to_numpy()}]
        probs = np.array([s["prob"] for s in plan_in])
        planned_soc = float(probs @ plan.soc[:, commit - 1])
        settled = solve(scenario, actual, state, fixed_u=plan.u[:, :commit], soc_target=planned_soc)

        frame = to_frame(settled, scenario, actual_idx)
        frame["demand"] = actual[0]["demand"]
        frame["renewable"] = actual[0]["renewable"]
        frame["plan_demand"] = probs @ np.array([s["demand"][:commit] for s in plan_in])
        frame["plan_renewable"] = probs @ np.array([s["renewable"][:commit] for s in plan_in])
        frames.append(frame)
        state = _advance(state, settled, commit, t.dt)

    ts = pd.concat(frames)
    _add_costs(ts, scenario, fleet)
    res = SimResult(scenario, strategy, ts)
    res.summary = _summarize(ts, scenario, names, time.perf_counter() - started)
    return res


def _advance(state: State, sol, commit: int, dt: float) -> State:
    u_now, tis = state.u.copy(), state.time_in_state.copy()
    for k in range(commit):
        same = sol.u[:, k] == u_now
        tis = np.where(same, tis + dt, dt)
        u_now = sol.u[:, k].copy()
    return State(u_now, sol.p[0, :, commit - 1].copy(), tis, float(sol.soc[0, commit - 1]))


def _add_costs(ts: pd.DataFrame, scenario: Scenario, fleet: pd.DataFrame) -> None:
    dt, pen, b = scenario.time.dt, scenario.penalties, scenario.battery
    ts["fuel_cost"] = dt * sum(ts[f"p_{n}"] * mc for n, mc in zip(fleet["name"], fleet["marginal_cost"]))
    ts["startup_cost"] = sum(ts[f"start_{n}"] * c for n, c in zip(fleet["name"], fleet["startup_cost"]))
    ts["battery_cost"] = dt * b.cycle_cost * (ts["charge"] + ts["discharge"])
    ts["penalty_cost"] = dt * (pen.voll * ts["unserved"] + pen.overgen * ts["overgen"]
                               + pen.reserve_shortfall * (ts["short_total"] + ts["short_spin"]))
    ts["total_cost"] = ts[["fuel_cost", "startup_cost", "battery_cost", "penalty_cost"]].sum(axis=1)


def _summarize(ts: pd.DataFrame, scenario: Scenario, names: list[str], seconds: float) -> dict:
    dt = scenario.time.dt
    energy = ts["demand"].sum() * dt
    op = ts[["fuel_cost", "startup_cost", "battery_cost"]].sum().sum()
    return {
        "total_cost": ts["total_cost"].sum(),
        "operating_cost": op,
        "penalty_cost": ts["penalty_cost"].sum(),
        "cost_per_mwh": ts["total_cost"].sum() / energy,
        "unserved_mwh": ts["unserved"].sum() * dt,
        "max_unserved_mw": ts["unserved"].max(),
        "hours_with_unserved": (ts["unserved"] > 0.1).sum() * dt,
        "curtailed_mwh": ts["curtail"].sum() * dt,
        "reserve_shortfall_mwh": (ts["short_total"] + ts["short_spin"]).sum() * dt,
        "starts": int(sum(ts[f"start_{n}"].sum() for n in names)),
        "runtime_s": seconds,
    }


def compare(scenario: Scenario, strategies: list[str] | None = None,
            prepared: Prepared | None = None) -> dict[str, SimResult]:
    """Run several strategies on the same data and forecasts."""
    prep = prepared or prepare(scenario)
    return {s: run(scenario, s, prep) for s in (strategies or scenario.strategies)}


def summary_table(results: dict[str, SimResult]) -> pd.DataFrame:
    return pd.DataFrame({k: r.summary for k, r in results.items()}).T
