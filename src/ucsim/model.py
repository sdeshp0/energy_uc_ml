"""Unit commitment as a (two-stage stochastic) MILP in Pyomo, solved with HiGHS.

One model covers every case: a deterministic solve is a single scenario
with probability 1. Commitment (u, startup s, shutdown v) is shared across
scenarios; dispatch, storage, curtailment and slacks are per scenario.

Per generator g, step t, scenario w (dt = step length in hours):
    u[g,t] - u[g,t-1] = s[g,t] - v[g,t]
    sum(s[g, t-UT+1 .. t]) <= u[g,t]          minimum up time
    sum(v[g, t-DT+1 .. t]) <= 1 - u[g,t]      minimum down time
    pmin*u <= p,  p + spin <= pmax*u,  spin <= ramp*response*u
    p[t] - p[t-1] <= ramp*dt*u[t-1] + SU*s[t]     SU = max(pmin, ramp*dt)
    p[t-1] - p[t] <= ramp*dt*u[t]   + SD*v[t]     SD = max(pmin, ramp*dt)
    sum_g p + renewable - curtail + discharge - charge + unserved - overgen = demand
    soc[t] = soc[t-1] + eta*charge*dt - discharge*dt/eta     eta = sqrt(round trip)
    sum_g (pmax*u - p) + short_total >= total_frac * net_load
    sum_g spin + short_spin >= spin_frac * net_load
Initial conditions (status, output, time in state, state of charge) come
from the previous solve, so min up/down and ramp limits hold across solves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from .config import Scenario


@dataclass
class State:
    """System state entering a solve."""
    u: np.ndarray            # (G,) on/off
    p: np.ndarray            # (G,) MW output in the previous step
    time_in_state: np.ndarray  # (G,) hours each unit has been in its current on/off state
    soc_mwh: float

    @classmethod
    def initial(cls, scenario: Scenario) -> State:
        G = len(scenario.fleet)
        b = scenario.battery
        return cls(np.zeros(G), np.zeros(G), np.full(G, 1e6), b.soc_init_frac * b.energy_mwh)


@dataclass
class Solution:
    u: np.ndarray            # (G,T)
    s: np.ndarray            # (G,T)
    p: np.ndarray            # (W,G,T)
    charge: np.ndarray       # (W,T)
    discharge: np.ndarray
    soc: np.ndarray
    curtail: np.ndarray
    unserved: np.ndarray
    overgen: np.ndarray
    short_total: np.ndarray
    short_spin: np.ndarray
    objective: float


def solve(scenario: Scenario, scenarios: list[dict], state: State,
          fixed_u: np.ndarray | None = None, soc_target: float | None = None,
          mip_gap: float = 0.005, time_limit: float = 60.0) -> Solution:
    """scenarios: list of {"prob", "demand", "renewable"} with equal-length arrays.

    fixed_u pins the commitment (settlement against actuals); fast-start
    units stay free even then, since they can be called on in real time. soc_target is a
    soft floor on the final state of charge, valued at the most expensive
    unit's marginal cost so the battery isn't drained just because the
    horizon ends.
    """
    fleet = scenario.fleet_frame()
    bat, res, pen = scenario.battery, scenario.reserves, scenario.penalties
    dt = scenario.time.dt
    G, W, T = len(fleet), len(scenarios), len(scenarios[0]["demand"])
    eta = math.sqrt(bat.round_trip_eff)
    pmin, pmax = fleet["pmin_mw"].to_numpy(), fleet["pmax_mw"].to_numpy()
    ramp = fleet["ramp_mw_per_hr"].to_numpy() * dt          # MW per step
    su = np.maximum(pmin, ramp)
    spin_cap = fleet["ramp_mw_per_hr"].to_numpy() * res.spin_response_min / 60
    ut = np.maximum(1, np.ceil(fleet["min_up_hr"].to_numpy() / dt)).astype(int)
    dn = np.maximum(1, np.ceil(fleet["min_down_hr"].to_numpy() / dt)).astype(int)
    mc = fleet["marginal_cost"].to_numpy()
    prob = np.array([s["prob"] for s in scenarios])
    demand = np.array([s["demand"] for s in scenarios])
    renew = np.array([s["renewable"] for s in scenarios])
    net = np.clip(demand - renew, 0, None)

    m = pyo.ConcreteModel()
    m.G, m.T, m.W = pyo.RangeSet(0, G - 1), pyo.RangeSet(0, T - 1), pyo.RangeSet(0, W - 1)
    m.u = pyo.Var(m.G, m.T, domain=pyo.Binary)
    m.s = pyo.Var(m.G, m.T, domain=pyo.Binary)
    m.v = pyo.Var(m.G, m.T, domain=pyo.Binary)
    nn = pyo.NonNegativeReals
    m.p = pyo.Var(m.W, m.G, m.T, domain=nn)
    m.spin = pyo.Var(m.W, m.G, m.T, domain=nn)
    m.ch = pyo.Var(m.W, m.T, bounds=(0, bat.power_mw))
    m.dis = pyo.Var(m.W, m.T, bounds=(0, bat.power_mw))
    m.soc = pyo.Var(m.W, m.T, bounds=(bat.soc_min_frac * bat.energy_mwh, bat.soc_max_frac * bat.energy_mwh))
    m.curt = pyo.Var(m.W, m.T, domain=nn)
    m.uns = pyo.Var(m.W, m.T, domain=nn)
    m.over = pyo.Var(m.W, m.T, domain=nn)
    m.sh_tot = pyo.Var(m.W, m.T, domain=nn)
    m.sh_spin = pyo.Var(m.W, m.T, domain=nn)
    m.soc_short = pyo.Var(m.W, domain=nn)

    def u_prev(g, t):
        return m.u[g, t - 1] if t > 0 else float(state.u[g])

    def p_prev(w, g, t):
        return m.p[w, g, t - 1] if t > 0 else float(state.p[g])

    # ---- commitment logic (first stage) ----
    m.logic = pyo.Constraint(m.G, m.T, rule=lambda m, g, t: m.u[g, t] - u_prev(g, t) == m.s[g, t] - m.v[g, t])
    m.min_up = pyo.Constraint(m.G, m.T, rule=lambda m, g, t:
                              sum(m.s[g, i] for i in range(max(0, t - ut[g] + 1), t + 1)) <= m.u[g, t])
    m.min_dn = pyo.Constraint(m.G, m.T, rule=lambda m, g, t:
                              sum(m.v[g, i] for i in range(max(0, t - dn[g] + 1), t + 1)) <= 1 - m.u[g, t])
    for g in range(G):  # carry-over of min up/down from before the horizon
        steps_in_state = int(state.time_in_state[g] / dt + 1e-9)
        need = (ut[g] if state.u[g] else dn[g]) - steps_in_state
        for t in range(min(max(need, 0), T)):
            m.u[g, t].fix(int(state.u[g]))
    if fixed_u is not None:
        for g in range(G):
            if fleet.at[g, "fast_start"]:
                continue
            for t in range(T):
                m.u[g, t].fix(round(fixed_u[g, t]))

    # ---- generator limits and ramps (per scenario) ----
    m.p_lo = pyo.Constraint(m.W, m.G, m.T, rule=lambda m, w, g, t: m.p[w, g, t] >= pmin[g] * m.u[g, t])
    m.p_hi = pyo.Constraint(m.W, m.G, m.T, rule=lambda m, w, g, t:
                            m.p[w, g, t] + m.spin[w, g, t] <= pmax[g] * m.u[g, t])
    m.spin_cap = pyo.Constraint(m.W, m.G, m.T, rule=lambda m, w, g, t: m.spin[w, g, t] <= spin_cap[g] * m.u[g, t])
    m.ramp_up = pyo.Constraint(m.W, m.G, m.T, rule=lambda m, w, g, t:
                               m.p[w, g, t] - p_prev(w, g, t) <= ramp[g] * u_prev(g, t) + su[g] * m.s[g, t])
    m.ramp_dn = pyo.Constraint(m.W, m.G, m.T, rule=lambda m, w, g, t:
                               p_prev(w, g, t) - m.p[w, g, t] <= ramp[g] * m.u[g, t] + su[g] * m.v[g, t])

    # ---- balance, curtailment, storage, reserves ----
    m.balance = pyo.Constraint(m.W, m.T, rule=lambda m, w, t:
                               sum(m.p[w, g, t] for g in m.G) + float(renew[w, t]) - m.curt[w, t]
                               + m.dis[w, t] - m.ch[w, t] + m.uns[w, t] - m.over[w, t] == float(demand[w, t]))
    m.curt_cap = pyo.Constraint(m.W, m.T, rule=lambda m, w, t: m.curt[w, t] <= float(renew[w, t]))
    m.soc_dyn = pyo.Constraint(m.W, m.T, rule=lambda m, w, t:
                               m.soc[w, t] == (m.soc[w, t - 1] if t > 0 else state.soc_mwh)
                               + eta * m.ch[w, t] * dt - m.dis[w, t] * dt / eta)
    if soc_target is not None:
        m.soc_end = pyo.Constraint(m.W, rule=lambda m, w: m.soc[w, T - 1] + m.soc_short[w] >= soc_target)
    if res.total_frac > 0:
        m.res_tot = pyo.Constraint(m.W, m.T, rule=lambda m, w, t:
                                   sum(pmax[g] * m.u[g, t] - m.p[w, g, t] for g in m.G) + m.sh_tot[w, t]
                                   >= res.total_frac * float(net[w, t]))
    if res.spin_frac > 0:
        m.res_spin = pyo.Constraint(m.W, m.T, rule=lambda m, w, t:
                                    sum(m.spin[w, g, t] for g in m.G) + m.sh_spin[w, t]
                                    >= res.spin_frac * float(net[w, t]))

    # ---- objective: startups + expected operating cost ----
    soc_value = mc.max() / eta
    startup = sum(float(fleet.at[g, "startup_cost"]) * m.s[g, t] for g in m.G for t in m.T)
    op = sum(
        float(prob[w]) * (
            dt * sum(float(mc[g]) * m.p[w, g, t] for g in m.G for t in m.T)
            + dt * sum(pen.voll * m.uns[w, t] + pen.overgen * m.over[w, t]
                       + pen.reserve_shortfall * (m.sh_tot[w, t] + m.sh_spin[w, t])
                       + bat.cycle_cost * (m.ch[w, t] + m.dis[w, t]) for t in m.T)
            + soc_value * m.soc_short[w]
        )
        for w in m.W
    )
    m.obj = pyo.Objective(expr=startup + op, sense=pyo.minimize)

    solver = pyo.SolverFactory("appsi_highs")
    solver.config.mip_gap = mip_gap
    solver.config.time_limit = time_limit
    result = solver.solve(m, load_solutions=False)
    tc = result.solver.termination_condition
    if tc not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.maxTimeLimit):
        raise RuntimeError(f"UC solve failed: {tc}")
    m.solutions.load_from(result)

    def arr(var, *shape):
        return np.array([pyo.value(var[k]) for k in var.index_set()]).reshape(shape)

    return Solution(
        u=np.round(arr(m.u, G, T)), s=np.round(arr(m.s, G, T)), p=arr(m.p, W, G, T),
        charge=arr(m.ch, W, T), discharge=arr(m.dis, W, T), soc=arr(m.soc, W, T),
        curtail=arr(m.curt, W, T), unserved=arr(m.uns, W, T), overgen=arr(m.over, W, T),
        short_total=arr(m.sh_tot, W, T), short_spin=arr(m.sh_spin, W, T),
        objective=pyo.value(m.obj),
    )


def to_frame(sol: Solution, scenario: Scenario, index: pd.DatetimeIndex, w: int = 0) -> pd.DataFrame:
    """One scenario's dispatch as a tidy time-indexed frame."""
    names = [g.name for g in scenario.fleet]
    df = pd.DataFrame(index=index)
    for i, n in enumerate(names):
        df[f"p_{n}"] = sol.p[w, i]
        df[f"u_{n}"] = sol.u[i]
        df[f"start_{n}"] = sol.s[i]
    for col in ("charge", "discharge", "soc", "curtail", "unserved", "overgen", "short_total", "short_spin"):
        df[col] = getattr(sol, col)[w]
    return df
