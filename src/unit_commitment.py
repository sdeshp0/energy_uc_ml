"""
Mixed-integer unit commitment with renewables + battery storage.

Solved via scipy.optimize.milp (HiGHS backend) so the project runs with
zero extra installs. For Phase 2 (stochastic/scenario UC, rolling horizon),
migrating this formulation to Pyomo is recommended -- the constraint logic
below maps directly, but Pyomo's indexed variables make multi-scenario
models much more readable than hand-built sparse matrices.

Decision variables per generator g, hour t:
    u[g,t]  in {0,1}   commitment (on/off)
    p[g,t]  continuous power output (MW)
    s[g,t]  in {0,1}   startup indicator

Battery, per hour t:
    c[t]    continuous charge power (MW)
    d[t]    continuous discharge power (MW)
    soc[t]  continuous state of charge (MWh)

Renewables are treated as must-take up to the forecast, with a curtailment
variable so the balance constraint stays feasible when renewable supply
would otherwise exceed demand.
    curt[t] continuous curtailed renewable energy (MW), 0 <= curt <= renewable_forecast[t]

Objective: minimize sum of marginal generation cost + startup costs.
(Curtailment and battery cycling carry no direct cost here -- a natural
Phase-2 extension is to add a small curtailment penalty / battery
degradation cost to shape behavior.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds


@dataclass
class UCResult:
    status: str
    total_cost: float
    dispatch: pd.DataFrame          # per-generator, per-hour output + commitment
    battery: pd.DataFrame           # per-hour charge/discharge/soc
    curtailment: np.ndarray         # per-hour curtailed renewable MW
    unserved: np.ndarray            # per-hour unserved demand MW (should be ~0 if feasible)


class UnitCommitmentModel:
    def __init__(self, fleet: pd.DataFrame, battery: dict, T: int = 24):
        self.fleet = fleet.reset_index(drop=True)
        self.G = len(fleet)
        self.T = T
        self.battery = battery

        # Variable layout (flat vector x): blocks in this order, each length T (or G*T)
        #   u[g,t]  : G*T   binary   commitment (on/off)
        #   p[g,t]  : G*T   continuous  power output
        #   s[g,t]  : G*T   binary   startup indicator
        #   v[g,t]  : G*T   binary   shutdown indicator (enforces min-down-time)
        #   c[t]    : T     continuous (battery charge)
        #   d[t]    : T     continuous (battery discharge)
        #   soc[t]  : T     continuous (battery state of charge)
        #   curt[t] : T     continuous (curtailed renewables)
        #   unserved[t]: T  continuous (unserved demand -- big penalty, should be ~0)
        self.n_u = self.G * T
        self.n_p = self.G * T
        self.n_s = self.G * T
        self.n_v = self.G * T
        self.n_c = T
        self.n_d = T
        self.n_soc = T
        self.n_curt = T
        self.n_unserved = T

        self.off_u = 0
        self.off_p = self.off_u + self.n_u
        self.off_s = self.off_p + self.n_p
        self.off_v = self.off_s + self.n_s
        self.off_c = self.off_v + self.n_v
        self.off_d = self.off_c + self.n_c
        self.off_soc = self.off_d + self.n_d
        self.off_curt = self.off_soc + self.n_soc
        self.off_unserved = self.off_curt + self.n_curt
        self.n_vars = self.off_unserved + self.n_unserved

    # -- index helpers --
    def iu(self, g, t): return self.off_u + g * self.T + t
    def ip(self, g, t): return self.off_p + g * self.T + t
    def is_(self, g, t): return self.off_s + g * self.T + t
    def iv(self, g, t): return self.off_v + g * self.T + t
    def ic(self, t): return self.off_c + t
    def id_(self, t): return self.off_d + t
    def isoc(self, t): return self.off_soc + t
    def icurt(self, t): return self.off_curt + t
    def iunserved(self, t): return self.off_unserved + t

    def build_and_solve(self, demand: np.ndarray, renewable_mw: np.ndarray,
                         u_prev: np.ndarray | None = None,
                         unserved_penalty: float = 5000.0,
                         fixed_commitment: np.ndarray | None = None,
                         price: np.ndarray | None = None,
                         soc_init_mwh: float | None = None,
                         soc_terminal_min_mwh: float | None = None) -> UCResult:
        """
        demand: length-T array of demand (MW)
        renewable_mw: length-T array of available wind+solar (MW), pre-summed
        u_prev: length-G array, commitment state in the hour before t=0 (default: all off)
        fixed_commitment: optional (G,T) 0/1 array. When given, u[g,t] is pinned to this
            value instead of being optimized -- used to evaluate the REALIZED cost of a
            day-ahead commitment schedule once actual renewables are known, as opposed to
            the (potentially misleading) planned cost computed under the forecast.
        soc_init_mwh: optional starting battery state of charge (MWh) at t=0. Defaults to
            None, which uses battery["soc_init_frac"] * capacity as before (a fixed
            starting point every solve). Pass the previous day's ENDING soc here when
            chaining solves across days (rolling-horizon simulation) so the battery's
            state actually carries over instead of resetting each day.
        soc_terminal_min_mwh: optional floor on soc[T-1] (the LAST hour's state of
            charge). Without this, a finite-horizon solve has no incentive to end
            with any charge above the physical minimum -- stored energy has no
            explicit value in the objective, so the optimizer will drain the
            battery by the final hours whenever that's weakly cost-reducing. Fine
            for a single isolated day, but fatal for a rolling multi-day
            simulation: every day would start crippled at the floor. Pass e.g.
            battery["soc_init_frac"] * capacity here to require each day end back
            at a normal operating level for the next day to start from.
        price: optional length-T array ($/MWh). When given, adds price[t]*charge[t] as a
            cost and price[t]*discharge[t] as a revenue credit (negative cost) to the
            objective -- a battery arbitrage incentive layered ON TOP OF production-cost
            minimization, not a replacement for it. This is a deliberately blended
            objective (minimize production cost, minus battery arbitrage revenue against
            an exogenous price), not a rigorous merchant/market reformulation: the
            resulting total_cost mixes true production cost with this arbitrage term, so
            it is NOT directly comparable to a total_cost from a price=None solve. For a
            fair comparison, recompute true production cost from the dispatch (e.g. via
            analysis.generator_economics) rather than reading UCResult.total_cost when
            price is used.
        """
        T, G, fleet = self.T, self.G, self.fleet
        if u_prev is None:
            u_prev = np.zeros(G)

        # ---------- objective ----------
        c_obj = np.zeros(self.n_vars)
        for g in range(G):
            for t in range(T):
                c_obj[self.ip(g, t)] = fleet.loc[g, "marginal_cost"]
                c_obj[self.is_(g, t)] = fleet.loc[g, "startup_cost"]
        for t in range(T):
            c_obj[self.iunserved(t)] = unserved_penalty
        if price is not None:
            for t in range(T):
                c_obj[self.ic(t)] += price[t]    # cost to charge (buying energy at market price)
                c_obj[self.id_(t)] += -price[t]  # revenue credit for discharging (selling at market price)

        constraints = []

        # ---------- power balance: sum(p) + (renewable - curt) + d - c + unserved = demand ----------
        for t in range(T):
            row = np.zeros(self.n_vars)
            for g in range(G):
                row[self.ip(g, t)] = 1.0
            row[self.icurt(t)] = -1.0
            row[self.id_(t)] = 1.0
            row[self.ic(t)] = -1.0
            row[self.iunserved(t)] = 1.0
            rhs = demand[t] - renewable_mw[t]
            constraints.append(LinearConstraint(row, rhs, rhs))

        # ---------- curtailment bound: curt[t] <= renewable_mw[t] ----------
        for t in range(T):
            row = np.zeros(self.n_vars)
            row[self.icurt(t)] = 1.0
            constraints.append(LinearConstraint(row, -np.inf, max(renewable_mw[t], 0.0)))

        # ---------- generator output linked to commitment: pmin*u <= p <= pmax*u ----------
        for g in range(G):
            pmin, pmax = fleet.loc[g, "pmin_mw"], fleet.loc[g, "pmax_mw"]
            for t in range(T):
                row_hi = np.zeros(self.n_vars)
                row_hi[self.ip(g, t)] = 1.0
                row_hi[self.iu(g, t)] = -pmax
                constraints.append(LinearConstraint(row_hi, -np.inf, 0.0))

                row_lo = np.zeros(self.n_vars)
                row_lo[self.ip(g, t)] = 1.0
                row_lo[self.iu(g, t)] = -pmin
                constraints.append(LinearConstraint(row_lo, 0.0, np.inf))

        # ---------- startup linking: s[g,t] >= u[g,t] - u[g,t-1] ----------
        for g in range(G):
            for t in range(T):
                row = np.zeros(self.n_vars)
                row[self.is_(g, t)] = 1.0
                row[self.iu(g, t)] = -1.0
                if t == 0:
                    rhs_lo = -u_prev[g]  # s - u >= -u_prev  =>  s >= u - u_prev
                    constraints.append(LinearConstraint(row, rhs_lo, np.inf))
                else:
                    row[self.iu(g, t - 1)] = 1.0
                    constraints.append(LinearConstraint(row, 0.0, np.inf))

        # ---------- shutdown linking: v[g,t] >= u[g,t-1] - u[g,t] ----------
        for g in range(G):
            for t in range(T):
                row = np.zeros(self.n_vars)
                row[self.iv(g, t)] = 1.0
                row[self.iu(g, t)] = 1.0
                if t == 0:
                    rhs_lo = u_prev[g]  # v + u >= u_prev  =>  v >= u_prev - u
                    constraints.append(LinearConstraint(row, rhs_lo, np.inf))
                else:
                    row[self.iu(g, t - 1)] = -1.0
                    constraints.append(LinearConstraint(row, 0.0, np.inf))

        # ---------- minimum up-time: if started within min_up hours of t, must still be on ----------
        for g in range(G):
            min_up = int(fleet.loc[g, "min_up_hr"])
            for t in range(T):
                window = range(max(0, t - min_up + 1), t + 1)
                row = np.zeros(self.n_vars)
                for i in window:
                    row[self.is_(g, i)] = 1.0
                row[self.iu(g, t)] = -1.0
                constraints.append(LinearConstraint(row, -np.inf, 0.0))

        # ---------- minimum down-time: if shut down within min_down hours of t, must still be off ----------
        for g in range(G):
            min_down = int(fleet.loc[g, "min_down_hr"])
            for t in range(T):
                window = range(max(0, t - min_down + 1), t + 1)
                row = np.zeros(self.n_vars)
                for i in window:
                    row[self.iv(g, i)] = 1.0
                row[self.iu(g, t)] = 1.0
                constraints.append(LinearConstraint(row, -np.inf, 1.0))

        # ---------- ramp limits ----------
        for g in range(G):
            ramp = fleet.loc[g, "ramp_mw_per_hr"]
            for t in range(1, T):
                row_up = np.zeros(self.n_vars)
                row_up[self.ip(g, t)] = 1.0
                row_up[self.ip(g, t - 1)] = -1.0
                constraints.append(LinearConstraint(row_up, -np.inf, ramp))

                row_dn = np.zeros(self.n_vars)
                row_dn[self.ip(g, t)] = 1.0
                row_dn[self.ip(g, t - 1)] = -1.0
                constraints.append(LinearConstraint(row_dn, -ramp, np.inf))

        # ---------- battery dynamics ----------
        cap = self.battery["capacity_mwh"]
        eff = self.battery["efficiency"]
        soc0 = soc_init_mwh if soc_init_mwh is not None else self.battery["soc_init_frac"] * cap
        for t in range(T):
            row = np.zeros(self.n_vars)
            row[self.isoc(t)] = 1.0
            row[self.ic(t)] = -eff
            row[self.id_(t)] = 1.0 / eff
            if t == 0:
                rhs = soc0
            else:
                row[self.isoc(t - 1)] = -1.0
                rhs = 0.0
            constraints.append(LinearConstraint(row, rhs, rhs))

        if soc_terminal_min_mwh is not None:
            row = np.zeros(self.n_vars)
            row[self.isoc(T - 1)] = 1.0
            constraints.append(LinearConstraint(row, soc_terminal_min_mwh, np.inf))

        # ---------- assemble bounds ----------
        lb = np.zeros(self.n_vars)
        ub = np.full(self.n_vars, np.inf)
        integrality = np.zeros(self.n_vars)

        for g in range(G):
            for t in range(T):
                integrality[self.iu(g, t)] = 1
                ub[self.iu(g, t)] = 1
                integrality[self.is_(g, t)] = 1
                ub[self.is_(g, t)] = 1
                integrality[self.iv(g, t)] = 1
                ub[self.iv(g, t)] = 1
                ub[self.ip(g, t)] = fleet.loc[g, "pmax_mw"]

        for t in range(T):
            ub[self.ic(t)] = self.battery["power_mw"]
            ub[self.id_(t)] = self.battery["power_mw"]
            lb[self.isoc(t)] = self.battery["soc_min_frac"] * cap
            ub[self.isoc(t)] = self.battery["soc_max_frac"] * cap
            ub[self.iunserved(t)] = np.inf

        if fixed_commitment is not None:
            for g in range(G):
                for t in range(T):
                    val = float(fixed_commitment[g, t])
                    lb[self.iu(g, t)] = val
                    ub[self.iu(g, t)] = val

        bounds = Bounds(lb, ub)

        res = milp(c_obj, constraints=constraints, integrality=integrality,
                   bounds=bounds, options={"time_limit": 60})

        return self._package_result(res, demand, renewable_mw)

    def _package_result(self, res, demand, renewable_mw) -> UCResult:
        if not res.success:
            return UCResult("infeasible_or_failed", np.nan, pd.DataFrame(), pd.DataFrame(),
                             np.zeros(self.T), np.zeros(self.T))

        x = res.x
        rows = []
        for g in range(self.G):
            for t in range(self.T):
                rows.append({
                    "generator": self.fleet.loc[g, "name"],
                    "hour": t,
                    "on": round(x[self.iu(g, t)]),
                    "power_mw": x[self.ip(g, t)],
                    "startup": round(x[self.is_(g, t)]),
                    "shutdown": round(x[self.iv(g, t)]),
                })
        dispatch = pd.DataFrame(rows)

        battery = pd.DataFrame({
            "hour": range(self.T),
            "charge_mw": [x[self.ic(t)] for t in range(self.T)],
            "discharge_mw": [x[self.id_(t)] for t in range(self.T)],
            "soc_mwh": [x[self.isoc(t)] for t in range(self.T)],
        })

        curt = np.array([x[self.icurt(t)] for t in range(self.T)])
        unserved = np.array([x[self.iunserved(t)] for t in range(self.T)])

        return UCResult("optimal", res.fun, dispatch, battery, curt, unserved)


@dataclass
class ScenarioOutcome:
    """Recourse (second-stage) outcome for one scenario, under the shared commitment."""
    probability: float
    demand_label: str
    renewable_label: str
    cost: float  # shared startup cost + this scenario's recourse cost, i.e. "if this
                 # scenario happens, given the commitment we chose, total cost would be..."
    dispatch: pd.DataFrame
    battery: pd.DataFrame
    curtailment: np.ndarray
    unserved: np.ndarray


@dataclass
class StochasticUCResult:
    status: str
    expected_cost: float       # startup cost + probability-weighted expected recourse cost
                                # (this is exactly the MILP's objective value)
    startup_cost: float        # first-stage, paid once regardless of which scenario happens
    commitment: pd.DataFrame   # shared on/off schedule: generator, hour, on, startup
    scenarios: list[ScenarioOutcome]


class StochasticUnitCommitmentModel:
    """
    Two-stage stochastic unit commitment: ONE shared commitment schedule
    (u/s/v -- decided before any scenario is known, matching how day-ahead UC
    actually works, since startup lead times mean you can't wait to see what
    demand/renewables actually do) but scenario-specific recourse (power
    output, battery operation, curtailment, unserved energy -- decided in
    real time once the scenario resolves). The objective minimizes shared
    startup cost plus the PROBABILITY-WEIGHTED EXPECTED recourse cost across
    scenarios.

    This is deliberately different from -- and more correct than -- solving
    each scenario independently with UnitCommitmentModel and combining the
    results afterward: independent solves would each pick their own
    commitment schedule, and you cannot fractionally blend several different
    discrete on/off decisions into one implementable plan. Here there is
    exactly one commitment decision, chosen to perform well in expectation
    across every scenario simultaneously.
    """

    def __init__(self, fleet: pd.DataFrame, battery: dict, scenarios: list[dict], T: int = 24):
        """scenarios: list of dicts, each with 'probability' (float), 'demand' (length-T
        array), 'renewable' (length-T array), and optionally 'demand_label'/'renewable_label'
        for reporting. Probabilities should sum to ~1 (not enforced, but the objective's
        expected-cost interpretation only holds if they do)."""
        self.fleet = fleet.reset_index(drop=True)
        self.G = len(fleet)
        self.T = T
        self.battery = battery
        self.scenarios = scenarios
        self.W = len(scenarios)

        G, T, W = self.G, self.T, self.W
        # First-stage (shared across scenarios): u, s, v -- G*T each
        self.n_u = G * T
        self.n_s = G * T
        self.n_v = G * T
        # Second-stage (per scenario): p is G*T*W; c/d/soc/curt/unserved are T*W each
        self.n_p = G * T * W
        self.n_c = T * W
        self.n_d = T * W
        self.n_soc = T * W
        self.n_curt = T * W
        self.n_unserved = T * W

        self.off_u = 0
        self.off_s = self.off_u + self.n_u
        self.off_v = self.off_s + self.n_s
        self.off_p = self.off_v + self.n_v
        self.off_c = self.off_p + self.n_p
        self.off_d = self.off_c + self.n_c
        self.off_soc = self.off_d + self.n_d
        self.off_curt = self.off_soc + self.n_soc
        self.off_unserved = self.off_curt + self.n_curt
        self.n_vars = self.off_unserved + self.n_unserved

    # -- index helpers -- first-stage (g,t); second-stage (g,t,w) or (t,w)
    def iu(self, g, t): return self.off_u + g * self.T + t
    def is_(self, g, t): return self.off_s + g * self.T + t
    def iv(self, g, t): return self.off_v + g * self.T + t
    def ip(self, g, t, w): return self.off_p + (g * self.T + t) * self.W + w
    def ic(self, t, w): return self.off_c + t * self.W + w
    def id_(self, t, w): return self.off_d + t * self.W + w
    def isoc(self, t, w): return self.off_soc + t * self.W + w
    def icurt(self, t, w): return self.off_curt + t * self.W + w
    def iunserved(self, t, w): return self.off_unserved + t * self.W + w

    def build_and_solve(self, u_prev: np.ndarray | None = None,
                         unserved_penalty: float = 5000.0,
                         soc_init_mwh: float | None = None,
                         soc_terminal_min_mwh: float | None = None) -> StochasticUCResult:
        """soc_init_mwh, soc_terminal_min_mwh: see UnitCommitmentModel.build_and_solve --
        same meaning, same rolling-horizon use case. soc_terminal_min_mwh is applied to
        EVERY scenario's ending soc (not just the expected one), since whichever scenario
        actually happens, the operator still needs a reasonable starting point for the
        next day -- this is what stops the stochastic model from draining the battery by
        hour 23 in every scenario the same way the single-scenario model would without it."""
        T, G, W, fleet = self.T, self.G, self.W, self.fleet
        if u_prev is None:
            u_prev = np.zeros(G)
        probs = np.array([s["probability"] for s in self.scenarios])

        # ---------- objective: shared startup cost + probability-weighted expected recourse ----------
        c_obj = np.zeros(self.n_vars)
        for g in range(G):
            for t in range(T):
                c_obj[self.is_(g, t)] = fleet.loc[g, "startup_cost"]
        for w in range(W):
            for g in range(G):
                for t in range(T):
                    c_obj[self.ip(g, t, w)] = probs[w] * fleet.loc[g, "marginal_cost"]
            for t in range(T):
                c_obj[self.iunserved(t, w)] = probs[w] * unserved_penalty

        constraints = []

        # ---------- power balance per (t,w) ----------
        for w in range(W):
            demand_w, renewable_w = self.scenarios[w]["demand"], self.scenarios[w]["renewable"]
            for t in range(T):
                row = np.zeros(self.n_vars)
                for g in range(G):
                    row[self.ip(g, t, w)] = 1.0
                row[self.icurt(t, w)] = -1.0
                row[self.id_(t, w)] = 1.0
                row[self.ic(t, w)] = -1.0
                row[self.iunserved(t, w)] = 1.0
                rhs = demand_w[t] - renewable_w[t]
                constraints.append(LinearConstraint(row, rhs, rhs))

                # curtailment bound: curt[t,w] <= renewable[w][t]
                row2 = np.zeros(self.n_vars)
                row2[self.icurt(t, w)] = 1.0
                constraints.append(LinearConstraint(row2, -np.inf, max(renewable_w[t], 0.0)))

        # ---------- generator output linked to SHARED commitment, per (g,t,w) ----------
        for g in range(G):
            pmin, pmax = fleet.loc[g, "pmin_mw"], fleet.loc[g, "pmax_mw"]
            for t in range(T):
                for w in range(W):
                    row_hi = np.zeros(self.n_vars)
                    row_hi[self.ip(g, t, w)] = 1.0
                    row_hi[self.iu(g, t)] = -pmax
                    constraints.append(LinearConstraint(row_hi, -np.inf, 0.0))

                    row_lo = np.zeros(self.n_vars)
                    row_lo[self.ip(g, t, w)] = 1.0
                    row_lo[self.iu(g, t)] = -pmin
                    constraints.append(LinearConstraint(row_lo, 0.0, np.inf))

        # ---------- startup/shutdown linking -- first-stage only, no scenario dependence ----------
        for g in range(G):
            for t in range(T):
                row_s = np.zeros(self.n_vars)
                row_s[self.is_(g, t)] = 1.0
                row_s[self.iu(g, t)] = -1.0
                if t == 0:
                    constraints.append(LinearConstraint(row_s, -u_prev[g], np.inf))
                else:
                    row_s[self.iu(g, t - 1)] = 1.0
                    constraints.append(LinearConstraint(row_s, 0.0, np.inf))

                row_v = np.zeros(self.n_vars)
                row_v[self.iv(g, t)] = 1.0
                row_v[self.iu(g, t)] = 1.0
                if t == 0:
                    constraints.append(LinearConstraint(row_v, u_prev[g], np.inf))
                else:
                    row_v[self.iu(g, t - 1)] = -1.0
                    constraints.append(LinearConstraint(row_v, 0.0, np.inf))

        # ---------- min up/down time -- first-stage only ----------
        for g in range(G):
            min_up = int(fleet.loc[g, "min_up_hr"])
            min_down = int(fleet.loc[g, "min_down_hr"])
            for t in range(T):
                window_up = range(max(0, t - min_up + 1), t + 1)
                row = np.zeros(self.n_vars)
                for i in window_up:
                    row[self.is_(g, i)] = 1.0
                row[self.iu(g, t)] = -1.0
                constraints.append(LinearConstraint(row, -np.inf, 0.0))

                window_down = range(max(0, t - min_down + 1), t + 1)
                row2 = np.zeros(self.n_vars)
                for i in window_down:
                    row2[self.iv(g, i)] = 1.0
                row2[self.iu(g, t)] = 1.0
                constraints.append(LinearConstraint(row2, -np.inf, 1.0))

        # ---------- ramp limits, per scenario (must hold along whichever path materializes) ----------
        for g in range(G):
            ramp = fleet.loc[g, "ramp_mw_per_hr"]
            for w in range(W):
                for t in range(1, T):
                    row_up = np.zeros(self.n_vars)
                    row_up[self.ip(g, t, w)] = 1.0
                    row_up[self.ip(g, t - 1, w)] = -1.0
                    constraints.append(LinearConstraint(row_up, -np.inf, ramp))

                    row_dn = np.zeros(self.n_vars)
                    row_dn[self.ip(g, t, w)] = 1.0
                    row_dn[self.ip(g, t - 1, w)] = -1.0
                    constraints.append(LinearConstraint(row_dn, -ramp, np.inf))

        # ---------- battery dynamics, per scenario (same known starting SoC for all) ----------
        cap = self.battery["capacity_mwh"]
        eff = self.battery["efficiency"]
        soc0 = soc_init_mwh if soc_init_mwh is not None else self.battery["soc_init_frac"] * cap
        for w in range(W):
            for t in range(T):
                row = np.zeros(self.n_vars)
                row[self.isoc(t, w)] = 1.0
                row[self.ic(t, w)] = -eff
                row[self.id_(t, w)] = 1.0 / eff
                if t == 0:
                    rhs = soc0
                else:
                    row[self.isoc(t - 1, w)] = -1.0
                    rhs = 0.0
                constraints.append(LinearConstraint(row, rhs, rhs))

            if soc_terminal_min_mwh is not None:
                row = np.zeros(self.n_vars)
                row[self.isoc(T - 1, w)] = 1.0
                constraints.append(LinearConstraint(row, soc_terminal_min_mwh, np.inf))

        # ---------- bounds ----------
        lb = np.zeros(self.n_vars)
        ub = np.full(self.n_vars, np.inf)
        integrality = np.zeros(self.n_vars)

        for g in range(G):
            for t in range(T):
                integrality[self.iu(g, t)] = 1
                ub[self.iu(g, t)] = 1
                integrality[self.is_(g, t)] = 1
                ub[self.is_(g, t)] = 1
                integrality[self.iv(g, t)] = 1
                ub[self.iv(g, t)] = 1
                for w in range(W):
                    ub[self.ip(g, t, w)] = fleet.loc[g, "pmax_mw"]

        for w in range(W):
            for t in range(T):
                ub[self.ic(t, w)] = self.battery["power_mw"]
                ub[self.id_(t, w)] = self.battery["power_mw"]
                lb[self.isoc(t, w)] = self.battery["soc_min_frac"] * cap
                ub[self.isoc(t, w)] = self.battery["soc_max_frac"] * cap

        bounds = Bounds(lb, ub)

        res = milp(c_obj, constraints=constraints, integrality=integrality,
                   bounds=bounds, options={"time_limit": 120})

        return self._package_result(res, probs, unserved_penalty)

    def _package_result(self, res, probs, unserved_penalty) -> StochasticUCResult:
        if not res.success:
            return StochasticUCResult("infeasible_or_failed", np.nan, np.nan, pd.DataFrame(), [])

        x = res.x
        T, G, W, fleet = self.T, self.G, self.W, self.fleet

        commitment_rows = []
        startup_cost_total = 0.0
        for g in range(G):
            for t in range(T):
                on = round(x[self.iu(g, t)])
                startup = round(x[self.is_(g, t)])
                commitment_rows.append({
                    "generator": fleet.loc[g, "name"], "hour": t, "on": on, "startup": startup,
                })
                startup_cost_total += startup * fleet.loc[g, "startup_cost"]
        commitment = pd.DataFrame(commitment_rows)

        scenario_outcomes = []
        for w in range(W):
            dispatch_rows = []
            for g in range(G):
                for t in range(T):
                    dispatch_rows.append({
                        "generator": fleet.loc[g, "name"], "hour": t,
                        "on": round(x[self.iu(g, t)]),
                        "power_mw": x[self.ip(g, t, w)],
                        "startup": round(x[self.is_(g, t)]),
                    })
            dispatch = pd.DataFrame(dispatch_rows)
            battery_df = pd.DataFrame({
                "hour": range(T),
                "charge_mw": [x[self.ic(t, w)] for t in range(T)],
                "discharge_mw": [x[self.id_(t, w)] for t in range(T)],
                "soc_mwh": [x[self.isoc(t, w)] for t in range(T)],
            })
            curt = np.array([x[self.icurt(t, w)] for t in range(T)])
            unserved = np.array([x[self.iunserved(t, w)] for t in range(T)])

            recourse_cost = sum(
                x[self.ip(g, t, w)] * fleet.loc[g, "marginal_cost"] for g in range(G) for t in range(T)
            ) + unserved.sum() * unserved_penalty
            scenario_cost = startup_cost_total + recourse_cost

            scenario_outcomes.append(ScenarioOutcome(
                probability=self.scenarios[w]["probability"],
                demand_label=self.scenarios[w].get("demand_label", f"scenario_{w}"),
                renewable_label=self.scenarios[w].get("renewable_label", ""),
                cost=scenario_cost, dispatch=dispatch, battery=battery_df,
                curtailment=curt, unserved=unserved,
            ))

        return StochasticUCResult(
            status="optimal", expected_cost=res.fun, startup_cost=startup_cost_total,
            commitment=commitment, scenarios=scenario_outcomes,
        )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_gen import generate_hourly_dataset, thermal_fleet_spec, battery_spec

    df = generate_hourly_dataset(n_days=30)
    day = df.iloc[-24:].reset_index(drop=True)
    demand = day["demand_mw"].values
    renewable_mw = (day["wind_mw"] + day["solar_mw"]).values

    model = UnitCommitmentModel(thermal_fleet_spec(), battery_spec(), T=24)
    result = model.build_and_solve(demand, renewable_mw)

    print("Status:", result.status)
    print("Total cost: $%.0f" % result.total_cost)
    print("\nDispatch (on hours only):")
    print(result.dispatch[result.dispatch["on"] == 1].to_string(index=False))
    print("\nMax unserved demand:", result.unserved.max())
    print("Total curtailment (MWh):", result.curtailment.sum())
    print("\nBattery:")
    print(result.battery.to_string(index=False))
