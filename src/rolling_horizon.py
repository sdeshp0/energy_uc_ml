"""
Rolling-horizon simulation: walk forward day by day over a multi-day window,
re-solving the day-ahead unit commitment each day and carrying battery SoC
and generator on/off status forward as the next day's starting condition --
instead of every day being an isolated 24h snapshot that resets to a fixed
initial state.

Two commitment approaches are run side by side, each day, against the SAME
actual demand/renewable realization, so their results are directly
comparable:
  - "p50_only": commit based on the P50 (median) forecast alone, then settle
    against actual demand/renewable -- what the main app does today, applied
    across many days instead of one.
  - "stochastic": build that day's 9-scenario hedge (same demand/renewable
    quantile forecasts, paired with a joint probability table) and solve
    StochasticUnitCommitmentModel, then settle its chosen commitment against
    actual demand/renewable the same way.

Each approach carries its OWN battery SoC and generator state forward
independently, since the two approaches generally commit differently.

Deliberate simplifications (see docs/CODE_WALKTHROUGH.md for the full list):
the demand/renewable forecast MODELS are fit once at the start of the
simulation window and then just applied (predicted on) each simulated day --
not refit daily. The joint scenario probability table is likewise computed
once from the initial training window's in-sample errors, not updated as
the simulation progresses. Both are reasonable first-pass choices (refitting
daily would be far slower for limited benefit over a few weeks) but are
exactly the kind of "rigor vs. cost" tradeoff flagged as future work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_gen import generate_hourly_dataset
from forecasting import fit_and_predict_range
from unit_commitment import UnitCommitmentModel, StochasticUnitCommitmentModel
from scenarios import (
    paired_errors_from_predictions, joint_scenario_probabilities, build_nine_scenarios,
)


@dataclass
class DayRecord:
    day_index: int
    date: str
    p50_only_cost: float
    p50_only_unserved_mw: float
    p50_only_curtailment_mwh: float
    p50_only_soc_end_mwh: float
    stochastic_cost: float
    stochastic_unserved_mw: float
    stochastic_curtailment_mwh: float
    stochastic_soc_end_mwh: float


@dataclass
class RollingHorizonResult:
    days: list[DayRecord] = field(default_factory=list)
    joint_probabilities: pd.DataFrame | None = None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(d) for d in self.days])

    def summary(self) -> dict:
        df = self.to_frame()
        return {
            "n_days": len(df),
            "p50_only_total_cost": df["p50_only_cost"].sum(),
            "stochastic_total_cost": df["stochastic_cost"].sum(),
            "p50_only_total_unserved_mwh": df["p50_only_unserved_mw"].sum(),
            "stochastic_total_unserved_mwh": df["stochastic_unserved_mw"].sum(),
            "p50_only_days_with_unserved": int((df["p50_only_unserved_mw"] > 0.01).sum()),
            "stochastic_days_with_unserved": int((df["stochastic_unserved_mw"] > 0.01).sum()),
        }


def _final_u(dispatch: pd.DataFrame, fleet_names: list[str], T: int) -> np.ndarray:
    """Extract each generator's on/off status in the LAST hour of a solved day,
    to hand forward as next day's u_prev."""
    last_hour = dispatch[dispatch["hour"] == T - 1].set_index("generator")
    return np.array([last_hour.loc[g, "on"] for g in fleet_names])


def simulate(fleet: pd.DataFrame, battery_cfg: dict, n_train_days: int = 200,
             n_sim_days: int = 14, T: int = 24, unserved_penalty: float = 5000.0,
             wind_capacity_mw: float = 300, solar_capacity_mw: float = 250) -> RollingHorizonResult:
    """Run the full rolling-horizon simulation. Returns per-day results for
    both approaches plus the (fixed, once-computed) joint probability table
    used for every simulated day's stochastic hedge."""
    total_days = n_train_days + n_sim_days
    df = generate_hourly_dataset(n_days=total_days)
    n_train_rows = n_train_days * T

    # Fit each target's quantile models ONCE, predict across the whole window
    # (training rows + every simulated day) from that single fit.
    demand_pred = fit_and_predict_range(df, "demand_mw", n_train_rows, clip_range=(0.0, None))
    wind_pred = fit_and_predict_range(df, "wind_cf", n_train_rows, clip_range=(0.0, 1.0))
    solar_pred = fit_and_predict_range(df, "solar_cf", n_train_rows, clip_range=(0.0, 1.0))

    # Joint scenario probabilities, computed once from the training window's
    # in-sample rows (same approach as scenarios.py elsewhere in the project).
    train_mask = demand_pred.index < n_train_rows
    paired = paired_errors_from_predictions(
        demand_pred[train_mask], wind_pred[train_mask], solar_pred[train_mask],
        wind_capacity_mw, solar_capacity_mw,
    )
    joint_probs = joint_scenario_probabilities(paired)

    fleet_names = list(fleet["name"])
    G = len(fleet)
    cap = battery_cfg["capacity_mwh"]
    soc_terminal_target = battery_cfg["soc_init_frac"] * cap  # require ending SoC back at this
    # level each day -- otherwise a finite-horizon solve has no reason not to drain the
    # battery to its floor by the last hour, crippling every subsequent day (see docstring).

    # Independent running state per approach
    u_prev_p50 = np.zeros(G)
    soc_p50 = battery_cfg["soc_init_frac"] * cap
    u_prev_stoch = np.zeros(G)
    soc_stoch = battery_cfg["soc_init_frac"] * cap

    records = []
    for day_i in range(n_sim_days):
        day_start_row = n_train_rows + day_i * T
        day_rows = df.iloc[day_start_row:day_start_row + T]
        actual_demand = day_rows["demand_mw"].values
        actual_renewable = (day_rows["wind_mw"] + day_rows["solar_mw"]).values
        date_label = str(day_rows["timestamp"].iloc[0].date())

        # this day's quantile forecasts, sliced from the single whole-window fit
        idx = day_rows.index
        d_fc = demand_pred.loc[demand_pred.index.isin(idx)]
        w_fc = wind_pred.loc[wind_pred.index.isin(idx)]
        s_fc = solar_pred.loc[solar_pred.index.isin(idx)]
        demand_p50 = d_fc["p50"].values
        renewable_p50 = w_fc["p50"].values * wind_capacity_mw + s_fc["p50"].values * solar_capacity_mw

        # ---------- approach 1: commit on P50 alone, settle against actual ----------
        plan_model = UnitCommitmentModel(fleet, battery_cfg, T=T)
        plan = plan_model.build_and_solve(demand_p50, renewable_p50, u_prev=u_prev_p50,
                                           unserved_penalty=unserved_penalty, soc_init_mwh=soc_p50,
                                           soc_terminal_min_mwh=soc_terminal_target)
        if plan.status == "optimal":
            committed = (plan.dispatch.pivot(index="generator", columns="hour", values="on")
                         .loc[fleet_names].values)
            settle_model = UnitCommitmentModel(fleet, battery_cfg, T=T)
            settled = settle_model.build_and_solve(actual_demand, actual_renewable, fixed_commitment=committed,
                                                     unserved_penalty=unserved_penalty, soc_init_mwh=soc_p50,
                                                     soc_terminal_min_mwh=soc_terminal_target)
            p50_cost = settled.total_cost
            p50_unserved = settled.unserved.max()
            p50_curt = settled.curtailment.sum()
            p50_soc_end = settled.battery["soc_mwh"].iloc[-1]
            u_prev_p50 = _final_u(settled.dispatch, fleet_names, T)
            soc_p50 = p50_soc_end
        else:
            p50_cost, p50_unserved, p50_curt, p50_soc_end = np.nan, np.nan, np.nan, soc_p50

        # ---------- approach 2: stochastic 9-scenario hedge, settle against actual ----------
        demand_fc_day = pd.DataFrame({
            "demand_p10_mw": d_fc["p10"].values, "demand_p50_mw": d_fc["p50"].values,
            "demand_p90_mw": d_fc["p90"].values,
        })
        renewable_fc_day = pd.DataFrame({
            "renewable_p10_mw": w_fc["p10"].values * wind_capacity_mw + s_fc["p10"].values * solar_capacity_mw,
            "renewable_p50_mw": w_fc["p50"].values * wind_capacity_mw + s_fc["p50"].values * solar_capacity_mw,
            "renewable_p90_mw": w_fc["p90"].values * wind_capacity_mw + s_fc["p90"].values * solar_capacity_mw,
        })
        day_scenarios = build_nine_scenarios(demand_fc_day, renewable_fc_day, joint_probs)

        stoch_model = StochasticUnitCommitmentModel(fleet, battery_cfg, day_scenarios, T=T)
        stoch = stoch_model.build_and_solve(u_prev=u_prev_stoch, unserved_penalty=unserved_penalty,
                                             soc_init_mwh=soc_stoch, soc_terminal_min_mwh=soc_terminal_target)
        if stoch.status == "optimal":
            committed = (stoch.commitment.pivot(index="generator", columns="hour", values="on")
                         .loc[fleet_names].values)
            settle_model = UnitCommitmentModel(fleet, battery_cfg, T=T)
            settled = settle_model.build_and_solve(actual_demand, actual_renewable, fixed_commitment=committed,
                                                     unserved_penalty=unserved_penalty, soc_init_mwh=soc_stoch,
                                                     soc_terminal_min_mwh=soc_terminal_target)
            stoch_cost = settled.total_cost
            stoch_unserved = settled.unserved.max()
            stoch_curt = settled.curtailment.sum()
            stoch_soc_end = settled.battery["soc_mwh"].iloc[-1]
            u_prev_stoch = _final_u(settled.dispatch, fleet_names, T)
            soc_stoch = stoch_soc_end
        else:
            stoch_cost, stoch_unserved, stoch_curt, stoch_soc_end = np.nan, np.nan, np.nan, soc_stoch

        records.append(DayRecord(
            day_index=day_i, date=date_label,
            p50_only_cost=p50_cost, p50_only_unserved_mw=p50_unserved,
            p50_only_curtailment_mwh=p50_curt, p50_only_soc_end_mwh=p50_soc_end,
            stochastic_cost=stoch_cost, stochastic_unserved_mw=stoch_unserved,
            stochastic_curtailment_mwh=stoch_curt, stochastic_soc_end_mwh=stoch_soc_end,
        ))

    return RollingHorizonResult(days=records, joint_probabilities=joint_probs)


if __name__ == "__main__":
    from data_gen import thermal_fleet_spec, battery_spec

    result = simulate(thermal_fleet_spec(), battery_spec(), n_train_days=200, n_sim_days=14)
    df = result.to_frame()
    print(df.to_string(index=False))
    print()
    print(result.summary())
    