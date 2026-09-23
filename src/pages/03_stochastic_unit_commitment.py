"""
Stochastic Unit Commitment (9-Path) page.

The real version of the "nine path probabilities" idea:
  1. Forecast demand AND renewables, each with P10/P50/P90 quantiles.
  2. Estimate the JOINT probability of each (demand level, renewable level)
     pair empirically from paired historical forecast errors -- not by
     assuming independence and multiplying two marginals together.
  3. Solve ONE two-stage stochastic MILP across all 9 scenarios: a single
     shared commitment schedule, chosen to minimize probability-weighted
     expected cost, with scenario-specific recourse (dispatch/battery/
     curtailment/unserved) for whichever scenario actually happens.
  4. Compare against the naive alternative: commit based on the P50
     scenario alone, then see what happens if a worse scenario occurs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from data_gen import thermal_fleet_spec, battery_spec
from forecasting import fit_forecast_and_history
from unit_commitment import UnitCommitmentModel, StochasticUnitCommitmentModel
from analysis import fuel_adjusted_fleet
from scenarios import (
    paired_errors_from_predictions, joint_scenario_probabilities,
    independence_baseline_probabilities, build_nine_scenarios, combine_renewable_forecast,
)
import scenario as scenario_mod

st.set_page_config(page_title="Stochastic Unit Commitment", layout="wide")
st.title("Stochastic Unit Commitment: the 9-path hedge")
st.markdown(
    "Demand and renewable output each get a P10/P50/P90 forecast, giving 9 possible "
    "(demand level, renewable level) combinations. Rather than assuming those two "
    "are independent (which would understate correlated risk -- e.g. a cold snap "
    "driving high demand *and* low wind at the same time), the probability of each "
    "combination is estimated **empirically** from paired historical forecast errors. "
    "Those 9 scenarios then feed a single **two-stage stochastic MILP**: one shared "
    "commitment schedule (decided before any scenario is known, since thermal units "
    "can't be started on a moment's notice) that performs well across all 9 weighted "
    "outcomes simultaneously."
)


def baseline(key):
    return st.session_state.get(key, scenario_mod.DEFAULTS[key])


with st.sidebar:
    st.header("Scenario setup")
    n_days_history = st.slider("Days of training history", 60, 400, int(baseline("n_days_history")), step=20)
    battery_power = st.slider("Battery power rating (MW)", 0, 150, int(baseline("battery_power")), step=10)
    battery_capacity = st.slider("Battery capacity (MWh)", 0, 500, int(baseline("battery_capacity")), step=25)
    coal_price = st.slider("Coal price ($/MMBtu)", 1.0, 8.0, float(baseline("coal_price")), step=0.10)
    gas_price = st.slider("Gas price ($/MMBtu)", 2.0, 12.0, float(baseline("gas_price")), step=0.10)
    use_empirical = st.checkbox("Use empirical joint probabilities (uncheck to use the independence assumption instead)",
                                 value=True)
    run_btn = st.button("Build scenarios & solve", type="primary")


@st.cache_data
def load_and_forecast(n_days_history):
    history, next_day = scenario_mod.load_data(n_days_history)
    # Fit each target's quantile models exactly once, reused for both the
    # next-day forecast and the in-sample historical error analysis --
    # fitting demand/wind/solar separately for each purpose would double
    # the (GBM x 3 quantiles x 3 targets) training cost for no benefit.
    demand_fc, demand_hist = fit_forecast_and_history(history, "demand_mw", next_day,
                                                        capacity_mw=1.0, clip_range=(0.0, None))
    wind_fc, wind_hist = fit_forecast_and_history(history, "wind_cf", next_day, capacity_mw=300)
    solar_fc, solar_hist = fit_forecast_and_history(history, "solar_cf", next_day, capacity_mw=250)

    renewable_fc = combine_renewable_forecast(wind_fc, solar_fc)
    paired = paired_errors_from_predictions(demand_hist, wind_hist, solar_hist)

    actual_demand = next_day["demand_mw"].values
    actual_renewable = (next_day["wind_mw"] + next_day["solar_mw"]).values
    return demand_fc, renewable_fc, paired, actual_demand, actual_renewable


def plot_probability_grid(probs: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(probs.values, cmap="Blues", vmin=0, vmax=probs.values.max() * 1.2)
    ax.set_xticks(range(3)); ax.set_xticklabels(probs.columns)
    ax.set_yticks(range(3)); ax.set_yticklabels(probs.index)
    ax.set_xlabel("Renewable level")
    ax.set_ylabel("Demand level")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{probs.values[i, j]:.1%}", ha="center", va="center", fontsize=11)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


demand_fc, renewable_fc, paired, actual_demand, actual_renewable = load_and_forecast(n_days_history)
fleet = fuel_adjusted_fleet(thermal_fleet_spec(), coal_price, gas_price)
battery = battery_spec()
battery["power_mw"] = battery_power
battery["capacity_mwh"] = battery_capacity

st.subheader("Joint scenario probabilities")
corr = paired["demand_error"].corr(paired["renewable_error"])
st.metric("Correlation between demand and renewable forecast errors (historical)", f"{corr:+.3f}")

joint = joint_scenario_probabilities(paired)
indep = independence_baseline_probabilities(joint)
col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_probability_grid(joint, "Empirical joint probabilities"))
with col2:
    st.pyplot(plot_probability_grid(indep, "Independence-assumption baseline"))
st.caption(
    "If these two grids look meaningfully different, the independence assumption would "
    "have mis-estimated risk -- watch the high-demand/low-renewable corner in particular."
)

probs_to_use = joint if use_empirical else indep
scenarios = build_nine_scenarios(demand_fc, renewable_fc, probs_to_use)

st.divider()

# ---------------------------------------------------------------------------
# Solve: stochastic hedge vs. naive P50-only commitment
# ---------------------------------------------------------------------------
if run_btn or "stoch_result" not in st.session_state:
    with st.spinner("Solving the two-stage stochastic MILP (9 scenarios)..."):
        stoch_model = StochasticUnitCommitmentModel(fleet, battery, scenarios, T=24)
        stoch_result = stoch_model.build_and_solve()

        p50_scenario = [s for s in scenarios if s["demand_label"] == "mid" and s["renewable_label"] == "mid"][0]
        p50_model = UnitCommitmentModel(fleet, battery, T=24)
        p50_plan = p50_model.build_and_solve(p50_scenario["demand"], p50_scenario["renewable"])
        p50_committed = None
        if p50_plan.status == "optimal":
            p50_committed = (p50_plan.dispatch.pivot(index="generator", columns="hour", values="on")
                              .loc[fleet["name"]].values)

        st.session_state["stoch_result"] = dict(
            stoch_result=stoch_result, p50_committed=p50_committed, scenarios=scenarios, fleet=fleet, battery=battery,
        )

saved = st.session_state["stoch_result"]
stoch_result, p50_committed = saved["stoch_result"], saved["p50_committed"]

st.subheader("Shared commitment: stochastic hedge vs. naive P50-only")

if stoch_result.status != "optimal":
    st.warning("Stochastic solve did not reach optimality -- try adjusting the sidebar settings.")
else:
    expected_cost = stoch_result.expected_cost
    st.metric("Expected cost across all 9 scenarios (stochastic hedge)", f"${expected_cost:,.0f}")

    st.markdown("**Per-scenario outcome under the stochastic hedge's shared commitment:**")
    rows = []
    for s in stoch_result.scenarios:
        rows.append({
            "demand": s.demand_label, "renewable": s.renewable_label, "probability": s.probability,
            "cost_if_this_happens": s.cost, "max_unserved_mw": s.unserved.max(),
            "curtailment_mwh": s.curtailment.sum(),
        })
    scen_df = pd.DataFrame(rows).sort_values(["demand", "renewable"])
    st.dataframe(scen_df.style.format({
        "probability": "{:.1%}", "cost_if_this_happens": "${:,.0f}",
        "max_unserved_mw": "{:.1f}", "curtailment_mwh": "{:.1f}",
    }), hide_index=True)

    st.markdown(
        "**Now the comparison that matters:** take the commitment schedule you'd get "
        "from committing based on the P50 scenario alone (what the main app does today), "
        "and see what happens if a WORSE scenario actually occurs -- same fleet, same "
        "battery, just settled against different demand/renewable realizations with that "
        "fixed, already-decided commitment."
    )

    if p50_committed is not None:
        compare_rows = []
        for s in scenarios:
            settle_model = UnitCommitmentModel(fleet, battery, T=24)
            settled = settle_model.build_and_solve(s["demand"], s["renewable"], fixed_commitment=p50_committed)
            stoch_outcome = next(so for so in stoch_result.scenarios
                                  if so.demand_label == s["demand_label"] and so.renewable_label == s["renewable_label"])
            compare_rows.append({
                "demand": s["demand_label"], "renewable": s["renewable_label"], "probability": s["probability"],
                "P50-only commitment: cost": settled.total_cost if settled.status == "optimal" else np.nan,
                "P50-only commitment: unserved_mw": settled.unserved.max() if settled.status == "optimal" else np.nan,
                "Stochastic hedge: cost": stoch_outcome.cost,
                "Stochastic hedge: unserved_mw": stoch_outcome.unserved.max(),
            })
        compare_df = pd.DataFrame(compare_rows).sort_values(["demand", "renewable"])
        st.dataframe(compare_df.style.format({
            "probability": "{:.1%}", "P50-only commitment: cost": "${:,.0f}",
            "P50-only commitment: unserved_mw": "{:.1f}", "Stochastic hedge: cost": "${:,.0f}",
            "Stochastic hedge: unserved_mw": "{:.1f}",
        }), hide_index=True)

        p50_expected = (compare_df["probability"] * compare_df["P50-only commitment: cost"]).sum()
        stoch_expected = (compare_df["probability"] * compare_df["Stochastic hedge: cost"]).sum()
        worst_p50_unserved = compare_df["P50-only commitment: unserved_mw"].max()
        worst_stoch_unserved = compare_df["Stochastic hedge: unserved_mw"].max()

        col1, col2, col3 = st.columns(3)
        col1.metric("P50-only: expected cost", f"${p50_expected:,.0f}")
        col2.metric("Stochastic hedge: expected cost", f"${stoch_expected:,.0f}",
                    delta=f"{stoch_expected - p50_expected:,.0f}", delta_color="inverse")
        col3.metric("Worst-case unserved demand", f"P50-only: {worst_p50_unserved:.0f} MW",
                    delta=f"Stochastic hedge: {worst_stoch_unserved:.0f} MW", delta_color="off")

        st.markdown(
            "The stochastic hedge typically costs a little more in expectation (it's paying "
            "an insurance premium -- committing slightly more thermal capacity than the P50 "
            "plan alone would) but should show dramatically lower worst-case unserved demand, "
            "since it never has to be redispatched beyond what was already committed."
        )
