"""
Rolling Horizon Simulation page.

Runs rolling_horizon.simulate() -- a multi-day walk-forward comparison of
committing on the P50 forecast alone vs. the two-stage stochastic hedge,
with battery SoC and generator status carried forward from each day's
ending state into the next day's starting condition (not reset daily).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from data_gen import thermal_fleet_spec, battery_spec
from analysis import fuel_adjusted_fleet
import rolling_horizon
import scenario as scenario_mod

st.set_page_config(page_title="Rolling Horizon Simulation", layout="wide")
st.title("Rolling Horizon Simulation")
st.markdown(
    "Every other page in this app looks at a single, isolated 24-hour day. This page "
    "walks forward day by day over a multi-day window, re-solving the day-ahead unit "
    "commitment each day and carrying the battery's state of charge and each "
    "generator's on/off status forward as the next day's starting condition -- so the "
    "stochastic hedge's value (or the P50-only approach's failures) can be seen as a "
    "trend across many days, not one illustrative example."
)


def baseline(key):
    return st.session_state.get(key, scenario_mod.DEFAULTS[key])


with st.sidebar:
    st.header("Simulation window")
    n_train_days = st.slider("Training days (forecast models fit once on this window)", 60, 300,
                              int(baseline("n_days_history")), step=20)
    n_sim_days = st.slider("Simulated days (walked forward after training)", 3, 30, 14, step=1,
                            help="Each simulated day costs a couple of MILP solves (~1-2s); "
                                 "the one-time model fit is the dominant cost (~15-20s).")
    st.caption(
        "Forecast models are fit ONCE at the start (not refit daily) and applied across "
        "the whole simulated window -- refitting daily would be far slower for limited "
        "benefit over a few weeks. See docs/CODE_WALKTHROUGH.md for the full list of "
        "deliberate simplifications."
    )
    battery_power = st.slider("Battery power rating (MW)", 0, 150, int(baseline("battery_power")), step=10)
    battery_capacity = st.slider("Battery capacity (MWh)", 0, 500, int(baseline("battery_capacity")), step=25)
    coal_price = st.slider("Coal price ($/MMBtu)", 1.0, 8.0, float(baseline("coal_price")), step=0.10)
    gas_price = st.slider("Gas price ($/MMBtu)", 2.0, 12.0, float(baseline("gas_price")), step=0.10)
    run_btn = st.button("Run simulation", type="primary")


@st.cache_data
def run_simulation(n_train_days, n_sim_days, battery_power, battery_capacity, coal_price, gas_price):
    fleet = fuel_adjusted_fleet(thermal_fleet_spec(), coal_price, gas_price)
    battery = battery_spec()
    battery["power_mw"] = battery_power
    battery["capacity_mwh"] = battery_capacity
    result = rolling_horizon.simulate(fleet, battery, n_train_days=n_train_days, n_sim_days=n_sim_days)
    return result.to_frame(), result.summary(), result.joint_probabilities


if run_btn or "rolling_result" not in st.session_state:
    with st.spinner(f"Fitting forecast models and simulating {n_sim_days} days "
                     f"({2 * n_sim_days} MILP solves)... this can take 20-40s"):
        st.session_state["rolling_result"] = run_simulation(
            n_train_days, n_sim_days, battery_power, battery_capacity, coal_price, gas_price
        )

df, summary, joint_probs = st.session_state["rolling_result"]

col1, col2, col3 = st.columns(3)
col1.metric("P50-only: total cost", f"${summary['p50_only_total_cost']:,.0f}",
            delta=f"{summary['p50_only_days_with_unserved']}/{summary['n_days']} days with unserved demand",
            delta_color="off")
col2.metric("Stochastic hedge: total cost", f"${summary['stochastic_total_cost']:,.0f}",
            delta=f"{summary['stochastic_days_with_unserved']}/{summary['n_days']} days with unserved demand",
            delta_color="off")
col3.metric("Cost difference", f"${summary['stochastic_total_cost'] - summary['p50_only_total_cost']:,.0f}",
            delta="negative = hedge is cheaper in aggregate", delta_color="off")

st.subheader("Daily cost")
cost_df = df.set_index("day_index")[["p50_only_cost", "stochastic_cost"]]
cost_df.columns = ["P50-only", "Stochastic hedge"]
st.line_chart(cost_df)

st.subheader("Daily unserved demand")
unserved_df = df.set_index("day_index")[["p50_only_unserved_mw", "stochastic_unserved_mw"]]
unserved_df.columns = ["P50-only", "Stochastic hedge"]
st.bar_chart(unserved_df)
st.caption(
    "If the stochastic hedge still shows unserved demand on the same day as P50-only, "
    "that day's actual conditions likely fell outside even the P90/P10 scenario coverage -- "
    "a hedge across 9 scenarios reduces risk, it doesn't eliminate every possible outcome."
)

st.subheader("Battery state of charge, end of each day")
soc_df = df.set_index("day_index")[["p50_only_soc_end_mwh", "stochastic_soc_end_mwh"]]
soc_df.columns = ["P50-only", "Stochastic hedge"]
st.line_chart(soc_df)
st.caption(
    "This should stay roughly stable across days (returning to the target level each night) "
    "rather than drifting toward zero -- a finite-horizon solve has no reason to preserve "
    "battery charge across a day boundary unless explicitly required to, which is enforced "
    "here via a terminal state-of-charge floor at each day's planning stage."
)

st.subheader("Day-by-day detail")
display_df = df.copy()
display_df.columns = [c.replace("_", " ") for c in display_df.columns]
st.dataframe(display_df.style.format({c: "${:,.0f}" for c in display_df.columns if "cost" in c} |
                                       {c: "{:.1f}" for c in display_df.columns if "mw" in c or "mwh" in c}),
             hide_index=True)

with st.expander("Joint scenario probabilities used throughout this simulation"):
    st.markdown(
        "Computed once from the training window's in-sample forecast errors (see page 3 for "
        "the empirical-vs-independence comparison) and held fixed for every simulated day."
    )
    st.dataframe(joint_probs.style.format("{:.1%}"))
