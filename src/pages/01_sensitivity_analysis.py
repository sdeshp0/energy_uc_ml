"""
Sensitivity Analysis page.

Run via the main app: streamlit run src/app.py, then open this page from
the sidebar nav (Streamlit auto-discovers files under src/pages/).

Sweeps one parameter at a time (fuel prices, battery power, battery
capacity) across a range, re-solving the MILP at each point, and plots
total cost / curtailment / generation mix against it. Each solve is
~0.25s, so an 8-10 point sweep is a couple seconds -- cheap enough to
run interactively rather than needing to be precomputed.
"""

import numpy as np
import streamlit as st

from data_gen import generate_hourly_dataset, thermal_fleet_spec, battery_spec
from analysis import (
    representative_day, sweep_fuel_price, sweep_battery_param,
    plot_sweep_cost, plot_sweep_generation_mix,
)

st.set_page_config(page_title="Sensitivity Analysis", layout="wide")
st.title("Sensitivity Analysis")
st.markdown(
    "Each chart below holds every parameter at its baseline value (from the "
    "main page's sliders, if you've visited it this session -- otherwise the "
    "app defaults) and sweeps **one** parameter across a range, re-solving the "
    "day-ahead unit commitment at every point. The dotted vertical line marks "
    "your current slider value."
)

DEFAULTS = {"coal_price": 2.20, "gas_price": 4.50, "battery_power": 60, "battery_capacity": 200,
            "n_days_history": 200}


def baseline(key):
    return st.session_state.get(key, DEFAULTS[key])


with st.sidebar:
    st.header("Representative day")
    day_type = st.select_slider(
        "Day type", options=["Typical (p50)", "Busy (p75)", "Extreme (p90)"], value="Busy (p75)",
        help="Percentile of daily peak residual load (demand - renewables) across "
             "the generated dataset. Sweeps use this single day throughout.",
    )
    quantile_map = {"Typical (p50)": 0.50, "Busy (p75)": 0.75, "Extreme (p90)": 0.90}

    st.header("Sweep resolution")
    n_points = st.slider("Points per sweep", 5, 15, 8,
                          help="More points = smoother curves, more solves (still fast).")

    run_btn = st.button("Run sweeps", type="primary")


@st.cache_data
def load_day(n_days_history, quantile):
    df = generate_hourly_dataset(n_days=n_days_history)
    return representative_day(df, quantile)


@st.cache_data
def run_all_sweeps(demand, renewable, coal_price, gas_price, battery_power, battery_capacity, n_points):
    fleet = thermal_fleet_spec()
    battery = battery_spec()
    battery["power_mw"] = battery_power
    battery["capacity_mwh"] = battery_capacity

    coal_sweep = sweep_fuel_price(fleet, battery, demand, renewable, fuel="coal",
                                   price_range=np.linspace(1.0, 8.0, n_points), other_fuel_price=gas_price)
    gas_sweep = sweep_fuel_price(fleet, battery, demand, renewable, fuel="gas",
                                  price_range=np.linspace(2.0, 12.0, n_points), other_fuel_price=coal_price)
    power_sweep = sweep_battery_param(fleet, battery, demand, renewable, "power_mw",
                                       np.linspace(0, 150, n_points))
    capacity_sweep = sweep_battery_param(fleet, battery, demand, renewable, "capacity_mwh",
                                          np.linspace(0, 500, n_points))
    return fleet, coal_sweep, gas_sweep, power_sweep, capacity_sweep


demand, renewable, day_label = load_day(baseline("n_days_history"), quantile_map[day_type])
st.caption(f"Representative day: {day_label}")

if run_btn or "sweeps" not in st.session_state:
    with st.spinner(f"Running {4 * n_points} solves..."):
        st.session_state["sweeps"] = run_all_sweeps(
            demand, renewable, baseline("coal_price"), baseline("gas_price"),
            baseline("battery_power"), baseline("battery_capacity"), n_points,
        )

fleet, coal_sweep, gas_sweep, power_sweep, capacity_sweep = st.session_state["sweeps"]
gen_names = list(fleet["name"])

st.subheader("Fuel prices")
col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_sweep_cost(coal_sweep, "Coal price ($/MMBtu)", "Cost vs. coal price",
                               baseline_value=baseline("coal_price")))
    st.pyplot(plot_sweep_generation_mix(coal_sweep, gen_names, "Coal price ($/MMBtu)",
                                         "Generation mix vs. coal price"))
with col2:
    st.pyplot(plot_sweep_cost(gas_sweep, "Gas price ($/MMBtu)", "Cost vs. gas price",
                               baseline_value=baseline("gas_price")))
    st.pyplot(plot_sweep_generation_mix(gas_sweep, gen_names, "Gas price ($/MMBtu)",
                                         "Generation mix vs. gas price"))

st.subheader("Battery parameters")
col3, col4 = st.columns(2)
with col3:
    st.pyplot(plot_sweep_cost(power_sweep, "Battery power (MW)", "Cost vs. battery power rating",
                               baseline_value=baseline("battery_power")))
    st.line_chart(power_sweep.set_index("value")[["battery_throughput_mwh"]])
    st.caption("Battery energy throughput (charge + discharge, MWh/day) vs. power rating.")
with col4:
    st.pyplot(plot_sweep_cost(capacity_sweep, "Battery capacity (MWh)", "Cost vs. battery capacity",
                               baseline_value=baseline("battery_capacity")))
    st.line_chart(capacity_sweep.set_index("value")[["battery_throughput_mwh"]])
    st.caption("Battery energy throughput (charge + discharge, MWh/day) vs. capacity.")

st.markdown(
    "**Reading these:** cost curves that flatten out show diminishing returns -- e.g. "
    "battery power typically saturates once it's no longer the binding constraint on "
    "how fast the battery can charge/discharge within a given capacity. The generation-mix "
    "area charts are what actually show a merit-order shift (one unit's share shrinking "
    "as another's grows), which the cost number alone doesn't reveal."
)
