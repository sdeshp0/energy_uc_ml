"""
Interactive dashboard for the energy UC + ML project.

Run with:  streamlit run src/app.py

Framing: demand (uncertain) minus renewable output (forecast, also
uncertain) leaves a residual load that thermal generation and the battery
must jointly cover. This app makes that residual, and the heterogeneous
thermal fleet that has to serve it (different heat rates, fuel costs,
ramp rates, min up/down times), visible and interactively explorable --
rather than just reporting a final cost number.
"""

import numpy as np
import pandas as pd
import streamlit as st

from data_gen import generate_hourly_dataset, thermal_fleet_spec, battery_spec
from forecasting import forecast_next_day
from unit_commitment import UnitCommitmentModel
from analysis import (
    fuel_adjusted_fleet, residual_load, thermal_and_battery_coverage,
    ramp_headroom, plot_commitment_gantt, plot_residual_load, plot_battery_soc,
)

st.set_page_config(page_title="Energy UC + ML Dashboard", layout="wide")
st.title("Day-Ahead Unit Commitment, driven by ML renewable forecasts")

with st.sidebar:
    st.header("Forecast & storage")
    quantile = st.select_slider(
        "Forecast quantile used for commitment decisions",
        options=["p10", "p50", "p90"], value="p50",
        help="p10 = conservative (assume less renewable than expected); "
             "p50 = median forecast; p90 = optimistic",
    )
    n_days_history = st.slider("Days of training history", 60, 400, 200, step=20, key="n_days_history")
    battery_power = st.slider("Battery power rating (MW)", 0, 150, 60, step=10, key="battery_power")
    battery_capacity = st.slider("Battery capacity (MWh)", 0, 500, 200, step=25, key="battery_capacity")

    st.header("Thermal fleet: fuel prices")
    coal_price = st.slider("Coal price ($/MMBtu)", 1.0, 8.0, 2.20, step=0.10, key="coal_price")
    gas_price = st.slider("Gas price ($/MMBtu)", 2.0, 12.0, 4.50, step=0.10, key="gas_price",
                           help="All three gas units (both CCGTs + the peaker) share this price.")

    run_btn = st.button("Run pipeline", type="primary")
    st.caption("See **Sensitivity Analysis** in the page nav above for parameter sweeps.")


@st.cache_data
def load_data(n_days_history):
    df = generate_hourly_dataset(n_days=n_days_history + 1)
    history = df.iloc[:-24].copy()
    next_day = df.iloc[-24:].copy().reset_index(drop=True)
    return history, next_day


def run_pipeline(quantile, n_days_history, battery_power, battery_capacity, coal_price, gas_price):
    history, next_day = load_data(n_days_history)
    fleet = fuel_adjusted_fleet(thermal_fleet_spec(), coal_price, gas_price)
    battery = battery_spec()
    battery["power_mw"] = battery_power
    battery["capacity_mwh"] = battery_capacity

    demand = next_day["demand_mw"].values
    actual_renewable = (next_day["wind_mw"] + next_day["solar_mw"]).values

    wind_fc = forecast_next_day(history, "wind_cf", next_day, capacity_mw=300)
    solar_fc = forecast_next_day(history, "solar_cf", next_day, capacity_mw=250)
    chosen_renewable = wind_fc[f"wind_{quantile}_mw"].values + solar_fc[f"solar_{quantile}_mw"].values

    model = UnitCommitmentModel(fleet, battery, T=24)
    planned = model.build_and_solve(demand, chosen_renewable)

    perfect_model = UnitCommitmentModel(fleet, battery, T=24)
    perfect = perfect_model.build_and_solve(demand, actual_renewable)

    realized_cost, unserved_max = None, None
    if planned.status == "optimal":
        committed = (planned.dispatch.pivot(index="generator", columns="hour", values="on")
                     .loc[fleet["name"]].values)
        settle_model = UnitCommitmentModel(fleet, battery, T=24)
        realized = settle_model.build_and_solve(demand, actual_renewable, fixed_commitment=committed)
        realized_cost = realized.total_cost
        unserved_max = realized.unserved.max()

    return dict(
        history=history, next_day=next_day, demand=demand, actual_renewable=actual_renewable,
        chosen_renewable=chosen_renewable, wind_fc=wind_fc, solar_fc=solar_fc,
        planned=planned, perfect=perfect, realized_cost=realized_cost, unserved_max=unserved_max,
        fleet=fleet, battery_spec=battery,
    )


if run_btn or "result" not in st.session_state:
    with st.spinner("Training forecaster and solving MILP..."):
        st.session_state["result"] = run_pipeline(
            quantile, n_days_history, battery_power, battery_capacity, coal_price, gas_price
        )

r = st.session_state["result"]
hours = np.arange(24)
fleet = r["fleet"]
dispatch = r["planned"].dispatch

col1, col2, col3 = st.columns(3)
col1.metric("Planned cost", f"${r['planned'].total_cost:,.0f}")
col2.metric("Realized cost", f"${r['realized_cost']:,.0f}" if r["realized_cost"] else "N/A",
            delta=f"{r['realized_cost'] - r['perfect'].total_cost:,.0f} vs perfect foresight"
            if r["realized_cost"] else None, delta_color="inverse")
col3.metric("Max unserved demand", f"{r['unserved_max']:.1f} MW" if r["unserved_max"] is not None else "N/A")

st.subheader("Demand vs. renewable forecast")
chart_df = pd.DataFrame({
    "hour": hours, "demand_mw": r["demand"], "actual_renewable_mw": r["actual_renewable"],
    "chosen_forecast_mw": r["chosen_renewable"],
}).set_index("hour")
st.line_chart(chart_df)

st.subheader("Residual load: what thermal + battery must cover")
st.markdown(
    "Demand minus renewable output leaves a **residual load**. Thermal generation "
    "and the battery are the only levers an operator has to meet it -- this is the "
    "actual control problem, once the (uncertain) renewable side is netted out."
)
resid = residual_load(r["demand"], r["chosen_renewable"])
coverage = thermal_and_battery_coverage(dispatch, r["planned"].battery)
fig_resid = plot_residual_load(hours, resid, coverage["thermal_total_mw"].values, coverage["battery_net_mw"].values)
st.pyplot(fig_resid)

st.subheader("Battery: state of charge")
st.markdown(
    "The battery has to plan ahead within the day: it can only discharge what it "
    "already has stored, and can only store what its remaining headroom allows. "
    "Watch how it charges during low-residual hours and discharges to cover the "
    "evening peak."
)
battery_cfg = r["battery_spec"]
fig_soc = plot_battery_soc(hours, r["planned"].battery, battery_cfg["capacity_mwh"],
                            battery_cfg["soc_min_frac"], battery_cfg["soc_max_frac"])
st.pyplot(fig_soc)

st.subheader("Thermal fleet")
st.markdown(
    "Cost is decomposed as **heat rate x fuel price + variable O&M** -- adjust fuel "
    "prices in the sidebar and watch the merit order (cheapest-to-most-expensive "
    "ordering) shift, especially between coal and gas."
)
fleet_display = fleet[[
    "name", "pmin_mw", "pmax_mw", "heat_rate_mmbtu_per_mwh", "fuel_type",
    "fuel_cost_per_mmbtu", "var_om_per_mwh", "marginal_cost",
    "ramp_mw_per_hr", "min_up_hr", "min_down_hr",
]].sort_values("marginal_cost").reset_index(drop=True)
fleet_display.columns = [
    "Generator", "Pmin (MW)", "Pmax (MW)", "Heat rate (MMBtu/MWh)", "Fuel",
    "Fuel price ($/MMBtu)", "Var O&M ($/MWh)", "Marginal cost ($/MWh)",
    "Ramp (MW/hr)", "Min up (hr)", "Min down (hr)",
]
st.dataframe(fleet_display, hide_index=True)

st.subheader("Commitment schedule")
st.markdown(
    "When a unit turns on it has to *stay* on for its minimum up-time, and once it "
    "shuts down it has to *stay* off for its minimum down-time -- that's why these "
    "blocks don't flicker on and off hour to hour."
)
fig_gantt = plot_commitment_gantt(dispatch, list(fleet["name"]))
st.pyplot(fig_gantt)

st.subheader("Dispatch stack")
pivot = dispatch.pivot(index="hour", columns="generator", values="power_mw")
st.bar_chart(pivot)

with st.expander("Ramp constraint detail (hours within 95% of a unit's ramp limit)"):
    rh = ramp_headroom(dispatch, fleet)
    near = rh[rh["near_limit"]].reset_index(drop=True)
    if len(near) == 0:
        st.write("No hours this run were close to binding on ramp limits.")
    else:
        st.dataframe(near)

with st.expander("Scenario detail (raw dispatch table)"):
    st.dataframe(dispatch[dispatch["on"] == 1].reset_index(drop=True))

st.caption(
    "Tip: switch the forecast quantile in the sidebar from p50 to p10 and watch the "
    "realized cost drop toward the perfect-foresight number -- that's the core finding "
    "of this project: a conservative forecast can beat a more 'accurate' one once you "
    "account for the asymmetric cost of under-committing thermal capacity."
)
