"""
Interactive dashboard for the energy UC + ML project.

Run with:  streamlit run src/app.py

Lets you pick which forecast quantile drives the day-ahead commitment
decision and immediately see the planned vs. realized cost tradeoff --
the same core finding as pipeline.py, but explorable interactively.
"""

import numpy as np
import pandas as pd
import streamlit as st

from data_gen import generate_hourly_dataset, thermal_fleet_spec, battery_spec
from forecasting import forecast_next_day
from unit_commitment import UnitCommitmentModel

st.set_page_config(page_title="Energy UC + ML Dashboard", layout="wide")
st.title("Day-Ahead Unit Commitment")
st.text("driven by ML renewable forecasts")

with st.sidebar:
    st.header("Settings")
    quantile = st.select_slider(
        "Forecast quantile used for commitment decisions",
        options=["p10", "p50", "p90"], value="p50",
        help="p10 = conservative (assume less renewable than expected); "
             "p50 = median forecast; p90 = optimistic",
    )
    n_days_history = st.slider("Days of training history", 60, 400, 200, step=20)
    battery_power = st.slider("Battery power rating (MW)", 0, 150, 60, step=10)
    battery_capacity = st.slider("Battery capacity (MWh)", 0, 500, 200, step=25)
    run_btn = st.button("Run pipeline", type="primary")


@st.cache_data
def load_data(n_days_history):
    df = generate_hourly_dataset(n_days=n_days_history + 1)
    history = df.iloc[:-24].copy()
    next_day = df.iloc[-24:].copy().reset_index(drop=True)
    return history, next_day


def run_pipeline(quantile, n_days_history, battery_power, battery_capacity):
    history, next_day = load_data(n_days_history)
    fleet = thermal_fleet_spec()
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
        fleet=fleet,
    )


if run_btn or "result" not in st.session_state:
    with st.spinner("Training forecaster and solving MILP..."):
        st.session_state["result"] = run_pipeline(quantile, n_days_history, battery_power, battery_capacity)

r = st.session_state["result"]
hours = np.arange(24)

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

st.subheader("Day-ahead dispatch (committed generators)")
dispatch = r["planned"].dispatch
pivot = dispatch.pivot(index="hour", columns="generator", values="power_mw")
st.bar_chart(pivot)

st.subheader("Scenario detail")
st.dataframe(dispatch[dispatch["on"] == 1].reset_index(drop=True))

st.caption(
    "Tip: switch the forecast quantile in the sidebar from p50 to p10 and watch the "
    "realized cost drop toward the perfect-foresight number -- that's the core finding "
    "of this project: a conservative forecast can beat a more 'accurate' one once you "
    "account for the asymmetric cost of under-committing thermal capacity."
)
