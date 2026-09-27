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
import streamlit as st

import cached_forecasts
from analysis import (
    residual_load, thermal_and_battery_coverage,
    ramp_headroom, plot_commitment_gantt, plot_residual_load, plot_battery_soc,
    plot_demand_vs_renewable_forecast, plot_dispatch_stack,
)
import scenario

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


def run_pipeline(quantile, n_days_history, battery_power, battery_capacity, coal_price, gas_price):
    prefetched = cached_forecasts.get_day_and_forecasts(n_days_history)
    return scenario.run_pipeline(quantile, n_days_history, battery_power, battery_capacity,
                                  coal_price, gas_price, prefetched=prefetched)


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
fig_demand = plot_demand_vs_renewable_forecast(hours, r["demand"], r["actual_renewable"],
                                                r["chosen_renewable"], quantile.upper())
st.pyplot(fig_demand)
st.caption(
    f"The {quantile.upper()} forecast is what the commitment decision below is actually "
    "based on -- compare it against the actual renewable line to see how much the "
    "chosen quantile over- or under-estimates on this particular day."
)

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
st.caption(
    "The dashed line and the solid thermal line should track closely, with the gold "
    "bars (battery net discharge/charge) making up the difference -- that's the battery "
    "actively smoothing what thermal generation alone would otherwise have to chase."
)

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
st.caption(
    "Top panel: state of charge should stay between the dotted (min) and dashed (max) "
    "bounds throughout. Bottom panel: green bars are discharge, red bars are charge -- "
    "these are what produce the top panel's curve hour by hour."
)

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
st.caption(
    "Each row is one generator; a blue block means that generator is committed (on) "
    "for that hour, green marks the specific hour it started up. Blocks that never "
    "appear (e.g. an idle CCGT) simply weren't needed this day."
)

st.subheader("Dispatch stack")
fig_dispatch = plot_dispatch_stack(dispatch, r["demand"], list(fleet["name"]))
st.pyplot(fig_dispatch)
st.caption(
    "Stacked bars are thermal output only, by generator; the dashed line is total "
    "demand. The gap between the top of the stack and the demand line is covered by "
    "renewables and the battery, which aren't thermal generators and so aren't stacked "
    "here -- see the residual-load chart above for how those two fill that gap."
)

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
