"""Compare strategies on one scenario, or sweep one parameter."""

import altair as alt
import pandas as pd
import streamlit as st
from common import STRATEGY_LABELS, describe, run_cached, scenario_sidebar

st.set_page_config(page_title="Compare", layout="wide")
st.title("Compare")

sc = scenario_sidebar()
st.caption(describe(sc))

SWEEPS = {
    "Gas price ($/MMBtu)": ("fuel_prices", "gas", "4, 6, 8, 10, 12"),
    "Coal price ($/MMBtu)": ("fuel_prices", "coal", "1.5, 2.2, 3, 4, 5"),
    "Battery power (MW)": ("battery", "power_mw", "0, 30, 60, 100, 150"),
    "Battery energy (MWh)": ("battery", "energy_mwh", "0, 100, 200, 400, 800"),
    "Total reserve (share of net load)": ("reserves", "total_frac", "0, 0.05, 0.1, 0.15, 0.2"),
    "Look-ahead (h)": ("time", "horizon_hr", "24, 36, 48"),
    "Value of lost load ($/MWh)": ("penalties", "voll", "1000, 5000, 20000"),
}

mode = st.radio("Compare", ["Strategies", "Parameter sweep"], horizontal=True)
if mode == "Strategies":
    chosen = st.multiselect("Strategies", list(STRATEGY_LABELS), default=sc.strategies,
                            format_func=STRATEGY_LABELS.get)
    runs = {STRATEGY_LABELS[s]: (sc, s) for s in chosen}
else:
    col1, col2, col3 = st.columns([2, 2, 2])
    label = col1.selectbox("Parameter", list(SWEEPS))
    section, field, default = SWEEPS[label]
    values = [float(v) for v in col2.text_input("Values", default, key=label).split(",") if v.strip()]
    strategy = col3.selectbox("Strategy", list(STRATEGY_LABELS), index=2, format_func=STRATEGY_LABELS.get)
    runs = {}
    for v in values:
        val = int(v) if section == "time" else v
        over = {section: {field: val}}
        if section == "time":
            over["time"]["commit_hr"] = min(sc.time.commit_hr, val)
        runs[f"{v:g}"] = (sc.with_overrides(over), strategy)

if not runs:
    st.stop()

results, bar = {}, st.progress(0.0, "Running...")
for i, (name, (scen, strat)) in enumerate(runs.items()):
    bar.progress(i / len(runs), f"Running {name}...")
    results[name] = run_cached(scen, strat)
bar.empty()

table = pd.DataFrame({k: r.summary for k, r in results.items()}).T
table.index.name = "run"
st.dataframe(table.style.format({
    "total_cost": "${:,.0f}", "operating_cost": "${:,.0f}", "penalty_cost": "${:,.0f}",
    "cost_per_mwh": "${:,.2f}", "unserved_mwh": "{:,.1f}", "max_unserved_mw": "{:,.1f}",
    "hours_with_unserved": "{:,.2f}", "curtailed_mwh": "{:,.1f}", "reserve_shortfall_mwh": "{:,.1f}",
    "starts": "{:.0f}", "runtime_s": "{:.1f}"}))

order = list(results)
costs = table[["operating_cost", "penalty_cost"]].rename(
    columns={"operating_cost": "Operating", "penalty_cost": "Penalties"}).reset_index().melt(
    "run", var_name="part", value_name="cost")
left, right = st.columns(2)
left.altair_chart(alt.Chart(costs).mark_bar().encode(
    x=alt.X("run:N", sort=order, title=None), y=alt.Y("cost:Q", title="Cost ($)"),
    color=alt.Color("part:N", title=None, legend=alt.Legend(orient="top"))).properties(
    title="Total cost", height=300), width="stretch")
right.altair_chart(alt.Chart(table.reset_index()).mark_bar().encode(
    x=alt.X("run:N", sort=order, title=None), y=alt.Y("unserved_mwh:Q", title="MWh")).properties(
    title="Unserved energy", height=300), width="stretch")

daily = pd.concat({k: r.daily()["total_cost"].cumsum() for k, r in results.items()}, names=["run", "date"])
st.altair_chart(alt.Chart(daily.rename("cost").reset_index()).mark_line(point=True).encode(
    x=alt.X("date:T", title=None), y=alt.Y("cost:Q", title="Cumulative cost ($)"),
    color=alt.Color("run:N", sort=order, title=None, legend=alt.Legend(orient="top"))).properties(
    title="Cumulative cost", height=300), width="stretch")
