"""Run one scenario with one strategy and inspect the result.

    streamlit run app/Run_Scenario.py
"""

import streamlit as st
from common import (
    STRATEGY_DESCRIPTIONS,
    STRATEGY_LABELS,
    commitment_chart,
    describe,
    dispatch_chart,
    forecast_chart,
    money,
    run_cached,
    scenario_sidebar,
    soc_chart,
    strategy_guide,
)

st.set_page_config(page_title="Unit commitment simulator", layout="wide")
st.title("Unit commitment simulator")

sc = scenario_sidebar()
strategy = st.radio("Commitment strategy", list(STRATEGY_LABELS), format_func=STRATEGY_LABELS.get,
                    index=list(STRATEGY_LABELS).index("p50"), horizontal=True)
st.info(STRATEGY_DESCRIPTIONS[strategy])
strategy_guide()
st.caption(describe(sc))

with st.spinner("Forecasting and solving..."):
    res = run_cached(sc, strategy)
s, ts = res.summary, res.timeseries

c = st.columns(5)
c[0].metric("Total cost", money(s["total_cost"]))
c[1].metric("Operating cost", money(s["operating_cost"]))
c[2].metric("Penalties", money(s["penalty_cost"]), help="Unserved energy, missed reserve, over-generation")
c[3].metric("Unserved energy", f"{s['unserved_mwh']:,.1f} MWh")
c[4].metric("Unit starts", s["starts"])

st.subheader("Plan vs. actual")
st.altair_chart(forecast_chart(ts), width="stretch")

st.subheader("Realized dispatch")
st.altair_chart(dispatch_chart(ts, sc), width="stretch")

left, right = st.columns([3, 2])
with left:
    st.subheader("Commitment")
    st.altair_chart(commitment_chart(ts, sc), width="stretch")
with right:
    st.subheader("Battery state of charge")
    st.altair_chart(soc_chart(ts), width="stretch")

st.subheader("By day")
st.dataframe(res.daily().style.format({"total_cost": "${:,.0f}", "unserved_mwh": "{:,.1f}",
                                       "curtailed_mwh": "{:,.1f}", "starts": "{:.0f}"}))

with st.expander("Fleet and merit order"):
    st.dataframe(sc.fleet_frame().sort_values("marginal_cost"), hide_index=True)
with st.expander("Raw time series"):
    st.dataframe(ts)
    st.download_button("Download CSV", ts.to_csv().encode(), f"{sc.name}_{strategy}.csv")
