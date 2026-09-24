"""
Market Prices & Battery Arbitrage page.

Two distinct things, kept separate on purpose:

1. Settlement reporting -- a simulated day-ahead price series applied AFTER
   the fact to the REALIZED dispatch (the schedule actually settled against
   actual renewables). This changes nothing about how units are dispatched;
   it just asks what that already-cost-minimal schedule would have been
   worth at market price. See analysis.generator_economics / battery_economics.

2. Battery arbitrage -- a controlled, independent pair of solves (baseline
   vs. price-aware) against the SAME actual demand/renewable, isolating
   what changes when the battery's objective includes a price incentive on
   top of production-cost minimization. Kept separate from (1) so the
   forecast-uncertainty story (covered on the main page) doesn't get mixed
   in with the arbitrage story.
"""

import numpy as np
import pandas as pd
import streamlit as st

from data_gen import thermal_fleet_spec, battery_spec, simulate_price_series
from unit_commitment import UnitCommitmentModel
from analysis import (
    fuel_adjusted_fleet, residual_load, generator_economics, battery_economics,
    plot_price_and_residual, plot_battery_pnl, plot_battery_soc,
)
import scenario

st.set_page_config(page_title="Market Prices & Battery Arbitrage", layout="wide")
st.title("Market Prices & Battery Arbitrage")
st.markdown(
    "The rest of this app minimizes **production cost** to serve fixed demand -- "
    "there's no market price anywhere in it, so the battery has no notion of "
    "'expensive' or 'cheap' electricity, only physical constraints. This page adds "
    "a simulated price series and asks two separate questions: *what would the "
    "existing schedule have been worth at market price?* and *does giving the "
    "battery an explicit price incentive change what it does?*"
)


def get_baseline_scenario():
    if "result" in st.session_state:
        return st.session_state["result"]
    with st.spinner("No scenario from the main page yet -- running one with default settings..."):
        return scenario.run_pipeline(
            scenario.DEFAULTS["quantile"], scenario.DEFAULTS["n_days_history"],
            scenario.DEFAULTS["battery_power"], scenario.DEFAULTS["battery_capacity"],
            scenario.DEFAULTS["coal_price"], scenario.DEFAULTS["gas_price"],
        )


r = get_baseline_scenario()
if "result" not in st.session_state:
    st.info("Showing a default scenario. Visit the main page first to use your own slider settings.")

fleet = r["fleet"]
battery_cfg = r["battery_spec"]
hours = np.arange(24)

with st.sidebar:
    st.header("Price model")
    st.caption(
        "The simulated price is a merit-order proxy: marginal cost of the last unit "
        "needed to cover residual load, plus noise. It's exogenous -- computed "
        "independently of the actual UC solve -- so the battery has a genuine "
        "external signal to react to rather than one derived from its own schedule."
    )
    oversupply_floor = st.slider("Oversupply price floor ($/MWh)", -50.0, 0.0, -15.0, step=5.0)
    scarcity_cap = st.slider("Scarcity price cap ($/MWh)", 100.0, 400.0, 220.0, step=10.0)
    noise_std = st.slider("Price noise (std dev, $/MWh)", 0.0, 15.0, 4.0, step=1.0)
    price_seed = st.number_input("Price randomness seed", value=7, step=1)

# ---------------------------------------------------------------------------
# 1. Settlement reporting: price applied to the REALIZED dispatch
# ---------------------------------------------------------------------------
st.subheader("Settlement: what the realized dispatch was worth")

if r["realized"] is None or r["realized"].status != "optimal":
    st.warning("No realized (settled) dispatch available for this scenario -- the planned schedule wasn't optimal.")
else:
    realized = r["realized"]
    demand, actual_renewable = r["demand"], r["actual_renewable"]
    rng = np.random.default_rng(int(price_seed))
    price = simulate_price_series(demand, actual_renewable, fleet, oversupply_floor, scarcity_cap, noise_std, rng)
    resid = residual_load(demand, actual_renewable)

    st.pyplot(plot_price_and_residual(hours, price, resid))

    st.markdown(
        "**Per-generator economics.** The marginal (most expensive committed) unit "
        "should show close to zero margin, and cheaper inframarginal units should "
        "show a healthy one -- that's standard merit-order economics, and a sanity "
        "check that the price series is behaving sensibly."
    )
    gen_econ = generator_economics(realized.dispatch, fleet, price)
    gen_econ_display = gen_econ.copy()
    for col in ["revenue", "fuel_var_om_cost", "startup_cost", "margin"]:
        gen_econ_display[col] = gen_econ_display[col].round(0)
    gen_econ_display["energy_mwh"] = gen_econ_display["energy_mwh"].round(1)
    st.dataframe(gen_econ_display, hide_index=True)

    bat_econ = battery_economics(realized.battery, price)
    col1, col2 = st.columns(2)
    col1.metric("Battery net P&L (settled day)", f"${bat_econ['net_pnl'].sum():,.0f}")
    col2.metric("System production cost (realized)", f"${r['realized_cost']:,.0f}")
    st.pyplot(plot_battery_pnl(hours, bat_econ, price))

st.divider()

# ---------------------------------------------------------------------------
# 2. Battery arbitrage: controlled baseline-vs-price-aware comparison
# ---------------------------------------------------------------------------
st.subheader("Battery arbitrage: does a price incentive change behavior?")
st.markdown(
    "A controlled comparison, independent of the settlement section above: **same** "
    "actual demand and renewable output, **same** fleet and battery, solved twice -- "
    "once with the battery blind to price (today's default), once with "
    "`price[t]*charge[t]` added as a cost and `price[t]*discharge[t]` as a revenue "
    "credit in the objective. This is a blended objective (production cost minus "
    "battery arbitrage revenue), not a full merchant-market reformulation -- so "
    "compare **production cost** (recomputed from dispatch, not the solver's raw "
    "`total_cost`, which mixes in the arbitrage term) rather than total_cost directly."
)

run_arb = st.button("Run arbitrage comparison", type="primary")

if run_arb or "arb_result" not in st.session_state:
    with st.spinner("Solving baseline and arbitrage-aware schedules..."):
        demand, actual_renewable = r["demand"], r["actual_renewable"]
        rng = np.random.default_rng(int(price_seed))
        price = simulate_price_series(demand, actual_renewable, fleet, oversupply_floor, scarcity_cap, noise_std, rng)

        model_base = UnitCommitmentModel(fleet, battery_cfg, T=24)
        res_base = model_base.build_and_solve(demand, actual_renewable)
        model_arb = UnitCommitmentModel(fleet, battery_cfg, T=24)
        res_arb = model_arb.build_and_solve(demand, actual_renewable, price=price)

        st.session_state["arb_result"] = dict(price=price, res_base=res_base, res_arb=res_arb)

price = st.session_state["arb_result"]["price"]
res_base = st.session_state["arb_result"]["res_base"]
res_arb = st.session_state["arb_result"]["res_arb"]

if res_base.status != "optimal" or res_arb.status != "optimal":
    st.warning("One of the two solves was not optimal -- try adjusting the price model sliders.")
else:
    econ_base = generator_economics(res_base.dispatch, fleet, price)
    econ_arb = generator_economics(res_arb.dispatch, fleet, price)
    true_cost_base = econ_base["fuel_var_om_cost"].sum() + econ_base["startup_cost"].sum()
    true_cost_arb = econ_arb["fuel_var_om_cost"].sum() + econ_arb["startup_cost"].sum()

    bat_econ_base = battery_economics(res_base.battery, price)
    bat_econ_arb = battery_economics(res_arb.battery, price)

    col1, col2, col3 = st.columns(3)
    col1.metric("True production cost (baseline)", f"${true_cost_base:,.0f}")
    col2.metric("True production cost (arbitrage-aware)", f"${true_cost_arb:,.0f}",
                delta=f"{true_cost_arb - true_cost_base:,.0f}", delta_color="inverse")
    col3.metric("Battery P&L improvement", f"${bat_econ_arb['net_pnl'].sum() - bat_econ_base['net_pnl'].sum():,.0f}",
                delta=f"baseline: ${bat_econ_base['net_pnl'].sum():,.0f} -> arbitrage: ${bat_econ_arb['net_pnl'].sum():,.0f}")

    st.markdown("**Charge/discharge timing, baseline vs. arbitrage-aware:**")
    compare_df = pd.DataFrame({
        "hour": hours, "price": price,
        "baseline_net_mw": res_base.battery["discharge_mw"].values - res_base.battery["charge_mw"].values,
        "arbitrage_net_mw": res_arb.battery["discharge_mw"].values - res_arb.battery["charge_mw"].values,
    }).set_index("hour")
    st.line_chart(compare_df[["baseline_net_mw", "arbitrage_net_mw"]])
    st.caption("Positive = net discharging, negative = net charging. Compare against the price sidebar chart above -- "
               "the arbitrage-aware line should hug 'charge when price is low, discharge when price is high' more closely.")

    st.markdown("**Battery state of charge, arbitrage-aware schedule:**")
    st.pyplot(plot_battery_soc(hours, res_arb.battery, battery_cfg["capacity_mwh"],
                                battery_cfg["soc_min_frac"], battery_cfg["soc_max_frac"]))
