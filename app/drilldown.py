"""Drill-down views into the fleet: thermal units, renewables, battery, costs."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from common import Scenario, legend_highlight, money, time_x, unit_colors, zoomable


def render(ts: pd.DataFrame, sc: Scenario) -> None:
    """Tabs for each part of the system, all limited to the rows in `ts`."""
    tabs = st.tabs(["Thermal units", "Renewables", "Battery", "Costs"])
    with tabs[0]:
        thermal(ts, sc)
    with tabs[1]:
        renewables(ts, sc)
    with tabs[2]:
        battery(ts, sc)
    with tabs[3]:
        costs(ts, sc)


# ---------------------------------------------------------------- thermal
def unit_table(ts: pd.DataFrame, sc: Scenario) -> pd.DataFrame:
    dt = sc.time.dt
    hours = len(ts) * dt
    rows = []
    for _, g in sc.fleet_frame().iterrows():
        n = g["name"]
        p, u, starts = ts[f"p_{n}"], ts[f"u_{n}"], ts[f"start_{n}"]
        energy = p.sum() * dt
        fuel = energy * g["marginal_cost"]
        startup = starts.sum() * g["startup_cost"]
        ramps = p.diff().abs().iloc[1:][(u.iloc[1:] > 0.5) & (u.shift().iloc[1:] > 0.5)]
        rows.append({
            "Unit": n,
            "Fuel": g["fuel"],
            "Marginal cost ($/MWh)": g["marginal_cost"],
            "Energy (MWh)": energy,
            "Capacity factor": energy / (g["pmax_mw"] * hours) if hours else 0.0,
            "Hours online": u.sum() * dt,
            "Starts": int(starts.sum()),
            "Avg output when on (MW)": p[u > 0.5].mean() if (u > 0.5).any() else 0.0,
            "Peak ramp used": (ramps.max() / (g["ramp_mw_per_hr"] * dt)) if len(ramps) else 0.0,
            "Fuel + O&M cost": fuel,
            "Startup cost": startup,
            "Average cost ($/MWh)": (fuel + startup) / energy if energy > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def thermal(ts: pd.DataFrame, sc: Scenario) -> None:
    table = unit_table(ts, sc)
    st.dataframe(table.style.format({
        "Marginal cost ($/MWh)": "${:,.2f}", "Energy (MWh)": "{:,.0f}", "Capacity factor": "{:.0%}",
        "Hours online": "{:,.0f}", "Avg output when on (MW)": "{:,.0f}", "Peak ramp used": "{:.0%}",
        "Fuel + O&M cost": "${:,.0f}", "Startup cost": "${:,.0f}", "Average cost ($/MWh)": "${:,.2f}",
    }), hide_index=True)
    st.caption("Capacity factor is energy over what the unit could make at full output for the whole period. "
               "Peak ramp used is the largest hour-to-hour change while online, as a share of its ramp limit. "
               "Average cost includes startups.")

    left, right = st.columns([3, 2])
    with left:
        names = [g.name for g in sc.fleet]
        unit = st.selectbox("Unit", names, key="drill_unit")
        output, ramping = unit_charts(ts, sc, unit)
        st.altair_chart(output, width="stretch")
        st.altair_chart(ramping, width="stretch")
    with right:
        st.markdown("**Merit order**")
        st.altair_chart(merit_order_chart(ts, sc), width="stretch")
        st.caption("Each step is a unit, as wide as its capacity, as high as its marginal cost. The shaded band "
                   "is the range of net load (demand minus renewables) the thermal fleet had to cover.")


def unit_charts(ts: pd.DataFrame, sc: Scenario, unit: str) -> tuple[alt.LayerChart, alt.LayerChart]:
    g = sc.fleet_frame().set_index("name").loc[unit]
    color = unit_colors(sc)[unit]
    step = pd.Timedelta(minutes=sc.time.resolution_min)
    df = pd.DataFrame({
        "output": ts[f"p_{unit}"],
        "on": ts[f"u_{unit}"],
        "headroom": (g["pmax_mw"] * ts[f"u_{unit}"] - ts[f"p_{unit}"]).clip(lower=0),
        "start": ts[f"start_{unit}"],
        "ramp": ts[f"p_{unit}"].diff().fillna(0),
    }).reset_index()
    df["end"] = df["timestamp"] + step
    base = alt.Chart(df)
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    online = base.transform_filter("datum.on > 0.5").mark_rect(color=color, opacity=0.12, clip=True).encode(
        x=time_x(), x2="end:T")
    limits = alt.Chart(pd.DataFrame({"MW": [g["pmin_mw"], g["pmax_mw"]], "limit": ["Pmin", "Pmax"]})).mark_rule(
        strokeDash=[4, 3], color="#888").encode(y="MW:Q", tooltip=["limit:N", "MW:Q"])
    output = base.mark_line(color=color, interpolate="step-after", clip=True).encode(
        x=time_x(), y=alt.Y("output:Q", title=f"{unit} output (MW)"),
        tooltip=["timestamp:T", alt.Tooltip("output:Q", format=",.0f"), alt.Tooltip("headroom:Q", format=",.0f")])
    starts = base.transform_filter("datum.start > 0.5").mark_point(shape="triangle-up", size=80, filled=True,
                                                                   color="#2ca02c", clip=True).encode(
        x="timestamp:T", y="output:Q", tooltip=[alt.Tooltip("timestamp:T", title="Start")])
    top = (online + limits + output.add_params(zoom) + starts).properties(
        height=220, title="Output (shaded = online, ▲ = start, dashed = Pmin / Pmax)")

    ramp_lim = g["ramp_mw_per_hr"] * sc.time.dt
    ramp_rules = alt.Chart(pd.DataFrame({"MW": [ramp_lim, -ramp_lim]})).mark_rule(strokeDash=[4, 3], color="#888").encode(
        y="MW:Q")
    ramp = base.mark_rule(clip=True, color=color, strokeWidth=2).encode(
        x=time_x(), y=alt.Y("ramp:Q", title="Change per step (MW)"), y2=alt.datum(0),
        tooltip=["timestamp:T", alt.Tooltip("ramp:Q", format="+,.0f")])
    bottom = (ramp.add_params(alt.selection_interval(bind="scales", encodings=["x"])) + ramp_rules).properties(
        height=220, title=f"Ramping (dashed = ±{ramp_lim:,.0f} MW limit; startups may jump to Pmin)")
    return top, bottom


def merit_order_chart(ts: pd.DataFrame, sc: Scenario) -> alt.LayerChart:
    fleet = sc.fleet_frame().sort_values("marginal_cost")
    fleet["x0"] = fleet["pmax_mw"].cumsum() - fleet["pmax_mw"]
    fleet["x1"] = fleet["pmax_mw"].cumsum()
    colors = unit_colors(sc)
    net = (ts["demand"] - ts["renewable"] + ts["curtail"]).clip(lower=0)
    band = alt.Chart(pd.DataFrame({"lo": [net.min()], "hi": [net.max()]})).mark_rect(
        color="#999", opacity=0.15).encode(x="lo:Q", x2="hi:Q")
    bars = alt.Chart(fleet).mark_rect(stroke="white").encode(
        x=alt.X("x0:Q", title="Cumulative capacity (MW)"), x2="x1:Q",
        y=alt.Y("marginal_cost:Q", title="Marginal cost ($/MWh)"), y2=alt.datum(0),
        color=alt.Color("name:N", legend=None,
                        scale=alt.Scale(domain=list(colors), range=list(colors.values()))),
        tooltip=["name:N", "fuel:N", alt.Tooltip("pmax_mw:Q", title="Capacity (MW)"),
                 alt.Tooltip("marginal_cost:Q", format="$,.2f")])
    labels = alt.Chart(fleet).mark_text(dy=-6, fontSize=10).encode(
        x=alt.X("mid:Q"), y="marginal_cost:Q", text="name:N").transform_calculate(mid="(datum.x0 + datum.x1) / 2")
    return (band + bars + labels).properties(height=300)


# ---------------------------------------------------------------- renewables
def renewables(ts: pd.DataFrame, sc: Scenario) -> None:
    dt = sc.time.dt
    avail, curt = ts["renewable"].sum() * dt, ts["curtail"].sum() * dt
    err = ts["renewable"] - ts["plan_renewable"]
    share = (ts["renewable"] - ts["curtail"]).sum() / ts["demand"].sum()
    c = st.columns(5)
    c[0].metric("Available", f"{avail:,.0f} MWh")
    c[1].metric("Curtailed", f"{curt:,.0f} MWh", f"{curt / avail:.1%} of available" if avail else None,
                delta_color="off")
    c[2].metric("Share of demand served", f"{share:.1%}")
    c[3].metric("Plan error (MAE)", f"{err.abs().mean():,.0f} MW", help="Mean absolute difference between the "
                "renewable output the strategy planned on and what actually happened.")
    c[4].metric("Plan bias", f"{err.mean():+,.0f} MW", help="Positive: more renewable output arrived than planned.",
                delta_color="off")

    df = pd.DataFrame({"Wind": ts["wind"], "Solar": ts["solar"]}).reset_index().melt(
        "timestamp", var_name="source", value_name="MW")
    pick = legend_highlight("source")
    stack = zoomable(alt.Chart(df).mark_area(interpolate="step-after", clip=True).encode(
        x=time_x(), y=alt.Y("MW:Q", stack=True, title="Available (MW)"),
        color=alt.Color("source:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=["Wind", "Solar"], range=["#3182bd", "#fdae6b"])),
        opacity=alt.condition(pick, alt.value(0.85), alt.value(0.15)),
        tooltip=["timestamp:T", "source:N", alt.Tooltip("MW:Q", format=",.0f")]).add_params(pick))
    curt_df = ts[["curtail"]].reset_index()
    curt_marks = alt.Chart(curt_df[curt_df["curtail"] > 0.1]).mark_bar(color="#d62728", clip=True).encode(
        x="timestamp:T", y=alt.Y("curtail:Q"), tooltip=["timestamp:T", alt.Tooltip("curtail:Q", format=",.0f")])
    st.altair_chart((stack + curt_marks).properties(
        height=260, title="Wind and solar available (red bars = curtailed)"), width="stretch")

    left, right = st.columns(2)
    with left:
        plan = pd.DataFrame({"Actual": ts["renewable"], "Planned": ts["plan_renewable"]}).reset_index().melt(
            "timestamp", var_name="series", value_name="MW")
        st.altair_chart(zoomable(alt.Chart(plan).mark_line(interpolate="step-after", clip=True).encode(
            x=time_x(), y="MW:Q",
            color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=["Actual", "Planned"], range=["#31a354", "#a1d99b"])),
            strokeDash=alt.condition("datum.series == 'Planned'", alt.value([4, 3]), alt.value([1, 0])),
            tooltip=["timestamp:T", "series:N", alt.Tooltip("MW:Q", format=",.0f")],
        )).properties(height=240, title="Planned vs. actual renewables"), width="stretch")
    with right:
        st.altair_chart(alt.Chart(err.rename("error").reset_index()).mark_bar().encode(
            x=alt.X("error:Q", bin=alt.Bin(maxbins=30), title="Actual − planned (MW)"),
            y=alt.Y("count():Q", title="Steps"), tooltip=["count():Q"],
        ).properties(height=240, title="Plan error distribution"), width="stretch")

    st.altair_chart(daily_profile(ts, {"wind": "Wind", "solar": "Solar", "curtail": "Curtailed"},
                                  ["#3182bd", "#fdae6b", "#d62728"], "Average MW"
                                  ).properties(title="Average day: output by hour"), width="stretch")


def daily_profile(ts: pd.DataFrame, cols: dict[str, str], colors: list[str], title: str) -> alt.Chart:
    hour = ts.index.hour + ts.index.minute / 60
    prof = ts[list(cols)].groupby(hour).mean().rename(columns=cols)
    prof.index.name = "hour"
    data = prof.reset_index().melt("hour", var_name="series", value_name="MW")
    return alt.Chart(data).mark_line(point=True).encode(
        x=alt.X("hour:Q", title="Hour of day", scale=alt.Scale(domain=[0, 24])),
        y=alt.Y("MW:Q", title=title),
        color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=list(cols.values()), range=colors)),
        tooltip=["hour:Q", "series:N", alt.Tooltip("MW:Q", format=",.0f")],
    ).properties(height=280)


# ---------------------------------------------------------------- battery
def battery(ts: pd.DataFrame, sc: Scenario) -> None:
    b, dt = sc.battery, sc.time.dt
    if b.power_mw <= 0 or b.energy_mwh <= 0:
        st.info("This scenario has no battery.")
        return
    charged, discharged = ts["charge"].sum() * dt, ts["discharge"].sum() * dt
    cycles = discharged / (b.energy_mwh * (b.soc_max_frac - b.soc_min_frac))
    lo, hi = b.soc_min_frac * b.energy_mwh, b.soc_max_frac * b.energy_mwh
    c = st.columns(5)
    c[0].metric("Charged", f"{charged:,.0f} MWh")
    c[1].metric("Discharged", f"{discharged:,.0f} MWh")
    c[2].metric("Losses", f"{charged - discharged:,.0f} MWh",
                help="Charged minus discharged; includes any change in stored energy over the period.")
    c[3].metric("Full cycles", f"{cycles:,.1f}", help="Discharged energy over usable capacity.")
    c[4].metric("Time at a limit", f"{((ts['soc'] <= lo + 0.5) | (ts['soc'] >= hi - 0.5)).mean():.0%}",
                help="Share of steps where the battery was full or empty, so it could not help in one direction.")

    soc = zoomable(alt.Chart(ts.reset_index()).mark_area(
        interpolate="step-after", clip=True, color="#e7ba52", opacity=0.5, line={"color": "#b8860b"}).encode(
        x=time_x(), y=alt.Y("soc:Q", title="State of charge (MWh)", scale=alt.Scale(domain=[0, b.energy_mwh])),
        tooltip=["timestamp:T", alt.Tooltip("soc:Q", format=",.0f")]))
    bounds = alt.Chart(pd.DataFrame({"MWh": [lo, hi], "limit": ["Min", "Max"]})).mark_rule(
        strokeDash=[4, 3], color="#888").encode(y="MWh:Q", tooltip=["limit:N", "MWh:Q"])
    st.altair_chart((soc + bounds).properties(height=220, title="State of charge (dashed = usable range)"),
                    width="stretch")

    flow = pd.DataFrame({"Discharging": ts["discharge"], "Charging": -ts["charge"]}).reset_index().melt(
        "timestamp", var_name="direction", value_name="net")
    st.altair_chart(zoomable(alt.Chart(flow).mark_area(interpolate="step-after", clip=True).encode(
        x=time_x(), y=alt.Y("net:Q", title="MW (+ discharge, − charge)", stack=None),
        color=alt.Color("direction:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=["Discharging", "Charging"], range=["#e6550d", "#3182bd"])),
        tooltip=["timestamp:T", alt.Tooltip("net:Q", format="+,.0f")],
    )).properties(height=280, title="Charge and discharge"), width="stretch")

    left, right = st.columns(2)
    with left:
        prof = ts.assign(net=ts["discharge"] - ts["charge"], net_load=ts["demand"] - ts["renewable"])
        st.altair_chart(daily_profile(prof, {"net": "Battery net output"}, ["#e6550d"], "Average MW").properties(
            title="Average day: when the battery works"), width="stretch")
    with right:
        scatter = pd.DataFrame({"net_load": ts["demand"] - ts["renewable"], "net": ts["discharge"] - ts["charge"]})
        st.altair_chart(alt.Chart(scatter).mark_circle(opacity=0.4, color="#e6550d").encode(
            x=alt.X("net_load:Q", title="Net load: demand − renewables (MW)"),
            y=alt.Y("net:Q", title="Battery net output (MW)"),
            tooltip=[alt.Tooltip("net_load:Q", format=",.0f"), alt.Tooltip("net:Q", format="+,.0f")],
        ).properties(height=280, title="Discharges at high net load, charges at low"), width="stretch")


# ---------------------------------------------------------------- costs
COST_PARTS = {"fuel_cost": "Fuel and O&M", "startup_cost": "Startups", "battery_cost": "Battery cycling",
              "penalty_cost": "Penalties"}


def costs(ts: pd.DataFrame, sc: Scenario) -> None:
    totals = ts[list(COST_PARTS)].sum()
    c = st.columns(len(COST_PARTS))
    for col, (k, label) in zip(c, COST_PARTS.items()):
        col.metric(label, money(totals[k]))

    daily = ts[list(COST_PARTS)].groupby(ts.index.normalize()).sum().rename(columns=COST_PARTS)
    daily.index.name = "day"
    data = daily.reset_index().melt("day", var_name="part", value_name="cost")
    pick = legend_highlight("part")
    st.altair_chart(alt.Chart(data).mark_bar().encode(
        x=alt.X("yearmonthdate(day):O", title=None), y=alt.Y("cost:Q", title="Cost ($)", stack=True),
        color=alt.Color("part:N", title=None, sort=list(COST_PARTS.values()), legend=alt.Legend(orient="top")),
        opacity=alt.condition(pick, alt.value(1), alt.value(0.2)),
        tooltip=[alt.Tooltip("yearmonthdate(day):O", title="Day"), "part:N", alt.Tooltip("cost:Q", format="$,.0f")],
    ).add_params(pick).properties(height=260, title="Cost by day"), width="stretch")

    left, right = st.columns(2)
    with left:
        units = unit_table(ts, sc)[["Unit", "Fuel + O&M cost", "Startup cost"]].melt(
            "Unit", var_name="part", value_name="cost")
        st.altair_chart(alt.Chart(units).mark_bar().encode(
            y=alt.Y("Unit:N", sort="-x", title=None), x=alt.X("cost:Q", title="Cost ($)", stack=True),
            color=alt.Color("part:N", title=None, legend=alt.Legend(orient="top")),
            tooltip=["Unit:N", "part:N", alt.Tooltip("cost:Q", format="$,.0f")],
        ).properties(height=220, title="Cost by unit"), width="stretch")
    with right:
        energy = sc.time.dt * ts["demand"]
        rate = (ts["total_cost"].rolling(max(1, sc.time.steps_per_day)).sum()
                / energy.rolling(max(1, sc.time.steps_per_day)).sum()).rename("rate").reset_index()
        st.altair_chart(zoomable(alt.Chart(rate).mark_line(clip=True).encode(
            x=time_x(), y=alt.Y("rate:Q", title="$/MWh"),
            tooltip=["timestamp:T", alt.Tooltip("rate:Q", format="$,.2f")],
        )).properties(height=220, title="Cost per MWh served (trailing 24h)"), width="stretch")
