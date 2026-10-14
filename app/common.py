"""Shared pieces of the Streamlit app: scenario editor, cached runs, charts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

try:
    import ucsim  # noqa: F401
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ucsim.config import Scenario, list_presets
from ucsim.simulate import (
    STRATEGY_DESCRIPTIONS,
    STRATEGY_LABELS,
    Prepared,
    SimResult,
    prepare,
    run,
)

PRESETS = list_presets()


# ---------------------------------------------------------------- caching
def _data_key(sc: Scenario) -> str:
    d = sc.to_dict()
    return json.dumps({k: d[k] for k in ("time", "data", "uncertainty")}, sort_keys=True)


@st.cache_data(show_spinner=False, max_entries=8)
def _prepare(data_key: str) -> Prepared:
    return prepare(Scenario.from_dict(json.loads(data_key)))


@st.cache_data(show_spinner=False, max_entries=64)
def _run(scenario_key: str, strategy: str) -> SimResult:
    sc = Scenario.from_dict(json.loads(scenario_key))
    return run(sc, strategy, _prepare(_data_key(sc)))


def run_cached(sc: Scenario, strategy: str) -> SimResult:
    return _run(sc.key(), strategy)


# ---------------------------------------------------------------- scenario editor
def scenario_sidebar() -> Scenario:
    """Pick a preset, then adjust it. Returns the edited scenario."""
    st.sidebar.header("Scenario")
    names = list(PRESETS)
    preset = st.sidebar.selectbox("Preset", names, index=names.index("base"), key="preset")
    base = Scenario.from_yaml(PRESETS[preset])
    st.sidebar.caption(base.description)
    k = lambda name: f"{preset}:{name}"
    t, b, r, p = base.time, base.battery, base.reserves, base.penalties

    with st.sidebar.expander("Time frame", expanded=True):
        sim_start = st.date_input("Simulation start", pd.Timestamp(t.sim_start), key=k("start"),
                                  min_value=pd.Timestamp("2023-08-01"), max_value=pd.Timestamp("2025-12-01"))
        sim_days = st.slider("Days simulated", 1, 30, t.sim_days, key=k("days"))
        resolution = st.select_slider("Time step (min)", [15, 30, 60], t.resolution_min, key=k("res"))
        horizon = st.slider("Look-ahead per solve (h)", 6, 72, t.horizon_hr, step=6, key=k("horizon"))
        commit_opts = [h for h in (1, 2, 3, 4, 6, 8, 12, 24, 48, 72) if h <= horizon]
        commit = st.select_slider("Re-commit every (h)", commit_opts,
                                  min(commit_opts, key=lambda h: abs(h - min(t.commit_hr, horizon))), key=k("commit"))

    with st.sidebar.expander("Prices and storage"):
        coal = st.slider("Coal ($/MMBtu)", 1.0, 10.0, float(base.fuel_prices.get("coal", 2.2)), 0.1, key=k("coal"))
        gas = st.slider("Gas ($/MMBtu)", 1.0, 15.0, float(base.fuel_prices.get("gas", 4.5)), 0.1, key=k("gas"))
        power = st.slider("Battery power (MW)", 0, 300, int(b.power_mw), 10, key=k("bp"))
        energy = st.slider("Battery energy (MWh)", 0, 1200, int(b.energy_mwh), 25, key=k("be"))
        voll = st.select_slider("Value of lost load ($/MWh)", [1000, 2000, 5000, 10000, 20000],
                                int(p.voll), key=k("voll"))

    with st.sidebar.expander("Reserves"):
        tot = st.slider("Total reserve (% of net load)", 0, 30, int(r.total_frac * 100), key=k("rt"))
        spin = st.slider("Spinning reserve (% of net load)", 0, 20, int(r.spin_frac * 100), key=k("rs"))

    with st.sidebar.expander("Uncertainty and data"):
        n_scen = st.slider("Stochastic scenarios", 2, 30, base.uncertainty.n_scenarios, key=k("ns"))
        seed = st.number_input("Weather seed", 0, 10_000, base.data.seed, key=k("seed"))
        train = st.slider("Training history (days)", 60, 365, base.data.train_days, 15, key=k("train"))

    with st.sidebar.expander("Fleet"):
        fleet = st.data_editor(pd.DataFrame([g.__dict__ for g in base.fleet]), num_rows="dynamic",
                               hide_index=True, key=k("fleet"))

    return base.with_overrides({
        "time": {"sim_start": str(pd.Timestamp(sim_start).date()), "sim_days": sim_days,
                 "resolution_min": resolution, "horizon_hr": horizon, "commit_hr": commit},
        "fuel_prices": {"coal": coal, "gas": gas},
        "battery": {"power_mw": power, "energy_mwh": energy},
        "penalties": {"voll": voll},
        "reserves": {"total_frac": tot / 100, "spin_frac": spin / 100},
        "uncertainty": {"n_scenarios": n_scen},
        "data": {"seed": int(seed), "train_days": train},
        "fleet": fleet.dropna(subset=["name"]).to_dict("records"),
    })


def strategy_guide(expanded: bool = False) -> None:
    """Expander explaining every commitment strategy."""
    with st.expander("What do the strategies mean?", expanded=expanded):
        st.markdown(
            "Each strategy decides which units to switch on using its own view of the "
            "future. The simulator then replays what actually happened with those units, "
            "so costs are always the real outcome, not the plan."
        )
        for key, label in STRATEGY_LABELS.items():
            st.markdown(f"**{label}.** {STRATEGY_DESCRIPTIONS[key]}")


def describe(sc: Scenario) -> str:
    t = sc.time
    return (f"{t.sim_days} day(s) from {t.sim_start} · {t.resolution_min}-min steps · "
            f"{t.horizon_hr}h look-ahead, re-committed every {t.commit_hr}h")


# ---------------------------------------------------------------- charts
# Every time-series chart zooms and pans along time: scroll to zoom, drag to
# pan, double-click to reset. Legends are clickable to highlight one series.
THERMAL_COLORS = ["#5b5b5b", "#1f77b4", "#6baed6", "#e6550d", "#fd8d3c", "#9e9ac8", "#bcbddc", "#8c6d31"]


def unit_colors(sc: Scenario) -> dict[str, str]:
    return {g.name: THERMAL_COLORS[i % len(THERMAL_COLORS)] for i, g in enumerate(sc.fleet)}


def zoomable(chart):
    """Scroll/drag zoom on the time axis only, so values stay readable."""
    return chart.add_params(alt.selection_interval(bind="scales", encodings=["x"]))


def legend_highlight(field: str):
    return alt.selection_point(fields=[field], bind="legend")


def time_x(title=None):
    return alt.X("timestamp:T", title=title, axis=alt.Axis(labelOverlap=True))


def window_picker(ts: pd.DataFrame, key: str = "window") -> pd.DataFrame:
    """Date-range slider applied to every chart on the page."""
    start, end = ts.index[0].to_pydatetime(), ts.index[-1].to_pydatetime()
    if start == end:
        return ts
    lo, hi = st.slider("Show period", min_value=start, max_value=end, value=(start, end),
                       format="MMM D, HH:mm", key=key,
                       help="Narrows every chart below. You can also scroll on a chart to zoom, "
                            "drag to pan, and double-click to reset.")
    return ts.loc[lo:hi]


def _long(df: pd.DataFrame, cols: dict[str, str], value="MW") -> pd.DataFrame:
    return df[list(cols)].rename(columns=cols).reset_index().melt("timestamp", var_name="series", value_name=value)


def forecast_chart(ts: pd.DataFrame) -> alt.Chart:
    data = _long(ts, {"demand": "Demand (actual)", "plan_demand": "Demand (plan)",
                      "renewable": "Renewables (actual)", "plan_renewable": "Renewables (plan)"})
    pick = legend_highlight("series")
    return zoomable(alt.Chart(data).mark_line(clip=True).encode(
        x=time_x(), y="MW:Q",
        color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top")),
        strokeDash=alt.condition("indexof(datum.series, 'plan') >= 0", alt.value([4, 3]), alt.value([1, 0])),
        opacity=alt.condition(pick, alt.value(1), alt.value(0.15)),
        tooltip=["timestamp:T", "series:N", alt.Tooltip("MW:Q", format=",.0f")],
    ).add_params(pick)).properties(height=260)


def dispatch_chart(ts: pd.DataFrame, sc: Scenario) -> alt.LayerChart:
    df = pd.DataFrame(index=ts.index)
    df["Renewables"] = ts["renewable"] - ts["curtail"]
    for g in sc.fleet:
        df[g.name] = ts[f"p_{g.name}"]
    df["Battery"] = ts["discharge"] - ts["charge"]
    df["Unserved"] = ts["unserved"]
    order = list(df.columns)
    colors = ["#31a354"] + list(unit_colors(sc).values()) + ["#e7ba52", "#d62728"]
    data = df.reset_index().melt("timestamp", var_name="source", value_name="MW")
    data["order"] = data["source"].map(order.index)
    pick = legend_highlight("source")
    area = zoomable(alt.Chart(data).mark_area(interpolate="step-after", clip=True).encode(
        x=time_x(), y=alt.Y("MW:Q", stack=True, title="MW"),
        color=alt.Color("source:N", sort=order, title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=order, range=colors)),
        order="order:Q", opacity=alt.condition(pick, alt.value(0.9), alt.value(0.15)),
        tooltip=["timestamp:T", "source:N", alt.Tooltip("MW:Q", format=",.0f")],
    ).add_params(pick))
    line = alt.Chart(ts.reset_index()).mark_line(color="black", strokeDash=[4, 2], interpolate="step-after", clip=True).encode(
        x="timestamp:T", y="demand:Q", tooltip=["timestamp:T", alt.Tooltip("demand:Q", format=",.0f")])
    return (area + line).properties(height=320)


def commitment_chart(ts: pd.DataFrame, sc: Scenario) -> alt.Chart:
    names = [g.name for g in sc.fleet]
    colors = unit_colors(sc)
    df = ts[[f"u_{n}" for n in names]].rename(columns=lambda c: c[2:])
    data = df.reset_index().melt("timestamp", var_name="unit", value_name="on")
    data = data[data["on"] > 0.5]
    step = pd.Timedelta(minutes=sc.time.resolution_min)
    data["end"] = data["timestamp"] + step
    return zoomable(alt.Chart(data).mark_rect(clip=True).encode(
        x=time_x(), x2="end:T", y=alt.Y("unit:N", sort=names, title=None),
        color=alt.Color("unit:N", legend=None, scale=alt.Scale(domain=names, range=[colors[n] for n in names])),
        tooltip=["unit:N", "timestamp:T"],
    )).properties(height=30 * len(names) + 20)


def soc_chart(ts: pd.DataFrame) -> alt.Chart:
    return zoomable(alt.Chart(ts.reset_index()).mark_line(interpolate="step-after", clip=True).encode(
        x=time_x(), y=alt.Y("soc:Q", title="Battery (MWh)"),
        tooltip=["timestamp:T", alt.Tooltip("soc:Q", format=",.0f")],
    )).properties(height=180)


def money(x: float) -> str:
    return f"${x:,.0f}"
