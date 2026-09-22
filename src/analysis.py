"""
Analysis helpers that make the thermal fleet + battery's role explicit:
given demand and renewable output, the residual load

    residual[t] = demand[t] - renewable[t]

is what thermal generation and the battery jointly have to cover. This
module is kept Streamlit-free on purpose so the logic can be unit-tested
and reused (e.g. from pipeline.py) without needing the app running.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def fuel_adjusted_fleet(fleet: pd.DataFrame, coal_price: float, gas_price: float) -> pd.DataFrame:
    """Recompute marginal_cost from heat_rate x fuel price + var_om, at the given fuel prices."""
    fleet = fleet.copy()
    price_map = {"coal": coal_price, "gas": gas_price}
    fleet["fuel_cost_per_mmbtu"] = fleet["fuel_type"].map(price_map)
    fleet["marginal_cost"] = (
        fleet["heat_rate_mmbtu_per_mwh"] * fleet["fuel_cost_per_mmbtu"] + fleet["var_om_per_mwh"]
    )
    return fleet


def residual_load(demand: np.ndarray, renewable_mw: np.ndarray) -> np.ndarray:
    """What thermal + battery must jointly cover, hour by hour. Can go negative
    (renewable oversupply) -- that's when curtailment or battery charging kicks in."""
    return demand - renewable_mw


def thermal_and_battery_coverage(dispatch: pd.DataFrame, battery: pd.DataFrame, T: int = 24) -> pd.DataFrame:
    """Per-hour thermal total output and net battery contribution (discharge - charge),
    for comparing against residual_load."""
    thermal_total = dispatch.groupby("hour")["power_mw"].sum().reindex(range(T), fill_value=0.0)
    battery_net = (battery.set_index("hour")["discharge_mw"] - battery.set_index("hour")["charge_mw"]) \
        .reindex(range(T), fill_value=0.0)
    return pd.DataFrame({"thermal_total_mw": thermal_total.values, "battery_net_mw": battery_net.values})


def commitment_matrix(dispatch: pd.DataFrame, fleet_order: list[str], T: int = 24) -> np.ndarray:
    """(G, T) 0/1 array of commitment status, in fleet_order."""
    pivot = dispatch.pivot(index="generator", columns="hour", values="on").reindex(
        index=fleet_order, columns=range(T), fill_value=0
    )
    return pivot.values


def ramp_headroom(dispatch: pd.DataFrame, fleet: pd.DataFrame, T: int = 24) -> pd.DataFrame:
    """For each generator/hour transition, how much of the ramp limit was used.
    Flags hours where the unit is within 5% of its ramp limit -- i.e. the ramp
    constraint is actually binding, not just present in the model."""
    rows = []
    for _, gen in fleet.iterrows():
        g = gen["name"]
        ramp_limit = gen["ramp_mw_per_hr"]
        power = dispatch[dispatch["generator"] == g].sort_values("hour")["power_mw"].values
        for t in range(1, T):
            delta = power[t] - power[t - 1]
            pct_of_limit = abs(delta) / ramp_limit if ramp_limit > 0 else 0.0
            rows.append({
                "generator": g, "hour": t, "delta_mw": delta,
                "ramp_limit_mw": ramp_limit, "pct_of_ramp_limit": pct_of_limit,
                "near_limit": pct_of_limit >= 0.95,
            })
    return pd.DataFrame(rows)


def plot_commitment_gantt(dispatch: pd.DataFrame, fleet_order: list[str], T: int = 24):
    """Matplotlib Gantt-style chart: one row per generator, colored blocks for on/startup.

    Startup/shutdown transitions are derived directly from the on/off commitment
    matrix (not from the solved s/v indicator variables) -- those indicators are
    only loosely constrained on already-off hours (nothing in the objective
    penalizes them), so trusting them directly can flag spurious "shutdowns" on
    hours where the unit was never on. The transition-from-`u` reading is
    unambiguous and matches what the operator actually sees.
    """
    mat = commitment_matrix(dispatch, fleet_order, T)

    fig, ax = plt.subplots(figsize=(11, 0.6 * len(fleet_order) + 1))
    on_color, startup_color = "#2b6cb0", "#38a169"
    for gi, g in enumerate(fleet_order):
        for t in range(T):
            if mat[gi, t]:
                is_startup = mat[gi, t] == 1 and (t == 0 or mat[gi, t - 1] == 0)
                color = startup_color if is_startup else on_color
                ax.barh(gi, 1, left=t, height=0.7, color=color, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(fleet_order)))
    ax.set_yticklabels(fleet_order)
    ax.set_xlabel("Hour")
    ax.set_xlim(0, T)
    ax.set_title("Commitment schedule (blue = on, green = startup hour)")
    ax.invert_yaxis()
    legend_handles = [
        mpatches.Patch(color=on_color, label="Committed (on)"),
        mpatches.Patch(color=startup_color, label="Startup hour"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_residual_load(hours: np.ndarray, residual: np.ndarray, thermal_total: np.ndarray,
                        battery_net: np.ndarray, curtailment: np.ndarray | None = None):
    """The core framing chart: residual load (demand - renewables) vs. what thermal +
    battery actually supplied to cover it."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(hours, residual, "k--", linewidth=2, label="Residual load (demand - renewables)")
    ax.plot(hours, thermal_total, color="#2b6cb0", linewidth=2, label="Thermal total output")
    ax.bar(hours, battery_net, width=0.5, color="gold", alpha=0.8, label="Battery net (discharge - charge)")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Hour")
    ax.set_ylabel("MW")
    ax.set_title("Residual load: what thermal generation + battery must jointly cover")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig
