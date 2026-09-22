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

from unit_commitment import UnitCommitmentModel


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


def plot_battery_soc(hours: np.ndarray, battery: pd.DataFrame, capacity_mwh: float,
                      soc_min_frac: float, soc_max_frac: float):
    """State of charge over the day (top) plus charge/discharge power (bottom),
    so you can see e.g. the battery running low late in the day, not just its
    net contribution folded into the residual-load chart."""
    fig, (ax_soc, ax_power) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True,
                                            gridspec_kw={"height_ratios": [1, 1]})

    ax_soc.fill_between(hours, 0, battery["soc_mwh"], step="mid", color="#805ad5", alpha=0.3)
    ax_soc.plot(hours, battery["soc_mwh"], color="#805ad5", linewidth=2, drawstyle="steps-mid",
                label="State of charge (MWh)")
    ax_soc.axhline(soc_min_frac * capacity_mwh, color="gray", linestyle=":", linewidth=1,
                   label=f"Min SoC ({soc_min_frac:.0%} of capacity)")
    ax_soc.axhline(soc_max_frac * capacity_mwh, color="gray", linestyle="--", linewidth=1,
                   label=f"Max SoC ({soc_max_frac:.0%} of capacity)")
    ax_soc.set_ylabel("MWh")
    ax_soc.set_title("Battery state of charge")
    ax_soc.legend(loc="upper right", fontsize=8)
    ax_soc.set_ylim(0, capacity_mwh * 1.05)

    ax_power.bar(hours, battery["discharge_mw"], color="#38a169", label="Discharge (MW)", width=0.8)
    ax_power.bar(hours, -battery["charge_mw"], color="#e53e3e", label="Charge (MW)", width=0.8)
    ax_power.axhline(0, color="gray", linewidth=0.8)
    ax_power.set_xlabel("Hour")
    ax_power.set_ylabel("MW")
    ax_power.set_title("Battery charge / discharge")
    ax_power.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sensitivity sweeps
# ---------------------------------------------------------------------------

def representative_day(df: pd.DataFrame, peak_residual_quantile: float = 0.75) -> tuple[np.ndarray, np.ndarray, str]:
    """Pick a single day out of a multi-day synthetic dataset whose peak residual
    load (demand - renewable) sits near the given quantile across all days --
    e.g. 0.5 for a 'typical' day, 0.75 for a 'busy' day. Returns (demand, renewable, label)."""
    d = df.copy()
    d["renewable_mw"] = d["wind_mw"] + d["solar_mw"]
    d["residual_mw"] = d["demand_mw"] - d["renewable_mw"]
    d["day"] = d["timestamp"].dt.date
    daily_peak = d.groupby("day")["residual_mw"].max()
    target = daily_peak.quantile(peak_residual_quantile)
    pick_day = (daily_peak - target).abs().idxmin()
    day_df = d[d["day"] == pick_day].reset_index(drop=True)
    label = f"{pick_day} (peak residual {daily_peak[pick_day]:.0f} MW, ~p{int(peak_residual_quantile*100)})"
    return day_df["demand_mw"].values, day_df["renewable_mw"].values, label


def run_sweep(base_fleet: pd.DataFrame, base_battery: dict, demand: np.ndarray, renewable: np.ndarray,
              apply_fn, values: list, T: int = 24) -> pd.DataFrame:
    """Generic sweep engine: for each v in values, apply_fn(base_fleet, base_battery, v)
    returns a (fleet, battery) pair to solve with. Records cost, curtailment, unserved
    demand, battery throughput, and per-generator energy for every point.

    apply_fn is the only thing that changes between a fuel-price sweep and a
    battery-parameter sweep -- everything else (solving, metric extraction) is shared.
    """
    rows = []
    gen_names = list(base_fleet["name"])
    for v in values:
        fleet, battery = apply_fn(base_fleet, base_battery, v)
        model = UnitCommitmentModel(fleet, battery, T=T)
        res = model.build_and_solve(demand, renewable)
        row = {"value": v, "status": res.status}
        if res.status == "optimal":
            energy_by_gen = res.dispatch.groupby("generator")["power_mw"].sum()
            row.update({
                "total_cost": res.total_cost,
                "curtailment_mwh": res.curtailment.sum(),
                "unserved_max_mw": res.unserved.max(),
                "battery_throughput_mwh": res.battery["charge_mw"].sum() + res.battery["discharge_mw"].sum(),
            })
            for g in gen_names:
                row[f"{g}_mwh"] = energy_by_gen.get(g, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_fuel_price(base_fleet: pd.DataFrame, base_battery: dict, demand: np.ndarray, renewable: np.ndarray,
                      fuel: str, price_range: np.ndarray, other_fuel_price: float) -> pd.DataFrame:
    """Sweep one fuel's price (coal or gas) while holding the other fixed at other_fuel_price."""
    def apply_fn(fleet, battery, v):
        coal_p = v if fuel == "coal" else other_fuel_price
        gas_p = v if fuel == "gas" else other_fuel_price
        return fuel_adjusted_fleet(fleet, coal_p, gas_p), battery
    return run_sweep(base_fleet, base_battery, demand, renewable, apply_fn, list(price_range))


def sweep_battery_param(base_fleet: pd.DataFrame, base_battery: dict, demand: np.ndarray, renewable: np.ndarray,
                         param: str, value_range: np.ndarray) -> pd.DataFrame:
    """Sweep one battery parameter (e.g. 'power_mw', 'capacity_mwh', 'efficiency')
    while holding the rest of the battery spec fixed."""
    def apply_fn(fleet, battery, v):
        return fleet, {**battery, param: v}
    return run_sweep(base_fleet, base_battery, demand, renewable, apply_fn, list(value_range))


def plot_sweep_cost(sweep_df: pd.DataFrame, x_label: str, title: str, baseline_value: float | None = None):
    """Total cost (and, if present, curtailment) vs. the swept parameter."""
    ok = sweep_df[sweep_df["status"] == "optimal"]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(ok["value"], ok["total_cost"], "o-", color="#2b6cb0", label="Total cost")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Total cost ($)", color="#2b6cb0")
    ax1.tick_params(axis="y", labelcolor="#2b6cb0")
    if baseline_value is not None:
        ax1.axvline(baseline_value, color="gray", linestyle=":", linewidth=1, label="Current slider value")

    ax2 = ax1.twinx()
    ax2.plot(ok["value"], ok["curtailment_mwh"], "s--", color="#38a169", alpha=0.7, label="Curtailment")
    ax2.set_ylabel("Curtailment (MWh)", color="#38a169")
    ax2.tick_params(axis="y", labelcolor="#38a169")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    ax1.set_title(title)
    fig.tight_layout()
    return fig


def plot_sweep_generation_mix(sweep_df: pd.DataFrame, gen_names: list[str], x_label: str, title: str):
    """Stacked area chart: how the generation mix (MWh/day per unit) shifts across the sweep --
    this is what actually shows a merit-order flip, not just the cost number moving."""
    ok = sweep_df[sweep_df["status"] == "optimal"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cols = [f"{g}_mwh" for g in gen_names if f"{g}_mwh" in ok.columns]
    ax.stackplot(ok["value"], [ok[c] for c in cols], labels=[c.replace("_mwh", "") for c in cols], alpha=0.85)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Energy (MWh/day)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig
