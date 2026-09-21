"""
End-to-end Phase 1 pipeline:
  1. Generate synthetic history + a "next day" (held out, with known actuals)
  2. Forecast next-day wind & solar (ML)
  3. Solve day-ahead unit commitment three ways and compare:
       a) Perfect foresight (uses actual renewable output) -- best-case lower bound on cost
       b) ML forecast (P50 median forecast) -- the actual proposed approach
       c) Naive baseline (yesterday's same-hour actuals, i.e. persistence)
  4. Re-solve with the ML (forecast) commitment schedule fixed, but actual
     renewables realized -- this exposes the real cost of forecast error
     (unserved energy penalty if the forecast was too optimistic).
  5. Save a comparison chart + CSV summary.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_gen import generate_hourly_dataset, thermal_fleet_spec, battery_spec
from forecasting import forecast_next_day
from unit_commitment import UnitCommitmentModel

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_hourly_dataset(n_days=400)
    history = df.iloc[:-24].copy()
    next_day = df.iloc[-24:].copy().reset_index(drop=True)

    fleet = thermal_fleet_spec()
    battery = battery_spec()

    demand = next_day["demand_mw"].values
    actual_renewable = (next_day["wind_mw"] + next_day["solar_mw"]).values

    # --- ML forecast (median) ---
    wind_fc = forecast_next_day(history, "wind_cf", next_day, capacity_mw=300)
    solar_fc = forecast_next_day(history, "solar_cf", next_day, capacity_mw=250)
    ml_forecast_renewable = wind_fc["wind_p50_mw"].values + solar_fc["solar_p50_mw"].values

    # --- naive baseline: persistence (yesterday's actuals at same hours) ---
    persistence_renewable = (history["wind_mw"].values[-24:] + history["solar_mw"].values[-24:])

    ml_forecast_conservative = wind_fc["wind_p10_mw"].values + solar_fc["solar_p10_mw"].values

    scenarios = {
        "Perfect foresight": actual_renewable,
        "ML forecast (P50)": ml_forecast_renewable,
        "ML forecast (P10, conservative)": ml_forecast_conservative,
        "Naive persistence": persistence_renewable,
    }

    results = {}
    for name, renewable_input in scenarios.items():
        model = UnitCommitmentModel(fleet, battery, T=24)
        res = model.build_and_solve(demand, renewable_input)
        results[name] = res
        print(f"{name:22s} -> status={res.status:10s} cost=${res.total_cost:,.0f} "
              f"unserved_max={res.unserved.max():.1f}MW curtailment={res.curtailment.sum():.1f}MWh")

    # --- realized cost: fix each plan's commitment schedule (the day-ahead decision that
    # was actually locked in), then re-settle against ACTUAL renewables. This is the fair
    # comparison -- "planned cost" under an optimistic forecast can look artificially cheap
    # because the optimizer under-commits thermal capacity it will actually need. ---
    print("\n--- Realized cost: fixed day-ahead commitment, settled against actual renewables ---")
    realized = {"Perfect foresight": results["Perfect foresight"].total_cost}
    for name in ["ML forecast (P50)", "ML forecast (P10, conservative)", "Naive persistence"]:
        planned = results[name]
        if planned.status != "optimal":
            continue
        committed = (planned.dispatch.pivot(index="generator", columns="hour", values="on")
                     .loc[fleet["name"]].values)  # (G,T) in fleet order
        model = UnitCommitmentModel(fleet, battery, T=24)
        realized_res = model.build_and_solve(demand, actual_renewable, fixed_commitment=committed)
        realized[name] = realized_res.total_cost
        gap = actual_renewable - scenarios[name]
        print(f"{name:22s}: planned cost=${planned.total_cost:,.0f}  ->  realized cost=${realized_res.total_cost:,.0f}"
              f"  (forecast MAE={np.abs(gap).mean():.1f} MW, unserved_max={realized_res.unserved.max():.1f} MW)")
    print(f"{'Perfect foresight':22s}: (reference / lower bound) cost=${realized['Perfect foresight']:,.0f}")

    # --- save comparison chart ---
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax = axes[0]
    hours = np.arange(24)
    ax.plot(hours, demand, "k--", label="Demand", linewidth=2)
    ax.plot(hours, actual_renewable, label="Actual renewable", linewidth=2)
    ax.plot(hours, ml_forecast_renewable, label="ML forecast (P50)", linewidth=2)
    ax.fill_between(hours, wind_fc["wind_p10_mw"] + solar_fc["solar_p10_mw"],
                     wind_fc["wind_p90_mw"] + solar_fc["solar_p90_mw"],
                     alpha=0.15, label="P10-P90 band")
    ax.plot(hours, persistence_renewable, ":", label="Naive persistence", linewidth=1.5)
    ax.set_ylabel("MW")
    ax.set_title("Demand vs. renewable forecast (day-ahead)")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    dispatch = results["ML forecast (P50)"].dispatch
    battery_df = results["ML forecast (P50)"].battery
    bottom = np.zeros(24)
    for g in fleet["name"]:
        gen_power = dispatch[dispatch["generator"] == g].sort_values("hour")["power_mw"].values
        ax.bar(hours, gen_power, bottom=bottom, label=g, width=0.9)
        bottom += gen_power
    ax.bar(hours, ml_forecast_renewable, bottom=bottom, label="Wind+Solar", width=0.9, color="tab:green", alpha=0.7)
    bottom += ml_forecast_renewable
    ax.bar(hours, battery_df["discharge_mw"].values, bottom=bottom, label="Battery discharge", width=0.9, color="gold")
    ax.plot(hours, demand, "k--", linewidth=2, label="Demand")
    ax.set_xlabel("Hour")
    ax.set_ylabel("MW")
    ax.set_title("Day-ahead dispatch stack (ML-forecast-driven UC)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dispatch_comparison.png", dpi=130)
    print("\nSaved chart to outputs/dispatch_comparison.png")

    # --- planned vs realized cost chart: the key finding ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    names = ["ML forecast (P50)", "ML forecast (P10, conservative)", "Naive persistence"]
    planned_vals = [results[n].total_cost for n in names]
    realized_vals = [realized[n] for n in names]
    x = np.arange(len(names))
    w = 0.35
    ax2.bar(x - w / 2, planned_vals, w, label="Planned cost (under forecast)", color="tab:orange")
    ax2.bar(x + w / 2, realized_vals, w, label="Realized cost (settled vs. actual)", color="tab:red")
    ax2.axhline(realized["Perfect foresight"], color="k", linestyle="--", linewidth=1.5,
                label="Perfect foresight (lower bound)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=9)
    ax2.set_ylabel("Cost ($)")
    ax2.set_title("Planned vs. realized cost: forecast choice matters more than accuracy")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "planned_vs_realized_cost.png", dpi=130)
    print("Saved chart to outputs/planned_vs_realized_cost.png")

    # --- save summary CSV ---
    summary = pd.DataFrame([
        {"scenario": name, "status": r.status, "planned_cost": r.total_cost,
         "realized_cost": realized.get(name, r.total_cost),
         "max_unserved_mw": r.unserved.max(), "total_curtailment_mwh": r.curtailment.sum()}
        for name, r in results.items()
    ])
    summary.to_csv(OUTPUT_DIR / "scenario_summary.csv", index=False)
    print("\n", summary.to_string(index=False))


if __name__ == "__main__":
    run()
