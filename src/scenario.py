"""
Shared scenario-building logic, extracted from app.py so both the main
dashboard and the pages/ scripts can build the same day-ahead scenario
without importing app.py itself (which has top-level Streamlit calls that
would re-execute on import). Kept plain/Streamlit-free like analysis.py --
caching decorators are applied where these are called, not here.
"""

from data_gen import generate_hourly_dataset, thermal_fleet_spec, battery_spec
from forecasting import forecast_next_day
from unit_commitment import UnitCommitmentModel
from analysis import fuel_adjusted_fleet

DEFAULTS = {
    "quantile": "p50", "n_days_history": 200,
    "battery_power": 60, "battery_capacity": 200,
    "coal_price": 2.20, "gas_price": 4.50,
}


def load_data(n_days_history):
    df = generate_hourly_dataset(n_days=n_days_history + 1)
    history = df.iloc[:-24].copy()
    next_day = df.iloc[-24:].copy().reset_index(drop=True)
    return history, next_day


def run_pipeline(quantile, n_days_history, battery_power, battery_capacity, coal_price, gas_price):
    history, next_day = load_data(n_days_history)
    fleet = fuel_adjusted_fleet(thermal_fleet_spec(), coal_price, gas_price)
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

    realized_cost, unserved_max, realized = None, None, None
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
        planned=planned, perfect=perfect, realized=realized,
        realized_cost=realized_cost, unserved_max=unserved_max,
        fleet=fleet, battery_spec=battery,
    )
