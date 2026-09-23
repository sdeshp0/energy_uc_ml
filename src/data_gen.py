"""
Synthetic data generator for the energy UC + ML project.

Generates:
  - An hourly historical dataset (demand, wind, solar) with realistic
    diurnal/seasonal shape + noise, suitable for training a forecasting model.
  - A small diversified thermal fleet spec (coal, CCGT, gas peaker, etc.)

Swap-in path for real data (documented in README):
  - Demand:        EIA hourly demand API, ENTSO-E transparency platform
  - Wind/Solar:    NREL Wind/Solar Integration Toolkit, renewables.ninja
  - Fleet specs:   EIA-860/923 generator-level data (heat rates, capacities)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def _diurnal_demand_shape(hour: np.ndarray) -> np.ndarray:
    """Two-peak (morning/evening) daily demand shape, normalized ~[0.6, 1.0]."""
    morning = 0.15 * np.exp(-((hour - 8) ** 2) / (2 * 2.5**2))
    evening = 0.25 * np.exp(-((hour - 19) ** 2) / (2 * 3.0**2))
    base = 0.65
    return base + morning + evening


def _solar_shape(hour: np.ndarray, day_of_year: np.ndarray) -> np.ndarray:
    """Bell-curve solar output during daylight hours, seasonal amplitude."""
    # Seasonal daylight length & intensity (peaks in summer, N. hemisphere)
    seasonal = 0.6 + 0.4 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
    daylight = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    return seasonal * daylight**1.3


def _wind_shape(n: int, day_of_year: np.ndarray) -> np.ndarray:
    """Wind: weakly seasonal (windier in winter), mean-reverting stochastic process."""
    seasonal = 0.55 + 0.2 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    # Ornstein-Uhlenbeck-ish mean-reverting noise for realistic autocorrelation
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.9 * noise[i - 1] + RNG.normal(0, 0.12)
    return np.clip(seasonal + noise, 0.02, 1.0)


def generate_hourly_dataset(n_days: int = 400, start_date: str = "2024-01-01") -> pd.DataFrame:
    """
    Returns an hourly DataFrame with columns:
      timestamp, hour, day_of_year, dow,
      demand_mw, wind_cf, solar_cf, wind_mw, solar_mw
    (cf = capacity factor in [0,1]; mw = capacity-factor * nameplate capacity)
    """
    n = n_days * 24
    timestamps = pd.date_range(start_date, periods=n, freq="h")
    hour = timestamps.hour.values
    day_of_year = timestamps.dayofyear.values
    dow = timestamps.dayofweek.values

    demand_shape = _diurnal_demand_shape(hour)
    weekend_derate = np.where(dow >= 5, 0.92, 1.0)
    demand_noise = RNG.normal(0, 0.03, n)
    demand_mw = 1000 * demand_shape * weekend_derate + 1000 * demand_noise

    solar_cf = np.clip(_solar_shape(hour, day_of_year) + RNG.normal(0, 0.04, n), 0, 1)
    wind_cf = _wind_shape(n, day_of_year)

    # Correlated weather shock ("cold snap"): a small fraction of days get a
    # simultaneous demand spike (extra heating/cooling load) AND a wind drought
    # (the still-air, high-pressure pattern that commonly accompanies a cold
    # snap) -- the classic correlated tail risk in power systems (e.g. Texas,
    # Feb 2021), and specifically the kind of event an independence assumption
    # between demand and renewable forecast errors would understate. Without
    # this, demand/wind/solar here are independent stochastic processes with
    # no shared driver, so there's nothing for a joint (non-independence)
    # scenario model to actually find -- see scenarios.py.
    cold_snap_day = RNG.random(n_days) < 0.05
    cold_snap_hourly = np.repeat(cold_snap_day, 24)
    demand_mw = demand_mw * np.where(cold_snap_hourly, 1.16, 1.0)
    wind_cf = np.clip(wind_cf * np.where(cold_snap_hourly, 0.4, 1.0), 0.02, 1.0)

    demand_mw = np.clip(demand_mw, 400, None)

    WIND_CAPACITY_MW = 300
    SOLAR_CAPACITY_MW = 250

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hour": hour,
            "day_of_year": day_of_year,
            "dow": dow,
            "demand_mw": demand_mw,
            "wind_cf": wind_cf,
            "solar_cf": solar_cf,
            "wind_mw": wind_cf * WIND_CAPACITY_MW,
            "solar_mw": solar_cf * SOLAR_CAPACITY_MW,
        }
    )
    return df


def thermal_fleet_spec(coal_price_per_mmbtu: float = 2.20,
                        gas_price_per_mmbtu: float = 4.50) -> pd.DataFrame:
    """
    A small, deliberately diversified thermal fleet: baseload coal, mid-merit
    CCGT, flexible CCGT, and a fast gas peaker.

    Cost is decomposed as heat_rate (MMBtu/MWh, how much fuel it takes to
    make a MWh -- efficiency) x fuel_cost ($/MMBtu) + var_om ($/MWh), rather
    than a single opaque marginal_cost number. This is what actually varies
    across a real fleet: a peaker isn't expensive because someone set
    "cost=78" -- it's expensive because its heat rate is worse (less
    efficient) AND it burns the same, pricier, fuel as the CCGTs. Passing
    different coal/gas prices lets you see the merit order shift, e.g. a
    gas price spike making coal relatively more attractive.
    """
    fleet = pd.DataFrame(
        [
            # name         pmin pmax heat_rate  fuel  var_om startup ramp  up down
            ["Coal_1",      130, 350,    10.5, "coal",   3.0,  13000,  70,  8,   8],
            ["CCGT_1",       70, 200,     7.0, "gas",    3.0,   4800,  95,  4,   3],
            ["CCGT_2",       50, 150,     7.6, "gas",    3.0,   3750,  90,  3,   3],
            ["GasPeaker_1",  20, 100,    11.5, "gas",    5.0,   1200, 100,  1,   1],
        ],
        columns=[
            "name", "pmin_mw", "pmax_mw", "heat_rate_mmbtu_per_mwh", "fuel_type", "var_om_per_mwh",
            "startup_cost", "ramp_mw_per_hr", "min_up_hr", "min_down_hr",
        ],
    )
    price_map = {"coal": coal_price_per_mmbtu, "gas": gas_price_per_mmbtu}
    fleet["fuel_cost_per_mmbtu"] = fleet["fuel_type"].map(price_map)
    fleet["marginal_cost"] = (
        fleet["heat_rate_mmbtu_per_mwh"] * fleet["fuel_cost_per_mmbtu"] + fleet["var_om_per_mwh"]
    )
    return fleet


def battery_spec() -> dict:
    return {
        "capacity_mwh": 200,
        "power_mw": 60,          # max charge/discharge rate
        "efficiency": 0.90,       # round-trip efficiency (applied on charge)
        "soc_init_frac": 0.5,
        "soc_min_frac": 0.10,
        "soc_max_frac": 0.95,
    }


def simulate_price_series(demand_mw: np.ndarray, renewable_mw: np.ndarray, fleet: pd.DataFrame,
                           oversupply_floor: float = -15.0, scarcity_cap: float = 220.0,
                           noise_std: float = 4.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Simulated hourly day-ahead electricity price -- an EXOGENOUS signal, not the
    literal shadow price of the UC solve. It's derived from a simplified merit-order
    stack of the thermal fleet (sorted by marginal cost): price = the marginal cost
    of the last unit needed to cover that hour's residual load (demand - renewable).

    This is deliberately independent of the UC's actual commitment/ramp/min-up-down
    constraints -- the point is to give the battery (and the reporting layer) a price
    signal it can react to that isn't circularly derived from the same optimization
    it's meant to influence. Real day-ahead prices are computed similarly in
    practice (marginal cost of the marginal unit), typically as the dual/shadow
    price of the full unit commitment + economic dispatch problem -- computing it
    that way here would make the "signal" and the "solve" the same thing, which
    defeats the purpose for the battery-arbitrage use case.

    residual <= 0 (renewable oversupply): price collapses toward oversupply_floor,
    scaling with the size of the oversupply -- a proxy for negative/near-zero
    pricing during curtailment events.
    residual > total fleet capacity: price caps at scarcity_cap (a simple
    value-of-lost-load-style ceiling).
    """
    if rng is None:
        rng = np.random.default_rng(7)
    residual = demand_mw - renewable_mw
    order = fleet.sort_values("marginal_cost")
    cum_cap = order["pmax_mw"].cumsum().values
    marginal_costs = order["marginal_cost"].values

    prices = np.empty(len(residual))
    for i, r in enumerate(residual):
        if r <= 0:
            oversupply_frac = min(1.0, -r / 200.0)
            prices[i] = oversupply_floor * oversupply_frac
        else:
            idx = np.searchsorted(cum_cap, r)
            prices[i] = scarcity_cap if idx >= len(marginal_costs) else marginal_costs[idx]

    prices = prices + rng.normal(0, noise_std, size=len(prices))
    return np.clip(prices, oversupply_floor, scarcity_cap)


if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = generate_hourly_dataset()
    df.to_csv(data_dir / "hourly_history.csv", index=False)
    thermal_fleet_spec().to_csv(data_dir / "thermal_fleet.csv", index=False)
    print(df.tail(48).to_string())
    print("\nFleet:\n", thermal_fleet_spec())
    print("\nBattery:\n", battery_spec())
