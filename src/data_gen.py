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
    demand_mw = np.clip(demand_mw, 400, None)

    solar_cf = np.clip(_solar_shape(hour, day_of_year) + RNG.normal(0, 0.04, n), 0, 1)
    wind_cf = _wind_shape(n, day_of_year)

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


def thermal_fleet_spec() -> pd.DataFrame:
    """
    A small, deliberately diversified thermal fleet:
    baseload coal, mid-merit CCGT, flexible CCGT, and a fast gas peaker.
    Costs are $/MWh (fuel + var O&M); startup costs in $.
    """
    fleet = pd.DataFrame(
        [
            # name        pmin  pmax  marg_cost  startup_cost  ramp_mw_per_hr  min_up  min_down
            ["Coal_1",     150,  400,     28.0,        15000,           80,        8,        8],
            ["CCGT_1",      80,  250,     42.0,         6000,          120,        4,        3],
            ["CCGT_2",      60,  200,     45.0,         5000,          120,        3,        3],
            ["GasPeaker_1", 20,  100,     78.0,         1200,          100,        1,        1],
        ],
        columns=[
            "name", "pmin_mw", "pmax_mw", "marginal_cost",
            "startup_cost", "ramp_mw_per_hr", "min_up_hr", "min_down_hr",
        ],
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
