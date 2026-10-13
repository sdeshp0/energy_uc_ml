"""Synthetic demand, wind and solar at any time resolution.

Every series is generated from a fixed epoch with its own seeded random
stream, so the value at a given timestamp depends only on (seed,
resolution) -- not on how long a window you ask for or what else was
generated first. Changing the training length therefore never changes
the weather on the days being simulated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .config import Data

EPOCH = pd.Timestamp("2023-01-01")


def _ar1(rng: np.random.Generator, n: int, phi: float, stationary_sd: float) -> np.ndarray:
    eps = rng.normal(0.0, stationary_sd * np.sqrt(1 - phi**2), n)
    return lfilter([1.0], [1.0, -phi], eps)


def generate(cfg: Data, resolution_min: int, end: pd.Timestamp) -> pd.DataFrame:
    """Series from EPOCH up to (not including) `end`, indexed by timestamp.

    Columns: demand_mw, wind_mw, solar_mw, renewable_mw, cold_snap.
    """
    dt = resolution_min / 60
    ts = pd.date_range(EPOCH, end, freq=f"{resolution_min}min", inclusive="left")
    n = len(ts)
    hour = ts.hour.values + ts.minute.values / 60
    doy = ts.dayofyear.values
    day_idx = ((ts - EPOCH) // pd.Timedelta(days=1)).values
    n_days = int(day_idx[-1]) + 1

    demand_rng, wind_rng, solar_rng, snap_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(cfg.seed).spawn(4)
    )

    # Demand: two daily peaks, winter/summer seasonality, weekend dip, AR(1) noise.
    shape = (0.50 + 0.18 * np.exp(-((hour - 8) ** 2) / (2 * 2.5**2))
             + 0.38 * np.exp(-((hour - 19) ** 2) / (2 * 3.0**2)))
    seasonal = 1 + 0.05 * np.cos(2 * np.pi * (doy - 15) / 365)
    weekend = np.where(ts.dayofweek.values >= 5, 0.92, 1.0)
    demand = cfg.demand_scale_mw * (shape * seasonal * weekend + _ar1(demand_rng, n, 0.8**dt, 0.03))

    # Solar: clear-sky bell curve scaled by season, with cloudiness noise.
    daylight = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) ** 1.3
    solar_season = 0.6 + 0.4 * np.cos(2 * np.pi * (doy - 172) / 365)
    clouds = np.clip(1 + _ar1(solar_rng, n, 0.9**dt, 0.15), 0.2, 1.1)
    solar_cf = np.clip(solar_season * daylight * clouds, 0, 1)

    # Wind: mean-reverting around a winter-heavy seasonal level.
    wind_season = 0.45 + 0.15 * np.cos(2 * np.pi * (doy - 15) / 365)
    wind_cf = np.clip(wind_season + _ar1(wind_rng, n, 0.93**dt, 0.25), 0.02, 1.0)

    # Cold snaps: whole days with +16% demand and a wind drought at the same
    # time, so demand and renewable errors are correlated in the tail.
    snap_day = snap_rng.random(n_days) < cfg.cold_snap_prob
    snap = snap_day[day_idx]
    demand = np.clip(demand * np.where(snap, 1.16, 1.0), 0.35 * cfg.demand_scale_mw, None)
    wind_cf = np.clip(wind_cf * np.where(snap, 0.35, 1.0), 0.02, 1.0)

    wind = wind_cf * cfg.wind_mw
    solar = solar_cf * cfg.solar_mw
    return pd.DataFrame(
        {"demand_mw": demand, "wind_mw": wind, "solar_mw": solar,
         "renewable_mw": wind + solar, "cold_snap": snap},
        index=pd.DatetimeIndex(ts, name="timestamp"),
    )
