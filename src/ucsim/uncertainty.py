"""Scenario trajectories for the stochastic strategy.

Scenarios are the P50 forecast plus a forecast-error path copied from the
held-out calibration window (actual - P50, measured out of sample). Each
path is one contiguous stretch of history starting at the same time of
day as the decision, and demand and renewable errors are taken from the
same stretch. That keeps hour-to-hour autocorrelation and the
demand/renewable correlation (cold snaps) without modelling either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def error_pool(forecasts: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Out-of-sample P50 errors on [start, end): columns demand, renewable."""
    d, r = forecasts["demand_mw"], forecasts["renewable_mw"]
    idx = d.index[(d.index >= start) & (d.index < end)].intersection(r.index)
    return pd.DataFrame({
        "demand": d.loc[idx, "actual"] - d.loc[idx, "p50"],
        "renewable": r.loc[idx, "actual"] - r.loc[idx, "p50"],
    })


def sample(pool: pd.DataFrame, demand_p50: np.ndarray, renewable_p50: np.ndarray,
           decision_time: pd.Timestamp, n: int, steps_per_day: int,
           renewable_cap: float, rng: np.random.Generator) -> list[dict]:
    """n equally likely scenarios over len(demand_p50) steps."""
    H = len(demand_p50)
    pos = np.arange(len(pool))
    tod = pool.index.hour * 60 + pool.index.minute
    target_tod = decision_time.hour * 60 + decision_time.minute
    starts = pos[(tod == target_tod) & (pos + H <= len(pool))]
    if len(starts) == 0:
        raise ValueError("calibration window is shorter than the horizon; raise train_days or calib_frac")
    picks = rng.choice(starts, size=n, replace=len(starts) < n)
    err_d, err_r = pool["demand"].to_numpy(), pool["renewable"].to_numpy()
    return [
        {
            "prob": 1.0 / n,
            "demand": np.clip(demand_p50 + err_d[s:s + H], 0, None),
            "renewable": np.clip(renewable_p50 + err_r[s:s + H], 0, renewable_cap),
        }
        for s in picks
    ]
