"""Quantile forecasts of demand and total renewable output.

Each target gets one gradient-boosted model per quantile, trained once on
the training window. Lag features only reach back to values that are
already known when a decision is made `horizon` ahead (the shortest lag is
a whole number of days >= the horizon), so forecasting any step of a
horizon never looks at the future.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
TARGETS = ("demand_mw", "renewable_mw")


def lead_days(horizon_hr: float) -> int:
    return max(1, math.ceil(horizon_hr / 24))


def features(df: pd.DataFrame, target: str, steps_per_day: int, lag_days: int) -> pd.DataFrame:
    idx = df.index
    hour = idx.hour + idx.minute / 60
    y = df[target]
    lag = lag_days * steps_per_day
    feat = pd.DataFrame({
        "sin_hour": np.sin(2 * np.pi * hour / 24),
        "cos_hour": np.cos(2 * np.pi * hour / 24),
        "dow": idx.dayofweek,
        "sin_doy": np.sin(2 * np.pi * idx.dayofyear / 365),
        "cos_doy": np.cos(2 * np.pi * idx.dayofyear / 365),
        "lag_a": y.shift(lag).values,
        "lag_b": y.shift(lag + steps_per_day).values,
        "lag_week": y.shift(max(7, lag_days + 1) * steps_per_day).values,
        "mean_last_day": y.shift(lag).rolling(steps_per_day).mean().values,
    }, index=idx)
    return feat


def fit_predict(df: pd.DataFrame, train_start: pd.Timestamp, train_end: pd.Timestamp, steps_per_day: int,
                horizon_hr: float, caps: dict[str, float | None]) -> dict[str, pd.DataFrame]:
    """Fit on rows in [train_start, train_end); predict every row with valid features.

    Returns {target: DataFrame[p10, p50, p90, persistence, actual]}.
    """
    lag_days = lead_days(horizon_hr)
    out = {}
    for target in TARGETS:
        feat = features(df, target, steps_per_day, lag_days)
        valid = feat.notna().all(axis=1)
        train = valid & (df.index >= train_start) & (df.index < train_end)
        pred = pd.DataFrame(index=df.index[valid])
        for name, q in QUANTILES.items():
            model = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=150, learning_rate=0.05,
                max_leaf_nodes=15, random_state=0,
            )
            model.fit(feat[train], df.loc[train, target])
            pred[name] = model.predict(feat[valid])
        cols = list(QUANTILES)
        pred[cols] = np.sort(pred[cols].values, axis=1)  # independent fits can cross
        pred["persistence"] = feat.loc[valid, "lag_a"]
        pred = pred.clip(lower=0, upper=caps.get(target))
        pred["actual"] = df.loc[valid, target]
        out[target] = pred
    return out
