"""
Day-ahead wind & solar forecasting.

Uses GradientBoostingRegressor with quantile loss to produce not just a
point forecast but a distribution (P10/P50/P90) for each hour. The P50
(median) forecast feeds the deterministic Phase-1 unit commitment; the
full quantile set is what Phase 2's scenario-based stochastic UC will
consume, so this module is written with that upgrade already in mind.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

QUANTILES = [0.1, 0.5, 0.9]


def _make_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Calendar features + autoregressive lags of the target itself.

    Day-ahead forecasts are issued using actuals known up to "now", so
    lag_24 (same hour yesterday) and lag_168 (same hour last week) are
    legitimate inputs, not leakage — they just require the target column
    to be present (with NaN for the day being forecast) in a *continuous*
    hourly frame so shift() lines up correctly across the history/future
    boundary. See _build_lagged_frame below.
    """
    feat = pd.DataFrame(index=df.index)
    feat["hour"] = df["hour"]
    feat["dow"] = df["dow"]
    feat["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    feat["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    feat["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    feat["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    feat["lag_24"] = df[target_col].shift(24)
    feat["lag_48"] = df[target_col].shift(48)
    feat["lag_168"] = df[target_col].shift(168)
    feat["roll_mean_24_72"] = df[target_col].shift(24).rolling(48, min_periods=24).mean()
    return feat


class QuantileForecaster:
    """One GBM per quantile, per target (wind_cf / solar_cf)."""

    def __init__(self, target_col: str, quantiles: list[float] = QUANTILES):
        self.target_col = target_col
        self.quantiles = quantiles
        self.models: dict[float, GradientBoostingRegressor] = {}
        self.feature_cols_: list[str] | None = None

    def fit(self, feat_train: pd.DataFrame, y_train: np.ndarray) -> "QuantileForecaster":
        self.feature_cols_ = list(feat_train.columns)
        for q in self.quantiles:
            model = GradientBoostingRegressor(
                loss="quantile", alpha=q,
                n_estimators=300, max_depth=3, learning_rate=0.04,
                subsample=0.8, random_state=0,
            )
            model.fit(feat_train, y_train)
            self.models[q] = model
        return self

    def predict(self, feat_future: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=feat_future.index)
        for q in self.quantiles:
            preds = self.models[q].predict(feat_future[self.feature_cols_])
            out[f"p{int(q * 100)}"] = np.clip(preds, 0, 1)
        # enforce monotonicity across quantiles (GBMs fit independently can cross)
        cols = [f"p{int(q * 100)}" for q in sorted(self.quantiles)]
        out[cols] = np.sort(out[cols].values, axis=1)
        return out


def forecast_next_day(history: pd.DataFrame, target_col: str, next_day_calendar: pd.DataFrame,
                       capacity_mw: float) -> pd.DataFrame:
    """
    Fit on `history` (must contain actual target_col values), forecast the
    24 hours described by `next_day_calendar` (hour/dow/day_of_year only —
    no target needed, that's what we're predicting), scale cf -> MW.
    """
    combined = pd.concat(
        [history[["hour", "dow", "day_of_year", target_col]],
         next_day_calendar[["hour", "dow", "day_of_year"]].assign(**{target_col: np.nan})],
        ignore_index=True,
    )
    all_feat = _make_features(combined, target_col)

    n_hist = len(history)
    feat_train_full = all_feat.iloc[:n_hist]
    y_train_full = combined[target_col].iloc[:n_hist]
    valid = feat_train_full.notna().all(axis=1)  # drop lag warm-up rows (first ~168h)
    feat_train, y_train = feat_train_full[valid], y_train_full[valid].values

    feat_future = all_feat.iloc[n_hist:].reset_index(drop=True)

    model = QuantileForecaster(target_col).fit(feat_train, y_train)
    cf_forecast = model.predict(feat_future)
    mw_forecast = cf_forecast * capacity_mw
    mw_forecast.columns = [f"{target_col.replace('_cf','')}_{c}_mw" for c in cf_forecast.columns]
    return mw_forecast


if __name__ == "__main__":
    from data_gen import generate_hourly_dataset

    df = generate_hourly_dataset(n_days=400)
    history = df.iloc[:-24].copy()
    next_day = df.iloc[-24:].copy().reset_index(drop=True)

    wind_fc = forecast_next_day(history, "wind_cf", next_day, capacity_mw=300)
    solar_fc = forecast_next_day(history, "solar_cf", next_day, capacity_mw=250)

    result = pd.concat([next_day[["hour", "wind_mw", "solar_mw"]], wind_fc, solar_fc], axis=1)
    print(result.to_string())

    # quick accuracy check on the median forecast
    wind_mae = (result["wind_mw"] - result["wind_p50_mw"]).abs().mean()
    solar_mae = (result["solar_mw"] - result["solar_p50_mw"]).abs().mean()
    print(f"\nWind P50 MAE:  {wind_mae:.1f} MW (capacity 300 MW)")
    print(f"Solar P50 MAE: {solar_mae:.1f} MW (capacity 250 MW)")
