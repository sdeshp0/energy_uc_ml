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
    """One GBM per quantile, per target. clip_range bounds predictions (min, max),
    either of which can be None to leave that side unbounded -- (0.0, 1.0) for a
    capacity factor like wind/solar, (0.0, None) for a raw MW quantity like demand
    that can't go negative but has no fixed upper bound."""

    def __init__(self, target_col: str, quantiles: list[float] = QUANTILES,
                 clip_range: tuple[float | None, float | None] = (0.0, 1.0)):
        self.target_col = target_col
        self.quantiles = quantiles
        self.clip_range = clip_range
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
        lo, hi = self.clip_range
        for q in self.quantiles:
            preds = self.models[q].predict(feat_future[self.feature_cols_])
            out[f"p{int(q * 100)}"] = np.clip(preds, lo, hi)
        # enforce monotonicity across quantiles (GBMs fit independently can cross)
        cols = [f"p{int(q * 100)}" for q in sorted(self.quantiles)]
        out[cols] = np.sort(out[cols].values, axis=1)
        return out


def _fit_quantile_model(history: pd.DataFrame, target_col: str,
                         clip_range: tuple[float | None, float | None]):
    """Shared fit step: build features on `history` alone (no future rows), drop
    the lag warm-up NaNs, and fit a QuantileForecaster. Returns (model, feat_train,
    y_train, valid_mask) so callers can either predict on future rows
    (forecast_next_day) or on these same training rows (historical_quantile_predictions,
    for in-sample residual analysis)."""
    feat = _make_features(history[["hour", "dow", "day_of_year", target_col]], target_col)
    y = history[target_col]
    valid = feat.notna().all(axis=1)
    feat_train, y_train = feat[valid], y[valid].values
    model = QuantileForecaster(target_col, clip_range=clip_range).fit(feat_train, y_train)
    return model, feat_train, y_train, valid


def historical_quantile_predictions(history: pd.DataFrame, target_col: str,
                                     clip_range: tuple[float | None, float | None] = (0.0, 1.0)) -> pd.DataFrame:
    """In-sample quantile predictions on the same historical rows used to train the
    model -- a fast proxy for backtested forecast error, used for estimating joint
    scenario probabilities (see scenarios.py). Two things worth being honest about:
    this is IN-SAMPLE (the model has seen these exact rows during training), so the
    resulting errors are optimistic relative to true walk-forward out-of-sample
    error -- a real backtest would retrain on a rolling window and predict strictly
    ahead each time, which is far more expensive (many refits) and out of scope
    here. What this DOES preserve correctly is the *correlation structure* between
    two targets' errors at the same hour (e.g. demand vs. renewable), which is the
    property scenarios.py actually needs -- it doesn't need the error magnitudes to
    be unbiased, just the co-movement between variables.
    """
    model, feat_train, y_train, valid = _fit_quantile_model(history, target_col, clip_range)
    preds = model.predict(feat_train)
    preds["actual"] = y_train
    preds.index = history.index[valid]
    return preds


def fit_forecast_and_history(history: pd.DataFrame, target_col: str, next_day_calendar: pd.DataFrame,
                              capacity_mw: float = 1.0,
                              clip_range: tuple[float | None, float | None] = (0.0, 1.0)):
    """Fit once, return (future_mw_forecast, historical_in_sample_predictions) --
    avoids double-fitting the same model when both are needed (e.g. scenarios.py
    wants in-sample historical error alongside the actual next-day forecast).
    forecast_next_day and historical_quantile_predictions are kept as the simple
    single-purpose entry points and both delegate here when used together; call
    this directly when you need both outputs to skip the redundant refit."""
    combined = pd.concat(
        [history[["hour", "dow", "day_of_year", target_col]],
         next_day_calendar[["hour", "dow", "day_of_year"]].assign(**{target_col: np.nan})],
        ignore_index=True,
    )
    all_feat = _make_features(combined, target_col)

    n_hist = len(history)
    feat_train_full = all_feat.iloc[:n_hist]
    y_train_full = combined[target_col].iloc[:n_hist]
    valid = feat_train_full.notna().all(axis=1)
    feat_train, y_train = feat_train_full[valid], y_train_full[valid].values
    feat_future = all_feat.iloc[n_hist:].reset_index(drop=True)

    model = QuantileForecaster(target_col, clip_range=clip_range).fit(feat_train, y_train)

    cf_forecast = model.predict(feat_future)
    mw_forecast = cf_forecast * capacity_mw
    base = target_col
    if base.endswith("_cf") or base.endswith("_mw"):
        base = base[:-3]
    mw_forecast.columns = [f"{base}_{c}_mw" for c in cf_forecast.columns]

    hist_pred = model.predict(feat_train)
    hist_pred["actual"] = y_train
    hist_pred.index = history.index[valid]

    return mw_forecast, hist_pred


def fit_and_predict_range(full_df: pd.DataFrame, target_col: str, n_train_rows: int,
                           clip_range: tuple[float | None, float | None] = (0.0, 1.0)) -> pd.DataFrame:
    """Fit ONCE on the first n_train_rows of full_df, then predict quantiles for
    every row in full_df from that single fit -- including rows well beyond the
    training window. For rolling-horizon simulation: refitting the model fresh
    for each of many simulated days would be far too slow (each fit is a few
    seconds x 3 quantiles x 3 targets), and isn't necessary -- a model fit once
    can predict on any future feature row just as well as tomorrow's.

    Lag features are computed on the full continuous series, so predictions for
    simulation days correctly reference real actual values up to that point (this
    is what makes it a fair simulation and not a lookahead: the model itself was
    only ever trained on the initial window, it just gets to see real lag values
    as the simulation walks forward, exactly as a deployed model would).

    Returns a DataFrame aligned to full_df's index (rows before the lag warm-up
    dropped) with p10/p50/p90 and 'actual' columns.
    """
    feat_full = _make_features(full_df[["hour", "dow", "day_of_year", target_col]], target_col)
    y_full = full_df[target_col]

    train_valid = feat_full.iloc[:n_train_rows].notna().all(axis=1)
    feat_train = feat_full.iloc[:n_train_rows][train_valid]
    y_train = y_full.iloc[:n_train_rows][train_valid].values
    model = QuantileForecaster(target_col, clip_range=clip_range).fit(feat_train, y_train)

    predict_valid = feat_full.notna().all(axis=1)
    preds = model.predict(feat_full[predict_valid])
    preds["actual"] = y_full[predict_valid].values
    preds.index = full_df.index[predict_valid]
    return preds


def forecast_next_day(history: pd.DataFrame, target_col: str, next_day_calendar: pd.DataFrame,
                       capacity_mw: float = 1.0,
                       clip_range: tuple[float | None, float | None] = (0.0, 1.0)) -> pd.DataFrame:
    """
    Fit on `history` (must contain actual target_col values), forecast the
    24 hours described by `next_day_calendar` (hour/dow/day_of_year only —
    no target needed, that's what we're predicting), scale by capacity_mw.

    capacity_mw scales a [0,1] capacity factor (wind/solar) up to MW; leave at
    the default 1.0 for a target already in MW (e.g. demand_mw), and pass
    clip_range=(0.0, None) in that case since demand isn't bounded at 1.

    If you also need in-sample historical predictions for this same target
    (e.g. for scenarios.py's paired-error analysis), call fit_forecast_and_history
    directly instead of calling this and historical_quantile_predictions
    separately -- that refits the same model twice for no reason.
    """
    mw_forecast, _ = fit_forecast_and_history(history, target_col, next_day_calendar, capacity_mw, clip_range)
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
    