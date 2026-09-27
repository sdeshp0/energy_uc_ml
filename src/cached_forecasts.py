"""
Shared Streamlit cache for the demand/wind/solar quantile forecast fit --
the single most expensive step in this app (~7s combined after the
n_estimators tuning in forecasting.py), and one that app.py, page 3, and
page 2's fallback path all separately needed.

Without a shared cache entry, each page independently re-fits the same
three models for the same n_days_history, multiplying that cost by the
number of pages visited in a session even though the underlying
computation is identical. This module exists to be the ONE place that
decorates the fit with @st.cache_data, so every caller -- regardless of
which page imports it -- hits the same cache entry for a given
n_days_history rather than each maintaining its own local cache.

This is the one caching module that imports Streamlit; the actual data
generation and model fitting stay in data_gen.py / forecasting.py, which
remain plain and independently testable.
"""

import streamlit as st

from data_gen import generate_hourly_dataset
from forecasting import fit_forecast_and_history


@st.cache_data
def get_day_and_forecasts(n_days_history: int) -> dict:
    """Generate the dataset and fit all three quantile models once. Returns
    history/next_day plus each target's next-day forecast AND in-sample
    historical predictions (the latter needed by scenarios.py's paired-error
    analysis, which would otherwise refit the same models a second time)."""
    df = generate_hourly_dataset(n_days=n_days_history + 1)
    history = df.iloc[:-24].copy()
    next_day = df.iloc[-24:].copy().reset_index(drop=True)

    demand_fc, demand_hist = fit_forecast_and_history(
        history, "demand_mw", next_day, capacity_mw=1.0, clip_range=(0.0, None))
    wind_fc, wind_hist = fit_forecast_and_history(history, "wind_cf", next_day, capacity_mw=300)
    solar_fc, solar_hist = fit_forecast_and_history(history, "solar_cf", next_day, capacity_mw=250)

    return dict(
        history=history, next_day=next_day,
        demand_fc=demand_fc, demand_hist=demand_hist,
        wind_fc=wind_fc, wind_hist=wind_hist,
        solar_fc=solar_fc, solar_hist=solar_hist,
    )