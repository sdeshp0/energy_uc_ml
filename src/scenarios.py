"""
Joint scenario construction for the two-stage stochastic unit commitment.

The key design choice here, and the reason this module exists rather than
just multiplying two marginal probability distributions together: demand
and renewable forecast errors are often CORRELATED in reality (a cold snap
driving both high heating demand and low wind simultaneously is the classic
case), and an independence assumption -- P(demand=lo, renewable=lo) =
P(demand=lo) x P(renewable=lo) -- will understate exactly the joint tail
event that a stochastic UC hedge is meant to protect against. So instead of
building two separate marginal distributions and taking their outer
product, this module builds the empirical JOINT distribution directly from
paired historical (demand_error, renewable_error) observations at the same
hours, which captures whatever correlation is actually present in the data
without having to assume a particular dependence structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting import historical_quantile_predictions, forecast_next_day

LEVELS = ["low", "mid", "high"]
LEVEL_TO_QUANTILE = {"low": "p10", "mid": "p50", "high": "p90"}


def paired_errors_from_predictions(demand_hist_pred: pd.DataFrame, wind_hist_pred: pd.DataFrame,
                                    solar_hist_pred: pd.DataFrame, wind_capacity_mw: float = 300,
                                    solar_capacity_mw: float = 250) -> pd.DataFrame:
    """Same as paired_historical_errors, but takes already-computed in-sample
    historical predictions (e.g. from forecasting.fit_forecast_and_history) instead
    of recomputing them from scratch -- use this when the caller is already fitting
    these same three models to build the next-day forecast, to avoid refitting them
    a second time just for the error analysis (refitting three GBM-per-quantile
    models twice roughly doubles this page's load time for no benefit)."""
    common_idx = demand_hist_pred.index.intersection(wind_hist_pred.index).intersection(solar_hist_pred.index)
    demand_hist_pred = demand_hist_pred.loc[common_idx]
    wind_hist_pred = wind_hist_pred.loc[common_idx]
    solar_hist_pred = solar_hist_pred.loc[common_idx]

    demand_error = demand_hist_pred["actual"] - demand_hist_pred["p50"]
    renewable_actual = wind_hist_pred["actual"] * wind_capacity_mw + solar_hist_pred["actual"] * solar_capacity_mw
    renewable_pred50 = wind_hist_pred["p50"] * wind_capacity_mw + solar_hist_pred["p50"] * solar_capacity_mw
    renewable_error = renewable_actual - renewable_pred50

    return pd.DataFrame({
        "demand_error": demand_error.values,
        "renewable_error": renewable_error.values,
    })


def paired_historical_errors(history: pd.DataFrame, wind_capacity_mw: float = 300,
                              solar_capacity_mw: float = 250) -> pd.DataFrame:
    """In-sample demand and combined-renewable (wind+solar) forecast errors
    (actual - P50), aligned by hour, for joint probability estimation.

    Combining wind and solar error by simple addition is exact regardless of
    correlation between them (error of a sum = sum of errors, always) -- that
    part needs no independence assumption. The renewable QUANTILE FORECASTS
    used elsewhere in this project (wind_p10 + solar_p10 as "renewable P10")
    are a separate, pre-existing simplification that does carry an implicit
    independence assumption; that's out of scope for this module, which only
    consumes the resulting P50 for computing errors.

    See historical_quantile_predictions for the in-sample-vs-walk-forward
    caveat -- this preserves the co-movement between demand and renewable
    errors correctly even though the error magnitudes are somewhat optimistic.

    Standalone convenience wrapper that fits its own models; if you're also
    building a next-day forecast for the same targets, use
    forecasting.fit_forecast_and_history + paired_errors_from_predictions
    instead to avoid fitting each target's models twice.
    """
    demand_pred = historical_quantile_predictions(history, "demand_mw", clip_range=(0.0, None))
    wind_pred = historical_quantile_predictions(history, "wind_cf", clip_range=(0.0, 1.0))
    solar_pred = historical_quantile_predictions(history, "solar_cf", clip_range=(0.0, 1.0))
    return paired_errors_from_predictions(demand_pred, wind_pred, solar_pred, wind_capacity_mw, solar_capacity_mw)


def tercile_labels(errors: pd.Series) -> pd.Series:
    """Equal-COUNT tercile split (standard) -- as opposed to an equal-error-MASS
    split, which doesn't correspond to a fixed probability and can be skewed by
    a handful of outliers. Returns 'low'/'mid'/'high' per observation."""
    q1, q2 = errors.quantile([1 / 3, 2 / 3])
    return pd.cut(errors, bins=[-np.inf, q1, q2, np.inf], labels=LEVELS)


def joint_scenario_probabilities(paired_errors: pd.DataFrame) -> pd.DataFrame:
    """Empirical joint frequency of (demand_tercile, renewable_tercile) pairs --
    the correlation-aware replacement for assuming independence and taking the
    outer product of two separately-estimated marginals. Returns a 3x3 DataFrame
    (index=demand level, columns=renewable level) of probabilities summing to 1.
    """
    d_labels = tercile_labels(paired_errors["demand_error"])
    r_labels = tercile_labels(paired_errors["renewable_error"])
    joint = pd.crosstab(d_labels, r_labels, normalize=True)
    return joint.reindex(index=LEVELS, columns=LEVELS, fill_value=0.0)


def independence_baseline_probabilities(joint_probs: pd.DataFrame) -> pd.DataFrame:
    """The independence-assumption alternative (marginal outer product), computed
    from the SAME data, purely so the two can be compared side by side -- this is
    what the naive approach would have produced instead of joint_scenario_probabilities."""
    demand_marginal = joint_probs.sum(axis=1)
    renewable_marginal = joint_probs.sum(axis=0)
    outer = pd.DataFrame(
        np.outer(demand_marginal.values, renewable_marginal.values),
        index=joint_probs.index, columns=joint_probs.columns,
    )
    return outer


def build_nine_scenarios(demand_fc: pd.DataFrame, renewable_fc: pd.DataFrame,
                          joint_probs: pd.DataFrame) -> list[dict]:
    """demand_fc: 24-row DataFrame with columns demand_p10_mw/demand_p50_mw/demand_p90_mw.
    renewable_fc: same shape, columns renewable_p10_mw/renewable_p50_mw/renewable_p90_mw
    (combined wind+solar -- see combine_renewable_forecast below).
    joint_probs: 3x3 DataFrame from joint_scenario_probabilities (or the independence
    baseline, for comparison).

    Returns a list of 9 dicts: probability, demand (len-T array), renewable (len-T
    array), demand_label, renewable_label -- ready to hand to
    StochasticUnitCommitmentModel.
    """
    scenarios = []
    for d_label in LEVELS:
        for r_label in LEVELS:
            prob = float(joint_probs.loc[d_label, r_label])
            demand_arr = demand_fc[f"demand_{LEVEL_TO_QUANTILE[d_label]}_mw"].values
            renewable_arr = renewable_fc[f"renewable_{LEVEL_TO_QUANTILE[r_label]}_mw"].values
            scenarios.append({
                "probability": prob, "demand": demand_arr, "renewable": renewable_arr,
                "demand_label": d_label, "renewable_label": r_label,
            })
    return scenarios


def combine_renewable_forecast(wind_fc: pd.DataFrame, solar_fc: pd.DataFrame) -> pd.DataFrame:
    """Combine separately-forecast wind/solar quantiles into a single 'renewable'
    quantile forecast by summing matching levels (renewable_p10 = wind_p10 +
    solar_p10, etc.) -- this is the project's existing convention (used elsewhere
    for the P10/P50/P90 renewable scenarios) and does carry an implicit
    independence assumption between wind and solar errors specifically. Left
    as-is here since fixing that is a separate, pre-existing simplification
    outside this module's scope (which is about demand-vs-renewable, not
    wind-vs-solar)."""
    out = pd.DataFrame()
    for level in ["p10", "p50", "p90"]:
        out[f"renewable_{level}_mw"] = wind_fc[f"wind_{level}_mw"].values + solar_fc[f"solar_{level}_mw"].values
    return out
