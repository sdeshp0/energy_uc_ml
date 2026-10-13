import numpy as np
import pandas as pd
import pytest

from ucsim import Scenario, list_presets, prepare, run
from ucsim.data import generate
from ucsim.model import State, solve

BASE = Scenario()
COAL = 0  # index of the coal unit in the default fleet


def flat(demand, renewable=None, prob=1.0):
    demand = np.asarray(demand, dtype=float)
    return {"prob": prob, "demand": demand,
            "renewable": np.zeros_like(demand) if renewable is None else np.asarray(renewable, dtype=float)}


def test_data_is_reproducible_and_window_independent():
    a = generate(BASE.data, 60, pd.Timestamp("2023-03-01"))
    b = generate(BASE.data, 60, pd.Timestamp("2023-03-01"))
    longer = generate(BASE.data, 60, pd.Timestamp("2023-06-01"))
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, longer.loc[a.index])


def test_coal_can_start_mid_horizon():
    # Coal's pmin (130 MW) exceeds its hourly ramp (70 MW); startups must still work.
    fixed = np.zeros((len(BASE.fleet), 24))
    fixed[COAL, 6:] = 1
    sol = solve(BASE, [flat([500] * 24)], State.initial(BASE), fixed_u=fixed)
    assert sol.u[COAL, :6].sum() == 0 and sol.u[COAL, 6:].all()
    assert sol.p[0, COAL, 6] >= 130 - 1e-6


def test_min_up_time_carries_over_from_previous_solve():
    pricey_coal = BASE.with_overrides({"fuel_prices": {"coal": 20.0}})  # optimizer wants coal off
    state = State.initial(pricey_coal)
    state.u[COAL], state.p[COAL], state.time_in_state[COAL] = 1, 150, 2  # on for 2 of its 8 required hours
    sol = solve(pricey_coal, [flat([450] * 24)], state)
    assert sol.u[COAL, :6].all()
    assert not sol.u[COAL, 8:].any()


def test_single_scenario_matches_duplicated_scenarios():
    demand = 600 + 300 * np.sin(np.linspace(0, np.pi, 24))
    one = solve(BASE, [flat(demand)], State.initial(BASE), mip_gap=0)
    two = solve(BASE, [flat(demand, prob=0.5), flat(demand, prob=0.5)], State.initial(BASE), mip_gap=0)
    assert one.objective == pytest.approx(two.objective, rel=1e-6)


def test_config_rejects_unknown_keys_and_loads_presets():
    with pytest.raises(KeyError):
        Scenario.from_dict({"time": {"horizon_hours": 24}})
    with pytest.raises(ValueError):
        Scenario.from_dict({"time": {"horizon_hr": 12, "commit_hr": 24}})
    presets = list_presets()
    assert "base" in presets
    for path in presets.values():
        Scenario.from_yaml(path)


def test_short_simulation_perfect_foresight_is_cheapest():
    sc = Scenario.from_dict({"time": {"sim_days": 2}})
    prep = prepare(sc)
    perfect = run(sc, "perfect", prep).summary["total_cost"]
    for strategy in ("p50", "persistence"):
        assert perfect <= run(sc, strategy, prep).summary["total_cost"] * 1.01


def test_sub_hourly_resolution_runs():
    sc = Scenario.from_dict({"time": {"resolution_min": 30, "horizon_hr": 12, "commit_hr": 6, "sim_days": 1}})
    res = run(sc, "p50")
    assert len(res.timeseries) == 48
    assert res.summary["unserved_mwh"] >= 0
