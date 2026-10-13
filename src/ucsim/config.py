"""Scenario configuration: one dataclass tree, loadable from YAML.

A scenario file only needs to list what differs from the defaults below;
everything else is filled in by a deep merge (see `Scenario.from_dict`).
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import yaml

STRATEGIES = ("perfect", "persistence", "p50", "conservative", "stochastic")


@dataclass
class Generator:
    name: str
    fuel: str
    pmin_mw: float
    pmax_mw: float
    heat_rate: float          # MMBtu/MWh
    var_om: float             # $/MWh
    startup_cost: float       # $ per start
    ramp_mw_per_hr: float
    min_up_hr: float
    min_down_hr: float
    fast_start: bool = False  # can be committed in real time, after the plan is locked


def _default_fleet() -> list[Generator]:
    # Sized so thermal + battery can cover the synthetic system's peak with
    # renewables near zero. See scenarios/tight_system.yaml for a short fleet.
    return [
        Generator("Coal", "coal", 130, 350, 10.5, 3.0, 13000, 70, 8, 8),
        Generator("CCGT_1", "gas", 80, 250, 7.0, 3.0, 5000, 120, 4, 3),
        Generator("CCGT_2", "gas", 60, 200, 7.6, 3.0, 4000, 100, 3, 3),
        Generator("Peaker_1", "gas", 20, 150, 11.5, 5.0, 1200, 150, 1, 1, fast_start=True),
        Generator("Peaker_2", "gas", 20, 150, 12.0, 5.0, 1200, 150, 1, 1, fast_start=True),
    ]


@dataclass
class Battery:
    power_mw: float = 60.0
    energy_mwh: float = 200.0
    round_trip_eff: float = 0.90
    soc_min_frac: float = 0.10
    soc_max_frac: float = 0.95
    soc_init_frac: float = 0.50
    cycle_cost: float = 2.0   # $/MWh charged or discharged; also stops simultaneous charge/discharge


@dataclass
class Reserves:
    total_frac: float = 0.0   # share of net load (demand - renewable) held as committed headroom
    spin_frac: float = 0.0    # share of net load deliverable within spin_response_min
    spin_response_min: float = 10.0


@dataclass
class Penalties:
    voll: float = 5000.0            # $/MWh of unserved energy
    reserve_shortfall: float = 1000.0  # $/MWh of missed reserve
    overgen: float = 5000.0         # $/MWh of thermal output that has nowhere to go


@dataclass
class TimeFrame:
    resolution_min: int = 60   # 60 or 15 (any divisor of 60 works)
    horizon_hr: int = 24       # look-ahead of each commitment solve
    commit_hr: int = 24        # how much of each solve is locked in before re-solving
    sim_start: str = "2024-01-08"
    sim_days: int = 14

    @property
    def dt(self) -> float:
        return self.resolution_min / 60

    @property
    def steps_per_day(self) -> int:
        return round(24 / self.dt)

    @property
    def horizon_steps(self) -> int:
        return round(self.horizon_hr / self.dt)

    @property
    def commit_steps(self) -> int:
        return round(self.commit_hr / self.dt)


@dataclass
class Data:
    seed: int = 42
    train_days: int = 180
    demand_scale_mw: float = 1000.0
    wind_mw: float = 300.0
    solar_mw: float = 250.0
    cold_snap_prob: float = 0.05


@dataclass
class Uncertainty:
    n_scenarios: int = 10      # sampled trajectories for the stochastic strategy
    calib_frac: float = 0.25   # share of the training window held out to measure forecast errors


@dataclass
class Scenario:
    name: str = "base"
    description: str = ""
    fleet: list[Generator] = field(default_factory=_default_fleet)
    fuel_prices: dict[str, float] = field(default_factory=lambda: {"coal": 2.2, "gas": 4.5})
    battery: Battery = field(default_factory=Battery)
    reserves: Reserves = field(default_factory=Reserves)
    penalties: Penalties = field(default_factory=Penalties)
    time: TimeFrame = field(default_factory=TimeFrame)
    data: Data = field(default_factory=Data)
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    strategies: list[str] = field(default_factory=lambda: list(STRATEGIES))

    def __post_init__(self):
        t = self.time
        if 60 % t.resolution_min:
            raise ValueError("resolution_min must divide 60")
        if not 0 < t.commit_hr <= t.horizon_hr:
            raise ValueError("need 0 < commit_hr <= horizon_hr")
        if (t.commit_hr * 60) % t.resolution_min:
            raise ValueError("commit_hr must be a whole number of time steps")
        unknown = set(self.strategies) - set(STRATEGIES)
        if unknown:
            raise ValueError(f"unknown strategies {sorted(unknown)}; choose from {STRATEGIES}")

    def fleet_frame(self) -> pd.DataFrame:
        df = pd.DataFrame([asdict(g) for g in self.fleet])
        df["fuel_price"] = df["fuel"].map(self.fuel_prices)
        if df["fuel_price"].isna().any():
            missing = sorted(df.loc[df["fuel_price"].isna(), "fuel"].unique())
            raise ValueError(f"no fuel price for {missing}")
        df["marginal_cost"] = df["heat_rate"] * df["fuel_price"] + df["var_om"]
        return df

    # ---- (de)serialization ----
    def to_dict(self) -> dict:
        return asdict(self)

    def key(self) -> str:
        """Stable string for caching."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, overrides: dict | None = None) -> Scenario:
        merged = _deep_merge(asdict(cls()), overrides or {})
        return cls(
            name=merged["name"],
            description=merged["description"],
            fleet=[Generator(**g) for g in merged["fleet"]],
            fuel_prices=merged["fuel_prices"],
            battery=Battery(**merged["battery"]),
            reserves=Reserves(**merged["reserves"]),
            penalties=Penalties(**merged["penalties"]),
            time=TimeFrame(**merged["time"]),
            data=Data(**merged["data"]),
            uncertainty=Uncertainty(**merged["uncertainty"]),
            strategies=list(merged["strategies"]),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Scenario:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        raw.setdefault("name", Path(path).stem)
        return cls.from_dict(raw)

    def with_overrides(self, overrides: dict) -> Scenario:
        return Scenario.from_dict(_deep_merge(self.to_dict(), overrides))


FREE_FORM_KEYS = {"fuel_prices"}  # mappings that may gain new keys


def _deep_merge(base: dict, override: dict, strict: bool = True) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k not in out:
            if strict:
                raise KeyError(f"unknown config key '{k}'")
            out[k] = copy.deepcopy(v)
        elif isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v, strict=k not in FREE_FORM_KEYS)
        else:
            out[k] = copy.deepcopy(v)  # lists (fleet, strategies) are replaced whole
    return out


SCENARIO_DIR = Path(__file__).resolve().parents[2] / "scenarios"


def list_presets(directory: Path = SCENARIO_DIR) -> dict[str, Path]:
    return {p.stem: p for p in sorted(directory.glob("*.yaml"))}
