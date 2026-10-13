"""Command line: run a scenario file and print a strategy comparison.

    python -m ucsim scenarios/base.yaml
    python -m ucsim scenarios/base.yaml --strategies p50 stochastic --set time.sim_days=3
    python -m ucsim scenarios/base.yaml --out outputs/base
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from .config import Scenario
from .simulate import STRATEGY_LABELS, compare, summary_table


def _parse_set(items: list[str]) -> dict:
    """['time.sim_days=3', 'fuel_prices.gas=8'] -> nested dict (values parsed as YAML)."""
    out: dict = {}
    for item in items:
        path, _, value = item.partition("=")
        node = out
        *parents, leaf = path.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ucsim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", type=Path, help="scenario YAML file")
    ap.add_argument("--strategies", nargs="+", choices=list(STRATEGY_LABELS), help="override the file's strategies")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override any field, e.g. time.sim_days=3")
    ap.add_argument("--out", type=Path, help="directory for summary.csv, daily.csv and timeseries_<strategy>.csv")
    args = ap.parse_args(argv)

    scenario = Scenario.from_yaml(args.scenario).with_overrides(_parse_set(args.set))
    t = scenario.time
    print(f"{scenario.name}: {scenario.description}")
    print(f"{t.sim_days} days from {t.sim_start}, {t.resolution_min}-min steps, "
          f"{t.horizon_hr}h horizon, re-committed every {t.commit_hr}h\n")

    results = compare(scenario, args.strategies)
    table = summary_table(results)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:,.1f}".format):
        print(table)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out / "summary.csv")
        pd.concat({k: r.daily() for k, r in results.items()}, names=["strategy", "date"]).to_csv(args.out / "daily.csv")
        for k, r in results.items():
            r.timeseries.to_csv(args.out / f"timeseries_{k}.csv")
        print(f"\nWrote results to {args.out}/")


if __name__ == "__main__":
    main()
