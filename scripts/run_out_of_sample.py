"""Run the pre-registered out-of-sample study (docs/OUT_OF_SAMPLE.md).

Usage:  .venv/bin/python scripts/run_out_of_sample.py [--out reports/oos] [--workers 6]

Each asset runs the frozen SPY configuration with only the permitted overrides, through the
same ablation as SPY. Writes one tear sheet and ablation CSV per asset, plus
``summary.csv`` (per asset x hypothesis) and ``verdicts.md`` (pre-registered decision rules).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from regime_engine.analytics.experiments import apply_overrides, run_ablation
from regime_engine.analytics.report import TearSheet
from regime_engine.core.config import EngineConfig
from regime_engine.data.ingestor import DataIngestor
from regime_engine.utils.logging import configure_logging

FROZEN_CONFIG = Path("config/spy_daily.toml")
FROZEN_FINGERPRINT = "ab697f666cba63e2"

ASSETS: Mapping[str, Mapping[str, Mapping[str, Any]]] = {
    "QQQ": {},
    "IWM": {},
    "EFA": {},
    "TLT": {},
    "GLD": {},
    "BTC-USD": {
        "run": {"annualization": 365},
        "data": {"start": "2014-09-17", "max_abs_log_return": 0.5},
    },
}

HYPOTHESES = {
    "H1": "overlay max drawdown < buy & hold max drawdown",
    "H2": "overlay Sharpe > buy & hold Sharpe",
    "H3": "overlay turnover < realised-vol baseline turnover",
    "H4": "overlay Sharpe >= realised-vol baseline Sharpe - 0.05",
    "H5": "tactical Sharpe < overlay Sharpe",
    "H6": "overlay deflated Sharpe > 0.95",
}


def asset_config(symbol: str) -> EngineConfig:
    base = EngineConfig.from_toml(FROZEN_CONFIG)
    if base.fingerprint() != FROZEN_FINGERPRINT:
        raise SystemExit(f"config/spy_daily.toml changed since registration ({base.fingerprint()})")
    extra = ASSETS[symbol]
    overrides: dict[str, dict[str, Any]] = {
        "data": {"symbol": symbol, "benchmark": symbol, **extra.get("data", {})},
        "logging": {"console_level": "ERROR", "json_file": False},
    }
    for section, values in extra.items():
        if section != "data":
            overrides[section] = dict(values)
    return apply_overrides(base, overrides)


def run_asset(symbol: str, out_dir: Path) -> tuple[str, pd.DataFrame, str, str]:
    cfg = asset_config(symbol)
    configure_logging(cfg.logging)
    bars = DataIngestor.from_config(cfg.data).load(symbol, cfg.data.start, cfg.data.end)
    ablation = run_ablation(cfg, bars)
    slug = symbol.lower().replace("-", "_")
    TearSheet(ablation, bars).render(out_dir / f"{slug}_tear_sheet.html")
    ablation.table.to_csv(out_dir / f"{slug}_ablation.csv")
    head = ablation.results[ablation.headline]
    start = head.first_decision
    period = f"{start.date()} -> {bars.index[-1].date()}" if start is not None else "n/a"
    return symbol, ablation.table, period, head.data_hash


def evaluate(table: pd.DataFrame) -> dict[str, bool]:
    t = table
    return {
        "H1": t.loc["overlay", "max_drawdown"] < t.loc["buy_and_hold", "max_drawdown"],
        "H2": t.loc["overlay", "sharpe"] > t.loc["buy_and_hold", "sharpe"],
        "H3": t.loc["overlay", "turnover"] < t.loc["realized_vol_overlay", "turnover"],
        "H4": t.loc["overlay", "sharpe"] >= t.loc["realized_vol_overlay", "sharpe"] - 0.05,
        "H5": t.loc["tactical", "sharpe"] < t.loc["overlay", "sharpe"],
        "H6": t.loc["overlay", "dsr"] > 0.95,
    }


def verdict(holds: int, n: int) -> str:
    if holds >= n - 1:
        return "replicates"
    if holds >= n // 2:
        return "partially replicates"
    return "fails to replicate"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reports/oos"))
    parser.add_argument("--workers", type=int, default=len(ASSETS))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_asset, symbol, args.out) for symbol in ASSETS]
        results = [f.result() for f in futures]

    rows = []
    for symbol, table, period, data_hash in results:
        checks = evaluate(table)
        rows.append(
            {
                "asset": symbol,
                "period": period,
                "data_hash": data_hash,
                "overlay_sharpe": table.loc["overlay", "sharpe"],
                "baseline_sharpe": table.loc["realized_vol_overlay", "sharpe"],
                "tactical_sharpe": table.loc["tactical", "sharpe"],
                "buy_hold_sharpe": table.loc["buy_and_hold", "sharpe"],
                "overlay_vs_bh_p": table.loc["buy_and_hold", "diff_p_value"],
                "overlay_mdd": table.loc["overlay", "max_drawdown"],
                "buy_hold_mdd": table.loc["buy_and_hold", "max_drawdown"],
                "overlay_cagr": table.loc["overlay", "cagr"],
                "buy_hold_cagr": table.loc["buy_and_hold", "cagr"],
                "overlay_turnover": table.loc["overlay", "turnover"],
                "baseline_turnover": table.loc["realized_vol_overlay", "turnover"],
                "overlay_dsr": table.loc["overlay", "dsr"],
                **checks,
            }
        )
    summary = pd.DataFrame(rows).set_index("asset")
    summary.to_csv(args.out / "summary.csv")

    n = len(summary)
    lines = ["| ID | Hypothesis | Holds on | Verdict |", "|---|---|---|---|"]
    for key, text in HYPOTHESES.items():
        holds = int(summary[key].sum())
        assets = ", ".join(summary.index[summary[key]]) or "none"
        lines.append(f"| {key} | {text} | {holds}/{n} ({assets}) | {verdict(holds, n)} |")
    (args.out / "verdicts.md").write_text("\n".join(lines) + "\n")
    print(summary.drop(columns=["data_hash"]).round(3).to_string())
    print("\n".join(lines))


if __name__ == "__main__":
    main()
