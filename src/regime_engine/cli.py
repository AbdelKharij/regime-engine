"""Command-line entry point.

regime-engine backtest --config config/spy_daily.toml [--out runs]
regime-engine report   --config config/spy_daily.toml [--out reports] [--runs runs]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from regime_engine.analytics.experiments import run_ablation
from regime_engine.analytics.metrics import summarize
from regime_engine.analytics.report import TearSheet
from regime_engine.backtest.engine import Backtester
from regime_engine.core.config import EngineConfig
from regime_engine.data.ingestor import DataIngestor
from regime_engine.utils.logging import configure_logging


def _backtest(args: argparse.Namespace) -> int:
    config = EngineConfig.from_toml(args.config)
    configure_logging(config.logging)
    bars = DataIngestor.from_config(config.data).load(
        config.data.symbol, config.data.start, config.data.end
    )
    result = Backtester(config).run(bars)
    run_dir = result.save(Path(args.out))
    summary = summarize(
        result.active_equity, result.active_benchmark, periods=config.run.annualization
    )
    print(
        json.dumps(
            {k: v for k, v in summary.as_dict().items() if k not in ("start", "end")},
            indent=2,
            default=str,
        )
    )
    print(f"results written to {run_dir}")
    return 0


def _report(args: argparse.Namespace) -> int:
    config = EngineConfig.from_toml(args.config)
    configure_logging(config.logging)
    bars = DataIngestor.from_config(config.data).load(
        config.data.symbol, config.data.start, config.data.end
    )
    ablation = run_ablation(config, bars, progress=lambda message: print(message, flush=True))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = TearSheet(ablation, bars).render(
        out_dir / f"{config.data.symbol.lower()}_tear_sheet.html"
    )
    ablation.table.to_csv(out_dir / f"{config.data.symbol.lower()}_ablation.csv")
    if args.runs:
        ablation.results[ablation.headline].save(Path(args.runs))
    print(ablation.table.drop(columns=["description"]).round(3).to_string())
    print(f"tear sheet written to {report}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="regime-engine", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    backtest = sub.add_parser("backtest", help="run the headline configuration and save results")
    backtest.add_argument("--config", type=Path, required=True)
    backtest.add_argument("--out", type=Path, default=Path("runs"))
    backtest.set_defaults(handler=_backtest)

    report = sub.add_parser("report", help="run the ablation study and render the tear sheet")
    report.add_argument("--config", type=Path, required=True)
    report.add_argument("--out", type=Path, default=Path("reports"))
    report.add_argument("--runs", type=Path, default=None, help="also save the headline run here")
    report.set_defaults(handler=_report)

    args = parser.parse_args(argv)
    handler = args.handler
    return int(handler(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
