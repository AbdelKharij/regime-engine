"""Cross-asset charts for the out-of-sample study (presentation only; no new analysis).

Reads ``reports/oos/summary.csv`` (written by run_out_of_sample.py) plus the SPY ablation, and
draws two dumbbell charts: max drawdown and Sharpe, overlay vs buy & hold, per asset.

Usage:  .venv/bin/python scripts/plot_out_of_sample.py [--out docs/images]
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

import pandas as pd
from matplotlib.lines import Line2D

from regime_engine.analytics.report import INK_2, MUTED, SERIES, _figure, _legend, _png


def load(oos_dir: Path, spy_table: Path) -> pd.DataFrame:
    summary = pd.read_csv(oos_dir / "summary.csv", index_col=0)
    spy = pd.read_csv(spy_table, index_col=0)
    spy_row = pd.DataFrame(
        {
            "overlay_mdd": [spy.loc["overlay", "max_drawdown"]],
            "buy_hold_mdd": [spy.loc["buy_and_hold", "max_drawdown"]],
            "overlay_sharpe": [spy.loc["overlay", "sharpe"]],
            "buy_hold_sharpe": [spy.loc["buy_and_hold", "sharpe"]],
        },
        index=["SPY (in-sample)"],
    )
    return pd.concat([spy_row, summary[spy_row.columns]])


def dumbbell(frame: pd.DataFrame, overlay: str, benchmark: str, xlabel: str, scale: float) -> str:
    fig, ax = _figure(width=8.0, height=0.48 * len(frame) + 1.0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.6)
    ax.spines["left"].set_visible(False)
    rows = frame.iloc[::-1]
    for y, (_, row) in enumerate(rows.iterrows()):
        a, b = row[overlay] * scale, row[benchmark] * scale
        ax.plot([a, b], [y, y], color=MUTED, linewidth=1.2, zorder=1)
        ax.scatter([b], [y], color=SERIES[1], s=46, zorder=3)
        ax.scatter([a], [y], color=SERIES[0], s=46, zorder=4)
    ax.set_yticks(range(len(rows)), list(rows.index), fontsize=9, color=INK_2)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    _legend(
        ax,
        [
            Line2D(
                [0], [0], color=SERIES[0], marker="o", linestyle="", markersize=7, label="Overlay"
            ),
            Line2D(
                [0],
                [0],
                color=SERIES[1],
                marker="o",
                linestyle="",
                markersize=7,
                label="Buy & hold",
            ),
        ],
    )
    return _png(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oos", type=Path, default=Path("reports/oos"))
    parser.add_argument("--spy", type=Path, default=Path("reports/spy_ablation.csv"))
    parser.add_argument("--out", type=Path, default=Path("docs/images"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frame = load(args.oos, args.spy)
    charts = {
        "oos_drawdown": dumbbell(
            frame, "overlay_mdd", "buy_hold_mdd", "Maximum drawdown (%)", 100.0
        ),
        "oos_sharpe": dumbbell(frame, "overlay_sharpe", "buy_hold_sharpe", "Sharpe ratio", 1.0),
    }
    for name, png in charts.items():
        (args.out / f"{name}.png").write_bytes(base64.b64decode(png))
        print(args.out / f"{name}.png")


if __name__ == "__main__":
    main()
