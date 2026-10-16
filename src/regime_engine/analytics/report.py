"""Static HTML tear sheet (single self-contained file, figures embedded as PNG).

Charts use matplotlib's object API (``Figure`` + Agg canvas, no pyplot global state) and a
validated categorical palette: slots 1-4 in fixed order, colour following the entity, one
y-axis per chart, recessive grid. Every chart is paired with a table so no number is
colour-only.
"""

from __future__ import annotations

import base64
import html
import io
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from regime_engine.analytics.diagnostics import (
    REGIME_NAMES,
    accuracy,
    decoded_vs_realized,
    filtered_vs_smoothed,
    mean_transition_matrix,
    probability_frame,
    regime_attribution,
    var_backtest_table,
)
from regime_engine.analytics.experiments import AblationResult
from regime_engine.analytics.metrics import drawdown_episodes, drawdown_series, summarize

# Validated reference palette (light surface): categorical slots in fixed order, chart chrome,
# a single-hue sequential ramp, and the reserved "critical" status colour.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
CRITICAL = "#d03b3b"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
REGIME_LABELS = {
    "BULL_LOW_VOL": "Bull / low vol",
    "BULL_HIGH_VOL": "Bull / high vol",
    "BEAR_LOW_VOL": "Bear / low vol",
    "BEAR_HIGH_VOL": "Bear / high vol (crash)",
}


# --------------------------------------------------------------------------------------------
# Figure helpers
# --------------------------------------------------------------------------------------------


def _figure(width: float = 10.0, height: float = 3.4) -> tuple[Figure, Axes]:
    fig = Figure(figsize=(width, height), facecolor=SURFACE)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    _style(ax)
    return fig, ax


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(which="both", colors=MUTED, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _legend(ax: Axes, handles: list[Line2D] | list[Patch], ncol: int | None = None) -> None:
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.16),
        ncol=ncol or len(handles),
        frameon=False,
        fontsize=8.5,
        handlelength=1.6,
        columnspacing=1.4,
    )
    for text in legend.get_texts():
        text.set_color(INK_2)


def _log_axis(ax: Axes) -> None:
    """Log y-axis with plain-number ticks at 1-2-5 steps (no scientific notation)."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _end_label_offsets(values: list[float], min_gap: float = 0.07) -> list[float]:
    """Vertical offsets (points) so end-of-line labels on a log axis don't overlap."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    offsets = [0.0] * len(values)
    previous: float | None = None
    shift = 0.0
    for i in order:
        position = math.log(values[i])
        shift = shift + 12.0 if previous is not None and position - previous < min_gap else 0.0
        offsets[i] = shift
        previous = position
    return offsets


def _png(fig: Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=144, facecolor=SURFACE, bbox_inches="tight")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _line_handles(names: list[str]) -> list[Line2D]:
    return [Line2D([0], [0], color=SERIES[i], linewidth=2, label=n) for i, n in enumerate(names)]


# --------------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------------


def growth_chart(series: Mapping[str, pd.Series]) -> str:
    fig, ax = _figure(height=3.8)
    names = list(series)
    growths = [values / values.iloc[0] for values in series.values()]
    offsets = _end_label_offsets([float(g.iloc[-1]) for g in growths])
    for i, growth in enumerate(growths):
        ax.plot(
            growth.index,
            growth.to_numpy(),
            color=SERIES[i],
            linewidth=1.6,
            zorder=len(growths) - i + 1,
        )
        ax.annotate(
            f"{growth.iloc[-1]:.2f}x",
            (growth.index[-1], growth.iloc[-1]),
            xytext=(4, offsets[i]),
            textcoords="offset points",
            fontsize=8.5,
            color=INK_2,
            va="center",
        )
    _log_axis(ax)
    ax.set_ylabel("Growth of $1 (log scale)", color=INK_2, fontsize=9)
    _legend(ax, _line_handles(names))
    return _png(fig)


def drawdown_chart(series: Mapping[str, pd.Series]) -> str:
    fig, ax = _figure(height=2.8)
    handles = []
    items = list(series.items())
    for i, (name, values) in enumerate(items):
        dd = -drawdown_series(values) * 100.0
        # The first series (headline) is drawn on top so the benchmark cannot hide it.
        ax.plot(dd.index, dd.to_numpy(), color=SERIES[i], linewidth=1.3, zorder=len(items) - i + 1)
        handles.append(Line2D([0], [0], color=SERIES[i], linewidth=2, label=name))
    ax.axhline(0.0, color=AXIS, linewidth=0.8)
    ax.set_ylabel("Drawdown (%)", color=INK_2, fontsize=9)
    _legend(ax, handles)
    return _png(fig)


def rolling_sharpe_chart(series: Mapping[str, pd.Series], window: int = 252) -> str:
    fig, ax = _figure(height=2.8)
    handles = []
    for i, (name, values) in enumerate(series.items()):
        r = values.pct_change()
        rolling = r.rolling(window).mean() / r.rolling(window).std() * math.sqrt(252)
        ax.plot(
            rolling.index,
            rolling.to_numpy(),
            color=SERIES[i],
            linewidth=1.3,
            zorder=len(series) - i + 1,
        )
        handles.append(Line2D([0], [0], color=SERIES[i], linewidth=2, label=name))
    ax.axhline(0.0, color=AXIS, linewidth=0.8)
    ax.set_ylabel(f"Rolling {window}-bar Sharpe", color=INK_2, fontsize=9)
    _legend(ax, handles)
    return _png(fig)


def regime_timeline_chart(close: pd.Series, regimes: pd.Series) -> str:
    fig, ax = _figure(height=3.4)
    price = close.reindex(regimes.index)
    for k in range(len(REGIME_NAMES)):
        ax.fill_between(
            regimes.index,
            0,
            1,
            where=(regimes == k).tolist(),
            color=SERIES[k],
            alpha=0.22,
            linewidth=0,
            step="post",
            transform=ax.get_xaxis_transform(),
        )
    ax.plot(price.index, price.to_numpy(), color=INK, linewidth=1.0)
    _log_axis(ax)
    ax.set_ylabel("Close (log scale)", color=INK_2, fontsize=9)
    _legend(
        ax,
        [
            Patch(facecolor=SERIES[k], alpha=0.5, label=REGIME_LABELS[n])
            for k, n in enumerate(REGIME_NAMES)
        ],
    )
    return _png(fig)


def probability_chart(probs: pd.DataFrame) -> str:
    fig, ax = _figure(height=2.8)
    ax.stackplot(
        probs.index,
        *[probs[c].to_numpy() for c in REGIME_NAMES],
        colors=SERIES,
        edgecolor=SURFACE,
        linewidth=0.3,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Filtered P(regime)", color=INK_2, fontsize=9)
    _legend(
        ax, [Patch(facecolor=SERIES[k], label=REGIME_LABELS[n]) for k, n in enumerate(REGIME_NAMES)]
    )
    return _png(fig)


def exposure_chart(weight: pd.Series) -> str:
    fig, ax = _figure(height=2.4)
    ax.plot(weight.index, weight.to_numpy(), color=SERIES[0], linewidth=1.0)
    ax.axhline(1.0, color=AXIS, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_ylabel("Portfolio weight", color=INK_2, fontsize=9)
    return _png(fig)


def heatmap(
    matrix: pd.DataFrame, *, fmt: Callable[[float], str], annotations: pd.DataFrame | None = None
) -> str:
    fig = Figure(figsize=(4.6, 3.9), facecolor=SURFACE)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    values = matrix.to_numpy(dtype=np.float64)
    ax.imshow(values, cmap=SEQUENTIAL, vmin=0.0, vmax=max(values.max(), 1e-12))
    labels = [
        REGIME_LABELS[n].replace(" / ", "/\n").replace(" (crash)", "") for n in matrix.columns
    ]
    ax.set_xticks(range(len(labels)), labels, fontsize=7.5, color=INK_2)
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5, color=INK_2)
    ax.set_xlabel(str(matrix.columns.name or "to"), color=INK_2, fontsize=8.5)
    ax.set_ylabel(str(matrix.index.name or "from"), color=INK_2, fontsize=8.5)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    threshold = 0.55 * values.max()
    for (i, j), value in np.ndenumerate(values):
        text = fmt(value)
        if annotations is not None:
            text += f"\n±{fmt(float(annotations.to_numpy(dtype=np.float64)[i, j]))}"
        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=7.5,
            color=SURFACE if value > threshold else INK,
        )
    return _png(fig)


def var_chart(var_forecasts: pd.DataFrame, column: str = "var_fhs") -> str:
    fig, ax = _figure(height=3.0)
    data = var_forecasts[[column, "realized_next"]].dropna()
    realized = data["realized_next"] * 100.0
    limit = -data[column] * 100.0
    breaches = realized < limit
    ax.scatter(realized.index, realized.to_numpy(), s=2, color=MUTED, linewidths=0)
    ax.plot(limit.index, limit.to_numpy(), color=SERIES[0], linewidth=1.1)
    ax.scatter(
        realized.index[breaches],
        realized[breaches].to_numpy(),
        s=14,
        color=CRITICAL,
        marker="v",
        linewidths=0,
    )
    ax.set_ylabel("Next-bar return (%)", color=INK_2, fontsize=9)
    _legend(
        ax,
        [
            Line2D([0], [0], color=SERIES[0], linewidth=2, label="−VaR 99% (filtered historical)"),
            Line2D(
                [0],
                [0],
                color=MUTED,
                marker="o",
                linestyle="",
                markersize=4,
                label="realised return",
            ),
            Line2D(
                [0],
                [0],
                color=CRITICAL,
                marker="v",
                linestyle="",
                markersize=6,
                label=f"breach ({int(breaches.sum())})",
            ),
        ],
    )
    return _png(fig)


def ablation_chart(table: pd.DataFrame, headline: str) -> str:
    """Sharpe ratio ± 1 standard error per variant; the headline is emphasised, others muted.

    A star marks variants whose Sharpe differs from the headline at p < 0.05 (paired
    stationary bootstrap).
    """
    rows = table.sort_values("sharpe")
    n = len(rows)
    fig, ax = _figure(width=10.0, height=0.42 * n + 0.9)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    head_sharpe = float(table["sharpe"].astype(float).to_dict()[headline])
    ax.axvline(head_sharpe, color=SERIES[0], linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    labels = []
    for y, (name, row) in enumerate(rows.iterrows()):
        sharpe, se = float(row["sharpe"]), float(row["sharpe_se"])
        color = SERIES[0] if name == headline else (SERIES[1] if name == "buy_and_hold" else MUTED)
        ax.errorbar(
            sharpe,
            y,
            xerr=se,
            fmt="o",
            color=color,
            markersize=6,
            elinewidth=1.4,
            capsize=0,
            zorder=3,
        )
        p_value = float(row["diff_p_value"])
        star = " *" if math.isfinite(p_value) and p_value < 0.05 else ""
        ax.annotate(
            f"{sharpe:.2f}{star}",
            (sharpe + se, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK_2,
        )
        labels.append(str(row["description"]).split(" (")[0])
    ax.set_yticks(range(n), labels, fontsize=8.5, color=INK_2)
    ax.set_xlabel(
        "Sharpe ratio (± 1 standard error)  ·  * differs from headline at p < 0.05",
        color=INK_2,
        fontsize=9,
    )
    ax.spines["left"].set_visible(False)
    return _png(fig)


# --------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------


def _pct(x: object, digits: int = 1) -> str:
    return (
        "–"
        if not isinstance(x, (int, float)) or not math.isfinite(float(x))
        else f"{float(x) * 100:.{digits}f}%"
    )


def _num(x: object, digits: int = 2) -> str:
    return (
        "–"
        if not isinstance(x, (int, float)) or not math.isfinite(float(x))
        else f"{float(x):.{digits}f}"
    )


def _money(x: object) -> str:
    return (
        "–"
        if not isinstance(x, (int, float)) or not math.isfinite(float(x))
        else f"${float(x):,.0f}"
    )


def _escape(value: object) -> str:
    return html.escape(str(value))


def _row_normalize(counts: pd.DataFrame) -> pd.DataFrame:
    return counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def _table(
    frame: pd.DataFrame, formats: Mapping[str, Callable[[object], str]], *, index_label: str = ""
) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [index_label, *frame.columns])
    body = []
    for idx, row in frame.iterrows():
        cells = [f"<th>{html.escape(str(idx))}</th>"]
        for col, value in row.items():
            render = formats.get(str(col), _escape)
            cells.append(f"<td>{render(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


_CSS = f"""
:root {{ color-scheme: light; --page:{PAGE}; --surface:{SURFACE}; --ink:{INK}; --ink2:{INK_2};
  --muted:{MUTED}; --grid:{GRID}; --border:rgba(11,11,11,0.10); }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--page); color:var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1080px; margin: 0 auto; padding: 32px 16px 64px; }}
h1 {{ font-size: 26px; margin: 0 0 4px; }}
h2 {{ font-size: 18px; margin: 40px 0 8px; padding-top: 8px; border-top: 1px solid var(--grid); }}
h3 {{ font-size: 14px; margin: 20px 0 6px; color: var(--ink2); }}
p.meta, p.note {{ color: var(--ink2); margin: 4px 0; }}
p.note {{ font-size: 13px; max-width: 78ch; }}
.tiles {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap:12px; margin: 20px 0; }}
.tile {{ background: var(--surface); border:1px solid var(--border); border-radius:8px; padding:12px 14px; }}
.tile .label {{ color: var(--ink2); font-size: 12px; }}
.tile .value {{ font-size: 24px; font-weight: 600; }}
.tile .sub {{ color: var(--muted); font-size: 12px; }}
figure {{ margin: 12px 0; background: var(--surface); border:1px solid var(--border); border-radius:8px; padding:8px; }}
figure img {{ width: 100%; height: auto; display:block; }}
figcaption {{ color: var(--ink2); font-size: 12px; padding: 4px 4px 0; }}
.pair {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; font-size: 12.5px; font-variant-numeric: tabular-nums; background: var(--surface); }}
th, td {{ padding: 5px 10px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }}
thead th {{ color: var(--ink2); font-weight: 600; border-bottom: 1px solid var(--muted); }}
tbody th {{ text-align: left; font-weight: 600; }}
code {{ font-size: 12.5px; }}
"""


# --------------------------------------------------------------------------------------------
# Tear sheet
# --------------------------------------------------------------------------------------------


class TearSheet:
    def __init__(self, ablation: AblationResult, bars: pd.DataFrame) -> None:
        self._ablation = ablation
        self._bars = bars

    def render(self, out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(self.html(), encoding="utf-8")
        return out_path

    def _curves(self) -> dict[str, pd.Series]:
        ab = self._ablation
        head = ab.results[ab.headline]
        start = head.first_decision
        if start is None:
            raise ValueError("headline run never became ready")
        bench = head.benchmark.loc[head.benchmark.index > start]
        curves: dict[str, pd.Series] = {
            "Overlay (headline)": head.active_equity.loc[bench.index[0] :]
        }
        if "realized_vol_overlay" in ab.results:
            curves["Realised-vol baseline"] = ab.results["realized_vol_overlay"].active_equity.loc[
                bench.index[0] :
            ]
        curves[f"{head.symbol} buy & hold"] = bench
        return curves

    def figures(self) -> dict[str, str]:
        """Every chart as a base64 PNG, keyed by name."""
        ab = self._ablation
        head = ab.results[ab.headline]
        close = self._bars["close"]
        start = head.first_decision
        curves = self._curves()
        transition_mean, transition_std = mean_transition_matrix(head)
        transition_mean.index.name, transition_mean.columns.name = "from", "to"
        return {
            "growth": growth_chart(curves),
            "drawdown": drawdown_chart(curves),
            "rolling": rolling_sharpe_chart(curves),
            "ablation": ablation_chart(ab.table, ab.headline),
            "timeline": regime_timeline_chart(close, head.regimes["regime"]),
            "probs": probability_chart(probability_frame(head)),
            "exposure": exposure_chart(head.equity["weight"].loc[start:]),
            "transition": heatmap(
                transition_mean, fmt=lambda v: f"{v:.3f}", annotations=transition_std
            ),
            "fvs": heatmap(_row_normalize(filtered_vs_smoothed(head)), fmt=lambda v: f"{v:.0%}"),
            "dvr": heatmap(
                _row_normalize(decoded_vs_realized(head, close)), fmt=lambda v: f"{v:.0%}"
            ),
            "var": var_chart(head.var_forecasts),
        }

    def export_figures(self, out_dir: Path, names: Sequence[str] | None = None) -> dict[str, Path]:
        """Write charts as PNG files (e.g. for a README); returns name -> path."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        figures = self.figures()
        written: dict[str, Path] = {}
        for name in names or list(figures):
            path = out_dir / f"{name}.png"
            path.write_bytes(base64.b64decode(figures[name]))
            written[name] = path
        return written

    def html(self) -> str:
        ab = self._ablation
        head = ab.results[ab.headline]
        cfg = head.config
        periods = cfg.run.annualization
        close = self._bars["close"]
        start = head.first_decision
        assert start is not None, "headline run never became ready"
        curves = self._curves()
        bench = curves[f"{head.symbol} buy & hold"]
        summaries = {
            name: summarize(series, None if "buy & hold" in name else bench, periods=periods)
            for name, series in curves.items()
        }

        summary_frame = pd.DataFrame({name: s.as_dict() for name, s in summaries.items()}).T
        summary_cols = [
            "cagr",
            "volatility",
            "sharpe",
            "sharpe_se",
            "psr",
            "sortino",
            "max_drawdown",
            "max_drawdown_duration",
            "calmar",
            "information_ratio",
            "beta",
        ]
        headline_summary = summaries["Overlay (headline)"]
        bench_summary = summaries[f"{head.symbol} buy & hold"]

        episodes = drawdown_episodes(curves["Overlay (headline)"])[:5]
        dd_frame = pd.DataFrame(
            [
                {
                    "peak": e.peak.date(),
                    "trough": e.trough.date(),
                    "recovery": e.recovery.date() if e.recovery is not None else "not yet",
                    "depth": e.depth,
                    "bars to trough": e.bars_to_trough,
                    "duration (bars)": e.duration_bars,
                }
                for e in episodes
            ],
            index=range(1, len(episodes) + 1),
        )

        transition_mean, _ = mean_transition_matrix(head)
        fvs = filtered_vs_smoothed(head)
        dvr = decoded_vs_realized(head, close)
        attribution = regime_attribution(head, close, periods)
        attribution.index = [REGIME_LABELS[i] for i in attribution.index]
        var_table = var_backtest_table(head)
        durations = pd.Series(
            1.0 / (1.0 - np.diag(head.final_params.transmat))
            if head.final_params is not None
            else np.nan,
            index=[REGIME_LABELS[n] for n in REGIME_NAMES],
        )
        interventions = (
            head.decisions["risk_rules"]
            .str.split(",")
            .explode()
            .replace("", np.nan)
            .dropna()
            .value_counts()
        )

        table = ab.table.copy()
        pct_cols = ("cagr", "volatility", "max_drawdown")
        ablation_formats: dict[str, Callable[[object], str]] = {c: _pct for c in pct_cols}
        ablation_formats |= {
            c: _num
            for c in (
                "sharpe",
                "sharpe_se",
                "sortino",
                "calmar",
                "information_ratio",
                "mean_weight",
                "sharpe_diff_vs_headline",
            )
        }
        ablation_formats |= {
            "psr": _pct,
            "dsr": _pct,
            "diff_p_value": lambda v: _num(v, 3),
            "turnover": lambda v: _num(v, 1) + "x",
            "costs": _money,
            "trial": lambda v: "yes" if v else "no",
        }

        tiles = "".join(
            f'<div class="tile"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'
            for label, value, sub in (
                ("CAGR", _pct(headline_summary.cagr), f"buy & hold {_pct(bench_summary.cagr)}"),
                (
                    "Sharpe",
                    _num(headline_summary.sharpe),
                    f"±{_num(headline_summary.sharpe_se)} SE · B&H {_num(bench_summary.sharpe)}",
                ),
                (
                    "Max drawdown",
                    _pct(headline_summary.max_drawdown),
                    f"buy & hold {_pct(bench_summary.max_drawdown)}",
                ),
                (
                    "Volatility",
                    _pct(headline_summary.volatility),
                    f"target {_pct(cfg.sizing.target_vol_annual, 0)}",
                ),
                (
                    "Deflated Sharpe",
                    _pct(float(ab.table["dsr"].astype(float).to_dict()[ab.headline])),
                    f"{int(ab.table['trial'].sum())} trials counted",
                ),
            )
        )

        figures = self.figures()

        def fig(key: str, caption: str) -> str:
            return f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{figures[key]}"><figcaption>{caption}</figcaption></figure>'

        generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        sections = [
            f"<h1>{html.escape(head.symbol)} regime engine: tear sheet</h1>",
            f'<p class="meta">Active period {start.date()} → {head.equity.index[-1].date()} · '
            f"{len(head.refits)} walk-forward HMM fits · config <code>{cfg.fingerprint()}</code> · "
            f"data <code>{head.data_hash}</code> · generated {generated}</p>",
            f'<div class="tiles">{tiles}</div>',
            "<h2>Performance</h2>",
            fig("growth", "Growth of $1 from the first tradable bar, after transaction costs."),
            fig("drawdown", "Underwater curves."),
            fig("rolling", "Rolling one-year Sharpe ratio."),
            '<h3>Summary</h3><div class="scroll">'
            + _table(
                summary_frame[summary_cols],
                {
                    **{c: _pct for c in ("cagr", "volatility", "max_drawdown", "psr")},
                    **{
                        c: _num
                        for c in (
                            "sharpe",
                            "sharpe_se",
                            "sortino",
                            "calmar",
                            "information_ratio",
                            "beta",
                        )
                    },
                    "max_drawdown_duration": lambda v: _num(v, 0),
                },
            )
            + "</div>",
            '<h3>Deepest drawdowns (headline)</h3><div class="scroll">'
            + _table(dd_frame, {"depth": _pct}, index_label="#")
            + "</div>",
            "<h2>Ablation study</h2>",
            '<p class="note">Each variant changes one layer of the engine. DSR deflates the Sharpe '
            "ratio for the number of configurations tried (all rows marked as trials, winners and "
            "losers). The p-value tests the Sharpe difference against the headline with a paired "
            "stationary bootstrap (mean block 21 bars).</p>",
            fig(
                "ablation",
                "Sharpe ratio by variant (± 1 standard error). Blue: headline. Orange: buy & hold.",
            ),
            '<div class="scroll">'
            + _table(table.drop(columns=["description"]), ablation_formats)
            + "</div>",
            '<div class="scroll">' + _table(table[["description"]], {}) + "</div>",
            "<h2>Regimes</h2>",
            fig(
                "timeline", "Close price shaded by the confirmed (causal, post-hysteresis) regime."
            ),
            fig(
                "probs",
                "Filtered regime probabilities P(S_t | x_1..t), the only regime estimate decisions use.",
            ),
            fig("exposure", "Portfolio weight (dashed line = fully invested)."),
            '<div class="pair">'
            + fig("transition", "Mean transition matrix across refits (± standard deviation).")
            + fig(
                "fvs",
                f"Rows: ex-post smoothed regime (final model). Columns: causal regime at the time. Agreement {accuracy(fvs):.0%}.",
            )
            + fig(
                "dvr",
                f"Rows: realised next-21-bar label. Columns: confirmed regime. Agreement {accuracy(dvr):.0%} (chance ≈ 25%).",
            )
            + "</div>",
            '<h3>Transition matrix (mean across refits)</h3><div class="scroll">'
            + _table(
                transition_mean.set_axis([REGIME_LABELS[n] for n in REGIME_NAMES], axis=0).set_axis(
                    [REGIME_LABELS[n] for n in REGIME_NAMES], axis=1
                ),
                {REGIME_LABELS[n]: lambda v: _num(v, 3) for n in REGIME_NAMES},
                index_label="from \\ to",
            )
            + "</div>",
            '<h3>Regime attribution (next-bar returns)</h3><div class="scroll">'
            + _table(
                attribution,
                {
                    **{
                        c: _pct
                        for c in (
                            "share_of_bars",
                            "asset_return_ann",
                            "asset_vol_ann",
                            "strategy_return_ann",
                            "strategy_vol_ann",
                            "strategy_contribution",
                        )
                    },
                    "mean_weight": _num,
                },
            )
            + "</div>",
            '<h3>Expected regime duration under the final model (bars)</h3><div class="scroll">'
            + _table(
                durations.to_frame("expected duration"), {"expected duration": lambda v: _num(v, 1)}
            )
            + "</div>",
            "<h2>Risk</h2>",
            fig("var", "One-bar 99% VaR per unit of exposure against realised next-bar returns."),
            '<h3>VaR backtests</h3><div class="scroll">'
            + _table(
                var_table,
                {
                    "n": lambda v: _num(v, 0),
                    "breaches": lambda v: _num(v, 0),
                    "breach_rate": lambda v: _pct(v, 2),
                    "expected_rate": lambda v: _pct(v, 2),
                    "kupiec_p": lambda v: _num(v, 3),
                    "christoffersen_p": lambda v: _num(v, 3),
                },
            )
            + "</div>",
            '<h3>Risk-desk interventions (decision bars)</h3><div class="scroll">'
            + _table(interventions.to_frame("bars"), {})
            + "</div>",
            "<h2>Method</h2>",
            '<p class="note">Walk-forward 4-state Gaussian HMM on (log return, Parkinson range vol, '
            "volume-change vol), refit every "
            f"{cfg.training.refit_every} bars on an {cfg.training.window_type.value} window; decisions use only "
            "filtered probabilities. Orders decided at the close fill at the next open with spread, "
            "square-root impact and commission. Causality is verified in the test suite by corrupting every bar "
            "after a cutoff and requiring the equity curve up to the cutoff to be bit-identical; an independent "
            "return-space recomputation matches the event-driven ledger to ~1e-15.</p>",
        ]
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(head.symbol)} Regime Tear Sheet</title><style>{_CSS}</style></head>"
            f"<body><main>{''.join(sections)}</main></body></html>"
        )
