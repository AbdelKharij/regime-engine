"""Ablation study: what does each layer of the engine contribute?

Every variant is a set of config overrides on the base config. Variants that share the
regime-relevant settings (``REGIME_SECTIONS``) replay one precomputed ``RegimeTrack``, so the
HMM is estimated once per distinct regime setup rather than once per variant.

Honesty rules built into the table:
* The Deflated Sharpe Ratio counts **every** variant marked ``counts_as_trial``, including the
  ones that did badly, because a configuration was "tried" whether or not it won.
* Each variant's Sharpe difference against the headline comes with a paired
  stationary-bootstrap p-value.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from regime_engine.analytics.metrics import (
    bootstrap_sharpe_difference,
    deflated_sharpe_ratio,
    sharpe_ratio,
    summarize,
)
from regime_engine.backtest.engine import Backtester, BacktestResult
from regime_engine.core.config import EngineConfig, StrategyConfig
from regime_engine.core.exceptions import ConfigError
from regime_engine.data.features import FeatureEngine
from regime_engine.models.hmm_engine import RegimeTrack, regime_fingerprint
from regime_engine.utils.logging import get_logger

_LOG = get_logger("analytics.experiments")

_TACTICAL: dict[str, Any] = {
    key: value.value
    for key, value in {
        "bull_low": StrategyConfig.tactical().bull_low,
        "bull_high": StrategyConfig.tactical().bull_high,
        "bear_low": StrategyConfig.tactical().bear_low,
        "bear_high": StrategyConfig.tactical().bear_high,
    }.items()
}


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    counts_as_trial: bool = True


HEADLINE = "overlay"

DEFAULT_VARIANTS: tuple[Variant, ...] = (
    Variant(HEADLINE, "Always long, HMM regime-mixture vol targeting (headline)"),
    Variant(
        "realized_vol_overlay",
        "Always long, trailing realised-vol targeting (no-HMM baseline)",
        {"sizing": {"method": "realized_vol_target"}},
    ),
    Variant("overlay_kelly", "Overlay + 0.25-Kelly cap", {"sizing": {"kelly_cap": True}}),
    Variant(
        "overlay_stops", "Overlay + 3-ATR trailing stops", {"risk": {"trailing_stop_atr": 3.0}}
    ),
    Variant(
        "overlay_no_cooldown", "Overlay without the regime-flip cooldown", {"cooldown": {"bars": 0}}
    ),
    Variant(
        "overlay_rolling",
        "Overlay, rolling 1260-bar training window",
        {"training": {"window_type": "rolling"}},
    ),
    Variant(
        "tactical", "Regime-routed strategies (momentum / MR / defensive)", {"strategy": _TACTICAL}
    ),
    Variant(
        "tactical_blended",
        "Regime-routed strategies, probability-blended",
        {"strategy": {**_TACTICAL, "routing": "blended"}},
    ),
    Variant(
        "tactical_stops",
        "Original design: regime-routed + 3-ATR stops",
        {"strategy": _TACTICAL, "risk": {"trailing_stop_atr": 3.0}},
    ),
    Variant(
        "overlay_zero_cost",
        "Overlay with zero transaction costs (sensitivity, not a trial)",
        {
            "execution": {
                "commission_bps": 0.0,
                "min_commission": 0.0,
                "half_spread_bps": 0.0,
                "impact_coefficient": 0.0,
                "borrow_fee_bps_annual": 0.0,
            }
        },
        counts_as_trial=False,
    ),
)


def apply_overrides(base: EngineConfig, overrides: Mapping[str, Mapping[str, Any]]) -> EngineConfig:
    raw = base.to_dict()
    for section, values in overrides.items():
        if section not in raw:
            raise ConfigError(f"unknown config section {section!r} in overrides")
        raw[section] = {**raw[section], **values}
    return EngineConfig.from_dict(raw)


def turnover(result: BacktestResult, periods: int = 252) -> float:
    """Annual traded notional as a multiple of average equity (active period)."""
    if result.fills.empty or result.first_decision is None:
        return 0.0
    fills = result.fills[result.fills["ts"] >= result.first_decision]
    equity = result.active_equity
    years = max(len(equity) - 1, 1) / periods
    return float((fills["quantity"] * fills["fill_price"]).sum() / equity.mean() / years)


@dataclass(frozen=True, eq=False)
class AblationResult:
    variants: tuple[Variant, ...]
    results: Mapping[str, BacktestResult]
    table: pd.DataFrame
    headline: str


def run_ablation(
    base: EngineConfig,
    bars: pd.DataFrame,
    variants: Sequence[Variant] = DEFAULT_VARIANTS,
    *,
    headline: str = HEADLINE,
    n_bootstrap: int = 2000,
    progress: Callable[[str], None] | None = None,
) -> AblationResult:
    if headline not in {v.name for v in variants}:
        raise ValueError(f"headline variant {headline!r} is not in the variant list")
    periods = base.run.annualization
    notify = progress or (lambda message: None)
    tracks: dict[str, RegimeTrack] = {}
    results: dict[str, BacktestResult] = {}
    features_cache: dict[str, pd.DataFrame] = {}

    for variant in variants:
        cfg = apply_overrides(base, variant.overrides)
        key = regime_fingerprint(cfg)
        if key not in tracks:
            started = time.perf_counter()
            feature_key = repr(cfg.features)
            if feature_key not in features_cache:
                features_cache[feature_key] = FeatureEngine(cfg.features).compute(bars)
            tracks[key] = RegimeTrack.build(cfg, features_cache[feature_key])
            notify(f"regime track {key} built in {time.perf_counter() - started:.0f}s")
        started = time.perf_counter()
        results[variant.name] = Backtester(cfg).run(bars, track=tracks[key])
        notify(f"{variant.name}: {time.perf_counter() - started:.1f}s")

    head = results[headline]
    start = head.first_decision
    benchmark = (
        head.benchmark.loc[head.benchmark.index > start] if start is not None else head.benchmark
    )
    trial_sharpes = [
        sharpe_ratio(results[v.name].returns(), periods) for v in variants if v.counts_as_trial
    ]
    head_returns = head.returns()

    rows = []
    for variant in variants:
        res = results[variant.name]
        summary = summarize(res.active_equity, benchmark, periods=periods)
        returns = res.returns()
        diff = (
            None
            if variant.name == headline
            else bootstrap_sharpe_difference(
                returns, head_returns, periods=periods, n_bootstrap=n_bootstrap
            )
        )
        weights = (
            res.equity["weight"].loc[res.first_decision :]
            if res.first_decision is not None
            else res.equity["weight"]
        )
        rows.append(
            {
                "variant": variant.name,
                "description": variant.description,
                "cagr": summary.cagr,
                "volatility": summary.volatility,
                "sharpe": summary.sharpe,
                "sharpe_se": summary.sharpe_se,
                "psr": summary.psr,
                "dsr": deflated_sharpe_ratio(returns, trial_sharpes, periods)
                if variant.counts_as_trial
                else np.nan,
                "sortino": summary.sortino,
                "max_drawdown": summary.max_drawdown,
                "calmar": summary.calmar,
                "information_ratio": summary.information_ratio,
                "mean_weight": float(weights.mean()),
                "turnover": turnover(res, periods),
                "costs": float(sum(res.costs.values())),
                "sharpe_diff_vs_headline": np.nan if diff is None else diff.difference,
                "diff_p_value": np.nan if diff is None else diff.p_value,
                "trial": variant.counts_as_trial,
            }
        )

    bench_summary = summarize(benchmark, periods=periods)
    bench_diff = bootstrap_sharpe_difference(
        benchmark.pct_change().dropna(), head_returns, periods=periods, n_bootstrap=n_bootstrap
    )
    rows.append(
        {
            "variant": "buy_and_hold",
            "description": f"{head.symbol} buy & hold (benchmark)",
            "cagr": bench_summary.cagr,
            "volatility": bench_summary.volatility,
            "sharpe": bench_summary.sharpe,
            "sharpe_se": bench_summary.sharpe_se,
            "psr": bench_summary.psr,
            "dsr": np.nan,
            "sortino": bench_summary.sortino,
            "max_drawdown": bench_summary.max_drawdown,
            "calmar": bench_summary.calmar,
            "information_ratio": np.nan,
            "mean_weight": 1.0,
            "turnover": 0.0,
            "costs": np.nan,
            "sharpe_diff_vs_headline": bench_diff.difference,
            "diff_p_value": bench_diff.p_value,
            "trial": False,
        }
    )
    table = pd.DataFrame(rows).set_index("variant")
    _LOG.info(
        "ABLATION_COMPLETED", "ablation study completed", variants=len(variants), tracks=len(tracks)
    )
    return AblationResult(tuple(variants), results, table, headline)
