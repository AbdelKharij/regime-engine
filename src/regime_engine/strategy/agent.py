"""Regime -> strategy router."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from regime_engine.core.config import RoutingMode, StrategyConfig, StrategyKind
from regime_engine.core.types import Regime, Signal
from regime_engine.strategy.base import MarketContext, Strategy
from regime_engine.strategy.library import (
    ConstantDirection,
    TimeSeriesMomentum,
    ZScoreMeanReversion,
)


class StrategyAgent:
    """Turns the current regime into a ``Signal``.

    HARD routing runs the strategy of the *confirmed* regime (after hysteresis). BLENDED
    routing runs all four and weights their directions by the filtered regime probabilities,
    which turns regime transitions into gradual changes instead of jumps.
    """

    def __init__(
        self, routing: Mapping[Regime, Strategy], mode: RoutingMode = RoutingMode.HARD
    ) -> None:
        missing = [r.name for r in Regime if r not in routing]
        if missing:
            raise ValueError(f"no strategy routed for {missing}")
        self._routing = MappingProxyType(dict(routing))
        self._mode = mode

    @property
    def routing(self) -> Mapping[Regime, Strategy]:
        return self._routing

    def signal(self, ctx: MarketContext) -> Signal:
        regime = ctx.regime.regime
        if self._mode is RoutingMode.HARD:
            strategy = self._routing[regime]
            direction, name = strategy.direction(ctx), strategy.name
        else:
            directions = np.array([self._routing[r].direction(ctx) for r in Regime])
            direction = float(ctx.regime.probs @ directions)
            name = "blended"
        return Signal(ctx.ts, float(np.clip(direction, -1.0, 1.0)), name, regime)


def build_strategy(kind: StrategyKind, config: StrategyConfig, *, damping: float = 1.0) -> Strategy:
    if kind is StrategyKind.LONG:
        return ConstantDirection(1.0, name="long")
    if kind is StrategyKind.MOMENTUM:
        return TimeSeriesMomentum(
            config.momentum_lookbacks,
            long_only=config.momentum_long_only,
            damping=damping,
            name="momentum" if damping == 1.0 else f"momentum_x{damping:g}",
        )
    if kind is StrategyKind.MEAN_REVERSION:
        return ZScoreMeanReversion(config.mean_reversion_window, scale=config.mean_reversion_scale)
    if kind is StrategyKind.DEFENSIVE:
        return ConstantDirection(config.defensive_direction, name="defensive")
    if kind is StrategyKind.FLAT:
        return ConstantDirection(0.0, name="flat")
    raise ValueError(f"unknown strategy kind {kind!r}")  # pragma: no cover


def build_strategy_agent(config: StrategyConfig) -> StrategyAgent:
    routing: dict[Regime, Strategy] = {
        Regime.BULL_LOW_VOL: build_strategy(config.bull_low, config),
        Regime.BULL_HIGH_VOL: build_strategy(
            config.bull_high, config, damping=config.bull_high_damping
        ),
        Regime.BEAR_LOW_VOL: build_strategy(config.bear_low, config),
        Regime.BEAR_HIGH_VOL: build_strategy(config.bear_high, config),
    }
    return StrategyAgent(routing, config.routing)
