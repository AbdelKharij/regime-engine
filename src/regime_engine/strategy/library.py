"""Concrete strategies. Each is a pure function of trailing closes."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from regime_engine.strategy.base import MarketContext


class TimeSeriesMomentum:
    """Volatility-normalised time-series momentum, blended over several lookbacks.

    For lookback L with trailing log return R_L and daily return std s (estimated over the
    longest available lookback), the score is tanh(R_L / (s * sqrt(L))): roughly the t-stat
    of the trend, squashed into (-1, 1). Scores are averaged over lookbacks that fit in the
    history; with no usable lookback the direction is 0. ``damping`` scales the output (used
    for the high-vol bull regime).
    """

    def __init__(
        self,
        lookbacks: Sequence[int] = (21, 63, 252),
        *,
        long_only: bool = True,
        damping: float = 1.0,
        name: str = "momentum",
    ) -> None:
        if not lookbacks or min(lookbacks) < 2:
            raise ValueError("lookbacks must be non-empty, each >= 2")
        if not 0.0 <= damping <= 1.0:
            raise ValueError("damping must be in [0, 1]")
        self._lookbacks = tuple(sorted(lookbacks))
        self._long_only = long_only
        self._damping = damping
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def direction(self, ctx: MarketContext) -> float:
        log_close = np.log(ctx.close)
        usable = [lb for lb in self._lookbacks if lb < len(log_close)]
        if not usable:
            return 0.0
        returns = np.diff(log_close[-(max(usable) + 1) :])
        daily_std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
        if not daily_std > 0.0:
            return 0.0
        scores = [
            math.tanh((log_close[-1] - log_close[-1 - lb]) / (daily_std * math.sqrt(lb)))
            for lb in usable
        ]
        value = self._damping * float(np.mean(scores))
        return max(0.0, value) if self._long_only else value


class ZScoreMeanReversion:
    """direction = -tanh(z / scale), z = (close - SMA_w) / std_w over the trailing window."""

    def __init__(
        self,
        window: int = 20,
        *,
        scale: float = 2.0,
        long_only: bool = False,
        name: str = "mean_reversion",
    ) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        if scale <= 0:
            raise ValueError("scale must be positive")
        self._window = window
        self._scale = scale
        self._long_only = long_only
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def direction(self, ctx: MarketContext) -> float:
        if len(ctx.close) < self._window:
            return 0.0
        window = ctx.close[-self._window :]
        std = float(window.std(ddof=1))
        if not std > 0.0:
            return 0.0
        z = (float(window[-1]) - float(window.mean())) / std
        value = -math.tanh(z / self._scale)
        return max(0.0, value) if self._long_only else value


class ConstantDirection:
    """Fixed conviction regardless of prices (defensive / flat sleeves)."""

    def __init__(self, direction: float, *, name: str = "constant") -> None:
        if not -1.0 <= direction <= 1.0:
            raise ValueError("direction must be in [-1, 1]")
        self._direction = float(direction)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def direction(self, ctx: MarketContext) -> float:
        return self._direction
