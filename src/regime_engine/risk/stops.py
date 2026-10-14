"""ATR-based trailing stop-loss and profit target, evaluated intrabar and gap-aware.

Causality: the stop and target levels checked on bar t were set at the close of bar t-1.
Within a bar, the order of the high and the low is unknown, so if both the stop and the
target are touched, the **stop wins** (the conservative assumption). If the bar opens beyond
the level (a gap), the fill is at the open, not the level.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from regime_engine.core.types import Side


def average_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Simple moving average of the true range over the trailing ``window`` bars (causal)."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1, skipna=True)
    return true_range.rolling(window, min_periods=window).mean().rename("atr")


class StopKind(str, Enum):
    STOP_LOSS = "stop_loss"
    PROFIT_TARGET = "profit_target"


@dataclass(frozen=True, slots=True)
class StopTrigger:
    ts: pd.Timestamp
    kind: StopKind
    side: Side
    level: float
    fill_price: float


class StopManager:
    """Tracks protective levels for a single net position."""

    def __init__(
        self, trailing_atr_mult: float | None = 3.0, target_atr_mult: float | None = None
    ) -> None:
        for value in (trailing_atr_mult, target_atr_mult):
            if value is not None and value <= 0:
                raise ValueError("ATR multiples must be positive")
        self._k_stop = trailing_atr_mult
        self._k_target = target_atr_mult
        self.reset()

    def reset(self) -> None:
        self._side: Side | None = None
        self._entry = float("nan")
        self._extreme = float("nan")
        self._stop: float | None = None
        self._target: float | None = None

    @property
    def side(self) -> Side | None:
        return self._side

    @property
    def stop_level(self) -> float | None:
        return self._stop

    @property
    def target_level(self) -> float | None:
        return self._target

    def on_position(self, weight: float, price: float, atr: float) -> None:
        """Call after the position changes. Levels are set only when a new position opens.

        Resizing in the same direction keeps the existing levels (the stop protects the
        trade, not the latest add-on). Going flat clears them; a sign flip re-opens.
        """
        side = None if weight == 0.0 else (Side.LONG if weight > 0 else Side.SHORT)
        if side is None:
            self.reset()
            return
        if side is self._side:
            return
        self._side, self._entry, self._extreme = side, price, price
        usable_atr = atr if np.isfinite(atr) and atr > 0 else None
        self._stop = (
            price - side * self._k_stop * usable_atr
            if self._k_stop is not None and usable_atr is not None
            else None
        )
        self._target = (
            price + side * self._k_target * usable_atr
            if self._k_target is not None and usable_atr is not None
            else None
        )

    def check_bar(
        self, ts: pd.Timestamp, open_: float, high: float, low: float
    ) -> StopTrigger | None:
        """Check bar ``ts`` against the levels set at the previous close."""
        if self._side is None:
            return None
        trigger: StopTrigger | None = None
        if self._side is Side.LONG:
            if self._stop is not None and low <= self._stop:
                trigger = StopTrigger(
                    ts, StopKind.STOP_LOSS, Side.LONG, self._stop, min(open_, self._stop)
                )
            elif self._target is not None and high >= self._target:
                trigger = StopTrigger(
                    ts, StopKind.PROFIT_TARGET, Side.LONG, self._target, max(open_, self._target)
                )
        else:
            if self._stop is not None and high >= self._stop:
                trigger = StopTrigger(
                    ts, StopKind.STOP_LOSS, Side.SHORT, self._stop, max(open_, self._stop)
                )
            elif self._target is not None and low <= self._target:
                trigger = StopTrigger(
                    ts, StopKind.PROFIT_TARGET, Side.SHORT, self._target, min(open_, self._target)
                )
        if trigger is not None:
            self.reset()
        return trigger

    def on_close(self, high: float, low: float, atr: float) -> None:
        """Ratchet the trailing stop with this bar's extreme. The stop never loosens."""
        if self._side is None or self._k_stop is None or not (np.isfinite(atr) and atr > 0):
            return
        if self._side is Side.LONG:
            self._extreme = max(self._extreme, high)
            candidate = self._extreme - self._k_stop * atr
            self._stop = candidate if self._stop is None else max(self._stop, candidate)
        else:
            self._extreme = min(self._extreme, low)
            candidate = self._extreme + self._k_stop * atr
            self._stop = candidate if self._stop is None else min(self._stop, candidate)
