"""Risk rules: each takes a proposed weight and the risk state and returns an allowed weight.

Rules are evaluated in order by ``RiskManager``. ``apply`` must be called exactly once per bar
for stateful rules (drawdown breaker), and ``tick`` once at the start of every bar for
countdown rules (cooldowns).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from regime_engine.core.config import CooldownMode
from regime_engine.utils.logging import LogEvent, get_logger

_LOG = get_logger("risk.rules")


@dataclass(frozen=True, slots=True)
class RiskState:
    ts: pd.Timestamp
    equity: float
    current_weight: float  # signed exposure / equity held going into the decision
    drawdown: float  # 1 - equity / running peak (portfolio level, for reporting)
    unit_var: float | None = None  # one-bar VaR per unit of exposure (loss fraction)


class RiskRule(Protocol):
    @property
    def name(self) -> str: ...

    def apply(self, weight: float, state: RiskState) -> tuple[float, str]:
        """Return (allowed weight, reason). The reason is empty if nothing changed."""
        ...


def reduce_only(proposed: float, current: float) -> float:
    """Largest-allowed move toward ``proposed`` that does not increase or flip exposure."""
    if proposed * current < 0.0:
        return 0.0
    if abs(proposed) > abs(current):
        return current
    return proposed


class CooldownRule:
    """Restricts trading for ``n_bars`` bars after ``arm()`` (inclusive of the arming bar).

    REDUCE_ONLY: only trades that shrink |exposure| without flipping sign are allowed.
    LITERAL_FREEZE: the current weight is held unchanged.
    Protective exits (stops, drawdown breaker) are separate and are never blocked.
    """

    def __init__(
        self, n_bars: int, mode: CooldownMode = CooldownMode.REDUCE_ONLY, *, name: str = "cooldown"
    ) -> None:
        if n_bars < 0:
            raise ValueError("n_bars must be >= 0")
        self._n_bars = n_bars
        self._mode = mode
        self._name = name
        self._remaining = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def active(self) -> bool:
        return self._remaining > 0

    @property
    def remaining(self) -> int:
        return self._remaining

    def tick(self) -> bool:
        """Advance one bar. Returns True if the cooldown just expired."""
        if self._remaining > 0:
            self._remaining -= 1
            return self._remaining == 0
        return False

    def arm(self) -> None:
        self._remaining = self._n_bars

    def apply(self, weight: float, state: RiskState) -> tuple[float, str]:
        if not self.active:
            return weight, ""
        if self._mode is CooldownMode.LITERAL_FREEZE:
            allowed = state.current_weight
            reason = f"frozen ({self._remaining} bar(s) left)"
        else:
            allowed = reduce_only(weight, state.current_weight)
            reason = f"reduce-only ({self._remaining} bar(s) left)"
        return allowed, reason if allowed != weight else ""


class DrawdownBreaker:
    """Flatten and halt new risk once equity falls ``threshold`` below its reference peak.

    The halt lasts ``halt_bars`` bars (including the tripping bar). When it ends, the reference
    peak is reset to the current equity, so the breaker re-arms instead of tripping again on
    the drawdown it already acted on.
    """

    def __init__(self, threshold: float, halt_bars: int, *, name: str = "drawdown_breaker") -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        if halt_bars < 1:
            raise ValueError("halt_bars must be >= 1")
        self._threshold = threshold
        self._halt_bars = halt_bars
        self._name = name
        self._peak: float | None = None
        self._halt_remaining = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def halted(self) -> bool:
        return self._halt_remaining > 0

    def apply(self, weight: float, state: RiskState) -> tuple[float, str]:
        if self._peak is None:
            self._peak = state.equity
        if self._halt_remaining > 0:
            self._halt_remaining -= 1
            if self._halt_remaining > 0:
                return 0.0, f"halted ({self._halt_remaining} bar(s) left)" if weight else ""
            self._peak = state.equity
            _LOG.info(
                LogEvent.DD_BREAKER_REARMED,
                "drawdown breaker re-armed",
                ts=state.ts,
                peak=self._peak,
            )
        self._peak = max(self._peak, state.equity)
        drawdown = 1.0 - state.equity / self._peak
        if drawdown >= self._threshold:
            self._halt_remaining = self._halt_bars
            _LOG.warning(
                LogEvent.DD_BREAKER_TRIPPED,
                "drawdown breaker tripped: flattening",
                ts=state.ts,
                drawdown=drawdown,
                threshold=self._threshold,
                halt_bars=self._halt_bars,
            )
            return 0.0, f"tripped at drawdown {drawdown:.1%}"
        return weight, ""


class VaRBudgetRule:
    """Scale |weight| so that one-bar VaR (per unit exposure x |weight|) <= ``budget``."""

    def __init__(self, budget: float, *, name: str = "var_budget") -> None:
        if not 0.0 < budget < 1.0:
            raise ValueError("budget must be in (0, 1)")
        self._budget = budget
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def apply(self, weight: float, state: RiskState) -> tuple[float, str]:
        unit_var = state.unit_var
        if unit_var is None or unit_var <= 0.0 or weight == 0.0:
            return weight, ""
        limit = self._budget / unit_var
        if abs(weight) <= limit:
            return weight, ""
        return math.copysign(limit, weight), (
            f"VaR {abs(weight) * unit_var:.2%} > budget {self._budget:.2%}"
        )


class LeverageRule:
    def __init__(
        self, max_leverage: float, *, allow_short: bool = False, name: str = "leverage"
    ) -> None:
        if max_leverage <= 0:
            raise ValueError("max_leverage must be positive")
        self._max = max_leverage
        self._allow_short = allow_short
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def apply(self, weight: float, state: RiskState) -> tuple[float, str]:
        lower = -self._max if self._allow_short else 0.0
        allowed = min(max(weight, lower), self._max)
        if allowed == weight:
            return weight, ""
        return allowed, "short selling disabled" if weight < 0 <= lower else (
            f"leverage capped at {self._max:g}"
        )
