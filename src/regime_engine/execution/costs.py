"""Transaction cost models.

For an order of Q shares at reference price P (the open of the fill bar, or a stop level):

    fill price  = P * (1 + side * (h + eta * sigma_d * sqrt(|Q| / ADV)))
    commission  = max(c_min, c_bps * |Q| * fill price)

h is the half-spread, and the square-root term is the standard market-impact law (impact
grows with the square root of participation). sigma_d and ADV are trailing estimates known
*before* the bar opens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from regime_engine.core.config import ExecutionCostConfig
from regime_engine.core.types import Side

_BPS = 1e-4


@dataclass(frozen=True, slots=True)
class MarketConditions:
    daily_vol: float  # std of daily log returns over the trailing window, as of the prior close
    adv: float  # average daily volume (shares) over the trailing window, as of the prior close


class CostModel(Protocol):
    def fill_price(
        self, side: Side, quantity: float, ref_price: float, market: MarketConditions
    ) -> float: ...

    def commission(self, quantity: float, price: float) -> float: ...


class SquareRootImpactCostModel:
    def __init__(
        self,
        *,
        half_spread_bps: float = 1.0,
        impact_coefficient: float = 0.1,
        commission_bps: float = 0.5,
        min_commission: float = 1.0,
    ) -> None:
        for value in (half_spread_bps, impact_coefficient, commission_bps, min_commission):
            if value < 0:
                raise ValueError("cost parameters must be non-negative")
        self._half_spread = half_spread_bps * _BPS
        self._eta = impact_coefficient
        self._commission_rate = commission_bps * _BPS
        self._min_commission = min_commission

    @classmethod
    def from_config(cls, config: ExecutionCostConfig) -> SquareRootImpactCostModel:
        return cls(
            half_spread_bps=config.half_spread_bps,
            impact_coefficient=config.impact_coefficient,
            commission_bps=config.commission_bps,
            min_commission=config.min_commission,
        )

    @property
    def half_spread(self) -> float:
        return self._half_spread

    @property
    def commission_rate(self) -> float:
        return self._commission_rate

    def impact(self, quantity: float, market: MarketConditions) -> float:
        """Fractional price impact. Zero when volatility or ADV is not yet known."""
        if not (math.isfinite(market.daily_vol) and math.isfinite(market.adv)) or market.adv <= 0:
            return 0.0
        return self._eta * market.daily_vol * math.sqrt(abs(quantity) / market.adv)

    def fill_price(
        self, side: Side, quantity: float, ref_price: float, market: MarketConditions
    ) -> float:
        return ref_price * (1.0 + int(side) * (self._half_spread + self.impact(quantity, market)))

    def commission(self, quantity: float, price: float) -> float:
        if quantity == 0.0:
            return 0.0
        return max(self._min_commission, self._commission_rate * abs(quantity) * price)
