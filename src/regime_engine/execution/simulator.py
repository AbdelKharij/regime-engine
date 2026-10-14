"""Order execution simulation with latency.

Orders are *target-weight* orders. A decision made at the close of bar t becomes eligible at
bar t + ``latency_bars`` and is resolved at that bar's open:
    Q_target = w * equity(open) / open,  trade = Q_target - Q_held.
Sizing an order off the open of the bar it fills on is legitimate, because that price is
known when the order executes. Protective exits (stops) bypass the queue and fill
immediately at their trigger price.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from regime_engine.core.types import Side
from regime_engine.execution.costs import CostModel, MarketConditions
from regime_engine.portfolio.portfolio import Portfolio
from regime_engine.utils.logging import LogEvent, get_logger

_LOG = get_logger("execution.simulator")
_MIN_QUANTITY = 1e-9


class OrderReason(str, Enum):
    REBALANCE = "rebalance"
    STOP_LOSS = "stop_loss"
    PROFIT_TARGET = "profit_target"


@dataclass(frozen=True, slots=True)
class TargetOrder:
    order_id: str
    created_ts: pd.Timestamp
    eligible_index: int  # position of the bar at whose open the order fills
    target_weight: float
    reason: OrderReason


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: str
    ts: pd.Timestamp
    side: Side
    quantity: float  # > 0
    ref_price: float
    fill_price: float
    commission: float
    reason: OrderReason

    @property
    def signed_quantity(self) -> float:
        return int(self.side) * self.quantity

    @property
    def slippage_cost(self) -> float:
        return self.quantity * abs(self.fill_price - self.ref_price)

    @property
    def notional(self) -> float:
        return self.quantity * self.fill_price


class ExecutionSimulator:
    def __init__(self, cost_model: CostModel, latency_bars: int = 1) -> None:
        if latency_bars < 1:
            raise ValueError("latency_bars must be >= 1 (same-bar fills are look-ahead)")
        self._costs = cost_model
        self._latency = latency_bars
        self._pending: list[TargetOrder] = []
        self._ids = itertools.count(1)

    @property
    def latency_bars(self) -> int:
        return self._latency

    @property
    def pending(self) -> tuple[TargetOrder, ...]:
        return tuple(self._pending)

    def submit_target(
        self,
        created_ts: pd.Timestamp,
        bar_index: int,
        target_weight: float,
        reason: OrderReason = OrderReason.REBALANCE,
    ) -> TargetOrder:
        order = TargetOrder(
            order_id=f"O{next(self._ids):06d}",
            created_ts=created_ts,
            eligible_index=bar_index + self._latency,
            target_weight=target_weight,
            reason=reason,
        )
        self._pending.append(order)
        _LOG.debug(
            LogEvent.ORDER_SUBMITTED,
            "target order queued",
            order_id=order.order_id,
            target_weight=target_weight,
            eligible_index=order.eligible_index,
        )
        return order

    def cancel_pending(self) -> int:
        n = len(self._pending)
        self._pending.clear()
        return n

    def fill_due(
        self,
        bar_index: int,
        ts: pd.Timestamp,
        open_price: float,
        portfolio: Portfolio,
        market: MarketConditions,
    ) -> list[Fill]:
        """Resolve every order eligible at this bar's open, oldest first."""
        due = [o for o in self._pending if o.eligible_index <= bar_index]
        self._pending = [o for o in self._pending if o.eligible_index > bar_index]
        fills: list[Fill] = []
        for order in due:
            target_qty = order.target_weight * portfolio.equity(open_price) / open_price
            delta = target_qty - portfolio.quantity
            if abs(delta) < _MIN_QUANTITY:
                continue
            fill = self.execute(order.order_id, ts, delta, open_price, market, order.reason)
            portfolio.apply(fill)
            fills.append(fill)
        return fills

    def execute(
        self,
        order_id: str,
        ts: pd.Timestamp,
        signed_quantity: float,
        ref_price: float,
        market: MarketConditions,
        reason: OrderReason,
    ) -> Fill:
        side = Side.LONG if signed_quantity > 0 else Side.SHORT
        quantity = abs(signed_quantity)
        price = self._costs.fill_price(side, quantity, ref_price, market)
        fill = Fill(
            order_id=order_id,
            ts=ts,
            side=side,
            quantity=quantity,
            ref_price=ref_price,
            fill_price=price,
            commission=self._costs.commission(quantity, price),
            reason=reason,
        )
        _LOG.debug(
            LogEvent.ORDER_FILLED,
            f"{side.name} {quantity:.4f} @ {price:.4f}",
            order_id=order_id,
            reason=reason,
            ref_price=ref_price,
            commission=fill.commission,
            slippage=fill.slippage_cost,
        )
        return fill

    def next_id(self) -> str:
        return f"O{next(self._ids):06d}"
