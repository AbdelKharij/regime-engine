"""Single-asset cash + position ledger.

Short-sale proceeds are credited to cash, and shorts accrue a borrow fee on their market
value every bar. Cash earns nothing (risk-free rate 0), consistent with metrics computed
against r_f = 0. Fractional shares are allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from regime_engine.execution.simulator import Fill


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    ts: pd.Timestamp
    cash: float
    quantity: float
    price: float
    equity: float
    weight: float
    drawdown: float


class Portfolio:
    def __init__(
        self,
        initial_cash: float,
        *,
        borrow_fee_bps_annual: float = 0.0,
        annualization: int = 252,
    ) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.cash = float(initial_cash)
        self.quantity = 0.0
        self._borrow_rate = borrow_fee_bps_annual * 1e-4 / annualization
        self._peak = float(initial_cash)
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_borrow = 0.0
        self.n_fills = 0

    def equity(self, price: float) -> float:
        return self.cash + self.quantity * price

    def weight(self, price: float) -> float:
        equity = self.equity(price)
        return 0.0 if equity <= 0 else self.quantity * price / equity

    def apply(self, fill: Fill) -> None:
        self.cash -= fill.signed_quantity * fill.fill_price + fill.commission
        self.quantity += fill.signed_quantity
        if abs(self.quantity) < 1e-9:
            self.quantity = 0.0
        self.total_commission += fill.commission
        self.total_slippage += fill.slippage_cost
        self.n_fills += 1

    def accrue_borrow(self, price: float) -> float:
        """Charge one bar of borrow fee on a short position. Returns the fee."""
        if self.quantity >= 0 or self._borrow_rate == 0.0:
            return 0.0
        fee = -self.quantity * price * self._borrow_rate
        self.cash -= fee
        self.total_borrow += fee
        return fee

    def mark(self, ts: pd.Timestamp, price: float) -> PortfolioSnapshot:
        equity = self.equity(price)
        self._peak = max(self._peak, equity)
        return PortfolioSnapshot(
            ts=ts,
            cash=self.cash,
            quantity=self.quantity,
            price=price,
            equity=equity,
            weight=self.weight(price),
            drawdown=1.0 - equity / self._peak if self._peak > 0 else 0.0,
        )
