"""Event-driven backtester and an independent return-space cross-check."""

from regime_engine.backtest.engine import Backtester, BacktestResult, return_space_equity

__all__ = ["BacktestResult", "Backtester", "return_space_equity"]
