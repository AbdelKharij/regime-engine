"""Phase 5 tests: costs, portfolio ledger, execution latency, FHS VaR, and the backtester."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from regime_engine.backtest import Backtester, BacktestResult, return_space_equity
from regime_engine.core.config import EngineConfig
from regime_engine.core.types import Side
from regime_engine.data.lookahead import LookAheadSanityChecker
from regime_engine.execution import (
    ExecutionSimulator,
    Fill,
    MarketConditions,
    OrderReason,
    SquareRootImpactCostModel,
)
from regime_engine.portfolio import Portfolio
from regime_engine.risk import FilteredHistoricalSimulation

from .conftest import OHLCVFactory

TS = pd.Timestamp("2024-01-02", tz="UTC")
NO_MARKET = MarketConditions(float("nan"), float("nan"))


def _fill(side: Side, qty: float, price: float, commission: float = 0.0) -> Fill:
    return Fill("O1", TS, side, qty, price, price, commission, OrderReason.REBALANCE)


def _config(**overrides: dict[str, object]) -> EngineConfig:
    raw: dict[str, dict[str, object]] = {
        "training": {"min_train_bars": 300, "refit_every": 100},
        "hmm": {"n_restarts": 2, "refit_restarts": 1},
        "risk": {"fhs_min_obs": 50, "trailing_stop_atr": 3.0},
        "strategy": {
            "bull_low": "momentum",
            "bull_high": "momentum",
            "bear_low": "mean_reversion",
            "bear_high": "defensive",
        },
    }
    for section, values in overrides.items():
        raw.setdefault(section, {}).update(values)
    return EngineConfig.from_dict(raw)


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    from .conftest import _make_ohlcv

    return _make_ohlcv(900, seed=3)


@pytest.fixture(scope="module")
def result(bars: pd.DataFrame) -> BacktestResult:
    return Backtester(_config()).run(bars)


# ============================================================================================
# Costs, ledger, execution
# ============================================================================================


class TestCosts:
    def test_fill_price_components(self) -> None:
        model = SquareRootImpactCostModel(half_spread_bps=2.0, impact_coefficient=0.5)
        market = MarketConditions(daily_vol=0.01, adv=1_000_000)
        impact = 0.5 * 0.01 * math.sqrt(10_000 / 1_000_000)
        assert model.fill_price(Side.LONG, 10_000, 100.0, market) == pytest.approx(
            100.0 * (1 + 2e-4 + impact)
        )
        assert model.fill_price(Side.SHORT, 10_000, 100.0, market) == pytest.approx(
            100.0 * (1 - 2e-4 - impact)
        )
        # Square-root law: 4x the size costs 2x the impact.
        assert model.impact(40_000, market) == pytest.approx(2 * model.impact(10_000, market))
        assert model.impact(10_000, NO_MARKET) == 0.0

    def test_commission(self) -> None:
        model = SquareRootImpactCostModel(commission_bps=0.5, min_commission=1.0)
        assert model.commission(10, 100.0) == 1.0  # minimum
        assert model.commission(100_000, 100.0) == pytest.approx(500.0)
        assert model.commission(0.0, 100.0) == 0.0


class TestPortfolio:
    def test_round_trip_accounting(self) -> None:
        book = Portfolio(10_000.0)
        book.apply(_fill(Side.LONG, 50, 100.0, commission=1.0))
        assert book.cash == pytest.approx(10_000 - 5_000 - 1)
        assert book.equity(110.0) == pytest.approx(10_000 - 1 + 500)
        assert book.weight(110.0) == pytest.approx(5_500 / 10_499)
        book.apply(_fill(Side.SHORT, 50, 110.0, commission=1.0))
        assert book.quantity == 0.0
        assert book.cash == pytest.approx(10_000 + 500 - 2)

    def test_short_borrow_and_drawdown(self) -> None:
        book = Portfolio(10_000.0, borrow_fee_bps_annual=252.0, annualization=252)
        book.apply(_fill(Side.SHORT, 10, 100.0))
        assert book.cash == pytest.approx(11_000.0)
        assert book.accrue_borrow(100.0) == pytest.approx(10 * 100 * 1e-4)
        snap = book.mark(TS, 120.0)
        assert snap.weight < 0.0
        assert snap.drawdown == pytest.approx(1 - snap.equity / 10_000.0)


class TestExecution:
    def test_latency_and_target_resolution(self) -> None:
        sim = ExecutionSimulator(
            SquareRootImpactCostModel(min_commission=0.0, commission_bps=0.0, half_spread_bps=0.0),
            2,
        )
        book = Portfolio(10_000.0)
        sim.submit_target(TS, bar_index=5, target_weight=0.5)
        assert sim.fill_due(6, TS, 100.0, book, NO_MARKET) == []  # not yet eligible
        fills = sim.fill_due(7, TS, 100.0, book, NO_MARKET)
        assert len(fills) == 1
        assert book.quantity == pytest.approx(0.5 * 10_000 / 100.0)
        assert sim.pending == ()

    def test_orders_resolve_oldest_first_and_cancel(self) -> None:
        sim = ExecutionSimulator(
            SquareRootImpactCostModel(min_commission=0.0, commission_bps=0.0, half_spread_bps=0.0),
            1,
        )
        book = Portfolio(10_000.0)
        sim.submit_target(TS, 0, 1.0)
        sim.submit_target(TS, 0, 0.25)
        fills = sim.fill_due(1, TS, 50.0, book, NO_MARKET)
        assert [f.side for f in fills] == [Side.LONG, Side.SHORT]
        assert book.weight(50.0) == pytest.approx(0.25)
        sim.submit_target(TS, 1, 1.0)
        assert sim.cancel_pending() == 1
        assert sim.fill_due(5, TS, 50.0, book, NO_MARKET) == []

    def test_same_bar_execution_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="look-ahead"):
            ExecutionSimulator(SquareRootImpactCostModel(), 0)


# ============================================================================================
# Filtered historical simulation
# ============================================================================================


class TestFHS:
    def test_scales_empirical_quantile_by_forecast(self) -> None:
        fhs = FilteredHistoricalSimulation(alpha=0.99, window=100_000, min_obs=100)
        assert fhs.var(0.0, 0.01) is None
        rng = np.random.default_rng(0)
        sigmas = rng.uniform(0.005, 0.03, 100_000)
        for s in sigmas:
            fhs.record(float(rng.normal(0.0, s)), 0.0, float(s))
        assert fhs.var(0.0, 0.02) == pytest.approx(0.02 * norm.ppf(0.99), rel=0.03)
        assert fhs.es(0.0, 0.02) > fhs.var(0.0, 0.02)  # type: ignore[operator]

    def test_fat_tails_are_captured(self) -> None:
        """Student-t(4) residuals: FHS beats a Gaussian VaR with the same forecast sigma."""
        rng = np.random.default_rng(1)
        z = rng.standard_t(4, 200_000) / math.sqrt(2.0)  # unit variance
        fhs = FilteredHistoricalSimulation(0.99, 200_000, 100)
        for value in z:
            fhs.record(float(value), 0.0, 1.0)
        assert fhs.var(0.0, 1.0) == pytest.approx(-np.quantile(z, 0.01), rel=1e-6)
        assert fhs.var(0.0, 1.0) > norm.ppf(0.99)  # type: ignore[operator]

    def test_ignores_invalid_forecasts(self) -> None:
        fhs = FilteredHistoricalSimulation(0.99, 252, 20)
        fhs.record(0.01, 0.0, 0.0)
        fhs.record(float("nan"), 0.0, 0.01)
        assert fhs.n_obs == 0


# ============================================================================================
# Backtester
# ============================================================================================


class TestBacktester:
    def test_ledger_identity_every_bar(self, result: BacktestResult) -> None:
        eq = result.equity
        np.testing.assert_allclose(
            eq["equity"], eq["cash"] + eq["quantity"] * eq["price"], rtol=1e-12
        )
        assert result.costs["commission"] == pytest.approx(result.fills["commission"].sum())
        assert result.costs["slippage"] == pytest.approx(result.fills["slippage_cost"].sum())

    def test_no_trading_before_first_decision_and_fill_timing(
        self, result: BacktestResult, bars: pd.DataFrame
    ) -> None:
        assert result.first_decision is not None
        first_fill = result.fills["ts"].iloc[0]
        first_decision_pos = bars.index.get_loc(result.first_decision)
        assert bars.index.get_loc(first_fill) >= first_decision_pos + 1
        before = result.equity.loc[: result.first_decision]
        assert (before["quantity"] == 0).all()
        rebalances = result.fills[result.fills["reason"] == "rebalance"]
        np.testing.assert_allclose(
            rebalances["ref_price"].to_numpy(),
            bars["open"].reindex(pd.DatetimeIndex(rebalances["ts"])).to_numpy(),
        )

    def test_orders_follow_decisions_with_latency(
        self, result: BacktestResult, bars: pd.DataFrame
    ) -> None:
        submitted = result.decisions.index[result.decisions["order_submitted"]]
        rebalance_days = set(result.fills.loc[result.fills["reason"] == "rebalance", "ts"])
        for ts in submitted[:-1]:
            nxt = bars.index[bars.index.get_loc(ts) + 1]
            stopped = nxt in set(result.stops["ts"]) if len(result.stops) else False
            assert nxt in rebalance_days or stopped

    def test_rebalance_threshold_respected(self, result: BacktestResult) -> None:
        d = result.decisions
        gap = (d["approved_weight"] - d["current_weight"]).abs()
        exits = (d["approved_weight"] == 0) & (d["current_weight"] != 0)
        expected = (gap > result.config.backtest.rebalance_threshold) | exits
        assert (d["order_submitted"] == expected).all()

    def test_risk_limits_hold(self, result: BacktestResult) -> None:
        assert (result.decisions["approved_weight"] >= 0).all()  # shorting disabled
        assert (result.decisions["approved_weight"] <= result.config.sizing.max_leverage).all()

    def test_stop_exits_are_gap_aware_and_flatten(self, result: BacktestResult) -> None:
        assert len(result.stops) > 0
        for _, stop in result.stops.iterrows():
            assert stop["fill_price"] <= stop["level"] + 1e-12  # long stops: at level or worse
            same_day = result.fills[
                (result.fills["ts"] == stop["ts"]) & (result.fills["reason"] == "stop_loss")
            ]
            assert len(same_day) == 1
            assert result.equity.loc[stop["ts"], "quantity"] == 0.0

    def test_var_table_alignment(self, result: BacktestResult, bars: pd.DataFrame) -> None:
        v = result.var_forecasts
        realized = np.log(bars["close"]).diff().shift(-1).reindex(v.index)
        pd.testing.assert_series_equal(v["realized_next"], realized, check_names=False)
        assert v["var_fhs"].iloc[:49].isna().all()
        assert v["var_fhs"].iloc[60:].notna().all()

    def test_benchmark(self, result: BacktestResult, bars: pd.DataFrame) -> None:
        entry = bars.index.get_loc(result.first_decision) + 1
        capital = result.config.run.initial_capital
        assert (result.benchmark.iloc[:entry] == capital).all()
        ratio = result.benchmark.iloc[-1] / capital
        assert ratio == pytest.approx(bars["close"].iloc[-1] / bars["open"].iloc[entry], rel=1e-3)

    def test_deterministic(self, bars: pd.DataFrame, result: BacktestResult) -> None:
        again = Backtester(_config()).run(bars)
        pd.testing.assert_frame_equal(again.equity, result.equity, check_exact=True)

    def test_save_writes_tables_and_manifest(self, result: BacktestResult, tmp_path: Path) -> None:
        run_dir = result.save(tmp_path)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["config_fingerprint"] == result.config.fingerprint()
        assert manifest["data_hash"] == result.data_hash
        for name in ("equity", "decisions", "fills", "var_forecasts", "regimes", "refits"):
            assert (run_dir / f"{name}.csv").is_file()

    def test_zero_cost_configuration(self, bars: pd.DataFrame) -> None:
        free = _config(
            execution={
                "commission_bps": 0.0,
                "min_commission": 0.0,
                "half_spread_bps": 0.0,
                "impact_coefficient": 0.0,
            }
        )
        r = Backtester(free).run(bars.iloc[:600])
        assert r.costs["commission"] == 0.0
        assert r.costs["slippage"] == 0.0


# ============================================================================================
# Cross-check and causality
# ============================================================================================


@pytest.mark.parametrize("allow_short", [False, True])
def test_event_engine_matches_return_space_computation(
    bars: pd.DataFrame, allow_short: bool
) -> None:
    """Two independent formulations (cash/share ledger vs. return-space recursion) agree."""
    cfg = _config(
        execution={
            "impact_coefficient": 0.0,
            "min_commission": 0.0,
            "half_spread_bps": 1.5,
            "commission_bps": 0.7,
            "borrow_fee_bps_annual": 80.0,
        },
        risk={"trailing_stop_atr": None, "allow_short": allow_short},
        backtest={"rebalance_threshold": 0.0},
        strategy={
            "defensive_direction": -0.5 if allow_short else 0.5,
            "momentum_long_only": not allow_short,
        },
    )
    result = Backtester(cfg).run(bars)
    orders = result.decisions.loc[result.decisions["order_submitted"], "approved_weight"]
    if allow_short:
        assert (orders < 0).any()
    expected = return_space_equity(
        bars,
        orders,
        initial_capital=cfg.run.initial_capital,
        latency_bars=cfg.execution.latency_bars,
        half_spread_bps=cfg.execution.half_spread_bps,
        commission_bps=cfg.execution.commission_bps,
        borrow_fee_bps_annual=cfg.execution.borrow_fee_bps_annual,
    )
    np.testing.assert_allclose(result.equity["equity"], expected, rtol=1e-10)


def test_full_backtest_has_no_lookahead(ohlcv_factory: OHLCVFactory) -> None:
    """The strongest causality check: corrupt every bar after T and re-run the entire stack;
    the equity curve, positions and cash up to T must be bit-identical."""
    cfg = _config(training={"min_train_bars": 250, "refit_every": 120})

    def equity_curve(ohlcv: pd.DataFrame) -> pd.DataFrame:
        return Backtester(cfg).run(ohlcv).equity

    data = ohlcv_factory(700, seed=41)
    checker = LookAheadSanityChecker(equity_curve)
    for cutoff in (420, 560, 690):
        assert checker.assert_no_lookahead(data, data.index[cutoff]).passed
