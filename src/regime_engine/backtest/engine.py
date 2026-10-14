"""Event-driven backtester.

Every bar ``t`` is processed in strict information order:

    1. start_bar           countdowns advance (regime cooldown, stop lockout)
    2. open                orders decided at t - latency fill at the open (with costs)
    3. intrabar            protective stops/targets are checked against high/low (gap-aware)
    4. close               borrow accrues; the portfolio is marked to the close
    5. regime              the engine's causal state for bar t (HMMEngine.update, or a replayed
                           RegimeTrack built the same way)
    6. decision            strategy -> sizing -> VaR -> risk desk -> target order for t + latency
    7. post-close          trailing stops ratchet with bar t's range and ATR

Features, ATR and market conditions are precomputed with trailing windows only. Causality is
verified end to end by running the backtest through ``LookAheadSanityChecker``: corrupting
every bar after T must leave the equity curve up to T bit-identical.
"""

from __future__ import annotations

import json
import math
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy

from regime_engine import __version__
from regime_engine.core.config import EngineConfig, VaRMethod
from regime_engine.core.exceptions import BacktestError, ConfigError
from regime_engine.data.features import FeatureEngine
from regime_engine.data.ingestor import DataIngestor
from regime_engine.execution.costs import MarketConditions, SquareRootImpactCostModel
from regime_engine.execution.simulator import ExecutionSimulator, Fill, OrderReason
from regime_engine.models.gaussian_hmm import HMMParams
from regime_engine.models.hmm_engine import RegimeTrack, regime_fingerprint
from regime_engine.models.regime_mapper import FeatureScaling
from regime_engine.portfolio.portfolio import Portfolio, PortfolioSnapshot
from regime_engine.risk.risk_manager import RiskManager
from regime_engine.risk.rules import RiskState
from regime_engine.risk.sizing import SizingInputs, build_sizer
from regime_engine.risk.stops import StopKind, StopTrigger, average_true_range
from regime_engine.risk.var import FilteredHistoricalSimulation, regime_mixture_moments
from regime_engine.strategy.agent import build_strategy_agent
from regime_engine.strategy.base import MarketContext
from regime_engine.utils.logging import LogEvent, get_logger, log_context, new_run_id

_LOG = get_logger("backtest.engine")


@dataclass(frozen=True, eq=False)
class BacktestResult:
    run_id: str
    symbol: str
    config: EngineConfig
    data_hash: str
    first_decision: pd.Timestamp | None
    equity: pd.DataFrame  # per bar: cash, quantity, price, equity, weight, drawdown
    benchmark: pd.Series  # buy-and-hold equity, entered at the strategy's first fill opportunity
    regimes: pd.DataFrame  # per decision bar: HMM engine states
    decisions: pd.DataFrame  # per decision bar: signal, sizing, risk outcome, order flag
    var_forecasts: pd.DataFrame  # per decision bar: next-bar VaR by method + realised return
    fills: pd.DataFrame
    stops: pd.DataFrame
    refits: pd.DataFrame
    costs: dict[str, float]
    features: pd.DataFrame  # the HMM observation vectors (for ex-post diagnostics)
    final_params: HMMParams | None  # canonical parameters of the last model in force
    final_scaling: FeatureScaling | None
    transition_matrices: npt.NDArray[np.float64]  # (n_refits, 4, 4), canonical order

    @property
    def active_equity(self) -> pd.Series:
        """Equity from the first decision bar on (the period the strategy could trade)."""
        series = self.equity["equity"]
        return series if self.first_decision is None else series.loc[self.first_decision :]

    @property
    def active_benchmark(self) -> pd.Series:
        return (
            self.benchmark
            if self.first_decision is None
            else self.benchmark.loc[self.first_decision :]
        )

    def returns(self) -> pd.Series:
        return self.active_equity.pct_change().dropna()

    def manifest(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "symbol": self.symbol,
            "created_utc": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "config_fingerprint": self.config.fingerprint(),
            "config": self.config.to_dict(),
            "data_hash": self.data_hash,
            "first_decision": None if self.first_decision is None else str(self.first_decision),
            "bars": len(self.equity),
            "final_equity": float(self.equity["equity"].iloc[-1]),
            "costs": self.costs,
            "versions": {
                "regime_engine": __version__,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
            },
        }

    def save(self, out_dir: Path) -> Path:
        """Write every table as CSV plus ``manifest.json`` into ``out_dir/<run_id>``."""
        run_dir = Path(out_dir) / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        tables = {
            "equity": self.equity,
            "benchmark": self.benchmark.to_frame("benchmark_equity"),
            "regimes": self.regimes,
            "decisions": self.decisions,
            "var_forecasts": self.var_forecasts,
            "fills": self.fills,
            "stops": self.stops,
            "refits": self.refits,
        }
        for name, table in tables.items():
            table.to_csv(run_dir / f"{name}.csv")
        (run_dir / "manifest.json").write_text(json.dumps(self.manifest(), indent=2, default=str))
        return run_dir


def _trailing(values: pd.Series, window: int, how: str) -> npt.NDArray[np.float64]:
    """Rolling statistic over bars t-window..t-1: what is known at the *open* of bar t."""
    rolling = values.rolling(window, min_periods=window)
    stat = rolling.std(ddof=1) if how == "std" else rolling.mean()
    return np.asarray(stat.shift(1).to_numpy(dtype=np.float64))


class Backtester:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self._cfg = config or EngineConfig()

    @property
    def config(self) -> EngineConfig:
        return self._cfg

    def run(
        self,
        bars: pd.DataFrame,
        *,
        symbol: str | None = None,
        track: RegimeTrack | None = None,
    ) -> BacktestResult:
        """Run the backtest. Pass a ``track`` built from the same bars and regime-relevant
        config to skip re-estimating the HMM (variants that differ only in strategy, sizing,
        risk or execution settings share one track)."""
        cfg = self._cfg
        symbol = symbol or cfg.data.symbol
        run_id = new_run_id()
        with log_context(run_id=run_id, symbol=symbol):
            features = FeatureEngine(cfg.features).compute(bars)
            if track is None:
                track = RegimeTrack.build(cfg, features)
            elif track.fingerprint != regime_fingerprint(cfg):
                raise ConfigError(
                    "regime track was built with different features/HMM/regime settings"
                )
            elif not set(track.points).issubset(features.index):
                raise ConfigError("regime track timestamps do not match these bars")
            return self._run(bars, features, track, symbol, run_id)

    def build_track(self, bars: pd.DataFrame) -> RegimeTrack:
        return RegimeTrack.build(self._cfg, FeatureEngine(self._cfg.features).compute(bars))

    def _run(
        self,
        bars: pd.DataFrame,
        features: pd.DataFrame,
        track: RegimeTrack,
        symbol: str,
        run_id: str,
    ) -> BacktestResult:
        cfg = self._cfg

        open_ = bars["open"].to_numpy(dtype=np.float64)
        high = bars["high"].to_numpy(dtype=np.float64)
        low = bars["low"].to_numpy(dtype=np.float64)
        close = bars["close"].to_numpy(dtype=np.float64)
        log_ret = np.concatenate([[np.nan], np.diff(np.log(close))])
        atr = average_true_range(
            bars["high"], bars["low"], bars["close"], cfg.risk.atr_window
        ).to_numpy()
        window = cfg.execution.adv_window
        daily_vol = _trailing(pd.Series(log_ret), window, "std")
        adv = _trailing(bars["volume"].astype(np.float64).reset_index(drop=True), window, "mean")

        agent = build_strategy_agent(cfg.strategy)
        sizer = build_sizer(cfg.sizing, annualization=cfg.run.annualization)
        desk = RiskManager.from_config(cfg)
        costs = SquareRootImpactCostModel.from_config(cfg.execution)
        execution = ExecutionSimulator(costs, cfg.execution.latency_bars)
        portfolio = Portfolio(
            cfg.run.initial_capital,
            borrow_fee_bps_annual=cfg.execution.borrow_fee_bps_annual,
            annualization=cfg.run.annualization,
        )
        fhs = FilteredHistoricalSimulation(
            cfg.risk.var_alpha, cfg.risk.var_window, cfg.risk.fhs_min_obs
        )

        snapshots: list[PortfolioSnapshot] = []
        decisions: list[dict[str, object]] = []
        var_rows: list[dict[str, object]] = []
        fills: list[Fill] = []
        stops: list[StopTrigger] = []
        first_decision_idx: int | None = None
        prev_forecast: tuple[float, float] | None = None

        _LOG.info(
            LogEvent.BACKTEST_STARTED,
            "backtest started",
            bars=len(bars),
            first=bars.index[0],
            last=bars.index[-1],
            config_fingerprint=cfg.fingerprint(),
        )
        index = bars.index
        for i, ts in enumerate(index):
            desk.start_bar(ts)
            market = MarketConditions(float(daily_vol[i]), float(adv[i]))
            prior_atr = float(atr[i - 1]) if i > 0 else float("nan")

            # 2. scheduled fills at the open
            opened = execution.fill_due(i, ts, open_[i], portfolio, market)
            if opened:
                fills.extend(opened)
                desk.on_position(portfolio.weight(open_[i]), open_[i], prior_atr)

            # 3. protective exits within the bar
            if portfolio.quantity != 0.0:
                trigger = desk.check_stops(ts, open_[i], high[i], low[i])
                if trigger is not None:
                    stops.append(trigger)
                    execution.cancel_pending()
                    reason = (
                        OrderReason.STOP_LOSS
                        if trigger.kind is StopKind.STOP_LOSS
                        else OrderReason.PROFIT_TARGET
                    )
                    exit_fill = execution.execute(
                        execution.next_id(),
                        ts,
                        -portfolio.quantity,
                        trigger.fill_price,
                        market,
                        reason,
                    )
                    portfolio.apply(exit_fill)
                    fills.append(exit_fill)
                    desk.on_position(0.0, exit_fill.fill_price, prior_atr)

            # 4. close: borrow, mark to market
            portfolio.accrue_borrow(close[i])
            snapshot = portfolio.mark(ts, close[i])
            snapshots.append(snapshot)
            if snapshot.equity <= 0.0:
                raise BacktestError(f"equity reached {snapshot.equity:.2f} at {ts}")

            # 5. regime
            point = track.points.get(ts)
            state = None if point is None else point.state
            desk.on_regime(state)

            # 6. decision
            if point is not None and state is not None:
                if first_decision_idx is None:
                    first_decision_idx = i
                if prev_forecast is not None:
                    fhs.record(float(log_ret[i]), *prev_forecast)

                predicted = point.forecast
                moments = point.moments
                mu, sigma = regime_mixture_moments(predicted, moments)
                trailing = log_ret[1 : i + 1]

                signal = agent.signal(MarketContext(ts, bars.iloc[: i + 1], state))
                target = sizer.size(signal, SizingInputs(trailing, predicted, moments))
                estimates = desk.var.estimate(trailing, predicted, moments)
                fhs_var = fhs.var(mu, sigma)
                unit_var = {
                    VaRMethod.FILTERED_HISTORICAL: fhs_var
                    if fhs_var is not None
                    else estimates.mixture,
                    VaRMethod.MIXTURE: estimates.mixture,
                    VaRMethod.HISTORICAL: estimates.historical,
                    VaRMethod.PARAMETRIC: estimates.parametric,
                }[cfg.risk.var_method]

                decision = desk.evaluate(
                    target,
                    RiskState(ts, snapshot.equity, snapshot.weight, snapshot.drawdown, unit_var),
                )
                approved = decision.approved.weight
                trade = abs(approved - snapshot.weight) > cfg.backtest.rebalance_threshold or (
                    approved == 0.0 and snapshot.weight != 0.0
                )
                if trade:
                    execution.submit_target(ts, i, approved)

                decisions.append(
                    {
                        "ts": ts,
                        "regime": state.regime.name,
                        "strategy": signal.strategy,
                        "direction": signal.direction,
                        "sized_weight": target.weight,
                        "approved_weight": approved,
                        "current_weight": snapshot.weight,
                        "unit_var": unit_var,
                        "risk_rules": ",".join(
                            o.rule for o in decision.trace if o.action.value != "pass"
                        ),
                        "order_submitted": trade,
                    }
                )
                var_rows.append(
                    {
                        "ts": ts,
                        "mu_forecast": mu,
                        "sigma_forecast": sigma,
                        "var_historical": estimates.historical,
                        "var_parametric": estimates.parametric,
                        "var_mixture": estimates.mixture,
                        "var_fhs": fhs_var,
                    }
                )
                prev_forecast = (mu, sigma)

            # 7. post-close: ratchet trailing stops
            desk.on_close(high[i], low[i], float(atr[i]))

        result = self._assemble(
            run_id,
            symbol,
            bars,
            log_ret,
            snapshots,
            decisions,
            var_rows,
            fills,
            stops,
            features,
            track,
            portfolio,
            costs,
            first_decision_idx,
        )
        _LOG.info(
            LogEvent.BACKTEST_COMPLETED,
            "backtest completed",
            final_equity=float(result.equity["equity"].iloc[-1]),
            fills=len(fills),
            stops=len(stops),
            refits=len(track.history),
            **result.costs,
        )
        return result

    def _assemble(
        self,
        run_id: str,
        symbol: str,
        bars: pd.DataFrame,
        log_ret: npt.NDArray[np.float64],
        snapshots: list[PortfolioSnapshot],
        decisions: list[dict[str, object]],
        var_rows: list[dict[str, object]],
        fills: list[Fill],
        stops: list[StopTrigger],
        features: pd.DataFrame,
        track: RegimeTrack,
        portfolio: Portfolio,
        costs: SquareRootImpactCostModel,
        first_decision_idx: int | None,
    ) -> BacktestResult:
        cfg = self._cfg
        equity = pd.DataFrame(
            [
                {
                    "cash": s.cash,
                    "quantity": s.quantity,
                    "price": s.price,
                    "equity": s.equity,
                    "weight": s.weight,
                    "drawdown": s.drawdown,
                }
                for s in snapshots
            ],
            index=bars.index,
        )
        next_return = pd.Series(log_ret, index=bars.index).shift(-1)
        var_frame = pd.DataFrame(var_rows).set_index("ts") if var_rows else pd.DataFrame()
        if not var_frame.empty:
            var_frame["realized_next"] = next_return.reindex(var_frame.index)
        decision_frame = pd.DataFrame(decisions).set_index("ts") if decisions else pd.DataFrame()
        fill_frame = pd.DataFrame(
            [
                {
                    "ts": f.ts,
                    "order_id": f.order_id,
                    "side": f.side.name,
                    "quantity": f.quantity,
                    "ref_price": f.ref_price,
                    "fill_price": f.fill_price,
                    "commission": f.commission,
                    "slippage_cost": f.slippage_cost,
                    "reason": f.reason.value,
                }
                for f in fills
            ]
        )
        stop_frame = pd.DataFrame(
            [
                {
                    "ts": s.ts,
                    "kind": s.kind.value,
                    "side": s.side.name,
                    "level": s.level,
                    "fill_price": s.fill_price,
                }
                for s in stops
            ]
        )
        refit_frame = pd.DataFrame(
            [
                {
                    "ts": r.ts,
                    "version": r.version,
                    "n_train": r.n_train,
                    "log_likelihood": r.fit.log_likelihood,
                    "bic": r.fit.bic,
                    "converged": r.fit.converged,
                    "n_iter": r.fit.n_iter,
                    "warm_started": r.warm_started,
                    "ambiguous": r.mapping.ambiguous,
                }
                for r in track.history
            ]
        )
        first_decision = None if first_decision_idx is None else bars.index[first_decision_idx]
        benchmark = self._benchmark(bars, costs, first_decision_idx)
        return BacktestResult(
            run_id=run_id,
            symbol=symbol,
            config=cfg,
            data_hash=DataIngestor.data_hash(bars),
            first_decision=first_decision,
            equity=equity,
            benchmark=benchmark,
            regimes=track.states_frame(),
            decisions=decision_frame,
            var_forecasts=var_frame,
            fills=fill_frame,
            stops=stop_frame,
            refits=refit_frame,
            costs={
                "commission": float(portfolio.total_commission),
                "slippage": float(portfolio.total_slippage),
                "borrow": float(portfolio.total_borrow),
            },
            features=features,
            final_params=track.final_params,
            final_scaling=track.final_scaling,
            transition_matrices=track.transition_matrices(),
        )

    def _benchmark(
        self, bars: pd.DataFrame, costs: SquareRootImpactCostModel, first_decision_idx: int | None
    ) -> pd.Series:
        """Buy and hold, entered (with costs) at the strategy's first possible fill."""
        capital = self._cfg.run.initial_capital
        series = pd.Series(capital, index=bars.index, dtype=np.float64, name="benchmark")
        if first_decision_idx is None:
            return series
        entry = first_decision_idx + self._cfg.execution.latency_bars
        if entry >= len(bars):
            return series
        open_price = float(bars["open"].iloc[entry])
        fill_price = open_price * (1.0 + costs.half_spread)
        quantity = capital / (fill_price * (1.0 + costs.commission_rate))
        cash = capital - quantity * fill_price - costs.commission(quantity, fill_price)
        values = series.to_numpy(dtype=np.float64, copy=True)
        values[entry:] = cash + quantity * bars["close"].to_numpy(dtype=np.float64)[entry:]
        return pd.Series(values, index=bars.index, name="benchmark")


def return_space_equity(
    bars: pd.DataFrame,
    orders: pd.Series,
    *,
    initial_capital: float,
    latency_bars: int = 1,
    half_spread_bps: float = 0.0,
    commission_bps: float = 0.0,
    borrow_fee_bps_annual: float = 0.0,
    annualization: int = 252,
) -> pd.Series:
    """Independent re-computation of the equity curve in return space (no cash/share ledger).

    ``orders`` maps decision timestamps to target weights. An order decided at bar d sets the
    weight at the open of bar d + latency. Between orders, the weight drifts with prices.
    Costs must be proportional (half-spread + commission; no impact, no minimum commission).
    Per bar j, with E the equity at the open before trading:

        dw      = w_j - w_drift_j
        cost_j  = |dw| (h + c (1 + sign(dw) h))
        E_close = E (1 - cost_j + w_j (C_j / O_j - 1) - borrow_j)
        E_next  = E (1 - cost_j + w_j (O_{j+1} / O_j - 1) - borrow_j)
        w_drift_{j+1} = w_j (O_{j+1} / O_j) / (E_next / E)

    With stops disabled and a zero rebalance threshold, this must match the event-driven
    ledger to floating-point precision.
    """
    h, c = half_spread_bps * 1e-4, commission_bps * 1e-4
    borrow_rate = borrow_fee_bps_annual * 1e-4 / annualization
    positions = {ts: i for i, ts in enumerate(bars.index)}
    position = {positions[ts] + latency_bars: float(w) for ts, w in orders.items()}
    o = bars["open"].to_numpy(dtype=np.float64)
    cl = bars["close"].to_numpy(dtype=np.float64)
    n = len(bars)
    equity_close = np.empty(n)
    e_open = float(initial_capital)
    w_drift = 0.0
    for j in range(n):
        w = position.get(j, w_drift)
        dw = w - w_drift
        cost = abs(dw) * (h + c * (1.0 + math.copysign(1.0, dw) * h)) if dw != 0.0 else 0.0
        borrow = -w * (cl[j] / o[j]) * borrow_rate if w < 0 else 0.0
        equity_close[j] = e_open * (1.0 - cost + w * (cl[j] / o[j] - 1.0) - borrow)
        if j + 1 < n:
            growth = 1.0 - cost + w * (o[j + 1] / o[j] - 1.0) - borrow
            w_drift = w * (o[j + 1] / o[j]) / growth if w != 0.0 else 0.0
            e_open *= growth
    return pd.Series(equity_close, index=bars.index, name="equity")
