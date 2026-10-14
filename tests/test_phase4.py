"""Phase 4 tests: strategies, regime routing, position sizing, VaR, risk rules, stops, risk desk."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, norm

from regime_engine.core.config import (
    CooldownMode,
    EngineConfig,
    RoutingMode,
    SizingMethod,
    StrategyKind,
)
from regime_engine.core.exceptions import ConfigError, LookAheadError
from regime_engine.core.types import (
    Regime,
    RegimeMoments,
    RegimeState,
    RiskAction,
    Side,
    Signal,
    TargetPosition,
)
from regime_engine.data.features import FeatureEngine
from regime_engine.models.hmm_engine import HMMEngine
from regime_engine.risk import (
    CooldownRule,
    DrawdownBreaker,
    LeverageRule,
    RealizedVolTargetSizer,
    RegimeVolTargetSizer,
    RiskManager,
    RiskState,
    SizingInputs,
    StopKind,
    StopManager,
    VaRBudgetRule,
    VaRCalculator,
    average_true_range,
    build_sizer,
    christoffersen_independence,
    gaussian_es,
    gaussian_var,
    historical_es,
    historical_var,
    kupiec_pof,
    mixture_var,
    parametric_var,
    reduce_only,
    regime_mixture_moments,
)
from regime_engine.strategy import (
    ConstantDirection,
    MarketContext,
    StrategyAgent,
    TimeSeriesMomentum,
    ZScoreMeanReversion,
    build_strategy_agent,
)

from .conftest import ListHandler, OHLCVFactory

TS = pd.Timestamp("2024-01-02", tz="UTC")

#: Per-bar regime moments shaped like the SPY walk-forward fits (≈ annual 19%/9% ... -11%/42%).
MOMENTS = {
    Regime.BULL_LOW_VOL: RegimeMoments(0.19 / 252, 0.09 / math.sqrt(252)),
    Regime.BULL_HIGH_VOL: RegimeMoments(0.11 / 252, 0.14 / math.sqrt(252)),
    Regime.BEAR_LOW_VOL: RegimeMoments(0.04 / 252, 0.19 / math.sqrt(252)),
    Regime.BEAR_HIGH_VOL: RegimeMoments(-0.11 / 252, 0.42 / math.sqrt(252)),
}


def _onehot(regime: Regime, mass: float = 1.0) -> np.ndarray:
    p = np.full(4, (1.0 - mass) / 3.0)
    p[regime] = mass
    return p


def _state(
    ts: pd.Timestamp = TS,
    regime: Regime = Regime.BULL_LOW_VOL,
    probs: np.ndarray | None = None,
    *,
    changed: bool = False,
) -> RegimeState:
    p = _onehot(regime) if probs is None else probs
    return RegimeState(
        ts=ts,
        probs=p,
        regime=regime,
        raw_argmax=Regime(int(np.argmax(p))),
        changed=changed,
        refit=False,
        model_version=1,
        mapping_ambiguous=False,
        predictive_loglik=0.0,
    )


def _history(closes: np.ndarray, end: pd.Timestamp = TS) -> pd.DataFrame:
    index = pd.bdate_range(end=end, periods=len(closes), tz="UTC", name="date")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": 1e6,
        },
        index=index,
    )


def _ctx(
    closes: np.ndarray, regime: Regime = Regime.BULL_LOW_VOL, probs: np.ndarray | None = None
) -> MarketContext:
    history = _history(closes)
    return MarketContext(history.index[-1], history, _state(history.index[-1], regime, probs))


def _signal(direction: float, regime: Regime = Regime.BULL_LOW_VOL) -> Signal:
    return Signal(TS, direction, "test", regime)


def _risk_state(
    current: float = 0.0, equity: float = 1.0, unit_var: float | None = None, ts: pd.Timestamp = TS
) -> RiskState:
    return RiskState(ts=ts, equity=equity, current_weight=current, drawdown=0.0, unit_var=unit_var)


# ============================================================================================
# Configuration
# ============================================================================================


class TestConfig:
    def test_defaults_follow_the_walk_forward_finding(self) -> None:
        cfg = EngineConfig()
        assert {cfg.strategy.bull_low, cfg.strategy.bear_high} == {StrategyKind.LONG}
        tactical = type(cfg.strategy).tactical()
        assert tactical.bear_high is StrategyKind.DEFENSIVE
        assert tactical.defensive_direction == 0.5  # reduced long, not short
        assert cfg.risk.trailing_stop_atr is None
        assert cfg.risk.allow_short is False
        assert cfg.sizing.method is SizingMethod.REGIME_VOL_TARGET
        assert cfg.sizing.kelly_cap is False
        assert cfg.risk.profit_target_atr is None

    @pytest.mark.parametrize(
        "raw",
        [
            {"strategy": {"defensive_direction": -1.5}},
            {"strategy": {"momentum_lookbacks": []}},
            {"sizing": {"target_vol_annual": 0.0}},
            {"risk": {"var_alpha": 0.4}},
            {"risk": {"trailing_stop_atr": -1.0}},
        ],
    )
    def test_invalid_values_rejected(self, raw: dict[str, object]) -> None:
        with pytest.raises(ConfigError):
            EngineConfig.from_dict(raw)


# ============================================================================================
# Strategies and routing
# ============================================================================================


class TestStrategies:
    def test_context_enforces_causality(self) -> None:
        history = _history(np.linspace(100, 110, 50))
        with pytest.raises(LookAheadError, match="must end at the decision bar"):
            MarketContext(history.index[-2], history, _state(history.index[-2]))
        with pytest.raises(LookAheadError, match="regime state"):
            MarketContext(history.index[-1], history, _state(history.index[-2]))

    def test_momentum_direction_and_bounds(self) -> None:
        rng = np.random.default_rng(0)
        up = 100 * np.exp(np.cumsum(0.002 + 0.005 * rng.standard_normal(300)))
        down = 100 * np.exp(np.cumsum(-0.002 + 0.005 * rng.standard_normal(300)))
        assert 0.5 < TimeSeriesMomentum().direction(_ctx(up)) <= 1.0
        assert TimeSeriesMomentum().direction(_ctx(down)) == 0.0  # long-only
        assert -1.0 <= TimeSeriesMomentum(long_only=False).direction(_ctx(down)) < -0.5
        damped = TimeSeriesMomentum(damping=0.5).direction(_ctx(up))
        assert damped == pytest.approx(0.5 * TimeSeriesMomentum().direction(_ctx(up)))

    def test_momentum_uses_available_lookbacks_only(self) -> None:
        short = 100 * np.exp(np.cumsum(np.full(30, 0.001) + np.tile([0.002, -0.002], 15)))
        assert TimeSeriesMomentum((21, 63, 252)).direction(_ctx(short)) > 0.0
        assert TimeSeriesMomentum((63, 252)).direction(_ctx(short)) == 0.0

    def test_mean_reversion_direction(self) -> None:
        base = 100 + np.sin(np.arange(40))
        stretched_up = np.append(base, 106.0)
        stretched_down = np.append(base, 94.0)
        assert ZScoreMeanReversion(20).direction(_ctx(stretched_up)) < -0.5
        assert ZScoreMeanReversion(20).direction(_ctx(stretched_down)) > 0.5
        assert ZScoreMeanReversion(20, long_only=True).direction(_ctx(stretched_up)) == 0.0
        assert ZScoreMeanReversion(20).direction(_ctx(np.full(30, 100.0))) == 0.0

    def test_hard_routing_uses_confirmed_regime(self) -> None:
        agent = StrategyAgent(
            {r: ConstantDirection(0.1 * (int(r) + 1), name=r.name) for r in Regime}
        )
        ctx = _ctx(
            np.linspace(100, 101, 30), Regime.BEAR_LOW_VOL, _onehot(Regime.BULL_LOW_VOL, 0.7)
        )
        signal = agent.signal(ctx)
        assert signal.direction == pytest.approx(0.3)
        assert signal.strategy == "BEAR_LOW_VOL"
        assert signal.regime is Regime.BEAR_LOW_VOL

    def test_blended_routing_weights_by_probability(self) -> None:
        directions = {
            r: ConstantDirection(d) for r, d in zip(Regime, (1.0, 0.5, -0.5, 0.0), strict=True)
        }
        probs = np.array([0.4, 0.3, 0.2, 0.1])
        signal = StrategyAgent(directions, RoutingMode.BLENDED).signal(
            _ctx(np.linspace(100, 101, 30), Regime.BULL_LOW_VOL, probs)
        )
        assert signal.direction == pytest.approx(0.4 + 0.15 - 0.1)
        assert signal.strategy == "blended"

    def test_default_agent_wiring(self) -> None:
        overlay = build_strategy_agent(EngineConfig().strategy)
        assert {s.name for s in overlay.routing.values()} == {"long"}
        assert (
            overlay.signal(_ctx(np.linspace(100, 90, 300), Regime.BEAR_HIGH_VOL)).direction == 1.0
        )
        agent = build_strategy_agent(type(EngineConfig().strategy).tactical())
        names = {r: s.name for r, s in agent.routing.items()}
        assert names == {
            Regime.BULL_LOW_VOL: "momentum",
            Regime.BULL_HIGH_VOL: "momentum_x0.5",
            Regime.BEAR_LOW_VOL: "mean_reversion",
            Regime.BEAR_HIGH_VOL: "defensive",
        }
        assert agent.signal(_ctx(np.linspace(100, 90, 300), Regime.BEAR_HIGH_VOL)).direction == 0.5
        with pytest.raises(ValueError, match="no strategy routed"):
            StrategyAgent({Regime.BULL_LOW_VOL: ConstantDirection(1.0)})


# ============================================================================================
# Sizing
# ============================================================================================


class TestSizing:
    def test_mixture_moments(self) -> None:
        p = np.array([0.5, 0.0, 0.0, 0.5])
        mu, sigma = regime_mixture_moments(p, MOMENTS)
        m = np.array(
            [MOMENTS[Regime.BULL_LOW_VOL].mean_return, MOMENTS[Regime.BEAR_HIGH_VOL].mean_return]
        )
        s = np.array(
            [MOMENTS[Regime.BULL_LOW_VOL].volatility, MOMENTS[Regime.BEAR_HIGH_VOL].volatility]
        )
        assert mu == pytest.approx(m.mean())
        assert sigma**2 == pytest.approx(0.5 * (s**2 + m**2).sum() - m.mean() ** 2)
        mu1, sigma1 = regime_mixture_moments(_onehot(Regime.BULL_HIGH_VOL), MOMENTS)
        assert (mu1, sigma1) == pytest.approx(
            (MOMENTS[Regime.BULL_HIGH_VOL].mean_return, MOMENTS[Regime.BULL_HIGH_VOL].volatility)
        )

    def test_vol_target_weight(self) -> None:
        sizer = RegimeVolTargetSizer(0.12, kelly_cap=False, max_leverage=5.0)
        inputs = SizingInputs(np.zeros(30), _onehot(Regime.BULL_HIGH_VOL), MOMENTS)
        target = sizer.size(_signal(0.8), inputs)
        assert target.weight == pytest.approx(0.8 * 0.12 / 0.14)
        assert sizer.size(_signal(0.0), inputs).weight == 0.0

    def test_leverage_cap(self) -> None:
        sizer = RegimeVolTargetSizer(0.12, kelly_cap=False, max_leverage=1.0)
        inputs = SizingInputs(np.zeros(30), _onehot(Regime.BULL_LOW_VOL), MOMENTS)
        assert sizer.size(_signal(1.0), inputs).weight == 1.0  # 0.12 / 0.09 = 1.33 -> 1.0

    def test_kelly_cap(self) -> None:
        sizer = RegimeVolTargetSizer(0.12, kelly_cap=True, kelly_fraction=0.25, max_leverage=5.0)
        bear_low = SizingInputs(np.zeros(30), _onehot(Regime.BEAR_LOW_VOL), MOMENTS)
        target = sizer.size(_signal(1.0), bear_low)
        m, s = MOMENTS[Regime.BEAR_LOW_VOL].mean_return, MOMENTS[Regime.BEAR_LOW_VOL].volatility
        assert target.weight == pytest.approx(0.25 * m / s**2)  # Kelly binds below 0.12 / 0.19
        assert "Kelly" in target.rationale
        crash = SizingInputs(np.zeros(30), _onehot(Regime.BEAR_HIGH_VOL), MOMENTS)
        assert sizer.size(_signal(0.5), crash).weight == 0.0  # drift opposes a long signal
        assert sizer.size(_signal(-0.5), crash).weight < 0.0
        bull = SizingInputs(np.zeros(30), _onehot(Regime.BULL_LOW_VOL), MOMENTS)
        uncapped = RegimeVolTargetSizer(0.12, kelly_cap=False, max_leverage=5.0)
        assert sizer.size(_signal(1.0), bull).weight == uncapped.size(_signal(1.0), bull).weight

    def test_realized_vol_baseline_uses_trailing_window(self) -> None:
        rng = np.random.default_rng(1)
        returns = np.concatenate([rng.normal(0, 0.03, 200), rng.normal(0, 0.01, 21)])
        sizer = RealizedVolTargetSizer(0.12, window=21, max_leverage=5.0)
        target = sizer.size(_signal(1.0), SizingInputs(returns))
        expected = 0.12 / (math.sqrt(252) * returns[-21:].std(ddof=1))
        assert target.weight == pytest.approx(expected)
        assert sizer.size(_signal(1.0), SizingInputs(np.array([0.01]))).weight == 0.0

    def test_build_sizer(self) -> None:
        cfg = EngineConfig.from_dict({"sizing": {"method": "realized_vol_target"}})
        assert isinstance(build_sizer(cfg.sizing), RealizedVolTargetSizer)
        assert isinstance(build_sizer(EngineConfig().sizing), RegimeVolTargetSizer)


# ============================================================================================
# VaR / ES and backtests
# ============================================================================================


@pytest.fixture(scope="module")
def normal_draws() -> np.ndarray:
    return np.random.default_rng(42).standard_normal(400_000)


class TestVaR:
    def test_historical_matches_normal_quantile(self, normal_draws: np.ndarray) -> None:
        assert historical_var(normal_draws, 0.99) == pytest.approx(norm.ppf(0.99), abs=0.02)
        assert historical_es(normal_draws, 0.99) == pytest.approx(
            norm.pdf(norm.ppf(0.99)) / 0.01, abs=0.03
        )

    def test_parametric_closed_forms(self, normal_draws: np.ndarray) -> None:
        assert gaussian_var(0.0, 1.0, 0.99) == pytest.approx(2.326347874, rel=1e-9)
        assert gaussian_es(0.0, 1.0, 0.975) == pytest.approx(2.337802, rel=1e-6)
        assert parametric_var(0.02 * normal_draws + 0.001, 0.95) == pytest.approx(
            gaussian_var(0.001, 0.02, 0.95), rel=0.01
        )
        assert gaussian_es(0.0, 1.0, 0.99) > gaussian_var(0.0, 1.0, 0.99)

    def test_mixture_var(self) -> None:
        single = mixture_var(_onehot(Regime.BULL_HIGH_VOL), MOMENTS, 0.99)
        m, s = MOMENTS[Regime.BULL_HIGH_VOL].mean_return, MOMENTS[Regime.BULL_HIGH_VOL].volatility
        assert single == pytest.approx(gaussian_var(m, s, 0.99), rel=1e-9)

        probs = np.array([0.7, 0.0, 0.0, 0.3])
        var = mixture_var(probs, MOMENTS, 0.99)
        rng = np.random.default_rng(3)
        regimes = rng.choice([0, 3], size=1_000_000, p=[0.7, 0.3])
        means = np.array([MOMENTS[Regime(k)].mean_return for k in range(4)])
        stds = np.array([MOMENTS[Regime(k)].volatility for k in range(4)])
        draws = means[regimes] + stds[regimes] * rng.standard_normal(len(regimes))
        assert var == pytest.approx(-np.quantile(draws, 0.01), rel=0.01)
        # The crash component fattens the tail beyond a Gaussian with the same variance.
        mu, sigma = regime_mixture_moments(probs, MOMENTS)
        assert var > gaussian_var(mu, sigma, 0.99)

    def test_kupiec_against_binomial_likelihood_ratio(self) -> None:
        n, alpha = 1000, 0.99
        on_target = kupiec_pof(np.arange(n) < 10, alpha)
        assert on_target.lr_statistic == pytest.approx(0.0, abs=1e-9)
        assert on_target.p_value == pytest.approx(1.0)
        too_many = kupiec_pof(np.arange(n) < 30, alpha)
        expected = 2 * (binom.logpmf(30, n, 0.03) - binom.logpmf(30, n, 0.01))
        assert too_many.lr_statistic == pytest.approx(expected, rel=1e-10)
        assert too_many.rejects()
        assert kupiec_pof(np.zeros(500, dtype=bool), alpha).lr_statistic > 0.0

    def test_christoffersen_detects_clustering(self) -> None:
        clustered = np.zeros(1000, dtype=bool)
        for start in (100, 400, 700):
            clustered[start : start + 5] = True
        assert christoffersen_independence(clustered).rejects()
        iid = np.random.default_rng(5).random(1000) < 0.02
        assert not christoffersen_independence(iid).rejects()

    def test_calculator(self) -> None:
        rng = np.random.default_rng(6)
        estimates = VaRCalculator(0.99, 252).estimate(
            rng.normal(0, 0.01, 600), _onehot(Regime.BULL_LOW_VOL), MOMENTS
        )
        assert estimates.historical > 0.0
        assert estimates.parametric == pytest.approx(0.0233, abs=0.003)
        assert estimates.mixture == pytest.approx(
            gaussian_var(
                MOMENTS[Regime.BULL_LOW_VOL].mean_return,
                MOMENTS[Regime.BULL_LOW_VOL].volatility,
                0.99,
            )
        )
        assert VaRCalculator().estimate(rng.normal(0, 0.01, 100)).mixture is None


# ============================================================================================
# Risk rules
# ============================================================================================


class TestRules:
    @pytest.mark.parametrize(
        ("proposed", "current", "allowed"),
        [
            (0.5, 1.0, 0.5),  # reduce: allowed
            (1.5, 1.0, 1.0),  # increase: held at current
            (-0.5, 1.0, 0.0),  # flip: only flatten
            (0.3, 0.0, 0.0),  # open from flat: blocked
            (-0.2, -0.6, -0.2),  # reduce a short
            (0.0, 0.7, 0.0),  # exit: allowed
        ],
    )
    def test_reduce_only(self, proposed: float, current: float, allowed: float) -> None:
        assert reduce_only(proposed, current) == allowed

    def test_cooldown_window_and_modes(self) -> None:
        rule = CooldownRule(2, CooldownMode.REDUCE_ONLY)
        assert rule.apply(1.5, _risk_state(1.0)) == (1.5, "")  # inactive
        rule.arm()
        assert rule.apply(1.5, _risk_state(1.0))[0] == 1.0
        assert rule.apply(0.5, _risk_state(1.0)) == (0.5, "")
        expired = rule.tick()
        assert not expired
        assert rule.apply(1.5, _risk_state(1.0))[0] == 1.0  # 2nd bar still active
        assert rule.tick()  # expires
        assert rule.apply(1.5, _risk_state(1.0)) == (1.5, "")

        freeze = CooldownRule(1, CooldownMode.LITERAL_FREEZE)
        freeze.arm()
        assert freeze.apply(0.2, _risk_state(1.0))[0] == 1.0  # even reductions are frozen
        assert CooldownRule(0).apply(2.0, _risk_state(0.0)) == (2.0, "")

    def test_drawdown_breaker_trips_halts_and_rearms(self) -> None:
        breaker = DrawdownBreaker(0.20, halt_bars=3)
        equity = [1.00, 1.10, 0.95, 0.87, 0.86, 0.85, 0.84, 0.80, 0.70]
        allowed = [breaker.apply(1.0, _risk_state(1.0, e))[0] for e in equity]
        # Peak 1.10; 0.87 / 1.10 = -20.9% trips. Halt 3 bars (0.87, 0.86, 0.85). Re-arm at
        # 0.84 with a new peak; 0.80 is -4.8%, 0.70 is -16.7%: both allowed.
        assert allowed == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        assert not breaker.halted

    def test_var_budget(self) -> None:
        rule = VaRBudgetRule(0.02)
        assert rule.apply(1.0, _risk_state(unit_var=0.04))[0] == pytest.approx(0.5)
        assert rule.apply(-1.0, _risk_state(unit_var=0.04))[0] == pytest.approx(-0.5)
        assert rule.apply(0.4, _risk_state(unit_var=0.04)) == (0.4, "")
        assert rule.apply(3.0, _risk_state(unit_var=None)) == (3.0, "")

    def test_leverage_and_short_permission(self) -> None:
        long_only = LeverageRule(1.5)
        assert long_only.apply(2.0, _risk_state())[0] == 1.5
        assert long_only.apply(-0.5, _risk_state()) == (0.0, "short selling disabled")
        assert LeverageRule(1.5, allow_short=True).apply(-2.0, _risk_state())[0] == -1.5


# ============================================================================================
# Stops
# ============================================================================================


class TestStops:
    def test_atr_is_causal_true_range_average(self, ohlcv: pd.DataFrame) -> None:
        atr = average_true_range(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)
        t = 100
        prev_close = ohlcv["close"].iloc[t - 14 : t].to_numpy()
        h = ohlcv["high"].iloc[t - 13 : t + 1].to_numpy()
        lo = ohlcv["low"].iloc[t - 13 : t + 1].to_numpy()
        tr = np.maximum.reduce([h - lo, np.abs(h - prev_close), np.abs(lo - prev_close)])
        assert atr.iloc[t] == pytest.approx(tr.mean())
        assert atr.iloc[:13].isna().all()
        truncated = average_true_range(
            ohlcv["high"].iloc[: t + 1],
            ohlcv["low"].iloc[: t + 1],
            ohlcv["close"].iloc[: t + 1],
            14,
        )
        assert truncated.iloc[-1] == atr.iloc[t]

    def test_long_trailing_stop_ratchets_and_fills(self) -> None:
        stops = StopManager(trailing_atr_mult=2.0)
        stops.on_position(1.0, price=100.0, atr=1.0)
        assert stops.stop_level == 98.0
        stops.on_close(high=105.0, low=101.0, atr=1.0)
        assert stops.stop_level == 103.0
        stops.on_close(high=104.0, low=99.0, atr=2.0)  # wider ATR would loosen: ignored
        assert stops.stop_level == 103.0
        assert stops.check_bar(TS, open_=104.0, high=106.0, low=103.5) is None
        trigger = stops.check_bar(TS, open_=104.0, high=104.5, low=102.0)
        assert trigger is not None
        assert (trigger.kind, trigger.side, trigger.fill_price) == (
            StopKind.STOP_LOSS,
            Side.LONG,
            103.0,
        )
        assert stops.side is None  # position closed by the stop

    def test_gap_through_stop_fills_at_open(self) -> None:
        stops = StopManager(trailing_atr_mult=2.0)
        stops.on_position(1.0, price=100.0, atr=1.0)
        trigger = stops.check_bar(TS, open_=95.0, high=96.0, low=94.0)
        assert trigger is not None
        assert trigger.level == 98.0
        assert trigger.fill_price == 95.0

    def test_stop_wins_when_both_levels_hit(self) -> None:
        stops = StopManager(trailing_atr_mult=2.0, target_atr_mult=3.0)
        stops.on_position(1.0, price=100.0, atr=1.0)
        trigger = stops.check_bar(TS, open_=100.0, high=104.0, low=97.0)
        assert trigger is not None
        assert trigger.kind is StopKind.STOP_LOSS
        stops.on_position(1.0, price=100.0, atr=1.0)
        target = stops.check_bar(TS, open_=104.0, high=105.0, low=101.0)
        assert target is not None
        assert (target.kind, target.fill_price) == (StopKind.PROFIT_TARGET, 104.0)

    def test_short_side_is_symmetric(self) -> None:
        stops = StopManager(trailing_atr_mult=2.0)
        stops.on_position(-0.5, price=100.0, atr=1.0)
        assert stops.stop_level == 102.0
        stops.on_close(high=99.0, low=95.0, atr=1.0)
        assert stops.stop_level == 97.0
        trigger = stops.check_bar(TS, open_=98.0, high=98.5, low=96.0)
        assert trigger is not None
        assert (trigger.side, trigger.fill_price) == (Side.SHORT, 98.0)

    def test_resizing_keeps_levels_and_flip_resets(self) -> None:
        stops = StopManager(trailing_atr_mult=2.0)
        stops.on_position(1.0, price=100.0, atr=1.0)
        stops.on_position(1.4, price=110.0, atr=1.0)
        assert stops.stop_level == 98.0
        stops.on_position(-1.0, price=110.0, atr=1.0)
        assert stops.stop_level == 112.0
        stops.on_position(0.0, price=110.0, atr=1.0)
        assert stops.side is None
        assert stops.check_bar(TS, 1.0, 1.0, 1.0) is None


# ============================================================================================
# Risk desk
# ============================================================================================


class TestRiskManager:
    def test_rule_order_and_trace(self) -> None:
        desk = RiskManager.from_config(EngineConfig())
        assert [r.name for r in desk.rules] == [
            "drawdown_breaker",
            "regime_cooldown",
            "stop_lockout",
            "var_budget",
            "leverage",
        ]
        decision = desk.evaluate(TargetPosition(TS, 3.0, "test"), _risk_state(0.5, unit_var=0.02))
        assert decision.approved.weight == pytest.approx(1.25)  # VaR budget 0.025 / 0.02
        assert [o.action for o in decision.trace] == [
            RiskAction.PASS,
            RiskAction.PASS,
            RiskAction.PASS,
            RiskAction.MODIFY,
            RiskAction.PASS,
        ]
        assert decision.modified

    def test_regime_flip_blocks_increases_but_not_exits(self, captured: ListHandler) -> None:
        desk = RiskManager.from_config(EngineConfig())
        desk.start_bar(TS)
        desk.on_regime(_state(regime=Regime.BEAR_HIGH_VOL, changed=True))
        assert desk.cooldown_active
        assert (
            desk.evaluate(TargetPosition(TS, 1.2, "add"), _risk_state(0.6)).approved.weight == 0.6
        )
        assert (
            desk.evaluate(TargetPosition(TS, 0.2, "cut"), _risk_state(0.6)).approved.weight == 0.2
        )
        desk.start_bar(TS)
        assert desk.cooldown_active  # 2-bar cooldown: still active on the next bar
        desk.start_bar(TS)
        assert not desk.cooldown_active
        assert len(captured.events("COOLDOWN_START")) == 1
        assert len(captured.events("COOLDOWN_END")) == 1

    def test_stop_out_arms_lockout(self, captured: ListHandler) -> None:
        desk = RiskManager.from_config(EngineConfig.from_dict({"risk": {"trailing_stop_atr": 3.0}}))
        desk.on_position(1.0, price=100.0, atr=1.0)
        desk.start_bar(TS)
        trigger = desk.check_stops(TS, open_=99.0, high=99.5, low=96.0)
        assert trigger is not None
        assert trigger.fill_price == 97.0  # 100 - 3 x ATR
        reentry = desk.evaluate(TargetPosition(TS, 1.0, "re-enter"), _risk_state(0.0))
        assert reentry.approved.weight == 0.0
        assert len(captured.events("STOP_TRIGGERED")) == 1
        for _ in range(5):
            desk.start_bar(TS)
        assert (
            desk.evaluate(TargetPosition(TS, 1.0, "later"), _risk_state(0.0)).approved.weight == 1.0
        )

    def test_breaker_overrides_everything(self) -> None:
        desk = RiskManager.from_config(EngineConfig())
        desk.evaluate(TargetPosition(TS, 1.0, "x"), _risk_state(1.0, equity=1.0))
        decision = desk.evaluate(TargetPosition(TS, 1.0, "x"), _risk_state(1.0, equity=0.75))
        assert decision.approved.weight == 0.0
        assert decision.trace[0].action is RiskAction.VETO


# ============================================================================================
# Decision pipeline on real engine output (synthetic bars)
# ============================================================================================


def test_decision_pipeline_end_to_end(ohlcv_factory: OHLCVFactory) -> None:
    cfg = EngineConfig.from_dict(
        {
            "training": {"min_train_bars": 300, "refit_every": 100},
            "hmm": {"n_restarts": 2, "refit_restarts": 1},
            "cooldown": {"bars": 3},
        }
    )
    bars = ohlcv_factory(800, seed=29)
    features = FeatureEngine(cfg.features)(bars)
    engine = HMMEngine(cfg)
    agent = build_strategy_agent(cfg.strategy)
    sizer = build_sizer(cfg.sizing)
    desk = RiskManager.from_config(cfg)
    log_returns = np.log(bars["close"]).diff()

    weight = 0.0
    approved: list[float] = []
    for ts, row in features.iterrows():
        state = engine.update(ts, row.to_numpy())
        if state is None:
            continue
        desk.start_bar(ts)
        desk.on_regime(state)
        ctx = MarketContext(ts, bars.loc[:ts], state)
        signal = agent.signal(ctx)
        predicted = engine.one_step_forecast()
        trailing = log_returns.loc[:ts].to_numpy()[1:]
        target = sizer.size(signal, SizingInputs(trailing, predicted, engine.mapping.moments))
        unit_var = desk.var.estimate(trailing, predicted, engine.mapping.moments).mixture
        decision = desk.evaluate(target, RiskState(ts, 1.0, weight, 0.0, unit_var))
        if state.changed:
            assert abs(decision.approved.weight) <= abs(weight) + 1e-12
        weight = decision.approved.weight
        approved.append(weight)

    arr = np.array(approved)
    assert len(arr) > 400
    assert np.isfinite(arr).all()
    assert (arr >= 0.0).all()
    assert (arr <= cfg.sizing.max_leverage).all()
