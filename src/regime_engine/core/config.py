"""Engine configuration schemas.

All configuration objects are frozen dataclasses validated in ``__post_init__``, so an invalid
configuration fails when it's built, not deep inside a backtest. ``EngineConfig`` can be
built from a nested mapping or a TOML file. Unknown keys and wrongly typed values are
rejected, so a typo can never silently fall back to a default.

Every window and delay is measured in **bars**, not wall-clock time, so the engine doesn't
depend on bar frequency (a "48-hour" cooldown is 2 bars on daily data or 48 on hourly).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from regime_engine.core.exceptions import ConfigError

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

_T = TypeVar("_T")


# --------------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------------


class CooldownMode(str, Enum):
    """Behaviour of the regime-flip cooldown.

    REDUCE_ONLY:    orders that reduce |exposure| are allowed; risk-increasing orders are blocked.
    LITERAL_FREEZE: no rebalancing orders at all (protective stops still fire).
    """

    REDUCE_ONLY = "reduce_only"
    LITERAL_FREEZE = "literal_freeze"


class DataSourceKind(str, Enum):
    CSV = "csv"
    YFINANCE = "yfinance"


class RangeVolMethod(str, Enum):
    """High-low range volatility estimator.

    PARKINSON (default): sqrt(mean(ln(H/L)^2) / (4 ln 2)), the *level* of range-implied
               volatility. It separates calm from turbulent markets.
    RANGE_STD: rolling standard deviation of ln(H/L), the *variability* of the daily range.
               Steady wide ranges and steady narrow ranges score alike, so it is a weaker
               regime separator. Kept for comparison.
    """

    RANGE_STD = "range_std"
    PARKINSON = "parkinson"


class ScalingMethod(str, Enum):
    ZSCORE = "zscore"
    MINMAX = "minmax"


class VolatilitySource(str, Enum):
    """Which per-state quantity ranks regimes by volatility during canonical mapping.

    RETURN_STD: sqrt of the state's return variance, Sigma_k[r, r], de-scaled (default).
    RANGE_VOL:  the state's mean high-low range volatility feature, de-scaled.
    """

    RETURN_STD = "return_std"
    RANGE_VOL = "range_vol"


class StrategyKind(str, Enum):
    LONG = "long"  # constant full-conviction long (direction 1): the risk-overlay default
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    DEFENSIVE = "defensive"  # constant reduced exposure (strategy.defensive_direction)
    FLAT = "flat"


class RoutingMode(str, Enum):
    """HARD: run the confirmed regime's strategy. BLENDED: probability-weight all four."""

    HARD = "hard"
    BLENDED = "blended"


class VaRMethod(str, Enum):
    """Which one-bar VaR estimate the VaR-budget rule uses.

    FILTERED_HISTORICAL (default): empirical quantile of returns standardised by the regime
    volatility forecast, rescaled by the current forecast (falls back to MIXTURE until enough
    standardised residuals exist).
    """

    FILTERED_HISTORICAL = "filtered_historical"
    MIXTURE = "mixture"
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"


class SizingMethod(str, Enum):
    """REGIME_VOL_TARGET: volatility from the HMM regime mixture (the engine's method).
    REALIZED_VOL_TARGET: volatility from trailing realised returns (the no-HMM baseline).
    UNIT: weight = direction (no volatility scaling), for diagnostics."""

    REGIME_VOL_TARGET = "regime_vol_target"
    REALIZED_VOL_TARGET = "realized_vol_target"
    UNIT = "unit"


class WindowType(str, Enum):
    ROLLING = "rolling"
    EXPANDING = "expanding"


# --------------------------------------------------------------------------------------------
# Section schemas
# --------------------------------------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


@dataclass(frozen=True, slots=True)
class RunConfig:
    seed: int = 42
    annualization: int = 252
    initial_capital: float = 1_000_000.0

    def __post_init__(self) -> None:
        _require(self.seed >= 0, "run.seed must be non-negative")
        _require(self.annualization > 0, "run.annualization must be positive")
        _require(self.initial_capital > 0, "run.initial_capital must be positive")


@dataclass(frozen=True, slots=True)
class DataConfig:
    symbol: str = "SPY"
    benchmark: str = "SPY"
    start: date = date(2005, 1, 1)
    end: date | None = None
    source: DataSourceKind = DataSourceKind.YFINANCE
    raw_dir: Path = Path("data/raw")
    cache_dir: Path = Path("data/cache")
    max_abs_log_return: float = 0.25
    max_calendar_gap_days: int = 7
    allow_zero_volume: bool = False
    extreme_return_allowlist: tuple[date, ...] = ()

    def __post_init__(self) -> None:
        _require(bool(self.symbol.strip()), "data.symbol must be non-empty")
        _require(bool(self.benchmark.strip()), "data.benchmark must be non-empty")
        _require(self.end is None or self.end > self.start, "data.end must be after data.start")
        _require(0.0 < self.max_abs_log_return <= 5.0, "data.max_abs_log_return must be in (0, 5]")
        _require(self.max_calendar_gap_days >= 1, "data.max_calendar_gap_days must be >= 1")


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    hl_window: int = 21
    volume_window: int = 21
    hl_method: RangeVolMethod = RangeVolMethod.PARKINSON

    def __post_init__(self) -> None:
        _require(self.hl_window >= 2, "features.hl_window must be >= 2")
        _require(self.volume_window >= 2, "features.volume_window must be >= 2")


@dataclass(frozen=True, slots=True)
class ScalingConfig:
    window: int = 252
    min_periods: int = 63
    method: ScalingMethod = ScalingMethod.ZSCORE
    ddof: int = 0
    include_current_bar: bool = True
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        _require(self.window >= 2, "scaling.window must be >= 2")
        _require(2 <= self.min_periods <= self.window, "scaling.min_periods must be in [2, window]")
        _require(self.ddof in (0, 1), "scaling.ddof must be 0 or 1")
        _require(self.epsilon > 0.0, "scaling.epsilon must be positive")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Walk-forward HMM training schedule.

    EXPANDING (default): each refit trains on every feature bar seen so far. Once a crisis
    enters the sample it stays there, so the Bear/High-Vol state keeps a genuine bear market
    to anchor on. ROLLING: each refit trains on the last ``train_window`` bars, which adapts
    faster but can lose its only bear market and end up labelling mild drawdowns as "crash".

    ``min_train_bars`` feature bars must be seen before the first fit (1260 is about 5 years
    of daily data). Refits then happen every ``refit_every`` bars.
    """

    window_type: WindowType = WindowType.EXPANDING
    train_window: int = 1260
    min_train_bars: int = 1260
    refit_every: int = 21

    def __post_init__(self) -> None:
        _require(self.train_window >= 50, "training.train_window must be >= 50")
        _require(self.min_train_bars >= 50, "training.min_train_bars must be >= 50")
        _require(
            self.window_type is WindowType.EXPANDING or self.min_train_bars <= self.train_window,
            "training.min_train_bars must not exceed train_window for rolling windows",
        )
        _require(self.refit_every >= 1, "training.refit_every must be >= 1")


@dataclass(frozen=True, slots=True)
class HMMConfig:
    """Gaussian HMM estimation (Baum-Welch with restarts)."""

    n_states: int = 4
    covariance_ridge: float = 1e-6
    max_iter: int = 500
    tol: float = 1e-4
    n_restarts: int = 10
    init_self_transition: float = 0.9
    sticky_prior: float = 0.0
    min_state_occupancy: float = 0.03
    kmeans_iter: int = 25
    refit_restarts: int = 2

    def __post_init__(self) -> None:
        _require(self.n_states == 4, "hmm.n_states must be 4 (the regime taxonomy is 2x2)")
        _require(self.covariance_ridge >= 0.0, "hmm.covariance_ridge must be non-negative")
        _require(self.max_iter >= 1, "hmm.max_iter must be >= 1")
        _require(self.tol > 0.0, "hmm.tol must be positive")
        _require(self.n_restarts >= 1, "hmm.n_restarts must be >= 1")
        _require(
            0.0 < self.init_self_transition < 1.0, "hmm.init_self_transition must be in (0, 1)"
        )
        _require(self.sticky_prior >= 0.0, "hmm.sticky_prior must be non-negative")
        _require(
            0.0 <= self.min_state_occupancy < 1.0 / self.n_states,
            "hmm.min_state_occupancy must be in [0, 1 / n_states)",
        )
        _require(self.kmeans_iter >= 1, "hmm.kmeans_iter must be >= 1")
        _require(
            1 <= self.refit_restarts <= self.n_restarts,
            "hmm.refit_restarts must be in [1, n_restarts]",
        )


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    """Hysteresis on the filtered regime probabilities.

    The confirmed regime switches to R' only after P(S_t = R' | x_1..t) >= ``confirm_prob``
    for ``confirm_bars`` consecutive bars. This filters flicker in *detection*. The trading
    cooldown (``CooldownConfig``) separately limits *trading* after a confirmed change.
    """

    confirm_prob: float = 0.6
    confirm_bars: int = 2

    def __post_init__(self) -> None:
        _require(0.5 <= self.confirm_prob <= 1.0, "regime.confirm_prob must be in [0.5, 1]")
        _require(self.confirm_bars >= 1, "regime.confirm_bars must be >= 1")


@dataclass(frozen=True, slots=True)
class MappingConfig:
    """Canonical regime labelling and its anomaly tolerances."""

    vol_source: VolatilitySource = VolatilitySource.RETURN_STD
    return_separation_tol: float = 1e-5
    vol_split_rel_tol: float = 0.05

    def __post_init__(self) -> None:
        _require(self.return_separation_tol >= 0.0, "mapping.return_separation_tol must be >= 0")
        _require(0.0 <= self.vol_split_rel_tol < 1.0, "mapping.vol_split_rel_tol must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Regime -> strategy routing and strategy parameters.

    Default: the **risk overlay**. Every regime is LONG, so the HMM acts only through sizing
    (volatility forecast) and the risk desk. The walk-forward evidence (docs/ARCHITECTURE.md
    §13) shows detected regimes classify volatility, not direction. A tactical
    direction layer cut Sharpe from 0.97 to 0.65 on SPY 2010-2026. ``tactical()`` returns that
    design (momentum / damped momentum / mean reversion / reduced-long defensive) for the
    ablation study.
    """

    bull_low: StrategyKind = StrategyKind.LONG
    bull_high: StrategyKind = StrategyKind.LONG
    bear_low: StrategyKind = StrategyKind.LONG
    bear_high: StrategyKind = StrategyKind.LONG
    routing: RoutingMode = RoutingMode.HARD
    momentum_lookbacks: tuple[int, ...] = (21, 63, 252)
    momentum_long_only: bool = True
    bull_high_damping: float = 0.5
    mean_reversion_window: int = 20
    mean_reversion_scale: float = 2.0
    defensive_direction: float = 0.5

    def __post_init__(self) -> None:
        _require(
            len(self.momentum_lookbacks) >= 1 and all(lb >= 2 for lb in self.momentum_lookbacks),
            "strategy.momentum_lookbacks must be non-empty, each >= 2",
        )
        _require(
            0.0 <= self.bull_high_damping <= 1.0, "strategy.bull_high_damping must be in [0, 1]"
        )
        _require(self.mean_reversion_window >= 2, "strategy.mean_reversion_window must be >= 2")
        _require(self.mean_reversion_scale > 0.0, "strategy.mean_reversion_scale must be positive")
        _require(
            -1.0 <= self.defensive_direction <= 1.0,
            "strategy.defensive_direction must be in [-1, 1]",
        )

    @classmethod
    def tactical(cls) -> StrategyConfig:
        """The Phase 4 regime-routed design, kept for the ablation study."""
        return cls(
            bull_low=StrategyKind.MOMENTUM,
            bull_high=StrategyKind.MOMENTUM,
            bear_low=StrategyKind.MEAN_REVERSION,
            bear_high=StrategyKind.DEFENSIVE,
        )


@dataclass(frozen=True, slots=True)
class SizingConfig:
    """Position sizing: weight = direction x target vol / forecast vol, capped.

    ``kelly_cap``: also cap |weight| at fractional Kelly lambda * mu / sigma^2, computed from the
    regime mixture (zero when the mixture drift opposes the signal). Off by default: the
    HMM's state means are contemporaneous, and on SPY the cap made the high-vol bear regime
    flat on 99% of bars (docs/ARCHITECTURE.md §13).
    """

    method: SizingMethod = SizingMethod.REGIME_VOL_TARGET
    target_vol_annual: float = 0.12
    max_leverage: float = 1.5
    kelly_cap: bool = False
    kelly_fraction: float = 0.25
    risk_free_annual: float = 0.0
    realized_vol_window: int = 21

    def __post_init__(self) -> None:
        _require(0.0 < self.target_vol_annual <= 1.0, "sizing.target_vol_annual must be in (0, 1]")
        _require(self.max_leverage > 0.0, "sizing.max_leverage must be positive")
        _require(0.0 < self.kelly_fraction <= 1.0, "sizing.kelly_fraction must be in (0, 1]")
        _require(self.realized_vol_window >= 2, "sizing.realized_vol_window must be >= 2")


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Automated risk desk.

    ``var_budget`` is the maximum one-bar VaR at ``var_alpha`` as a fraction of equity.
    ``dd_breaker`` flattens and halts new risk for ``dd_halt_bars`` once equity falls that far
    below its peak (the peak is reset when trading resumes). Stops are ATR multiples; a
    stop-out blocks risk-increasing trades for ``stop_lockout_bars``.
    """

    allow_short: bool = False
    var_alpha: float = 0.99
    var_window: int = 252
    var_budget: float = 0.025
    dd_breaker: float = 0.20
    dd_halt_bars: int = 21
    trailing_stop_atr: float | None = None
    profit_target_atr: float | None = None
    atr_window: int = 14
    stop_lockout_bars: int = 5
    var_method: VaRMethod = VaRMethod.FILTERED_HISTORICAL
    fhs_min_obs: int = 100

    def __post_init__(self) -> None:
        _require(0.5 < self.var_alpha < 1.0, "risk.var_alpha must be in (0.5, 1)")
        _require(self.var_window >= 20, "risk.var_window must be >= 20")
        _require(0.0 < self.var_budget < 1.0, "risk.var_budget must be in (0, 1)")
        _require(0.0 < self.dd_breaker < 1.0, "risk.dd_breaker must be in (0, 1)")
        _require(self.dd_halt_bars >= 1, "risk.dd_halt_bars must be >= 1")
        for name in ("trailing_stop_atr", "profit_target_atr"):
            value = getattr(self, name)
            _require(value is None or value > 0.0, f"risk.{name} must be positive or null")
        _require(self.atr_window >= 2, "risk.atr_window must be >= 2")
        _require(self.stop_lockout_bars >= 0, "risk.stop_lockout_bars must be >= 0")
        _require(
            20 <= self.fhs_min_obs <= self.var_window,
            "risk.fhs_min_obs must be in [20, var_window]",
        )


@dataclass(frozen=True, slots=True)
class CooldownConfig:
    mode: CooldownMode = CooldownMode.REDUCE_ONLY
    bars: int = 2

    def __post_init__(self) -> None:
        _require(self.bars >= 0, "cooldown.bars must be >= 0")


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Event-driven backtest settings.

    A rebalance order is sent only when |target - current weight| exceeds
    ``rebalance_threshold`` (or when exiting to flat), which avoids paying costs to chase
    small drifts.
    """

    rebalance_threshold: float = 0.02

    def __post_init__(self) -> None:
        _require(
            0.0 <= self.rebalance_threshold < 1.0, "backtest.rebalance_threshold must be in [0, 1)"
        )


@dataclass(frozen=True, slots=True)
class ExecutionCostConfig:
    """Transaction-cost model parameters. bps = basis points (1e-4)."""

    commission_bps: float = 0.5
    min_commission: float = 1.0
    half_spread_bps: float = 1.0
    borrow_fee_bps_annual: float = 50.0
    impact_coefficient: float = 0.1
    adv_window: int = 20
    latency_bars: int = 1

    def __post_init__(self) -> None:
        for name in (
            "commission_bps",
            "min_commission",
            "half_spread_bps",
            "borrow_fee_bps_annual",
            "impact_coefficient",
        ):
            _require(getattr(self, name) >= 0.0, f"execution.{name} must be non-negative")
        _require(self.adv_window >= 1, "execution.adv_window must be >= 1")
        # A signal formed on the close of bar t cannot be filled on bar t: that would be
        # trading on information at a price that no longer exists.
        _require(
            self.latency_bars >= 1,
            "execution.latency_bars must be >= 1 (same-bar fills introduce look-ahead bias)",
        )


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    level: str = "INFO"  # minimum level written to events.jsonl
    console: bool = True
    console_level: str = "WARNING"  # minimum level echoed to the terminal
    json_file: bool = True
    log_dir: Path = Path("runs")

    def __post_init__(self) -> None:
        for name in ("level", "console_level"):
            _require(
                getattr(self, name).upper() in _LOG_LEVELS,
                f"logging.{name} must be one of {sorted(_LOG_LEVELS)}",
            )


@dataclass(frozen=True, slots=True)
class EngineConfig:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    cooldown: CooldownConfig = field(default_factory=CooldownConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    execution: ExecutionCostConfig = field(default_factory=ExecutionCostConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        _require(
            self.scaling.window <= self.training.train_window,
            "scaling.window must not exceed training.train_window "
            "(the scaler is fit inside the training window)",
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EngineConfig:
        return _build(cls, raw, path="")

    @classmethod
    def from_toml(cls, path: Path | str) -> EngineConfig:
        p = Path(path)
        try:
            with p.open("rb") as fh:
                raw = tomllib.load(fh)
        except FileNotFoundError as exc:
            raise ConfigError(f"config file not found: {p}") from exc
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"invalid TOML in {p}: {exc}") from exc
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = _to_plain(dataclasses.asdict(self))
        return result

    def fingerprint(self) -> str:
        """Stable short hash of the full configuration, recorded in every run manifest."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------------------------
# Strict mapping -> dataclass construction
# --------------------------------------------------------------------------------------------


def _build(cls: type[_T], raw: Any, path: str) -> _T:
    where = path or "<root>"
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where}: expected a table, got {type(raw).__name__}")
    if not dataclasses.is_dataclass(cls):
        raise ConfigError(f"{where}: {cls!r} is not a config schema")
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(known)}")
    kwargs = {
        name: _coerce(value, hints[name], f"{path}.{name}" if path else name)
        for name, value in raw.items()
    }
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"{where}: {exc}") from exc


def _coerce(value: Any, tp: Any, path: str) -> Any:
    origin = get_origin(tp)

    if origin is Union or origin is types.UnionType:
        args = get_args(tp)
        if value is None and type(None) in args:
            return None
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) != 1:
            raise ConfigError(f"{path}: unsupported union type {tp}")
        return _coerce(value, non_none[0], path)

    if origin is tuple:
        args = get_args(tp)
        if len(args) != 2 or args[1] is not Ellipsis:
            raise ConfigError(f"{path}: only homogeneous tuple[T, ...] fields are supported")
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        return tuple(_coerce(v, args[0], f"{path}[{i}]") for i, v in enumerate(value))

    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return _build(tp, value, path)

    if isinstance(tp, type) and issubclass(tp, Enum):
        if isinstance(value, tp):
            return value
        try:
            return tp(value)
        except ValueError as exc:
            allowed = [m.value for m in tp]
            raise ConfigError(f"{path}: {value!r} is not one of {allowed}") from exc

    if tp is Path:
        if isinstance(value, (str, Path)):
            return Path(value)
        raise ConfigError(f"{path}: expected a path string, got {type(value).__name__}")

    if tp is date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError as exc:
                raise ConfigError(f"{path}: {value!r} is not an ISO date (YYYY-MM-DD)") from exc
        raise ConfigError(f"{path}: expected a date, got {type(value).__name__}")

    if tp is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{path}: expected a boolean, got {type(value).__name__}")

    if tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise ConfigError(f"{path}: expected an integer, got {type(value).__name__}")

    if tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise ConfigError(f"{path}: expected a number, got {type(value).__name__}")

    if tp is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"{path}: expected a string, got {type(value).__name__}")

    raise ConfigError(f"{path}: unsupported field type {tp!r}")


def _to_plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value
