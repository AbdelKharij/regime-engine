"""Structured logging pipeline built on the standard ``logging`` module.

Design
------
* Every engine component logs through an ``EventLogger``. Each record carries a
  machine-readable ``event`` name, the ``component`` that emitted it, a ``data`` payload, and
  the ambient execution ``context`` (run id, symbol, bar timestamp, ...).
* Execution context is held in a ``ContextVar`` and set with ``log_context(...)``. Nested
  contexts merge, and context does not leak across threads or async tasks.
* ``configure_logging`` installs a human-readable console handler and a JSON-lines file
  handler (``runs/<run_id>/events.jsonl``). The JSONL file can be replayed for diagnostics.
* ``RegimeOrderingAnomalyChannel`` is a dedicated warning channel (logger
  ``regime_engine.anomaly.regime_ordering``, plus its own ``anomalies.jsonl`` file) for
  suspicious results when HMM states are re-ordered into canonical regimes.

This module is ``regime_engine.utils.logging``. Inside it, ``import logging`` resolves to the
standard library because imports are absolute.
"""

from __future__ import annotations

import json
import logging
import math
import secrets
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from regime_engine.core.config import LoggingConfig
from regime_engine.core.types import Regime, RegimeMoments

ROOT_LOGGER_NAME: Final = "regime_engine"
ANOMALY_COMPONENT: Final = "anomaly.regime_ordering"
ANOMALY_LOGGER_NAME: Final = f"{ROOT_LOGGER_NAME}.{ANOMALY_COMPONENT}"
_HANDLER_TAG: Final = "_regime_engine_handler"

_EMPTY_CONTEXT: Mapping[str, Any] = MappingProxyType({})
_CONTEXT: ContextVar[Mapping[str, Any]] = ContextVar(
    "regime_engine_log_context", default=_EMPTY_CONTEXT
)


class LogEvent(str, Enum):
    """Canonical event names. Logs are queried by event, never by message text."""

    LOGGING_CONFIGURED = "LOGGING_CONFIGURED"
    CONFIG_LOADED = "CONFIG_LOADED"
    DATA_FETCHED = "DATA_FETCHED"
    DATA_CACHE_HIT = "DATA_CACHE_HIT"
    DATA_CACHE_WRITE = "DATA_CACHE_WRITE"
    DATA_CACHE_SKIPPED = "DATA_CACHE_SKIPPED"
    DATA_VALIDATED = "DATA_VALIDATED"
    DATA_VALIDATION_WARNING = "DATA_VALIDATION_WARNING"
    DATA_VALIDATION_FAILED = "DATA_VALIDATION_FAILED"
    DATA_LOADED = "DATA_LOADED"
    DATA_ALIGNED = "DATA_ALIGNED"
    DATA_ALIGNMENT_FAILED = "DATA_ALIGNMENT_FAILED"
    FEATURES_BUILT = "FEATURES_BUILT"
    SCALER_FIT = "SCALER_FIT"
    LOOKAHEAD_CHECK_PASSED = "LOOKAHEAD_CHECK_PASSED"
    LOOKAHEAD_CHECK_FAILED = "LOOKAHEAD_CHECK_FAILED"
    REGIME_ORDERING_ANOMALY = "REGIME_ORDERING_ANOMALY"
    HMM_RESTART_REJECTED = "HMM_RESTART_REJECTED"
    HMM_LOGLIK_DECREASE = "HMM_LOGLIK_DECREASE"
    HMM_FIT = "HMM_FIT"
    REGIME_MAPPING = "REGIME_MAPPING"
    REGIME_MAPPING_DISAGREEMENT = "REGIME_MAPPING_DISAGREEMENT"
    ENGINE_READY = "ENGINE_READY"
    HMM_REFIT = "HMM_REFIT"
    HMM_REFIT_FAILED = "HMM_REFIT_FAILED"
    REGIME_CHANGE = "REGIME_CHANGE"
    RISK_MODIFY = "RISK_MODIFY"
    RISK_VETO = "RISK_VETO"
    COOLDOWN_START = "COOLDOWN_START"
    COOLDOWN_END = "COOLDOWN_END"
    STOP_TRIGGERED = "STOP_TRIGGERED"
    DD_BREAKER_TRIPPED = "DD_BREAKER_TRIPPED"
    DD_BREAKER_REARMED = "DD_BREAKER_REARMED"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    BACKTEST_STARTED = "BACKTEST_STARTED"
    BACKTEST_COMPLETED = "BACKTEST_COMPLETED"


# --------------------------------------------------------------------------------------------
# Execution context
# --------------------------------------------------------------------------------------------


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Attach key/value context to every record logged inside the ``with`` block."""
    merged = {**_CONTEXT.get(), **fields}
    token = _CONTEXT.set(MappingProxyType(merged))
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def current_context() -> dict[str, Any]:
    return dict(_CONTEXT.get())


# --------------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------------


def to_jsonable(value: Any) -> Any:
    """Convert numpy/pandas/enum/path values into plain JSON-serialisable Python values."""
    if isinstance(value, Enum):  # before str: str-mixin enums must render as their value
        return to_jsonable(value.value)
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, int):
        return value
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    return str(value)


def _utc_iso(created: float) -> str:
    stamp = datetime.fromtimestamp(created, tz=timezone.utc).isoformat(timespec="milliseconds")
    return stamp.replace("+00:00", "Z")


def _record_component(record: logging.LogRecord) -> str:
    component = getattr(record, "component", None)
    if isinstance(component, str):
        return component
    prefix = ROOT_LOGGER_NAME + "."
    return record.name[len(prefix) :] if record.name.startswith(prefix) else record.name


def _record_event(record: logging.LogRecord) -> str:
    event = getattr(record, "event", None)
    return event if isinstance(event, str) else "LOG"


def _record_mapping(record: logging.LogRecord, attr: str) -> Mapping[str, Any]:
    value = getattr(record, attr, None)
    return value if isinstance(value, Mapping) else {}


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, component, event, message, context, data."""

    def format(self, record: logging.LogRecord) -> str:
        context = _record_mapping(record, "context") or current_context()
        payload: dict[str, Any] = {
            "ts": _utc_iso(record.created),
            "level": record.levelname,
            "component": _record_component(record),
            "event": _record_event(record),
            "message": record.getMessage(),
            "context": to_jsonable(context),
            "data": to_jsonable(_record_mapping(record, "data")),
            "logger": record.name,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, allow_nan=False)


def _compact(value: Any) -> str:
    plain = to_jsonable(value)
    if isinstance(plain, float):
        return f"{plain:.6g}"
    if isinstance(plain, (dict, list)):
        return json.dumps(plain, separators=(",", ":"))
    return str(plain)


class ConsoleFormatter(logging.Formatter):
    """Human-readable single-line format that keeps the structured fields visible."""

    def format(self, record: logging.LogRecord) -> str:
        context = _record_mapping(record, "context") or current_context()
        fields = {**context, **_record_mapping(record, "data")}
        kv = " ".join(f"{k}={_compact(v)}" for k, v in fields.items())
        line = (
            f"{_utc_iso(record.created)} | {record.levelname:<8} | "
            f"{_record_component(record):<24} | {_record_event(record):<24} | "
            f"{record.getMessage()}"
        )
        if kv:
            line = f"{line} | {kv}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


# --------------------------------------------------------------------------------------------
# Event logger
# --------------------------------------------------------------------------------------------


class EventLogger:
    """Thin typed wrapper that enforces the (event, message, **data) logging convention."""

    __slots__ = ("_logger", "component")

    def __init__(self, component: str) -> None:
        self.component = component
        self._logger = logging.getLogger(f"{ROOT_LOGGER_NAME}.{component}")

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def log(
        self,
        level: int,
        event: LogEvent | str,
        message: str,
        /,
        *,
        exc_info: bool = False,
        **data: Any,
    ) -> None:
        if not self._logger.isEnabledFor(level):
            return
        name = event.value if isinstance(event, LogEvent) else event
        self._logger.log(
            level,
            message,
            extra={
                "event": name,
                "component": self.component,
                "context": current_context(),
                "data": data,
            },
            exc_info=exc_info,
            stacklevel=3,
        )

    def debug(self, event: LogEvent | str, message: str, /, **data: Any) -> None:
        self.log(logging.DEBUG, event, message, **data)

    def info(self, event: LogEvent | str, message: str, /, **data: Any) -> None:
        self.log(logging.INFO, event, message, **data)

    def warning(self, event: LogEvent | str, message: str, /, **data: Any) -> None:
        self.log(logging.WARNING, event, message, **data)

    def error(
        self, event: LogEvent | str, message: str, /, *, exc_info: bool = False, **data: Any
    ) -> None:
        self.log(logging.ERROR, event, message, exc_info=exc_info, **data)


def get_logger(component: str) -> EventLogger:
    """Return the event logger for a dotted component name, e.g. ``"data.ingestor"``."""
    return EventLogger(component)


# --------------------------------------------------------------------------------------------
# Pipeline configuration
# --------------------------------------------------------------------------------------------


def new_run_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)


def _tag(handler: logging.Handler) -> logging.Handler:
    setattr(handler, _HANDLER_TAG, True)
    return handler


def _remove_engine_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            logger.removeHandler(handler)
            handler.close()


def reset_logging() -> None:
    """Remove the engine's handlers and restore default propagation (tests, notebooks)."""
    root = logging.getLogger(ROOT_LOGGER_NAME)
    _remove_engine_handlers(root)
    _remove_engine_handlers(logging.getLogger(ANOMALY_LOGGER_NAME))
    root.setLevel(logging.NOTSET)
    root.propagate = True


def configure_logging(config: LoggingConfig, run_id: str | None = None) -> Path | None:
    """Install the engine's handlers. Calling it more than once is safe.

    Returns the run directory holding ``events.jsonl`` and ``anomalies.jsonl`` when file
    logging is enabled, otherwise ``None``.
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)
    anomaly = logging.getLogger(ANOMALY_LOGGER_NAME)
    _remove_engine_handlers(root)
    _remove_engine_handlers(anomaly)

    root.setLevel(config.level.upper())
    root.propagate = False

    if config.console:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(config.console_level.upper())
        console.setFormatter(ConsoleFormatter())
        root.addHandler(_tag(console))

    run_dir: Path | None = None
    if config.json_file:
        run_dir = config.log_dir / (run_id or new_run_id())
        run_dir.mkdir(parents=True, exist_ok=True)

        events = logging.FileHandler(run_dir / "events.jsonl", encoding="utf-8")
        events.setFormatter(JsonFormatter())
        root.addHandler(_tag(events))

        # Dedicated anomaly sink. Records still propagate to the root handlers as well.
        anomalies = logging.FileHandler(run_dir / "anomalies.jsonl", encoding="utf-8")
        anomalies.setLevel(logging.WARNING)
        anomalies.setFormatter(JsonFormatter())
        anomaly.addHandler(_tag(anomalies))

    get_logger("utils.logging").info(
        LogEvent.LOGGING_CONFIGURED,
        "logging pipeline configured",
        level=config.level.upper(),
        run_dir=run_dir,
    )
    return run_dir


# --------------------------------------------------------------------------------------------
# Regime-ordering anomaly channel
# --------------------------------------------------------------------------------------------


class OrderingAnomalyKind(str, Enum):
    MISSING_REGIME = "missing_regime"
    NON_MONOTONIC_RETURNS = "non_monotonic_returns"
    INSEPARABLE_RETURNS = "inseparable_returns"
    BULL_NON_POSITIVE_MEAN = "bull_non_positive_mean"
    BEAR_NON_NEGATIVE_MEAN = "bear_non_negative_mean"
    CRASH_NOT_LOWEST_RETURN = "crash_not_lowest_return"
    VOL_ORDER_INVERTED = "vol_order_inverted"
    DEGENERATE_VOL_SPLIT = "degenerate_vol_split"


@dataclass(frozen=True, slots=True)
class OrderingAnomaly:
    kind: OrderingAnomalyKind
    detail: str


_BULL: Final = (Regime.BULL_LOW_VOL, Regime.BULL_HIGH_VOL)
_BEAR: Final = (Regime.BEAR_LOW_VOL, Regime.BEAR_HIGH_VOL)


def detect_ordering_anomalies(
    moments: Mapping[Regime, RegimeMoments],
    *,
    return_separation_tol: float = 1e-5,
    vol_split_rel_tol: float = 0.05,
) -> tuple[OrderingAnomaly, ...]:
    """Check canonically ordered regime moments against the assumptions behind the labels.

    The canonical mapping (rank states by mean return, then by volatility within each pair)
    always produces *some* labelling. This function flags when that labelling is
    economically doubtful, e.g. a "bull" state with a negative drift, or a "crash" state that
    is not the worst-returning state.
    """
    missing = [r for r in Regime if r not in moments]
    if missing:
        return (
            OrderingAnomaly(
                OrderingAnomalyKind.MISSING_REGIME,
                f"moments missing for {[r.name for r in missing]}",
            ),
        )

    mu = {r: moments[r].mean_return for r in Regime}
    vol = {r: moments[r].volatility for r in Regime}
    found: list[OrderingAnomaly] = []

    min_bull = min(mu[r] for r in _BULL)
    max_bear = max(mu[r] for r in _BEAR)
    if min_bull < max_bear:
        found.append(
            OrderingAnomaly(
                OrderingAnomalyKind.NON_MONOTONIC_RETURNS,
                f"min bull mean {min_bull:.3e} < max bear mean {max_bear:.3e}: "
                "mean returns are not monotonic across the canonical bull/bear split",
            )
        )
    elif min_bull - max_bear < return_separation_tol:
        found.append(
            OrderingAnomaly(
                OrderingAnomalyKind.INSEPARABLE_RETURNS,
                f"bull/bear mean-return gap {min_bull - max_bear:.3e} "
                f"< tolerance {return_separation_tol:.1e}",
            )
        )

    for regime in _BULL:
        if mu[regime] <= 0.0:
            found.append(
                OrderingAnomaly(
                    OrderingAnomalyKind.BULL_NON_POSITIVE_MEAN,
                    f"{regime.name} has non-positive mean return {mu[regime]:.3e}",
                )
            )
    for regime in _BEAR:
        if mu[regime] >= 0.0:
            found.append(
                OrderingAnomaly(
                    OrderingAnomalyKind.BEAR_NON_NEGATIVE_MEAN,
                    f"{regime.name} has non-negative mean return {mu[regime]:.3e}",
                )
            )

    if mu[Regime.BEAR_HIGH_VOL] > min(mu.values()):
        worst = min(Regime, key=lambda r: mu[r])
        found.append(
            OrderingAnomaly(
                OrderingAnomalyKind.CRASH_NOT_LOWEST_RETURN,
                f"BEAR_HIGH_VOL mean {mu[Regime.BEAR_HIGH_VOL]:.3e} is above "
                f"{worst.name} mean {mu[worst]:.3e}",
            )
        )

    for low, high in (_BULL, _BEAR):
        if vol[high] <= vol[low]:
            found.append(
                OrderingAnomaly(
                    OrderingAnomalyKind.VOL_ORDER_INVERTED,
                    f"{high.name} vol {vol[high]:.3e} <= {low.name} vol {vol[low]:.3e}",
                )
            )
        elif (vol[high] - vol[low]) / vol[high] < vol_split_rel_tol:
            found.append(
                OrderingAnomaly(
                    OrderingAnomalyKind.DEGENERATE_VOL_SPLIT,
                    f"{low.name}/{high.name} vol split "
                    f"{(vol[high] - vol[low]) / vol[high]:.2%} < {vol_split_rel_tol:.0%}",
                )
            )

    return tuple(found)


class RegimeOrderingAnomalyChannel:
    """Dedicated WARNING channel for anomalies flagged during state re-ordering."""

    def __init__(
        self, *, return_separation_tol: float = 1e-5, vol_split_rel_tol: float = 0.05
    ) -> None:
        self._return_tol = return_separation_tol
        self._vol_tol = vol_split_rel_tol
        self._log = get_logger(ANOMALY_COMPONENT)

    def inspect(
        self,
        moments: Mapping[Regime, RegimeMoments],
        *,
        refit_ts: datetime | None = None,
        raw_state_order: Sequence[int] | None = None,
    ) -> tuple[OrderingAnomaly, ...]:
        anomalies = detect_ordering_anomalies(
            moments, return_separation_tol=self._return_tol, vol_split_rel_tol=self._vol_tol
        )
        snapshot = {
            r.name: {"mean_return": m.mean_return, "volatility": m.volatility}
            for r, m in moments.items()
        }
        for anomaly in anomalies:
            self._log.warning(
                LogEvent.REGIME_ORDERING_ANOMALY,
                anomaly.detail,
                kind=anomaly.kind,
                moments=snapshot,
                refit_ts=refit_ts,
                raw_state_order=list(raw_state_order) if raw_state_order is not None else None,
            )
        return anomalies
