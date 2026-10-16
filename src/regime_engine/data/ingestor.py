"""Look-ahead-safe data ingestion and validation.

Pipeline: ``source.fetch`` → ``standardize`` → date-range slice → ``OHLCVValidator.validate``
→ (optional) cache.

Two rules shape this module:

1. **Never repair silently.** Unsorted, duplicated, missing or non-finite data raises a
   ``DataValidationError`` that lists *every* issue found. Silent sorting, forward-filling or
   dropping hides upstream problems and can itself introduce look-ahead (a backward-filled
   gap copies tomorrow's price into today).
2. **One canonical schema.** Downstream code can rely on a tz-aware UTC ``DatetimeIndex``
   named ``date`` (one row per trading date, normalized to midnight) and float64 columns
   ``open, high, low, close, volume``.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from regime_engine.core.config import DataConfig
from regime_engine.core.exceptions import DataAlignmentError, DataValidationError
from regime_engine.data.sources import MarketDataSource, build_data_source
from regime_engine.utils.logging import LogEvent, get_logger, log_context

OHLCV_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close")
INDEX_NAME: Final = "date"

_COLUMN_ALIASES: Final[Mapping[str, str]] = {
    "date": "date",
    "datetime": "date",
    "timestamp": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj_close": "adj_close",
    "adjclose": "adj_close",
    "adjusted_close": "adj_close",
    "volume": "volume",
}
_MAX_EXAMPLES: Final = 5
_NS_PER_DAY: Final = 86_400 * 10**9

_LOG = get_logger("data.ingestor")


# --------------------------------------------------------------------------------------------
# Validation report types
# --------------------------------------------------------------------------------------------


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class IssueCode(str, Enum):
    INVALID_INDEX = "invalid_index"
    MISSING_COLUMNS = "missing_columns"
    EMPTY = "empty"
    NON_NUMERIC = "non_numeric"
    NOT_SORTED = "not_sorted"
    DUPLICATE_TIMESTAMPS = "duplicate_timestamps"
    NAN_VALUES = "nan_values"
    INF_VALUES = "inf_values"
    NON_POSITIVE_PRICE = "non_positive_price"
    NEGATIVE_VOLUME = "negative_volume"
    ZERO_VOLUME = "zero_volume"
    OHLC_INCONSISTENT = "ohlc_inconsistent"
    EXTREME_RETURN = "extreme_return"
    CALENDAR_GAP = "calendar_gap"
    FUTURE_TIMESTAMP = "future_timestamp"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: IssueCode
    severity: IssueSeverity
    message: str
    count: int = 0
    examples: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidationReport:
    symbol: str
    n_rows: int
    issues: tuple[ValidationIssue, ...]

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is IssueSeverity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is IssueSeverity.WARNING)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def codes(self) -> frozenset[IssueCode]:
        return frozenset(i.code for i in self.issues)

    def summary(self) -> str:
        if not self.issues:
            return f"{self.symbol}: {self.n_rows} rows, no issues"
        lines = [f"{self.symbol}: {self.n_rows} rows, {len(self.issues)} issue(s)"]
        for issue in self.issues:
            examples = f" e.g. {', '.join(issue.examples)}" if issue.examples else ""
            lines.append(
                f"  [{issue.severity.value}] {issue.code.value}: {issue.message}{examples}"
            )
        return "\n".join(lines)


def _epoch_ns(index: pd.DatetimeIndex) -> npt.NDArray[np.int64]:
    return index.to_numpy(dtype="datetime64[ns]").astype(np.int64)


def _examples(index: pd.Index, mask: npt.NDArray[np.bool_]) -> tuple[str, ...]:
    positions = np.flatnonzero(mask)[:_MAX_EXAMPLES]
    return tuple(str(index[p]) for p in positions)


# --------------------------------------------------------------------------------------------
# Validator
# --------------------------------------------------------------------------------------------


class OHLCVValidator:
    """Strict validation of a standardized OHLCV frame.

    ``inspect`` gathers every issue without raising. ``validate`` raises if any ERROR-level
    issue exists. Return-based and gap-based checks run only when the index is sorted and
    unique, because both are meaningless on an unordered series.
    """

    def __init__(
        self,
        *,
        max_abs_log_return: float = 0.25,
        max_calendar_gap_days: int = 7,
        allow_zero_volume: bool = False,
        extreme_return_allowlist: Iterable[date] = (),
        ohlc_rel_tol: float = 1e-9,
        clock: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        self._max_abs_log_return = max_abs_log_return
        self._max_gap_days = max_calendar_gap_days
        self._allow_zero_volume = allow_zero_volume
        self._allowlist = frozenset(extreme_return_allowlist)
        self._ohlc_rel_tol = ohlc_rel_tol
        self._clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))

    @classmethod
    def from_config(cls, config: DataConfig) -> OHLCVValidator:
        return cls(
            max_abs_log_return=config.max_abs_log_return,
            max_calendar_gap_days=config.max_calendar_gap_days,
            allow_zero_volume=config.allow_zero_volume,
            extreme_return_allowlist=config.extreme_return_allowlist,
        )

    def inspect(self, df: pd.DataFrame, symbol: str) -> ValidationReport:
        issues: list[ValidationIssue] = []

        def report() -> ValidationReport:
            return ValidationReport(symbol, len(df), tuple(issues))

        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            issues.append(
                ValidationIssue(
                    IssueCode.INVALID_INDEX,
                    IssueSeverity.ERROR,
                    f"index must be a DatetimeIndex, got {type(idx).__name__}",
                )
            )
            return report()
        if idx.tz is None or str(idx.tz) != "UTC":
            issues.append(
                ValidationIssue(
                    IssueCode.INVALID_INDEX,
                    IssueSeverity.ERROR,
                    f"index must be tz-aware UTC, got tz={idx.tz}",
                )
            )

        missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing:
            issues.append(
                ValidationIssue(
                    IssueCode.MISSING_COLUMNS,
                    IssueSeverity.ERROR,
                    f"missing required columns {missing}",
                    count=len(missing),
                )
            )
            return report()
        if df.empty:
            issues.append(ValidationIssue(IssueCode.EMPTY, IssueSeverity.ERROR, "no rows"))
            return report()

        # --- ordering -------------------------------------------------------------------
        stamps = _epoch_ns(idx)
        backwards = np.zeros(len(idx), dtype=bool)
        backwards[1:] = np.diff(stamps) < 0
        if backwards.any():
            examples = tuple(
                f"{idx[p - 1]} -> {idx[p]}" for p in np.flatnonzero(backwards)[:_MAX_EXAMPLES]
            )
            issues.append(
                ValidationIssue(
                    IssueCode.NOT_SORTED,
                    IssueSeverity.ERROR,
                    "timestamps are not in ascending chronological order",
                    count=int(backwards.sum()),
                    examples=examples,
                )
            )
        duplicated = np.asarray(idx.duplicated(keep="first"), dtype=bool)
        if duplicated.any():
            issues.append(
                ValidationIssue(
                    IssueCode.DUPLICATE_TIMESTAMPS,
                    IssueSeverity.ERROR,
                    "duplicate timestamps",
                    count=int(duplicated.sum()),
                    examples=_examples(idx, duplicated),
                )
            )
        ordered = not backwards.any() and not duplicated.any()

        # --- numeric content -------------------------------------------------------------
        try:
            values = df.loc[:, list(OHLCV_COLUMNS)].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            issues.append(
                ValidationIssue(
                    IssueCode.NON_NUMERIC, IssueSeverity.ERROR, f"non-numeric OHLCV data: {exc}"
                )
            )
            return report()

        nan_mask = np.isnan(values)
        if nan_mask.any():
            per_col = {
                c: int(n) for c, n in zip(OHLCV_COLUMNS, nan_mask.sum(axis=0), strict=True) if n
            }
            issues.append(
                ValidationIssue(
                    IssueCode.NAN_VALUES,
                    IssueSeverity.ERROR,
                    f"missing (NaN) values per column: {per_col}",
                    count=int(nan_mask.sum()),
                    examples=_examples(idx, nan_mask.any(axis=1)),
                )
            )
        inf_mask = np.isinf(values)
        if inf_mask.any():
            per_col = {
                c: int(n) for c, n in zip(OHLCV_COLUMNS, inf_mask.sum(axis=0), strict=True) if n
            }
            issues.append(
                ValidationIssue(
                    IssueCode.INF_VALUES,
                    IssueSeverity.ERROR,
                    f"infinite values per column: {per_col}",
                    count=int(inf_mask.sum()),
                    examples=_examples(idx, inf_mask.any(axis=1)),
                )
            )

        finite_rows = np.isfinite(values).all(axis=1)
        prices = values[:, :4]
        volume = values[:, 4]

        non_positive = ((prices <= 0.0) & np.isfinite(prices)).any(axis=1)
        if non_positive.any():
            issues.append(
                ValidationIssue(
                    IssueCode.NON_POSITIVE_PRICE,
                    IssueSeverity.ERROR,
                    "prices must be strictly positive",
                    count=int(non_positive.sum()),
                    examples=_examples(idx, non_positive),
                )
            )
        negative_volume = volume < 0.0
        if negative_volume.any():
            issues.append(
                ValidationIssue(
                    IssueCode.NEGATIVE_VOLUME,
                    IssueSeverity.ERROR,
                    "volume must be non-negative",
                    count=int(negative_volume.sum()),
                    examples=_examples(idx, negative_volume),
                )
            )
        zero_volume = volume == 0.0
        if zero_volume.any() and not self._allow_zero_volume:
            issues.append(
                ValidationIssue(
                    IssueCode.ZERO_VOLUME,
                    IssueSeverity.ERROR,
                    "zero-volume bars (log-volume features are undefined); "
                    "set data.allow_zero_volume to accept them",
                    count=int(zero_volume.sum()),
                    examples=_examples(idx, zero_volume),
                )
            )

        o, h, lo, c = prices.T
        tol = self._ohlc_rel_tol * np.abs(prices).max(axis=1, initial=0.0)
        with np.errstate(invalid="ignore"):
            inconsistent = finite_rows & (
                (h < np.maximum(o, c) - tol) | (lo > np.minimum(o, c) + tol) | (h < lo)
            )
        if inconsistent.any():
            issues.append(
                ValidationIssue(
                    IssueCode.OHLC_INCONSISTENT,
                    IssueSeverity.ERROR,
                    "high must be >= max(open, close) and low <= min(open, close)",
                    count=int(inconsistent.sum()),
                    examples=_examples(idx, inconsistent),
                )
            )

        # --- time-series checks (only meaningful on an ordered index) ----------------------
        if ordered and len(idx) > 1:
            close_ok = np.isfinite(c) & (c > 0.0)
            if close_ok.all():
                log_ret = np.abs(np.diff(np.log(c)))
                extreme = np.zeros(len(idx), dtype=bool)
                extreme[1:] = log_ret > self._max_abs_log_return
                if self._allowlist:
                    allowed = np.array([ts.date() in self._allowlist for ts in idx], dtype=bool)
                    extreme &= ~allowed
                if extreme.any():
                    issues.append(
                        ValidationIssue(
                            IssueCode.EXTREME_RETURN,
                            IssueSeverity.ERROR,
                            f"|log return| > {self._max_abs_log_return} (bad tick or "
                            "unadjusted corporate action); allowlist the date if genuine",
                            count=int(extreme.sum()),
                            examples=_examples(idx, extreme),
                        )
                    )

            gaps = np.zeros(len(idx), dtype=bool)
            gaps[1:] = np.diff(_epoch_ns(idx.normalize())) > self._max_gap_days * _NS_PER_DAY
            if gaps.any():
                issues.append(
                    ValidationIssue(
                        IssueCode.CALENDAR_GAP,
                        IssueSeverity.WARNING,
                        f"gap longer than {self._max_gap_days} calendar days",
                        count=int(gaps.sum()),
                        examples=_examples(idx, gaps),
                    )
                )

        now = self._clock()
        horizon = now + pd.Timedelta(days=1)
        if idx.tz is None:  # already reported as INVALID_INDEX; compare like with like
            horizon = horizon.tz_convert(None)
        future = np.asarray(idx > horizon, dtype=bool)
        if future.any():
            issues.append(
                ValidationIssue(
                    IssueCode.FUTURE_TIMESTAMP,
                    IssueSeverity.ERROR,
                    f"timestamps after the current time {now.isoformat()}",
                    count=int(future.sum()),
                    examples=_examples(idx, future),
                )
            )

        return report()

    def validate(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Return ``df`` unchanged if valid, otherwise raise ``DataValidationError``."""
        report = self.inspect(df, symbol)
        for warning in report.warnings:
            _LOG.warning(
                LogEvent.DATA_VALIDATION_WARNING,
                warning.message,
                symbol=symbol,
                code=warning.code,
                count=warning.count,
                examples=warning.examples,
            )
        if not report.is_valid:
            _LOG.error(
                LogEvent.DATA_VALIDATION_FAILED,
                "data validation failed",
                symbol=symbol,
                codes=sorted(i.code.value for i in report.errors),
                rows=report.n_rows,
            )
            raise DataValidationError(
                report.summary(), symbol=symbol, codes=(i.code.value for i in report.errors)
            )
        _LOG.debug(
            LogEvent.DATA_VALIDATED,
            "data validated",
            symbol=symbol,
            rows=report.n_rows,
            warnings=len(report.warnings),
        )
        return df


# --------------------------------------------------------------------------------------------
# Ingestor
# --------------------------------------------------------------------------------------------


def _canonical_column(name: object) -> str:
    key = re.sub(r"[\s\-]+", "_", str(name).strip().lower())
    return _COLUMN_ALIASES.get(key, key)


class DataIngestor:
    """Fetches, standardizes, validates and caches daily OHLCV bars."""

    def __init__(
        self,
        source: MarketDataSource,
        validator: OHLCVValidator | None = None,
        *,
        cache_dir: Path | None = None,
    ) -> None:
        self._source = source
        self._validator = validator or OHLCVValidator()
        self._cache_dir = cache_dir

    @classmethod
    def from_config(
        cls, config: DataConfig, source: MarketDataSource | None = None
    ) -> DataIngestor:
        return cls(
            source or build_data_source(config),
            OHLCVValidator.from_config(config),
            cache_dir=config.cache_dir,
        )

    # ---- public API ------------------------------------------------------------------------

    def load(
        self, symbol: str, start: date, end: date | None = None, *, refresh: bool = False
    ) -> pd.DataFrame:
        """Load validated bars for ``symbol`` with dates in ``[start, end]`` (inclusive)."""
        if end is not None and end < start:
            raise ValueError(f"end {end} precedes start {start}")
        with log_context(symbol=symbol, source=self._source.name):
            cache_path = self._cache_path(symbol, start, end)
            from_cache = cache_path is not None and cache_path.is_file() and not refresh
            if from_cache:
                assert cache_path is not None
                raw = pd.read_csv(cache_path, float_precision="round_trip")
                _LOG.debug(LogEvent.DATA_CACHE_HIT, "loaded from cache", path=cache_path)
            else:
                raw = self._source.fetch(symbol, start, end)
                _LOG.debug(LogEvent.DATA_FETCHED, "fetched from source", rows=len(raw))

            df = self.standardize(raw)
            in_range = np.asarray(df.index >= pd.Timestamp(start, tz="UTC"), dtype=bool)
            if end is not None:
                in_range &= np.asarray(df.index <= pd.Timestamp(end, tz="UTC"), dtype=bool)
            df = df.loc[in_range]
            self._validator.validate(df, symbol)

            if not from_cache:
                self._write_cache(df, cache_path)

            _LOG.info(
                LogEvent.DATA_LOADED,
                "bars loaded",
                rows=len(df),
                first=df.index[0],
                last=df.index[-1],
                sha256=self.data_hash(df),
                from_cache=from_cache,
            )
            return df

    def load_aligned(
        self, symbols: Sequence[str], start: date, end: date | None = None
    ) -> dict[str, pd.DataFrame]:
        """Load several symbols and require identical trading calendars."""
        frames = {s: self.load(s, start, end) for s in dict.fromkeys(symbols)}
        self.assert_aligned(frames)
        return frames

    @staticmethod
    def standardize(raw: pd.DataFrame) -> pd.DataFrame:
        """Map an arbitrary raw frame onto the canonical schema. It doesn't sort or fill.

        * Columns are matched case-insensitively (``Adj Close`` → ``adj_close``).
        * If ``adj_close`` is present, OHLC are scaled by ``adj_close / close``. Volume is left
          unchanged (vendors already split-adjust volume).
        * The index becomes the trading date: tz-aware timestamps keep their exchange-local
          date, then everything is normalized to midnight UTC.
        """
        df = raw.copy()
        df.columns = pd.Index([_canonical_column(c) for c in df.columns])
        if "date" in df.columns:
            df = df.set_index("date")

        try:
            idx = pd.DatetimeIndex(pd.to_datetime(df.index))
        except (TypeError, ValueError):
            try:
                # Mixed UTC offsets (e.g. across a DST change) can only be parsed as UTC.
                idx = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True))
            except (TypeError, ValueError) as exc:
                raise DataValidationError(
                    f"cannot parse index as dates: {exc}", codes=(IssueCode.INVALID_INDEX.value,)
                ) from exc
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        idx = idx.normalize().tz_localize("UTC")
        idx.name = INDEX_NAME
        df.index = idx

        missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing:
            raise DataValidationError(
                f"missing required columns {missing}; got {list(df.columns)}",
                codes=(IssueCode.MISSING_COLUMNS.value,),
            )

        numeric = pd.DataFrame(df.apply(pd.to_numeric, errors="coerce"), dtype=np.float64)
        out = numeric.loc[:, list(OHLCV_COLUMNS)].copy()
        if "adj_close" in numeric.columns:
            factor = numeric["adj_close"] / numeric["close"]
            for col in PRICE_COLUMNS:
                out[col] = numeric[col] * factor
        return out

    @staticmethod
    def assert_aligned(frames: Mapping[str, pd.DataFrame]) -> None:
        """Raise ``DataAlignmentError`` unless all frames share exactly the same index."""
        if len(frames) < 2:
            return
        names = list(frames)
        reference = frames[names[0]].index
        problems: list[str] = []
        for name in names[1:]:
            other = frames[name].index
            if reference.equals(other):
                continue
            only_ref = reference.difference(other)
            only_other = other.difference(reference)
            problems.append(
                f"{names[0]} vs {name}: {len(only_ref)} date(s) only in {names[0]}"
                f" (e.g. {[str(t.date()) for t in only_ref[:3]]}), "
                f"{len(only_other)} only in {name}"
                f" (e.g. {[str(t.date()) for t in only_other[:3]]})"
            )
        if problems:
            _LOG.error(LogEvent.DATA_ALIGNMENT_FAILED, "indexes misaligned", problems=problems)
            raise DataAlignmentError(
                "time indexes are not aligned (no forward-filling is performed):\n  "
                + "\n  ".join(problems),
                codes=("misaligned",),
            )
        _LOG.debug(LogEvent.DATA_ALIGNED, "indexes aligned", symbols=names, rows=len(reference))

    @staticmethod
    def data_hash(df: pd.DataFrame) -> str:
        """Content hash of the data (index + values), recorded in run manifests."""
        hashed = pd.util.hash_pandas_object(df, index=True).to_numpy()
        return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]

    # ---- cache -------------------------------------------------------------------------------

    def _cache_path(self, symbol: str, start: date, end: date | None) -> Path | None:
        # An open-ended request ("up to today") changes daily, so it is never cached.
        if self._cache_dir is None or end is None:
            return None
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", symbol)
        return self._cache_dir / f"{safe}_{start.isoformat()}_{end.isoformat()}.csv"

    def _write_cache(self, df: pd.DataFrame, path: Path | None) -> None:
        if path is None:
            _LOG.debug(LogEvent.DATA_CACHE_SKIPPED, "cache disabled or open-ended range")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index_label=INDEX_NAME)
        _LOG.debug(LogEvent.DATA_CACHE_WRITE, "cached bars", path=path, rows=len(df))
