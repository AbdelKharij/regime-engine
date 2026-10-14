"""Phase 1 tests: configuration, logging, ingestion/validation, look-ahead checks, features,
and rolling-window scaling."""

from __future__ import annotations

import json
import logging
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_engine.core.config import (
    CooldownMode,
    DataConfig,
    EngineConfig,
    FeatureConfig,
    RangeVolMethod,
    ScalingMethod,
)
from regime_engine.core.exceptions import (
    ConfigError,
    DataAlignmentError,
    DataValidationError,
    InsufficientDataError,
    LookAheadError,
    ScalerNotFittedError,
)
from regime_engine.core.types import Regime, RegimeMoments
from regime_engine.data.features import FeatureEngine, RollingWindowScaler
from regime_engine.data.ingestor import (
    DataIngestor,
    IssueCode,
    IssueSeverity,
    OHLCVValidator,
)
from regime_engine.data.lookahead import (
    CorruptionMode,
    LookAheadSanityChecker,
)
from regime_engine.data.sources import InMemoryDataSource
from regime_engine.utils.logging import (
    ANOMALY_LOGGER_NAME,
    JsonFormatter,
    LogEvent,
    OrderingAnomalyKind,
    RegimeOrderingAnomalyChannel,
    get_logger,
    log_context,
)

from .conftest import ListHandler, OHLCVFactory

REPO_ROOT = Path(__file__).resolve().parents[1]


def _codes(exc_info: pytest.ExceptionInfo[DataValidationError]) -> frozenset[str]:
    return exc_info.value.codes


# ============================================================================================
# Configuration
# ============================================================================================


class TestConfig:
    def test_defaults(self) -> None:
        cfg = EngineConfig()
        assert cfg.data.symbol == "SPY"
        assert cfg.cooldown.mode is CooldownMode.REDUCE_ONLY
        assert cfg.features.hl_method is RangeVolMethod.PARKINSON
        assert cfg.execution.latency_bars >= 1

    def test_repo_config_file_loads(self) -> None:
        cfg = EngineConfig.from_toml(REPO_ROOT / "config" / "spy_daily.toml")
        assert cfg.data.start == date(2005, 1, 1)
        assert cfg.features.hl_method is RangeVolMethod.PARKINSON
        assert cfg.scaling.method is ScalingMethod.ZSCORE
        assert cfg.cooldown.mode is CooldownMode.REDUCE_ONLY

    def test_from_dict_coerces_nested_types(self) -> None:
        cfg = EngineConfig.from_dict(
            {
                "data": {
                    "symbol": "BTC-USD",
                    "start": "2018-01-01",
                    "end": "2024-12-31",
                    "cache_dir": "/tmp/cache",
                    "extreme_return_allowlist": ["2020-03-12"],
                },
                "cooldown": {"mode": "literal_freeze", "bars": 48},
                "execution": {"commission_bps": 1},
            }
        )
        assert cfg.data.end == date(2024, 12, 31)
        assert cfg.data.cache_dir == Path("/tmp/cache")
        assert cfg.data.extreme_return_allowlist == (date(2020, 3, 12),)
        assert cfg.cooldown.mode is CooldownMode.LITERAL_FREEZE
        assert isinstance(cfg.execution.commission_bps, float)

    @pytest.mark.parametrize(
        ("raw", "fragment"),
        [
            ({"data": {"symbl": "SPY"}}, "unknown key"),
            ({"cooldown": {"mode": "sometimes"}}, "not one of"),
            ({"execution": {"commission_bps": -1.0}}, "non-negative"),
            ({"execution": {"latency_bars": 0}}, "look-ahead"),
            ({"features": {"hl_window": "21"}}, "expected an integer"),
            ({"scaling": {"window": 2000}}, "must not exceed"),
            ({"data": {"start": "2020-01-01", "end": "2019-01-01"}}, "after data.start"),
        ],
    )
    def test_invalid_config_rejected(self, raw: dict[str, object], fragment: str) -> None:
        with pytest.raises(ConfigError, match=fragment):
            EngineConfig.from_dict(raw)

    def test_fingerprint_is_stable_and_sensitive(self) -> None:
        a, b = EngineConfig(), EngineConfig()
        assert a.fingerprint() == b.fingerprint()
        c = EngineConfig.from_dict({"features": {"hl_window": 10}})
        assert c.fingerprint() != a.fingerprint()


# ============================================================================================
# Structured logging
# ============================================================================================


class TestLogging:
    def test_json_record_carries_structure(self, captured: ListHandler) -> None:
        log = get_logger("test.component")
        with log_context(run_id="r1", symbol="SPY"):
            log.info(LogEvent.DATA_LOADED, "bars loaded", rows=10, last=pd.Timestamp("2024-01-02"))
        payload = json.loads(JsonFormatter().format(captured.records[-1]))
        assert payload["level"] == "INFO"
        assert payload["component"] == "test.component"
        assert payload["event"] == "DATA_LOADED"
        assert payload["context"] == {"run_id": "r1", "symbol": "SPY"}
        assert payload["data"]["rows"] == 10
        assert payload["data"]["last"].startswith("2024-01-02")
        assert payload["ts"].endswith("Z")

    def test_enum_values_render_plainly(self) -> None:
        from regime_engine.utils.logging import ConsoleFormatter, OrderingAnomalyKind, to_jsonable

        assert to_jsonable(OrderingAnomalyKind.BULL_NON_POSITIVE_MEAN) == "bull_non_positive_mean"
        record = logging.LogRecord("regime_engine.x", logging.WARNING, "", 0, "m", None, None)
        record.data = {"kind": OrderingAnomalyKind.BULL_NON_POSITIVE_MEAN}
        assert "kind=bull_non_positive_mean" in ConsoleFormatter().format(record)

    def test_console_level_is_separate(self, tmp_path: Path) -> None:
        from regime_engine.core.config import LoggingConfig
        from regime_engine.utils.logging import configure_logging, reset_logging

        try:
            configure_logging(LoggingConfig(level="DEBUG", console_level="ERROR", log_dir=tmp_path))
            handlers = logging.getLogger("regime_engine").handlers
            levels = sorted(h.level for h in handlers)
            assert levels == [logging.NOTSET, logging.ERROR]  # file: inherits DEBUG; console: ERROR
        finally:
            reset_logging()

    def test_context_does_not_leak(self, captured: ListHandler) -> None:
        log = get_logger("test.component")
        with log_context(run_id="r1"):
            pass
        log.info("EVENT", "outside")
        assert json.loads(JsonFormatter().format(captured.records[-1]))["context"] == {}

    def test_anomaly_channel_flags_non_monotonic_returns(self, captured: ListHandler) -> None:
        moments = {
            Regime.BULL_LOW_VOL: RegimeMoments(-0.0010, 0.008),  # "bull" with negative drift
            Regime.BULL_HIGH_VOL: RegimeMoments(0.0004, 0.015),
            Regime.BEAR_LOW_VOL: RegimeMoments(0.0002, 0.009),
            Regime.BEAR_HIGH_VOL: RegimeMoments(-0.0005, 0.030),
        }
        anomalies = RegimeOrderingAnomalyChannel().inspect(moments)
        kinds = {a.kind for a in anomalies}
        assert OrderingAnomalyKind.NON_MONOTONIC_RETURNS in kinds
        assert OrderingAnomalyKind.BULL_NON_POSITIVE_MEAN in kinds
        assert OrderingAnomalyKind.CRASH_NOT_LOWEST_RETURN in kinds
        warnings = [r for r in captured.records if r.name == ANOMALY_LOGGER_NAME]
        assert len(warnings) == len(anomalies)
        assert all(r.levelno == logging.WARNING for r in warnings)
        assert all(r.__dict__["event"] == "REGIME_ORDERING_ANOMALY" for r in warnings)

    def test_anomaly_channel_silent_on_clean_ordering(self, captured: ListHandler) -> None:
        moments = {
            Regime.BULL_LOW_VOL: RegimeMoments(0.0008, 0.007),
            Regime.BULL_HIGH_VOL: RegimeMoments(0.0005, 0.014),
            Regime.BEAR_LOW_VOL: RegimeMoments(-0.0002, 0.010),
            Regime.BEAR_HIGH_VOL: RegimeMoments(-0.0030, 0.035),
        }
        assert RegimeOrderingAnomalyChannel().inspect(moments) == ()
        assert not [r for r in captured.records if r.name == ANOMALY_LOGGER_NAME]


# ============================================================================================
# Data validation
# ============================================================================================


@pytest.fixture
def validator() -> OHLCVValidator:
    return OHLCVValidator()


class TestValidation:
    def test_clean_data_passes(self, validator: OHLCVValidator, ohlcv: pd.DataFrame) -> None:
        report = validator.inspect(ohlcv, "TEST")
        assert report.is_valid, report.summary()
        assert validator.validate(ohlcv, "TEST") is ohlcv

    def test_out_of_order_dates_flagged(
        self, validator: OHLCVValidator, ohlcv: pd.DataFrame
    ) -> None:
        order = np.arange(len(ohlcv))
        order[[100, 101]] = order[[101, 100]]
        shuffled = ohlcv.iloc[order]
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate(shuffled, "TEST")
        assert IssueCode.NOT_SORTED.value in _codes(exc_info)

    def test_fully_reversed_dates_flagged(
        self, validator: OHLCVValidator, ohlcv: pd.DataFrame
    ) -> None:
        report = validator.inspect(ohlcv.iloc[::-1], "TEST")
        assert IssueCode.NOT_SORTED in report.codes
        assert not report.is_valid

    def test_duplicate_timestamps_flagged(
        self, validator: OHLCVValidator, ohlcv: pd.DataFrame
    ) -> None:
        duplicated = pd.concat([ohlcv.iloc[:50], ohlcv.iloc[49:]])
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate(duplicated, "TEST")
        assert IssueCode.DUPLICATE_TIMESTAMPS.value in _codes(exc_info)

    @pytest.mark.parametrize("column", ["open", "high", "low", "close", "volume"])
    def test_missing_values_flagged(
        self, validator: OHLCVValidator, ohlcv: pd.DataFrame, column: str
    ) -> None:
        broken = ohlcv.copy()
        broken.iloc[200, broken.columns.get_loc(column)] = np.nan
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate(broken, "TEST")
        assert IssueCode.NAN_VALUES.value in _codes(exc_info)

    def test_inf_values_flagged(self, validator: OHLCVValidator, ohlcv: pd.DataFrame) -> None:
        broken = ohlcv.copy()
        broken.iloc[10, broken.columns.get_loc("volume")] = np.inf
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate(broken, "TEST")
        assert IssueCode.INF_VALUES.value in _codes(exc_info)

    def test_missing_column_flagged(self, validator: OHLCVValidator, ohlcv: pd.DataFrame) -> None:
        report = validator.inspect(ohlcv.drop(columns=["volume"]), "TEST")
        assert IssueCode.MISSING_COLUMNS in report.codes

    def test_ohlc_inconsistency_flagged(
        self, validator: OHLCVValidator, ohlcv: pd.DataFrame
    ) -> None:
        broken = ohlcv.copy()
        broken.iloc[5, broken.columns.get_loc("high")] = broken["low"].iloc[5] * 0.99
        assert IssueCode.OHLC_INCONSISTENT in validator.inspect(broken, "TEST").codes

    def test_non_positive_price_flagged(
        self, validator: OHLCVValidator, ohlcv: pd.DataFrame
    ) -> None:
        broken = ohlcv.copy()
        broken.iloc[5, broken.columns.get_loc("low")] = 0.0
        assert IssueCode.NON_POSITIVE_PRICE in validator.inspect(broken, "TEST").codes

    def test_naive_index_flagged(self, validator: OHLCVValidator, ohlcv: pd.DataFrame) -> None:
        naive = ohlcv.tz_localize(None)
        assert IssueCode.INVALID_INDEX in validator.inspect(naive, "TEST").codes

    def test_extreme_return_flagged_unless_allowlisted(self, ohlcv: pd.DataFrame) -> None:
        broken = ohlcv.copy()
        broken.iloc[300:, :4] *= 0.5  # ~ -69% log return: a bad tick or an unadjusted split
        shock_date = broken.index[300].date()
        assert IssueCode.EXTREME_RETURN in OHLCVValidator().inspect(broken, "T").codes
        allowed = OHLCVValidator(extreme_return_allowlist=[shock_date])
        assert allowed.inspect(broken, "T").is_valid

    def test_future_timestamps_flagged(self, ohlcv: pd.DataFrame) -> None:
        frozen_clock = OHLCVValidator(clock=lambda: pd.Timestamp("2015-06-01", tz="UTC"))
        assert IssueCode.FUTURE_TIMESTAMP in frozen_clock.inspect(ohlcv, "T").codes

    def test_calendar_gap_is_warning_not_error(
        self, validator: OHLCVValidator, ohlcv: pd.DataFrame
    ) -> None:
        gapped = ohlcv.drop(ohlcv.index[100:110])
        report = validator.inspect(gapped, "T")
        assert IssueCode.CALENDAR_GAP in report.codes
        assert report.is_valid
        assert all(i.severity is IssueSeverity.WARNING for i in report.issues)

    def test_all_issues_reported_together(
        self, validator: OHLCVValidator, ohlcv: pd.DataFrame
    ) -> None:
        broken = ohlcv.iloc[::-1].copy()
        broken.iloc[3, broken.columns.get_loc("close")] = np.nan
        codes = validator.inspect(broken, "T").codes
        assert {IssueCode.NOT_SORTED, IssueCode.NAN_VALUES} <= codes


# ============================================================================================
# Ingestion
# ============================================================================================


def _vendor_format(df: pd.DataFrame, adj_factor: float = 1.0) -> pd.DataFrame:
    """Re-express canonical bars the way a vendor would deliver them."""
    raw = df.rename(columns=str.title).copy()
    raw["Adj Close"] = raw["Close"] * adj_factor
    # Vendors stamp daily bars at exchange-local midnight.
    raw.index = df.index.tz_localize(None).tz_localize("America/New_York")
    raw.index.name = "Date"
    return raw


class TestIngestor:
    def test_standardizes_vendor_format(self, ohlcv: pd.DataFrame) -> None:
        source = InMemoryDataSource({"SPY": _vendor_format(ohlcv, adj_factor=0.5)})
        df = DataIngestor(source).load("SPY", date(2015, 1, 1), date(2030, 1, 1))
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert str(df.index.tz) == "UTC"
        assert df.index.name == "date"
        assert df.index.equals(ohlcv.index)  # local trading date preserved
        np.testing.assert_allclose(df["close"], ohlcv["close"] * 0.5, rtol=1e-15)
        np.testing.assert_allclose(df["volume"], ohlcv["volume"], rtol=0)

    def test_does_not_silently_sort(self, ohlcv: pd.DataFrame) -> None:
        source = InMemoryDataSource({"SPY": ohlcv.iloc[::-1]})
        with pytest.raises(DataValidationError) as exc_info:
            DataIngestor(source).load("SPY", date(2015, 1, 1), date(2030, 1, 1))
        assert IssueCode.NOT_SORTED.value in _codes(exc_info)

    def test_date_range_is_inclusive(self, ohlcv: pd.DataFrame) -> None:
        start, end = ohlcv.index[10].date(), ohlcv.index[50].date()
        df = DataIngestor(InMemoryDataSource({"SPY": ohlcv})).load("SPY", start, end)
        assert df.index[0].date() == start
        assert df.index[-1].date() == end
        assert len(df) == 41

    def test_cache_round_trip_is_bit_identical(self, ohlcv: pd.DataFrame, tmp_path: Path) -> None:
        source = InMemoryDataSource({"SPY": ohlcv})
        ingestor = DataIngestor(source, cache_dir=tmp_path)
        first = ingestor.load("SPY", date(2015, 1, 1), date(2030, 1, 1))
        second = ingestor.load("SPY", date(2015, 1, 1), date(2030, 1, 1))
        assert source.fetch_count == 1
        assert DataIngestor.data_hash(first) == DataIngestor.data_hash(second)
        pd.testing.assert_frame_equal(first, second, check_exact=True, check_freq=False)

    def test_from_config_wires_validator(self, ohlcv: pd.DataFrame) -> None:
        broken = ohlcv.copy()
        broken.iloc[300:, :4] *= 0.5
        cfg = DataConfig(extreme_return_allowlist=(broken.index[300].date(),))
        ingestor = DataIngestor.from_config(cfg, source=InMemoryDataSource({"SPY": broken}))
        assert len(ingestor.load("SPY", date(2015, 1, 1), date(2030, 1, 1))) == len(broken)

    def test_misaligned_symbols_rejected(self, ohlcv_factory: OHLCVFactory) -> None:
        a = ohlcv_factory(300, seed=1)
        b = ohlcv_factory(300, seed=2).drop(a.index[150])
        ingestor = DataIngestor(InMemoryDataSource({"A": a, "B": b}))
        with pytest.raises(DataAlignmentError, match="1 date"):
            ingestor.load_aligned(["A", "B"], date(2015, 1, 1), date(2030, 1, 1))


# ============================================================================================
# Feature engineering
# ============================================================================================


class TestFeatureEngine:
    def test_exactly_three_named_features(self, ohlcv: pd.DataFrame) -> None:
        features = FeatureEngine().compute(ohlcv)
        assert tuple(features.columns) == ("log_ret", "hl_vol", "vol_vol")
        assert not features.isna().to_numpy().any()

    def test_warmup_rows_dropped_not_filled(self, ohlcv: pd.DataFrame) -> None:
        engine = FeatureEngine(FeatureConfig(hl_window=10, volume_window=15))
        features = engine.compute(ohlcv)
        assert engine.warmup_bars == 15
        assert features.index[0] == ohlcv.index[15]
        raw = engine.compute(ohlcv, drop_warmup=False)
        assert raw["vol_vol"].iloc[:15].isna().all()
        assert raw["hl_vol"].iloc[:9].isna().all()
        assert raw["hl_vol"].iloc[9:].notna().all()

    def test_log_returns_match_definition(self, ohlcv: pd.DataFrame) -> None:
        features = FeatureEngine().compute(ohlcv)
        t = features.index[100]
        pos = ohlcv.index.get_loc(t)
        expected = math.log(ohlcv["close"].iloc[pos] / ohlcv["close"].iloc[pos - 1])
        assert features.loc[t, "log_ret"] == pytest.approx(expected, rel=1e-12)

    def test_range_std_uses_only_trailing_window(self, ohlcv: pd.DataFrame) -> None:
        w = 21
        engine = FeatureEngine(FeatureConfig(hl_window=w, hl_method=RangeVolMethod.RANGE_STD))
        features = engine.compute(ohlcv)
        pos = 250
        window = ohlcv.iloc[pos - w + 1 : pos + 1]
        expected = np.log(window["high"] / window["low"]).std(ddof=1)
        assert features.loc[ohlcv.index[pos], "hl_vol"] == pytest.approx(expected, rel=1e-10)

    def test_parkinson_uses_only_trailing_window(self, ohlcv: pd.DataFrame) -> None:
        w = 21
        engine = FeatureEngine(FeatureConfig(hl_window=w, hl_method=RangeVolMethod.PARKINSON))
        features = engine.compute(ohlcv)
        pos = 250
        window = ohlcv.iloc[pos - w + 1 : pos + 1]
        expected = math.sqrt(
            float((np.log(window["high"] / window["low"]) ** 2).mean()) / (4 * math.log(2))
        )
        assert features.loc[ohlcv.index[pos], "hl_vol"] == pytest.approx(expected, rel=1e-10)

    def test_volume_volatility_uses_only_trailing_window(self, ohlcv: pd.DataFrame) -> None:
        w = 21
        features = FeatureEngine(FeatureConfig(volume_window=w)).compute(ohlcv)
        pos = 250
        volume = ohlcv["volume"].iloc[pos - w : pos + 1].to_numpy()
        expected = np.diff(np.log(volume)).std(ddof=1)
        assert features.loc[ohlcv.index[pos], "vol_vol"] == pytest.approx(expected, rel=1e-10)

    def test_prefix_invariance(self, ohlcv: pd.DataFrame) -> None:
        """Features on a truncated history equal the full-history features, bit for bit."""
        engine = FeatureEngine()
        full = engine.compute(ohlcv)
        for k in (60, 200, 451):
            partial = engine.compute(ohlcv.iloc[:k])
            pd.testing.assert_frame_equal(partial, full.loc[partial.index], check_exact=True)

    def test_rejects_unsorted_and_non_positive_input(self, ohlcv: pd.DataFrame) -> None:
        engine = FeatureEngine()
        with pytest.raises(DataValidationError):
            engine.compute(ohlcv.iloc[::-1])
        zero_volume = ohlcv.copy()
        zero_volume.iloc[5, zero_volume.columns.get_loc("volume")] = 0.0
        with pytest.raises(DataValidationError):
            engine.compute(zero_volume)
        with pytest.raises(InsufficientDataError):
            engine.compute(ohlcv.iloc[:10])


# ============================================================================================
# Look-ahead sanity checker
# ============================================================================================


def _leak_centered_window(df: pd.DataFrame) -> pd.DataFrame:
    """Centered rolling mean: each bar sees 2 bars of the future."""
    return pd.DataFrame({"x": df["close"].rolling(5, center=True).mean()})


def _leak_forward_return(df: pd.DataFrame) -> pd.DataFrame:
    """Next-bar return stamped on the current bar (the classic label leak)."""
    return pd.DataFrame({"x": np.log(df["close"]).diff().shift(-1)})


def _leak_full_sample_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizing with full-sample moments: every bar 'knows' the future distribution."""
    r = np.log(df["close"]).diff()
    return pd.DataFrame({"x": (r - r.mean()) / r.std()})


def _leak_weekly_backfill(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly close back-filled onto daily bars: Monday 'knows' Friday's close."""
    weekly = df["close"].resample("W-FRI").last()
    return pd.DataFrame({"x": weekly.reindex(df.index, method="bfill")})


class TestLookAheadSanityChecker:
    @pytest.mark.parametrize("method", list(RangeVolMethod))
    @pytest.mark.parametrize("cutoff_pos", [60, 300, 598])
    def test_feature_engine_is_causal(
        self, ohlcv: pd.DataFrame, method: RangeVolMethod, cutoff_pos: int
    ) -> None:
        engine = FeatureEngine(FeatureConfig(hl_method=method))
        checker = LookAheadSanityChecker(engine, trials_per_mode=2, seed=cutoff_pos)
        report = checker.assert_no_lookahead(ohlcv, ohlcv.index[cutoff_pos])
        assert report.passed
        assert report.n_history_rows == cutoff_pos + 1 - engine.warmup_bars
        assert report.n_future_rows == len(ohlcv) - cutoff_pos - 1

    @pytest.mark.parametrize(
        "leaky_fn",
        [
            _leak_centered_window,
            _leak_forward_return,
            _leak_full_sample_zscore,
            _leak_weekly_backfill,
        ],
    )
    def test_detects_leaky_transforms(self, ohlcv: pd.DataFrame, leaky_fn: object) -> None:
        checker = LookAheadSanityChecker(leaky_fn)  # type: ignore[arg-type]
        # Mid-week cutoff: the weekly leak is invisible when the cutoff is the week's last bar.
        cutoff = next(ts for ts in ohlcv.index[300:] if ts.dayofweek == 2)
        with pytest.raises(LookAheadError):
            checker.assert_no_lookahead(ohlcv, cutoff)

    def test_divergence_pinpoints_the_leak(self, ohlcv: pd.DataFrame) -> None:
        cutoff = ohlcv.index[300]
        report = LookAheadSanityChecker(_leak_forward_return).run(ohlcv, cutoff)
        assert not report.passed
        # A one-bar forward leak contaminates exactly the last history bar.
        assert all(d.first_divergent_ts == cutoff for d in report.divergences)
        assert all(d.n_cells == 1 for d in report.divergences)

        report = LookAheadSanityChecker(_leak_centered_window).run(ohlcv, cutoff)
        # A centered 5-bar window looks 2 bars ahead.
        assert all(d.first_divergent_ts == ohlcv.index[299] for d in report.divergences)

    def test_compare_catches_one_ulp_tampering(self, ohlcv: pd.DataFrame) -> None:
        """Any modification of a historical value, however small, is caught."""
        features = FeatureEngine().compute(ohlcv)
        cutoff = features.index[200]
        tampered = features.copy()
        cell = tampered.iloc[150, 1]
        tampered.iloc[150, 1] = np.nextafter(cell, np.inf)
        divergence = LookAheadSanityChecker.compare(features, tampered, cutoff)
        assert divergence is not None
        assert divergence.first_divergent_ts == features.index[150]
        assert divergence.columns == ("hl_vol",)
        assert 0.0 < divergence.max_abs_diff < 1e-15

        # Tampering strictly after the cutoff is irrelevant to the history comparison.
        tampered_future = features.copy()
        tampered_future.iloc[250, 0] += 1.0
        assert LookAheadSanityChecker.compare(features, tampered_future, cutoff) is None

    def test_compare_catches_structural_tampering(self, ohlcv: pd.DataFrame) -> None:
        features = FeatureEngine().compute(ohlcv)
        cutoff = features.index[200]
        dropped = features.drop(features.index[100])
        divergence = LookAheadSanityChecker.compare(features, dropped, cutoff)
        assert divergence is not None
        assert divergence.structural is not None
        nan_injected = features.copy()
        nan_injected.iloc[10, 0] = np.nan
        divergence = LookAheadSanityChecker.compare(features, nan_injected, cutoff)
        assert divergence is not None
        assert math.isinf(divergence.max_abs_diff)

    @pytest.mark.parametrize("mode", list(CorruptionMode))
    def test_corruption_changes_only_the_future(
        self, ohlcv: pd.DataFrame, mode: CorruptionMode
    ) -> None:
        cutoff = ohlcv.index[400]
        corrupted = LookAheadSanityChecker.corrupt(ohlcv, cutoff, mode, np.random.default_rng(0))
        history = ohlcv.index <= cutoff
        pd.testing.assert_frame_equal(corrupted.loc[history], ohlcv.loc[history], check_exact=True)
        future_changed = (corrupted.loc[~history] != ohlcv.loc[~history]).any(axis=1)
        assert future_changed.mean() > 0.9
        # Corrupted rows stay valid, so a correct pipeline never crashes on them.
        assert (
            OHLCVValidator(max_abs_log_return=10.0).inspect(corrupted.loc[~history], "T").is_valid
        )

    def test_detects_input_mutation(self, ohlcv: pd.DataFrame) -> None:
        def mutating(df: pd.DataFrame) -> pd.DataFrame:
            df["close"] = df["close"] * 1.0 + 1.0
            return FeatureEngine().compute(df)

        report = LookAheadSanityChecker(mutating).run(ohlcv, ohlcv.index[300])
        assert any(d.structural and "mutated" in d.structural for d in report.divergences)

    def test_cutoff_edge_cases(self, ohlcv: pd.DataFrame) -> None:
        checker = LookAheadSanityChecker(FeatureEngine())
        with pytest.raises(InsufficientDataError):
            checker.run(ohlcv, ohlcv.index[-1])  # nothing to corrupt
        with pytest.raises(InsufficientDataError):
            checker.run(ohlcv, ohlcv.index[5])  # inside the warm-up period

    def test_accepts_date_cutoff(self, ohlcv: pd.DataFrame) -> None:
        checker = LookAheadSanityChecker(FeatureEngine())
        assert checker.run(ohlcv, ohlcv.index[300].date()).passed


# ============================================================================================
# Rolling-window scaler
# ============================================================================================


@pytest.fixture
def features(ohlcv_factory: OHLCVFactory) -> pd.DataFrame:
    return FeatureEngine().compute(ohlcv_factory(1200, seed=11))


class TestRollingWindowScaler:
    def test_rolling_zscore_matches_trailing_window(self, features: pd.DataFrame) -> None:
        w = 126
        scaled = RollingWindowScaler(w, min_periods=30).rolling_transform(features)
        for pos in (29, 125, 400, len(features) - 1):
            window = features.iloc[max(0, pos - w + 1) : pos + 1]
            expected = (features.iloc[pos] - window.mean()) / window.std(ddof=0)
            np.testing.assert_allclose(scaled.iloc[pos], expected, rtol=1e-9, atol=1e-12)
        assert scaled.iloc[:29].isna().all().all()

    def test_exclude_current_uses_strictly_prior_window(self, features: pd.DataFrame) -> None:
        w = 63
        scaler = RollingWindowScaler(w, include_current=False)
        scaled = scaler.rolling_transform(features)
        pos = 500
        window = features.iloc[pos - w : pos]
        expected = (features.iloc[pos] - window.mean()) / window.std(ddof=0)
        np.testing.assert_allclose(scaled.iloc[pos], expected, rtol=1e-9)
        assert scaled.iloc[:w].isna().all().all()

    @pytest.mark.parametrize("window", [5, 21, 252])
    def test_zscore_respects_samuelson_bound(self, features: pd.DataFrame, window: int) -> None:
        """With ddof=0 and the current bar in-window, |z| <= sqrt(n - 1) for n in-window obs.

        The bound is a hard mathematical limit that only holds if the statistics come from the
        bar's own trailing window, so it doubles as a check that no other data entered.
        """
        min_periods = max(2, window // 2)
        scaler = RollingWindowScaler(window, min_periods=min_periods)
        scaled = scaler.rolling_transform(features)
        n_obs = features.rolling(window, min_periods=1).count()
        bound = np.sqrt(n_obs - 1.0)
        within = (scaled.abs() <= bound + 1e-9) | scaled.isna()
        assert within.all().all()
        assert scaled.notna().sum().sum() > 0

    def test_rolling_scaled_variance_is_bounded(self, features: pd.DataFrame) -> None:
        """Each bar is scaled by its own trailing window, so the series of rolling z-scores is
        not exactly unit-variance (persistent features drift within a window), but it must
        stay within a band around 1. Hard per-bar bounds are tested via Samuelson above."""
        w = 252
        scaled = RollingWindowScaler(w, min_periods=w).rolling_transform(features).dropna()
        variance = scaled.var(ddof=0)
        assert ((variance > 0.5) & (variance < 2.0)).all(), variance

    def test_minmax_within_unit_interval(self, features: pd.DataFrame) -> None:
        scaler = RollingWindowScaler(63, min_periods=10, method=ScalingMethod.MINMAX)
        scaled = scaler.rolling_transform(features).dropna()
        assert (scaled >= 0.0).all().all()
        assert (scaled <= 1.0).all().all()

    def test_rolling_scaler_is_causal(self, ohlcv_factory: OHLCVFactory) -> None:
        engine = FeatureEngine()
        scaler = RollingWindowScaler(126, min_periods=30)
        pipeline = lambda df: scaler.rolling_transform(engine(df))  # noqa: E731
        data = ohlcv_factory(800, seed=3)
        report = LookAheadSanityChecker(pipeline, trials_per_mode=2).assert_no_lookahead(
            data, data.index[500]
        )
        assert report.passed

    def test_future_variance_shift_does_not_leak(self, features: pd.DataFrame) -> None:
        """A 10x volatility regime after bar k must not change scaled values before k.

        A full-sample scaler would shrink every pre-shift z-score (the classic leak); the
        rolling scaler's pre-shift output is bit-identical with or without the shift.
        """
        k = 700
        shifted = features.copy()
        shifted.iloc[k:] = shifted.iloc[k:] * 10.0
        scaler = RollingWindowScaler(252, min_periods=63)
        base = scaler.rolling_transform(features)
        after = scaler.rolling_transform(shifted)
        pd.testing.assert_frame_equal(base.iloc[:k], after.iloc[:k], check_exact=True)

        pre = after.iloc[63:k].dropna()
        assert ((pre.var(ddof=0) > 0.5) & (pre.var(ddof=0) < 2.0)).all()
        leaky = ((shifted - shifted.mean()) / shifted.std(ddof=0)).iloc[63:k]
        assert (leaky.var(ddof=0) < 0.1).all()

    def test_window_fit_has_unit_variance_in_sample(self, features: pd.DataFrame) -> None:
        w = 504
        scaled = RollingWindowScaler(w).fit_transform(features)
        assert len(scaled) == w
        np.testing.assert_allclose(scaled.mean(), 0.0, atol=1e-12)
        np.testing.assert_allclose(scaled.var(ddof=0), 1.0, rtol=1e-12)

    def test_fit_uses_only_trailing_window(self, features: pd.DataFrame) -> None:
        w = 252
        full = RollingWindowScaler(w).fit(features).params
        tail = RollingWindowScaler(w).fit(features.iloc[-w:]).params
        pd.testing.assert_series_equal(full.location, tail.location, check_exact=True)
        pd.testing.assert_series_equal(full.scale, tail.scale, check_exact=True)
        assert full.fit_start == features.index[-w]
        assert full.fit_end == features.index[-1]

    def test_transform_refuses_rows_before_fit_window(self, features: pd.DataFrame) -> None:
        w = 252
        train, test = features.iloc[300 : 300 + w], features.iloc[300 + w :]
        scaler = RollingWindowScaler(w).fit(train)
        out_of_sample = scaler.transform(test)  # strictly later rows: allowed
        assert out_of_sample.index.equals(test.index)
        with pytest.raises(LookAheadError):
            scaler.transform(features.iloc[:400])

    def test_unfitted_and_degenerate_cases(self, features: pd.DataFrame) -> None:
        with pytest.raises(ScalerNotFittedError):
            RollingWindowScaler(10).transform(features)
        constant = features.assign(vol_vol=1.0)
        scaled = RollingWindowScaler(50, min_periods=10).rolling_transform(constant)
        assert (scaled["vol_vol"].dropna() == 0.0).all()
        fitted = RollingWindowScaler(50).fit_transform(constant)
        assert np.isfinite(fitted.to_numpy()).all()

    def test_inverse_transform_round_trip(self, features: pd.DataFrame) -> None:
        scaler = RollingWindowScaler(252).fit(features)
        tail = features.iloc[-252:]
        np.testing.assert_allclose(
            scaler.inverse_transform(scaler.transform(tail)), tail, rtol=1e-12
        )
