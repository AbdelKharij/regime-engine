"""Causal feature engineering and look-ahead-safe scaling.

Every transformation here obeys one rule: **the value stamped at bar t is a function of bars
<= t only.** Rolling windows are trailing (``pandas.rolling`` default right alignment, never
``center=True``), nothing is backward-filled, and no statistic is estimated on the full
sample. ``LookAheadSanityChecker`` checks this empirically in the test suite.

Features (per bar t, window w)
------------------------------
* ``log_ret``  r_t = ln(C_t / C_{t-1})
* ``hl_vol``   PARKINSON (default): sqrt( mean{ ln(H_i / L_i)^2 } / (4 ln 2) )
               RANGE_STD: sample std of { ln(H_i / L_i) : i in (t-w, t] }
* ``vol_vol``  sample std of { ln V_i - ln V_{i-1} : i in (t-w, t] }
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Final

import numpy as np
import pandas as pd

from regime_engine.core.config import FeatureConfig, RangeVolMethod, ScalingConfig, ScalingMethod
from regime_engine.core.exceptions import (
    DataValidationError,
    InsufficientDataError,
    LookAheadError,
    ScalerNotFittedError,
)
from regime_engine.data.ingestor import OHLCV_COLUMNS
from regime_engine.utils.logging import LogEvent, get_logger

_LOG = get_logger("data.features")
_PARKINSON_CONST: Final = 1.0 / (4.0 * math.log(2.0))


def _require_time_series(frame: pd.DataFrame, what: str) -> None:
    idx = frame.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise DataValidationError(
            f"{what}: index must be a DatetimeIndex, got {type(idx).__name__}",
            codes=("invalid_index",),
        )
    if idx.has_duplicates:
        raise DataValidationError(
            f"{what}: index has duplicate timestamps", codes=("duplicate_timestamps",)
        )
    if not idx.is_monotonic_increasing:
        raise DataValidationError(
            f"{what}: index must be in ascending chronological order", codes=("not_sorted",)
        )


# --------------------------------------------------------------------------------------------
# Feature engine
# --------------------------------------------------------------------------------------------


class FeatureEngine:
    """Builds the 3-dimensional HMM observation vector from validated OHLCV bars."""

    FEATURE_COLUMNS: ClassVar[tuple[str, str, str]] = ("log_ret", "hl_vol", "vol_vol")

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self._cfg = config or FeatureConfig()

    @property
    def config(self) -> FeatureConfig:
        return self._cfg

    @property
    def warmup_bars(self) -> int:
        """Leading input rows that cannot produce a complete feature vector.

        ``log_ret`` is first defined at row 1. ``hl_vol`` needs ``hl_window`` ranges, so it is
        first defined at row ``hl_window - 1``. ``vol_vol`` needs ``volume_window`` volume
        *changes*, so it is first defined at row ``volume_window``.
        """
        return max(1, self._cfg.hl_window - 1, self._cfg.volume_window)

    def __call__(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        return self.compute(ohlcv)

    def compute(self, ohlcv: pd.DataFrame, *, drop_warmup: bool = True) -> pd.DataFrame:
        """Return a frame with columns ``FEATURE_COLUMNS`` on the input's index.

        With ``drop_warmup=True`` (default), the first ``warmup_bars`` rows are dropped, not
        filled, and the result contains no NaN.
        """
        self._check_input(ohlcv)
        features = pd.DataFrame(
            {
                "log_ret": self.log_returns(ohlcv["close"]),
                "hl_vol": self.range_volatility(
                    ohlcv["high"], ohlcv["low"], self._cfg.hl_window, self._cfg.hl_method
                ),
                "vol_vol": self.volume_volatility(ohlcv["volume"], self._cfg.volume_window),
            },
            index=ohlcv.index,
        )
        if drop_warmup:
            features = features.iloc[self.warmup_bars :]
            if features.isna().to_numpy().any():
                raise DataValidationError(
                    "NaN feature values after warm-up; input contains gaps", codes=("nan_values",)
                )
        _LOG.debug(
            LogEvent.FEATURES_BUILT,
            "features computed",
            rows=len(features),
            first=features.index[0] if len(features) else None,
            last=features.index[-1] if len(features) else None,
            hl_method=self._cfg.hl_method,
        )
        return features

    # ---- individual features (pure, causal) ----------------------------------------------

    @staticmethod
    def log_returns(close: pd.Series) -> pd.Series:
        values = close.to_numpy(dtype=np.float64)
        out = np.full(values.shape, np.nan)
        out[1:] = np.diff(np.log(values))
        return pd.Series(out, index=close.index, name="log_ret")

    @staticmethod
    def range_volatility(
        high: pd.Series,
        low: pd.Series,
        window: int,
        method: RangeVolMethod = RangeVolMethod.PARKINSON,
    ) -> pd.Series:
        log_range = pd.Series(
            np.log(high.to_numpy(dtype=np.float64) / low.to_numpy(dtype=np.float64)),
            index=high.index,
        )
        if method is RangeVolMethod.RANGE_STD:
            vol = log_range.rolling(window, min_periods=window).std(ddof=1)
        elif method is RangeVolMethod.PARKINSON:
            mean_sq = (log_range**2).rolling(window, min_periods=window).mean()
            vol = pd.Series(np.sqrt(_PARKINSON_CONST * mean_sq.to_numpy()), index=high.index)
        else:  # pragma: no cover - exhaustive over the enum
            raise ValueError(f"unknown range volatility method {method!r}")
        return vol.rename("hl_vol")

    @staticmethod
    def volume_volatility(volume: pd.Series, window: int) -> pd.Series:
        values = volume.to_numpy(dtype=np.float64)
        dlogv = np.full(values.shape, np.nan)
        dlogv[1:] = np.diff(np.log(values))
        series = pd.Series(dlogv, index=volume.index)
        return series.rolling(window, min_periods=window).std(ddof=1).rename("vol_vol")

    # ---- input checks ---------------------------------------------------------------------

    def _check_input(self, ohlcv: pd.DataFrame) -> None:
        missing = [c for c in OHLCV_COLUMNS if c not in ohlcv.columns]
        if missing:
            raise DataValidationError(
                f"missing OHLCV columns {missing}", codes=("missing_columns",)
            )
        _require_time_series(ohlcv, "FeatureEngine input")
        values = ohlcv.loc[:, ["high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise DataValidationError(
                "non-finite high/low/close/volume; validate data before feature engineering",
                codes=("nan_values",),
            )
        if (values <= 0.0).any():
            raise DataValidationError(
                "high/low/close/volume must be strictly positive (log transforms)",
                codes=("non_positive_price",),
            )
        if (values[:, 0] < values[:, 1]).any():
            raise DataValidationError("high < low", codes=("ohlc_inconsistent",))
        if len(ohlcv) <= self.warmup_bars:
            raise InsufficientDataError(
                f"need more than {self.warmup_bars} bars for one feature row, got {len(ohlcv)}"
            )


# --------------------------------------------------------------------------------------------
# Rolling-window scaler
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScalerParams:
    """Statistics of a scaler fit on one trailing window.

    ZSCORE: location = mean, scale = std(ddof). MINMAX: location = min, scale = max - min.
    """

    method: ScalingMethod
    location: pd.Series
    scale: pd.Series
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    n_obs: int


class RollingWindowScaler:
    """Feature scaling estimated only from a trailing window of historical bars.

    There are two modes, and neither ever looks at a future bar:

    * **Per-bar rolling** (``rolling_transform``): the value at bar t is scaled by statistics
      of the window ending at t (``include_current=True``) or at t-1 (``False``). This is fully
      causal and suits features consumed bar by bar.
    * **Window fit** (``fit`` / ``transform``): statistics are estimated on the *last*
      ``window`` rows passed to ``fit``. This mirrors how the HMM is trained on a trailing
      window. Rows inside the fit window are in-sample. Rows after it are out-of-sample.
      Transforming rows *before* the window is refused, because their scaled values would
      depend on statistics from their future.

    Variance guarantee (ZSCORE, ddof=0, include_current=True): by Samuelson's inequality,
    |z_t| <= sqrt(n_t - 1), where n_t <= window is the number of observations in the window.
    """

    def __init__(
        self,
        window: int = 252,
        *,
        min_periods: int | None = None,
        method: ScalingMethod = ScalingMethod.ZSCORE,
        ddof: int = 0,
        include_current: bool = True,
        epsilon: float = 1e-12,
    ) -> None:
        resolved_min = window if min_periods is None else min_periods
        if window < 2:
            raise ValueError("window must be >= 2")
        if not 2 <= resolved_min <= window:
            raise ValueError("min_periods must be in [2, window]")
        if ddof not in (0, 1):
            raise ValueError("ddof must be 0 or 1")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.window = window
        self.min_periods = resolved_min
        self.method = method
        self.ddof = ddof
        self.include_current = include_current
        self.epsilon = epsilon
        self._params: ScalerParams | None = None

    @classmethod
    def from_config(cls, config: ScalingConfig) -> RollingWindowScaler:
        return cls(
            config.window,
            min_periods=config.min_periods,
            method=config.method,
            ddof=config.ddof,
            include_current=config.include_current_bar,
            epsilon=config.epsilon,
        )

    @property
    def degenerate_fill(self) -> float:
        """Output for a constant window (zero spread): the window's centre."""
        return 0.0 if self.method is ScalingMethod.ZSCORE else 0.5

    @property
    def is_fitted(self) -> bool:
        return self._params is not None

    @property
    def params(self) -> ScalerParams:
        if self._params is None:
            raise ScalerNotFittedError("RollingWindowScaler.fit has not been called")
        return self._params

    # ---- window fit ----------------------------------------------------------------------

    def fit(self, X: pd.DataFrame) -> RollingWindowScaler:
        """Estimate statistics from the trailing ``window`` rows of ``X`` (the rest is ignored)."""
        self._check_frame(X, "fit")
        block = X.iloc[-self.window :]
        if len(block) < self.min_periods:
            raise InsufficientDataError(
                f"scaler needs at least {self.min_periods} rows, got {len(block)}"
            )
        if self.method is ScalingMethod.ZSCORE:
            location = block.mean()
            scale = block.std(ddof=self.ddof)
        else:
            location = block.min()
            scale = block.max() - location
        self._params = ScalerParams(
            method=self.method,
            location=location,
            scale=scale,
            fit_start=block.index[0],
            fit_end=block.index[-1],
            n_obs=len(block),
        )
        _LOG.debug(
            LogEvent.SCALER_FIT,
            "scaler fit on trailing window",
            method=self.method,
            fit_start=block.index[0],
            fit_end=block.index[-1],
            n_obs=len(block),
            location=location.to_dict(),
            scale=scale.to_dict(),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        params = self.params
        self._check_frame(X, "transform")
        if list(X.columns) != list(params.location.index):
            raise DataValidationError(
                f"columns {list(X.columns)} differ from fitted {list(params.location.index)}"
            )
        early = np.asarray(X.index < params.fit_start, dtype=bool)
        if early.any():
            raise LookAheadError(
                f"{int(early.sum())} row(s) precede the scaler's fit window "
                f"(starts {params.fit_start}); scaling them would use statistics from "
                "their future"
            )
        degenerate = params.scale <= self.epsilon
        safe_scale = params.scale.mask(degenerate, 1.0)
        out = (X - params.location) / safe_scale
        if degenerate.any():
            out.loc[:, degenerate.to_numpy()] = self.degenerate_fill
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit on the trailing window of ``X`` and return the scaled rows of that window."""
        self.fit(X)
        return self.transform(X.iloc[-self.window :])

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        params = self.params
        return X * params.scale + params.location

    # ---- per-bar rolling -----------------------------------------------------------------

    def rolling_statistics(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(location, scale) per bar, each computed from the trailing window only."""
        self._check_frame(X, "rolling_statistics")
        source = X if self.include_current else X.shift(1)
        roll = source.rolling(self.window, min_periods=self.min_periods)
        if self.method is ScalingMethod.ZSCORE:
            location = roll.mean()
            scale = roll.std(ddof=self.ddof)
        else:
            location = roll.min()
            scale = roll.max() - location
        return location, scale

    def rolling_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Causal per-bar scaling. Warm-up rows (fewer than ``min_periods`` bars) are NaN."""
        location, scale = self.rolling_statistics(X)
        degenerate = (scale <= self.epsilon) & location.notna()
        out = (X - location) / scale.mask(degenerate)
        return out.mask(degenerate, self.degenerate_fill)

    # ---- checks ----------------------------------------------------------------------------

    @staticmethod
    def _check_frame(X: pd.DataFrame, what: str) -> None:
        _require_time_series(X, f"RollingWindowScaler.{what}")
        values = X.to_numpy(dtype=np.float64)
        if np.isinf(values).any():
            raise DataValidationError(f"RollingWindowScaler.{what}: infinite values in input")
        if what in ("fit", "transform") and np.isnan(values).any():
            raise DataValidationError(f"RollingWindowScaler.{what}: NaN values in input")
