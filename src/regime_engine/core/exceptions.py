"""Exception hierarchy for the regime engine.

Every error the engine raises deliberately derives from ``RegimeEngineError``, so callers can
separate engine-detected problems (bad data, look-ahead, bad config) from programming errors.
"""

from __future__ import annotations

from collections.abc import Iterable


class RegimeEngineError(Exception):
    """Base class for all errors raised deliberately by the engine."""


class ConfigError(RegimeEngineError, ValueError):
    """Configuration is malformed or violates an invariant."""


class DataValidationError(RegimeEngineError):
    """Market data failed a validation check.

    ``codes`` holds the machine-readable issue codes (``IssueCode`` values) so tests and callers
    can assert on the exact failure instead of parsing the message.
    """

    def __init__(
        self, message: str, *, symbol: str | None = None, codes: Iterable[str] = ()
    ) -> None:
        super().__init__(message)
        self.symbol = symbol
        self.codes: frozenset[str] = frozenset(codes)


class DataAlignmentError(DataValidationError):
    """Two or more series that must share a time index do not."""


class InsufficientDataError(RegimeEngineError):
    """Not enough observations to perform the requested computation."""


class LookAheadError(RegimeEngineError):
    """Information from after a decision timestamp influenced a value stamped at or before it."""


class ScalerNotFittedError(RegimeEngineError):
    """A scaler was used before ``fit`` was called."""


class HMMParameterError(RegimeEngineError, ValueError):
    """HMM parameters are malformed (non-stochastic matrices, non-PD covariances, shapes)."""


class HMMNumericalError(RegimeEngineError):
    """An HMM recursion underflowed or produced a non-finite value."""


class HMMFitError(RegimeEngineError):
    """No Baum-Welch restart produced an acceptable model."""


class RegimeMappingError(RegimeEngineError):
    """Fitted HMM states cannot be mapped onto the canonical regimes."""


class BacktestError(RegimeEngineError):
    """The simulation reached an invalid state (e.g. non-positive equity)."""
