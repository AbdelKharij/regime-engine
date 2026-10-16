"""Data ingestion, validation, look-ahead checks and feature engineering."""

from regime_engine.data.features import FeatureEngine, RollingWindowScaler, ScalerParams
from regime_engine.data.ingestor import (
    DataIngestor,
    IssueCode,
    IssueSeverity,
    OHLCVValidator,
    ValidationIssue,
    ValidationReport,
)
from regime_engine.data.lookahead import (
    CorruptionMode,
    Divergence,
    LookAheadReport,
    LookAheadSanityChecker,
)
from regime_engine.data.sources import (
    CSVDataSource,
    InMemoryDataSource,
    MarketDataSource,
    YFinanceDataSource,
)

__all__ = [
    "CSVDataSource",
    "CorruptionMode",
    "DataIngestor",
    "Divergence",
    "FeatureEngine",
    "InMemoryDataSource",
    "IssueCode",
    "IssueSeverity",
    "LookAheadReport",
    "LookAheadSanityChecker",
    "MarketDataSource",
    "OHLCVValidator",
    "RollingWindowScaler",
    "ScalerParams",
    "ValidationIssue",
    "ValidationReport",
    "YFinanceDataSource",
]
