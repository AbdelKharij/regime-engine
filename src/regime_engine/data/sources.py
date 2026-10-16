"""Market data sources.

A source only *fetches* raw bars. Standardization, validation, slicing and caching are the
ingestor's job, so every source goes through exactly the same checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from regime_engine.core.config import DataConfig, DataSourceKind
from regime_engine.core.exceptions import DataValidationError


@runtime_checkable
class MarketDataSource(Protocol):
    """Returns raw daily bars. The date may be the index or a ``date`` column.

    Column names are case-insensitive. ``Open, High, Low, Close, Volume`` are required, and
    ``Adj Close`` is used for split/dividend adjustment when present.
    """

    @property
    def name(self) -> str: ...

    def fetch(self, symbol: str, start: date, end: date | None) -> pd.DataFrame: ...


class CSVDataSource:
    """Reads ``<directory>/<symbol>.csv`` (template configurable)."""

    def __init__(self, directory: Path, filename_template: str = "{symbol}.csv") -> None:
        self._directory = Path(directory)
        self._template = filename_template

    @property
    def name(self) -> str:
        return "csv"

    def path_for(self, symbol: str) -> Path:
        return self._directory / self._template.format(symbol=symbol)

    def fetch(self, symbol: str, start: date, end: date | None) -> pd.DataFrame:
        path = self.path_for(symbol)
        if not path.is_file():
            raise FileNotFoundError(f"no CSV data for {symbol!r} at {path}")
        # round_trip parsing guarantees floats read back bit-identical to what was written.
        return pd.read_csv(path, float_precision="round_trip")


class YFinanceDataSource:
    """Yahoo Finance daily bars via ``yfinance`` (optional dependency: ``pip install .[yahoo]``).

    With ``auto_adjust=True`` (the default), Yahoo returns OHLC already adjusted for splits and
    dividends, so returns and ranges stay consistent across corporate actions.
    """

    def __init__(self, *, auto_adjust: bool = True) -> None:
        self._auto_adjust = auto_adjust

    @property
    def name(self) -> str:
        return "yfinance"

    def fetch(self, symbol: str, start: date, end: date | None) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise ImportError(
                "yfinance is not installed: `pip install -e .[yahoo]` or use the CSV source"
            ) from exc
        # yfinance treats `end` as exclusive. Add a day so the requested end date is included.
        end_exclusive = (end or date.today()) + timedelta(days=1)
        frame: pd.DataFrame = yf.Ticker(symbol).history(
            start=start.isoformat(),
            end=end_exclusive.isoformat(),
            interval="1d",
            auto_adjust=self._auto_adjust,
            actions=False,
        )
        if frame.empty:
            raise DataValidationError(
                f"yfinance returned no rows for {symbol}", symbol=symbol, codes=("empty",)
            )
        return frame


class InMemoryDataSource:
    """Serves pre-built frames. Used for tests, notebooks and replaying cached snapshots."""

    def __init__(self, frames: Mapping[str, pd.DataFrame]) -> None:
        self._frames = dict(frames)
        self.fetch_count = 0

    @property
    def name(self) -> str:
        return "memory"

    def fetch(self, symbol: str, start: date, end: date | None) -> pd.DataFrame:
        self.fetch_count += 1
        if symbol not in self._frames:
            raise KeyError(f"no in-memory data for {symbol!r}")
        return self._frames[symbol].copy()


def build_data_source(config: DataConfig) -> MarketDataSource:
    if config.source is DataSourceKind.CSV:
        return CSVDataSource(config.raw_dir)
    if config.source is DataSourceKind.YFINANCE:
        return YFinanceDataSource()
    raise ValueError(f"unsupported data source {config.source!r}")  # pragma: no cover
