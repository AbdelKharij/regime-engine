"""Shared fixtures: deterministic synthetic OHLCV data and engine log capture."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
import pytest

from regime_engine.models.gaussian_hmm import HMMParams

OHLCVFactory = Callable[..., pd.DataFrame]


def _make_ohlcv(
    n: int = 600, *, seed: int = 7, start: str = "2015-01-02", daily_vol: float = 0.01
) -> pd.DataFrame:
    """Geometric random walk with internally consistent OHLC and log-normal volume."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=n, tz="UTC", name="date")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, daily_vol, n)))
    prev_close = np.concatenate([[100.0], close[:-1]])
    open_ = prev_close * np.exp(rng.normal(0.0, daily_vol / 4, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, daily_vol / 2, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, daily_vol / 2, n)))
    volume = np.exp(rng.normal(np.log(5e7), 0.3, n))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )


@pytest.fixture
def ohlcv_factory() -> OHLCVFactory:
    return _make_ohlcv


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return _make_ohlcv()


class ListHandler(logging.Handler):
    """Collects every record emitted under the ``regime_engine`` logger."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def events(self, name: str) -> list[logging.LogRecord]:
        return [r for r in self.records if r.__dict__.get("event") == name]


@pytest.fixture
def captured() -> Iterator[ListHandler]:
    handler = ListHandler()
    root = logging.getLogger("regime_engine")
    previous_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    yield handler
    root.removeHandler(handler)
    root.setLevel(previous_level)


# --------------------------------------------------------------------------------------------
# A known 4-regime model
# --------------------------------------------------------------------------------------------


def _covariance(stds: tuple[float, float, float], corr_01: float) -> np.ndarray:
    corr = np.eye(3)
    corr[0, 1] = corr[1, 0] = corr_01
    d = np.diag(stds)
    return d @ corr @ d


def make_canonical_truth() -> HMMParams:
    """A 4-regime model already in canonical order (BULL_LOW, BULL_HIGH, BEAR_LOW, BEAR_HIGH).

    Dimension 0 plays the return, dimension 1 the range volatility, dimension 2 the volume
    volatility. Return/vol orderings are unambiguous by construction.
    """
    return HMMParams(
        startprob=[0.4, 0.3, 0.2, 0.1],
        transmat=[
            [0.97, 0.01, 0.015, 0.005],
            [0.02, 0.94, 0.02, 0.02],
            [0.02, 0.01, 0.95, 0.02],
            [0.01, 0.03, 0.06, 0.90],
        ],
        means=[
            [0.8, -1.0, -0.5],
            [0.4, 0.3, 0.4],
            [-0.3, -0.4, 0.0],
            [-1.2, 1.8, 1.2],
        ],
        covars=np.stack(
            [
                _covariance((0.4, 0.3, 0.5), -0.2),
                _covariance((0.9, 0.4, 0.5), -0.3),
                _covariance((0.5, 0.3, 0.5), -0.1),
                _covariance((1.6, 0.6, 0.5), -0.5),
            ]
        ),
    )
