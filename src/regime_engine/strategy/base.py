"""Strategy interface and the causal market context it receives.

A strategy maps a ``MarketContext`` to a *direction* in [-1, 1], i.e. conviction, never a size.
Sizing and risk are separate layers. ``MarketContext`` enforces causality by construction:
its price history ends exactly at the decision bar, and its regime state is for that bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from regime_engine.core.exceptions import LookAheadError
from regime_engine.core.types import RegimeState
from regime_engine.models.gaussian_hmm import FloatArray


@dataclass(frozen=True, eq=False)
class MarketContext:
    ts: pd.Timestamp
    history: pd.DataFrame  # OHLCV bars up to and including ``ts``
    regime: RegimeState

    def __post_init__(self) -> None:
        if len(self.history) == 0:
            raise LookAheadError("market context has no history")
        last = self.history.index[-1]
        if last != self.ts:
            raise LookAheadError(
                f"history must end at the decision bar {self.ts}, but ends at {last}"
            )
        if self.regime.ts != self.ts:
            raise LookAheadError(f"regime state is for {self.regime.ts}, not {self.ts}")

    @cached_property
    def close(self) -> FloatArray:
        return np.asarray(self.history["close"].to_numpy(dtype=np.float64))


@runtime_checkable
class Strategy(Protocol):
    @property
    def name(self) -> str: ...

    def direction(self, ctx: MarketContext) -> float:
        """Conviction for the next bar in [-1, 1]."""
        ...
