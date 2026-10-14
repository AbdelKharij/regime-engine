"""Core domain types shared across layers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum, IntEnum

import numpy as np
import numpy.typing as npt
import pandas as pd


class Regime(IntEnum):
    """Canonical market regimes.

    The integer values are the canonical state order: after every HMM refit, the fitted
    parameters are permuted so that state ``k`` always means ``Regime(k)``.
    """

    BULL_LOW_VOL = 0
    BULL_HIGH_VOL = 1
    BEAR_LOW_VOL = 2
    BEAR_HIGH_VOL = 3  # "Crash"

    @property
    def is_bull(self) -> bool:
        return self in (Regime.BULL_LOW_VOL, Regime.BULL_HIGH_VOL)

    @property
    def is_high_vol(self) -> bool:
        return self in (Regime.BULL_HIGH_VOL, Regime.BEAR_HIGH_VOL)


@dataclass(frozen=True, slots=True)
class RegimeMoments:
    """Per-regime return moments in original (unscaled) units, per bar."""

    mean_return: float
    volatility: float


@dataclass(frozen=True, slots=True, eq=False)
class RegimeState:
    """The engine's causal view of the market regime at the close of bar ``ts``."""

    ts: pd.Timestamp
    probs: npt.NDArray[np.float64]  # (4,) filtered P(S_t = k | x_1..t), canonical order
    regime: Regime  # confirmed regime (after hysteresis)
    raw_argmax: Regime  # most probable regime this bar, before hysteresis
    changed: bool  # confirmed regime differs from the previous bar's
    refit: bool  # the model was re-estimated on this bar
    model_version: int  # number of successful fits so far (1 = first model)
    mapping_ambiguous: bool  # current model's regime labels raised ordering anomalies
    predictive_loglik: float  # log p(x_t | x_1..t-1) under the current model ("surprise")

    def __post_init__(self) -> None:
        probs = np.array(self.probs, dtype=np.float64)
        probs.setflags(write=False)
        object.__setattr__(self, "probs", probs)

    @property
    def confidence(self) -> float:
        return float(self.probs.max())


class Side(IntEnum):
    LONG = 1
    SHORT = -1


@dataclass(frozen=True, slots=True)
class Signal:
    """Strategy conviction for the next bar. ``direction`` is in [-1, 1] and is NOT a size."""

    ts: pd.Timestamp
    direction: float
    strategy: str
    regime: Regime

    def __post_init__(self) -> None:
        if not -1.0 <= self.direction <= 1.0:
            raise ValueError(f"signal direction {self.direction} outside [-1, 1]")


@dataclass(frozen=True, slots=True)
class TargetPosition:
    """Desired signed exposure as a fraction of equity (|weight| > 1 means leverage)."""

    ts: pd.Timestamp
    weight: float
    rationale: str
    components: Mapping[str, float] = field(default_factory=dict)


class RiskAction(str, Enum):
    PASS = "pass"
    MODIFY = "modify"
    VETO = "veto"


@dataclass(frozen=True, slots=True)
class RuleOutcome:
    rule: str
    action: RiskAction
    weight_before: float
    weight_after: float
    reason: str


@dataclass(frozen=True, slots=True)
class RiskDecision:
    ts: pd.Timestamp
    requested: TargetPosition
    approved: TargetPosition
    trace: tuple[RuleOutcome, ...]

    @property
    def modified(self) -> bool:
        return any(o.action is not RiskAction.PASS for o in self.trace)
