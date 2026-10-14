"""Position sizing: direction -> signed portfolio weight.

* ``RegimeVolTargetSizer`` (the engine's method): volatility is forecast from the HMM regime
  mixture with **one-step-ahead** probabilities p_{t+1} = alpha_t A, because the position is
  held over the next bar:

      w = d * sigma_target / (sqrt(N) * sigma_mix)

  With ``kelly_cap``, |w| is also capped at fractional Kelly, lambda (mu_mix - r_f) / sigma_mix^2
  (the log-optimal fraction for Gaussian returns, scaled down because mu is estimated with a
  lot of error). The cap is zero when the mixture drift opposes the signal.
* ``RealizedVolTargetSizer`` (the no-HMM baseline): the same rule with sigma estimated from
  trailing realised returns. Comparing the two isolates what the HMM adds.
* ``UnitSizer``: w = d, for diagnostics.

All sizers clip |w| at ``max_leverage``. Shorting permissions belong to the risk desk, not here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from regime_engine.core.config import SizingConfig, SizingMethod
from regime_engine.core.types import Regime, RegimeMoments, Signal, TargetPosition
from regime_engine.risk.var import regime_mixture_moments

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class SizingInputs:
    """Everything a sizer may use. All of it is known at the close of the decision bar."""

    trailing_returns: FloatArray  # per-bar log returns up to and including the decision bar
    predicted_probs: FloatArray | None = None  # P(S_{t+1} | x_1..t), canonical order
    moments: Mapping[Regime, RegimeMoments] | None = None


class PositionSizer(Protocol):
    def size(self, signal: Signal, inputs: SizingInputs) -> TargetPosition: ...


def _clip(weight: float, max_leverage: float) -> float:
    return float(np.clip(weight, -max_leverage, max_leverage))


class RegimeVolTargetSizer:
    def __init__(
        self,
        target_vol_annual: float = 0.12,
        *,
        annualization: int = 252,
        max_leverage: float = 1.5,
        kelly_cap: bool = True,
        kelly_fraction: float = 0.25,
        risk_free_annual: float = 0.0,
    ) -> None:
        self._target = target_vol_annual
        self._n = annualization
        self._max_leverage = max_leverage
        self._kelly_cap = kelly_cap
        self._kelly_fraction = kelly_fraction
        self._rf = risk_free_annual / annualization

    def size(self, signal: Signal, inputs: SizingInputs) -> TargetPosition:
        if inputs.predicted_probs is None or inputs.moments is None:
            raise ValueError("regime sizing needs predicted regime probabilities and moments")
        mu, sigma = regime_mixture_moments(inputs.predicted_probs, inputs.moments)
        d = signal.direction
        components = {"direction": d, "mu_mix": mu, "sigma_mix": sigma}
        if d == 0.0 or sigma <= 0.0:
            return TargetPosition(signal.ts, 0.0, "zero direction", components)

        weight = d * self._target / (math.sqrt(self._n) * sigma)
        components["vol_weight"] = weight
        rationale = f"vol target {self._target:.0%} / forecast {sigma * math.sqrt(self._n):.1%}"
        if self._kelly_cap:
            kelly = self._kelly_fraction * (mu - self._rf) / sigma**2
            cap = max(0.0, kelly * math.copysign(1.0, d))
            components["kelly_cap"] = cap
            if abs(weight) > cap:
                weight = math.copysign(cap, d)
                rationale += f"; capped by {self._kelly_fraction:g}-Kelly"
        clipped = _clip(weight, self._max_leverage)
        if clipped != weight:
            rationale += f"; leverage cap {self._max_leverage:g}"
        return TargetPosition(signal.ts, clipped, rationale, components)


class RealizedVolTargetSizer:
    def __init__(
        self,
        target_vol_annual: float = 0.12,
        *,
        annualization: int = 252,
        max_leverage: float = 1.5,
        window: int = 21,
    ) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self._target = target_vol_annual
        self._n = annualization
        self._max_leverage = max_leverage
        self._window = window

    def size(self, signal: Signal, inputs: SizingInputs) -> TargetPosition:
        recent = np.asarray(inputs.trailing_returns, dtype=np.float64)[-self._window :]
        recent = recent[np.isfinite(recent)]
        d = signal.direction
        if len(recent) < 2:
            return TargetPosition(signal.ts, 0.0, "insufficient return history", {"direction": d})
        sigma = float(recent.std(ddof=1))
        components = {"direction": d, "sigma_realized": sigma}
        if d == 0.0 or sigma <= 0.0:
            return TargetPosition(signal.ts, 0.0, "zero direction", components)
        weight = d * self._target / (math.sqrt(self._n) * sigma)
        components["vol_weight"] = weight
        return TargetPosition(
            signal.ts,
            _clip(weight, self._max_leverage),
            f"vol target {self._target:.0%} / realized {sigma * math.sqrt(self._n):.1%}",
            components,
        )


class UnitSizer:
    def __init__(self, *, max_leverage: float = 1.0) -> None:
        self._max_leverage = max_leverage

    def size(self, signal: Signal, inputs: SizingInputs) -> TargetPosition:
        return TargetPosition(
            signal.ts,
            _clip(signal.direction, self._max_leverage),
            "unit sizing",
            {"direction": signal.direction},
        )


def build_sizer(config: SizingConfig, *, annualization: int = 252) -> PositionSizer:
    if config.method is SizingMethod.REGIME_VOL_TARGET:
        return RegimeVolTargetSizer(
            config.target_vol_annual,
            annualization=annualization,
            max_leverage=config.max_leverage,
            kelly_cap=config.kelly_cap,
            kelly_fraction=config.kelly_fraction,
            risk_free_annual=config.risk_free_annual,
        )
    if config.method is SizingMethod.REALIZED_VOL_TARGET:
        return RealizedVolTargetSizer(
            config.target_vol_annual,
            annualization=annualization,
            max_leverage=config.max_leverage,
            window=config.realized_vol_window,
        )
    return UnitSizer(max_leverage=config.max_leverage)
