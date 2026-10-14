"""Canonical regime labelling: solving HMM label switching.

HMM state indices are arbitrary and change from one refit to the next. ``RegimeMapper`` turns
a fitted model into canonical form, where state ``k`` always means ``Regime(k)``:

1. Compute each state's mean return and volatility in **original units**, de-scaling with the
   affine transform the features were scaled by.
2. Rank states by mean return: the top two are Bull, the bottom two are Bear.
3. Within each pair, the lower-volatility state is Low-Vol and the higher one is High-Vol.
4. Permute **all** parameters together (pi, A, mu, Sigma), so ``A'[i, j]`` keeps the same
   economic meaning, e.g. P(BULL_LOW_VOL -> BEAR_HIGH_VOL), across refits.

The ranking always produces *a* labelling. Whether that labelling is economically sound
(e.g. a "bull" state with negative drift) is checked by the regime-ordering anomaly channel.
If the previous mapping is supplied, a Hungarian assignment on moment distance is computed as
an independent cross-check, and any disagreement with the rank-based labels is logged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from regime_engine.core.config import MappingConfig, VolatilitySource
from regime_engine.core.exceptions import RegimeMappingError
from regime_engine.core.types import Regime, RegimeMoments
from regime_engine.data.features import ScalerParams
from regime_engine.models.gaussian_hmm import FloatArray, HMMParams, IntArray
from regime_engine.utils.logging import (
    LogEvent,
    OrderingAnomaly,
    RegimeOrderingAnomalyChannel,
    get_logger,
)

_LOG = get_logger("models.regime_mapper")
_N_REGIMES = len(Regime)


@dataclass(frozen=True, eq=False)
class FeatureScaling:
    """Affine map from model space back to original units: x = z * scale + location."""

    names: tuple[str, ...]
    location: FloatArray
    scale: FloatArray

    def __post_init__(self) -> None:
        location = np.array(self.location, dtype=np.float64)
        scale = np.array(self.scale, dtype=np.float64)
        if location.shape != (len(self.names),) or scale.shape != (len(self.names),):
            raise RegimeMappingError("location/scale must have one entry per feature name")
        if not (np.isfinite(location).all() and np.isfinite(scale).all() and (scale > 0).all()):
            raise RegimeMappingError("scaling location must be finite and scale positive")
        location.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def identity(cls, names: Sequence[str]) -> FeatureScaling:
        n = len(names)
        return cls(tuple(names), np.zeros(n), np.ones(n))

    @classmethod
    def from_scaler_params(cls, params: ScalerParams) -> FeatureScaling:
        names = tuple(str(name) for name in params.location.index)
        scale = params.scale.to_numpy(dtype=np.float64)
        # A degenerate (constant) feature is scaled with 1.0 by the scaler; mirror that here.
        return cls(
            names, params.location.to_numpy(dtype=np.float64), np.where(scale > 0, scale, 1.0)
        )

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise RegimeMappingError(f"feature {name!r} not in {list(self.names)}") from exc


@dataclass(frozen=True, eq=False)
class RegimeMapping:
    order: IntArray  # order[k] = raw state index assigned to Regime(k)
    canonical_params: HMMParams
    moments: Mapping[Regime, RegimeMoments]
    anomalies: tuple[OrderingAnomaly, ...]
    hungarian_agrees: bool | None  # None when no previous mapping was supplied

    @property
    def ambiguous(self) -> bool:
        return bool(self.anomalies)

    @property
    def raw_to_regime(self) -> dict[int, Regime]:
        return {int(raw): Regime(k) for k, raw in enumerate(self.order)}

    def relabel_states(self, raw_states: npt.ArrayLike) -> IntArray:
        """Map raw state indices (e.g. a Viterbi path of the raw model) to canonical ones."""
        inverse = np.empty(_N_REGIMES, dtype=np.int64)
        inverse[self.order] = np.arange(_N_REGIMES)
        return np.asarray(inverse[np.asarray(raw_states, dtype=np.int64)], dtype=np.int64)

    def relabel_probs(self, raw_probs: npt.ArrayLike) -> FloatArray:
        """Reorder the last axis of raw-state probabilities into canonical regime order."""
        return np.asarray(np.asarray(raw_probs, dtype=np.float64)[..., self.order])


class RegimeMapper:
    def __init__(
        self,
        config: MappingConfig | None = None,
        *,
        return_feature: str = "log_ret",
        range_feature: str = "hl_vol",
        channel: RegimeOrderingAnomalyChannel | None = None,
    ) -> None:
        self._cfg = config or MappingConfig()
        self._return_feature = return_feature
        self._range_feature = range_feature
        self._channel = channel or RegimeOrderingAnomalyChannel(
            return_separation_tol=self._cfg.return_separation_tol,
            vol_split_rel_tol=self._cfg.vol_split_rel_tol,
        )

    def state_moments(
        self, params: HMMParams, scaling: FeatureScaling
    ) -> tuple[FloatArray, FloatArray]:
        """Per-state (mean return, volatility) in original units, in raw state order."""
        if len(scaling.names) != params.n_features:
            raise RegimeMappingError(
                f"scaling describes {len(scaling.names)} features, model has {params.n_features}"
            )
        r = scaling.index(self._return_feature)
        mean_return = params.means[:, r] * scaling.scale[r] + scaling.location[r]
        if self._cfg.vol_source is VolatilitySource.RETURN_STD:
            volatility = np.sqrt(params.covars[:, r, r]) * scaling.scale[r]
        else:
            h = scaling.index(self._range_feature)
            volatility = params.means[:, h] * scaling.scale[h] + scaling.location[h]
        return np.asarray(mean_return, dtype=np.float64), np.asarray(volatility, dtype=np.float64)

    def map(
        self,
        params: HMMParams,
        scaling: FeatureScaling,
        *,
        previous: RegimeMapping | None = None,
        refit_ts: datetime | None = None,
    ) -> RegimeMapping:
        if params.n_states != _N_REGIMES:
            raise RegimeMappingError(
                f"canonical mapping needs exactly {_N_REGIMES} states, got {params.n_states}"
            )
        mean_return, volatility = self.state_moments(params, scaling)

        by_return = np.argsort(-mean_return, kind="stable")
        bull, bear = by_return[:2], by_return[2:]
        bull = bull[np.argsort(volatility[bull], kind="stable")]
        bear = bear[np.argsort(volatility[bear], kind="stable")]
        order = np.concatenate([bull, bear]).astype(np.int64)
        order.setflags(write=False)

        moments = MappingProxyType(
            {
                Regime(k): RegimeMoments(float(mean_return[raw]), float(volatility[raw]))
                for k, raw in enumerate(order)
            }
        )
        anomalies = self._channel.inspect(
            moments, refit_ts=refit_ts, raw_state_order=order.tolist()
        )
        hungarian_agrees = (
            None
            if previous is None
            else self._hungarian_check(order, mean_return, volatility, previous)
        )

        mapping = RegimeMapping(
            order=order,
            canonical_params=params.permuted(order),
            moments=moments,
            anomalies=anomalies,
            hungarian_agrees=hungarian_agrees,
        )
        _LOG.info(
            LogEvent.REGIME_MAPPING,
            "HMM states mapped to canonical regimes",
            refit_ts=refit_ts,
            raw_state_order=order.tolist(),
            moments={r.name: [m.mean_return, m.volatility] for r, m in moments.items()},
            n_anomalies=len(anomalies),
            hungarian_agrees=hungarian_agrees,
        )
        return mapping

    def _hungarian_check(
        self,
        order: IntArray,
        mean_return: FloatArray,
        volatility: FloatArray,
        previous: RegimeMapping,
    ) -> bool:
        """Match new raw states to previous canonical regimes by standardized moment distance."""
        prev_mean = np.array([previous.moments[r].mean_return for r in Regime])
        prev_vol = np.array([previous.moments[r].volatility for r in Regime])
        mean_unit = max(float(prev_mean.std()), 1e-12)
        vol_unit = max(float(prev_vol.std()), 1e-12)
        cost = ((mean_return[:, None] - prev_mean[None, :]) / mean_unit) ** 2 + (
            (volatility[:, None] - prev_vol[None, :]) / vol_unit
        ) ** 2
        raw_idx, regime_idx = linear_sum_assignment(cost)
        hungarian_order = np.empty(_N_REGIMES, dtype=np.int64)
        hungarian_order[regime_idx] = raw_idx
        agrees = bool(np.array_equal(hungarian_order, order))
        if not agrees:
            _LOG.warning(
                LogEvent.REGIME_MAPPING_DISAGREEMENT,
                "rank-based labels differ from nearest-previous-regime assignment",
                rank_order=order.tolist(),
                hungarian_order=hungarian_order.tolist(),
            )
        return agrees
