"""Walk-forward regime engine: scheduled refits, online filtering, hysteresis.

Each bar, ``HMMEngine.update(ts, features)`` returns the engine's *causal* view of the regime
at the close of that bar. Every quantity in it is a function of feature bars <= ts:

* **Refit schedule.** The first fit happens once ``min_train_bars`` feature bars have been
  seen. Later refits happen every ``refit_every`` bars. EXPANDING windows (default) train on
  all history so far. ROLLING windows train on the last ``train_window`` bars.
* **Scaling.** Each refit fits a z-score (or min-max) scaler on exactly its training block.
  Bars between refits are scaled with that frozen transform, never with statistics that
  include them.
* **Labels.** Each fitted model goes through ``RegimeMapper``, so state ``k`` always means
  ``Regime(k)``. The previous mapping is passed in for the Hungarian cross-check.
* **Warm starts.** The previous model is re-expressed in the new scaled space
  (``rescale_params``) and seeds EM, alongside fresh k-means++ restarts.
* **Filtering.** On a refit bar, the filtered distribution is recomputed by running the new
  model forward over its training block. On other bars it advances one ``filter_step``.
* **Hysteresis.** ``RegimeConfirmation`` decides the *confirmed* regime.
* **Failure isolation.** If a refit fails (EM rejected, mapping error), the previous model
  stays in force and the failure is logged. The next attempt is ``refit_every`` bars later.

Random seeds are derived from the attempt number, so a run is deterministic and a refit can
never depend on how many bars come after it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
import pandas as pd

from regime_engine.core.config import EngineConfig, RegimeConfig, WindowType
from regime_engine.core.exceptions import (
    DataValidationError,
    HMMFitError,
    HMMNumericalError,
    HMMParameterError,
    InsufficientDataError,
    LookAheadError,
    RegimeMappingError,
)
from regime_engine.core.types import Regime, RegimeMoments, RegimeState
from regime_engine.data.features import FeatureEngine, RollingWindowScaler
from regime_engine.models.gaussian_hmm import (
    FitResult,
    FloatArray,
    GaussianHMM,
    HMMParams,
    filter_step,
    forward,
    predict_next,
)
from regime_engine.models.regime_mapper import FeatureScaling, RegimeMapper, RegimeMapping
from regime_engine.utils.logging import LogEvent, get_logger, log_context

_LOG = get_logger("models.hmm_engine")

PROB_COLUMNS: tuple[str, ...] = tuple(f"p_{r.name}" for r in Regime)


# --------------------------------------------------------------------------------------------
# Hysteresis
# --------------------------------------------------------------------------------------------


class RegimeConfirmation:
    """Confirmed-regime state machine.

    The first observation confirms its argmax immediately. After that, the confirmed regime
    switches to R' only once P(R') >= ``confirm_prob`` for ``confirm_bars`` consecutive bars.
    A bar that breaks the streak (a different candidate, or the probability dipping below
    the threshold) resets it.
    """

    def __init__(self, confirm_prob: float = 0.6, confirm_bars: int = 2) -> None:
        if not 0.5 <= confirm_prob <= 1.0:
            raise ValueError("confirm_prob must be in [0.5, 1]")
        if confirm_bars < 1:
            raise ValueError("confirm_bars must be >= 1")
        self._threshold = confirm_prob
        self._required = confirm_bars
        self._confirmed: Regime | None = None
        self._candidate: Regime | None = None
        self._streak = 0

    @classmethod
    def from_config(cls, config: RegimeConfig) -> RegimeConfirmation:
        return cls(config.confirm_prob, config.confirm_bars)

    @property
    def confirmed(self) -> Regime | None:
        return self._confirmed

    def update(self, probs: npt.ArrayLike) -> tuple[Regime, Regime, bool]:
        """Return (confirmed regime, raw argmax, changed)."""
        p = np.asarray(probs, dtype=np.float64)
        raw = Regime(int(p.argmax()))
        if self._confirmed is None:
            self._confirmed = raw
            return raw, raw, False
        if raw is not self._confirmed and p[raw] >= self._threshold:
            if raw is self._candidate:
                self._streak += 1
            else:
                self._candidate, self._streak = raw, 1
            if self._streak >= self._required:
                self._confirmed = raw
                self._candidate, self._streak = None, 0
                return raw, raw, True
        else:
            self._candidate, self._streak = None, 0
        return self._confirmed, raw, False


# --------------------------------------------------------------------------------------------
# Re-expressing a model in a new scaled space
# --------------------------------------------------------------------------------------------


def rescale_params(params: HMMParams, old: FeatureScaling, new: FeatureScaling) -> HMMParams:
    """Express a model fitted in ``old`` scaled space in ``new`` scaled space.

    With original units x = z_old * s_old + l_old and z_new = (x - l_new) / s_new, we get
    z_new = r * z_old + c with r = s_old / s_new and c = (l_old - l_new) / s_new. So
    mu_new = r * mu_old + c and Sigma_new = diag(r) Sigma_old diag(r). Filtered state
    probabilities are unchanged, because the change of variables adds the same Jacobian
    constant to every state's log-density.
    """
    if old.names != new.names:
        raise RegimeMappingError(f"feature names differ: {old.names} vs {new.names}")
    ratio = old.scale / new.scale
    shift = (old.location - new.location) / new.scale
    covars = params.covars * ratio[None, :, None] * ratio[None, None, :]
    return HMMParams(
        startprob=params.startprob,
        transmat=params.transmat,
        means=params.means * ratio + shift,
        covars=0.5 * (covars + covars.transpose(0, 2, 1)),
    )


# --------------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RefitRecord:
    ts: pd.Timestamp
    version: int
    n_train: int
    fit: FitResult
    mapping: RegimeMapping
    scaling: FeatureScaling
    warm_started: bool


_FIT_ERRORS = (
    HMMFitError,
    HMMNumericalError,
    HMMParameterError,
    InsufficientDataError,
    RegimeMappingError,
)


class HMMEngine:
    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        feature_names: Sequence[str] = FeatureEngine.FEATURE_COLUMNS,
    ) -> None:
        self._cfg = config or EngineConfig()
        self._names = tuple(feature_names)
        self._mapper = RegimeMapper(self._cfg.mapping)
        self._confirmation = RegimeConfirmation.from_config(self._cfg.regime)

        self._timestamps: list[pd.Timestamp] = []
        self._observations: list[FloatArray] = []
        self._last_attempt_n: int | None = None
        self._attempts = 0

        self._params: HMMParams | None = None
        self._scaling: FeatureScaling | None = None
        self._mapping: RegimeMapping | None = None
        self._alpha: FloatArray | None = None
        self._refit_loglik = 0.0
        self._history: list[RefitRecord] = []

    # ---- read-only views -------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._params is not None

    @property
    def model_version(self) -> int:
        return len(self._history)

    @property
    def n_bars_seen(self) -> int:
        return len(self._observations)

    @property
    def history(self) -> tuple[RefitRecord, ...]:
        return tuple(self._history)

    @property
    def canonical_params(self) -> HMMParams:
        if self._params is None:
            raise HMMParameterError("engine has not completed its first fit")
        return self._params

    @property
    def scaling(self) -> FeatureScaling:
        if self._scaling is None:
            raise HMMParameterError("engine has not completed its first fit")
        return self._scaling

    @property
    def mapping(self) -> RegimeMapping:
        if self._mapping is None:
            raise HMMParameterError("engine has not completed its first fit")
        return self._mapping

    def one_step_forecast(self) -> FloatArray:
        """P(S_{t+1} | x_1..t) in canonical regime order."""
        if self._alpha is None or self._params is None:
            raise HMMParameterError("engine has not completed its first fit")
        return predict_next(self._alpha, self._params)

    # ---- main loop ---------------------------------------------------------------------------

    def update(self, ts: pd.Timestamp | str, features: npt.ArrayLike) -> RegimeState | None:
        """Consume the feature vector for bar ``ts``. Returns ``None`` until the first fit."""
        stamp = pd.Timestamp(ts)
        if self._timestamps and stamp <= self._timestamps[-1]:
            raise LookAheadError(
                f"bar {stamp} does not follow {self._timestamps[-1]}: bars must arrive in "
                "strictly increasing time order"
            )
        vector = np.array(features, dtype=np.float64).reshape(-1)
        if vector.shape != (len(self._names),):
            raise DataValidationError(
                f"expected {len(self._names)} features {self._names}, got shape {vector.shape}"
            )
        if not np.isfinite(vector).all():
            raise DataValidationError(f"non-finite feature vector at {stamp}: {vector.tolist()}")

        self._timestamps.append(stamp)
        self._observations.append(vector)

        refitted = self._refit_due() and self._refit(stamp)
        if self._params is None or self._scaling is None or self._mapping is None:
            return None

        if refitted:
            assert self._alpha is not None
            alpha, loglik = self._alpha, self._refit_loglik
        else:
            scaled = (vector - self._scaling.location) / self._scaling.scale
            alpha, loglik = filter_step(self._alpha, scaled, self._params)
            self._alpha = alpha

        confirmed, raw, changed = self._confirmation.update(alpha)
        if changed:
            _LOG.info(
                LogEvent.REGIME_CHANGE,
                f"confirmed regime -> {confirmed.name}",
                ts=stamp,
                to=confirmed.name,
                probs={r.name: float(alpha[r]) for r in Regime},
                model_version=self.model_version,
            )
        return RegimeState(
            ts=stamp,
            probs=alpha,
            regime=confirmed,
            raw_argmax=raw,
            changed=changed,
            refit=refitted,
            model_version=self.model_version,
            mapping_ambiguous=self._mapping.ambiguous,
            predictive_loglik=float(loglik),
        )

    def run(self, features: pd.DataFrame) -> pd.DataFrame:
        """Feed every row of ``features`` through ``update`` and tabulate the regime states."""
        if tuple(str(c) for c in features.columns) != self._names:
            raise DataValidationError(
                f"feature columns {list(features.columns)} != expected {list(self._names)}"
            )
        states: list[RegimeState] = []
        for ts, row in zip(features.index, features.to_numpy(dtype=np.float64), strict=True):
            state = self.update(ts, row)
            if state is not None:
                states.append(state)
        return states_to_frame(states)

    # ---- refitting -----------------------------------------------------------------------

    def _refit_due(self) -> bool:
        n = len(self._observations)
        if self._last_attempt_n is None:
            return n >= self._cfg.training.min_train_bars
        return n - self._last_attempt_n >= self._cfg.training.refit_every

    def _refit(self, ts: pd.Timestamp) -> bool:
        training = self._cfg.training
        n = len(self._observations)
        self._last_attempt_n = n
        attempt = self._attempts
        self._attempts += 1

        start = (
            0 if training.window_type is WindowType.EXPANDING else max(0, n - training.train_window)
        )
        block = pd.DataFrame(
            np.vstack(self._observations[start:]),
            index=pd.DatetimeIndex(self._timestamps[start:]),
            columns=list(self._names),
        )
        scaling_cfg = self._cfg.scaling
        scaler = RollingWindowScaler(
            len(block),
            min_periods=len(block),
            method=scaling_cfg.method,
            ddof=scaling_cfg.ddof,
            epsilon=scaling_cfg.epsilon,
        )
        previous = (
            (self._params, self._scaling)
            if self._params is not None and self._scaling is not None
            else None
        )
        warm = previous is not None
        hmm_cfg = self._cfg.hmm
        if warm:
            hmm_cfg = dataclasses.replace(hmm_cfg, n_restarts=hmm_cfg.refit_restarts)
        model = GaussianHMM.from_config(hmm_cfg, seed=self._cfg.run.seed + attempt)

        with log_context(refit_ts=ts, attempt=attempt):
            try:
                scaled = scaler.fit_transform(block).to_numpy(dtype=np.float64)
                scaling = FeatureScaling.from_scaler_params(scaler.params)
                init = None if previous is None else rescale_params(*previous, scaling)
                fit = model.fit(scaled, init=init)
                mapping = self._mapper.map(
                    fit.params, scaling, previous=self._mapping, refit_ts=ts.to_pydatetime()
                )
                params = mapping.canonical_params
                filtered = forward(scaled, params)
            except _FIT_ERRORS as exc:
                _LOG.warning(
                    LogEvent.HMM_REFIT_FAILED,
                    "refit failed; keeping the previous model"
                    if self._params is not None
                    else "first fit failed; engine not ready",
                    error=type(exc).__name__,
                    detail=str(exc),
                    n_train=len(block),
                    model_version=self.model_version,
                )
                return False

        self._params, self._scaling, self._mapping = params, scaling, mapping
        self._alpha = filtered.filtered[-1].copy()
        self._refit_loglik = float(filtered.log_scale[-1])
        self._history.append(
            RefitRecord(
                ts=ts,
                version=len(self._history) + 1,
                n_train=len(block),
                fit=fit,
                mapping=mapping,
                scaling=scaling,
                warm_started=warm,
            )
        )
        _LOG.info(
            LogEvent.HMM_REFIT,
            "model re-estimated",
            ts=ts,
            model_version=self.model_version,
            n_train=len(block),
            train_start=block.index[0],
            log_likelihood=fit.log_likelihood,
            bic=fit.bic,
            converged=fit.converged,
            n_iter=fit.n_iter,
            warm_started=warm,
            ambiguous=mapping.ambiguous,
            durations={
                r.name: float(d) for r, d in zip(Regime, params.expected_durations(), strict=True)
            },
        )
        if self.model_version == 1:
            _LOG.info(LogEvent.ENGINE_READY, "first model fitted; regime states available", ts=ts)
        return True


def states_to_frame(states: Sequence[RegimeState]) -> pd.DataFrame:
    """Tabulate regime states, one row per bar, indexed by timestamp."""
    index = pd.DatetimeIndex([s.ts for s in states], name="date")
    probs = np.vstack([s.probs for s in states]) if states else np.empty((0, len(Regime)))
    frame = pd.DataFrame(probs, index=index, columns=list(PROB_COLUMNS))
    frame["regime"] = np.array([int(s.regime) for s in states], dtype=np.int64)
    frame["raw_argmax"] = np.array([int(s.raw_argmax) for s in states], dtype=np.int64)
    frame["confidence"] = np.array([s.confidence for s in states], dtype=np.float64)
    frame["changed"] = np.array([s.changed for s in states], dtype=bool)
    frame["refit"] = np.array([s.refit for s in states], dtype=bool)
    frame["model_version"] = np.array([s.model_version for s in states], dtype=np.int64)
    frame["mapping_ambiguous"] = np.array([s.mapping_ambiguous for s in states], dtype=bool)
    frame["predictive_loglik"] = np.array([s.predictive_loglik for s in states], dtype=np.float64)
    return frame


# --------------------------------------------------------------------------------------------
# Precomputed regime path
# --------------------------------------------------------------------------------------------

#: Config sections that determine the regime path. Changing anything else (strategy, sizing,
#: risk, execution) leaves the path unchanged, so it can be computed once and replayed.
REGIME_SECTIONS: tuple[str, ...] = ("features", "scaling", "training", "hmm", "mapping", "regime")


def regime_fingerprint(config: EngineConfig) -> str:
    import hashlib
    import json

    full = config.to_dict()
    relevant = {name: full[name] for name in REGIME_SECTIONS} | {"seed": full["run"]["seed"]}
    blob = json.dumps(relevant, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, eq=False)
class TrackPoint:
    state: RegimeState
    forecast: FloatArray  # P(S_{t+1} | x_1..t), canonical order
    moments: Mapping[Regime, RegimeMoments]  # per-regime moments of the model in force at t


@dataclass(frozen=True, eq=False)
class RegimeTrack:
    """The engine's causal output for every feature bar, replayable across backtests.

    Building the track runs ``HMMEngine.update`` bar by bar, exactly as a live loop would.
    Each point depends only on features up to its own timestamp, so replaying it inside a
    backtest is equivalent to running the engine inline.
    """

    fingerprint: str
    points: Mapping[pd.Timestamp, TrackPoint]
    history: tuple[RefitRecord, ...]

    @classmethod
    def build(cls, config: EngineConfig, features: pd.DataFrame) -> RegimeTrack:
        engine = HMMEngine(config, feature_names=tuple(str(c) for c in features.columns))
        points: dict[pd.Timestamp, TrackPoint] = {}
        for ts, row in zip(features.index, features.to_numpy(dtype=np.float64), strict=True):
            state = engine.update(ts, row)
            if state is not None:
                points[ts] = TrackPoint(state, engine.one_step_forecast(), engine.mapping.moments)
        return cls(regime_fingerprint(config), MappingProxyType(points), engine.history)

    @property
    def final_params(self) -> HMMParams | None:
        return self.history[-1].mapping.canonical_params if self.history else None

    @property
    def final_scaling(self) -> FeatureScaling | None:
        return self.history[-1].scaling if self.history else None

    def transition_matrices(self) -> FloatArray:
        """Canonical transition matrices of every refit, shape (n_refits, 4, 4)."""
        if not self.history:
            return np.empty((0, len(Regime), len(Regime)))
        return np.stack([r.mapping.canonical_params.transmat for r in self.history])

    def states_frame(self) -> pd.DataFrame:
        return states_to_frame([p.state for p in self.points.values()])
