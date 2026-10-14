"""Phase 3 tests: walk-forward HMM engine, online filtering, hysteresis, causality."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from regime_engine.core.config import EngineConfig, TrainingConfig, WindowType
from regime_engine.core.exceptions import (
    ConfigError,
    DataValidationError,
    HMMFitError,
    HMMParameterError,
    LookAheadError,
)
from regime_engine.core.types import Regime
from regime_engine.data.features import FeatureEngine
from regime_engine.data.lookahead import LookAheadSanityChecker
from regime_engine.models import gaussian_hmm
from regime_engine.models.gaussian_hmm import HMMParams, forward, sample
from regime_engine.models.hmm_engine import (
    PROB_COLUMNS,
    HMMEngine,
    RegimeConfirmation,
    rescale_params,
)
from regime_engine.models.regime_mapper import FeatureScaling

from .conftest import ListHandler, OHLCVFactory, make_canonical_truth

FEATURES = ("log_ret", "hl_vol", "vol_vol")
MIN_TRAIN = 500
REFIT_EVERY = 100


def _config(**training: object) -> EngineConfig:
    return EngineConfig.from_dict(
        {
            "training": {"min_train_bars": MIN_TRAIN, "refit_every": REFIT_EVERY, **training},
            "hmm": {"n_restarts": 3, "refit_restarts": 2},
        }
    )


@pytest.fixture(scope="module")
def regime_data() -> tuple[pd.DataFrame, np.ndarray]:
    """1,600 feature bars drawn from a known canonical 4-regime model."""
    X, states = sample(1600, make_canonical_truth(), np.random.default_rng(7))
    index = pd.bdate_range("2010-01-04", periods=len(X), tz="UTC", name="date")
    return pd.DataFrame(X, index=index, columns=list(FEATURES)), states


@pytest.fixture(scope="module")
def engine_run(regime_data: tuple[pd.DataFrame, np.ndarray]) -> tuple[HMMEngine, pd.DataFrame]:
    features, _ = regime_data
    engine = HMMEngine(_config())
    return engine, engine.run(features)


# ============================================================================================
# Configuration
# ============================================================================================


class TestConfig:
    def test_expanding_window_is_default(self) -> None:
        cfg = EngineConfig()
        assert cfg.training.window_type is WindowType.EXPANDING
        assert cfg.regime.confirm_prob == 0.6
        assert cfg.regime.confirm_bars == 2

    def test_rolling_window_constraints(self) -> None:
        TrainingConfig(window_type=WindowType.EXPANDING, min_train_bars=3000, train_window=1000)
        with pytest.raises(ConfigError, match="must not exceed"):
            TrainingConfig(window_type=WindowType.ROLLING, min_train_bars=3000, train_window=1000)

    @pytest.mark.parametrize(
        "raw",
        [
            {"regime": {"confirm_prob": 0.4}},
            {"regime": {"confirm_bars": 0}},
            {"hmm": {"n_restarts": 2, "refit_restarts": 3}},
        ],
    )
    def test_invalid_values_rejected(self, raw: dict[str, object]) -> None:
        with pytest.raises(ConfigError):
            EngineConfig.from_dict(raw)


# ============================================================================================
# Hysteresis
# ============================================================================================


def _p(regime: Regime, prob: float) -> np.ndarray:
    out = np.full(4, (1.0 - prob) / 3.0)
    out[regime] = prob
    return out


class TestRegimeConfirmation:
    def test_first_observation_confirms_immediately(self) -> None:
        confirmation = RegimeConfirmation(0.6, 2)
        assert confirmation.update(_p(Regime.BEAR_LOW_VOL, 0.4)) == (
            Regime.BEAR_LOW_VOL,
            Regime.BEAR_LOW_VOL,
            False,
        )

    def test_single_bar_spike_is_ignored(self) -> None:
        confirmation = RegimeConfirmation(0.6, 2)
        confirmation.update(_p(Regime.BULL_LOW_VOL, 0.9))
        assert confirmation.update(_p(Regime.BEAR_HIGH_VOL, 0.95)) == (
            Regime.BULL_LOW_VOL,
            Regime.BEAR_HIGH_VOL,
            False,
        )
        assert confirmation.update(_p(Regime.BULL_LOW_VOL, 0.9))[0] is Regime.BULL_LOW_VOL
        assert confirmation.update(_p(Regime.BEAR_HIGH_VOL, 0.95))[2] is False

    def test_switches_after_required_consecutive_bars(self) -> None:
        confirmation = RegimeConfirmation(0.6, 3)
        confirmation.update(_p(Regime.BULL_LOW_VOL, 0.9))
        flags = [confirmation.update(_p(Regime.BEAR_LOW_VOL, 0.7)) for _ in range(4)]
        assert [f[0] for f in flags] == [
            Regime.BULL_LOW_VOL,
            Regime.BULL_LOW_VOL,
            Regime.BEAR_LOW_VOL,
            Regime.BEAR_LOW_VOL,
        ]
        assert [f[2] for f in flags] == [False, False, True, False]

    def test_below_threshold_never_switches(self) -> None:
        confirmation = RegimeConfirmation(0.6, 2)
        confirmation.update(_p(Regime.BULL_LOW_VOL, 0.9))
        for _ in range(10):
            confirmed, raw, changed = confirmation.update(_p(Regime.BEAR_LOW_VOL, 0.55))
            assert raw is Regime.BEAR_LOW_VOL
            assert confirmed is Regime.BULL_LOW_VOL
            assert not changed

    def test_streak_resets_on_interruption_or_new_candidate(self) -> None:
        confirmation = RegimeConfirmation(0.6, 2)
        confirmation.update(_p(Regime.BULL_LOW_VOL, 0.9))
        confirmation.update(_p(Regime.BEAR_LOW_VOL, 0.7))
        confirmation.update(_p(Regime.BEAR_HIGH_VOL, 0.7))  # different candidate: streak = 1
        assert confirmation.confirmed is Regime.BULL_LOW_VOL
        assert confirmation.update(_p(Regime.BEAR_HIGH_VOL, 0.7))[2] is True

    def test_single_bar_confirmation(self) -> None:
        confirmation = RegimeConfirmation(0.5, 1)
        confirmation.update(_p(Regime.BULL_LOW_VOL, 0.9))
        assert confirmation.update(_p(Regime.BULL_HIGH_VOL, 0.5)) == (
            Regime.BULL_HIGH_VOL,
            Regime.BULL_HIGH_VOL,
            True,
        )


# ============================================================================================
# Re-expressing a model in a new scaled space
# ============================================================================================


def test_rescaled_params_give_identical_filtered_probabilities() -> None:
    truth = make_canonical_truth()
    z_old, _ = sample(400, truth, np.random.default_rng(3))
    old = FeatureScaling(FEATURES, np.array([0.0004, 0.01, 0.3]), np.array([0.012, 0.004, 0.07]))
    new = FeatureScaling(FEATURES, np.array([0.0002, 0.012, 0.25]), np.array([0.010, 0.005, 0.09]))
    original = z_old * old.scale + old.location
    z_new = (original - new.location) / new.scale
    moved = rescale_params(truth, old, new)
    np.testing.assert_allclose(
        forward(z_new, moved).filtered, forward(z_old, truth).filtered, rtol=1e-10, atol=1e-13
    )


# ============================================================================================
# Engine behaviour
# ============================================================================================


class TestHMMEngine:
    def test_not_ready_before_min_train_bars(
        self, regime_data: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        features, _ = regime_data
        engine = HMMEngine(_config())
        for ts, row in zip(features.index[: MIN_TRAIN - 1], features.to_numpy(), strict=False):
            assert engine.update(ts, row) is None
        assert not engine.is_ready
        with pytest.raises(HMMParameterError):
            engine.one_step_forecast()
        state = engine.update(features.index[MIN_TRAIN - 1], features.iloc[MIN_TRAIN - 1])
        assert state is not None
        assert state.refit
        assert state.model_version == 1

    def test_refit_schedule_and_expanding_window(
        self,
        engine_run: tuple[HMMEngine, pd.DataFrame],
        regime_data: tuple[pd.DataFrame, np.ndarray],
    ) -> None:
        engine, states = engine_run
        features, _ = regime_data
        assert states.index[0] == features.index[MIN_TRAIN - 1]
        expected_positions = list(range(MIN_TRAIN - 1, len(features), REFIT_EVERY))
        refit_positions = [features.index.get_loc(ts) for ts in states.index[states["refit"]]]
        assert refit_positions == expected_positions
        assert [r.n_train for r in engine.history] == [p + 1 for p in expected_positions]
        assert [r.version for r in engine.history] == list(range(1, len(expected_positions) + 1))
        assert [r.warm_started for r in engine.history] == [False] + [True] * (
            len(expected_positions) - 1
        )
        assert states["model_version"].is_monotonic_increasing

    def test_rolling_window_caps_training_size(
        self, regime_data: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        features, _ = regime_data
        engine = HMMEngine(_config(window_type="rolling", train_window=600))
        engine.run(features.iloc[:1000])
        # Fits at bars 500, 600, ..., 1000; the window is capped at 600 bars.
        assert [r.n_train for r in engine.history] == [500, 600, 600, 600, 600, 600]

    def test_online_filter_matches_batch_forward_per_model(
        self,
        engine_run: tuple[HMMEngine, pd.DataFrame],
        regime_data: tuple[pd.DataFrame, np.ndarray],
    ) -> None:
        """Between refits, bar-by-bar filtering equals a batch forward pass with that model."""
        engine, states = engine_run
        features, _ = regime_data
        raw = features.to_numpy()
        for record in engine.history:
            rows = states[states["model_version"] == record.version]
            last = features.index.get_loc(rows.index[-1])
            scaled = (raw[: last + 1] - record.scaling.location) / record.scaling.scale
            batch = forward(scaled, record.mapping.canonical_params).filtered
            first = features.index.get_loc(rows.index[0])
            np.testing.assert_allclose(
                rows[list(PROB_COLUMNS)].to_numpy(), batch[first:], rtol=1e-9, atol=1e-12
            )

    def test_tracks_true_regimes(
        self,
        engine_run: tuple[HMMEngine, pd.DataFrame],
        regime_data: tuple[pd.DataFrame, np.ndarray],
    ) -> None:
        _, states = engine_run
        features, true_states = regime_data
        truth = pd.Series(true_states, index=features.index).loc[states.index]
        assert (states["raw_argmax"] == truth).mean() > 0.85
        assert (states["regime"] == truth).mean() > 0.8
        assert not states["mapping_ambiguous"].any()

    def test_hysteresis_reduces_switching_and_changes_are_logged(
        self, regime_data: tuple[pd.DataFrame, np.ndarray], captured: ListHandler
    ) -> None:
        features, _ = regime_data
        states = HMMEngine(_config()).run(features)
        raw_switches = int((states["raw_argmax"].diff().fillna(0) != 0).sum())
        confirmed_switches = int(states["changed"].sum())
        assert 0 < confirmed_switches < raw_switches
        np.testing.assert_array_equal(
            states["changed"].to_numpy()[1:], np.diff(states["regime"].to_numpy()) != 0
        )
        assert len(captured.events("REGIME_CHANGE")) == confirmed_switches
        assert len(captured.events("ENGINE_READY")) == 1
        assert len(captured.events("HMM_REFIT")) == int(states["refit"].sum())

    def test_probabilities_and_forecast(self, engine_run: tuple[HMMEngine, pd.DataFrame]) -> None:
        engine, states = engine_run
        np.testing.assert_allclose(states[list(PROB_COLUMNS)].sum(axis=1), 1.0, atol=1e-12)
        last = states[list(PROB_COLUMNS)].iloc[-1].to_numpy()
        forecast = engine.one_step_forecast()
        np.testing.assert_allclose(forecast, last @ engine.canonical_params.transmat)
        assert forecast.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(states["confidence"], states[list(PROB_COLUMNS)].max(axis=1))

    def test_deterministic(self, regime_data: tuple[pd.DataFrame, np.ndarray]) -> None:
        features, _ = regime_data
        a = HMMEngine(_config()).run(features.iloc[:900])
        b = HMMEngine(_config()).run(features.iloc[:900])
        pd.testing.assert_frame_equal(a, b, check_exact=True)

    def test_rejects_bad_input(self, regime_data: tuple[pd.DataFrame, np.ndarray]) -> None:
        features, _ = regime_data
        engine = HMMEngine(_config())
        engine.update(features.index[1], features.iloc[1])
        with pytest.raises(LookAheadError):
            engine.update(features.index[0], features.iloc[0])
        with pytest.raises(LookAheadError):
            engine.update(features.index[1], features.iloc[1])
        with pytest.raises(DataValidationError, match="non-finite"):
            engine.update(features.index[2], [0.1, np.nan, 0.2])
        with pytest.raises(DataValidationError, match="expected 3"):
            engine.update(features.index[2], [0.1, 0.2])
        with pytest.raises(DataValidationError, match="columns"):
            HMMEngine(_config()).run(features[["hl_vol", "log_ret", "vol_vol"]])


# ============================================================================================
# Failure isolation
# ============================================================================================


def _fail_on_calls(monkeypatch: pytest.MonkeyPatch, failing_calls: set[int]) -> Callable[[], int]:
    original = gaussian_hmm.GaussianHMM.fit
    calls = {"n": 0}

    def flaky_fit(
        self: gaussian_hmm.GaussianHMM, X: np.ndarray, *, init: HMMParams | None = None
    ) -> gaussian_hmm.FitResult:
        calls["n"] += 1
        if calls["n"] in failing_calls:
            raise HMMFitError("injected failure")
        return original(self, X, init=init)

    monkeypatch.setattr(gaussian_hmm.GaussianHMM, "fit", flaky_fit)
    return lambda: calls["n"]


class TestRefitFailures:
    def test_failed_refit_keeps_previous_model(
        self,
        regime_data: tuple[pd.DataFrame, np.ndarray],
        monkeypatch: pytest.MonkeyPatch,
        captured: ListHandler,
    ) -> None:
        features, _ = regime_data
        _fail_on_calls(monkeypatch, {2})
        states = HMMEngine(_config()).run(features.iloc[:800])
        # Attempts at bars 500 (ok), 600 (injected failure), 700 (ok), 800 (ok).
        refit_rows = states.loc[states["refit"]].index.map(features.index.get_loc).tolist()
        assert refit_rows == [499, 699, 799]
        version_at = states["model_version"]
        assert (version_at.iloc[: 699 - 499] == 1).all()
        assert version_at.iloc[699 - 499] == 2
        assert version_at.iloc[-1] == 3
        assert len(captured.events("HMM_REFIT_FAILED")) == 1

    def test_failed_first_fit_retries_later(
        self, regime_data: tuple[pd.DataFrame, np.ndarray], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        features, _ = regime_data
        _fail_on_calls(monkeypatch, {1})
        engine = HMMEngine(_config())
        states = engine.run(features.iloc[:700])
        assert states.index[0] == features.index[MIN_TRAIN - 1 + REFIT_EVERY]
        assert engine.history[0].n_train == MIN_TRAIN + REFIT_EVERY
        assert not engine.history[0].warm_started


# ============================================================================================
# End-to-end causality: OHLCV -> features -> walk-forward HMM -> regime states
# ============================================================================================


def test_full_regime_pipeline_has_no_lookahead(ohlcv_factory: OHLCVFactory) -> None:
    cfg = EngineConfig.from_dict(
        {
            "training": {"min_train_bars": 300, "refit_every": 150},
            "hmm": {"n_restarts": 2, "refit_restarts": 1},
        }
    )
    feature_engine = FeatureEngine(cfg.features)

    def pipeline(ohlcv: pd.DataFrame) -> pd.DataFrame:
        return HMMEngine(cfg).run(feature_engine(ohlcv))

    data = ohlcv_factory(900, seed=23)
    first_ready = feature_engine.warmup_bars + 300 - 1  # OHLCV row of the first fit
    checker = LookAheadSanityChecker(pipeline)
    for cutoff_pos in (first_ready + 149, first_ready + 150, first_ready + 151, 820):
        report = checker.assert_no_lookahead(data, data.index[cutoff_pos])
        assert report.passed
