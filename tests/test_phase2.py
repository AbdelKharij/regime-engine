"""Phase 2 tests: Gaussian HMM (math, hmmlearn parity, recovery) and canonical regime mapping."""

from __future__ import annotations

import dataclasses
import itertools
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from regime_engine.core.config import EngineConfig, HMMConfig, MappingConfig, VolatilitySource
from regime_engine.core.exceptions import (
    ConfigError,
    HMMFitError,
    HMMParameterError,
    InsufficientDataError,
    RegimeMappingError,
)
from regime_engine.core.types import Regime
from regime_engine.data.features import FeatureEngine, RollingWindowScaler
from regime_engine.models.gaussian_hmm import (
    GaussianHMM,
    HMMParams,
    filter_step,
    forward,
    log_emission,
    predict_next,
    sample,
    score,
    select_n_states,
    smooth,
    viterbi,
)
from regime_engine.models.regime_mapper import FeatureScaling, RegimeMapper
from regime_engine.utils.logging import OrderingAnomalyKind

from .conftest import ListHandler, OHLCVFactory, make_canonical_truth

FEATURES = ("log_ret", "hl_vol", "vol_vol")


@pytest.fixture(scope="module")
def truth() -> HMMParams:
    return make_canonical_truth()


@pytest.fixture(scope="module")
def sampled(truth: HMMParams) -> tuple[np.ndarray, np.ndarray]:
    return sample(5000, truth, np.random.default_rng(2024))


def _small_model() -> HMMParams:
    return HMMParams(
        startprob=[0.5, 0.3, 0.2],
        transmat=[[0.8, 0.15, 0.05], [0.1, 0.7, 0.2], [0.25, 0.25, 0.5]],
        means=[[0.0, 1.0], [2.0, -1.0], [-1.5, 0.5]],
        covars=[
            [[1.0, 0.3], [0.3, 0.8]],
            [[0.5, -0.1], [-0.1, 0.6]],
            [[1.5, 0.0], [0.0, 0.4]],
        ],
    )


# ============================================================================================
# Parameters
# ============================================================================================


class TestHMMParams:
    @pytest.mark.parametrize(
        ("field", "value", "fragment"),
        [
            ("startprob", [0.5, 0.6, -0.1], "probability vector"),
            ("transmat", [[0.9, 0.2, 0.0], [0.1, 0.7, 0.2], [0.25, 0.25, 0.5]], "rows"),
            ("means", [[0.0, 1.0], [2.0, -1.0]], "shape"),
            ("covars", [[[1.0, 0.0], [0.0, -1.0]]] * 3, "positive definite"),
            ("covars", [[[1.0, 0.5], [0.0, 1.0]]] * 3, "symmetric"),
            ("means", [[0.0, np.nan], [2.0, -1.0], [-1.5, 0.5]], "non-finite"),
        ],
    )
    def test_invalid_parameters_rejected(self, field: str, value: object, fragment: str) -> None:
        good = _small_model()
        kwargs = {f.name: getattr(good, f.name) for f in dataclasses.fields(good)}
        kwargs[field] = value
        with pytest.raises(HMMParameterError, match=fragment):
            HMMParams(**kwargs)

    def test_arrays_are_immutable_copies(self) -> None:
        source = np.array([0.5, 0.3, 0.2])
        model = dataclasses.replace(_small_model(), startprob=source)
        source[0] = 0.0
        assert model.startprob[0] == 0.5
        with pytest.raises(ValueError, match="read-only"):
            model.means[0, 0] = 1.0

    def test_permutation_preserves_likelihood(self, truth: HMMParams) -> None:
        X, _ = sample(300, truth, np.random.default_rng(1))
        reordered = truth.permuted([2, 0, 3, 1])
        assert score(X, reordered) == pytest.approx(score(X, truth), rel=1e-12)
        assert reordered.transmat[0, 1] == truth.transmat[2, 0]
        restored = reordered.permuted(np.argsort([2, 0, 3, 1]))
        assert restored.allclose(truth, rtol=0, atol=0)
        with pytest.raises(HMMParameterError):
            truth.permuted([0, 0, 1, 2])

    def test_stationary_distribution_and_durations(self, truth: HMMParams) -> None:
        pi = truth.stationary_distribution()
        np.testing.assert_allclose(pi @ truth.transmat, pi, atol=1e-12)
        assert pi.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(
            truth.expected_durations(), 1.0 / (1.0 - np.diag(truth.transmat))
        )

    def test_free_parameter_count(self, truth: HMMParams) -> None:
        # (K-1) + K(K-1) + K*d + K*d(d+1)/2 = 3 + 12 + 12 + 24
        assert truth.n_free_parameters == 51


# ============================================================================================
# Inference against first principles
# ============================================================================================


def _brute_force(
    X: np.ndarray, params: HMMParams
) -> tuple[float, np.ndarray, tuple[int, ...], float]:
    """Enumerate all K^T paths: (log-likelihood, smoothed posteriors, MAP path, MAP log-prob)."""
    k, n = params.n_states, len(X)
    log_b = np.column_stack(
        [multivariate_normal(params.means[j], params.covars[j]).logpdf(X) for j in range(k)]
    )
    paths = list(itertools.product(range(k), repeat=n))
    log_joint = np.empty(len(paths))
    for i, path in enumerate(paths):
        lp = math.log(params.startprob[path[0]]) + log_b[0, path[0]]
        for t in range(1, n):
            lp += math.log(params.transmat[path[t - 1], path[t]]) + log_b[t, path[t]]
        log_joint[i] = lp
    log_lik = float(logsumexp(log_joint))
    weights = np.exp(log_joint - log_lik)
    posterior = np.zeros((n, k))
    for w, path in zip(weights, paths, strict=True):
        posterior[np.arange(n), path] += w
    best = int(log_joint.argmax())
    return log_lik, posterior, paths[best], float(log_joint[best])


class TestInference:
    def test_log_emission_matches_scipy(self, truth: HMMParams) -> None:
        X, _ = sample(50, truth, np.random.default_rng(3))
        expected = np.column_stack(
            [multivariate_normal(truth.means[k], truth.covars[k]).logpdf(X) for k in range(4)]
        )
        np.testing.assert_allclose(log_emission(X, truth), expected, rtol=1e-12)

    def test_forward_backward_viterbi_match_path_enumeration(self) -> None:
        params = _small_model()
        X, _ = sample(6, params, np.random.default_rng(5))
        log_lik, posterior, map_path, map_logprob = _brute_force(X, params)
        assert score(X, params) == pytest.approx(log_lik, rel=1e-12)
        np.testing.assert_allclose(smooth(X, params), posterior, atol=1e-12)
        path, logprob = viterbi(X, params)
        assert tuple(path.tolist()) == map_path
        assert logprob == pytest.approx(map_logprob, rel=1e-12)

    def test_filter_step_reproduces_batch_forward(self, truth: HMMParams) -> None:
        X, _ = sample(400, truth, np.random.default_rng(7))
        batch = forward(X, truth)
        alpha: np.ndarray | None = None
        increments = []
        for t in range(len(X)):
            alpha, inc = filter_step(alpha, X[t], truth)
            np.testing.assert_allclose(alpha, batch.filtered[t], rtol=1e-12, atol=1e-15)
            increments.append(inc)
        assert sum(increments) == pytest.approx(batch.log_likelihood, rel=1e-12)

    def test_filtered_probabilities_are_causal(self, truth: HMMParams) -> None:
        """Filtered P(S_t | x_1..t) on a prefix equals the full-sample value; smoothed does not."""
        X, _ = sample(1000, truth, np.random.default_rng(9))
        full_filtered = forward(X, truth).filtered
        full_smoothed = smooth(X, truth)
        for k in (50, 333, 999):
            prefix_filtered = forward(X[:k], truth).filtered
            np.testing.assert_allclose(prefix_filtered, full_filtered[:k], rtol=1e-12, atol=1e-15)
        prefix_smoothed = smooth(X[:500], truth)
        assert np.abs(prefix_smoothed - full_smoothed[:500]).max() > 1e-3  # uses the future

    def test_last_filtered_equals_last_smoothed(self, truth: HMMParams) -> None:
        X, _ = sample(300, truth, np.random.default_rng(11))
        np.testing.assert_allclose(forward(X, truth).filtered[-1], smooth(X, truth)[-1], atol=1e-12)

    def test_probabilities_are_normalised(self, truth: HMMParams) -> None:
        X, _ = sample(300, truth, np.random.default_rng(12))
        filtered = forward(X, truth).filtered
        np.testing.assert_allclose(filtered.sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(smooth(X, truth).sum(axis=1), 1.0, atol=1e-12)
        nxt = predict_next(filtered[-1], truth)
        np.testing.assert_allclose(nxt, filtered[-1] @ truth.transmat)
        assert nxt.sum() == pytest.approx(1.0)

    def test_sampler_matches_transition_matrix(self, truth: HMMParams) -> None:
        X, states = sample(40_000, truth, np.random.default_rng(13))
        assert X.shape == (40_000, 3)
        counts = np.zeros((4, 4))
        np.add.at(counts, (states[:-1], states[1:]), 1.0)
        empirical = counts / counts.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(empirical, truth.transmat, atol=0.015)
        for k in range(4):
            np.testing.assert_allclose(X[states == k].mean(axis=0), truth.means[k], atol=0.1)


# ============================================================================================
# Parity with hmmlearn (independent reference implementation)
# ============================================================================================


hmmlearn_hmm = pytest.importorskip("hmmlearn.hmm")


def _hmmlearn_model(params: HMMParams, n_iter: int) -> object:
    model = hmmlearn_hmm.GaussianHMM(
        n_components=params.n_states,
        covariance_type="full",
        n_iter=n_iter,
        tol=1e-300,
        init_params="",
        params="stmc",
        covars_prior=0.0,
        covars_weight=0.0,
    )
    model.startprob_ = np.array(params.startprob)
    model.transmat_ = np.array(params.transmat)
    model.means_ = np.array(params.means)
    model.covars_ = np.array(params.covars)
    return model


class TestHmmlearnParity:
    @pytest.fixture
    def data(self, truth: HMMParams) -> np.ndarray:
        X, _ = sample(800, truth, np.random.default_rng(21))
        return X

    @pytest.fixture
    def perturbed(self, truth: HMMParams) -> HMMParams:
        """A deliberately wrong starting point so EM has work to do."""
        return HMMParams(
            startprob=np.full(4, 0.25),
            transmat=np.full((4, 4), 0.1) + 0.6 * np.eye(4),
            means=truth.means + 0.3,
            covars=truth.covars * 1.5,
        )

    def test_inference_matches(self, truth: HMMParams, data: np.ndarray) -> None:
        reference = _hmmlearn_model(truth, n_iter=1)
        assert score(data, truth) == pytest.approx(reference.score(data), rel=1e-10)  # type: ignore[attr-defined]
        np.testing.assert_allclose(
            smooth(data, truth),
            reference.predict_proba(data),  # type: ignore[attr-defined]
            atol=1e-10,
        )
        ref_logprob, ref_path = reference.decode(data, algorithm="viterbi")  # type: ignore[attr-defined]
        path, logprob = viterbi(data, truth)
        np.testing.assert_array_equal(path, ref_path)
        assert logprob == pytest.approx(ref_logprob, rel=1e-10)

    @pytest.mark.parametrize("n_iter", [1, 25])
    def test_baum_welch_updates_match(
        self, data: np.ndarray, perturbed: HMMParams, n_iter: int
    ) -> None:
        reference = _hmmlearn_model(perturbed, n_iter=n_iter)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference.fit(data)  # type: ignore[attr-defined]
        ours = GaussianHMM(4, covariance_ridge=0.0, max_iter=n_iter, tol=1e-300, n_restarts=1).fit(
            data, init=perturbed
        )
        # Both implementations stop early once the improvement is exactly zero.
        assert ours.n_iter <= n_iter
        np.testing.assert_allclose(ours.params.startprob, reference.startprob_, atol=1e-8)  # type: ignore[attr-defined]
        np.testing.assert_allclose(ours.params.transmat, reference.transmat_, atol=1e-8)  # type: ignore[attr-defined]
        np.testing.assert_allclose(ours.params.means, reference.means_, atol=1e-8)  # type: ignore[attr-defined]
        np.testing.assert_allclose(ours.params.covars, reference.covars_, atol=1e-8)  # type: ignore[attr-defined]


# ============================================================================================
# Estimation
# ============================================================================================


@pytest.fixture(scope="module")
def recovered(sampled: tuple[np.ndarray, np.ndarray]) -> object:
    X, _ = sampled
    return GaussianHMM(4, n_restarts=5, seed=3).fit(X)


class TestEstimation:
    def test_em_is_monotone_without_regularisation(self, truth: HMMParams) -> None:
        X, _ = sample(1500, truth, np.random.default_rng(31))
        result = GaussianHMM(4, covariance_ridge=0.0, n_restarts=1, max_iter=200, seed=4).fit(X)
        history = np.array(result.loglik_history)
        assert (np.diff(history) >= -1e-8 * np.abs(history[:-1])).all()
        assert result.log_likelihood == history[-1]

    def test_recovers_generating_parameters(
        self, truth: HMMParams, sampled: tuple[np.ndarray, np.ndarray], recovered: object
    ) -> None:
        """Fit on 5,000 draws from a known model, map to canonical order, compare to the truth."""
        X, true_states = sampled
        fit = recovered
        assert fit.converged  # type: ignore[attr-defined]
        mapping = RegimeMapper().map(fit.params, FeatureScaling.identity(FEATURES))  # type: ignore[attr-defined]
        est = mapping.canonical_params
        assert not mapping.ambiguous
        # Tight: EM must recover the moments of the sample it was given, state by state.
        for k in range(4):
            members = X[true_states == k]
            np.testing.assert_allclose(est.means[k], members.mean(axis=0), atol=0.03)
            np.testing.assert_allclose(est.covars[k], np.cov(members.T, bias=True), atol=0.05)
        # Loose: the sample itself deviates from the generating model by sampling noise
        # (the rare crash state has ~600 draws here).
        np.testing.assert_allclose(est.means, truth.means, atol=0.12)
        np.testing.assert_allclose(
            np.diagonal(est.covars, axis1=1, axis2=2),
            np.diagonal(truth.covars, axis1=1, axis2=2),
            rtol=0.3,
        )
        np.testing.assert_allclose(np.diag(est.transmat), np.diag(truth.transmat), atol=0.03)
        # The fitted model can't beat the truth by much on its own training data, and must not
        # be much worse.
        assert fit.log_likelihood >= score(X, truth) - 5.0  # type: ignore[attr-defined]

        decoded, _ = viterbi(X, est)
        assert (decoded == true_states).mean() > 0.9
        filtered = forward(X, est).filtered.argmax(axis=1)
        assert (filtered == true_states).mean() > 0.85

    def test_selects_best_restart_deterministically(self, truth: HMMParams) -> None:
        X, _ = sample(1200, truth, np.random.default_rng(41))
        a = GaussianHMM(4, n_restarts=4, seed=9).fit(X)
        b = GaussianHMM(4, n_restarts=4, seed=9).fit(X)
        assert a.params.allclose(b.params, rtol=0, atol=0)
        accepted = [ll for ll in a.restart_log_likelihoods if ll is not None]
        assert a.log_likelihood == max(accepted)
        assert a.restart_log_likelihoods[a.best_restart] == a.log_likelihood

    def test_warm_start_converges_quickly(
        self, truth: HMMParams, sampled: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, _ = sampled
        result = GaussianHMM(4, n_restarts=1).fit(X, init=truth)
        assert result.converged
        assert result.n_iter < 30
        assert result.log_likelihood >= score(X, truth)

    def test_occupancy_floor_rejects_degenerate_fits(self, truth: HMMParams) -> None:
        X, _ = sample(600, truth, np.random.default_rng(51))
        with pytest.raises(HMMFitError, match="occupancy"):
            GaussianHMM(4, n_restarts=2, min_state_occupancy=0.2499).fit(X)

    def test_insufficient_data(self) -> None:
        X = np.random.default_rng(0).normal(size=(40, 3))
        with pytest.raises(InsufficientDataError):
            GaussianHMM(4).fit(X)

    def test_sticky_prior_lengthens_regimes(self, truth: HMMParams) -> None:
        X, _ = sample(1500, truth, np.random.default_rng(61))
        plain = GaussianHMM(4, n_restarts=2, seed=1).fit(X)
        sticky = GaussianHMM(4, n_restarts=2, seed=1, sticky_prior=300.0).fit(X)
        assert np.diag(sticky.params.transmat).mean() > np.diag(plain.params.transmat).mean()

    def test_bic_identifies_number_of_states(self) -> None:
        two_state = HMMParams(
            [0.5, 0.5],
            [[0.95, 0.05], [0.05, 0.95]],
            [[1.0, 0.0, 0.0], [-1.0, 1.5, 1.0]],
            np.stack([0.3 * np.eye(3), 0.6 * np.eye(3)]),
        )
        X, _ = sample(2000, two_state, np.random.default_rng(71))
        table = select_n_states(X, candidates=(1, 2, 3), n_restarts=3, max_iter=150, seed=2)
        assert min(table, key=lambda k: table[k].bic) == 2

    def test_config_wiring(self) -> None:
        cfg = EngineConfig.from_dict({"hmm": {"n_restarts": 3, "sticky_prior": 5.0}})
        model = GaussianHMM.from_config(cfg.hmm, seed=cfg.run.seed)
        assert (model.n_states, model.n_restarts, model.sticky_prior) == (4, 3, 5.0)
        with pytest.raises(ConfigError, match="n_states must be 4"):
            HMMConfig(n_states=3)


# ============================================================================================
# Canonical regime mapping
# ============================================================================================


class TestRegimeMapper:
    @pytest.mark.parametrize("perm", list(itertools.permutations(range(4))))
    def test_any_raw_labelling_maps_to_the_same_canonical_model(
        self, truth: HMMParams, perm: tuple[int, ...]
    ) -> None:
        raw = truth.permuted(perm)
        mapping = RegimeMapper().map(raw, FeatureScaling.identity(FEATURES))
        assert mapping.canonical_params.allclose(truth, rtol=0, atol=0)
        assert [mapping.raw_to_regime[raw_idx] for raw_idx in mapping.order] == list(Regime)

    def test_relabelled_inference_is_consistent(self, truth: HMMParams) -> None:
        raw = truth.permuted([3, 1, 0, 2])
        mapping = RegimeMapper().map(raw, FeatureScaling.identity(FEATURES))
        X, _ = sample(500, truth, np.random.default_rng(81))
        np.testing.assert_allclose(
            mapping.relabel_probs(forward(X, raw).filtered),
            forward(X, mapping.canonical_params).filtered,
            rtol=1e-12,
            atol=1e-15,
        )
        raw_path, _ = viterbi(X, raw)
        canonical_path, _ = viterbi(X, mapping.canonical_params)
        np.testing.assert_array_equal(mapping.relabel_states(raw_path), canonical_path)

    def test_moments_are_reported_in_original_units(self, truth: HMMParams) -> None:
        scaling = FeatureScaling(
            FEATURES, np.array([0.0004, 0.008, 0.3]), np.array([0.011, 0.004, 0.07])
        )
        mapping = RegimeMapper().map(truth, scaling)
        bull_low = mapping.moments[Regime.BULL_LOW_VOL]
        assert bull_low.mean_return == pytest.approx(0.8 * 0.011 + 0.0004)
        assert bull_low.volatility == pytest.approx(0.4 * 0.011)

        by_range = RegimeMapper(MappingConfig(vol_source=VolatilitySource.RANGE_VOL)).map(
            truth, scaling
        )
        assert by_range.moments[Regime.BEAR_HIGH_VOL].volatility == pytest.approx(
            1.8 * 0.004 + 0.008
        )

    def test_scaling_from_fitted_scaler(self, ohlcv_factory: OHLCVFactory) -> None:
        features = FeatureEngine().compute(ohlcv_factory(400, seed=5))
        scaler = RollingWindowScaler(252).fit(features)
        scaling = FeatureScaling.from_scaler_params(scaler.params)
        assert scaling.names == FEATURES
        np.testing.assert_allclose(scaling.location, scaler.params.location.to_numpy())

    def test_economically_doubtful_labels_are_flagged(
        self, truth: HMMParams, captured: ListHandler
    ) -> None:
        bearish = dataclasses.replace(truth, means=truth.means - np.array([2.0, 0.0, 0.0]))
        mapping = RegimeMapper().map(bearish, FeatureScaling.identity(FEATURES))
        kinds = {a.kind for a in mapping.anomalies}
        assert mapping.ambiguous
        assert OrderingAnomalyKind.BULL_NON_POSITIVE_MEAN in kinds
        assert len(captured.events("REGIME_ORDERING_ANOMALY")) == len(mapping.anomalies)

    def test_hungarian_cross_check(self, truth: HMMParams, captured: ListHandler) -> None:
        mapper = RegimeMapper()
        scaling = FeatureScaling.identity(FEATURES)
        first = mapper.map(truth.permuted([1, 0, 2, 3]), scaling)
        second = mapper.map(truth.permuted([3, 2, 0, 1]), scaling, previous=first)
        assert second.hungarian_agrees is True
        assert not captured.events("REGIME_MAPPING_DISAGREEMENT")

        swapped = dict(first.moments)
        swapped[Regime.BULL_LOW_VOL], swapped[Regime.BULL_HIGH_VOL] = (
            first.moments[Regime.BULL_HIGH_VOL],
            first.moments[Regime.BULL_LOW_VOL],
        )
        stale = dataclasses.replace(first, moments=swapped)
        third = mapper.map(truth, scaling, previous=stale)
        assert third.hungarian_agrees is False
        assert len(captured.events("REGIME_MAPPING_DISAGREEMENT")) == 1
        assert third.canonical_params.allclose(truth, rtol=0, atol=0)  # rank labels still win

    def test_rejects_wrong_shapes(self, truth: HMMParams) -> None:
        with pytest.raises(RegimeMappingError, match="exactly 4"):
            RegimeMapper().map(_small_model(), FeatureScaling.identity(("log_ret", "hl_vol")))
        with pytest.raises(RegimeMappingError, match="not in"):
            RegimeMapper().map(truth, FeatureScaling.identity(("a", "b", "c")))
        with pytest.raises(RegimeMappingError):
            FeatureScaling(FEATURES, np.zeros(3), np.array([1.0, 0.0, 1.0]))


# ============================================================================================
# End to end: bars -> features -> scaled -> HMM -> canonical regimes
# ============================================================================================


def test_phase2_pipeline_on_synthetic_bars(ohlcv_factory: OHLCVFactory) -> None:
    cfg = EngineConfig.from_dict({"hmm": {"n_restarts": 3}})
    features = FeatureEngine(cfg.features).compute(ohlcv_factory(900, seed=17))
    scaler = RollingWindowScaler.from_config(cfg.scaling)
    train = scaler.fit_transform(features.iloc[:600])
    fit = GaussianHMM.from_config(cfg.hmm, seed=cfg.run.seed).fit(train.to_numpy())
    mapping = RegimeMapper(cfg.mapping).map(
        fit.params, FeatureScaling.from_scaler_params(scaler.params)
    )
    out_of_sample = scaler.transform(features.loc[features.index > scaler.params.fit_end])
    filtered = pd.DataFrame(
        forward(out_of_sample.to_numpy(), mapping.canonical_params).filtered,
        index=out_of_sample.index,
        columns=[r.name for r in Regime],
    )
    np.testing.assert_allclose(filtered.sum(axis=1), 1.0, atol=1e-12)
    returns = [mapping.moments[r].mean_return for r in Regime]
    assert min(returns[:2]) >= max(returns[2:])
