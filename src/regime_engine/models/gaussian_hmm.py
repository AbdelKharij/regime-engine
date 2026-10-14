"""Gaussian hidden Markov model, implemented from scratch in NumPy.

Model
-----
Hidden state S_t in {0..K-1}. Parameters theta = (pi, A, {mu_k, Sigma_k}):

    P(S_1 = k)              = pi_k
    P(S_{t+1} = j | S_t = i) = A_ij
    x_t | S_t = k           ~ N(mu_k, Sigma_k)      (full covariance)

Inference (all recursions are scaled, so nothing underflows)
-------------------------------------------------------------
* ``forward`` / ``filter_step``: the *filtered* posterior P(S_t | x_1..t). This is causal and
  is the only regime estimate a trading decision may use.
* ``smooth``: P(S_t | x_1..T) via forward-backward. It uses future observations, so it is for
  training and ex-post diagnostics only.
* ``viterbi``: the single most likely state path. Also uses the future, so diagnostics only.

Estimation
----------
``GaussianHMM.fit`` runs Baum-Welch (EM) from several k-means++ initialisations and keeps the
best accepted restart. Two optional regularisers are available: a covariance ridge (keeps
Sigma_k positive definite when a state collapses onto a few outlier bars) and a "sticky"
Dirichlet prior on the diagonal of A (encourages persistent regimes). With both set to zero,
the update equations match ``hmmlearn`` (``covars_prior=0``), which the test suite checks.

Inference is exposed as pure module-level functions of ``(X, params)``. ``GaussianHMM``
holds only estimation hyperparameters. Fitted parameters are immutable ``HMMParams`` values.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_triangular

from regime_engine.core.config import HMMConfig
from regime_engine.core.exceptions import (
    HMMFitError,
    HMMNumericalError,
    HMMParameterError,
    InsufficientDataError,
)
from regime_engine.utils.logging import LogEvent, get_logger

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

_LOG = get_logger("models.hmm")
_LOG_2PI: Final = math.log(2.0 * math.pi)
_STOCHASTIC_ATOL: Final = 1e-8
_MIN_STATE_MASS: Final = 1e-10


def count_free_parameters(n_states: int, n_features: int) -> int:
    """(K-1) start + K(K-1) transition + K*d means + K*d(d+1)/2 covariance parameters."""
    k, d = n_states, n_features
    return (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2


def _frozen(values: npt.ArrayLike, ndim: int, name: str) -> FloatArray:
    arr = np.array(values, dtype=np.float64, copy=True)
    if arr.ndim != ndim:
        raise HMMParameterError(f"{name} must be {ndim}-dimensional, got shape {arr.shape}")
    arr.setflags(write=False)
    return arr


# --------------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class HMMParams:
    """Immutable, validated HMM parameters. Arrays are copied and made read-only."""

    startprob: FloatArray  # (K,)
    transmat: FloatArray  # (K, K), rows sum to 1
    means: FloatArray  # (K, d)
    covars: FloatArray  # (K, d, d), symmetric positive definite

    def __post_init__(self) -> None:
        object.__setattr__(self, "startprob", _frozen(self.startprob, 1, "startprob"))
        object.__setattr__(self, "transmat", _frozen(self.transmat, 2, "transmat"))
        object.__setattr__(self, "means", _frozen(self.means, 2, "means"))
        object.__setattr__(self, "covars", _frozen(self.covars, 3, "covars"))
        self._validate()

    def _validate(self) -> None:
        k, d = self.means.shape
        if k < 1 or d < 1:
            raise HMMParameterError(f"means must be (K>=1, d>=1), got {self.means.shape}")
        expected = {
            "startprob": (k,),
            "transmat": (k, k),
            "covars": (k, d, d),
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise HMMParameterError(f"{name} has shape {actual}, expected {shape}")
        for name in ("startprob", "transmat", "means", "covars"):
            if not np.isfinite(getattr(self, name)).all():
                raise HMMParameterError(f"{name} contains non-finite values")
        if (self.startprob < 0).any() or abs(self.startprob.sum() - 1.0) > _STOCHASTIC_ATOL:
            raise HMMParameterError("startprob must be a probability vector")
        if (self.transmat < 0).any() or not np.allclose(
            self.transmat.sum(axis=1), 1.0, rtol=0.0, atol=_STOCHASTIC_ATOL
        ):
            raise HMMParameterError("transmat rows must be probability vectors")
        scale = max(1.0, float(np.abs(self.covars).max()))
        if not np.allclose(self.covars, self.covars.transpose(0, 2, 1), rtol=0, atol=1e-10 * scale):
            raise HMMParameterError("covariance matrices must be symmetric")
        _ = self.cholesky  # raises if any covariance is not positive definite

    # ---- derived quantities --------------------------------------------------------------

    @property
    def n_states(self) -> int:
        return int(self.means.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.means.shape[1])

    @cached_property
    def cholesky(self) -> FloatArray:
        """Lower Cholesky factors L_k with Sigma_k = L_k L_k^T, shape (K, d, d)."""
        try:
            chol = np.asarray(np.linalg.cholesky(self.covars), dtype=np.float64)
        except np.linalg.LinAlgError as exc:
            raise HMMParameterError("a covariance matrix is not positive definite") from exc
        chol.setflags(write=False)
        return chol

    @cached_property
    def log_det_covars(self) -> FloatArray:
        diag = np.diagonal(self.cholesky, axis1=1, axis2=2)
        return np.asarray(2.0 * np.log(diag).sum(axis=1), dtype=np.float64)

    @property
    def n_free_parameters(self) -> int:
        return count_free_parameters(self.n_states, self.n_features)

    def stationary_distribution(self) -> FloatArray:
        """Left eigenvector of A for eigenvalue 1 (pi A = pi), via least squares."""
        k = self.n_states
        system = np.vstack([self.transmat.T - np.eye(k), np.ones((1, k))])
        target = np.zeros(k + 1)
        target[-1] = 1.0
        solution, *_ = np.linalg.lstsq(system, target, rcond=None)
        solution = np.clip(solution, 0.0, None)
        return np.asarray(solution / solution.sum(), dtype=np.float64)

    def expected_durations(self) -> FloatArray:
        """E[D_k] = 1 / (1 - A_kk) bars (infinite for an absorbing state)."""
        stay = np.diag(self.transmat)
        with np.errstate(divide="ignore"):
            return np.asarray(np.where(stay < 1.0, 1.0 / (1.0 - stay), np.inf), dtype=np.float64)

    def permuted(self, order: npt.ArrayLike) -> HMMParams:
        """Relabel states: new state i is old state ``order[i]``.

        All parameters are permuted together (pi' = pi[o], A' = A[o][:, o], mu' = mu[o],
        Sigma' = Sigma[o]), so the model and its likelihood are unchanged.
        """
        perm = np.asarray(order, dtype=np.int64)
        if sorted(perm.tolist()) != list(range(self.n_states)):
            raise HMMParameterError(
                f"{perm.tolist()} is not a permutation of 0..{self.n_states - 1}"
            )
        return HMMParams(
            startprob=self.startprob[perm],
            transmat=self.transmat[np.ix_(perm, perm)],
            means=self.means[perm],
            covars=self.covars[perm],
        )

    def allclose(self, other: HMMParams, *, rtol: float = 1e-7, atol: float = 1e-9) -> bool:
        return all(
            np.allclose(getattr(self, name), getattr(other, name), rtol=rtol, atol=atol)
            for name in ("startprob", "transmat", "means", "covars")
        )


# --------------------------------------------------------------------------------------------
# Inference (pure functions)
# --------------------------------------------------------------------------------------------


def _as_observations(X: npt.ArrayLike, n_features: int) -> FloatArray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :] if n_features > 1 else arr[:, None]
    if arr.ndim != 2 or arr.shape[1] != n_features:
        raise HMMParameterError(f"observations must have shape (T, {n_features}), got {arr.shape}")
    if len(arr) == 0:
        raise InsufficientDataError("no observations")
    if not np.isfinite(arr).all():
        raise HMMNumericalError("observations contain NaN or inf")
    return arr


def log_emission(X: npt.ArrayLike, params: HMMParams) -> FloatArray:
    """log N(x_t; mu_k, Sigma_k) for every t and k, shape (T, K). Cholesky-based."""
    obs = _as_observations(X, params.n_features)
    out = np.empty((len(obs), params.n_states))
    const = params.n_features * _LOG_2PI
    for k in range(params.n_states):
        whitened = solve_triangular(
            params.cholesky[k], (obs - params.means[k]).T, lower=True, check_finite=False
        )
        mahalanobis = np.einsum("ij,ij->j", whitened, whitened)
        out[:, k] = -0.5 * (const + params.log_det_covars[k] + mahalanobis)
    return out


def _scaled_step(predicted: FloatArray, log_b: FloatArray) -> tuple[FloatArray, float]:
    """One normalised forward step: alpha_t ∝ predicted ∘ b(x_t). Returns (alpha_t, log c_t)."""
    shift = float(log_b.max())
    unnormalised = predicted * np.exp(log_b - shift)
    total = float(unnormalised.sum())
    if not (total > 0.0 and math.isfinite(total)):
        raise HMMNumericalError(
            "forward recursion collapsed (observation impossible under every reachable state)"
        )
    return unnormalised / total, math.log(total) + shift


@dataclass(frozen=True, eq=False)
class ForwardResult:
    filtered: FloatArray  # (T, K): P(S_t = k | x_1..t)
    log_scale: FloatArray  # (T,): log p(x_t | x_1..t-1); sums to the log-likelihood

    @property
    def log_likelihood(self) -> float:
        return float(self.log_scale.sum())


def forward(X: npt.ArrayLike, params: HMMParams) -> ForwardResult:
    """Filtered state probabilities. Row t depends only on x_1..x_t."""
    log_b = log_emission(X, params)
    n_obs = len(log_b)
    filtered = np.empty((n_obs, params.n_states))
    log_scale = np.empty(n_obs)
    predicted: FloatArray = params.startprob
    for t in range(n_obs):
        filtered[t], log_scale[t] = _scaled_step(predicted, log_b[t])
        with np.errstate(all="ignore"):  # spurious BLAS flags; _scaled_step checks finiteness
            predicted = filtered[t] @ params.transmat
    return ForwardResult(filtered, log_scale)


def filter_step(
    previous: FloatArray | None, x_t: npt.ArrayLike, params: HMMParams
) -> tuple[FloatArray, float]:
    """Advance the filter by one observation (online use).

    ``previous`` is P(S_{t-1} | x_1..t-1), or ``None`` for the first observation. Returns
    (P(S_t | x_1..t), log p(x_t | x_1..t-1)).
    """
    with np.errstate(all="ignore"):  # spurious BLAS flags; _scaled_step checks finiteness
        predicted = params.startprob if previous is None else previous @ params.transmat
    log_b = log_emission(np.atleast_2d(np.asarray(x_t, dtype=np.float64)), params)[0]
    return _scaled_step(predicted, log_b)


def predict_next(filtered_t: FloatArray, params: HMMParams) -> FloatArray:
    """One-step-ahead regime forecast P(S_{t+1} | x_1..t) = alpha_t A."""
    return np.asarray(filtered_t @ params.transmat, dtype=np.float64)


def score(X: npt.ArrayLike, params: HMMParams) -> float:
    """Log-likelihood log p(x_1..T | theta)."""
    return forward(X, params).log_likelihood


def _forward_backward(log_b: FloatArray, params: HMMParams) -> tuple[FloatArray, FloatArray, float]:
    """Return (gamma (T,K), sum_t xi_t (K,K), log-likelihood) from emission log-densities."""
    # Some BLAS backends (e.g. Apple Accelerate with numpy 2.x) raise spurious floating-point
    # flags inside matmul on finite inputs. Flags are silenced here and every output is
    # checked explicitly for finiteness instead.
    with np.errstate(all="ignore"):
        gamma, xi_sum, log_likelihood = _forward_backward_unchecked(log_b, params)
    if not (
        np.isfinite(gamma).all() and np.isfinite(xi_sum).all() and math.isfinite(log_likelihood)
    ):
        raise HMMNumericalError("forward-backward produced non-finite values")
    return gamma, xi_sum, log_likelihood


def _forward_backward_unchecked(
    log_b: FloatArray, params: HMMParams
) -> tuple[FloatArray, FloatArray, float]:
    n_obs, k = log_b.shape
    shift = log_b.max(axis=1)
    b = np.exp(log_b - shift[:, None])
    transmat = params.transmat

    alpha = np.empty((n_obs, k))
    scale = np.empty(n_obs)
    predicted: FloatArray = params.startprob
    for t in range(n_obs):
        unnormalised = predicted * b[t]
        total = unnormalised.sum()
        if not (total > 0.0 and np.isfinite(total)):
            raise HMMNumericalError(f"forward recursion collapsed at t={t}")
        scale[t] = total
        alpha[t] = unnormalised / total
        predicted = alpha[t] @ transmat

    beta = np.empty((n_obs, k))
    beta[-1] = 1.0
    for t in range(n_obs - 2, -1, -1):
        beta[t] = transmat @ (b[t + 1] * beta[t + 1]) / scale[t + 1]

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    if n_obs > 1:
        weighted_next = b[1:] * beta[1:] / scale[1:, None]
        xi_sum = transmat * (alpha[:-1].T @ weighted_next)
    else:
        xi_sum = np.zeros((k, k))
    log_likelihood = float(np.log(scale).sum() + shift.sum())
    return gamma, xi_sum, log_likelihood


def smooth(X: npt.ArrayLike, params: HMMParams) -> FloatArray:
    """Smoothed posteriors P(S_t | x_1..T). **Uses the future**: diagnostics/training only."""
    gamma, _, _ = _forward_backward(log_emission(X, params), params)
    return gamma


def viterbi(X: npt.ArrayLike, params: HMMParams) -> tuple[IntArray, float]:
    """Most likely state path and its joint log-probability. **Uses the future.**"""
    log_b = log_emission(X, params)
    n_obs, k = log_b.shape
    with np.errstate(divide="ignore"):
        log_a = np.log(params.transmat)
        log_pi = np.log(params.startprob)
    delta = log_pi + log_b[0]
    backpointer = np.zeros((n_obs, k), dtype=np.int64)
    columns = np.arange(k)
    for t in range(1, n_obs):
        candidates = delta[:, None] + log_a
        backpointer[t] = candidates.argmax(axis=0)
        delta = candidates[backpointer[t], columns] + log_b[t]
    path = np.empty(n_obs, dtype=np.int64)
    path[-1] = int(delta.argmax())
    for t in range(n_obs - 1, 0, -1):
        path[t - 1] = backpointer[t, path[t]]
    return path, float(delta.max())


def sample(n_obs: int, params: HMMParams, rng: np.random.Generator) -> tuple[FloatArray, IntArray]:
    """Draw (X, states) of length ``n_obs`` from the model."""
    if n_obs < 1:
        raise ValueError("n_obs must be >= 1")
    k = params.n_states
    cumulative = np.cumsum(params.transmat, axis=1)
    uniforms = rng.random(n_obs)
    states = np.empty(n_obs, dtype=np.int64)
    states[0] = int(rng.choice(k, p=params.startprob))
    for t in range(1, n_obs):
        row = cumulative[states[t - 1]]
        states[t] = min(int(np.searchsorted(row, uniforms[t], side="right")), k - 1)
    noise = rng.standard_normal((n_obs, params.n_features))
    X = params.means[states] + np.einsum("tij,tj->ti", params.cholesky[states], noise)
    return X, states


# --------------------------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class FitResult:
    params: HMMParams
    log_likelihood: float
    n_iter: int
    converged: bool
    n_obs: int
    state_occupancy: FloatArray  # mean smoothed posterior per state
    loglik_history: tuple[float, ...]
    restart_log_likelihoods: tuple[float | None, ...]  # None = rejected restart
    best_restart: int

    @property
    def n_parameters(self) -> int:
        return self.params.n_free_parameters

    @property
    def bic(self) -> float:
        return -2.0 * self.log_likelihood + self.n_parameters * math.log(self.n_obs)

    @property
    def aic(self) -> float:
        return -2.0 * self.log_likelihood + 2.0 * self.n_parameters


@dataclass(frozen=True, eq=False)
class _RestartOutcome:
    params: HMMParams
    log_likelihood: float
    n_iter: int
    converged: bool
    occupancy: FloatArray
    history: tuple[float, ...]


def _kmeans(
    X: FloatArray, k: int, rng: np.random.Generator, n_iter: int
) -> tuple[FloatArray, IntArray]:
    """k-means++ seeding followed by Lloyd iterations. Empty clusters are re-seeded."""
    n_obs = len(X)
    centers = np.empty((k, X.shape[1]))
    centers[0] = X[rng.integers(n_obs)]
    closest_sq = ((X - centers[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        total = closest_sq.sum()
        pick = (
            int(rng.choice(n_obs, p=closest_sq / total)) if total > 0 else int(rng.integers(n_obs))
        )
        centers[j] = X[pick]
        closest_sq = np.minimum(closest_sq, ((X - centers[j]) ** 2).sum(axis=1))

    labels = np.zeros(n_obs, dtype=np.int64)
    for _ in range(n_iter):
        sq_dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = sq_dist.argmin(axis=1)
        updated = centers.copy()
        for j in range(k):
            members = X[labels == j]
            if len(members):
                updated[j] = members.mean(axis=0)
            else:
                updated[j] = X[int(sq_dist.min(axis=1).argmax())]
        moved = not np.allclose(updated, centers)
        centers = updated
        if not moved:
            break
    labels = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return centers, labels.astype(np.int64)


class GaussianHMM:
    """Baum-Welch estimator for a full-covariance Gaussian HMM.

    Stateless apart from hyperparameters. ``fit`` returns a ``FitResult``, and inference uses
    the module-level functions with ``result.params``.
    """

    def __init__(
        self,
        n_states: int = 4,
        *,
        covariance_ridge: float = 1e-6,
        max_iter: int = 500,
        tol: float = 1e-4,
        n_restarts: int = 10,
        init_self_transition: float = 0.9,
        sticky_prior: float = 0.0,
        min_state_occupancy: float = 0.0,
        kmeans_iter: int = 25,
        seed: int = 0,
    ) -> None:
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        if covariance_ridge < 0 or sticky_prior < 0:
            raise ValueError("covariance_ridge and sticky_prior must be non-negative")
        if max_iter < 1 or n_restarts < 1 or kmeans_iter < 1:
            raise ValueError("max_iter, n_restarts and kmeans_iter must be >= 1")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if not 0.0 < init_self_transition < 1.0 and n_states > 1:
            raise ValueError("init_self_transition must be in (0, 1)")
        if not 0.0 <= min_state_occupancy < 1.0 / n_states:
            raise ValueError("min_state_occupancy must be in [0, 1 / n_states)")
        self.n_states = n_states
        self.covariance_ridge = covariance_ridge
        self.max_iter = max_iter
        self.tol = tol
        self.n_restarts = n_restarts
        self.init_self_transition = init_self_transition
        self.sticky_prior = sticky_prior
        self.min_state_occupancy = min_state_occupancy
        self.kmeans_iter = kmeans_iter
        self.seed = seed

    @classmethod
    def from_config(cls, config: HMMConfig, *, seed: int = 0) -> GaussianHMM:
        return cls(
            config.n_states,
            covariance_ridge=config.covariance_ridge,
            max_iter=config.max_iter,
            tol=config.tol,
            n_restarts=config.n_restarts,
            init_self_transition=config.init_self_transition,
            sticky_prior=config.sticky_prior,
            min_state_occupancy=config.min_state_occupancy,
            kmeans_iter=config.kmeans_iter,
            seed=seed,
        )

    # ---- public ----------------------------------------------------------------------------

    def fit(self, X: npt.ArrayLike, *, init: HMMParams | None = None) -> FitResult:
        """Fit by EM with restarts. ``init``, if given, seeds the first restart (warm start)."""
        obs = np.asarray(X, dtype=np.float64)
        if obs.ndim != 2:
            raise HMMParameterError(f"X must be 2-dimensional (T, d), got shape {obs.shape}")
        obs = _as_observations(obs, obs.shape[1])
        if init is not None and (init.n_states, init.n_features) != (self.n_states, obs.shape[1]):
            raise HMMParameterError("init parameters do not match n_states / n_features")
        n_params = count_free_parameters(self.n_states, obs.shape[1])
        if len(obs) <= n_params:
            raise InsufficientDataError(
                f"{len(obs)} observations cannot identify {n_params} free parameters"
            )

        rng = np.random.default_rng(self.seed)
        accepted: list[tuple[int, _RestartOutcome]] = []
        restart_lls: list[float | None] = []
        reasons: list[str] = []
        for restart in range(self.n_restarts):
            start = init if (init is not None and restart == 0) else self._initial_params(obs, rng)
            try:
                outcome = self._em(obs, start)
            except (HMMNumericalError, HMMParameterError) as exc:
                reasons.append(f"restart {restart}: {exc}")
                restart_lls.append(None)
                _LOG.warning(
                    LogEvent.HMM_RESTART_REJECTED,
                    "EM restart failed",
                    restart=restart,
                    reason=str(exc),
                )
                continue
            min_occupancy = float(outcome.occupancy.min())
            if min_occupancy < self.min_state_occupancy:
                reasons.append(f"restart {restart}: min state occupancy {min_occupancy:.4f}")
                restart_lls.append(None)
                _LOG.debug(
                    LogEvent.HMM_RESTART_REJECTED,
                    "degenerate state occupancy",
                    restart=restart,
                    min_occupancy=min_occupancy,
                    threshold=self.min_state_occupancy,
                )
                continue
            accepted.append((restart, outcome))
            restart_lls.append(outcome.log_likelihood)

        if not accepted:
            raise HMMFitError(
                f"all {self.n_restarts} EM restarts were rejected:\n  " + "\n  ".join(reasons)
            )
        best_restart, best = max(accepted, key=lambda item: item[1].log_likelihood)
        result = FitResult(
            params=best.params,
            log_likelihood=best.log_likelihood,
            n_iter=best.n_iter,
            converged=best.converged,
            n_obs=len(obs),
            state_occupancy=best.occupancy,
            loglik_history=best.history,
            restart_log_likelihoods=tuple(restart_lls),
            best_restart=best_restart,
        )
        _LOG.info(
            LogEvent.HMM_FIT,
            "Gaussian HMM fitted",
            n_obs=len(obs),
            n_states=self.n_states,
            log_likelihood=result.log_likelihood,
            bic=result.bic,
            n_iter=result.n_iter,
            converged=result.converged,
            accepted_restarts=len(accepted),
            best_restart=best_restart,
            occupancy=result.state_occupancy,
        )
        return result

    # ---- internals -------------------------------------------------------------------------

    def _initial_params(self, X: FloatArray, rng: np.random.Generator) -> HMMParams:
        k, d = self.n_states, X.shape[1]
        centers, labels = _kmeans(X, k, rng, self.kmeans_iter)
        global_cov = np.atleast_2d(np.cov(X.T, bias=True))
        floor = 1e-3 * np.diag(np.diag(global_cov)) + self.covariance_ridge * np.eye(d)
        covars = np.empty((k, d, d))
        for j in range(k):
            members = X[labels == j]
            cov = np.atleast_2d(np.cov(members.T, bias=True)) if len(members) > d else global_cov
            covars[j] = 0.5 * (cov + cov.T) + floor
        if k == 1:
            transmat = np.ones((1, 1))
        else:
            off = (1.0 - self.init_self_transition) / (k - 1)
            transmat = np.full((k, k), off)
            np.fill_diagonal(transmat, self.init_self_transition)
        return HMMParams(np.full(k, 1.0 / k), transmat, centers, covars)

    def _m_step(
        self, X: FloatArray, gamma: FloatArray, xi_sum: FloatArray, previous: HMMParams
    ) -> HMMParams:
        k, d = previous.n_states, previous.n_features
        startprob = gamma[0] / gamma[0].sum()

        counts = xi_sum + self.sticky_prior * np.eye(k)
        row_mass = counts.sum(axis=1, keepdims=True)
        has_mass = row_mass > _MIN_STATE_MASS
        transmat = np.where(has_mass, counts / np.where(has_mass, row_mass, 1.0), previous.transmat)

        state_mass = gamma.sum(axis=0)
        means = previous.means.copy()
        covars = previous.covars.copy()
        ridge = self.covariance_ridge * np.eye(d)
        for j in range(k):
            if state_mass[j] <= _MIN_STATE_MASS:
                continue  # unused state: keep previous emission parameters
            weights = gamma[:, j]
            with np.errstate(all="ignore"):  # spurious BLAS flags; checked just below
                means[j] = weights @ X / state_mass[j]
                centred = X - means[j]
                cov = (weights[:, None] * centred).T @ centred / state_mass[j]
            if not (np.isfinite(means[j]).all() and np.isfinite(cov).all()):
                raise HMMNumericalError(f"M-step produced non-finite parameters for state {j}")
            covars[j] = 0.5 * (cov + cov.T) + ridge
        return HMMParams(startprob, transmat, means, covars)

    def _em(self, X: FloatArray, start: HMMParams) -> _RestartOutcome:
        params = start
        gamma, xi_sum, log_likelihood = _forward_backward(log_emission(X, params), params)
        history = [log_likelihood]
        converged = False
        n_iter = 0
        while n_iter < self.max_iter:
            candidate = self._m_step(X, gamma, xi_sum, params)
            new_gamma, new_xi, new_ll = _forward_backward(log_emission(X, candidate), candidate)
            n_iter += 1
            history.append(new_ll)
            if new_ll < log_likelihood - 1e-8 * max(1.0, abs(log_likelihood)):
                # Exact EM never decreases the likelihood. With a ridge or sticky prior it
                # maximises a penalised objective instead, so small decreases are possible.
                _LOG.debug(
                    LogEvent.HMM_LOGLIK_DECREASE,
                    "log-likelihood decreased during EM",
                    iteration=n_iter,
                    before=log_likelihood,
                    after=new_ll,
                )
            improvement = new_ll - log_likelihood
            params, gamma, xi_sum, log_likelihood = candidate, new_gamma, new_xi, new_ll
            if abs(improvement) < self.tol:
                converged = True
                break
        return _RestartOutcome(
            params=params,
            log_likelihood=log_likelihood,
            n_iter=n_iter,
            converged=converged,
            occupancy=gamma.mean(axis=0),
            history=tuple(history),
        )


def select_n_states(
    X: npt.ArrayLike, candidates: Iterable[int] = (2, 3, 4, 5), **hmm_kwargs: object
) -> dict[int, FitResult]:
    """Fit one model per candidate K. Compare ``result.bic`` (lower is better) to justify K."""
    results: dict[int, FitResult] = {}
    for k in candidates:
        results[k] = GaussianHMM(k, **hmm_kwargs).fit(X)  # type: ignore[arg-type]
    return results
