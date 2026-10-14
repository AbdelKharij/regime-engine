"""Value at Risk, Expected Shortfall and VaR backtests.

Conventions: all functions take per-bar *returns* and report VaR / ES as positive *loss
fractions* per unit of exposure (multiply by |weight| x equity for currency). ``alpha`` is the
confidence level, e.g. 0.99 means "loss exceeded on 1% of bars".

Methods
-------
* historical:  VaR = -Q_{1-alpha}(r)
* parametric:  VaR = -(mu + z_{1-alpha} sigma),   ES = sigma phi(z_alpha) / (1 - alpha) - mu
* regime mixture (HMM-native): solve sum_k p_k Phi((q - m_k) / s_k) = 1 - alpha for q, with
  p = one-step-ahead regime probabilities. VaR = -q. The fat left tail comes from the
  crash regime's weight, not from a distributional assumption on the whole sample.
* filtered historical simulation (Barone-Adesi et al. 1999): standardise each realised return
  by the forecast made for it, z = (r - mu_hat) / sigma_hat, then
  VaR = -(mu + sigma Q_{1-alpha}(z)).
  The regime model supplies the volatility *dynamics*; the empirical z supplies the *tails*.

Backtests
---------
* Kupiec (1995) proportion of failures: LR = -2 ln[L(p) / L(p_hat)] ~ chi2(1).
* Christoffersen (1998) independence: are breaches clustered? LR ~ chi2(1).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from scipy.stats import chi2, norm

from regime_engine.core.exceptions import InsufficientDataError
from regime_engine.core.types import Regime, RegimeMoments

FloatArray = npt.NDArray[np.float64]


def _returns(values: npt.ArrayLike, minimum: int = 2) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if len(arr) < minimum:
        raise InsufficientDataError(f"need at least {minimum} finite returns, got {len(arr)}")
    return np.asarray(arr, dtype=np.float64)


def _check_alpha(alpha: float) -> None:
    if not 0.5 < alpha < 1.0:
        raise ValueError("alpha must be in (0.5, 1)")


# --------------------------------------------------------------------------------------------
# Point estimates
# --------------------------------------------------------------------------------------------


def historical_var(returns: npt.ArrayLike, alpha: float) -> float:
    _check_alpha(alpha)
    return float(-np.quantile(_returns(returns), 1.0 - alpha))


def historical_es(returns: npt.ArrayLike, alpha: float) -> float:
    r = _returns(returns)
    threshold = np.quantile(r, 1.0 - alpha)
    return float(-r[r <= threshold].mean())


def gaussian_var(mu: float, sigma: float, alpha: float) -> float:
    _check_alpha(alpha)
    return float(-(mu + norm.ppf(1.0 - alpha) * sigma))


def gaussian_es(mu: float, sigma: float, alpha: float) -> float:
    _check_alpha(alpha)
    return float(sigma * norm.pdf(norm.ppf(alpha)) / (1.0 - alpha) - mu)


def parametric_var(returns: npt.ArrayLike, alpha: float) -> float:
    r = _returns(returns)
    return gaussian_var(float(r.mean()), float(r.std(ddof=1)), alpha)


def parametric_es(returns: npt.ArrayLike, alpha: float) -> float:
    r = _returns(returns)
    return gaussian_es(float(r.mean()), float(r.std(ddof=1)), alpha)


def regime_mixture_moments(
    probs: npt.ArrayLike, moments: Mapping[Regime, RegimeMoments]
) -> tuple[float, float]:
    """Mean and std of the regime-weighted Gaussian mixture of per-bar returns.

    mu = sum p_k m_k ;  sigma^2 = sum p_k (s_k^2 + m_k^2) - mu^2
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.shape != (len(Regime),) or not np.isfinite(p).all() or abs(p.sum() - 1.0) > 1e-8:
        raise ValueError(f"regime probabilities must be a length-{len(Regime)} distribution")
    m = np.array([moments[r].mean_return for r in Regime])
    s = np.array([moments[r].volatility for r in Regime])
    mu = float(p @ m)
    variance = float(p @ (s**2 + m**2)) - mu**2
    return mu, math.sqrt(max(variance, 0.0))


def mixture_var(
    probs: npt.ArrayLike, moments: Mapping[Regime, RegimeMoments], alpha: float
) -> float:
    """VaR of the regime mixture: root of F(q) = 1 - alpha (Brent's method)."""
    _check_alpha(alpha)
    p = np.asarray(probs, dtype=np.float64)
    m = np.array([moments[r].mean_return for r in Regime])
    s = np.array([moments[r].volatility for r in Regime])
    if (s <= 0).any():
        raise ValueError("regime volatilities must be positive")
    target = 1.0 - alpha

    def excess_cdf(q: float) -> float:
        return float(p @ norm.cdf((q - m) / s)) - target

    lo = float((m - 12.0 * s).min())
    hi = float((m + 12.0 * s).max())
    return float(-brentq(excess_cdf, lo, hi, xtol=1e-14, rtol=1e-12))


# --------------------------------------------------------------------------------------------
# Backtests
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoverageTest:
    n_obs: int
    n_breaches: int
    expected_rate: float
    observed_rate: float
    lr_statistic: float
    p_value: float

    def rejects(self, significance: float = 0.05) -> bool:
        return self.p_value < significance


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * math.log(y)


def kupiec_pof(breaches: npt.ArrayLike, alpha: float) -> CoverageTest:
    """Unconditional coverage: is the breach rate consistent with 1 - alpha?"""
    _check_alpha(alpha)
    b = np.asarray(breaches, dtype=bool).reshape(-1)
    n, x = len(b), int(b.sum())
    if n == 0:
        raise InsufficientDataError("no observations to backtest")
    p, p_hat = 1.0 - alpha, x / n
    log_null = _xlogy(n - x, 1.0 - p) + _xlogy(x, p)
    log_alt = _xlogy(n - x, 1.0 - p_hat) + _xlogy(x, p_hat)
    lr = max(0.0, -2.0 * (log_null - log_alt))
    return CoverageTest(n, x, p, p_hat, lr, float(chi2.sf(lr, df=1)))


def christoffersen_independence(breaches: npt.ArrayLike) -> CoverageTest:
    """Independence: does a breach today change the probability of a breach tomorrow?"""
    b = np.asarray(breaches, dtype=bool).reshape(-1)
    if len(b) < 2:
        raise InsufficientDataError("need at least 2 observations")
    prev, curr = b[:-1], b[1:]
    n00 = int((~prev & ~curr).sum())
    n01 = int((~prev & curr).sum())
    n10 = int((prev & ~curr).sum())
    n11 = int((prev & curr).sum())
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (len(curr))
    log_null = _xlogy(n00 + n10, 1.0 - pi) + _xlogy(n01 + n11, pi)
    log_alt = (
        _xlogy(n00, 1.0 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1.0 - pi11) + _xlogy(n11, pi11)
    )
    lr = max(0.0, -2.0 * (log_null - log_alt))
    return CoverageTest(len(curr), int(curr.sum()), pi, pi11, lr, float(chi2.sf(lr, df=1)))


# --------------------------------------------------------------------------------------------
# Rolling calculator
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VaREstimates:
    historical: float
    parametric: float
    mixture: float | None


class VaRCalculator:
    """One-bar VaR per unit of exposure by all three methods over a trailing window."""

    def __init__(self, alpha: float = 0.99, window: int = 252) -> None:
        _check_alpha(alpha)
        if window < 20:
            raise ValueError("window must be >= 20")
        self.alpha = alpha
        self.window = window

    def estimate(
        self,
        trailing_returns: npt.ArrayLike,
        probs: npt.ArrayLike | None = None,
        moments: Mapping[Regime, RegimeMoments] | None = None,
    ) -> VaREstimates:
        r = _returns(trailing_returns, minimum=20)[-self.window :]
        mixture = (
            mixture_var(probs, moments, self.alpha)
            if probs is not None and moments is not None
            else None
        )
        return VaREstimates(
            historical=historical_var(r, self.alpha),
            parametric=parametric_var(r, self.alpha),
            mixture=mixture,
        )


class FilteredHistoricalSimulation:
    """Online FHS VaR/ES from paired (realised return, forecast mu, forecast sigma).

    Causality: ``record`` must pair the return of bar t+1 with the forecast made at the close
    of bar t; ``var`` then rescales the trailing standardised residuals by the *next* forecast.
    """

    def __init__(self, alpha: float = 0.99, window: int = 252, min_obs: int = 100) -> None:
        _check_alpha(alpha)
        if not 20 <= min_obs <= window:
            raise ValueError("min_obs must be in [20, window]")
        self.alpha = alpha
        self.min_obs = min_obs
        self._residuals: deque[float] = deque(maxlen=window)

    @property
    def n_obs(self) -> int:
        return len(self._residuals)

    def record(self, realized: float, mu: float, sigma: float) -> None:
        if math.isfinite(realized) and math.isfinite(mu) and math.isfinite(sigma) and sigma > 0:
            self._residuals.append((realized - mu) / sigma)

    def _quantile(self) -> float | None:
        if len(self._residuals) < self.min_obs:
            return None
        return float(np.quantile(np.fromiter(self._residuals, dtype=np.float64), 1.0 - self.alpha))

    def var(self, mu: float, sigma: float) -> float | None:
        q = self._quantile()
        return None if q is None else float(-(mu + sigma * q))

    def es(self, mu: float, sigma: float) -> float | None:
        q = self._quantile()
        if q is None:
            return None
        z = np.fromiter(self._residuals, dtype=np.float64)
        return float(-(mu + sigma * z[z <= q].mean()))
