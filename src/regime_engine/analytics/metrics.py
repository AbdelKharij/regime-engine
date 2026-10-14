"""Performance metrics and their statistical significance.

Conventions: ``returns`` are simple per-bar returns, ``periods`` bars per year (252 for daily
equities, 365 for crypto), and the risk-free rate is annual. Ratios are annualised unless the
name says otherwise.

Significance
------------
* Sharpe standard error (Lo 2002, Mertens 2002, non-normal returns):
  SE(SR) = sqrt((1 - g3 SR + (g4 - 1)/4 SR^2) / (n - 1)) per period.
* Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012): P(true SR > SR*) =
  Phi((SR - SR*) sqrt(n - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2)).
* Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): PSR against the Sharpe ratio that the
  best of N independent *unskilled* trials would be expected to reach,
  SR* = sqrt(V) ((1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e))). This corrects for
  having tried several configurations.
* Paired stationary-bootstrap test of a Sharpe-ratio difference (Politis & Romano 1994),
  which keeps serial dependence and the cross-correlation between the two strategies.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import kurtosis, norm, skew

FloatArray = npt.NDArray[np.float64]
_EULER_GAMMA = 0.5772156649015329


def _degenerate(std: float, scale: float) -> bool:
    """True when a standard deviation is zero up to floating-point noise."""
    return not std > 1e-12 * max(1.0, abs(scale))


def _clean(returns: pd.Series | FloatArray) -> FloatArray:
    arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    return np.asarray(arr[np.isfinite(arr)], dtype=np.float64)


# --------------------------------------------------------------------------------------------
# Point metrics
# --------------------------------------------------------------------------------------------


def cagr(equity: pd.Series, periods: int = 252) -> float:
    n = len(equity) - 1
    if n <= 0 or equity.iloc[0] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (periods / n) - 1.0)


def annualized_volatility(returns: pd.Series | FloatArray, periods: int = 252) -> float:
    r = _clean(returns)
    return float(r.std(ddof=1) * math.sqrt(periods)) if len(r) > 1 else float("nan")


def sharpe_ratio(
    returns: pd.Series | FloatArray, periods: int = 252, risk_free_annual: float = 0.0
) -> float:
    r = _clean(returns) - risk_free_annual / periods
    if len(r) < 2 or _degenerate(float(r.std(ddof=1)), float(np.abs(r).max())):
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods))


def sortino_ratio(
    returns: pd.Series | FloatArray, periods: int = 252, target_annual: float = 0.0
) -> float:
    r = _clean(returns) - target_annual / periods
    downside = math.sqrt(float(np.mean(np.minimum(r, 0.0) ** 2))) if len(r) else 0.0
    if downside == 0:
        return float("nan")
    return float(r.mean() / downside * math.sqrt(periods))


def information_ratio(
    returns: pd.Series, benchmark_returns: pd.Series, periods: int = 252
) -> tuple[float, float]:
    """(information ratio, annualised tracking error) of ``returns`` against a benchmark."""
    active = (returns - benchmark_returns).dropna().to_numpy(dtype=np.float64)
    if len(active) < 2 or _degenerate(float(active.std(ddof=1)), float(np.abs(active).max())):
        return float("nan"), float("nan")
    te = float(active.std(ddof=1) * math.sqrt(periods))
    return float(active.mean() * periods / te), te


def beta_alpha(
    returns: pd.Series, benchmark_returns: pd.Series, periods: int = 252
) -> tuple[float, float]:
    """(beta, annualised alpha) from an OLS regression on the benchmark."""
    joined = pd.concat([returns, benchmark_returns], axis=1).dropna().to_numpy(dtype=np.float64)
    if len(joined) < 3:
        return float("nan"), float("nan")
    r, b = joined[:, 0], joined[:, 1]
    beta = float(np.cov(r, b, ddof=1)[0, 1] / b.var(ddof=1))
    return beta, float((r.mean() - beta * b.mean()) * periods)


@dataclass(frozen=True, slots=True)
class DrawdownEpisode:
    peak: pd.Timestamp
    trough: pd.Timestamp
    recovery: pd.Timestamp | None  # None = not yet recovered
    depth: float  # positive fraction
    bars_to_trough: int
    duration_bars: int  # peak to recovery (or to the end of the sample if ongoing)


def drawdown_series(equity: pd.Series) -> pd.Series:
    return (1.0 - equity / equity.cummax()).rename("drawdown")


def drawdown_episodes(equity: pd.Series) -> list[DrawdownEpisode]:
    """Every peak-to-recovery episode, deepest first."""
    values = equity.to_numpy(dtype=np.float64)
    index = equity.index
    episodes: list[DrawdownEpisode] = []
    peak_i = 0
    i = 1
    n = len(values)
    while i < n:
        if values[i] >= values[peak_i]:
            peak_i = i
            i += 1
            continue
        start = peak_i
        trough_i = i
        while i < n and values[i] < values[start]:
            if values[i] < values[trough_i]:
                trough_i = i
            i += 1
        recovered = i < n
        end_i = i if recovered else n - 1
        episodes.append(
            DrawdownEpisode(
                peak=index[start],
                trough=index[trough_i],
                recovery=index[end_i] if recovered else None,
                depth=float(1.0 - values[trough_i] / values[start]),
                bars_to_trough=trough_i - start,
                duration_bars=end_i - start,
            )
        )
        if recovered:
            peak_i = i
            i += 1
    return sorted(episodes, key=lambda e: e.depth, reverse=True)


# --------------------------------------------------------------------------------------------
# Significance
# --------------------------------------------------------------------------------------------


def _moments(r: FloatArray) -> tuple[float, float, float]:
    """Per-period Sharpe ratio, skewness, and (non-excess) kurtosis."""
    return (
        float(r.mean() / r.std(ddof=1)),
        float(skew(r, bias=False)),
        float(kurtosis(r, fisher=False, bias=False)),
    )


def sharpe_standard_error(returns: pd.Series | FloatArray, periods: int = 252) -> float:
    """Annualised standard error of the Sharpe ratio under non-normal returns."""
    r = _clean(returns)
    sr, g3, g4 = _moments(r)
    return float(
        math.sqrt((1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2) / (len(r) - 1)) * math.sqrt(periods)
    )


def probabilistic_sharpe_ratio(
    returns: pd.Series | FloatArray, sr_benchmark_annual: float = 0.0, periods: int = 252
) -> float:
    r = _clean(returns)
    sr, g3, g4 = _moments(r)
    sr_star = sr_benchmark_annual / math.sqrt(periods)
    denominator = math.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2, 1e-12))
    return float(norm.cdf((sr - sr_star) * math.sqrt(len(r) - 1) / denominator))


def expected_max_sharpe(n_trials: int, sr_variance_annual: float, periods: int = 252) -> float:
    """Annualised Sharpe the best of ``n_trials`` zero-skill trials is expected to reach."""
    if n_trials < 2:
        return 0.0
    sd = math.sqrt(sr_variance_annual)
    z = (1.0 - _EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_trials) + _EULER_GAMMA * norm.ppf(
        1.0 - 1.0 / (n_trials * math.e)
    )
    return float(sd * z)


def deflated_sharpe_ratio(
    returns: pd.Series | FloatArray,
    trial_sharpes_annual: Sequence[float],
    periods: int = 252,
) -> float:
    """PSR against the expected maximum Sharpe of the trials actually run."""
    sharpes = np.asarray([s for s in trial_sharpes_annual if math.isfinite(s)], dtype=np.float64)
    if len(sharpes) < 2:
        return probabilistic_sharpe_ratio(returns, 0.0, periods)
    threshold = expected_max_sharpe(len(sharpes), float(sharpes.var(ddof=1)), periods)
    return probabilistic_sharpe_ratio(returns, threshold, periods)


@dataclass(frozen=True, slots=True)
class SharpeDifference:
    difference: float  # annualised SR(a) - SR(b)
    ci_low: float
    ci_high: float
    p_value: float  # two-sided, H0: equal Sharpe ratios
    n_bootstrap: int
    mean_block: int


def _stationary_bootstrap_indices(
    n: int, n_boot: int, mean_block: int, rng: np.random.Generator
) -> npt.NDArray[np.int64]:
    restart = rng.random((n_boot, n)) < 1.0 / mean_block
    starts = rng.integers(0, n, size=(n_boot, n))
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(restart[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap_sharpe_difference(
    returns_a: pd.Series,
    returns_b: pd.Series,
    *,
    periods: int = 252,
    n_bootstrap: int = 2000,
    mean_block: int = 21,
    confidence: float = 0.95,
    seed: int = 0,
) -> SharpeDifference:
    """Paired stationary-bootstrap test of SR(a) - SR(b) on their common dates."""
    joined = pd.concat([returns_a, returns_b], axis=1).dropna().to_numpy(dtype=np.float64)
    a, b = joined[:, 0], joined[:, 1]
    observed = sharpe_ratio(a, periods) - sharpe_ratio(b, periods)
    idx = _stationary_bootstrap_indices(
        len(a), n_bootstrap, mean_block, np.random.default_rng(seed)
    )
    ra, rb = a[idx], b[idx]
    sr_a = ra.mean(axis=1) / ra.std(axis=1, ddof=1)
    sr_b = rb.mean(axis=1) / rb.std(axis=1, ddof=1)
    diffs = (sr_a - sr_b) * math.sqrt(periods)
    tail = (1.0 - confidence) / 2.0
    p_value = float(np.mean(np.abs(diffs - diffs.mean()) >= abs(observed)))
    return SharpeDifference(
        difference=observed,
        ci_low=float(np.quantile(diffs, tail)),
        ci_high=float(np.quantile(diffs, 1.0 - tail)),
        p_value=p_value,
        n_bootstrap=n_bootstrap,
        mean_block=mean_block,
    )


# --------------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    start: pd.Timestamp
    end: pd.Timestamp
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sharpe_se: float
    psr: float  # P(true Sharpe > 0)
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_duration: int  # bars, peak to recovery (or to end if ongoing)
    skewness: float
    excess_kurtosis: float
    hit_rate: float
    information_ratio: float | None
    tracking_error: float | None
    beta: float | None
    alpha: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize(
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    *,
    periods: int = 252,
    risk_free_annual: float = 0.0,
) -> PerformanceSummary:
    returns = equity.pct_change().dropna()
    r: FloatArray = np.asarray(returns.to_numpy(), dtype=np.float64)
    episodes = drawdown_episodes(equity)
    mdd = episodes[0].depth if episodes else 0.0
    growth = cagr(equity, periods)
    ir = te = beta = alpha = None
    if benchmark is not None:
        bench_returns = benchmark.reindex(equity.index).pct_change().dropna()
        ir, te = information_ratio(returns, bench_returns, periods)
        beta, alpha = beta_alpha(returns, bench_returns, periods)
    return PerformanceSummary(
        start=equity.index[0],
        end=equity.index[-1],
        total_return=float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        cagr=growth,
        volatility=annualized_volatility(r, periods),
        sharpe=sharpe_ratio(r, periods, risk_free_annual),
        sharpe_se=sharpe_standard_error(r, periods),
        psr=probabilistic_sharpe_ratio(r, 0.0, periods),
        sortino=sortino_ratio(r, periods),
        calmar=growth / mdd if mdd > 0 else float("nan"),
        max_drawdown=mdd,
        max_drawdown_duration=max((e.duration_bars for e in episodes), default=0),
        skewness=float(skew(r, bias=False)),
        excess_kurtosis=float(kurtosis(r, fisher=True, bias=False)),
        hit_rate=float((r > 0).mean()),
        information_ratio=ir,
        tracking_error=te,
        beta=beta,
        alpha=alpha,
    )
