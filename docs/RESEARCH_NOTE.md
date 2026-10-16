# Hidden Markov regimes as a risk model, not an alpha model

*Research note. The code, data hashes and every number below are reproducible from this
repository (`make report`, `make oos`).*

## Abstract

I fit a four-state Gaussian hidden Markov model to daily returns, intraday range volatility
and volume volatility. It is re-estimated every 21 trading days on an expanding window, and
trading decisions use only the filtered (causal) state probabilities. On SPY (2010–2026,
after costs), using the regimes to choose *direction* (momentum, mean reversion or defensive
by regime) cuts the Sharpe ratio from 0.97 to 0.65 (bootstrap p = 0.03). Using them only to
forecast *volatility* for position sizing gives a Sharpe of 0.97 against 0.89 for buy and
hold, and roughly halves the maximum drawdown (15% vs 34%). A pre-registered test on six
assets never used in development (QQQ, IWM, EFA, TLT, GLD, BTC-USD) replicates the drawdown
reduction (6/6), the turnover advantage over realised-volatility sizing (6/6), and the
failure of regime-routed direction signals (6/6). It does **not** replicate a statistically
meaningful Sharpe improvement over buy and hold: deflated Sharpe exceeds 0.95 on 1/6 assets.

## 1. Question

Regime-switching models (Hamilton 1989) are a natural way to describe markets that alternate
between calm uptrends and turbulent selloffs. In a backtest, the tempting use is tactical:
detect the regime and switch strategies. The question here is narrower and testable. **Does
the information a causally-filtered HMM has about the current regime help a trader more as a
direction signal or as a risk signal?**

## 2. Data and features

Daily SPY bars from Yahoo Finance (split- and dividend-adjusted), 2005-01-03 to 2026-09-30,
5,470 bars. Bars are validated strictly (ordering, duplicates, NaN/inf, OHLC consistency,
extreme moves, gaps) and never repaired silently. The observation vector at bar *t*:

* r_t = ln(C_t / C_{t−1}): drift
* the Parkinson (1980) range estimator over 21 bars: the level of intraday volatility
* the 21-bar standard deviation of log-volume changes: liquidity shocks

## 3. Model

A full-covariance Gaussian HMM with K = 4 states, estimated by Baum-Welch (Rabiner 1989) with
scaled forward/backward recursions. Each fit uses 10 k-means++ restarts, a covariance ridge,
and a minimum-occupancy filter. The implementation is written from scratch and verified
against brute-force path enumeration and against `hmmlearn` (likelihood, posteriors and
Viterbi path; Baum-Welch updates to 1e-8).

**Label switching.** State indices are arbitrary and change between refits. After every
refit, states are ranked by de-scaled mean return (top two = bull) and then by volatility
(low/high). All parameters are permuted together, so state *k* always means the same
economic regime. A Hungarian assignment against the previous model cross-checks the labels.

**Causality.** The first model is fit on 2005–2010 (1,260 bars). Refits happen every 21
bars. Features are standardised with statistics from the training window only. Decisions
use the filtered P(S_t | x_1..t) after a hysteresis rule (0.6 probability for 2 consecutive
bars). Smoothed probabilities and Viterbi paths appear only in diagnostics.

## 4. Strategy, sizing and risk

* **Direction.** Either *always long* (the overlay), or *regime-routed* (time-series momentum
  in bull regimes, z-score mean reversion in bear/low-vol, a reduced long in bear/high-vol).
* **Sizing.** w = d · σ* / σ̂, with σ* = 12% annualised. Here σ̂ is the standard deviation of
  the regime mixture of per-state return distributions under the one-step-ahead
  probabilities α_t A. The no-HMM baseline replaces σ̂ with 21-day realised volatility.
* **Risk desk.** A one-day 99% VaR budget using filtered historical simulation (Barone-Adesi
  et al. 1999), a 20% drawdown circuit breaker, a reduce-only cooldown after regime flips,
  and a 1.5× leverage cap. Every intervention is logged per rule.
* **Execution.** An order decided at the close fills at the next open, with a 1 bp half
  spread, 0.5 bp commission, and square-root market impact (Almgren et al. 2005).

## 5. Verification

Backtest bugs, look-ahead above all, produce most published "alpha", so verification gets
the same weight as modelling:

* **Whole-stack look-ahead canary.** Every bar after a cutoff is corrupted and the full
  pipeline reruns. Equity, positions and cash up to the cutoff must be bit-identical. The same
  checker detects centred windows, forward-shifted labels, full-sample scaling and
  back-filled data.
* **Two independent accounting engines** (a cash/share ledger and a return-space recursion)
  agree to ~1e-15.
* **Calibrated statistics.** PSR is uniform under the null. Kupiec matches an independent
  binomial likelihood ratio.

## 6. Results on SPY (in sample for design decisions)

| | CAGR | Vol | Sharpe ± SE | Max DD | Turnover |
|---|---:|---:|---:|---:|---:|
| Overlay (HMM vol target) | 11.6% | 12.1% | 0.97 ± 0.25 | 15.4% | 3.5× |
| Realised-vol target | 11.1% | 12.1% | 0.93 ± 0.25 | 15.5% | 5.8× |
| Regime-routed strategies | 3.9% | 6.2% | 0.65 ± 0.25 | 12.4% | 13.1× |
| Buy & hold | 14.6% | 17.0% | 0.89 ± 0.25 | 33.7% | – |

**Why direction fails.** Detected regimes classify the *current* volatility state well: the
causal call agrees 83% with the ex-post smoothed one. But they carry little information about
the next month's direction: 32% agreement with a realised forward label, against 25% chance.
Detection lags the drawdown, and the rebound follows. Next-day SPY returns after a confirmed
bear/high-vol regime averaged **+38% annualised**, so strategies that de-risk directionally
in those regimes miss the recovery.

**Why sizing works.** Volatility is persistent and forecastable. Scaling exposure inversely
to forecast volatility keeps risk roughly constant. Moreira & Muir (2017) document this
"volatility-managed portfolio" effect. The HMM forecast is smoother than realised volatility,
which cuts turnover by 40% at the same Sharpe.

## 7. Out-of-sample test (pre-registered)

The protocol, hypotheses and pass/fail rules were committed *before* the run
([OUT_OF_SAMPLE.md](OUT_OF_SAMPLE.md), commit `0f2d61b`). The frozen SPY configuration then
ran once on six unseen assets.

| Hypothesis | Holds on | Verdict |
|---|---|---|
| Overlay drawdown < buy & hold | 6/6 | replicates |
| Overlay Sharpe > buy & hold | 4/6 | partially |
| Overlay turnover < realised-vol baseline | 6/6 | replicates |
| Overlay Sharpe ≥ baseline − 0.05 | 6/6 | replicates |
| Regime-routed Sharpe < overlay | 6/6 | replicates |
| Overlay deflated Sharpe > 0.95 | 1/6 | fails |

The risk claims generalise. The return claim does not: no overlay-vs-buy-and-hold Sharpe
difference is significant on any asset (p from 0.27 to 0.87). This agrees with Cederburg et
al. (2020), who find that volatility-managed portfolios rarely beat their unmanaged versions
reliably out of sample. One SPY-based design decision also failed to transfer: trailing stops
hurt on SPY, QQQ, IWM, EFA and TLT but helped on GLD and BTC-USD.

## 8. Limitations and next steps

* 16 years and seven assets is a small sample for Sharpe-ratio inference; standard errors
  are about 0.25.
* K = 4 is imposed by the 2×2 regime taxonomy; BIC prefers K = 5 on SPY.
* Single-asset portfolios. The natural extension is cross-asset: use regime probabilities to
  allocate risk between equities, bonds and gold, where the regime's information about
  *relative* volatility may matter more than its information about direction.
* Costs are calibrated heuristically (η = 0.1 impact) rather than from fills.

## References

* Almgren, R., Thum, C., Hauptmann, E., & Li, H. (2005). Direct estimation of equity market impact. *Risk*, 18(7).
* Bailey, D. H., & López de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk*, 15(2).
* Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*, 40(5).
* Barone-Adesi, G., Giannopoulos, K., & Vosper, L. (1999). VaR without correlations for portfolios of derivative securities. *Journal of Futures Markets*, 19(5).
* Cederburg, S., O'Doherty, M. S., Wang, F., & Yan, X. (2020). On the performance of volatility-managed portfolios. *Journal of Financial Economics*, 138(1).
* Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4).
* Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2).
* Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2).
* Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4).
* Moreira, A., & Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance*, 72(4).
* Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *Journal of Business*, 53(1).
* Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428).
* Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2).
