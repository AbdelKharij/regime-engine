# Regime Engine

A regime-switching trading and risk engine built from scratch in Python: a walk-forward
**4-state Gaussian hidden Markov model** (written in NumPy, checked against `hmmlearn`), a
regime-aware sizing and risk desk, an **event-driven backtester** with realistic execution,
and an ablation study with formal significance tests.

The research conclusion matters as much as the code. **On SPY, the HMM's regimes classify
volatility, not direction.** Used as a risk overlay, they beat buy-and-hold on risk-adjusted
return and halve the maximum drawdown. Used to pick direction (momentum / mean-reversion /
defensive by regime), they significantly destroy value. Both results are reported below,
with the statistics to back them.

## Headline result

SPY, daily bars. Walk-forward from **2010-02-03 to 2026-09-30** (the first model is fit on
2005–2010 data, so nothing before the first decision is out of sample). All figures are after
spread, square-root market impact and commission, with next-open fills.

| | Overlay (headline) | Realised-vol baseline | SPY buy & hold |
|---|---:|---:|---:|
| CAGR | 11.6% | 11.1% | 14.6% |
| Volatility | 12.1% | 12.1% | 17.0% |
| Sharpe (± SE) | **0.97 ± 0.25** | 0.93 ± 0.25 | 0.89 ± 0.25 |
| Sortino | 1.34 | 1.28 | 1.25 |
| Max drawdown | **15.4%** | 15.5% | 33.7% |
| Calmar | 0.75 | 0.72 | 0.43 |
| Turnover (×/yr) | 3.5 | 5.8 | – |
| Total costs | $24k | $39k | – |
| Deflated Sharpe | 99.7% (9 trials) | 99.5% | – |

**The overlay:** always long SPY, sized each day to a 12% volatility target using the HMM's
regime-mixture volatility forecast for the next bar, behind a risk desk (VaR budget,
drawdown breaker, a cooldown after each regime flip).

What the numbers do and don't say:
* It **halves the drawdown** (15% vs 34%) and has a higher Sharpe than buy-and-hold, but it
  runs at 12% vol instead of 17%, so CAGR is lower. The Sharpe difference vs buy-and-hold
  is **not statistically significant** (paired stationary bootstrap p = 0.43).
* Against plain realised-volatility targeting, the HMM's edge is **not in Sharpe** (+0.04,
  p = 0.27) but in **trading cost**: its forecast moves more smoothly, giving 40% less
  turnover and 38% lower costs.
* The Deflated Sharpe Ratio corrects for all 9 configurations tried, including the losing
  ones. The overlay's Sharpe stays above what the best of 9 zero-skill trials would reach by
  luck.

## Ablation: what each layer contributes

Each row changes one thing relative to the headline. The p-value is a paired stationary
bootstrap test (mean block 21 days) of the Sharpe difference.

| Variant | CAGR | Vol | Sharpe | Max DD | ΔSharpe | p |
|---|---:|---:|---:|---:|---:|---:|
| **Overlay (headline)** | 11.6% | 12.1% | 0.97 | 15.4% | – | – |
| Realised-vol sizing (no HMM) | 11.1% | 12.1% | 0.93 | 15.5% | −0.04 | 0.27 |
| + 0.25-Kelly cap | 9.4% | 11.1% | 0.87 | 15.8% | −0.10 | 0.18 |
| + 3-ATR trailing stops | 5.5% | 9.6% | 0.60 | 21.5% | **−0.37** | **0.02** |
| No regime-flip cooldown | 11.6% | 12.1% | 0.97 | 15.4% | +0.00 | 0.83 |
| Rolling (vs expanding) window | 10.5% | 11.5% | 0.92 | 16.9% | −0.05 | 0.47 |
| Regime-routed strategies | 3.9% | 6.2% | 0.65 | 12.4% | **−0.32** | **0.03** |
| Regime-routed, probability-blended | 4.2% | 6.3% | 0.69 | 11.8% | −0.28 | 0.05 |
| Regime-routed + stops (original design) | 2.2% | 5.2% | 0.45 | 11.5% | **−0.52** | **0.01** |
| Zero transaction costs (sensitivity) | 11.7% | 12.1% | 0.97 | 15.4% | +0.01 | – |
| SPY buy & hold | 14.6% | 17.0% | 0.89 | 33.7% | −0.09 | 0.43 |

The original design (regime-routed strategies plus stops) was the worst configuration
tested. The data showed why: next-day returns *after* a confirmed bear or high-vol regime
were higher, not lower (bear/high-vol: +30% annualised at 39% vol), because detection lags
the drawdown and the rebound follows. That evidence, and each decision it drove, is in the
[decision log](docs/ARCHITECTURE.md#13-decision--findings-log).

## What the regime model actually does

| Diagnostic (2010–2026, causal) | Result |
|---|---|
| Causal regime vs. ex-post smoothed regime (final model) | 83% agreement |
| Decoded regime vs. *realised* next-21-day regime label | 32% agreement (chance ≈ 25%) |
| Mean self-transition probability across 200 refits | 0.96–0.98 (regimes persist for weeks) |
| Share of bars / overlay weight, bull/low-vol regime | 44.6% / 1.24 |
| Share of bars / overlay weight, bear/high-vol regime | 5.9% / 0.23 |

The model detects the *current* volatility state well, but carries little information about
the next month's direction. That is why it works as a risk model and not as an alpha model.
The price is visible in the attribution table: SPY returned +38% annualised in the days after
a confirmed crash regime, while the overlay held only 23% exposure.

**VaR backtests** (one-day 99%, per unit exposure, 4,100+ days):

| Method | Breaches | Kupiec p (rate) | Christoffersen p (clustering) |
|---|---:|---:|---:|
| Filtered historical simulation (used by the risk desk) | 1.37% | 0.025 | 0.008 |
| Regime mixture (Gaussian within regime) | 2.70% | <0.001 | 0.28 |
| Historical | 1.50% | 0.002 | <0.001 |
| Parametric Gaussian | 2.67% | <0.001 | <0.001 |

Each method fails differently. The regime mixture tracks volatility clustering (its breaches
are independent) but its within-regime tails are too thin. Historical VaR has the opposite
problem. Filtered historical simulation, which scales the empirical standardised residuals
by the regime volatility forecast, comes closest on both tests. The VaR budget built on it
is the risk desk's most active rule (1,123 decision bars).

## How it works

```
OHLCV ──► DataIngestor ──► FeatureEngine ──► HMMEngine (walk-forward) ──► RegimeState(t)
          validate,         log return,        Baum-Welch refit every 21 bars
          never repair      Parkinson vol,     on an expanding window, label
                            volume-change vol  mapping, filtered P(S_t|x_1..t),
                                               hysteresis
                                                   │
          ┌────────────────────────────────────────┘
          ▼
   StrategyAgent ──► Sizer ──► RiskManager ──► ExecutionSimulator ──► Portfolio
   direction only    regime-     DD breaker,     next-open fills,       cash/shares,
   (default: long)   mixture     cooldown, VaR   spread + sqrt impact   borrow
                     vol target  budget (FHS),   + commission
                                 leverage, stops
```

* **HMM from scratch.** Scaled forward/backward recursions, Baum-Welch with k-means++
  restarts, a covariance ridge, an optional sticky Dirichlet prior, Viterbi, sampling and
  BIC. The library is verified against brute-force path enumeration and against `hmmlearn`
  (likelihood, posteriors, Viterbi path, and Baum-Welch updates after 1 and 25 iterations,
  all to 1e-8).
* **Label switching solved.** After every refit, states are ranked by de-scaled mean return
  (bull/bear) and then volatility (low/high), and **all** parameters are permuted together.
  A Hungarian-assignment cross-check and an anomaly log channel flag economically doubtful
  labellings.
* **Strictly causal.** Decisions use only filtered probabilities. The smoothed posterior and
  the Viterbi path are used for diagnostics only. Scalers are fit inside each training
  window, and fills happen no earlier than the next bar.
* **Risk desk.** Regime-mixture, historical, parametric and filtered-historical-simulation
  VaR/ES, with Kupiec and Christoffersen backtests. Also a drawdown circuit breaker,
  reduce-only cooldowns, ATR trailing stops (gap-aware, stop wins ties), and a per-rule
  audit trail on every decision.
* **Analytics.** Sharpe with a non-normal standard error, Probabilistic and Deflated Sharpe,
  paired stationary-bootstrap tests, drawdown episodes, regime confusion matrices, and
  per-regime attribution, all rendered into a single-file HTML tear sheet.

## Verification

252 tests (`pytest`), `mypy --strict` and `ruff` all pass. The tests that matter most:

* **Look-ahead canary, whole stack.** Every bar after a cutoff T is corrupted (random walk,
  shuffled, scaled), then the full pipeline reruns, from raw prices through features, HMM
  refits, strategy, risk and execution. Equity, positions and cash up to T must be
  **bit-identical**. The same checker catches centred windows, forward-shifted labels,
  full-sample scaling, back-filled weekly data and single-ULP tampering.
* **Two independent accounting engines.** The event-driven cash/share ledger and a
  return-space recursion agree to **~1e-15** over hundreds of orders, including shorts and
  borrow.
* **Statistics are calibrated.** PSR is uniform under the null (≈5% exceed 0.95). The
  expected-max-Sharpe formula matches a Monte Carlo simulation. Kupiec matches an independent
  binomial likelihood ratio. Mixture VaR matches a 1-million-draw simulation.
* **Recovery.** On 5,000 bars simulated from a known 4-regime model, EM recovers the
  per-state sample moments and decodes the true state on more than 90% of bars.

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev,yahoo]"
.venv/bin/pytest -q
.venv/bin/regime-engine backtest --config config/spy_daily.toml --out runs
.venv/bin/regime-engine report   --config config/spy_daily.toml --out reports --runs runs
```

`report` runs the full ablation (about 3 minutes: the regime path is estimated once per
distinct HMM setup and replayed across variants). It writes `reports/spy_tear_sheet.html`
and `reports/spy_ablation.csv`. Every run saves its tables plus a `manifest.json` (config
fingerprint, data hash, package versions) and a JSON-lines event log.

## Limitations

* **One asset, one market history.** 16.6 years of SPY is one bull-dominated sample.
  Standard errors on Sharpe ratios (≈0.25) are larger than most of the differences in the
  table.
* **Fixed K = 4.** It is required by the 2×2 regime taxonomy, but BIC keeps improving up to
  K = 5 on SPY.
* **Some design choices were revisited after seeing results** (the overlay headline, the
  Kelly cap off, stops off). Each change is in the decision log and was cross-checked by the
  ablation and the DSR. Still, the honest out-of-sample test is a second asset or a
  hold-out period that has never been used for any decision.
* **Execution model.** Fractional shares, cash earns no interest, impact is calibrated
  heuristically (η = 0.1), and SPY's real costs are lower than modelled.

## Layout

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design document, the maths, and
the decision & findings log (§13).
