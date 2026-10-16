# Out-of-sample protocol (pre-registered)

**Status: registered before any result on these assets was computed.** This file and the
analysis script `scripts/run_out_of_sample.py` are committed together *before* the run. The
git history is the timestamp. Results are appended below the line at the end, without
editing anything above it.

## Why

Every design decision so far was made while looking at SPY 2010–2026: expanding window,
Parkinson range volatility, risk-overlay headline, Kelly cap off, stops off (see
`ARCHITECTURE.md` §13). The SPY numbers are therefore in-sample with respect to those
choices. This protocol tests whether the conclusions hold on assets that played no part in
any decision.

## Frozen configuration

* `config/spy_daily.toml` as committed at the time of this file
  (blob `eeb5da3b7bf83884383f66d4c28f81ce7999ef2b`, config fingerprint `ab697f666cba63e2`).
* Code: the commit containing this file. No code or parameter changes after it, except
  under the bug rule below.

**Only these per-asset overrides are permitted:**

| Asset | Overrides |
|---|---|
| QQQ, IWM, EFA, TLT, GLD | `data.symbol`, `data.benchmark` = the asset |
| BTC-USD | as above, plus `run.annualization = 365` (trades every day), `data.start = 2014-09-17` (first Yahoo bar), `data.max_abs_log_return = 0.5` (data validation only: crypto has genuine daily moves above 25%) |

All bar-denominated settings (feature windows, 1260-bar first fit, 21-bar refits, cooldown)
are unchanged. Cost parameters are unchanged.

**Period:** 2005-01-01 to 2026-09-30 (BTC-USD: 2014-09-17 to 2026-09-30). As on SPY, the first
~5 years of each series only train the first model.

**Assets:**

| Asset | What it is | Why it is included |
|---|---|---|
| QQQ | Nasdaq-100 | Equity, higher beta than SPY |
| IWM | Russell 2000 | Small caps |
| EFA | Developed ex-US equities | Non-US equity |
| TLT | 20+ year Treasuries | Different asset class; its "bear" regimes are rate shocks |
| GLD | Gold | Commodity |
| BTC-USD | Bitcoin | Crypto: 24/7 trading, extreme volatility |

## Variants

The same ten variants as the SPY ablation (`analytics.experiments.DEFAULT_VARIANTS`), plus buy
and hold, run **once** per asset.

## Hypotheses and decision rules

These are the SPY claims, stated so they can fail. "Overlay" is the headline variant.

| ID | Claim | Holds on an asset if |
|---|---|---|
| H1 | The overlay reduces drawdowns | overlay max drawdown < buy-and-hold max drawdown |
| H2 | The overlay improves risk-adjusted return | overlay Sharpe > buy-and-hold Sharpe (point estimate; bootstrap p reported) |
| H3 | HMM sizing trades less than realised-vol sizing | overlay turnover < realised-vol baseline turnover |
| H4 | HMM sizing is not worse than realised-vol sizing | overlay Sharpe ≥ baseline Sharpe − 0.05 |
| H5 | Regime-routed direction signals hurt | tactical Sharpe < overlay Sharpe |
| H6 | The overlay's Sharpe survives deflation | overlay DSR > 0.95 |

**Verdict per hypothesis, across the 6 assets:**
* **replicates:** holds on ≥ 5 of 6
* **partially replicates:** holds on 3–4 of 6
* **fails to replicate:** holds on ≤ 2 of 6

All six assets are reported whatever the outcome. No asset is dropped or added after the run.

## Bug rule

If the run crashes or exposes a bug, a fix is allowed only if it leaves the committed SPY
ablation table unchanged to the last digit. The fix and that check are documented under
Results.

---

## Results

*Appended after the run, which was executed once on 2026-10-15 at commit `0f2d61b`. The
protocol above is unchanged. The bug rule was not needed: no crash and no code change.*

### Per asset (walk-forward, after costs)

| Asset | Active period | Overlay CAGR | B&H CAGR | Overlay Sharpe | Baseline Sharpe | Tactical Sharpe | B&H Sharpe | Overlay vs B&H p | Overlay MDD | B&H MDD | Turnover overlay / baseline | Overlay DSR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QQQ | 2010-02 → 2026-09 | 12.7% | 19.8% | 1.02 | 1.02 | 0.67 | 0.98 | 0.58 | 18.2% | 35.1% | 2.3× / 4.7× | 99.8% |
| IWM | 2010-02 → 2026-09 | 5.7% | 11.3% | 0.53 | 0.43 | 0.36 | 0.59 | 0.48 | 20.3% | 41.1% | 1.7× / 4.4× | 93.0% |
| EFA | 2010-02 → 2026-09 | 3.9% | 7.5% | 0.37 | 0.34 | 0.05 | 0.49 | 0.27 | 22.2% | 34.2% | 2.5× / 5.7× | 66.2% |
| TLT | 2010-02 → 2026-09 | 2.7% | 1.9% | 0.28 | 0.26 | 0.09 | 0.20 | 0.32 | 37.7% | 48.4% | 2.1× / 6.1× | 71.9% |
| GLD | 2010-02 → 2026-09 | 6.9% | 8.1% | 0.61 | 0.58 | 0.42 | 0.55 | 0.30 | 39.0% | 45.6% | 2.1× / 5.9× | 93.0% |
| BTC-USD | 2018-03 → 2026-09 | 8.6% | 30.0% | 0.76 | 0.60 | 0.36 | 0.74 | 0.87 | 26.4% | 76.6% | 2.0× / 2.7× | 89.9% |

Data hashes for each asset are recorded in `reports/oos/summary.csv`. Each asset also has a
tear sheet and an ablation CSV in `reports/oos/`.

### Verdicts (pre-registered rules)

| ID | Hypothesis | Holds on | Verdict |
|---|---|---|---|
| H1 | overlay max drawdown < buy & hold max drawdown | 6/6 | **replicates** |
| H2 | overlay Sharpe > buy & hold Sharpe | 4/6 (not IWM, EFA) | **partially replicates** |
| H3 | overlay turnover < realised-vol baseline turnover | 6/6 | **replicates** |
| H4 | overlay Sharpe ≥ realised-vol baseline Sharpe − 0.05 | 6/6 | **replicates** |
| H5 | tactical Sharpe < overlay Sharpe | 6/6 | **replicates** |
| H6 | overlay deflated Sharpe > 0.95 | 1/6 (QQQ) | **fails to replicate** |

### Interpretation

* **What generalises:**
  * **The drawdown reduction.** Max drawdown is lower on every asset, with reductions between
    7 points (GLD) and 50 points (BTC).
  * **HMM sizing is cheaper to trade.** It needs 26–65% less turnover than realised-vol sizing.
  * **Regime-routed direction signals lose.** They underperform the overlay on all six assets.
* **What does not:**
  * **No significant risk-adjusted outperformance.** No overlay-vs-buy-and-hold Sharpe
    difference is significant (p from 0.27 to 0.87), and the point estimate is worse on IWM
    and EFA.
  * **The SPY deflated Sharpe does not carry over.** Except on QQQ, the overlay's Sharpe
    cannot be distinguished from the best of nine zero-skill trials (H6). On SPY it could
    (DSR 99.7%). That result was helped by SPY's strong 2010–2026 run and should not be read
    as general.
* **The cost of the overlay is lower CAGR.** It runs at about 12% volatility, so on
    high-return assets (QQQ, BTC) it gives up a lot of return in exchange for the shallower
    drawdowns.

### Exploratory observations (not pre-registered, so treat as hypotheses)

* HMM sizing beat realised-vol sizing on Sharpe on 5 of 6 assets, with a tie on QQQ. On SPY
  it was also ahead. No single difference is significant (p from 0.08 to 1.0).
* **Stops are asset-dependent.** 3-ATR stops hurt on QQQ, IWM, EFA and TLT, as on SPY, but
  *helped* on GLD (Sharpe 0.89 vs 0.61) and BTC-USD (1.06 vs 0.76). The SPY-based decision to
  turn stops off by default does not generalise to trending, high-volatility assets.
* The 0.25-Kelly cap lowered Sharpe on 5 of 6 assets, consistent with SPY.
