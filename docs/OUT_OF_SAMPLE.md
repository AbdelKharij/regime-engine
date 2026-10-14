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

*(Appended after the run.)*
