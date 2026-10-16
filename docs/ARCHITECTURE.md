# Regime-Switching Trading & Risk Engine — System Architecture

**Status:** Draft v0.1, for review before implementation
**Scope:** Single-asset (SPY daily by default; BTC-USD through the same interface), 4-state Gaussian HMM, event-driven backtester, automated risk desk, performance tear sheet.

---

## 0. Design Principles

1. **Causality comes first.** Every number used to make a decision at bar `t` must be computable from information available at the close of bar `t`. Section 3.6 lists the specific ways an HMM backtest can leak future data and how each is blocked.
2. **Alpha, sizing, risk, and execution are kept separate.** Strategies only say *which direction and how strongly*. Sizing says *how much*. Risk says *whether it's allowed*. Execution says *what it actually cost*. Each layer can be tested on its own and swapped out.
3. **Functional core, imperative shell.** All math (HMM, features, metrics, VaR) is in pure, stateless functions or immutable model objects. All state (portfolio, cooldown timers, pending orders) is in a small number of explicit stateful classes driven by the backtest loop.
4. **Protocols over inheritance.** Components depend on `typing.Protocol` interfaces, so a test double or an alternative implementation (e.g. `hmmlearn` instead of the NumPy HMM) can be injected without subclassing.
5. **Reproducibility.** Every run writes a manifest with the config hash, data hash, seed, and package versions.
6. **Honest evaluation.** Report deflated Sharpe, costs, and regime-detection lag. A regime model usually wins on drawdown and risk-adjusted return, not on raw return against SPY buy-and-hold. The tear sheet should show that plainly.

---

## 1. Folder Structure (as built)

```
regime-engine/
├── pyproject.toml                 # deps, `regime-engine` CLI entry point, ruff / mypy --strict / pytest
├── README.md
├── config/spy_daily.toml          # headline experiment; unknown keys are rejected
├── docs/ARCHITECTURE.md           # this document (§13 = decision & findings log)
├── src/regime_engine/
│   ├── cli.py                     # `regime-engine backtest|report --config ...`
│   ├── core/                      # config.py (frozen dataclasses + strict TOML loader), types.py, exceptions.py
│   ├── utils/logging.py           # JSON-lines + console logging, log_context, regime-ordering anomaly channel
│   ├── data/
│   │   ├── sources.py             # CSV / yfinance / in-memory sources
│   │   ├── ingestor.py            # DataIngestor + OHLCVValidator (never repairs silently)
│   │   ├── lookahead.py           # LookAheadSanityChecker (corrupt the future, demand bit-identical past)
│   │   └── features.py            # FeatureEngine (log ret, Parkinson vol, volume-change vol), RollingWindowScaler
│   ├── models/
│   │   ├── gaussian_hmm.py        # NumPy Baum-Welch, scaled forward/backward, Viterbi, sampling, BIC
│   │   ├── regime_mapper.py       # label switching → canonical regimes (+ Hungarian cross-check)
│   │   └── hmm_engine.py          # walk-forward HMMEngine, hysteresis, RegimeTrack (replayable regime path)
│   ├── strategy/                  # direction-only strategies + regime router (default: always long)
│   ├── risk/                      # sizing (regime vol target, Kelly cap), VaR/ES/FHS + backtests, rules, ATR stops, RiskManager
│   ├── execution/                 # cost model (spread + sqrt impact + commission), latency queue
│   ├── portfolio/                 # cash/position ledger, borrow accrual
│   ├── backtest/engine.py         # event-driven Backtester + independent return-space cross-check
│   └── analytics/                 # metrics & significance (PSR/DSR/bootstrap), diagnostics, ablation, HTML tear sheet
└── tests/                         # one file per layer: unit, property, hmmlearn parity, causality canaries
```

**Dependencies:** `numpy`, `pandas`, `scipy`, `pydantic>=2`, `pyyaml`, `yfinance`, `pyarrow`, `matplotlib`, `jinja2`. Dev: `pytest`, `hypothesis`, `mypy`, `ruff`, `hmmlearn` (used **only** as a test oracle, see §3.5). Python ≥ 3.11.

---

## 2. Data Schemas

All time series use a **tz-aware UTC `DatetimeIndex`** that is monotonic increasing and unique. Bar timestamps mark the *close* of the bar.

### 2.1 Raw OHLCV (`DataIngestor` output)

| column      | dtype   | constraint |
|-------------|---------|------------|
| `open`      | float64 | > 0 |
| `high`      | float64 | ≥ max(open, close) |
| `low`       | float64 | ≤ min(open, close), > 0 |
| `close`     | float64 | > 0 |
| `adj_close` | float64 | > 0 (split/dividend-adjusted) |
| `volume`    | float64 | ≥ 0; = 0 flagged |

OHLC is rescaled by `adj_close / close` so returns and ranges are corporate-action consistent.

**Validation (`data/validation.py`) raises `DataValidationError` on:** NaN/inf in any column, a non-monotonic or duplicate index, OHLC inconsistency, a single-bar |log return| > configurable threshold (default 25% for SPY) unless allowlisted, or a calendar gap > N sessions. Benchmark and asset frames go through an explicit **inner-join alignment check**. Mismatched indexes are reported, never silently forward-filled.

### 2.2 Feature Frame (`FeatureEngineer` output)

| column       | definition (see §3.1) |
|--------------|------------|
| `log_ret`    | `r_t` |
| `hl_vol`     | Parkinson range volatility, rolling `w` |
| `vol_of_vol` | rolling std of log-volume changes, rolling `w` |

Warm-up rows (the first `w`) are **dropped, not filled**. The frame keeps a `feature_ts` attribute equal to the raw index so alignment can be asserted.

### 2.3 Domain Types (`types.py`, all `@dataclass(frozen=True, slots=True)`)

```python
class Regime(IntEnum):
    BULL_LOW_VOL = 0
    BULL_HIGH_VOL = 1
    BEAR_LOW_VOL = 2
    BEAR_HIGH_VOL = 3      # "Crash"

class Side(IntEnum):
    BUY = 1
    SELL = -1

@dataclass(frozen=True, slots=True)
class RegimeState:
    ts: pd.Timestamp
    probs: npt.NDArray[np.float64]      # shape (4,), filtered P(S_t=k | x_1:t), canonical order
    regime: Regime                       # confirmed regime (after hysteresis)
    raw_argmax: Regime                   # argmax before hysteresis
    confidence: float                    # max(probs)
    changed: bool                        # confirmed regime differs from previous bar

@dataclass(frozen=True, slots=True)
class Signal:
    ts: pd.Timestamp
    direction: float                     # in [-1, 1]: strategy conviction, NOT size
    strategy: str
    regime: Regime

@dataclass(frozen=True, slots=True)
class TargetPosition:
    ts: pd.Timestamp
    weight: float                        # signed fraction of equity (leverage allowed)
    rationale: str

@dataclass(frozen=True, slots=True)
class RiskDecision:
    ts: pd.Timestamp
    requested: TargetPosition
    approved: TargetPosition
    rule_trace: tuple[tuple[str, str], ...]   # (rule_name, action) audit trail

@dataclass(frozen=True, slots=True)
class Order:
    order_id: str
    created_ts: pd.Timestamp
    eligible_ts: pd.Timestamp            # created_ts + latency bars
    side: Side
    quantity: float                      # > 0
    reason: str                          # "rebalance" | "stop" | "target" | "breaker"

@dataclass(frozen=True, slots=True)
class Fill:
    order_id: str
    ts: pd.Timestamp
    side: Side
    quantity: float
    ref_price: float                     # open of fill bar (or stop level)
    fill_price: float                    # after spread + impact
    commission: float
    slippage_cost: float

@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    ts: pd.Timestamp
    cash: float
    position_qty: float
    price: float
    equity: float
    gross_exposure: float
    weight: float
    drawdown: float
```

### 2.4 Configuration (`config.py`, Pydantic, frozen, loaded from YAML)

```yaml
run:        { seed: 42, run_dir: runs/, annualization: 252 }
data:       { symbol: SPY, benchmark: SPY, start: 2005-01-01, end: 2026-09-30, source: yfinance }
features:   { window: 21 }
hmm:        { n_states: 4, covariance: full, n_restarts: 10, max_iter: 500, tol: 1.0e-6,
              train_window: 1260, refit_every: 21, window_type: rolling, min_state_occupancy: 0.03 }
regime:     { confirm_prob: 0.60, confirm_bars: 2 }
strategy:   { bull_low: trend, bull_high: trend, bear_low: mean_reversion, bear_high: defensive }
sizing:     { method: vol_target, target_vol_annual: 0.12, kelly_fraction: 0.25, max_leverage: 1.5 }
risk:       { cooldown_bars: 2, cooldown_mode: reduce_only, var_alpha: 0.99, var_window: 252,
              var_budget: 0.025, dd_breaker: 0.20, dd_halt_bars: 21,
              trailing_stop_atr: 3.0, profit_target_atr: null }
execution:  { latency_bars: 1, commission_bps: 0.5, min_commission: 1.0, half_spread_bps: 1.0,
              impact_eta: 0.1, borrow_bps_annual: 50 }
```

"48 hours" is expressed as `cooldown_bars`: 2 bars on daily data, or 48 on hourly crypto data. Everything is bar-denominated so the engine doesn't depend on bar frequency.

---

## 3. Mathematical Formulation

### 3.1 Features (all causal, using bars ≤ t)

**Log return (drift):**
$$r_t = \ln\frac{C_t}{C_{t-1}}$$

**High-Low range volatility (Parkinson, 1980):** more efficient than close-to-close for intraday variance (~5× for a driftless diffusion).
$$\sigma^{HL}_t = \sqrt{\frac{1}{4\ln 2}\cdot\frac{1}{w}\sum_{i=0}^{w-1}\left(\ln\frac{H_{t-i}}{L_{t-i}}\right)^2}$$

**Realized volume volatility (liquidity shocks):**
$$u_t = \ln V_t - \ln V_{t-1},\qquad \sigma^{V}_t = \operatorname{std}\left(u_{t-w+1},\dots,u_t\right)$$

Feature vector $x_t = (r_t,\ \ln\sigma^{HL}_t,\ \sigma^V_t)^\top \in \mathbb{R}^3$. The log of range vol makes its emission distribution closer to Gaussian, which matters because the emission model is Gaussian.

**Standardization:** $\tilde{x}_t = (x_t - \hat\mu_{\text{train}}) / \hat\sigma_{\text{train}}$, where the moments come **only from the current training window**. The scaler is refit with the HMM and versioned with it.

### 3.2 Gaussian HMM

Hidden state $S_t \in \{1,\dots,K\}$, $K = 4$. Parameters $\theta = (\pi, A, \{\mu_k, \Sigma_k\})$:

- Initial distribution: $\pi_k = P(S_1 = k)$
- Transition matrix: $A_{ij} = P(S_{t+1}=j \mid S_t=i)$, rows sum to 1
- Emissions: $b_k(x_t) = \mathcal{N}(x_t;\ \mu_k, \Sigma_k) = (2\pi)^{-d/2}|\Sigma_k|^{-1/2}\exp\!\left(-\tfrac12 (x_t-\mu_k)^\top\Sigma_k^{-1}(x_t-\mu_k)\right)$

Joint likelihood:
$$p(x_{1:T}, s_{1:T}\mid\theta) = \pi_{s_1} b_{s_1}(x_1)\prod_{t=2}^{T} A_{s_{t-1}s_t}\, b_{s_t}(x_t)$$

**Forward recursion (scaled, to avoid underflow):**
$$\hat\alpha_1(k) = \frac{\pi_k b_k(x_1)}{c_1},\qquad \hat\alpha_t(j) = \frac{b_j(x_t)\sum_i \hat\alpha_{t-1}(i)A_{ij}}{c_t}$$
where $c_t$ normalizes $\hat\alpha_t$ to sum to 1. Then $\hat\alpha_t(k) = P(S_t = k \mid x_{1:t})$ is **the filtered probability, the only regime estimate the strategy is allowed to see.** Log-likelihood $\ell(\theta) = \sum_t \ln c_t$.

**Backward recursion** (training only): $\hat\beta_T(i)=1$, $\hat\beta_t(i) = \frac{1}{c_{t+1}}\sum_j A_{ij} b_j(x_{t+1})\hat\beta_{t+1}(j)$.

**Smoothed posteriors** (training only, they use the future):
$$\gamma_t(k) = \hat\alpha_t(k)\hat\beta_t(k),\qquad \xi_t(i,j) = \frac{\hat\alpha_t(i)A_{ij}b_j(x_{t+1})\hat\beta_{t+1}(j)}{c_{t+1}}$$

**Baum-Welch M-step:**
$$\pi_k = \gamma_1(k),\quad A_{ij} = \frac{\sum_{t=1}^{T-1}\xi_t(i,j)}{\sum_{t=1}^{T-1}\gamma_t(i)},\quad \mu_k = \frac{\sum_t\gamma_t(k)x_t}{\sum_t\gamma_t(k)},\quad \Sigma_k = \frac{\sum_t\gamma_t(k)(x_t-\mu_k)(x_t-\mu_k)^\top}{\sum_t\gamma_t(k)} + \epsilon I$$

The $\epsilon I$ ridge (default $10^{-6}$) prevents covariance collapse onto a few outlier bars, which is a real failure mode around crash days. An optional Dirichlet prior on rows of $A$ (MAP-EM) adds diagonal mass and encourages regime persistence:
$$A_{ij} \propto \sum_t \xi_t(i,j) + (\kappa_{ij} - 1),\qquad \kappa_{ii} > \kappa_{i\neq j}$$

**Initialization and restarts:** k-means++ on $\tilde x$ for $\mu_k$, pooled covariance for $\Sigma_k$, diagonal-heavy $A$ (0.9 on the diagonal). Run `n_restarts` random seeds and keep $\arg\max_\theta \ell(\theta)$. Reject any fit where some state's expected occupancy $\frac1T\sum_t\gamma_t(k) <$ `min_state_occupancy`.

**Model selection evidence:** BIC is reported for $K\in\{2,3,4,5\}$ to justify, or challenge, the fixed $K=4$:
$$\text{BIC} = -2\ell + p\ln T,\qquad p = (K-1) + K(K-1) + Kd + K\tfrac{d(d+1)}{2}$$
For $K=4, d=3$: $p = 3 + 12 + 12 + 24 = 51$.

**One-step regime forecast:** $P(S_{t+1}=j\mid x_{1:t}) = \sum_i \hat\alpha_t(i)A_{ij}$.
**Expected regime duration:** $\mathbb{E}[D_k] = 1/(1-A_{kk})$, reported per regime.

**Viterbi** (most likely full path, ex-post diagnostics only):
$$\delta_t(j) = \max_i\left[\delta_{t-1}(i) + \ln A_{ij}\right] + \ln b_j(x_t)$$

### 3.3 Regime Labelling: solving label switching

HMM state indices are arbitrary and **change between refits**. `RegimeMapper` produces a permutation $\rho$ from raw state index to canonical `Regime`:

1. Compute each state's mean return $m_k$ and volatility level $v_k$ in **original (unscaled) units**: $m_k = \mu_k^{(r)}\hat\sigma_r + \hat\mu_r$, and $v_k$ from the `hl_vol` component (equivalently $\sqrt{\Sigma_k^{(r,r)}}$; configurable).
2. Sort states by $m_k$: the top two are **Bull**, the bottom two are **Bear**.
3. Within each pair, the lower $v_k$ is **Low-Vol** and the higher is **High-Vol**.
4. Build the permutation matrix $P$ and reorder **all** parameters into canonical order:
$$\pi' = P\pi,\quad A' = PAP^\top,\quad \mu'_k = \mu_{\rho^{-1}(k)},\quad \Sigma'_k = \Sigma_{\rho^{-1}(k)}$$

The transition matrix isn't "sorted" on its own. It's permuted *jointly* with the emission parameters, so $A'_{ij}$ always means "P(BULL_LOW → BEAR_HIGH)" and so on, across every refit.

**Degeneracy guards:** if both "bull" states have $m_k < 0$, or the vol split within a pair is below a tolerance, the mapper still returns a mapping but logs a `REGIME_MAPPING_AMBIGUOUS` warning with the moments. The diagnostics report counts these.

**Cross-refit stability (optional):** match new states to the previous model's canonical means with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) on the Mahalanobis distance. Rank-sorting and Hungarian matching are logged when they disagree.

### 3.4 Regime Confirmation (hysteresis)

A raw argmax flicker shouldn't start a regime change. The confirmed regime moves from $R$ to $R'$ only when
$$\hat\alpha_t(R') \ge p_{\text{confirm}}\ \text{for}\ n_{\text{confirm}}\ \text{consecutive bars.}$$
A *confirmed* change is what triggers the cooldown (§5.1). This is separate from the cooldown: hysteresis filters noise in **detection**, while cooldown limits **trading** after a real detection.

### 3.5 Implementation choice: custom NumPy HMM, `hmmlearn` as oracle

`hmmlearn.predict_proba` returns **smoothed** posteriors (forward-backward), which use the future, and it has no incremental single-step filter API. The strategy needs $\hat\alpha_t$ updated one bar at a time. So:

- `GaussianHMM` is implemented from scratch in NumPy (log-space emissions via Cholesky, scaled forward/backward, vectorized E-step).
- The test suite checks it against `hmmlearn` on the same initial parameters: log-likelihood matches to `1e-6`, and M-step parameters match after one iteration.
- A synthetic-recovery test samples 5,000 bars from a known 4-state HMM and asserts that the fitted parameters (after mapping) recover the truth within tolerance.

This gives the project full ownership of the code plus independent correctness evidence.

### 3.6 Causality Contract: look-ahead vectors and mitigations

| Vector | Mitigation |
|---|---|
| Smoothed / Viterbi states used for trading | Strategy only receives `RegimeState.probs` = forward-filtered $\hat\alpha_t$ |
| HMM fit on the full sample | Walk-forward: fit on $[t-W, t]$, then filter forward until the next refit |
| Scaler fit on the full sample | `CausalScaler` is fit inside each training window |
| Fill at the same close as the signal | `latency_bars ≥ 1`: signal at close $t$, fill at open $t+1$ |
| Stops filled at the stop price through gaps | Gap-aware: fill at `min(stop, open)` for longs |
| Benchmark misalignment | Inner-join plus assertion; no forward-fill |
| Adjusted-price revisions | Raw cached snapshot is hashed into the run manifest |

**Look-ahead canary test (integration):** run the backtest, then corrupt every bar after $t^*$ (randomize prices) and rerun. All `RiskDecision`s and `Fill`s with `ts ≤ t*` must be **bit-identical**. If anything leaks, this test fails.

---

## 4. Trading Strategy Layer

Strategies output a `Signal.direction ∈ [-1, 1]` (conviction) and **never** a position size.

| Regime | Default strategy | Direction logic |
|---|---|---|
| BULL_LOW_VOL | `TimeSeriesMomentum` | $\text{sign}$ / $\tanh$ of blended 21/63/252-bar trailing returns, scaled by their vol; long-only clip $[0,1]$ |
| BULL_HIGH_VOL | `TimeSeriesMomentum` (dampened) | Same signal × 0.5; long-only |
| BEAR_LOW_VOL | `ZScoreMeanReversion` | $-\tanh(z_t/2)$, $z_t = (C_t - \text{SMA}_{20})/\text{std}_{20}$; symmetric |
| BEAR_HIGH_VOL | `DefensiveShort` | $-0.5$ (synthetic short with borrow cost) or $0$ (flat), configurable |

**A note on two items in the original spec:**
- *"Inverse tracking" short via an inverse ETF (e.g. SH):* daily-rebalanced inverse ETFs suffer volatility decay, which is worst in exactly the high-vol regimes where they'd be used. The engine models a **direct short with borrow cost** instead, which is cleaner and more realistic for a fund.
- *"Market-neutral" mean reversion:* with a single traded asset nothing is truly market-neutral. The honest single-asset version is a small symmetric mean-reversion book with low gross exposure. A real market-neutral sleeve needs a second instrument (see open question 2).

**StrategyAgent** routes by *confirmed* regime by default. An optional **probability-blended** mode computes $\text{direction} = \sum_k \hat\alpha_t(k)\cdot d_k$, which turns transitions into smooth changes rather than hard switches. This mode is a good ablation for the write-up.

---

## 5. Risk Desk

### 5.1 Regime-Flip Cooldown

When a confirmed regime change happens at bar $t$, the cooldown is active for bars $t, \dots, t + N - 1$.

`cooldown_mode` options:
- `freeze`: no orders at all (the literal spec).
- `reduce_only` **(default)**: orders that **reduce** $|w|$ are allowed; orders that increase risk are blocked.

**Rationale:** a pure freeze right after a decoded flip *into* BEAR_HIGH_VOL would lock in the old bull-sized position during the crash. Whipsaw protection should stop the engine from *adding* risk on a noisy signal, not stop it from cutting risk. Stops and the drawdown breaker are **always** active and ignore the cooldown.

### 5.2 Dynamic Volatility Sizing

The regime-conditional return distribution is a Gaussian mixture weighted by filtered probabilities. Using the return-component moments $(m_k, s_k^2)$ of each state in original units:
$$\mu_{\text{mix}} = \sum_k \hat\alpha_t(k)\, m_k,\qquad \sigma^2_{\text{mix}} = \sum_k \hat\alpha_t(k)\left(s_k^2 + m_k^2\right) - \mu_{\text{mix}}^2$$

**Volatility targeting / risk parity (default):**
$$w^{\text{vol}}_t = d_t \cdot \frac{\sigma_{\text{target}}}{\sqrt{252}\,\sigma_{\text{mix}}}$$

**Fractional Kelly (cap / alternative):** for log-utility with Gaussian returns, $f^* = \mu/\sigma^2$:
$$w^{\text{kelly}}_t = \lambda\cdot\frac{\mu_{\text{mix}} - r_f}{\sigma^2_{\text{mix}}},\qquad \lambda \in [0.25, 0.5]$$

**Final weight:** $w_t = \text{clip}\left(\text{sign}(d_t)\min\left(|w^{\text{vol}}_t|, |w^{\text{kelly}}_t|\right),\ -L_{\max},\ L_{\max}\right)$ when the `kelly_cap` mode is enabled.

Full Kelly on an *estimated* $\mu$ overbets badly because estimation error in $\mu$ dominates. So Kelly is used as a cap and vol-targeting as the primary sizing method. This choice is documented as deliberate.

### 5.3 Value at Risk and Expected Shortfall (rolling window $W$, confidence $\alpha$)

- **Historical:** $\text{VaR}^{H}_\alpha = -\,Q_{1-\alpha}\!\left(\{r_{t-W+1},\dots,r_t\}\right)\cdot |w_t| \cdot E_t$
- **Parametric (Gaussian):** $\text{VaR}^{P}_\alpha = -\left(\hat\mu + z_{1-\alpha}\hat\sigma\right)|w_t|E_t$
- **Regime-mixture (the HMM-native measure):** solve $\sum_k \hat p_{t+1}(k)\,\Phi\!\left(\frac{q - m_k}{s_k}\right) = 1-\alpha$ for $q$ with `scipy.optimize.brentq`, using one-step-ahead probabilities $\hat p_{t+1} = \hat\alpha_t A$. Then $\text{VaR}^{M}_\alpha = -q\,|w_t|E_t$.
- **Expected Shortfall:** $\text{ES}_\alpha = -\mathbb{E}[r \mid r \le -\text{VaR}_\alpha]$, historical and Gaussian closed form $\hat\sigma\,\phi(z_\alpha)/(1-\alpha) - \hat\mu$.

**VaR budget rule:** if the proposed position's $\text{VaR}^{M}_\alpha / E_t >$ `var_budget`, scale $w$ down proportionally.

**VaR backtesting:** the Kupiec proportion-of-failures likelihood ratio test checks the breach rate $\hat p = x/n$ against $1-\alpha$:
$$LR_{POF} = -2\ln\frac{(1-p)^{n-x}p^x}{(1-\hat p)^{n-x}\hat p^x} \sim \chi^2_1$$
It's reported for all three VaR methods.

### 5.4 Drawdown, Stops, Targets

- **Drawdown:** $DD_t = 1 - E_t / \max_{s\le t}E_s$. **Circuit breaker:** if $DD_t >$ `dd_breaker`, flatten and halt new risk for `dd_halt_bars`.
- **Trailing stop (ATR-based):** for longs, $\text{stop}_t = \max_{s \in \text{trade}} H_s - k\cdot\text{ATR}_{14}$. It's checked **intrabar** against $L_{t}$. The fill is $\min(\text{stop}, O_t)$ to model gap-through. Shorts are symmetric.
- **Profit target:** optional, $\text{entry} \pm k_{pt}\cdot\text{ATR}$, same intrabar logic. If both the stop and the target are hit in the same bar, **the stop wins** (conservative assumption, since the path within the bar is unknown).

### 5.5 RiskManager pipeline

`RiskManager` applies an **ordered** list of `RiskRule`s to a requested `TargetPosition`. Each rule can pass, modify, or veto. The full trace is kept in `RiskDecision.rule_trace` and logged. Default order: `DrawdownBreaker → Cooldown → VaRBudget → MaxLeverage`. Stops and targets are evaluated separately, earlier in the bar loop (§6.2), because they're intrabar events and not rebalance decisions.

---

## 6. Execution & Backtesting

### 6.1 Execution Cost Model

For an order of $Q$ shares at reference price $P_{\text{ref}}$ (the open of the fill bar):
$$P_{\text{fill}} = P_{\text{ref}}\left(1 + \text{side}\cdot\left[\underbrace{h}_{\text{half-spread}} + \underbrace{\eta\,\sigma_{d}\sqrt{|Q|/\text{ADV}_{20}}}_{\text{square-root impact}}\right]\right)$$
$$\text{commission} = \max\left(c_{\min},\ c_{\text{bps}}\cdot|Q|P_{\text{fill}}\right),\qquad \text{borrow}_t = |Q_{\text{short}}|P_t\cdot\frac{b_{\text{annual}}}{252}$$

**Latency:** an order created at close $t$ becomes eligible at bar $t + \text{latency\_bars}$ and fills at that bar's open. On daily bars this is the only meaningful form of latency. The `ExecutionSimulator` uses a FIFO queue keyed by `eligible_ts`, so moving to intraday bars only changes configuration.

### 6.2 Event-Driven Loop (per bar $t$)

```
for bar t in index[warmup:]:
  1. ExecutionSimulator.fill_eligible(t, bar)        → fills at open_t, Portfolio.apply(fills)
  2. Stops/targets: check intrabar vs high_t/low_t   → immediate gap-aware exit fills
  3. Portfolio.mark_to_market(close_t), accrue borrow
  4. HMMEngine: if refit due → fit on [t-W, t]; then filter step → RegimeState_t
  5. StrategyAgent.signal(context_t)                 → Signal
  6. Sizer.size(signal, regime_state, context_t)     → TargetPosition (requested)
  7. RiskManager.evaluate(requested, state_t)        → RiskDecision (approved)
  8. If |Δw| > rebalance_threshold → Order(eligible = t + latency)
  9. Record PortfolioSnapshot, RegimeState, RiskDecision; structured log events
```

Why event-driven instead of vectorized: cooldowns, trailing stops, drawdown halts, and refit-time label mapping are all **path-dependent**. A vectorized engine would have to approximate them. A pure vectorized `quick_backtest()` (no stops or cooldown) is kept for the ablation study and as a cross-check. With all risk rules disabled, the two engines must agree within a tolerance.

**Benchmark:** buy-and-hold of `data.benchmark` with the same one-time entry cost.

---

## 7. Analytics & Dashboard

### 7.1 Metrics (daily returns $r_t$, benchmark $b_t$, annualization $N$ = 252 or 365 for crypto)

| Metric | Formula |
|---|---|
| Total / CAGR | $E_T/E_0 - 1$; $(E_T/E_0)^{N/T} - 1$ |
| Sharpe | $\sqrt N\,\overline{(r - r_f)}/\sigma(r - r_f)$ |
| Sortino | $\sqrt N\,\overline{(r - r_f)}/\text{DD}$, $\text{DD} = \sqrt{\frac1T\sum\min(r_t - \text{MAR}, 0)^2}$ |
| Information Ratio | $\sqrt N\,\overline{(r - b)}/\sigma(r - b)$ |
| Max Drawdown | $\max_t DD_t$; **duration** = longest peak-to-recovery span in bars (ongoing if not yet recovered) |
| Calmar | CAGR / MDD |
| Probabilistic Sharpe | $\Phi\!\left(\frac{(\widehat{SR}-SR^*)\sqrt{T-1}}{\sqrt{1-\hat\gamma_3\widehat{SR}+\frac{\hat\gamma_4-1}{4}\widehat{SR}^2}}\right)$ |
| Deflated Sharpe | PSR with $SR^*$ set by the number of configurations tried (Bailey & López de Prado, 2014) |
| Turnover, cost drag, exposure, hit rate | per run and per regime |

### 7.2 Regime Diagnostics

There's no ground-truth regime, so a "confusion matrix" has to say what it compares. Two are produced:
1. **Filtered vs. smoothed:** causal $\arg\max\hat\alpha_t$ against ex-post $\arg\max\gamma_t$ (same model). The off-diagonal mass **measures detection lag**, which is the cost of not having hindsight.
2. **Decoded vs. realized:** decoded regime at $t$ against an ex-post label from the *next* 21 bars (sign of forward return × above/below median forward realized vol). This measures whether the regimes carry forward-looking information.

Also reported: the canonical transition matrix heatmap (averaged across refits, with dispersion), expected durations, per-regime return attribution, and the count of mapping-ambiguity events.

### 7.3 Tear Sheet (`report.py` → `runs/<run_id>/report.html`)

Panels: (1) log equity vs. benchmark; (2) underwater curve with the longest drawdown highlighted; (3) price shaded by confirmed regime, with cooldown windows hatched; (4) stacked filtered-probability area chart; (5) transition matrix heatmap; (6) both confusion matrices; (7) rolling VaR (3 methods) with breaches marked plus the Kupiec table; (8) metrics table (strategy vs. benchmark); (9) ablation table: no-HMM vol-target baseline, hard vs. blended routing, cooldown on/off, costs on/off.

The ablation table matters most in an interview. It shows *what the HMM contributes* compared with plain volatility targeting, which captures a large share of the drawdown benefit on its own.

---

## 8. Class Blueprints

Signatures only. Implementations follow after sign-off.

```python
# ---------- data ----------
class MarketDataSource(Protocol):
    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame: ...

class DataIngestor:
    def __init__(self, source: MarketDataSource, cache_dir: Path,
                 validator: OHLCVValidator, logger: logging.Logger) -> None: ...
    def load(self, symbol: str, start: date, end: date, *, refresh: bool = False) -> pd.DataFrame: ...
    def load_aligned(self, symbols: Sequence[str], start: date, end: date) -> dict[str, pd.DataFrame]: ...
    def data_hash(self, df: pd.DataFrame) -> str: ...

class OHLCVValidator:
    def __init__(self, cfg: ValidationConfig) -> None: ...
    def validate(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame: ...   # raises DataValidationError
    def assert_aligned(self, a: pd.DataFrame, b: pd.DataFrame) -> None: ...

# ---------- features ----------
class FeatureEngineer:
    FEATURE_NAMES: ClassVar[tuple[str, ...]] = ("log_ret", "hl_vol", "vol_of_vol")
    def __init__(self, window: int) -> None: ...
    def transform(self, ohlcv: pd.DataFrame) -> pd.DataFrame: ...
    @staticmethod
    def parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series: ...

class CausalScaler:
    def fit(self, X: npt.NDArray[np.float64]) -> Self: ...
    def transform(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def inverse_transform_col(self, values: npt.NDArray[np.float64], col: int) -> npt.NDArray[np.float64]: ...

# ---------- models ----------
@dataclass(frozen=True, slots=True)
class HMMParams:
    startprob: npt.NDArray[np.float64]   # (K,)
    transmat: npt.NDArray[np.float64]    # (K, K)
    means: npt.NDArray[np.float64]       # (K, d)
    covars: npt.NDArray[np.float64]      # (K, d, d)
    def permuted(self, perm: npt.NDArray[np.int_]) -> "HMMParams": ...
    def validate(self) -> None: ...      # stochasticity, PSD covariances

@dataclass(frozen=True, slots=True)
class FitResult:
    params: HMMParams
    log_likelihood: float
    n_iter: int
    converged: bool
    bic: float
    state_occupancy: npt.NDArray[np.float64]

class GaussianHMM:
    def __init__(self, n_states: int, *, covariance_ridge: float = 1e-6,
                 max_iter: int = 500, tol: float = 1e-6, n_restarts: int = 10,
                 transition_prior: float | None = None, seed: int = 0) -> None: ...
    def fit(self, X: npt.NDArray[np.float64]) -> FitResult: ...
    def log_emission(self, X: npt.NDArray[np.float64], params: HMMParams) -> npt.NDArray[np.float64]: ...
    def forward(self, X: npt.NDArray[np.float64], params: HMMParams) -> tuple[npt.NDArray[np.float64], float]: ...
    def filter_step(self, alpha_prev: npt.NDArray[np.float64], x_t: npt.NDArray[np.float64],
                    params: HMMParams) -> npt.NDArray[np.float64]: ...
    def smooth(self, X: npt.NDArray[np.float64], params: HMMParams) -> npt.NDArray[np.float64]: ...   # diagnostics only
    def viterbi(self, X: npt.NDArray[np.float64], params: HMMParams) -> npt.NDArray[np.int_]: ...     # diagnostics only
    def sample(self, n: int, params: HMMParams, rng: np.random.Generator) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]: ...

@dataclass(frozen=True, slots=True)
class RegimeMapping:
    perm: npt.NDArray[np.int_]           # raw index -> canonical Regime index
    regime_moments: dict[Regime, tuple[float, float]]   # (mean ret, vol) in original units
    ambiguous: bool

class RegimeMapper:
    def __init__(self, vol_source: Literal["hl_vol", "ret_var"] = "hl_vol") -> None: ...
    def map(self, params: HMMParams, scaler: CausalScaler,
            previous: HMMParams | None = None) -> RegimeMapping: ...

class HMMEngine:
    """Owns walk-forward refits, scaler versioning, canonical mapping, and online filtering."""
    def __init__(self, model: GaussianHMM, mapper: RegimeMapper, cfg: HMMConfig,
                 regime_cfg: RegimeConfig, logger: logging.Logger) -> None: ...
    def needs_refit(self, bar_idx: int) -> bool: ...
    def refit(self, features: pd.DataFrame, end_idx: int) -> FitResult: ...
    def update(self, ts: pd.Timestamp, x_t: npt.NDArray[np.float64]) -> RegimeState: ...
    def one_step_forecast(self) -> npt.NDArray[np.float64]: ...
    @property
    def canonical_params(self) -> HMMParams: ...
    @property
    def fit_history(self) -> Sequence[tuple[pd.Timestamp, FitResult, RegimeMapping]]: ...

# ---------- strategy ----------
@dataclass(frozen=True, slots=True)
class MarketContext:
    ts: pd.Timestamp
    history: pd.DataFrame        # OHLCV up to and including ts (view, read-only)
    regime: RegimeState

class Strategy(Protocol):
    name: str
    def generate(self, ctx: MarketContext) -> float: ...     # direction in [-1, 1]

class TimeSeriesMomentum:      # implements Strategy
    def __init__(self, lookbacks: Sequence[int] = (21, 63, 252), long_only: bool = True,
                 damping: float = 1.0) -> None: ...
class ZScoreMeanReversion:     # implements Strategy
    def __init__(self, window: int = 20, scale: float = 2.0) -> None: ...
class DefensiveShort:          # implements Strategy
    def __init__(self, short_fraction: float = 0.5) -> None: ...

class StrategyAgent:
    def __init__(self, routing: Mapping[Regime, Strategy],
                 mode: Literal["hard", "blended"] = "hard") -> None: ...
    def signal(self, ctx: MarketContext) -> Signal: ...

# ---------- risk ----------
class PositionSizer(Protocol):
    def size(self, signal: Signal, regime: RegimeState, moments: Mapping[Regime, tuple[float, float]],
             equity: float) -> TargetPosition: ...

class VolTargetSizer: ...        # implements PositionSizer
class FractionalKellySizer: ...  # implements PositionSizer
class CappedSizer:               # min(|vol|, |kelly|) composite
    def __init__(self, primary: PositionSizer, cap: PositionSizer, max_leverage: float) -> None: ...

class VaRCalculator:
    def __init__(self, alpha: float, window: int) -> None: ...
    def historical(self, returns: npt.NDArray[np.float64]) -> float: ...
    def parametric(self, returns: npt.NDArray[np.float64]) -> float: ...
    def regime_mixture(self, probs: npt.NDArray[np.float64],
                       moments: Mapping[Regime, tuple[float, float]]) -> float: ...
    def expected_shortfall(self, returns: npt.NDArray[np.float64],
                           method: Literal["historical", "parametric"]) -> float: ...
    @staticmethod
    def kupiec_pof(breaches: npt.NDArray[np.bool_], alpha: float) -> tuple[float, float]: ...  # (LR, p-value)

@dataclass(frozen=True, slots=True)
class RiskState:
    ts: pd.Timestamp
    equity: float
    current_weight: float
    drawdown: float
    regime: RegimeState
    var_estimate: float

class RiskRule(Protocol):
    name: str
    def apply(self, target: TargetPosition, state: RiskState) -> tuple[TargetPosition, str]: ...

class CooldownRule:              # stateful: tracks bars remaining
    def __init__(self, n_bars: int, mode: Literal["freeze", "reduce_only"]) -> None: ...
    def on_regime_state(self, rs: RegimeState) -> None: ...
    @property
    def active(self) -> bool: ...
class DrawdownBreaker: ...
class VaRBudgetRule: ...
class MaxLeverageRule: ...

class StopManager:
    def __init__(self, trailing_atr_mult: float | None, target_atr_mult: float | None) -> None: ...
    def on_entry(self, ts: pd.Timestamp, price: float, side: Side, atr: float) -> None: ...
    def check(self, bar: pd.Series, atr: float) -> Order | None: ...     # gap-aware, stop wins ties
    def reset(self) -> None: ...

class RiskManager:
    def __init__(self, rules: Sequence[RiskRule], stops: StopManager,
                 var: VaRCalculator, logger: logging.Logger) -> None: ...
    def evaluate(self, requested: TargetPosition, state: RiskState) -> RiskDecision: ...
    def check_stops(self, bar: pd.Series, atr: float) -> Order | None: ...
    def on_regime_state(self, rs: RegimeState) -> None: ...

# ---------- execution / portfolio ----------
class CostModel(Protocol):
    def fill_price(self, order: Order, ref_price: float, ctx: ExecutionContext) -> float: ...
    def commission(self, quantity: float, price: float) -> float: ...

class SqrtImpactCostModel: ...   # implements CostModel

class ExecutionSimulator:
    def __init__(self, cost_model: CostModel, latency_bars: int, logger: logging.Logger) -> None: ...
    def submit(self, order: Order) -> None: ...
    def fill_eligible(self, ts: pd.Timestamp, bar: pd.Series, ctx: ExecutionContext) -> list[Fill]: ...
    def fill_immediate(self, order: Order, price: float, ctx: ExecutionContext) -> Fill: ...  # stops
    @property
    def pending(self) -> Sequence[Order]: ...

class Portfolio:
    def __init__(self, initial_cash: float, borrow_bps_annual: float) -> None: ...
    def apply(self, fill: Fill) -> None: ...
    def mark_to_market(self, ts: pd.Timestamp, price: float) -> PortfolioSnapshot: ...
    def accrue_borrow(self, price: float, n_days: float) -> None: ...
    def orders_for_target(self, target: TargetPosition, price: float, ts: pd.Timestamp,
                          latency: int) -> list[Order]: ...
    @property
    def equity(self) -> float: ...

# ---------- backtest / analytics ----------
@dataclass(frozen=True)
class BacktestResult:
    run_id: str
    config: EngineConfig
    snapshots: pd.DataFrame       # PortfolioSnapshot rows
    regimes: pd.DataFrame         # ts, probs[4], regime, changed, cooldown_active
    decisions: pd.DataFrame       # RiskDecision rows (flattened trace)
    fills: pd.DataFrame
    benchmark_equity: pd.Series
    fit_history: pd.DataFrame
    def returns(self) -> pd.Series: ...

class Backtester:
    def __init__(self, cfg: EngineConfig, ingestor: DataIngestor, features: FeatureEngineer,
                 hmm: HMMEngine, agent: StrategyAgent, sizer: PositionSizer,
                 risk: RiskManager, execution: ExecutionSimulator,
                 portfolio_factory: Callable[[], Portfolio], logger: logging.Logger) -> None: ...
    def run(self) -> BacktestResult: ...

def build_backtester(cfg: EngineConfig) -> Backtester: ...   # composition root (the only place that wires concretes)

class PerformanceAnalyzer:
    def __init__(self, result: BacktestResult, annualization: int, risk_free: float = 0.0) -> None: ...
    def summary(self) -> pd.DataFrame: ...                    # strategy vs benchmark
    def drawdowns(self, top_n: int = 5) -> pd.DataFrame: ...  # depth, start, trough, recovery, duration
    def regime_attribution(self) -> pd.DataFrame: ...
    def probabilistic_sharpe(self, sr_benchmark: float = 0.0) -> float: ...
    def deflated_sharpe(self, n_trials: int, sr_variance: float) -> float: ...

class RegimeDiagnostics:
    def __init__(self, result: BacktestResult, model: GaussianHMM) -> None: ...
    def filtered_vs_smoothed(self) -> pd.DataFrame: ...       # 4x4 confusion
    def decoded_vs_realized(self, horizon: int = 21) -> pd.DataFrame: ...
    def mean_transition_matrix(self) -> tuple[pd.DataFrame, pd.DataFrame]: ...   # mean, std across refits
    def var_backtest(self) -> pd.DataFrame: ...                # Kupiec per method

class TearSheet:
    def __init__(self, perf: PerformanceAnalyzer, diag: RegimeDiagnostics) -> None: ...
    def render(self, out_dir: Path) -> Path: ...               # returns path to report.html
```

**Exceptions (`exceptions.py`):**
`RegimeEngineError` → `DataValidationError`, `DataAlignmentError`, `LookAheadError`, `ModelNotFittedError`, `HMMConvergenceError`, `RegimeMappingError`, `ConfigError`.

---

## 9. Logging

- **`logging` + a custom `JsonFormatter`.** Each record is one JSON line in `runs/<run_id>/events.jsonl`, with a human-readable console handler at INFO.
- **A `ContextFilter` adds `run_id` and `bar_ts`** to every record.
- **Event taxonomy (`extra={"event": ...}`):** `DATA_INGESTED`, `DATA_VALIDATION_WARN`, `FEATURES_BUILT`, `HMM_FIT` (loglik, iters, converged, bic, occupancy), `REGIME_MAPPING_AMBIGUOUS`, `REGIME_CHANGE` (from, to, probs), `COOLDOWN_START` / `COOLDOWN_END`, `RISK_MODIFY` / `RISK_VETO` (rule, before, after), `ORDER_SUBMITTED`, `ORDER_FILLED` (prices, costs), `STOP_TRIGGERED`, `DD_BREAKER_TRIPPED`, `RUN_COMPLETE` (summary metrics).
- **The events file can be replayed.** Diagnostics can be rebuilt from `events.jsonl` alone.

---

## 10. Testing Strategy

| Layer | Tests |
|---|---|
| HMM | hmmlearn parity (loglik, one M-step); synthetic recovery; rows of $A$ sum to 1; $\Sigma_k$ PSD; filter_step ≡ batch forward |
| Mapper | Permutation invariance: shuffling raw states gives the same canonical params; ambiguity flag |
| Features | Parkinson against a hand-computed value; no NaN after warm-up; causality (feature at t unchanged by appending bars) |
| Risk | Cooldown reduce-only semantics; gap-through stop fill; stop-wins-tie; VaR against closed form on Gaussian data; Kupiec against known values |
| Execution | Costs are always ≥ 0 and always adverse (hypothesis); latency: no fill before `eligible_ts` |
| Engine | **Look-ahead canary**; event-driven ≡ vectorized with risk off; deterministic given seed |
| Metrics | Sharpe/Sortino/MDD/duration on hand-built series |

CI target: `ruff`, `mypy --strict`, and `pytest` all pass, with ≥ 90% coverage on `models/`, `risk/`, and `execution/`.

---

## 11. Implementation Phases (proposed)

1. **Foundation:** config, types, logging, data ingestion and validation, features, with tests.
2. **HMM core:** NumPy GaussianHMM, mapper, hmmlearn parity and synthetic recovery tests.
3. **HMMEngine:** walk-forward refit, online filtering, hysteresis, look-ahead canary scaffold.
4. **Strategy, sizing, risk:** strategies, agent, sizers, VaR, rules, stops.
5. **Execution and backtester:** cost model, simulator, portfolio, event loop, vectorized cross-check.
6. **Analytics and tear sheet:** metrics, diagnostics, ablations, HTML report, README with results.

---

## 12. Open Decisions

1. **Primary asset:** SPY daily from 2005 (covers 2008, 2011, 2015, 2018, 2020, 2022, and gives about 1,260 bars of warm-up before 2008) — *recommended*. BTC-USD runs through the same code path with `annualization: 365`.
2. **Bear-regime strategy:** single-asset (direct short with borrow, or flat) — *recommended for v1*. Or add a defensive second asset (TLT / GLD / BIL rotation) for a real allocation sleeve. That turns `Portfolio` multi-asset, which is a moderate scope increase.
3. **Cooldown semantics:** `reduce_only` (recommended) vs. literal `freeze`. Both are implemented; which one is the default?
4. **HMM implementation:** custom NumPy with an hmmlearn oracle (recommended) vs. hmmlearn plus a custom filter only.

---

## 13. Decision & Findings Log

| Date | Decision / finding | Rationale / evidence |
|---|---|---|
| 2026-10-13 | SPY daily from 2005; `cooldown_mode = reduce_only`; custom NumPy HMM with `hmmlearn` as test oracle | Open decisions 1, 3, 4 (§12) |
| 2026-10-13 | `hl_vol` defaults to the **Parkinson** estimator | A rolling std of ln(H/L) measures how much the range *varies*, not its *level*, so it separates calm from turbulent markets poorly |
| 2026-10-13 | Inference is exposed as pure functions `forward/filter_step/smooth/viterbi(X, params)`; `GaussianHMM` only estimates | Functional core (§0.3); immutable `HMMParams` |
| 2026-10-13 | Training window defaults to **expanding** | A rolling 1260-bar window ending 2026 contains no bear market: both "bear" states had positive drift and the "crash" state earned +13.6%/yr. With expanding windows, BEAR_HIGH_VOL keeps negative drift in all 200 walk-forward fits (−62% → −11% ann.) |
| 2026-10-13 | **Finding:** causally detected regimes classify *volatility*, not *direction* | Walk-forward SPY 2010–2026, next-day realized return by confirmed regime: BULL_LOW 7.6% (vol 11%), BULL_HIGH 15.6% (15%), BEAR_LOW 17.9% (20%), BEAR_HIGH 30.5% (39%). State means are contemporaneous; detection lags the drawdown (COVID: peak 2020-02-19, confirmed BEAR_HIGH 2020-03-04), and the rebound follows. Implication for Phase 4: shorting in bear regimes is not supported by the data; regimes should drive *risk* (sizing, exposure), and the no-HMM vol-target baseline is essential |
| 2026-10-13 | **Finding:** BEAR_LOW_VOL drifts slightly positive (+4–6%) in 172/200 fits | Over 2010–2026 only one state has genuinely negative drift; BEAR_LOW behaves as "choppy / low-return". The anomaly channel reports it as `bear_non_negative_mean` |
| 2026-10-13 | Phase 4 defaults: BEAR_HIGH_VOL → `defensive` reduced long (direction 0.5); `allow_short = false`; regime vol-target sizing with 0.25-Kelly cap | Approved plan following the "volatility, not direction" finding |
| 2026-10-13 | **Finding:** the Kelly cap overrides the defensive policy | SPY walk-forward decision pipeline (no fills yet): Kelly binds on 99% of BEAR_HIGH bars (mean weight 0.01, i.e. flat) and 72% of BEAR_LOW bars. The cap uses the HMM's *contemporaneous* state means (BEAR_HIGH ≈ −11%/yr), which the forward-return finding shows are not what follows detection. Overall: mean weight 0.40, flat 26% of the time |
| 2026-10-13 | **Finding:** each VaR method fails a different backtest | 1-day 99% VaR, 2010–2026: regime-mixture breaches 2.70% (Kupiec rejects; Christoffersen p=0.28, breaches independent): it adapts to vol clustering but Gaussian-within-regime tails are too thin. Historical breaches 1.50% (Kupiec p=0.002) and are clustered (Christoffersen p<0.001) |
| 2026-10-13 | Kelly cap **off** by default; filtered historical simulation (FHS) VaR added and used by the VaR budget | User decision after Phase 4 findings |
| 2026-10-13 | **Finding (Phase 5, first full backtest):** SPY 2010-02 → 2026-09, with costs, next-open fills | Full engine (tactical routing + 3-ATR stops): CAGR 2.2%, vol 5.2%, Sharpe 0.45, MDD 11.5%. Buy & hold: 14.6%, 17.0%, 0.89, 33.7% |
| 2026-10-13 | **Finding: decomposition by layer** (same period, costs included) | Stops off: Sharpe 0.65 (stops cost 0.20 Sharpe, 121 stop-outs + 5-bar lockouts). Always-long + HMM vol target, no stops: CAGR 11.6%, vol 12.1%, **Sharpe 0.97, MDD 15.4%**, costs $24k. Always-long + realised-vol target: 11.1%, 12.1%, 0.93, 15.5%, costs $39k. The HMM adds value as a **volatility forecaster for sizing** (beats buy & hold on Sharpe, halves MDD, smoother than realised vol → ~40% less cost); the tactical direction layer **destroys** value (0.97 → 0.65), consistent with "regimes classify volatility, not direction". HMM vs realised-vol Sharpe gap (0.04) is well within one standard error (≈0.25 over 16.6 years) and the decomposition was run after seeing results; Phase 6 must quantify significance (PSR/DSR, bootstrap) |
| 2026-10-13 | FHS VaR: breaches 1.37% (Kupiec p=0.025, Christoffersen p=0.008) vs mixture 2.70%, historical 1.50%, parametric 2.67% | Best calibrated of the four, still slightly under-covering, with some clustering |
| 2026-10-13 | **Headline = risk overlay**: all regimes `long`, stops off by default; tactical design kept as `StrategyConfig.tactical()` | User decision after the Phase 5 decomposition |
| 2026-10-13 | `RegimeTrack`: the HMM path is computed once per regime-relevant config (`REGIME_SECTIONS`) and replayed across variants | Ablation of 10 variants in ~3 min instead of ~25; the whole-stack look-ahead canary still passes because the track is built bar by bar from the same causal engine |
| 2026-10-13 | **Final ablation (SPY 2010-02 → 2026-09, costs included)** | Overlay: CAGR 11.6%, vol 12.1%, Sharpe 0.97 ± 0.25, MDD 15.4%, DSR 99.7% (9 trials). Buy & hold: 14.6%, 17.0%, 0.89, 33.7% (ΔSharpe p = 0.43, not significant). Realised-vol sizing: Sharpe 0.93 (p = 0.27) but 5.8× vs 3.5× turnover, $39k vs $24k costs. Significantly worse (p < 0.05): stops (−0.37), regime-routed strategies (−0.32), original design (−0.52). Not significant: Kelly cap, cooldown, rolling window |
| 2026-10-13 | **Diagnostics**: causal vs smoothed regime agreement 83%; decoded vs realised next-21-day label 32% (chance ≈ 25%); the VaR budget is the most active rule (1,123 bars) | The model tracks the current volatility state, not forward direction |
| 2026-10-15 | **Pre-registered out-of-sample test** on QQQ, IWM, EFA, TLT, GLD, BTC-USD ([OUT_OF_SAMPLE.md](OUT_OF_SAMPLE.md)) | Protocol committed (`0f2d61b`) before the run. Replicates: drawdown reduction, lower turnover, HMM sizing ≥ realised-vol sizing, regime-routed signals hurt (all 6/6). Partially: Sharpe > buy & hold (4/6, none significant). Fails: DSR > 0.95 (1/6). Exploratory: stops help on GLD and BTC-USD, so "stops off" is SPY-specific |
