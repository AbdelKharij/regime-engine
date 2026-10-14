"""Regime-model diagnostics.

No ground-truth regime exists, so each confusion matrix says what it compares:

* **Filtered vs smoothed.** The causal argmax P(S_t | x_1..t) actually used for trading,
  against the ex-post argmax P(S_t | x_1..T) under the final model. Off-diagonal mass is the
  price of not having hindsight (detection lag and false alarms).
* **Decoded vs realised.** The confirmed regime at t, against a label built from what
  happened over the next ``horizon`` bars: forward return sign (bull/bear) x forward realised
  volatility above/below its full-sample median (high/low). This measures whether the
  detected regimes carry *forward-looking* information. It does not affect any decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_engine.backtest.engine import BacktestResult
from regime_engine.core.types import Regime
from regime_engine.models.gaussian_hmm import smooth
from regime_engine.models.hmm_engine import PROB_COLUMNS
from regime_engine.risk.var import christoffersen_independence, kupiec_pof

REGIME_NAMES: tuple[str, ...] = tuple(r.name for r in Regime)


def _confusion(actual: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    """Counts with rows = ``actual`` labels and columns = ``predicted`` labels."""
    table = pd.crosstab(actual, predicted).reindex(index=range(4), columns=range(4), fill_value=0)
    table.index = pd.Index(REGIME_NAMES, name=actual.name)
    table.columns = pd.Index(REGIME_NAMES, name=predicted.name)
    return table.astype(np.int64)


def filtered_vs_smoothed(result: BacktestResult, *, confirmed: bool = False) -> pd.DataFrame:
    """Rows: ex-post smoothed regime (final model). Columns: causal regime used at the time."""
    if result.final_params is None or result.final_scaling is None or result.regimes.empty:
        raise ValueError("backtest has no fitted regime model")
    features = result.features.loc[result.regimes.index]
    scaled = (
        features.to_numpy(dtype=np.float64) - result.final_scaling.location
    ) / result.final_scaling.scale
    gamma = smooth(scaled, result.final_params)
    ex_post = pd.Series(gamma.argmax(axis=1), index=features.index, name="smoothed")
    causal = result.regimes["regime" if confirmed else "raw_argmax"].rename("filtered")
    return _confusion(ex_post, causal)


def realized_regime_labels(close: pd.Series, horizon: int = 21) -> pd.Series:
    """Ex-post 2x2 label from the next ``horizon`` bars (NaN where the window is incomplete)."""
    log_close = pd.Series(np.log(close.to_numpy(dtype=np.float64)), index=close.index)
    fwd_return = log_close.shift(-horizon) - log_close
    daily = log_close.diff()
    fwd_vol = daily[::-1].rolling(horizon, min_periods=horizon).std()[::-1].shift(-1)
    high_vol = fwd_vol > fwd_vol.median()
    labels = np.where(
        fwd_return > 0,
        np.where(high_vol, Regime.BULL_HIGH_VOL, Regime.BULL_LOW_VOL),
        np.where(high_vol, Regime.BEAR_HIGH_VOL, Regime.BEAR_LOW_VOL),
    ).astype(np.float64)
    labels[(fwd_return.isna() | fwd_vol.isna()).to_numpy()] = np.nan
    return pd.Series(labels, index=close.index, name="realized")


def decoded_vs_realized(
    result: BacktestResult, close: pd.Series, *, horizon: int = 21
) -> pd.DataFrame:
    """Rows: realised forward label. Columns: confirmed regime decoded at the time."""
    realized = realized_regime_labels(close, horizon).reindex(result.regimes.index).dropna()
    decoded = result.regimes["regime"].reindex(realized.index).rename("decoded")
    return _confusion(realized.astype(np.int64), decoded)


def accuracy(confusion: pd.DataFrame) -> float:
    values = confusion.to_numpy(dtype=np.float64)
    return float(np.trace(values) / values.sum()) if values.sum() else float("nan")


def mean_transition_matrix(result: BacktestResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean and standard deviation of the canonical transition matrix across refits."""
    stack = result.transition_matrices
    if len(stack) == 0:
        raise ValueError("no refits recorded")
    mean = pd.DataFrame(stack.mean(axis=0), index=REGIME_NAMES, columns=REGIME_NAMES)
    std = pd.DataFrame(stack.std(axis=0), index=REGIME_NAMES, columns=REGIME_NAMES)
    return mean, std


def regime_attribution(
    result: BacktestResult, close: pd.Series, periods: int = 252
) -> pd.DataFrame:
    """Per confirmed regime: time share, exposure, and next-bar asset vs strategy returns.

    The return on bar t+1 is attributed to the regime confirmed at the close of bar t, which
    is the information the position was decided on.
    """
    regimes = result.regimes["regime"]
    weight = result.equity["weight"].reindex(regimes.index)
    asset_next = close.pct_change().shift(-1).reindex(regimes.index)
    strategy_next = result.equity["equity"].pct_change().shift(-1).reindex(regimes.index)
    rows = []
    for regime in Regime:
        mask = (regimes == int(regime)).to_numpy()
        a, s = asset_next[mask].dropna(), strategy_next[mask].dropna()
        rows.append(
            {
                "regime": regime.name,
                "share_of_bars": float(mask.mean()),
                "mean_weight": float(weight[mask].mean()) if mask.any() else float("nan"),
                "asset_return_ann": float(a.mean() * periods) if len(a) else float("nan"),
                "asset_vol_ann": float(a.std() * np.sqrt(periods)) if len(a) > 1 else float("nan"),
                "strategy_return_ann": float(s.mean() * periods) if len(s) else float("nan"),
                "strategy_vol_ann": float(s.std() * np.sqrt(periods))
                if len(s) > 1
                else float("nan"),
                "strategy_contribution": float(s.sum()),
            }
        )
    return pd.DataFrame(rows).set_index("regime")


def var_backtest_table(result: BacktestResult) -> pd.DataFrame:
    """Breach rates and coverage tests for each VaR method (per unit of exposure)."""
    alpha = result.config.risk.var_alpha
    table = result.var_forecasts
    rows = []
    for column, label in (
        ("var_fhs", "filtered historical"),
        ("var_mixture", "regime mixture"),
        ("var_historical", "historical"),
        ("var_parametric", "parametric (Gaussian)"),
    ):
        valid = table[[column, "realized_next"]].dropna()
        if valid.empty:
            continue
        breaches = (valid["realized_next"] < -valid[column]).to_numpy()
        kupiec = kupiec_pof(breaches, alpha)
        independence = christoffersen_independence(breaches)
        rows.append(
            {
                "method": label,
                "n": len(breaches),
                "breaches": int(breaches.sum()),
                "breach_rate": float(breaches.mean()),
                "expected_rate": 1.0 - alpha,
                "kupiec_p": kupiec.p_value,
                "christoffersen_p": independence.p_value,
            }
        )
    return pd.DataFrame(rows).set_index("method")


def probability_frame(result: BacktestResult) -> pd.DataFrame:
    return result.regimes[list(PROB_COLUMNS)].set_axis(list(REGIME_NAMES), axis=1)
