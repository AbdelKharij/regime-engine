"""Analytics: metrics and significance, regime diagnostics, ablation study, tear sheet, CLI."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_engine.analytics import (
    DEFAULT_VARIANTS,
    TearSheet,
    accuracy,
    apply_overrides,
    bootstrap_sharpe_difference,
    cagr,
    decoded_vs_realized,
    deflated_sharpe_ratio,
    drawdown_episodes,
    expected_max_sharpe,
    filtered_vs_smoothed,
    mean_transition_matrix,
    probabilistic_sharpe_ratio,
    realized_regime_labels,
    regime_attribution,
    run_ablation,
    sharpe_ratio,
    sortino_ratio,
    summarize,
    var_backtest_table,
)
from regime_engine.analytics.experiments import AblationResult, Variant
from regime_engine.analytics.metrics import (
    _stationary_bootstrap_indices,
    beta_alpha,
    information_ratio,
)
from regime_engine.cli import main
from regime_engine.core.config import EngineConfig
from regime_engine.core.exceptions import ConfigError
from regime_engine.core.types import Regime
from regime_engine.models import hmm_engine
from regime_engine.utils.logging import reset_logging

from .conftest import _make_ohlcv

SMALL = {
    "training": {"min_train_bars": 300, "refit_every": 80, "train_window": 400},
    "hmm": {"n_restarts": 2, "refit_restarts": 1},
    "risk": {"fhs_min_obs": 50},
}


# ============================================================================================
# Metrics
# ============================================================================================


class TestMetrics:
    def test_point_metrics_against_hand_calculation(self) -> None:
        r = np.array([0.01, -0.01, 0.02, 0.0])
        assert sharpe_ratio(r, periods=1) == pytest.approx(r.mean() / r.std(ddof=1))
        assert sortino_ratio(r, periods=1) == pytest.approx(
            r.mean() / math.sqrt(np.mean(np.minimum(r, 0) ** 2))
        )
        equity = pd.Series([100.0, 110.0, 121.0])
        assert cagr(equity, periods=1) == pytest.approx(0.10)
        assert math.isnan(sharpe_ratio(np.full(10, 0.01)))

    def test_beta_alpha_information_ratio(self) -> None:
        rng = np.random.default_rng(0)
        b = pd.Series(rng.normal(0.0004, 0.01, 5000))
        r = 0.5 * b + 0.0002 + pd.Series(rng.normal(0, 0.002, 5000))
        beta, alpha = beta_alpha(r, b)
        assert beta == pytest.approx(0.5, abs=0.01)
        assert alpha == pytest.approx(0.0002 * 252, abs=0.02)
        ir, te = information_ratio(r, b)
        active = (r - b).to_numpy()
        assert te == pytest.approx(active.std(ddof=1) * math.sqrt(252))
        assert ir == pytest.approx(active.mean() * 252 / te)

    def test_drawdown_episodes(self) -> None:
        idx = pd.bdate_range("2020-01-01", periods=7)
        equity = pd.Series([100.0, 110.0, 99.0, 105.0, 111.0, 90.0, 95.0], index=idx)
        deepest, second = drawdown_episodes(equity)
        assert deepest.depth == pytest.approx(1 - 90 / 111)
        assert deepest.recovery is None
        assert deepest.duration_bars == 2  # ongoing: peak (4) to end of sample (6)
        assert second.depth == pytest.approx(0.1)
        assert (second.peak, second.trough, second.recovery) == (idx[1], idx[2], idx[4])
        assert (second.bars_to_trough, second.duration_bars) == (1, 3)
        summary = summarize(equity, periods=252)
        assert summary.max_drawdown == pytest.approx(deepest.depth)
        assert summary.max_drawdown_duration == 3

    def test_psr_is_calibrated_under_the_null(self) -> None:
        """With a true Sharpe of 0, P(PSR > 0.95) should be about 5%."""
        rng = np.random.default_rng(1)
        psr = np.array(
            [probabilistic_sharpe_ratio(rng.normal(0.0, 0.01, 1000)) for _ in range(600)]
        )
        assert 0.02 < (psr > 0.95).mean() < 0.09

    def test_expected_max_sharpe_matches_monte_carlo(self) -> None:
        rng = np.random.default_rng(2)
        simulated = rng.standard_normal((200_000, 10)).max(axis=1).mean()
        assert expected_max_sharpe(10, 1.0) == pytest.approx(simulated, rel=0.05)
        assert expected_max_sharpe(100, 1.0) > expected_max_sharpe(10, 1.0)
        assert expected_max_sharpe(1, 1.0) == 0.0

    def test_deflation_penalises_many_trials(self) -> None:
        r = np.random.default_rng(3).normal(0.0005, 0.01, 2500)
        psr = probabilistic_sharpe_ratio(r)
        assert deflated_sharpe_ratio(r, [0.2, 0.9, -0.3, 0.5, 1.1, 0.0]) < psr
        assert deflated_sharpe_ratio(r, [0.7]) == pytest.approx(psr)

    def test_bootstrap_sharpe_difference(self) -> None:
        rng = np.random.default_rng(4)
        noise = pd.Series(rng.normal(0.0, 0.01, 3000))
        same = bootstrap_sharpe_difference(noise, noise, n_bootstrap=300)
        assert same.difference == 0.0
        assert same.p_value == 1.0
        better = bootstrap_sharpe_difference(noise + 0.002, noise, n_bootstrap=500)
        assert better.difference > 0
        assert better.ci_low > 0
        assert better.p_value < 0.01

    def test_stationary_bootstrap_blocks(self) -> None:
        idx = _stationary_bootstrap_indices(1000, 50, 20, np.random.default_rng(5))
        assert idx.min() >= 0
        assert idx.max() < 1000
        continuation = ((idx[:, 1:] - idx[:, :-1]) % 1000 == 1).mean()
        assert continuation == pytest.approx(1 - 1 / 20, abs=0.01)


# ============================================================================================
# Ablation + diagnostics on a small synthetic study
# ============================================================================================


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return _make_ohlcv(1100, seed=8)


@pytest.fixture(scope="module")
def ablation(bars: pd.DataFrame) -> AblationResult:
    base = EngineConfig.from_dict(SMALL)
    variants = [
        v
        for v in DEFAULT_VARIANTS
        if v.name
        in ("overlay", "realized_vol_overlay", "overlay_rolling", "tactical", "overlay_zero_cost")
    ]
    return run_ablation(base, bars, variants, n_bootstrap=200)


class TestAblation:
    def test_table_structure(self, ablation: AblationResult) -> None:
        table = ablation.table
        assert list(table.index) == [v.name for v in ablation.variants] + ["buy_and_hold"]
        assert math.isnan(table.loc["overlay", "sharpe_diff_vs_headline"])
        assert table.loc["overlay_zero_cost", "trial"] == False  # noqa: E712
        assert math.isnan(table.loc["overlay_zero_cost", "dsr"])
        assert (table.loc[table["trial"], "dsr"] <= table.loc[table["trial"], "psr"] + 1e-12).all()
        assert table.loc["overlay_zero_cost", "costs"] == 0.0
        assert table.loc["overlay_zero_cost", "cagr"] >= table.loc["overlay", "cagr"]

    def test_regime_track_is_shared(
        self, bars: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        builds: list[str] = []
        original = hmm_engine.RegimeTrack.build.__func__  # type: ignore[attr-defined]

        def counting(cls: type, config: EngineConfig, features: pd.DataFrame) -> object:
            builds.append(hmm_engine.regime_fingerprint(config))
            return original(cls, config, features)

        monkeypatch.setattr(hmm_engine.RegimeTrack, "build", classmethod(counting))
        variants = [
            Variant("overlay", "a"),
            Variant("b", "b", {"sizing": {"method": "realized_vol_target"}}),
            Variant("c", "c", {"risk": {"trailing_stop_atr": 3.0}}),
            Variant("d", "d", {"training": {"window_type": "rolling"}}),
        ]
        run_ablation(EngineConfig.from_dict(SMALL), bars.iloc[:700], variants, n_bootstrap=50)
        assert len(builds) == 2  # expanding + rolling, not one per variant

    def test_override_validation(self) -> None:
        base = EngineConfig()
        assert apply_overrides(base, {"sizing": {"kelly_cap": True}}).sizing.kelly_cap is True
        with pytest.raises(ConfigError, match="unknown config section"):
            apply_overrides(base, {"sizng": {"kelly_cap": True}})
        with pytest.raises(ValueError, match="headline"):
            run_ablation(base, pd.DataFrame(), [Variant("x", "x")])


class TestDiagnostics:
    def test_confusions(self, ablation: AblationResult, bars: pd.DataFrame) -> None:
        head = ablation.results["overlay"]
        fvs = filtered_vs_smoothed(head)
        assert fvs.shape == (4, 4)
        assert int(fvs.to_numpy().sum()) == len(head.regimes)
        dvr = decoded_vs_realized(head, bars["close"])
        assert 0 < int(dvr.to_numpy().sum()) <= len(head.regimes)
        assert 0.0 <= accuracy(dvr) <= 1.0

    def test_realized_labels(self) -> None:
        rng = np.random.default_rng(6)
        calm_up = np.cumsum(0.003 + 0.002 * rng.standard_normal(200))
        wild_down = calm_up[-1] + np.cumsum(-0.004 + 0.03 * rng.standard_normal(200))
        close = pd.Series(np.exp(np.concatenate([calm_up, wild_down])) * 100)
        labels = realized_regime_labels(close, horizon=21)
        assert labels.iloc[:150].mode().iloc[0] == Regime.BULL_LOW_VOL
        assert labels.iloc[250:370].isin([Regime.BEAR_HIGH_VOL, Regime.BULL_HIGH_VOL]).mean() > 0.9
        assert labels.iloc[-21:].isna().all()

    def test_transition_attribution_and_var(
        self, ablation: AblationResult, bars: pd.DataFrame
    ) -> None:
        head = ablation.results["overlay"]
        mean, std = mean_transition_matrix(head)
        np.testing.assert_allclose(mean.sum(axis=1), 1.0)
        assert (std.to_numpy() >= 0).all()
        attribution = regime_attribution(head, bars["close"])
        assert attribution["share_of_bars"].sum() == pytest.approx(1.0)
        table = var_backtest_table(head)
        v = head.var_forecasts.dropna(subset=["var_fhs", "realized_next"])
        expected = (v["realized_next"] < -v["var_fhs"]).mean()
        assert table.loc["filtered historical", "breach_rate"] == pytest.approx(expected)


# ============================================================================================
# Tear sheet and CLI
# ============================================================================================


def test_tear_sheet_renders(ablation: AblationResult, bars: pd.DataFrame, tmp_path: Path) -> None:
    path = TearSheet(ablation, bars).render(tmp_path / "sheet.html")
    text = path.read_text()
    for heading in ("Performance", "Ablation study", "Regimes", "Risk", "Method"):
        assert f"<h2>{heading}</h2>" in text
    assert text.count("data:image/png;base64,") == 11
    assert "nan%" not in text.lower()
    exported = TearSheet(ablation, bars).export_figures(tmp_path / "img", ["growth", "ablation"])
    assert sorted(exported) == ["ablation", "growth"]
    assert all(p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for p in exported.values())


def test_cli_backtest_and_report(tmp_path: Path) -> None:
    data = _make_ohlcv(800, seed=12)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    data.to_csv(raw_dir / "TEST.csv", index_label="date")
    config = tmp_path / "test.toml"
    config.write_text(
        f"""
[data]
symbol = "TEST"
benchmark = "TEST"
start = 2015-01-01
end = 2019-01-01
source = "csv"
raw_dir = "{raw_dir}"
cache_dir = "{tmp_path / "cache"}"

[training]
min_train_bars = 300
refit_every = 100
train_window = 400

[hmm]
n_restarts = 2
refit_restarts = 1

[risk]
fhs_min_obs = 50

[logging]
console = false
json_file = true
log_dir = "{tmp_path / "logs"}"
"""
    )
    try:
        assert main(["backtest", "--config", str(config), "--out", str(tmp_path / "runs")]) == 0
        runs = list((tmp_path / "runs").iterdir())
        assert len(runs) == 1
        assert (runs[0] / "manifest.json").is_file()
        assert main(["report", "--config", str(config), "--out", str(tmp_path / "reports")]) == 0
        assert (tmp_path / "reports" / "test_tear_sheet.html").is_file()
        assert (tmp_path / "reports" / "test_ablation.csv").is_file()
        events = list((tmp_path / "logs").glob("*/events.jsonl"))
        assert events
        assert "BACKTEST_COMPLETED" in events[0].read_text()
    finally:
        reset_logging()
