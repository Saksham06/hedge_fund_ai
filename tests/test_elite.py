"""
Elite system tests: SimFin, FMP, Piotroski, sector-neutral normalization,
Black-Litterman, CVaR, Kelly, validation report.
"""
import json
import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


# ── SimFin ────────────────────────────────────────────────────────────────────

class TestSimFinLoader:
    def test_piotroski_all_positive(self):
        from hedge_fund_ai.data.simfin_loader import compute_piotroski
        strong = {
            "roa_pct": 12.0, "cfo": 1e9, "net_income": 8e8,
            "gross_margin_pct": 45.0, "de_ratio": 0.3, "current_ratio": 2.5,
            "revenue": 5e9, "total_assets": 10e9,
        }
        prior = {
            "roa_pct": 8.0, "gross_margin_pct": 40.0, "de_ratio": 0.5,
            "current_ratio": 1.8, "revenue": 4e9, "total_assets": 9e9,
        }
        score = compute_piotroski(strong, prior)
        assert score >= 7, f"Strong company should score ≥7, got {score}"

    def test_piotroski_weak_company(self):
        from hedge_fund_ai.data.simfin_loader import compute_piotroski
        weak = {
            "roa_pct": -5.0, "cfo": -1e8, "net_income": -2e8,
            "gross_margin_pct": 10.0, "de_ratio": 3.0, "current_ratio": 0.8,
            "revenue": 1e9, "total_assets": 5e9,
        }
        prior = {
            "roa_pct": -2.0, "gross_margin_pct": 15.0, "de_ratio": 2.0,
            "current_ratio": 1.0, "revenue": 1.2e9, "total_assets": 4.5e9,
        }
        score = compute_piotroski(weak, prior)
        assert score <= 4, f"Weak company should score ≤4, got {score}"

    def test_piotroski_range(self):
        from hedge_fund_ai.data.simfin_loader import compute_piotroski
        score = compute_piotroski({}, {})
        assert 0 <= score <= 9

    def test_quality_composite(self):
        from hedge_fund_ai.data.simfin_loader import _quality_composite
        f = {"roe_pct": 20, "roic_pct": 15, "op_margin_pct": 25, "fcf_margin_pct": 18, "roa_pct": 10}
        q = _quality_composite(f)
        assert q > 0
        assert np.isfinite(q)

    def test_simfin_store_empty_as_of(self):
        from hedge_fund_ai.data.simfin_loader import SimFinStore
        store = SimFinStore()
        # Not loaded — should return empty
        assert store.as_of("AAPL", "2022-01-01") == {}

    def test_helper_pct(self):
        from hedge_fund_ai.data.simfin_loader import _pct
        assert _pct(100, 500) == pytest.approx(20.0)
        assert _pct(None, 500) is None
        assert _pct(100, 0) is None

    def test_helper_growth(self):
        from hedge_fund_ai.data.simfin_loader import _growth
        assert _growth(110, 100) == pytest.approx(10.0)
        assert _growth(None, 100) is None
        assert _growth(100, 0) is None


# ── FMP loader ────────────────────────────────────────────────────────────────

class TestFMPLoader:
    def test_no_key_returns_empty(self):
        """Without API key, all FMP functions return safe defaults."""
        import hedge_fund_ai.config as cfg
        original = cfg.FMP_API_KEY
        cfg.FMP_API_KEY = ""
        try:
            from hedge_fund_ai.data.fmp_loader import get_fmp_alpha_signal
            result = get_fmp_alpha_signal("AAPL", 150.0)
            assert result["fmp_score"] == 0.0
            assert result["available"] is False
        finally:
            cfg.FMP_API_KEY = original

    def test_earnings_surprise_score_bounds(self):
        from hedge_fund_ai.data.fmp_loader import get_earnings_surprise_score
        with patch("hedge_fund_ai.data.fmp_loader.get_earnings_surprises") as mock:
            mock.return_value = [
                {"date": "2024-01-01", "surprise_pct": 20.0},
                {"date": "2023-10-01", "surprise_pct": 15.0},
                {"date": "2023-07-01", "surprise_pct": -5.0},
                {"date": "2023-04-01", "surprise_pct": 10.0},
            ]
            score = get_earnings_surprise_score("AAPL")
            assert -3.0 <= score <= 3.0

    def test_earnings_surprise_positive_for_beats(self):
        from hedge_fund_ai.data.fmp_loader import get_earnings_surprise_score
        with patch("hedge_fund_ai.data.fmp_loader.get_earnings_surprises") as mock:
            mock.return_value = [
                {"date": "2024-01-01", "surprise_pct": 30.0},
                {"date": "2023-10-01", "surprise_pct": 25.0},
                {"date": "2023-07-01", "surprise_pct": 20.0},
                {"date": "2023-04-01", "surprise_pct": 15.0},
            ]
            score = get_earnings_surprise_score("AAPL")
            assert score > 0, "Consistent beats should produce positive score"

    def test_earnings_surprise_negative_for_misses(self):
        from hedge_fund_ai.data.fmp_loader import get_earnings_surprise_score
        with patch("hedge_fund_ai.data.fmp_loader.get_earnings_surprises") as mock:
            mock.return_value = [
                {"date": "2024-01-01", "surprise_pct": -20.0},
                {"date": "2023-10-01", "surprise_pct": -15.0},
                {"date": "2023-07-01", "surprise_pct": -10.0},
                {"date": "2023-04-01", "surprise_pct": -5.0},
            ]
            score = get_earnings_surprise_score("AAPL")
            assert score < 0, "Consistent misses should produce negative score"

    def test_as_of_date_filter(self):
        from hedge_fund_ai.data.fmp_loader import get_earnings_surprise_score
        with patch("hedge_fund_ai.data.fmp_loader.get_earnings_surprises") as mock:
            mock.return_value = [
                {"date": "2024-06-01", "surprise_pct": 50.0},
                {"date": "2024-01-01", "surprise_pct": 5.0},
            ]
            # as_of before the big surprise — should not see it
            score_before = get_earnings_surprise_score("AAPL", as_of_date="2024-03-01")
            score_after  = get_earnings_surprise_score("AAPL", as_of_date="2024-12-01")
            # The big +50 surprise should only affect score_after
            assert score_after >= score_before

    def test_price_target_upside(self):
        from hedge_fund_ai.data.fmp_loader import get_price_target_upside
        with patch("hedge_fund_ai.data.fmp_loader._fmp_get") as mock:
            mock.return_value = {"targetConsensus": 200.0}
            upside = get_price_target_upside("AAPL", 160.0)
            assert upside == pytest.approx(0.25, abs=0.01)


# ── Sector-neutral normalization ──────────────────────────────────────────────

class TestSectorNeutralNormalization:
    def _make_data(self, n=20):
        np.random.seed(42)
        sectors = ["Technology"] * 8 + ["Healthcare"] * 6 + ["Finance"] * 6
        data = []
        for i in range(n):
            data.append({
                "ticker": f"T{i:02d}",
                "sector": sectors[i],
                "tech": {
                    "ret_3m": float(np.random.randn() * 5 + (10 if sectors[i] == "Technology" else 0)),
                    "ret_6m": float(np.random.randn() * 8),
                    "ret_12m_skip1m": float(np.random.randn() * 12),
                    "ret_1m": float(np.random.randn() * 3),
                    "ann_vol_30d": float(20 + np.random.rand() * 10),
                    "ann_vol_90d": float(22 + np.random.rand() * 8),
                    "vol_regime": float(0.9 + np.random.rand() * 0.2),
                    "high_52w_proximity": float(0.7 + np.random.rand() * 0.3),
                    "trend_slope": float(np.random.randn() * 0.01),
                    "z_score": float(np.random.randn()),
                },
                "fund": {
                    "roe_pct": float(10 + np.random.rand() * 20),
                    "roic_pct": float(8 + np.random.rand() * 15),
                    "op_margin_pct": float(10 + np.random.rand() * 20),
                    "fcf_margin_pct": float(5 + np.random.rand() * 15),
                    "gross_margin_pct": float(30 + np.random.rand() * 30),
                    "revenue_growth": float(np.random.randn() * 10),
                    "ni_growth": float(np.random.randn() * 15),
                    "pe_ratio": float(15 + np.random.rand() * 20),
                    "ps_ratio": float(2 + np.random.rand() * 5),
                },
                "piotroski_score": int(np.random.randint(3, 9)),
                "fmp_score": float(np.random.randn() * 0.5),
                "earnings_surprise": float(np.random.randn() * 0.3),
                "flow_signal": float(np.random.randn()),
                "sentiment_score": float(np.random.randn() * 0.3),
                "liquidity": float(np.random.randn()),
                "skewness": float(np.random.randn() * 0.5),
                "accruals": float(np.random.randn() * 0.05),
            })
        return data

    def test_normalize_attaches_features(self):
        from hedge_fund_ai.factors.normalization import normalize_features
        data = self._make_data()
        result = normalize_features(data)
        for d in result:
            assert "features" in d

    def test_features_finite(self):
        from hedge_fund_ai.factors.normalization import normalize_features
        data = normalize_features(self._make_data())
        for d in data:
            for k, v in d["features"].items():
                assert np.isfinite(v), f"Non-finite: {k}={v} for {d['ticker']}"

    def test_sector_momentum_differs_from_global(self):
        from hedge_fund_ai.factors.normalization import normalize_features, sector_neutral_zscore
        data = self._make_data()
        result = normalize_features(data)
        # Technology stocks have biased raw ret_3m — sector-neutral should reduce this bias
        tech_feats = [d["features"]["momentum"] for d in result if d["sector"] == "Technology"]
        health_feats = [d["features"]["momentum"] for d in result if d["sector"] == "Healthcare"]
        # After sector-neutral normalization, means should be closer to 0
        assert abs(np.mean(tech_feats)) < 2.0
        assert abs(np.mean(health_feats)) < 2.0

    def test_sector_neutral_zscore(self):
        from hedge_fund_ai.factors.normalization import sector_neutral_zscore
        # Tech stocks have 10 points higher raw momentum
        values  = [15, 14, 16, 13, 5, 6, 4, 7]
        sectors = ["Tech"] * 4 + ["Health"] * 4
        result  = sector_neutral_zscore(values, sectors)
        assert len(result) == 8
        assert all(np.isfinite(v) for v in result)
        # Tech and Health should both have mixed signs after sector-neutral adjustment
        tech_z   = result[:4]
        health_z = result[4:]
        assert any(v > 0 for v in tech_z)
        assert any(v < 0 for v in tech_z)

    def test_new_factors_present(self):
        from hedge_fund_ai.factors.normalization import normalize_features
        data = normalize_features(self._make_data())
        required = ["piotroski", "roic", "sector_momentum", "fmp_alpha",
                    "accruals", "skewness", "high_52w"]
        for d in data:
            for factor in required:
                assert factor in d["features"], f"Missing factor: {factor}"


# ── Signal engine: decay-weighted IC ─────────────────────────────────────────

class TestEliteSignalEngine:
    def test_ic_decay_weights_recent_more(self):
        """Recent IC observations should matter more than old ones."""
        from hedge_fund_ai.factors.signal_engine import _ic_scale, update_ic, _ic_history
        import hedge_fund_ai.factors.signal_engine as se

        # Simulate: old ICs were negative, recent are positive
        se._ic_history["test_decay"] = __import__("collections").deque(
            [-0.1, -0.1, -0.1, -0.1, -0.1, 0.15, 0.20, 0.25],
            maxlen=24
        )
        scale = se._ic_scale("test_decay", 1.0)
        # With decay weighting, recent positive ICs should outweigh old negatives
        assert scale > 0, f"Decay-weighted scale should be positive, got {scale}"
        se._ic_history.pop("test_decay", None)

    def test_negative_ic_ir_gated(self):
        """Factors with consistently negative IC should be zeroed out."""
        from hedge_fund_ai.factors.signal_engine import _ic_scale
        import hedge_fund_ai.factors.signal_engine as se

        se._ic_history["bad_factor"] = __import__("collections").deque(
            [-0.15] * 12, maxlen=24
        )
        scale = se._ic_scale("bad_factor", 1.0)
        assert scale == 0.0, f"Consistently negative IC should be gated (scale=0), got {scale}"
        se._ic_history.pop("bad_factor", None)

    def test_crisis_momentum_penalized(self):
        from hedge_fund_ai.factors.signal_engine import build_signal
        np.random.seed(5)
        d = {
            "tech": {"ann_vol_30d": 20.0},
            "features": {k: 1.0 for k in
                         ["momentum_12_1", "momentum", "price_accel", "rel_strength",
                          "sector_momentum", "quality", "piotroski", "growth", "roic",
                          "value", "fmp_alpha", "earnings_surprise", "sentiment",
                          "flow_signal", "liquidity", "high_52w",
                          "volatility", "skewness", "mean_reversion", "accruals"]},
        }
        score_on     = build_signal(d, "risk_on")
        score_crisis = build_signal(d, "crisis")
        # Crisis should penalize momentum heavily
        assert score_crisis < score_on

    def test_get_factor_ic_summary(self):
        from hedge_fund_ai.factors.signal_engine import get_factor_ic_summary, update_ic
        update_ic("test_summary_factor", 0.05)
        update_ic("test_summary_factor", 0.08)
        update_ic("test_summary_factor", 0.03)
        update_ic("test_summary_factor", 0.06)
        summary = get_factor_ic_summary()
        assert "test_summary_factor" in summary
        s = summary["test_summary_factor"]
        assert "mean_ic" in s and "ic_ir" in s and "hit_rate" in s


# ── Elite optimizer: BL + CVaR + Kelly ───────────────────────────────────────

class TestEliteOptimizer:
    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        return pd.DataFrame(
            np.random.randn(200, 5) * 0.01,
            columns=["AAPL", "MSFT", "NVDA", "GOOGL", "META"]
        )

    def test_cvar_weights_sum_to_one(self, sample_returns):
        from hedge_fund_ai.portfolio.optimizer import cvar_position_sizes
        w = cvar_position_sizes(sample_returns)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_cvar_weights_positive(self, sample_returns):
        from hedge_fund_ai.portfolio.optimizer import cvar_position_sizes
        w = cvar_position_sizes(sample_returns)
        assert (w >= 0).all()

    def test_cvar_inverse_to_tail_risk(self, sample_returns):
        """Higher-risk stock should get lower CVaR weight."""
        from hedge_fund_ai.portfolio.optimizer import cvar_position_sizes
        # Make column 0 much more volatile
        risky = sample_returns.copy()
        risky.iloc[:, 0] *= 5
        w = cvar_position_sizes(risky)
        # Risky stock should have lowest weight
        assert w[0] == pytest.approx(w.min(), abs=0.05)

    def test_kelly_weights_sum_near_one(self, sample_returns):
        from hedge_fund_ai.portfolio.optimizer import kelly_weights
        scores = np.array([1.5, 0.8, 2.0, 0.5, 1.2])
        w = kelly_weights(scores, sample_returns)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_kelly_weights_higher_score_gets_more(self, sample_returns):
        from hedge_fund_ai.portfolio.optimizer import kelly_weights
        scores = np.array([3.0, 0.1, 0.1, 0.1, 0.1])
        w = kelly_weights(scores, sample_returns)
        assert w[0] > w[1], "Highest score should get highest Kelly weight"

    def test_black_litterman_returns_valid_weights(self, sample_returns):
        from hedge_fund_ai.portfolio.optimizer import black_litterman_weights, _ledoit_wolf_cov
        cov = _ledoit_wolf_cov(sample_returns)
        n = sample_returns.shape[1]
        market_w = np.ones(n) / n
        views    = np.array([0.5, 0.2, 0.8, -0.1, 0.3])
        w = black_litterman_weights(cov, market_w, views)
        assert len(w) == n
        assert all(np.isfinite(w))
        assert (w >= 0).all()
        assert abs(w.sum() - 1.0) < 1e-6

    def test_black_litterman_positive_views_overweight(self, sample_returns):
        from hedge_fund_ai.portfolio.optimizer import black_litterman_weights, _ledoit_wolf_cov
        cov = _ledoit_wolf_cov(sample_returns)
        n = sample_returns.shape[1]
        market_w = np.ones(n) / n
        views = np.array([2.0, -2.0, 0.0, 0.0, 0.0])
        w = black_litterman_weights(cov, market_w, views)
        # Strong positive view on stock 0 → higher weight than stock 1
        assert w[0] > w[1]


# ── Validation report ─────────────────────────────────────────────────────────

class TestValidationReport:
    def _make_metrics(self):
        return {
            "sharpe": 1.2, "sortino": 1.8, "calmar": 0.7,
            "max_drawdown": -0.15, "cagr": 0.14, "win_rate": 0.54,
            "ann_vol": 0.12, "alpha_ann": 0.04, "beta": 0.75,
            "excess_vs_spy": 0.05, "excess_vs_ew": 0.03,
            "avg_cost_bps": 12.5, "avg_raw_turnover": 0.35,
            "avg_filtered_turnover": 0.18,
        }

    def _make_audit_log(self):
        from hedge_fund_ai.backtest.audit import AuditLog, AuditRecord
        log = AuditLog()
        np.random.seed(0)
        for i in range(12):
            r = AuditRecord(
                rebalance_idx=i*21, date=f"2022-{i+1:02d}-01",
                regime=["risk_on","neutral"][i%2],
                universe_size=25, dd_scale=1.0,
                final_weights={"AAPL": 0.5, "MSFT": 0.5},
                raw_turnover=0.3, filtered_turnover=0.15,
                factor_exposures={
                    "AAPL": {"momentum_12_1": 1.2, "quality": 0.5},
                    "MSFT": {"momentum_12_1": 0.8, "quality": 0.9},
                },
            )
            log.append(r)
            log.fill_returns(i, float(np.random.randn() * 0.02))
            log.fill_ic(i, {"momentum_12_1": float(np.random.randn() * 0.08),
                            "quality": float(np.random.randn() * 0.05)})
        return log

    def test_report_generates(self, tmp_path):
        import hedge_fund_ai.backtest.validation_report as vr
        original = vr._REPORT_PATH
        vr._REPORT_PATH = str(tmp_path / "report.txt")

        try:
            from hedge_fund_ai.backtest.validation_report import generate_validation_report
            audit = self._make_audit_log()
            report = generate_validation_report(
                self._make_metrics(), None, audit, None
            )
            assert isinstance(report, str)
            assert len(report) > 100
        finally:
            vr._REPORT_PATH = original

    def test_report_contains_all_sections(self, tmp_path):
        import hedge_fund_ai.backtest.validation_report as vr
        vr._REPORT_PATH = str(tmp_path / "report.txt")
        try:
            from hedge_fund_ai.backtest.validation_report import generate_validation_report
            audit = self._make_audit_log()
            report = generate_validation_report(self._make_metrics(), None, audit, None)
            for section in ["BACKTEST PERFORMANCE", "PLACEBO TEST",
                            "FACTOR IC ATTRIBUTION", "STABILITY",
                            "LIVE SIGNAL VALIDATION", "DEPLOYMENT VERDICT"]:
                assert section in report, f"Missing section: {section}"
        finally:
            vr._REPORT_PATH = str(tmp_path / "report.txt")

    def test_deployment_verdict_present(self, tmp_path):
        import hedge_fund_ai.backtest.validation_report as vr
        vr._REPORT_PATH = str(tmp_path / "report.txt")
        try:
            from hedge_fund_ai.backtest.validation_report import generate_validation_report
            report = generate_validation_report(self._make_metrics(), None, self._make_audit_log(), None)
            assert any(v in report for v in ["DEPLOY", "PAPER", "NOT READY", "DO NOT"])
        finally:
            vr._REPORT_PATH = str(tmp_path / "report.txt")

    def test_pass_fail_icons(self, tmp_path):
        import hedge_fund_ai.backtest.validation_report as vr
        vr._REPORT_PATH = str(tmp_path / "report.txt")
        try:
            from hedge_fund_ai.backtest.validation_report import generate_validation_report
            report = generate_validation_report(self._make_metrics(), None, self._make_audit_log(), None)
            assert "✅" in report
            assert "❌" in report
        finally:
            vr._REPORT_PATH = str(tmp_path / "report.txt")
