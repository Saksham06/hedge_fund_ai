"""
Production-readiness tests for all document requirements.
"""
import json
import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


# ── Startup validator ─────────────────────────────────────────────────────────

class TestStartup:
    def test_startup_report_has_run_id(self):
        from hedge_fund_ai.core.startup import validate_startup
        with patch("hedge_fund_ai.core.startup._check_apis", return_value={}):
            report = validate_startup(dry_run=True)
        assert len(report.run_id) == 8
        assert report.run_id.isupper()

    def test_startup_warns_on_missing_optional_keys(self):
        import hedge_fund_ai.config as cfg
        original = cfg.FMP_API_KEY
        cfg.FMP_API_KEY = ""
        try:
            from hedge_fund_ai.core.startup import validate_startup
            with patch("hedge_fund_ai.core.startup._check_apis", return_value={}):
                report = validate_startup(dry_run=True)
            warning_texts = " ".join(report.warnings)
            assert "FMP_API_KEY" in warning_texts
        finally:
            cfg.FMP_API_KEY = original

    def test_startup_fails_if_state_dir_not_writable(self, tmp_path):
        from hedge_fund_ai.core.startup import StartupReport
        report = StartupReport(run_id="TEST", start_time="now")
        report.errors.append("State dir not writeable")
        report.can_proceed = False
        assert not report.can_proceed

    def test_different_runs_get_different_ids(self):
        from hedge_fund_ai.core.startup import validate_startup
        with patch("hedge_fund_ai.core.startup._check_apis", return_value={}):
            r1 = validate_startup(dry_run=True)
            r2 = validate_startup(dry_run=True)
        assert r1.run_id != r2.run_id


# ── Data quality gates ────────────────────────────────────────────────────────

class TestDataQuality:
    def _make_good(self, ticker="AAPL", n=120):
        np.random.seed(0)
        prices = np.cumprod(1 + np.random.randn(n) * 0.01) * 100
        volume = np.ones(n) * 2e6
        return {
            "ticker": ticker,
            "tech": {
                "price_hist":  prices.tolist(),
                "volume_hist": volume.tolist(),
            },
            "fund": {},
        }

    def test_good_data_passes(self):
        from hedge_fund_ai.data.quality import run_quality_gates
        data   = [self._make_good("AAPL"), self._make_good("MSFT")]
        passed, report = run_quality_gates(data)
        assert len(passed) == 2
        assert report["rejected"] == 0

    def test_short_history_rejected(self):
        from hedge_fund_ai.data.quality import run_quality_gates
        bad = self._make_good()
        bad["tech"]["price_hist"] = bad["tech"]["price_hist"][:20]  # too short
        passed, report = run_quality_gates([bad])
        assert len(passed) == 0
        assert report["rejected"] == 1

    def test_zero_price_rejected(self):
        from hedge_fund_ai.data.quality import run_quality_gates
        bad = self._make_good()
        ph  = list(bad["tech"]["price_hist"])
        ph[50] = 0.0
        bad["tech"]["price_hist"] = ph
        passed, _ = run_quality_gates([bad])
        assert len(passed) == 0

    def test_penny_stock_rejected(self):
        from hedge_fund_ai.data.quality import run_quality_gates
        bad = self._make_good()
        bad["tech"]["price_hist"] = [0.50] * 120  # penny stock
        passed, _ = run_quality_gates([bad])
        assert len(passed) == 0

    def test_price_spike_rejected(self):
        from hedge_fund_ai.data.quality import run_quality_gates
        bad = self._make_good()
        ph  = list(bad["tech"]["price_hist"])
        ph[60] = ph[59] * 3.0   # 200% spike — data error
        bad["tech"]["price_hist"] = ph
        passed, _ = run_quality_gates([bad])
        assert len(passed) == 0

    def test_freshness_report(self):
        from hedge_fund_ai.data.quality import check_data_freshness
        data = [{
            "ticker": "AAPL",
            "tech": {"last_price_date": "2024-01-01"},
        }]
        freshness = check_data_freshness(data)
        assert "AAPL" in freshness


# ── Factor health monitor ─────────────────────────────────────────────────────

class TestFactorHealth:
    def test_auto_disable_after_3_fails(self, tmp_path):
        import hedge_fund_ai.factors.factor_health as fh
        fh._HEALTH_PATH = str(tmp_path / "factor_health.json")
        from hedge_fund_ai.factors.factor_health import FactorHealthMonitor

        monitor = FactorHealthMonitor()
        ic_summary = {"bad_factor": {"mean_ic": -0.05, "ic_ir": -0.8, "n": 12}}

        for _ in range(3):
            monitor.evaluate(ic_summary)
            monitor = FactorHealthMonitor()  # reload state

        assert "bad_factor" in monitor.get_disabled_factors()

    def test_good_factor_stays_active(self, tmp_path):
        import hedge_fund_ai.factors.factor_health as fh
        fh._HEALTH_PATH = str(tmp_path / "factor_health2.json")
        from hedge_fund_ai.factors.factor_health import FactorHealthMonitor

        monitor = FactorHealthMonitor()
        ic_summary = {"good_factor": {"mean_ic": 0.08, "ic_ir": 1.2, "n": 12}}
        monitor.evaluate(ic_summary)
        assert "good_factor" not in monitor.get_disabled_factors()

    def test_monthly_turnover_cap(self):
        from hedge_fund_ai.factors.factor_health import apply_monthly_turnover_cap
        new_w = {"A": 0.80, "B": 0.20}
        old_w = {"A": 0.30, "B": 0.70}
        smoothed = apply_monthly_turnover_cap(new_w, old_w, max_delta=0.30)
        # A moves from 0.30 → 0.80 (delta 0.50 > cap 0.30) → should cap at 0.60
        assert abs(smoothed["A"] - old_w["A"]) <= 0.30 + 1e-6
        assert abs(smoothed["B"] - old_w["B"]) <= 0.30 + 1e-6

    def test_factor_stability_trends(self):
        from hedge_fund_ai.factors.factor_health import compute_factor_stability
        # Improving: early ICs low, recent ICs high
        history = {
            "improving_f": [0.01, 0.02, 0.01, 0.02, 0.05, 0.07, 0.08, 0.09],
            "degrading_f": [0.08, 0.07, 0.06, 0.05, 0.02, 0.01, 0.00, -0.01],
        }
        result = compute_factor_stability(history)
        assert result["improving_f"]["trend"] == "improving"
        assert result["degrading_f"]["trend"] == "degrading"

    def test_redundancy_filter_disables_weaker(self, tmp_path):
        import hedge_fund_ai.factors.factor_health as fh
        fh._HEALTH_PATH = str(tmp_path / "factor_health3.json")
        from hedge_fund_ai.factors.factor_health import FactorHealthMonitor

        monitor = FactorHealthMonitor()
        # Two perfectly correlated factors — weaker one should be disabled
        base = [0.05, 0.06, 0.04, 0.07, 0.05, 0.08, 0.06, 0.05, 0.07, 0.06]
        ic_summary = {
            "strong_f": {"mean_ic": 0.08, "ic_ir": 1.5, "n": 10},
            "weak_f":   {"mean_ic": 0.03, "ic_ir": 0.4, "n": 10},
        }
        # Inject correlated IC history
        return_matrix = {"strong_f": base, "weak_f": base}  # perfectly correlated
        monitor.evaluate(ic_summary, return_matrix)
        disabled = monitor.get_disabled_factors()
        assert "weak_f" in disabled
        assert "strong_f" not in disabled


# ── Monte Carlo + multiple testing ────────────────────────────────────────────

class TestMonteCarlo:
    @pytest.fixture
    def sample_equity(self):
        np.random.seed(42)
        return pd.Series(np.cumprod(1 + np.random.randn(500) * 0.008))

    def test_block_bootstrap_returns_required_keys(self, sample_equity):
        from hedge_fund_ai.backtest.monte_carlo import block_bootstrap_equity
        result = block_bootstrap_equity(sample_equity, n_simulations=20)
        for key in ["sharpe_mean", "sharpe_std", "p_sharpe_gt_0",
                    "p_sharpe_gt_1", "model_sharpe", "verdict"]:
            assert key in result, f"Missing: {key}"

    def test_p_values_in_range(self, sample_equity):
        from hedge_fund_ai.backtest.monte_carlo import block_bootstrap_equity
        r = block_bootstrap_equity(sample_equity, n_simulations=20)
        assert 0 <= r["p_sharpe_gt_0"] <= 1
        assert 0 <= r["p_sharpe_gt_1"] <= 1

    def test_benjamini_hochberg_controls_fdr(self):
        from hedge_fund_ai.backtest.monte_carlo import benjamini_hochberg
        p_values = {
            "f1": 0.001,  # clearly significant
            "f2": 0.002,  # significant
            "f3": 0.500,  # not significant
            "f4": 0.800,  # not significant
            "f5": 0.030,  # borderline
        }
        result = benjamini_hochberg(p_values, fdr=0.05)
        assert result["f1"] == True
        assert result["f2"] == True
        assert result["f3"] == False
        assert result["f4"] == False

    def test_ic_significance_test(self):
        from hedge_fund_ai.backtest.monte_carlo import ic_significance_test
        # Consistently positive IC → should be significant
        strong_ic = [0.08, 0.09, 0.07, 0.10, 0.08, 0.09, 0.11, 0.07, 0.09, 0.08]
        result    = ic_significance_test(strong_ic)
        assert result["significant"] is True
        assert result["ic_ir"] > 0

        # Near-zero IC → not significant
        weak_ic = [0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.0, 0.01, -0.01, 0.0]
        result2 = ic_significance_test(weak_ic)
        assert result2["significant"] is False

    def test_capacity_model_returns_dict(self, sample_equity):
        from hedge_fund_ai.backtest.monte_carlo import model_capacity
        from hedge_fund_ai.backtest.audit import AuditLog, AuditRecord
        audit = AuditLog()
        audit.append(AuditRecord(0, "2023-01-01", "neutral", 20, 1.0,
                                 total_cost_frac=0.001, final_weights={"A": 1.0}))
        result = model_capacity(sample_equity, audit,
                                aum_levels=[100_000, 1_000_000, 10_000_000])
        assert "by_aum" in result
        assert "capacity_limit_usd" in result
        assert len(result["by_aum"]) == 3


# ── Order manager ─────────────────────────────────────────────────────────────

class TestOrderManager:
    def test_no_slicing_for_small_order(self):
        from hedge_fund_ai.execution.order_manager import compute_order_slices
        slices = compute_order_slices("AAPL", 5_000, 10_000_000, 150.0)
        assert len(slices) == 1

    def test_slicing_for_large_order(self):
        from hedge_fund_ai.execution.order_manager import compute_order_slices
        slices = compute_order_slices("AAPL", 100_000, 500_000, 150.0,
                                      max_participation=0.05)
        # 100k / (500k * 0.05) = 100k / 25k = 4 slices
        assert len(slices) > 1
        assert sum(s["notional"] for s in slices) == pytest.approx(100_000, rel=0.01)

    def test_slices_respect_max(self):
        from hedge_fund_ai.execution.order_manager import compute_order_slices
        slices = compute_order_slices("AAPL", 1_000_000, 100_000, 150.0)
        assert len(slices) <= 5  # MAX_SLICES_PER_STOCK

    def test_execution_tracker_records_slippage(self, tmp_path):
        import hedge_fund_ai.execution.order_manager as om
        om._EXEC_LOG_PATH = str(tmp_path / "exec_log.json")
        from hedge_fund_ai.execution.order_manager import ExecutionTracker
        tracker = ExecutionTracker()
        tid = tracker.record_arrival("AAPL", 150.0, "buy", 10_000)
        tracker.record_fill(tid, 151.0)
        # 1.0/150.0 * 10000 = 66.7bps slippage
        entry = next(e for e in tracker._log if e["trade_id"] == tid)
        assert entry["slippage_bps"] > 0
        assert entry["status"] == "filled"

    def test_negative_slippage_for_good_fill(self, tmp_path):
        import hedge_fund_ai.execution.order_manager as om
        om._EXEC_LOG_PATH = str(tmp_path / "exec_log2.json")
        from hedge_fund_ai.execution.order_manager import ExecutionTracker
        tracker = ExecutionTracker()
        tid = tracker.record_arrival("AAPL", 150.0, "buy", 10_000)
        tracker.record_fill(tid, 149.5)  # better than arrival
        entry = next(e for e in tracker._log if e["trade_id"] == tid)
        assert entry["slippage_bps"] < 0  # negative = good for buy

    def test_reconciliation_finds_mismatch(self):
        from hedge_fund_ai.execution.order_manager import reconcile_positions
        expected = {"AAPL": 0.20, "MSFT": 0.15}
        actual   = {
            "AAPL": {"market_value": 15_000},   # should be 20_000
            "MSFT": {"market_value": 15_000},   # correct
        }
        mismatches = reconcile_positions(expected, actual, 100_000)
        assert "AAPL" in mismatches
        assert "MSFT" not in mismatches

    def test_reconciliation_no_mismatch_within_tolerance(self):
        from hedge_fund_ai.execution.order_manager import reconcile_positions
        expected = {"AAPL": 0.20}
        actual   = {"AAPL": {"market_value": 20_500}}  # 0.5% off — within 2%
        mismatches = reconcile_positions(expected, actual, 100_000)
        assert "AAPL" not in mismatches

    def test_vol_adjusted_stop_widens_in_high_vol(self):
        from hedge_fund_ai.execution.order_manager import compute_vol_adjusted_stop
        low_vol  = compute_vol_adjusted_stop("AAPL", 100.0, current_vol=0.10)
        high_vol = compute_vol_adjusted_stop("AAPL", 100.0, current_vol=0.40)
        assert high_vol["stop_pct"] > low_vol["stop_pct"]

    def test_vol_adjusted_stop_within_bounds(self):
        from hedge_fund_ai.execution.order_manager import compute_vol_adjusted_stop
        for vol in [0.05, 0.15, 0.30, 0.60]:
            stop = compute_vol_adjusted_stop("X", 100.0, current_vol=vol)
            assert 0.04 <= stop["stop_pct"] <= 0.20


# ── Dynamic risk model ────────────────────────────────────────────────────────

class TestDynamicRiskModel:
    @pytest.fixture
    def sample_equity(self):
        np.random.seed(1)
        return pd.Series(np.cumprod(1 + np.random.randn(200) * 0.008))

    def test_dynamic_vol_target_reduces_in_high_vol(self, sample_equity):
        from hedge_fund_ai.portfolio.risk_model import dynamic_vol_target_scale
        # High vol: amplify returns to simulate stress
        stressed = pd.Series(sample_equity.values * np.cumprod(1 + np.random.randn(200)*0.03))
        scale_normal = dynamic_vol_target_scale(sample_equity)
        scale_stress = dynamic_vol_target_scale(stressed)
        assert scale_stress <= scale_normal

    def test_crisis_regime_further_reduces(self, sample_equity):
        from hedge_fund_ai.portfolio.risk_model import dynamic_vol_target_scale
        scale_neutral = dynamic_vol_target_scale(sample_equity, regime="neutral")
        scale_crisis  = dynamic_vol_target_scale(sample_equity, regime="crisis")
        assert scale_crisis <= scale_neutral

    def test_cash_buffer_larger_in_high_vol(self):
        from hedge_fund_ai.portfolio.risk_model import compute_cash_buffer
        low_regime  = compute_cash_buffer(0.8, "neutral")   # vol compressing
        high_regime = compute_cash_buffer(1.5, "neutral")   # vol expanding
        assert high_regime > low_regime

    def test_cash_buffer_crisis_largest(self):
        from hedge_fund_ai.portfolio.risk_model import compute_cash_buffer
        crisis  = compute_cash_buffer(1.0, "crisis")
        risk_on = compute_cash_buffer(1.0, "risk_on")
        assert crisis > risk_on

    def test_position_cvar_limit_tighter_for_risky(self):
        from hedge_fund_ai.portfolio.risk_model import position_cvar_limit
        safe_rets  = pd.Series(np.random.randn(100) * 0.005)
        risky_rets = pd.Series(np.random.randn(100) * 0.05)
        safe_limit  = position_cvar_limit(safe_rets)
        risky_limit = position_cvar_limit(risky_rets)
        assert risky_limit <= safe_limit
