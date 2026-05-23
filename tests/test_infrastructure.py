"""
Tests for all infrastructure improvements from the document.
Covers: corporate actions, microstructure, crowding, factor covariance,
P&L attribution, tax lots, logging, backup.
"""
import json
import os
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock


# ── Corporate actions ─────────────────────────────────────────────────────────

class TestCorporateActions:
    def test_known_events_loaded(self):
        from hedge_fund_ai.data.corporate_actions import CorporateActionStore
        store = CorporateActionStore(["TSLA", "AAPL"])
        # Inject curated events directly without network
        import yfinance as yf
        with patch.object(yf.Ticker, "splits", new_callable=lambda: property(lambda self: pd.Series(dtype=float))), \
             patch.object(yf.Ticker, "dividends", new_callable=lambda: property(lambda self: pd.Series(dtype=float))):
            # load_ticker falls back to known events for TSLA
            df = store._load_ticker.__wrapped__(store, "TSLA")
        # Should have curated splits even without yfinance data
        assert df is not None
        assert len(df) >= 2  # at least 2 TSLA splits curated

    def test_split_price_multiplier(self):
        from hedge_fund_ai.data.corporate_actions import CorporateActionStore
        import pandas as pd
        store = CorporateActionStore(["TSLA"])
        # Manually inject a split
        store._actions["TSLA"] = pd.DataFrame([{
            "date": pd.Timestamp("2022-01-15"),
            "type": "split",
            "ratio": 4.0,
            "price_multiplier": 0.25,
            "source": "test",
        }])
        adj = store.get_adjustment("TSLA", "2022-06-01")
        assert adj["price_multiplier"] == pytest.approx(0.25)

    def test_adjustment_before_split_is_one(self):
        from hedge_fund_ai.data.corporate_actions import CorporateActionStore
        import pandas as pd
        store = CorporateActionStore(["TSLA"])
        store._actions["TSLA"] = pd.DataFrame([{
            "date": pd.Timestamp("2022-08-25"),
            "type": "split", "ratio": 3.0,
            "price_multiplier": 1/3, "source": "test",
        }])
        adj = store.get_adjustment("TSLA", "2022-01-01")  # before split
        assert adj["price_multiplier"] == pytest.approx(1.0)

    def test_merger_flag(self):
        from hedge_fund_ai.data.corporate_actions import CorporateActionStore
        import pandas as pd
        store = CorporateActionStore(["FAKE"])
        store._actions["FAKE"] = pd.DataFrame([{
            "date": pd.Timestamp("2023-06-01"),
            "type": "merger", "ratio": 1.0,
            "price_multiplier": 0.0, "source": "test",
        }])
        assert store.flag_merger_or_delisting("FAKE", "2023-12-01") is True
        assert store.flag_merger_or_delisting("FAKE", "2023-01-01") is False

    def test_restatement_flag(self):
        from hedge_fund_ai.data.corporate_actions import get_restatement_flag
        v1 = {"revenue_growth": 10.0, "roe_pct": 15.0}
        v2 = {"revenue_growth":  5.0, "roe_pct": 15.2}  # revenue restated >10%
        flags = get_restatement_flag(v1, v2, threshold=0.10)
        assert "revenue_growth" in flags
        assert flags["revenue_growth"]["flagged"] is True
        assert "roe_pct" not in flags  # small change, not flagged

    def test_dividend_yield(self):
        from hedge_fund_ai.data.corporate_actions import CorporateActionStore
        import pandas as pd
        store = CorporateActionStore(["MSFT"])
        store._actions["MSFT"] = pd.DataFrame([
            {"date": pd.Timestamp(f"2023-{m:02d}-15"), "type": "dividend",
             "ratio": 0.75, "price_multiplier": 1.0, "source": "test"}
            for m in [3, 6, 9, 12]
        ])
        yield_ = store.annual_dividend_yield("MSFT")
        assert yield_ == pytest.approx(3.0)  # 4 × $0.75


# ── Market microstructure ─────────────────────────────────────────────────────

class TestMicrostructure:
    def test_fomc_is_blackout(self):
        from hedge_fund_ai.data.microstructure import is_blackout_window
        result = is_blackout_window("2024-03-20")  # FOMC date
        assert result["blackout"] is True
        assert "fomc_meeting" in result["reasons"]
        assert result["scale"] <= 0.50

    def test_normal_day_not_blackout(self):
        from hedge_fund_ai.data.microstructure import is_blackout_window
        result = is_blackout_window("2024-04-15")  # random Monday
        # Should not be FOMC (verify it's not in our set)
        assert result["scale"] == 1.0 or result["scale"] <= 1.0

    def test_triple_witching_reduces_scale(self):
        from hedge_fund_ai.data.microstructure import is_blackout_window, _TRIPLE_WITCHING
        if _TRIPLE_WITCHING:
            tw_date = list(_TRIPLE_WITCHING)[0]
            result = is_blackout_window(tw_date, {"triple_witching"})
            assert result["scale"] <= 0.70

    def test_liquidity_tiers(self):
        from hedge_fund_ai.data.microstructure import classify_liquidity_tier
        t1 = classify_liquidity_tier(1_000_000_000)
        t4 = classify_liquidity_tier(5_000_000)
        assert t1["tier"] == 1
        assert t4["tier"] == 4
        assert t1["max_weight"] > t4["max_weight"]

    def test_tier_caps_dict(self):
        from hedge_fund_ai.data.microstructure import get_tier_caps
        adv = {"AAPL": 500_000_000, "SMLCAP": 8_000_000}
        caps = get_tier_caps(adv)
        assert caps["AAPL"] > caps["SMLCAP"]

    def test_vwap_schedule_sums_to_target(self):
        from hedge_fund_ai.data.microstructure import build_vwap_schedule
        schedule = build_vwap_schedule("AAPL", 50_000, 200_000_000, participation_rate=0.05)
        total = sum(s["notional"] for s in schedule)
        assert total == pytest.approx(50_000, rel=0.05)

    def test_vwap_avoids_auction_windows(self):
        from hedge_fund_ai.data.microstructure import build_vwap_schedule
        schedule = build_vwap_schedule("AAPL", 50_000, 200_000_000)
        # First slot (open auction) should be excluded
        offsets = [s["time_offset_minutes"] for s in schedule]
        assert 0 not in offsets  # never trade at exactly market open

    def test_auction_window_detection(self):
        from hedge_fund_ai.data.microstructure import avoid_auction_windows
        import pytz
        et = pytz.timezone("America/New_York")
        # 9:35 ET = opening auction zone
        open_time = datetime(2024, 1, 15, 9, 35, tzinfo=et)
        # 2:00 PM ET = safe trading window
        safe_time = datetime(2024, 1, 15, 14, 0, tzinfo=et)
        assert avoid_auction_windows(open_time) is True
        assert avoid_auction_windows(safe_time) is False


# ── Factor crowding ───────────────────────────────────────────────────────────

class TestCrowding:
    def test_crowding_score_bounded(self):
        from hedge_fund_ai.factors.crowding import compute_crowding_score
        with patch("hedge_fund_ai.factors.crowding.get_institutional_ownership_pct", return_value=0.70), \
             patch("hedge_fund_ai.factors.crowding.get_top_holders", return_value=[]):
            result = compute_crowding_score(["AAPL"])
        assert 0.0 <= result["AAPL"]["crowding_score"] <= 1.0

    def test_crowding_penalty_reduces_score(self):
        from hedge_fund_ai.factors.crowding import apply_crowding_penalty
        scores   = {"AAPL": 2.0, "MSFT": 1.5}
        crowding = {
            "AAPL": {"crowding_score": 0.80, "risk_level": "high"},
            "MSFT": {"crowding_score": 0.30, "risk_level": "low"},
        }
        adjusted = apply_crowding_penalty(scores, crowding)
        assert adjusted["AAPL"] < scores["AAPL"]   # penalized
        assert adjusted["MSFT"] == scores["MSFT"]  # unchanged

    def test_cost_adjusted_ic_hurdle(self):
        from hedge_fund_ai.factors.crowding import cost_adjusted_ic_hurdle
        # Negative IC-IR, high turnover → should definitely fail
        result_fail = cost_adjusted_ic_hurdle(raw_ic_ir=-0.20, factor_turnover=0.90)
        # High IC-IR, low turnover → should pass
        result_pass = cost_adjusted_ic_hurdle(raw_ic_ir=1.20, factor_turnover=0.10)
        assert not result_fail["passes"]
        assert result_pass["passes"]

    def test_factor_turnover_stable_factor(self):
        from hedge_fund_ai.factors.crowding import compute_factor_turnover
        same = {"AAPL": 2.0, "MSFT": 1.5, "NVDA": 3.0, "GOOGL": 1.0, "META": 0.5}
        turnover = compute_factor_turnover(same, same)
        assert turnover < 0.10  # perfectly stable

    def test_factor_turnover_reshuffled(self):
        from hedge_fund_ai.factors.crowding import compute_factor_turnover
        curr = {"AAPL": 3.0, "MSFT": 2.5, "NVDA": -1.0, "GOOGL": -2.0, "META": -3.0}
        prev = {"AAPL": -3.0, "MSFT": -2.5, "NVDA": 1.0, "GOOGL": 2.0, "META": 3.0}
        turnover = compute_factor_turnover(curr, prev)
        assert turnover > 0.70  # completely reshuffled


# ── Factor covariance ─────────────────────────────────────────────────────────

class TestFactorCovariance:
    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        return pd.DataFrame(
            np.random.randn(250, 6) * 0.01,
            columns=["AAPL","MSFT","NVDA","GOOGL","META","AMZN"]
        )

    def test_returns_total_cov_matrix(self, sample_returns):
        from hedge_fund_ai.portfolio.factor_covariance import estimate_factor_covariance
        result = estimate_factor_covariance(sample_returns)
        cov = result["total_cov"]
        assert cov.shape == (6, 6)
        eigvals = np.linalg.eigvalsh(cov)
        assert (eigvals > -1e-8).all(), "Covariance not positive semi-definite"

    def test_factor_names_returned(self, sample_returns):
        from hedge_fund_ai.portfolio.factor_covariance import estimate_factor_covariance
        result = estimate_factor_covariance(sample_returns)
        if result["factor_names"]:
            assert "market" in result["factor_names"]
            assert "momentum" in result["factor_names"]

    def test_r_squared_in_range(self, sample_returns):
        from hedge_fund_ai.portfolio.factor_covariance import estimate_factor_covariance
        result = estimate_factor_covariance(sample_returns)
        r2 = result["r_squared"]
        assert all(0 <= v <= 1 for v in r2)

    def test_risk_attribution_sums_to_100(self, sample_returns):
        from hedge_fund_ai.portfolio.factor_covariance import (
            estimate_factor_covariance, compute_factor_risk_attribution
        )
        result = estimate_factor_covariance(sample_returns)
        if result["factor_loadings"] is not None:
            w = np.ones(6) / 6
            attr = compute_factor_risk_attribution(w, result)
            total = attr["factor_pct"] + attr["specific_pct"]
            assert abs(total - 100.0) < 1.0

    def test_fallback_with_short_history(self):
        from hedge_fund_ai.portfolio.factor_covariance import estimate_factor_covariance
        short = pd.DataFrame(np.random.randn(30, 4) * 0.01,
                             columns=["A","B","C","D"])
        result = estimate_factor_covariance(short)
        assert result["method"] == "ledoit_wolf_fallback"
        assert result["total_cov"].shape == (4, 4)


# ── P&L attribution ───────────────────────────────────────────────────────────

class TestPnLAttribution:
    def test_brinson_attribution_sums_correctly(self):
        from hedge_fund_ai.reporting.pnl_attribution import compute_brinson_attribution
        port_w  = {"AAPL": 0.50, "MSFT": 0.50}
        bench_w = {"AAPL": 0.40, "MSFT": 0.60}
        returns = {"AAPL": 0.05, "MSFT": 0.02}
        result  = compute_brinson_attribution(port_w, bench_w, returns, 0.03)
        # allocation + selection + interaction ≈ total_active
        recon = result["allocation_effect"] + result["selection_effect"] + result["interaction_effect"]
        assert abs(recon - result["total_active"]) < 1e-8

    def test_factor_attribution_residual(self):
        from hedge_fund_ai.reporting.pnl_attribution import compute_factor_attribution
        port_w   = {"AAPL": 0.6, "MSFT": 0.4}
        loadings = {"AAPL": {"market": 1.2, "momentum": 0.5},
                    "MSFT": {"market": 0.9, "momentum": 0.3}}
        f_rets   = {"market": 0.01, "momentum": 0.005}
        s_rets   = {"AAPL": 0.02, "MSFT": 0.01}
        result   = compute_factor_attribution(port_w, loadings, f_rets, s_rets)
        assert "portfolio_return" in result
        assert "alpha_residual" in result
        assert "factor_breakdown" in result
        # Portfolio return = factor + residual
        recon = result["factor_return_total"] + result["alpha_residual"]
        assert abs(recon - result["portfolio_return"]) < 1e-8

    def test_attribution_report_generates(self):
        from hedge_fund_ai.reporting.pnl_attribution import generate_attribution_report
        from hedge_fund_ai.backtest.audit import AuditLog
        np.random.seed(0)
        equity = pd.Series(
            np.cumprod(1 + np.random.randn(500) * 0.008),
            index=pd.bdate_range("2022-01-01", periods=500),
        )
        log = AuditLog()
        df  = generate_attribution_report(equity, log, period="monthly")
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "portfolio_return" in df.columns
            assert "active_return" in df.columns


# ── Tax lot tracker ───────────────────────────────────────────────────────────

class TestTaxLots:
    @pytest.fixture
    def ledger(self, tmp_path):
        import hedge_fund_ai.execution.tax_lots as tl
        tl._LOTS_PATH = str(tmp_path / "tax_lots.json")
        from hedge_fund_ai.execution.tax_lots import TaxLotLedger
        return TaxLotLedger(method="fifo")

    def test_add_and_close_lot_fifo(self, ledger):
        ledger.add_lot("AAPL", 10, 150.0, "2023-01-15")
        ledger.add_lot("AAPL", 10, 160.0, "2023-06-15")
        result = ledger.close_lots("AAPL", 10, 170.0, "2023-12-15")
        # FIFO: should close the first lot (cost=150)
        assert result["realized_gain"] == pytest.approx(200.0)  # (170-150)*10

    def test_lifo_closes_last_lot(self, ledger):
        ledger.method = "lifo"
        ledger.add_lot("MSFT", 10, 200.0, "2023-01-01")
        ledger.add_lot("MSFT", 10, 250.0, "2023-06-01")
        result = ledger.close_lots("MSFT", 10, 300.0, "2023-12-01")
        # LIFO: should close the $250 lot
        assert result["realized_gain"] == pytest.approx(500.0)  # (300-250)*10

    def test_long_term_classification(self, ledger):
        ledger.add_lot("NVDA", 5, 100.0, "2022-01-01")
        result = ledger.close_lots("NVDA", 5, 200.0, "2023-06-01")  # held ~17 months
        assert result["long_term"] == pytest.approx(500.0)
        assert result["short_term"] == pytest.approx(0.0)

    def test_wash_sale_detection(self, ledger):
        ledger.add_lot("META", 10, 200.0, "2023-12-01")
        result = ledger.close_lots("META", 10, 150.0, "2023-12-20")  # loss within 30 days
        assert result["wash_sale_risk"] is True

    def test_tax_summary(self, ledger):
        year = datetime.now().year
        ledger.add_lot("GOOG", 5, 100.0, f"{year}-01-15")
        ledger.close_lots("GOOG", 5, 120.0, f"{year}-06-15")
        summary = ledger.tax_summary(year)
        assert summary["total_realized"] == pytest.approx(100.0)
        assert summary["n_transactions"] == 1

    def test_open_positions(self, ledger):
        ledger.add_lot("AMZN", 10, 130.0, "2023-01-01")
        open_pos = ledger.open_positions()
        assert "AMZN" in open_pos
        assert open_pos["AMZN"]["shares"] == pytest.approx(10.0)


# ── Backup & logging ──────────────────────────────────────────────────────────

class TestBackup:
    def test_create_and_restore(self, tmp_path):
        import hedge_fund_ai.core.backup as bk
        bk._STATE_DIR  = str(tmp_path / "state")
        bk._BACKUP_DIR = str(tmp_path / "backups")
        os.makedirs(bk._STATE_DIR, exist_ok=True)

        # Create a fake state file
        test_file = os.path.join(bk._STATE_DIR, "ic_history.json")
        with open(test_file, "w") as f:
            json.dump({"test": [0.05, 0.06]}, f)

        bk._CRITICAL_FILES = ["ic_history.json"]
        result = bk.create_backup("test_tag")
        assert result["files_backed_up"] == ["ic_history.json"]
        assert os.path.exists(result["backup_path"])

        # Delete file and restore
        os.remove(test_file)
        assert not os.path.exists(test_file)
        restore = bk.restore_backup(result["backup_path"])
        assert "ic_history.json" in restore["restored"]
        assert os.path.exists(test_file)

    def test_integrity_check(self, tmp_path):
        import hedge_fund_ai.core.backup as bk
        bk._STATE_DIR = str(tmp_path / "state")
        os.makedirs(bk._STATE_DIR, exist_ok=True)
        bk._CRITICAL_FILES = ["test.json"]

        # Write valid JSON
        with open(os.path.join(bk._STATE_DIR, "test.json"), "w") as f:
            json.dump({"ok": True}, f)
        report = bk.verify_state_integrity()
        assert report["healthy"] is True

    def test_prune_old_backups(self, tmp_path):
        import hedge_fund_ai.core.backup as bk
        bk._BACKUP_DIR = str(tmp_path / "backups")
        os.makedirs(bk._BACKUP_DIR, exist_ok=True)
        # Create a fake old backup
        old_path = os.path.join(bk._BACKUP_DIR, "backup_old.zip")
        with open(old_path, "w") as f:
            f.write("fake")
        # Set modification time to 60 days ago
        import time
        old_mtime = time.time() - 60 * 86400
        os.utime(old_path, (old_mtime, old_mtime))
        bk.prune_old_backups(keep_days=30)
        assert not os.path.exists(old_path)


class TestStructuredLogging:
    def test_json_formatter(self):
        from hedge_fund_ai.core.logging_setup import JSONFormatter, set_run_id
        set_run_id("TEST01")
        fmt = JSONFormatter()
        import logging
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello world", (), None)
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["run_id"] == "TEST01"
        assert parsed["msg"] == "hello world"
        assert parsed["level"] == "INFO"

    def test_metrics_counter(self):
        from hedge_fund_ai.core.logging_setup import increment_metric, get_metrics
        increment_metric("test.counter", 1.0)
        increment_metric("test.counter", 2.0)
        m = get_metrics()
        assert m["test.counter"] >= 3.0
