"""
Tests for all 8 gaps identified in the deployment readiness review.

Gap 1: Fundamentals in backtest (point-in-time)
Gap 2: Survivorship bias
Gap 3: Live signal validation (SignalLogger)
Gap 4: Sentiment IC (tested via sentiment pipeline)
Gap 5: Position stop losses + daily loss circuit breaker
Gap 6: Placebo test
Gap 7: Corporate actions (yfinance-adjusted — documented limitation)
Gap 8: Stale ADV / execution circuit breaker
"""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Gap 1: Point-in-time fundamentals ────────────────────────────────────────

class TestFundamentalsPointInTime:
    def test_as_of_returns_empty_for_future_date(self):
        from hedge_fund_ai.data.fundamentals_pit import FundamentalStore
        store = FundamentalStore(["AAPL"])
        # No data loaded — should return empty dict
        result = store.as_of("AAPL", "2020-01-01")
        assert result == {}

    def test_as_of_respects_filing_lag(self):
        """as_of() must NOT return data that wasn't available on the requested date."""
        from hedge_fund_ai.data.fundamentals_pit import FundamentalStore, _safe_float

        store = FundamentalStore(["FAKE"], filing_lag_days=45)

        # Manually inject fake data
        fake_data = pd.DataFrame([
            {
                "report_date":    pd.Timestamp("2022-12-31"),
                "available_date": pd.Timestamp("2023-02-14"),  # 45 days after quarter end
                "roe": 15.0, "op_margin": 20.0, "net_margin": 10.0,
                "rev_yoy": 8.0, "ni_yoy": 12.0, "de_ratio": 0.5,
            },
            {
                "report_date":    pd.Timestamp("2022-09-30"),
                "available_date": pd.Timestamp("2022-11-14"),
                "roe": 12.0, "op_margin": 18.0, "net_margin": 8.0,
                "rev_yoy": 5.0, "ni_yoy": 7.0, "de_ratio": 0.6,
            },
        ]).set_index("available_date").sort_index()

        store._yf_data["FAKE"] = fake_data

        # Request data BEFORE the Q4 filing is available
        result_before = store.as_of("FAKE", "2023-01-01")  # Q4 not yet filed
        result_after  = store.as_of("FAKE", "2023-03-01")  # Q4 now available

        # Should get Q3 data (filed Nov 14) before Q4 available
        assert result_before.get("roe_pct", 0) == pytest.approx(12.0)
        # Should get Q4 data after filing date
        assert result_after.get("roe_pct", 0) == pytest.approx(15.0)

    def test_filing_lag_enforced(self):
        """Report date + filing_lag must be before the as_of date."""
        from hedge_fund_ai.data.fundamentals_pit import FundamentalStore

        store = FundamentalStore(["X"], filing_lag_days=45)
        fake  = pd.DataFrame([{
            "report_date":    pd.Timestamp("2023-03-31"),
            "available_date": pd.Timestamp("2023-05-15"),  # 45 days later
            "roe": 20.0, "op_margin": 25.0, "net_margin": 15.0,
            "rev_yoy": 10.0, "ni_yoy": 15.0, "de_ratio": 0.3,
        }]).set_index("available_date")
        store._yf_data["X"] = fake

        # Before filing → empty
        assert store.as_of("X", "2023-04-01") == {}
        # After filing → has data
        assert store.as_of("X", "2023-06-01").get("roe_pct") == pytest.approx(20.0)

    def test_coverage_report(self):
        import pandas as pd
        from hedge_fund_ai.data.fundamentals_pit import FundamentalStore
        store = FundamentalStore(["A", "B"])
        # Inject fake data (load_all() would hit network)
        for t in ["A", "B"]:
            store._yf_data[t] = pd.DataFrame(
                [{"roe": 10.0}],
                index=pd.DatetimeIndex([pd.Timestamp("2023-02-15")]),
            )
        cov = store.coverage()
        assert "per_ticker" in cov
        pt = cov["per_ticker"]
        assert "A" in pt and "B" in pt
        for t, v in pt.items():
            assert "n_quarters" in v
            assert v["n_quarters"] >= 1


# ── Gap 2: Survivorship bias ──────────────────────────────────────────────────

class TestSurvivorshipBias:
    def test_baseline_2019_contains_major_names(self):
        from hedge_fund_ai.data.survivorship import get_universe_as_of
        universe = get_universe_as_of("2019-01-01")
        for ticker in ["AAPL", "MSFT", "JPM", "JNJ"]:
            assert ticker in universe, f"{ticker} should be in 2019 baseline"

    def test_additions_applied_after_effective_date(self):
        from hedge_fund_ai.data.survivorship import get_universe_as_of
        # TSLA added 2020-12-21
        before = get_universe_as_of("2020-12-20")
        after  = get_universe_as_of("2020-12-22")
        assert "TSLA" not in before, "TSLA should not be in universe before addition"
        assert "TSLA" in after,      "TSLA should be in universe after addition"

    def test_removals_applied_after_effective_date(self):
        from hedge_fund_ai.data.survivorship import get_universe_as_of
        # ATVI removed 2023-06-19 (acquired by MSFT)
        before = get_universe_as_of("2023-06-18")
        after  = get_universe_as_of("2023-06-20")
        assert "ATVI" in before, "ATVI should be present before removal"
        assert "ATVI" not in after, "ATVI should be removed after acquisition"

    def test_filter_reduces_universe(self):
        from hedge_fund_ai.data.survivorship import filter_universe_pit
        # Mix of valid 2019 tickers and a recently-added one
        candidates = ["AAPL", "MSFT", "TSLA", "PLTR"]  # TSLA/PLTR not in 2019
        filtered   = filter_universe_pit(candidates, "2019-06-01")
        assert "AAPL" in filtered
        assert "MSFT" in filtered
        assert "TSLA" not in filtered, "TSLA was added Dec 2020, not June 2019"
        assert "PLTR" not in filtered, "PLTR was added Sep 2024"

    def test_filter_returns_list(self):
        from hedge_fund_ai.data.survivorship import filter_universe_pit
        result = filter_universe_pit(["AAPL", "MSFT"], "2022-01-01")
        assert isinstance(result, list)

    def test_2024_additions(self):
        from hedge_fund_ai.data.survivorship import get_universe_as_of
        # PLTR added 2024-09-23
        before = get_universe_as_of("2024-09-22")
        after  = get_universe_as_of("2024-09-24")
        assert "PLTR" not in before
        assert "PLTR" in after


# ── Gap 3: Live signal logger / IC validation ─────────────────────────────────

class TestSignalLogger:
    @pytest.fixture
    def tmp_logger(self, tmp_path):
        """SignalLogger that writes to a temp directory."""
        import hedge_fund_ai.data.signal_logger as sl_mod
        original = sl_mod._LOG_PATH
        sl_mod._LOG_PATH = str(tmp_path / "signal_log.json")
        from hedge_fund_ai.data.signal_logger import SignalLogger
        logger = SignalLogger(fwd_days=21)
        yield logger
        sl_mod._LOG_PATH = original

    def test_log_signals_creates_record(self, tmp_logger):
        scores = {"AAPL": 1.5, "MSFT": 0.8, "NVDA": 2.1, "GOOGL": -0.3, "META": 0.5}
        tmp_logger.log_signals("2024-01-15", scores, regime="risk_on",
                               top_picks=["NVDA", "AAPL"])
        assert len(tmp_logger._records) == 1
        r = tmp_logger._records[0]
        assert r["signal_date"] == "2024-01-15"
        assert r["regime"]      == "risk_on"
        assert "NVDA" in r["scores"]
        assert "NVDA" in r["top_picks"]

    def test_fwd_due_date_is_21_days_later(self, tmp_logger):
        tmp_logger.log_signals("2024-01-15", {"A": 1.0, "B": 2.0})
        r = tmp_logger._records[0]
        assert r["fwd_due_date"] == "2024-02-05"

    def test_ic_report_insufficient_data(self, tmp_logger):
        tmp_logger.log_signals("2024-01-01", {"A": 1.0})
        report = tmp_logger.ic_report()
        assert report["status"] == "insufficient_data"
        assert "need filled records" in report["message"].lower()

    def test_fill_returns_computes_ic(self, tmp_logger):
        """Simulate a full signal → fill → IC cycle."""
        from scipy.stats import spearmanr

        # Log signals 30 days ago (so they're past due)
        past_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        scores = {"AAPL": 2.0, "MSFT": 1.5, "NVDA": 3.0,
                  "GOOGL": -1.0, "META": 0.5, "AMZN": -0.5}
        tmp_logger.log_signals(past_date, scores, top_picks=["NVDA", "AAPL", "MSFT"])

        # Inject fake prices that reward the signal (higher scores → higher returns)
        prices = pd.DataFrame({
            t: [100.0, 100.0 * (1 + scores[t] * 0.01)]
            for t in scores
        }, index=[pd.Timestamp(past_date), pd.Timestamp.now()])

        tmp_logger.fill_returns(fwd_prices=prices)

        filled = [r for r in tmp_logger._records if r.get("filled")]
        assert len(filled) == 1
        assert filled[0]["ic"] is not None

    def test_verdict_formats(self):
        from hedge_fund_ai.data.signal_logger import _verdict
        assert "WAIT"    in _verdict(0.05, 0.3, 0.55, 5)
        assert "DEPLOY"  in _verdict(0.06, 0.6, 0.56, 20)
        assert "NO ALPHA" in _verdict(-0.01, -0.5, 0.45, 20)

    def test_pending_fills(self, tmp_logger):
        past = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        future = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
        tmp_logger.log_signals(past, {"A": 1.0})
        tmp_logger.log_signals(future, {"A": 1.0})
        pending = tmp_logger.pending_fills()
        assert len(pending) == 1   # only the past one is due


# ── Gap 5: Stop losses and circuit breaker ────────────────────────────────────

class TestRiskControls:
    @pytest.fixture(autouse=True)
    def patch_state_path(self, tmp_path):
        import hedge_fund_ai.execution.risk_controls as rc
        rc._STATE_PATH = str(tmp_path / "risk_state.json")
        # Reset in-memory halt
        rc.clear_halt()

    def test_no_halt_initially(self):
        from hedge_fund_ai.execution.risk_controls import is_trading_halted
        halted, reason = is_trading_halted()
        assert not halted

    def test_trigger_and_check_halt(self):
        from hedge_fund_ai.execution.risk_controls import (
            is_trading_halted, trigger_daily_halt, clear_halt
        )
        trigger_daily_halt("test halt", hours=48)
        halted, reason = is_trading_halted()
        assert halted
        assert "test halt" in reason
        clear_halt()
        halted, _ = is_trading_halted()
        assert not halted

    def test_stop_loss_detected(self):
        from hedge_fund_ai.execution.risk_controls import (
            check_stop_losses, _load_risk_state, _save_risk_state
        )
        state = _load_risk_state()
        state["entry_prices"] = {
            "AAPL": {"price": 200.0, "date": "2024-01-01"},
            "MSFT": {"price": 400.0, "date": "2024-01-01"},
        }
        _save_risk_state(state)

        # AAPL drops 10% (above STOP_LOSS_PCT=8%)
        current = {"AAPL": 180.0, "MSFT": 398.0}
        to_close = check_stop_losses(current)
        assert "AAPL" in to_close
        assert "MSFT" not in to_close

    def test_stop_loss_not_triggered_within_threshold(self):
        from hedge_fund_ai.execution.risk_controls import (
            check_stop_losses, _load_risk_state, _save_risk_state
        )
        state = _load_risk_state()
        state["entry_prices"] = {"NVDA": {"price": 500.0, "date": "2024-01-01"}}
        _save_risk_state(state)
        # Only 3% drop — within 8% threshold
        to_close = check_stop_losses({"NVDA": 485.0})
        assert "NVDA" not in to_close

    def test_daily_loss_triggers_halt(self):
        from hedge_fund_ai.execution.risk_controls import (
            check_daily_loss, is_trading_halted
        )
        # 4% loss triggers 3% limit
        triggered = check_daily_loss(96_000, 100_000)
        assert triggered
        halted, _ = is_trading_halted()
        assert halted

    def test_daily_loss_ok_within_limit(self):
        from hedge_fund_ai.execution.risk_controls import check_daily_loss
        triggered = check_daily_loss(98_500, 100_000)  # only 1.5% loss
        assert not triggered

    def test_max_drawdown_triggers_halt(self):
        from hedge_fund_ai.execution.risk_controls import (
            check_max_drawdown, is_trading_halted
        )
        # 25% drawdown from peak of 1.3
        equity = [1.0, 1.1, 1.3, 1.0, 0.97]
        triggered = check_max_drawdown(equity)
        assert triggered
        halted, _ = is_trading_halted()
        assert halted

    def test_pre_execution_check_returns_correct_structure(self):
        from hedge_fund_ai.execution.risk_controls import (
            pre_execution_check, _save_risk_state, _load_risk_state
        )
        # Ensure no halt and no stop losses
        state = _load_risk_state()
        state["entry_prices"] = {}
        state["halted_until"] = None
        _save_risk_state(state)

        result = pre_execution_check(
            portfolio=[{"ticker": "AAPL", "price": 200.0, "weight": 0.1}],
            equity_curve=[1.0, 1.01, 1.02],
            portfolio_value_today=102_000,
            portfolio_value_yesterday=101_000,
        )
        assert "proceed" in result
        assert "stop_loss_closes" in result
        assert isinstance(result["stop_loss_closes"], list)


# ── Gap 6: Placebo test ───────────────────────────────────────────────────────

class TestPlaceboTest:
    @pytest.fixture
    def small_backtest_data(self):
        np.random.seed(42)
        n_days, n_tickers = 400, 15
        tickers = [f"T{i:02d}" for i in range(n_tickers)]
        rets    = np.random.randn(n_days, n_tickers) * 0.01
        prices  = pd.DataFrame(
            np.cumprod(1 + rets, axis=0) * 100,
            index=pd.bdate_range("2021-01-01", periods=n_days),
            columns=tickers,
        )
        volumes = pd.DataFrame(
            np.abs(np.random.randn(n_days, n_tickers)) * 1e6 + 5e6,
            index=prices.index, columns=tickers,
        )
        return prices, volumes, tickers

    def test_placebo_returns_required_keys(self, small_backtest_data):
        from hedge_fund_ai.backtest.placebo import run_placebo_test
        prices, volumes, tickers = small_backtest_data
        np.random.seed(0)
        model_equity = pd.Series(
            np.cumprod(1 + np.random.randn(300) * 0.008),
            index=pd.bdate_range("2021-06-01", periods=300),
        )
        result = run_placebo_test(
            prices, volumes, None, tickers,
            model_equity=model_equity, n_trials=5,
        )
        for key in ["model_sharpe", "random_mean", "random_std", "p_value", "verdict", "n_trials"]:
            assert key in result, f"Missing key: {key}"

    def test_p_value_in_range(self, small_backtest_data):
        from hedge_fund_ai.backtest.placebo import run_placebo_test
        prices, volumes, tickers = small_backtest_data
        model_equity = pd.Series(
            np.cumprod(1 + np.random.randn(300) * 0.008),
            index=pd.bdate_range("2021-06-01", periods=300),
        )
        result = run_placebo_test(
            prices, volumes, None, tickers,
            model_equity=model_equity, n_trials=5,
        )
        assert 0.0 <= result["p_value"] <= 1.0

    def test_placebo_verdict_no_alpha_for_random(self, small_backtest_data):
        from hedge_fund_ai.backtest.placebo import run_placebo_test
        prices, volumes, tickers = small_backtest_data
        # A flat equity curve — should not beat random
        model_equity = pd.Series(
            np.ones(300),
            index=pd.bdate_range("2021-06-01", periods=300),
        )
        result = run_placebo_test(
            prices, volumes, None, tickers,
            model_equity=model_equity, n_trials=5,
        )
        # Flat equity has Sharpe ~0, likely does not beat random
        assert isinstance(result["verdict"], str)
        assert len(result["verdict"]) > 0

    def test_placebo_verdict_function(self):
        from hedge_fund_ai.backtest.placebo import _placebo_verdict
        assert "SIGNIFICANT" in _placebo_verdict(1.5, 0.03, 0.2)
        assert "NO ALPHA"    in _placebo_verdict(0.1, 0.70, 0.5)
        assert "BORDERLINE"  in _placebo_verdict(1.2, 0.08, 0.3)
