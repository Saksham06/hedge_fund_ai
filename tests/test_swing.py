"""Tests for all swing trading components."""
import json
import os
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch


# ── Price cache ───────────────────────────────────────────────────────────────

class TestPriceCache:
    @pytest.fixture(autouse=True)
    def tmp_cache(self, tmp_path):
        import hedge_fund_ai.config as cfg
        cfg.PRICE_CACHE_DIR = str(tmp_path / "cache")
        cfg.PRICE_CACHE_TTL_HOURS = 20
        import hedge_fund_ai.data.price_cache as pc
        pc._cache = None
        yield
        pc._cache = None

    def test_miss_on_empty_cache(self):
        from hedge_fund_ai.data.price_cache import get_cache
        c, v = get_cache().get("AAPL")
        assert c is None and v is None

    def test_put_then_get(self):
        from hedge_fund_ai.data.price_cache import get_cache
        close  = pd.Series([100.0, 101.0, 102.0])
        volume = pd.Series([1e6, 1.1e6, 0.9e6])
        get_cache().put("MSFT", close, volume)
        c, v = get_cache().get("MSFT")
        assert c is not None
        assert len(c) == 3
        assert abs(float(c.iloc[-1]) - 102.0) < 0.01

    def test_stale_returns_none(self, tmp_path):
        import hedge_fund_ai.config as cfg
        cfg.PRICE_CACHE_TTL_HOURS = 0  # immediately stale
        from hedge_fund_ai.data.price_cache import get_cache, PriceCache
        import hedge_fund_ai.data.price_cache as pc
        pc._cache = None
        cache = get_cache()
        cache.put("NVDA", pd.Series([500.0]), None)
        import time; time.sleep(0.01)
        c, v = cache.get("NVDA")
        assert c is None

    def test_bulk_get(self):
        from hedge_fund_ai.data.price_cache import get_cache
        cache = get_cache()
        for t in ["AAPL","MSFT","NVDA"]:
            cache.put(t, pd.Series([100.0, 101.0]), None)
        result = cache.get_bulk(["AAPL","MSFT","NVDA","GOOGL"])
        assert "AAPL" in result and "MSFT" in result and "NVDA" in result
        assert "GOOGL" not in result  # not cached

    def test_stats(self):
        from hedge_fund_ai.data.price_cache import get_cache
        cache = get_cache()
        cache.put("X", pd.Series([1.0, 2.0]), None)
        s = cache.stats()
        assert s["fresh"] >= 1
        assert s["size_mb"] >= 0


# ── Swing signals ─────────────────────────────────────────────────────────────

class TestSwingSignals:
    def _make_data(self, momentum=1.0, flow=0.5):
        return {
            "tech": {"ann_vol_30d": 18.0},
            "features": {
                "momentum":          momentum,
                "price_accel":       0.5,
                "rel_strength":      0.3,
                "sector_momentum":   0.2,
                "flow_signal":       flow,
                "mean_reversion":    0.1,
                "earnings_surprise": 0.4,
                "sentiment":         0.2,
                "high_52w":          0.6,
                "liquidity":         0.3,
                "quality":           0.1,
                "value":             0.0,
                "volatility":        -0.2,
                "skewness":          0.0,
                "accruals":          0.0,
            },
        }

    def test_build_swing_signal_finite(self):
        from hedge_fund_ai.factors.swing_signals import build_swing_signal
        for regime in ["risk_on","neutral","risk_off","crisis"]:
            s = build_swing_signal(self._make_data(), regime)
            assert np.isfinite(s), f"Non-finite for regime {regime}"

    def test_risk_on_higher_than_crisis(self):
        from hedge_fund_ai.factors.swing_signals import build_swing_signal
        d = self._make_data(momentum=1.5, flow=1.0)
        assert build_swing_signal(d, "risk_on") > build_swing_signal(d, "crisis")

    def test_positive_momentum_positive_score(self):
        from hedge_fund_ai.factors.swing_signals import build_swing_signal
        s = build_swing_signal(self._make_data(momentum=2.0), "risk_on")
        assert s > 0

    def test_score_components_returned(self):
        from hedge_fund_ai.factors.swing_signals import swing_score_components
        comps = swing_score_components(self._make_data(), "risk_on")
        assert isinstance(comps, dict)
        assert "momentum" in comps
        assert "flow_signal" in comps
        # Swing should NOT have momentum_12_1 (too slow)
        assert "momentum_12_1" not in comps

    def test_crisis_near_zero(self):
        from hedge_fund_ai.factors.swing_signals import build_swing_signal
        # In crisis, momentum-heavy data should give near-zero or negative score
        d = self._make_data(momentum=1.0, flow=0.5)
        s = build_swing_signal(d, "crisis")
        # crisis weights are very small so |score| should be small
        assert abs(s) < 3.0

    def test_weights_sum_to_one(self):
        from hedge_fund_ai.factors.swing_signals import _SWING_BASE
        for regime, weights in _SWING_BASE.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, f"{regime} weights sum to {total}"


# ── Swing tracker ─────────────────────────────────────────────────────────────

class TestSwingTracker:
    @pytest.fixture(autouse=True)
    def tmp_paths(self, tmp_path):
        import hedge_fund_ai.execution.swing_tracker as st
        st._TRACKER_PATH = str(tmp_path / "positions.json")
        st._TRADES_PATH  = str(tmp_path / "trades.json")
        yield

    def test_open_position(self):
        from hedge_fund_ai.execution.swing_tracker import SwingTracker
        t = SwingTracker()
        t.open_position("AAPL", 150.0, 0.10, "risk_on", 1.5)
        assert "AAPL" in t._open
        assert t._open["AAPL"]["entry_price"] == 150.0

    def test_profit_target_exit(self):
        from hedge_fund_ai.execution.swing_tracker import SwingTracker
        import hedge_fund_ai.execution.swing_tracker as st_mod
        st_mod.SWING_MIN_HOLD_DAYS  = 0
        st_mod.SWING_MAX_HOLD_DAYS  = 60
        st_mod.SWING_PROFIT_TARGET  = 0.12
        st_mod.SWING_STOP_LOSS      = 0.05
        t = SwingTracker()
        t.open_position("MSFT", 300.0, 0.10)
        # Entry 10 days ago
        from datetime import date
        t._open["MSFT"]["entry_date"] = (
            datetime.now(timezone.utc).date() - timedelta(days=10)
        ).strftime("%Y-%m-%d")
        exits = t.check_exits({"MSFT": 340.0})  # +13.3% > 12% target
        assert "MSFT" in exits
        assert exits["MSFT"] == "profit_target"

    def test_stop_loss_exit(self):
        from hedge_fund_ai.execution.swing_tracker import SwingTracker
        import hedge_fund_ai.execution.swing_tracker as st_mod
        st_mod.SWING_MIN_HOLD_DAYS = 0
        t = SwingTracker()
        t.open_position("NVDA", 500.0, 0.10)
        t._open["NVDA"]["entry_date"] = (
            datetime.now(timezone.utc).date() - timedelta(days=5)
        ).strftime("%Y-%m-%d")
        exits = t.check_exits({"NVDA": 472.0})  # -5.6% < -5% stop
        assert "NVDA" in exits
        assert exits["NVDA"] == "stop_loss"

    def test_max_hold_exit(self):
        from hedge_fund_ai.execution.swing_tracker import SwingTracker
        import hedge_fund_ai.execution.swing_tracker as st_mod
        st_mod.SWING_MIN_HOLD_DAYS = 0
        st_mod.SWING_MAX_HOLD_DAYS = 60
        t = SwingTracker()
        t.open_position("GOOGL", 170.0, 0.10)
        t._open["GOOGL"]["entry_date"] = (
            datetime.now(timezone.utc).date() - timedelta(days=61)
        ).strftime("%Y-%m-%d")
        exits = t.check_exits({"GOOGL": 172.0})  # within profit/stop but too old
        assert "GOOGL" in exits
        assert exits["GOOGL"] == "max_hold"

    def test_min_hold_prevents_exit(self):
        from hedge_fund_ai.execution.swing_tracker import SwingTracker
        import hedge_fund_ai.execution.swing_tracker as st_mod
        st_mod.SWING_MIN_HOLD_DAYS = 3
        st_mod.SWING_STOP_LOSS     = 0.05
        t = SwingTracker()
        t.open_position("META", 500.0, 0.10)
        t._open["META"]["entry_date"] = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
        exits = t.check_exits({"META": 470.0})  # would be stop loss but <3 days
        assert "META" not in exits

    def test_close_records_trade(self):
        from hedge_fund_ai.execution.swing_tracker import SwingTracker
        import hedge_fund_ai.execution.swing_tracker as st_mod
        st_mod.SWING_MIN_HOLD_DAYS = 0
        t = SwingTracker()
        t.open_position("AMZN", 180.0, 0.10)
        t._open["AMZN"]["entry_date"] = (
            datetime.now(timezone.utc).date() - timedelta(days=10)
        ).strftime("%Y-%m-%d")
        t.close_position("AMZN", 198.0, "profit_target")
        assert "AMZN" not in t._open
        assert len(t._closed) == 1
        assert t._closed[0]["win"] is True
        assert t._closed[0]["net_pnl"] > 0

    def test_metrics_empty(self):
        from hedge_fund_ai.execution.swing_tracker import SwingTracker
        t = SwingTracker()
        m = t.compute_metrics()
        assert m["n_trades"] == 0

    def test_metrics_with_trades(self):
        from hedge_fund_ai.execution.swing_tracker import SwingTracker
        import hedge_fund_ai.execution.swing_tracker as st_mod
        st_mod.SWING_MIN_HOLD_DAYS = 0
        t = SwingTracker()
        # Simulate 10 closed trades: 6 wins 4 losses
        for i in range(10):
            t.open_position(f"T{i}", 100.0, 0.05)
            t._open[f"T{i}"]["entry_date"] = (
                datetime.now(timezone.utc).date() - timedelta(days=15)
            ).strftime("%Y-%m-%d")
            exit_px = 112.0 if i < 6 else 95.0  # win or loss
            t.close_position(f"T{i}", exit_px, "profit_target" if i < 6 else "stop_loss")
        m = t.compute_metrics()
        assert m["n_trades"] == 10
        assert abs(m["win_rate"] - 0.6) < 0.01
        assert m["profit_factor"] > 1.0
        assert "swing_verdict" in m


# ── Swing metrics ─────────────────────────────────────────────────────────────

class TestSwingMetrics:
    def _make_audit(self):
        from hedge_fund_ai.backtest.audit import AuditLog, AuditRecord
        log = AuditLog()
        np.random.seed(42)
        regimes = ["risk_on","neutral","risk_off","neutral","risk_on"] * 8
        for i in range(40):
            r = AuditRecord(
                rebalance_idx=i*5, date=f"2024-{(i//20)+1:02d}-{(i%20)+1:02d}",
                regime=regimes[i % len(regimes)],
                universe_size=30, dd_scale=1.0,
                final_weights={"AAPL":0.1,"MSFT":0.1},
                raw_turnover=0.3, filtered_turnover=0.15,
            )
            log.append(r)
            # 55% win rate
            ret = abs(np.random.randn() * 0.02) if i % 2 == 0 else -abs(np.random.randn() * 0.015)
            log.fill_returns(i, float(ret))
        return log

    def test_swing_metrics_keys(self):
        from hedge_fund_ai.backtest.swing_metrics import compute_swing_metrics
        equity = pd.Series(np.cumprod(1 + np.random.randn(200)*0.005))
        log    = self._make_audit()
        m      = compute_swing_metrics(equity, log)
        for key in ["win_rate","profit_factor","max_consecutive_losses",
                    "signal_to_noise","swing_verdict","swing_sharpe"]:
            assert key in m, f"Missing: {key}"

    def test_win_rate_in_range(self):
        from hedge_fund_ai.backtest.swing_metrics import compute_swing_metrics
        equity = pd.Series(np.cumprod(1 + np.random.randn(200)*0.005))
        m = compute_swing_metrics(equity, self._make_audit())
        assert 0.0 <= m["win_rate"] <= 1.0

    def test_profit_factor_positive(self):
        from hedge_fund_ai.backtest.swing_metrics import compute_swing_metrics
        equity = pd.Series(np.cumprod(1 + np.random.randn(200)*0.005))
        m = compute_swing_metrics(equity, self._make_audit())
        assert m["profit_factor"] > 0

    def test_win_rate_by_regime(self):
        from hedge_fund_ai.backtest.swing_metrics import compute_swing_metrics
        equity = pd.Series(np.cumprod(1 + np.random.randn(200)*0.005))
        m = compute_swing_metrics(equity, self._make_audit())
        assert "win_rate_by_regime" in m
        for regime, stats in m["win_rate_by_regime"].items():
            assert 0.0 <= stats["win_rate"] <= 1.0

    def test_verdict_string(self):
        from hedge_fund_ai.backtest.swing_metrics import compute_swing_metrics
        equity = pd.Series(np.cumprod(1 + np.random.randn(200)*0.005))
        m = compute_swing_metrics(equity, self._make_audit())
        assert isinstance(m["swing_verdict"], str)
        assert len(m["swing_verdict"]) > 5


# ── Config swing params ───────────────────────────────────────────────────────

class TestSwingConfig:
    def test_swing_params_defined(self):
        from hedge_fund_ai.config import (
            SWING_REBALANCE_DAYS, SWING_MIN_HOLD_DAYS, SWING_MAX_HOLD_DAYS,
            SWING_TOP_N, SWING_TARGET_VOL, SWING_STOP_LOSS, SWING_PROFIT_TARGET,
        )
        assert SWING_REBALANCE_DAYS == 5
        assert SWING_MIN_HOLD_DAYS  == 3
        assert SWING_MAX_HOLD_DAYS  == 60
        assert SWING_TOP_N          == 12
        assert SWING_STOP_LOSS      == 0.05
        assert SWING_PROFIT_TARGET  == 0.12
        assert 0.10 < SWING_TARGET_VOL < 0.30

    def test_horizon_params_swing(self):
        from hedge_fund_ai.run import _horizon_params
        p = _horizon_params("swing")
        assert p["rebalance_days"] == 5
        assert p["stop_loss"]      == 0.05
        assert p["profit_target"]  == 0.12

    def test_horizon_params_longterm(self):
        from hedge_fund_ai.run import _horizon_params
        p = _horizon_params("longterm")
        assert p["rebalance_days"] >= 60
        assert p["stop_loss"]      > 0.05   # wider stop for long-term


# ── Scheduler ────────────────────────────────────────────────────────────────

class TestScheduler:
    def test_next_run_is_weekday(self):
        from hedge_fund_ai.scheduler import _next_run_dt
        dt = _next_run_dt()
        assert dt.weekday() < 5, f"Scheduled on weekend: {dt.strftime('%A')}"

    def test_next_run_in_future(self):
        from hedge_fund_ai.scheduler import _next_run_dt
        from datetime import timezone
        dt = _next_run_dt()
        assert dt > datetime.now(timezone.utc)

    def test_build_cmd_contains_horizon(self):
        import hedge_fund_ai.scheduler as sch
        sch.SCHEDULE_HORIZON = "swing"
        cmd = sch._build_cmd()
        assert "--horizon" in cmd
        assert "swing" in cmd

    def test_build_cmd_no_backtest(self):
        import hedge_fund_ai.scheduler as sch
        sch.SCHEDULE_NO_BACKTEST = True
        cmd = sch._build_cmd()
        assert "--no-backtest" in cmd
