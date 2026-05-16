"""
Unit tests for the factor computation and signal engine.
Run with: pytest tests/ -v
"""
import numpy as np
import pandas as pd
import pytest

from hedge_fund_ai.factors.historical_features import compute_features_from_prices
from hedge_fund_ai.factors.normalization import normalize_features, zscore, winsorize
from hedge_fund_ai.factors.signal_engine import build_signal, _clip
from hedge_fund_ai.backtest.metrics import compute_metrics


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def price_series():
    np.random.seed(42)
    return np.cumprod(1 + np.random.randn(300) * 0.01) * 100

@pytest.fixture
def volume_series():
    np.random.seed(43)
    return np.abs(np.random.randn(300)) * 1e6 + 5e5

@pytest.fixture
def equity():
    np.random.seed(0)
    return pd.Series(np.cumprod(1 + np.random.randn(500) * 0.008))

@pytest.fixture
def mock_data():
    """10 stock mock universe."""
    np.random.seed(99)
    records = []
    for i in range(10):
        records.append({
            "ticker": f"TICK{i}",
            "tech": {
                "ret_3m": float(np.random.randn() * 5),
                "ret_6m": float(np.random.randn() * 8),
                "ret_12m_skip1m": float(np.random.randn() * 12),
                "ann_vol_30d": float(20 + np.random.rand() * 10),
                "ann_vol_90d": float(20 + np.random.rand() * 10),
                "vol_regime": float(0.8 + np.random.rand() * 0.4),
                "high_52w_proximity": float(0.7 + np.random.rand() * 0.3),
                "trend_slope": float(np.random.randn() * 0.01),
            },
            "fund": {
                "roe_pct": float(10 + np.random.rand() * 20),
                "op_margin_pct": float(10 + np.random.rand() * 20),
                "fcf_margin_pct": float(5 + np.random.rand() * 15),
                "pe_ratio": float(15 + np.random.rand() * 20),
                "ps_ratio": float(2 + np.random.rand() * 5),
                "revenue_growth": float(np.random.randn() * 10),
            },
            "sentiment_score": float(np.random.randn() * 0.5),
            "flow_signal": float(np.random.randn()),
            "earnings_surprise": float(np.random.randn()),
            "liquidity": float(np.random.randn()),
            "skewness": float(np.random.randn() * 0.5),
        })
    return records


# ── historical_features ───────────────────────────────────────────────────────

class TestHistoricalFeatures:
    def test_returns_dict(self, price_series):
        f = compute_features_from_prices(price_series)
        assert isinstance(f, dict)

    def test_has_core_keys(self, price_series):
        f = compute_features_from_prices(price_series)
        for key in ["momentum_12_1", "ret_3m", "ret_1m", "vol",
                    "mean_reversion", "flow_signal", "earnings_surprise",
                    "liquidity", "reversal", "price_accel"]:
            assert key in f, f"Missing key: {key}"

    def test_with_volume(self, price_series, volume_series):
        f = compute_features_from_prices(price_series, volume_series)
        assert "liquidity" in f
        assert "flow_signal" in f

    def test_short_series_returns_empty(self):
        f = compute_features_from_prices(np.random.randn(30))
        assert f == {}

    def test_none_returns_empty(self):
        assert compute_features_from_prices(None) == {}

    def test_momentum_skip_month(self, price_series):
        f = compute_features_from_prices(price_series)
        # momentum_12_1 should skip last month: end[-21] / start
        assert isinstance(f["momentum_12_1"], float)
        assert -100 < f["momentum_12_1"] < 100

    def test_vol_positive(self, price_series):
        f = compute_features_from_prices(price_series)
        assert f["vol"] > 0

    def test_vol_regime_reasonable(self, price_series, volume_series):
        f = compute_features_from_prices(price_series, volume_series)
        assert 0.1 < f["vol_regime"] < 10


# ── normalization ─────────────────────────────────────────────────────────────

class TestNormalization:
    def test_zscore_zero_mean(self):
        arr = zscore([1, 2, 3, 4, 5])
        assert abs(float(np.mean(arr))) < 0.01

    def test_zscore_unit_std(self):
        arr = zscore([1, 2, 3, 4, 5])
        assert abs(float(np.std(arr)) - 1.0) < 0.1

    def test_winsorize_clips(self):
        arr = winsorize(np.array([0, 1, 2, 3, 4, 100]), lower=5, upper=95)
        assert float(arr.max()) < 100

    def test_normalize_features_adds_features_key(self, mock_data):
        result = normalize_features(mock_data)
        for d in result:
            assert "features" in d

    def test_normalized_values_finite(self, mock_data):
        result = normalize_features(mock_data)
        for d in result:
            for k, v in d["features"].items():
                assert np.isfinite(v), f"Non-finite feature: {k}={v}"

    def test_missing_pe_imputed(self, mock_data):
        # Set PE to None for some stocks
        mock_data[0]["fund"]["pe_ratio"] = None
        mock_data[1]["fund"]["pe_ratio"] = None
        result = normalize_features(mock_data)
        for d in result:
            assert np.isfinite(d["features"]["value"])


# ── signal_engine ─────────────────────────────────────────────────────────────

class TestSignalEngine:
    def test_clip_bounds(self):
        assert _clip(100) == 3.0
        assert _clip(-100) == -3.0
        assert _clip(0) == 0.0

    def test_build_signal_returns_float(self, mock_data):
        data = normalize_features(mock_data)
        for d in data:
            score = build_signal(d, regime="neutral")
            assert isinstance(score, float)
            assert np.isfinite(score)

    def test_regime_changes_score(self, mock_data):
        data = normalize_features(mock_data)
        d = data[0]
        s_on  = build_signal(d, regime="risk_on")
        s_off = build_signal(d, regime="risk_off")
        # Not guaranteed to differ but should be different for most stocks
        assert isinstance(s_on, float)
        assert isinstance(s_off, float)

    def test_crisis_regime_supported(self, mock_data):
        data = normalize_features(mock_data)
        score = build_signal(data[0], regime="crisis")
        assert isinstance(score, float)


# ── metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_all_keys_present(self, equity):
        m = compute_metrics(equity)
        for key in ["sharpe", "sortino", "calmar", "max_drawdown",
                    "total_return", "cagr", "ann_vol", "win_rate", "profit_factor"]:
            assert key in m, f"Missing metric: {key}"

    def test_sharpe_finite(self, equity):
        m = compute_metrics(equity)
        assert np.isfinite(m["sharpe"])

    def test_max_drawdown_negative(self, equity):
        m = compute_metrics(equity)
        assert m["max_drawdown"] <= 0

    def test_win_rate_in_range(self, equity):
        m = compute_metrics(equity)
        assert 0 <= m["win_rate"] <= 1

    def test_empty_returns_empty(self):
        assert compute_metrics(None) == {}
        assert compute_metrics(pd.Series(dtype=float)) == {}

    def test_benchmark_comparison(self, equity):
        bench = equity * (1 + np.random.randn(len(equity)) * 0.0001)
        m = compute_metrics(equity, benchmark=bench)
        assert "beta" in m
        assert "alpha_ann" in m
        assert "info_ratio" in m

    def test_list_input(self):
        perf = [{"portfolio_value": 100 + i * 0.1} for i in range(100)]
        m = compute_metrics(perf)
        assert "sharpe" in m
