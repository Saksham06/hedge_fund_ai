"""Tests for portfolio optimizer and constraints."""
import numpy as np
import pandas as pd
import pytest

from hedge_fund_ai.portfolio.optimizer import (
    _ledoit_wolf_cov, _herfindahl_smooth, _cap_positions,
    _apply_position_limits, _sector_constraints,
)
from hedge_fund_ai.portfolio.risk_model import drawdown_control


@pytest.fixture
def sample_returns():
    np.random.seed(1)
    return pd.DataFrame(
        np.random.randn(200, 5) * 0.01,
        columns=["A", "B", "C", "D", "E"],
    )


class TestOptimizer:
    def test_ledoit_wolf_positive_definite(self, sample_returns):
        cov = _ledoit_wolf_cov(sample_returns)
        assert cov.shape == (5, 5)
        eigvals = np.linalg.eigvalsh(cov)
        assert (eigvals > 0).all()

    def test_herfindahl_smooth_reduces_concentration(self):
        w = np.array([0.90, 0.04, 0.03, 0.03])
        hhi_before = float(np.sum(w**2))
        smoothed   = _herfindahl_smooth(w.copy(), max_ratio=1.5)
        hhi_after  = float(np.sum(smoothed**2))
        assert hhi_after < hhi_before

    def test_cap_positions_enforced(self):
        # 6 stocks so the capped weight can redistribute without exceeding cap
        w = np.array([0.50, 0.20, 0.12, 0.08, 0.06, 0.04])
        w /= w.sum()
        capped = _cap_positions(w, cap=0.20)
        assert capped.max() <= 0.20 + 1e-4
        assert abs(capped.sum() - 1.0) < 1e-6

    def test_apply_position_limits_dict(self):
        weights = {"A": 0.50, "B": 0.20, "C": 0.12, "D": 0.08, "E": 0.06, "F": 0.04}
        result  = _apply_position_limits(weights)
        for t, w in result.items():
            assert w <= 0.20 + 1e-4, f"{t}: {w} > MAX"
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_sector_constraints_enforced(self):
        weights = {"A": 0.40, "B": 0.35, "C": 0.15, "D": 0.10}
        sectors = {"A": "Tech", "B": "Tech", "C": "Health", "D": "Finance"}
        result  = _sector_constraints(weights, sectors)
        sec_tot = {}
        for t, w in result.items():
            s = sectors[t]
            sec_tot[s] = sec_tot.get(s, 0) + w
        for s, total in sec_tot.items():
            assert total <= 0.35 + 0.005, f"{s}: {total} > MAX_SECTOR"


class TestRiskModel:
    def test_drawdown_control_no_dd(self):
        equity = pd.Series([1.0, 1.01, 1.02, 1.03])
        assert drawdown_control(equity) == 1.0

    def test_drawdown_control_moderate_dd(self):
        equity = pd.Series([1.0, 1.1, 1.0, 0.95])
        scale  = drawdown_control(equity)
        assert scale < 1.0

    def test_drawdown_control_severe_dd(self):
        equity = pd.Series([1.0, 1.2, 1.0, 0.7])
        scale  = drawdown_control(equity)
        assert scale <= 0.65
