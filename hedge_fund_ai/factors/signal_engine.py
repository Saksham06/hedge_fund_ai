"""
Elite Signal Engine.

Improvements over previous version:
  1. Exponential IC decay: recent IC observations weighted more than old ones
  2. Factor IC-IR gating: factors with IC-IR < -0.3 are zeroed out entirely
  3. FMP alpha signal as a separate factor (earnings surprise + revisions + ownership)
  4. Piotroski F-score as a standalone quality factor
  5. Sector-relative momentum (stock vs sector median, not just universe median)
  6. Earnings quality factor (accruals)
  7. ROIC > cost-of-capital filter (capital allocation quality)
"""
import json
import logging
import os
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

_IC_FILE   = os.path.join(os.path.dirname(__file__), "..", "state", "ic_history.json")

from hedge_fund_ai.config import IC_WINDOW, IC_DECAY_HALF_LIFE

_ic_history: dict[str, deque] = {}
_ic_dirty   = False


def _load_ic():
    global _ic_history
    if not os.path.exists(_IC_FILE):
        return
    try:
        with open(_IC_FILE) as f:
            raw = json.load(f)
        for k, v in raw.items():
            _ic_history[k] = deque(list(v)[-IC_WINDOW:], maxlen=IC_WINDOW)
    except Exception as e:
        logger.warning("IC load failed: %s", e)


def _save_ic():
    try:
        os.makedirs(os.path.dirname(_IC_FILE), exist_ok=True)
        with open(_IC_FILE, "w") as f:
            json.dump({k: list(v) for k, v in _ic_history.items()}, f)
    except Exception as e:
        logger.warning("IC save failed: %s", e)


_load_ic()


def update_ic(factor: str, ic_val: float):
    global _ic_dirty
    if factor not in _ic_history:
        _ic_history[factor] = deque(maxlen=IC_WINDOW)
    _ic_history[factor].append(float(ic_val))
    _ic_dirty = True


def flush_ic():
    global _ic_dirty
    if _ic_dirty:
        _save_ic()
        _ic_dirty = False


def _ic_scale(factor: str, base: float) -> float:
    """
    Exponentially decay-weighted IC scaling.
    Recent observations count more. Factors with IC-IR < -0.3 are zeroed.
    """
    hist = _ic_history.get(factor)
    if not hist or len(hist) < 4:
        return base

    arr = np.array(hist, dtype=float)
    n   = len(arr)

    # Exponential decay weights: most recent = weight 1.0
    half = float(IC_DECAY_HALF_LIFE)
    weights = np.array([2.0 ** (-(n - 1 - i) / half) for i in range(n)])
    weights /= weights.sum()

    w_mean = float(np.dot(weights, arr))
    w_std  = float(np.sqrt(np.dot(weights, (arr - w_mean) ** 2))) + 1e-6
    ic_ir  = w_mean / w_std

    # Gate: negative IC-IR factors disabled
    if ic_ir < -0.3:
        return 0.0

    return base * float(np.clip(1.0 + ic_ir, 0.1, 2.5))


def get_factor_ic_summary() -> dict:
    """Return current IC stats for all tracked factors."""
    summary = {}
    for factor, hist in _ic_history.items():
        if not hist:
            continue
        arr = np.array(hist, dtype=float)
        summary[factor] = {
            "mean_ic":  round(float(np.mean(arr)), 4),
            "ic_ir":    round(float(np.mean(arr) / (np.std(arr) + 1e-6)), 3),
            "hit_rate": round(float(np.mean(arr > 0)), 3),
            "n":        len(arr),
            "current_scale": round(_ic_scale(factor, 1.0), 3),
        }
    return summary


# ── Regime-dependent base weights ─────────────────────────────────────────────

_BASE: dict[str, dict[str, float]] = {
    "risk_on": {
        # Momentum cluster
        "momentum_12_1":     0.16,
        "momentum":          0.06,
        "price_accel":       0.04,
        "rel_strength":      0.08,
        "sector_momentum":   0.05,
        # Quality / growth cluster
        "quality":           0.04,
        "piotroski":         0.04,
        "growth":            0.08,
        "roic":              0.03,
        # Value
        "value":             0.02,
        # Alternative alpha
        "fmp_alpha":         0.10,
        "earnings_surprise": 0.07,
        "sentiment":         0.07,
        "flow_signal":       0.07,
        # Risk structure
        "liquidity":         0.04,
        "high_52w":          0.05,
    },
    "neutral": {
        "momentum_12_1":     0.10,
        "momentum":          0.05,
        "price_accel":       0.03,
        "rel_strength":      0.06,
        "sector_momentum":   0.04,
        "quality":           0.09,
        "piotroski":         0.06,
        "growth":            0.07,
        "roic":              0.05,
        "value":             0.07,
        "fmp_alpha":         0.08,
        "earnings_surprise": 0.06,
        "sentiment":         0.06,
        "flow_signal":       0.06,
        "liquidity":         0.07,
        "high_52w":          0.05,
    },
    "risk_off": {
        "momentum_12_1":     0.03,
        "momentum":          0.02,
        "price_accel":       0.02,
        "rel_strength":      0.04,
        "sector_momentum":   0.03,
        "quality":           0.18,
        "piotroski":         0.10,
        "growth":            0.05,
        "roic":              0.08,
        "value":             0.12,
        "fmp_alpha":         0.06,
        "earnings_surprise": 0.04,
        "sentiment":         0.04,
        "flow_signal":       0.04,
        "liquidity":         0.12,
        "high_52w":          0.03,
    },
    "crisis": {
        "momentum_12_1":     0.00,
        "momentum":          0.00,
        "price_accel":       0.01,
        "rel_strength":      0.03,
        "sector_momentum":   0.02,
        "quality":           0.22,
        "piotroski":         0.12,
        "growth":            0.04,
        "roic":              0.08,
        "value":             0.14,
        "fmp_alpha":         0.04,
        "earnings_surprise": 0.03,
        "sentiment":         0.02,
        "flow_signal":       0.03,
        "liquidity":         0.18,
        "high_52w":          0.02,
    },
}


def _clip(x: float, lo: float = -3.0, hi: float = 3.0) -> float:
    try:
        return float(max(lo, min(hi, float(x or 0))))
    except Exception:
        return 0.0


def score_components(d: dict, regime: str = "neutral") -> dict[str, float]:
    """Per-factor contribution breakdown for attribution."""
    features = d.get("features", {}) or {}
    regime_k = regime.lower() if regime.lower() in _BASE else "neutral"
    return {
        factor: round(_ic_scale(factor, bw) * _clip(features.get(factor, 0.0)), 5)
        for factor, bw in _BASE[regime_k].items()
    }


def build_signal(d: dict, regime: str = "neutral") -> float:
    """
    Composite alpha score.
    All feature inputs must be cross-sectionally z-scored before calling.
    """
    features = d.get("features", {}) or {}
    tech     = d.get("tech", {}) or {}
    regime_k = regime.lower() if regime.lower() in _BASE else "neutral"

    score = sum(
        _ic_scale(f, bw) * _clip(features.get(f, 0.0))
        for f, bw in _BASE[regime_k].items()
    )

    # Always-on penalties (not regime-dependent)
    vol_z  = _clip(features.get("volatility", 0.0))
    skew_z = _clip(features.get("skewness", 0.0))
    mr_z   = _clip(features.get("mean_reversion", 0.0))
    acc_z  = _clip(features.get("accruals", 0.0))        # earnings quality
    vol_pct = float(tech.get("ann_vol_30d") or 20.0)

    score -= 0.08 * vol_z     # BAB: penalize high beta
    score -= 0.03 * skew_z    # tail risk
    score -= 0.03 * mr_z      # overextension
    score -= 0.04 * acc_z     # high accruals = earnings manipulation risk
    score -= 0.005 * vol_pct  # raw vol anchor

    if regime_k == "crisis":
        score -= 0.15 * _clip(features.get("momentum_12_1", 0.0))
        score += 0.10 * _clip(features.get("piotroski", 0.0))

    return float(score)


def apply_turnover_penalty(score: float, ticker: str, prev_portfolio: list,
                           penalty: float = 0.025) -> float:
    if not prev_portfolio:
        return score
    prev = {p.get("ticker"): float(p.get("weight", 0))
            for p in prev_portfolio if p.get("ticker")}
    if prev.get(ticker, 0.0) == 0.0:
        score -= penalty * 0.10
    return score
