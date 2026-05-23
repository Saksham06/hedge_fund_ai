"""
Swing Trading Signal Engine.

Factor weights tuned specifically for 1-week to 3-month holding periods.

Key differences from the base engine:
  1. Short-term momentum (3-5 day price acceleration) dominates
  2. 12-month momentum REMOVED — too slow for swing, reversal risk
  3. Mean-reversion signal added — capture snap-backs
  4. Flow signal (OBV) heavily weighted — institutional accumulation visible in volume
  5. Quality/value MINIMIZED — fundamentals don't move in weeks
  6. Earnings surprise gets a large weight — post-earnings drift is a real swing edge
  7. Volatility penalty stronger — swing traders want liquid, less volatile names

Regime overlays:
  risk_on:  full momentum + flow
  neutral:  balanced, reduce momentum slightly
  risk_off: shift to mean-reversion + defensive
  crisis:   go flat (all weights near zero, cash is a position)
"""

import json
import logging
import os
from collections import deque

import numpy as np

from hedge_fund_ai.config import IC_WINDOW, IC_DECAY_HALF_LIFE

logger = logging.getLogger(__name__)

_IC_FILE = os.path.join(os.path.dirname(__file__), "..", "state", "swing_ic_history.json")

_ic_history: dict[str, deque] = {}
_ic_dirty = False


def _load_ic():
    global _ic_history
    if not os.path.exists(_IC_FILE):
        return
    try:
        with open(_IC_FILE) as f:
            raw = json.load(f)
        for k, v in raw.items():
            _ic_history[k] = deque(list(v)[-IC_WINDOW:], maxlen=IC_WINDOW)
    except Exception:
        pass


def _save_ic():
    try:
        os.makedirs(os.path.dirname(_IC_FILE), exist_ok=True)
        with open(_IC_FILE, "w") as f:
            json.dump({k: list(v) for k, v in _ic_history.items()}, f)
    except Exception:
        pass


_load_ic()


def update_swing_ic(factor: str, ic_val: float):
    global _ic_dirty
    if factor not in _ic_history:
        _ic_history[factor] = deque(maxlen=IC_WINDOW)
    _ic_history[factor].append(float(ic_val))
    _ic_dirty = True


def flush_swing_ic():
    global _ic_dirty
    if _ic_dirty:
        _save_ic()
        _ic_dirty = False


def _ic_scale(factor: str, base: float) -> float:
    """Decay-weighted IC scaling. Negative IC-IR → zero weight."""
    hist = _ic_history.get(factor)
    if not hist or len(hist) < 4:
        return base
    arr = np.array(hist, dtype=float)
    n   = len(arr)
    half = float(IC_DECAY_HALF_LIFE)
    weights = np.array([2.0 ** (-(n-1-i)/half) for i in range(n)])
    weights /= weights.sum()
    w_mean = float(np.dot(weights, arr))
    w_std  = float(np.sqrt(np.dot(weights, (arr-w_mean)**2))) + 1e-6
    ir     = w_mean / w_std
    if ir < -0.3:
        return 0.0
    return base * float(np.clip(1.0 + ir, 0.1, 2.5))


# ── Swing factor weights per regime ───────────────────────────────────────────
_SWING_BASE = {
    "risk_on": {
        # SHORT-TERM MOMENTUM (dominant in swing)
        "momentum":          0.18,  # 3-month price trend
        "price_accel":       0.12,  # acceleration of trend (2nd derivative)
        "rel_strength":      0.10,  # vs universe median
        "sector_momentum":   0.08,  # sector tailwind
        # FLOW / VOLUME (institutional accumulation)
        "flow_signal":       0.14,  # OBV-based accumulation
        # MEAN REVERSION (snap-back after oversold)
        "mean_reversion":    0.08,  # price vs 50d SMA
        # CATALYSTS (event-driven swing edge)
        "earnings_surprise": 0.12,  # post-earnings drift
        "sentiment":         0.06,  # news momentum
        # STRUCTURAL
        "high_52w":          0.06,  # proximity to 52w high (momentum filter)
        "liquidity":         0.04,  # prefer liquid names
        # MINIMAL QUALITY (not a swing factor)
        "quality":           0.02,
    },
    "neutral": {
        "momentum":          0.14,
        "price_accel":       0.10,
        "rel_strength":      0.08,
        "sector_momentum":   0.06,
        "flow_signal":       0.12,
        "mean_reversion":    0.10,
        "earnings_surprise": 0.10,
        "sentiment":         0.08,
        "high_52w":          0.06,
        "liquidity":         0.08,
        "quality":           0.04,
        "value":             0.04,
    },
    "risk_off": {
        # In risk-off, favour mean-reversion and defensive
        "momentum":          0.05,
        "price_accel":       0.04,
        "rel_strength":      0.04,
        "sector_momentum":   0.03,
        "flow_signal":       0.10,
        "mean_reversion":    0.20,  # oversold snap-backs dominate
        "earnings_surprise": 0.08,
        "sentiment":         0.06,
        "high_52w":          0.02,
        "liquidity":         0.16,  # prefer high-liquidity names
        "quality":           0.12,
        "value":             0.10,
    },
    "crisis": {
        # Crisis: go to near-zero — cash is the position
        "momentum":          0.02,
        "price_accel":       0.02,
        "rel_strength":      0.02,
        "sector_momentum":   0.01,
        "flow_signal":       0.05,
        "mean_reversion":    0.10,
        "earnings_surprise": 0.03,
        "sentiment":         0.02,
        "high_52w":          0.01,
        "liquidity":         0.30,  # liquidity is everything in crisis
        "quality":           0.20,
        "value":             0.22,
    },
}


def _clip(x: float, lo: float = -3.0, hi: float = 3.0) -> float:
    try:
        return float(max(lo, min(hi, float(x or 0))))
    except Exception:
        return 0.0


def is_trending_market(macro: dict) -> dict:
    """
    Trend filter to prevent swing entries in choppy/ranging markets.
    Two conditions (either passes):
      1. SPY 20-day SMA > 50-day SMA (uptrend)
      2. VIX < 22 (low fear = more directional moves)

    In ranging market: reduce TOP_N from 12 → 6 (higher conviction only).
    Returns {trending, reason, recommended_top_n}
    """
    vix      = float(macro.get("vix", 20) or 20)
    spy_3m   = float(macro.get("spy_3m", 0) or 0)     # 3m return proxy for trend
    vol_reg  = float(macro.get("vol_regime", 1) or 1)  # short/long vol ratio

    low_vix      = vix < TREND_VIX_THRESHOLD          # VIX < 22
    uptrend      = spy_3m > 0                          # SPY positive 3m return
    stable_vol   = vol_reg < 1.2                       # vol not expanding

    trending = low_vix or uptrend
    ranging  = not trending or vol_reg > 1.5

    if ranging:
        reason = f"RANGING (VIX={vix:.1f} spy3m={spy_3m:.1f}% vol_reg={vol_reg:.2f})"
        top_n  = 6   # concentrate on highest-conviction only
    elif trending:
        reason = f"TRENDING (VIX={vix:.1f} spy3m={spy_3m:.1f}%)"
        top_n  = 12
    else:
        reason = "NEUTRAL"
        top_n  = 9

    return {"trending": trending, "ranging": ranging,
            "reason": reason, "recommended_top_n": top_n}


def build_swing_signal(d: dict, regime: str = "neutral") -> float:
    """
    Composite swing alpha score.
    Input features must be cross-sectionally z-scored before calling.
    """
    features = d.get("features", {}) or {}
    tech     = d.get("tech", {}) or {}
    regime_k = regime.lower() if regime.lower() in _SWING_BASE else "neutral"

    score = sum(
        _ic_scale(f, bw) * _clip(features.get(f, 0.0))
        for f, bw in _SWING_BASE[regime_k].items()
    )

    # Always-on swing penalties
    vol_pct = float(tech.get("ann_vol_30d") or 20.0)
    skew_z  = _clip(features.get("skewness", 0.0))
    acc_z   = _clip(features.get("accruals", 0.0))

    score -= 0.10 * _clip(features.get("volatility", 0.0))  # penalise high vol harder for swing
    score -= 0.04 * skew_z
    score -= 0.04 * acc_z
    score -= 0.006 * vol_pct

    # In crisis: additional momentum penalty (avoid catching falling knives)
    if regime_k == "crisis":
        score -= 0.20 * _clip(features.get("momentum", 0.0))

    return float(score)


def swing_score_components(d: dict, regime: str = "neutral") -> dict:
    features = d.get("features", {}) or {}
    regime_k = regime.lower() if regime.lower() in _SWING_BASE else "neutral"
    return {
        f: round(_ic_scale(f, bw) * _clip(features.get(f, 0.0)), 5)
        for f, bw in _SWING_BASE[regime_k].items()
    }
