"""
Trend Filter — prevents new swing entries in choppy/ranging markets.

Logic:
  TRENDING:  SPY 20d SMA > 50d SMA  OR  VIX < 22
  CHOPPY:    SPY 20d SMA < 50d SMA  AND VIX > 22

In choppy markets:
  - Reduce TOP_N from 12 → 6 (only highest conviction)
  - Tighten profit target from 12% → 8% (take profits faster)
  - Skip new entries for scores below 0.5 standard deviations

This prevents the "stopped out 3x in a week" scenario that erodes capital
through repeated 15bps round-trip costs with no wins.
"""
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VIX_THRESHOLD    = float(os.getenv("TREND_VIX_THRESHOLD",   "22.0"))
CHOPPY_TOP_N     = int(os.getenv("TREND_CHOPPY_TOP_N",       "6"))
CHOPPY_MIN_SCORE = float(os.getenv("TREND_CHOPPY_MIN_SCORE", "0.5"))


def assess_market_trend(macro: dict) -> dict:
    """
    Assess whether the market is trending or choppy using macro data.

    Returns:
        {
          "is_trending":  bool,
          "regime_label": "trending" | "choppy" | "unknown",
          "top_n_adj":    int    — adjusted TOP_N for this regime,
          "min_score":    float  — minimum signal score to enter,
          "reasons":      list of strings
        }
    """
    vix     = float(macro.get("vix",       20.0) or 20.0)
    spy_3m  = float(macro.get("spy_3m",     0.0) or 0.0)
    vol_reg = float(macro.get("vol_regime", 1.0) or 1.0)
    regime  = str(macro.get("regime",  "neutral"))

    reasons = []
    trend_signals = 0
    chop_signals  = 0

    # Signal 1: VIX level
    if vix < VIX_THRESHOLD:
        trend_signals += 1
        reasons.append(f"VIX={vix:.1f} < {VIX_THRESHOLD} (low fear = trending)")
    else:
        chop_signals += 1
        reasons.append(f"VIX={vix:.1f} >= {VIX_THRESHOLD} (elevated fear = choppy risk)")

    # Signal 1b: VIX spike leading indicator (pre-emptive regime detection)
    # If VIX has spiked >15% in 5 days, choppiness is incoming — act early
    vix_5d_change = float(macro.get("vix_5d_change", 0.0) or 0.0)
    if vix_5d_change > 15.0:
        chop_signals += 1
        reasons.append(f"VIX spike +{vix_5d_change:.1f}% in 5d → pre-emptive choppy warning")

    # Signal 2: SPY 3-month trend
    if spy_3m > 2.0:
        trend_signals += 1
        reasons.append(f"SPY 3m={spy_3m:+.1f}% (uptrend)")
    elif spy_3m < -3.0:
        chop_signals += 1
        reasons.append(f"SPY 3m={spy_3m:+.1f}% (downtrend → choppy for longs)")

    # Signal 3: Vol regime (expanding vol = choppy)
    if vol_reg > 1.3:
        chop_signals += 1
        reasons.append(f"vol_regime={vol_reg:.2f} > 1.3 (expanding vol = choppy)")
    elif vol_reg < 0.8:
        trend_signals += 1
        reasons.append(f"vol_regime={vol_reg:.2f} < 0.8 (compressing vol = trending)")

    # Signal 4: Crisis/risk_off regime
    if regime in ("crisis", "risk_off"):
        chop_signals += 2
        reasons.append(f"regime={regime} (high-risk = reduce entries)")
    elif regime == "risk_on":
        trend_signals += 1
        reasons.append(f"regime=risk_on (favourable for swing entries)")

    is_trending   = trend_signals > chop_signals
    regime_label  = "trending" if is_trending else "choppy"

    # Adjust parameters for choppy markets
    if is_trending:
        top_n_adj  = int(os.getenv("SWING_TOP_N", "12"))
        min_score  = 0.0
    else:
        top_n_adj  = CHOPPY_TOP_N      # 6 — only highest conviction
        min_score  = CHOPPY_MIN_SCORE  # 0.5 std dev minimum
        logger.warning(
            "TREND FILTER: Choppy market detected (VIX=%.1f, SPY3m=%.1f%%) → "
            "reducing entries from 12 to %d, min_score=%.1f",
            vix, spy_3m, top_n_adj, min_score
        )

    result = {
        "is_trending":  is_trending,
        "regime_label": regime_label,
        "top_n_adj":    top_n_adj,
        "min_score":    min_score,
        "trend_score":  trend_signals,
        "chop_score":   chop_signals,
        "reasons":      reasons,
        "vix":          vix,
        "spy_3m":       spy_3m,
    }

    logger.info("Trend filter: %s (trend=%d chop=%d) | top_n=%d",
                regime_label.upper(), trend_signals, chop_signals, top_n_adj)
    return result


def apply_trend_filter(top_stocks: list, scores: dict[str, float],
                       trend_assessment: dict) -> list:
    """
    Filter top_stocks based on trend assessment.
    In choppy markets: keep only top top_n_adj stocks above min_score threshold.
    """
    if trend_assessment.get("is_trending", True):
        return top_stocks  # no filtering in trending market

    top_n     = trend_assessment["top_n_adj"]
    min_score = trend_assessment["min_score"]

    # Filter by score threshold
    qualified = [t for t in top_stocks
                 if scores.get(t if isinstance(t, str) else t.get("ticker", ""), 0) >= min_score]

    # Limit to top_n
    filtered = qualified[:top_n]

    removed = len(top_stocks) - len(filtered)
    if removed > 0:
        logger.info("Trend filter removed %d low-conviction entries (%d → %d positions)",
                    removed, len(top_stocks), len(filtered))

    return filtered
