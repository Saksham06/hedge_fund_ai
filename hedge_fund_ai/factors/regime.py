"""
Four-state market regime detection using macro indicators.

States: risk_on | neutral | risk_off | crisis

Inputs (all computed from price data, no premium data feed needed):
  - VIX level and trend
  - SPY 3-month momentum
  - Credit spread proxy (HYG vs IEF)
  - Yield curve proxy (TLT vs SHY relative return)
  - Realized volatility regime (SPY 30d vs 90d vol)

Uses a rule-based ensemble by default (fast, no refit each call).
Falls back to KMeans clustering on historical macro snapshots
for regime smoothing when enough history is available.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _rule_score(vix: float, spy_3m: float, credit: float, yc: float, vol_regime: float) -> str:
    """
    Weighted rule-based regime classifier.
    Returns one of: risk_on | neutral | risk_off | crisis
    """
    score = 0.0

    # VIX
    if   vix < 15:  score += 2.0
    elif vix < 20:  score += 1.0
    elif vix < 25:  score -= 0.5
    elif vix < 30:  score -= 1.5
    else:           score -= 3.0

    # SPY momentum
    if   spy_3m > 5:  score += 2.0
    elif spy_3m > 0:  score += 0.5
    elif spy_3m > -5: score -= 1.0
    else:             score -= 2.5

    # Credit (HYG outperforms IEF → risk appetite)
    if   credit > 0.01:  score += 1.5
    elif credit > 0:     score += 0.5
    elif credit > -0.01: score -= 0.5
    else:                score -= 1.5

    # Yield curve (TLT underperforms → growth expectations)
    if yc > 0.02:  score += 1.0   # steepening = growth
    elif yc < -0.02: score -= 1.0  # flattening = slow

    # Vol regime
    if vol_regime < 0.8:  score += 1.0   # vol compressing
    elif vol_regime > 1.3: score -= 1.5  # vol expanding

    if score >= 3.5:    return "risk_on"
    elif score >= 0.5:  return "neutral"
    elif score >= -2.0: return "risk_off"
    else:               return "crisis"


def detect_regime(macro_history: list[dict]) -> str:
    """Classify current regime from list of macro snapshots."""
    if not macro_history:
        return "neutral"

    latest = macro_history[-1]
    vix         = float(latest.get("vix",         20) or 20)
    spy_3m      = float(latest.get("spy_3m",       0) or 0)
    credit      = float(latest.get("credit_spread_signal", 0) or 0)
    yc          = float(latest.get("yield_curve",  0) or 0)
    vol_regime  = float(latest.get("vol_regime",   1) or 1)

    rule_regime = _rule_score(vix, spy_3m, credit, yc, vol_regime)

    # Smooth with KMeans when we have 30+ snapshots
    if len(macro_history) >= 30:
        try:
            from sklearn.cluster import KMeans
            X = np.array([
                [m.get("vix", 20), m.get("spy_3m", 0),
                 m.get("credit_spread_signal", 0), m.get("vol_regime", 1)]
                for m in macro_history
            ], dtype=float)
            X = np.nan_to_num(X)
            km = KMeans(n_clusters=4, random_state=42, n_init="auto")
            labels = km.fit_predict(X)
            current_centroid = km.cluster_centers_[labels[-1]]
            # Map centroid to regime by rule
            c_vix, c_spy, c_cred, c_vol = current_centroid
            ml_regime = _rule_score(c_vix, c_spy, c_cred, 0.0, c_vol)
            # Majority vote between rule-based and ML
            if ml_regime == rule_regime:
                return rule_regime
            # Use recent trend to break tie
            recent = macro_history[-5:]
            scores = [_rule_score(
                float(m.get("vix", 20) or 20), float(m.get("spy_3m", 0) or 0),
                float(m.get("credit_spread_signal", 0) or 0),
                float(m.get("yield_curve", 0) or 0),
                float(m.get("vol_regime", 1) or 1)
            ) for m in recent]
            from collections import Counter
            return Counter(scores).most_common(1)[0][0]
        except Exception as e:
            logger.debug("KMeans regime failed: %s", e)

    return rule_regime


def compute_regime_series(spy_prices, vix_series) -> pd.Series:
    """
    Compute a regime label for each trading day in the historical dataset.
    Used by the walk-forward backtester.
    """
    def _to_series(v, name):
        if isinstance(v, pd.DataFrame):
            return v.iloc[:, 0].rename(name)
        if isinstance(v, pd.Series):
            return v.rename(name)
        return pd.Series(v, name=name)

    spy = _to_series(spy_prices, "spy")
    vix = _to_series(vix_series, "vix")

    df = pd.DataFrame({"spy": spy, "vix": vix}).dropna()
    df["spy_ret_3m"]  = df["spy"].pct_change(63) * 100
    df["vol_30d"]     = df["spy"].pct_change().rolling(30).std() * np.sqrt(252)
    df["vol_90d"]     = df["spy"].pct_change().rolling(90).std() * np.sqrt(252)
    df["vol_regime"]  = df["vol_30d"] / (df["vol_90d"] + 1e-6)
    df = df.dropna()

    def _row_regime(row):
        return _rule_score(
            float(row["vix"]),
            float(row["spy_ret_3m"]),
            0.0,   # credit not available in historical loop
            0.0,
            float(row["vol_regime"]),
        )

    return df.apply(_row_regime, axis=1).rename("regime")
