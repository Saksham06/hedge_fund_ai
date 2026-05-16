"""
Risk Model — dynamic volatility targeting + drawdown control.

Improvements:
  - Dynamic vol targeting: scales exposure using realized vol ratio
    rather than a fixed annual target
  - Regime-aware scaling: more aggressive reduction in crisis regimes
  - Cash buffer sizing: proportional to vol regime
  - CVaR-based position limits
"""
import logging

import numpy as np
import pandas as pd

from hedge_fund_ai.config import TARGET_VOL

logger = logging.getLogger(__name__)


def vol_target_weights(returns: pd.Series, target_vol: float = TARGET_VOL) -> float:
    vol = float(returns.std() * np.sqrt(252)) + 1e-8
    return float(min(target_vol / vol, 2.0))


def risk_parity_weights(vols: list | np.ndarray) -> np.ndarray:
    inv = 1.0 / (np.array(vols, dtype=float) + 1e-6)
    return inv / inv.sum()


def drawdown_control(equity_curve: pd.Series) -> float:
    """3-level drawdown scaling."""
    if len(equity_curve) < 2:
        return 1.0
    peak = equity_curve.cummax()
    dd   = float((equity_curve.iloc[-1] - peak.iloc[-1]) / (peak.iloc[-1] + 1e-10))
    if dd < -0.25: return 0.40
    if dd < -0.15: return 0.65
    if dd < -0.08: return 0.85
    return 1.0


def dynamic_vol_target_scale(
    equity: pd.Series,
    target_vol: float = TARGET_VOL,
    lookback_short: int = 21,
    lookback_long:  int = 63,
    regime: str = "neutral",
) -> float:
    """
    Dynamic volatility targeting with regime overlay.

    Logic:
      - Compute realized vol over short (21d) and long (63d) windows
      - Scale factor = target_vol / realized_vol (capped at 1.0)
      - In crisis regime: additional 0.75x multiplier
      - In risk_off regime: additional 0.90x multiplier

    Returns a scale factor in [0.20, 1.00].
    """
    if equity is None or len(equity) < lookback_short + 2:
        return 1.0

    rets = equity.pct_change().dropna()

    # Short-window vol (responsive to recent stress)
    if len(rets) >= lookback_short:
        vol_short = float(rets.tail(lookback_short).std() * np.sqrt(252))
    else:
        vol_short = TARGET_VOL

    # Long-window vol (stable baseline)
    if len(rets) >= lookback_long:
        vol_long = float(rets.tail(lookback_long).std() * np.sqrt(252))
    else:
        vol_long = vol_short

    # Use max of short/long (conservative — respond to vol spikes quickly)
    realized_vol = max(vol_short, vol_long, 1e-4)
    scale = float(min(target_vol / realized_vol, 1.0))

    # Regime overlay
    regime_multiplier = {
        "crisis":   0.70,
        "risk_off": 0.88,
        "neutral":  1.00,
        "risk_on":  1.00,
    }.get(regime.lower(), 1.00)

    final_scale = float(np.clip(scale * regime_multiplier, 0.20, 1.00))

    if final_scale < 0.95:
        logger.info(
            "Vol targeting: realized=%.1f%% target=%.1f%% → scale=%.2f (regime=%s → %.2f final)",
            realized_vol * 100, target_vol * 100, scale, regime, final_scale
        )

    return final_scale


def compute_cash_buffer(vol_regime: float, regime: str = "neutral") -> float:
    """
    Compute cash reserve as a fraction of portfolio.
    Higher vol regime → larger cash buffer.

    vol_regime: ratio of short-vol to long-vol (>1 = expanding vol)
    Returns fraction to hold in cash [0, 0.30].
    """
    base_cash = {
        "crisis":   0.20,
        "risk_off": 0.10,
        "neutral":  0.05,
        "risk_on":  0.02,
    }.get(regime.lower(), 0.05)

    # Amplify by vol regime
    vol_amp = max(0, (vol_regime - 1.0) * 0.10)
    return float(np.clip(base_cash + vol_amp, 0.0, 0.30))


def position_cvar_limit(
    returns: pd.Series,
    max_cvar_contribution: float = 0.025,
    confidence: float = 0.95,
) -> float:
    """
    Maximum weight for a position given its CVaR and a portfolio CVaR budget.
    Returns max weight as a fraction.
    """
    if returns is None or len(returns) < 20:
        return 0.20  # default cap

    rets = returns.dropna().values
    cutoff = np.percentile(rets, (1 - confidence) * 100)
    tail   = rets[rets <= cutoff]
    cvar   = float(abs(np.mean(tail))) if len(tail) > 0 else 0.02

    # Max weight = CVaR budget / position CVaR
    max_w = max_cvar_contribution / (cvar + 1e-6)
    return float(np.clip(max_w, 0.03, 0.20))
