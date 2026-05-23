"""
Point-in-time feature computation from price + volume history.
No lookahead. All windows are strictly backward-looking.

New in this version:
  - Short-term reversal factor (1-week)
  - Betting-Against-Beta proxy (market beta estimate)
  - Price acceleration (2nd derivative of momentum)
  - 52-week high proximity (anchoring / breakout signal)
  - Realized skewness (tail risk measure)
"""
import numpy as np


def _safe_div(a, b, default=0.0):
    try:
        b = float(b)
        return float(a) / b if abs(b) > 1e-12 else default
    except Exception:
        return default


def _linreg_slope(y: np.ndarray) -> float:
    """Compute OLS slope of y vs equally-spaced x."""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x -= x.mean()
    y = y - y.mean()
    return float(_safe_div(np.dot(x, y), np.dot(x, x)))


def compute_features_from_prices(price_series, volume_series=None):
    """
    Returns a feature dict for a single stock.
    price_series: array-like of close prices (oldest first)
    volume_series: matching array of volumes (optional)
    """
    if price_series is None:
        return {}
    c = np.array(price_series, dtype=float)
    if len(c) < 60:
        return {}

    v = (
        np.array(volume_series, dtype=float)
        if volume_series is not None and len(volume_series) == len(c)
        else None
    )

    f = {}

    # ── MOMENTUM ──────────────────────────────────────────────────────────────
    # 12-1 skip-month (Jegadeesh-Titman)
    if len(c) >= 252:
        f["momentum_12_1"] = (_safe_div(c[-21], c[-252]) - 1.0) * 100
    elif len(c) >= 84:
        f["momentum_12_1"] = (_safe_div(c[-21], c[0]) - 1.0) * 100
    else:
        f["momentum_12_1"] = 0.0

    f["ret_6m"]  = (_safe_div(c[-1], c[-126]) - 1.0) * 100 if len(c) >= 126 else 0.0
    f["ret_3m"]  = (_safe_div(c[-1], c[-63])  - 1.0) * 100 if len(c) >= 63  else 0.0
    f["ret_1m"]  = (_safe_div(c[-1], c[-21])  - 1.0) * 100 if len(c) >= 21  else 0.0
    f["ret_1w"]  = (_safe_div(c[-1], c[-5])   - 1.0) * 100 if len(c) >= 5   else 0.0

    # Short-term reversal: 1-week return (contrarian signal — invert before use)
    f["reversal"] = -f["ret_1w"]  # negative 1-week → potential bounce

    # Price acceleration: 3m momentum minus 6m momentum (is momentum strengthening?)
    f["price_accel"] = f["ret_3m"] - f["ret_6m"]

    # ── VOLATILITY ────────────────────────────────────────────────────────────
    window = min(31, len(c))
    r_30 = np.diff(c[-window:]) / (c[-window:-1] + 1e-10)
    vol_30d = float(np.nanstd(r_30) * np.sqrt(252) * 100)
    f["vol"] = vol_30d

    if len(c) >= 91:
        r_90 = np.diff(c[-91:]) / (c[-91:-1] + 1e-10)
        vol_90d = float(np.nanstd(r_90) * np.sqrt(252) * 100)
        f["vol_regime"] = _safe_div(vol_30d, vol_90d + 1e-6)
    else:
        f["vol_regime"] = 1.0

    # Realized skewness (negative skew = fat left tail = more risky)
    if len(r_30) >= 20:
        mu, sigma = np.mean(r_30), np.std(r_30) + 1e-10
        f["skewness"] = float(np.mean(((r_30 - mu) / sigma) ** 3))
    else:
        f["skewness"] = 0.0

    # ── MEAN REVERSION ────────────────────────────────────────────────────────
    if len(c) >= 50:
        sma50 = float(np.mean(c[-50:]))
        f["mean_reversion"] = _safe_div(c[-1] - sma50, vol_30d / 100 * sma50 + 1e-6)
    else:
        f["mean_reversion"] = 0.0

    # ── TREND ─────────────────────────────────────────────────────────────────
    if len(c) >= 200:
        f["trend_strength"] = 1.0 if c[-1] > float(np.mean(c[-200:])) else -1.0
        # Distance from 52-week high (proximity = breakout signal)
        high_52w = float(np.max(c[-252:])) if len(c) >= 252 else float(np.max(c))
        f["high_52w_proximity"] = _safe_div(c[-1], high_52w)
    else:
        f["trend_strength"]    = 0.0
        f["high_52w_proximity"] = 0.5

    # Slope of price / SMA200 over past 20 days (trend acceleration)
    if len(c) >= 220:
        sma200_recent = np.array([float(np.mean(c[i-200:i])) for i in range(len(c)-20, len(c))])
        price_ratio = c[-20:] / (sma200_recent + 1e-6)
        f["trend_slope"] = _linreg_slope(price_ratio)
    else:
        f["trend_slope"] = 0.0

    # ── OBV TREND + AMIHUD ────────────────────────────────────────────────────
    if v is not None and len(v) >= 21:
        price_chg = np.sign(np.diff(c[-21:]))   # 20 elements
        obv = np.cumsum(price_chg * v[-20:])    # both 20
        obv_std = np.std(obv) + 1e-6
        f["flow_signal"] = float(_safe_div(obv[-1] - obv[0], obv_std * len(obv)))

        # Amihud illiquidity
        daily_ret_abs = np.abs(np.diff(c[-21:]) / (c[-21:-1] + 1e-10))  # 20 elements
        dollar_vol = (c[-20:] * v[-20:]) + 1e-6                           # 20 elements
        amihud = float(np.mean(daily_ret_abs / dollar_vol))
        f["liquidity"] = float(-np.log1p(amihud * 1e6))

        up_days   = v[-20:][price_chg > 0] if (price_chg > 0).any() else np.array([0.0])
        down_days = v[-20:][price_chg < 0] if (price_chg < 0).any() else np.array([0.0])
        f["volume_trend"] = _safe_div(float(np.mean(up_days)), float(np.mean(down_days)) + 1e-6)
    else:
        f["flow_signal"]  = float(np.sum(np.sign(np.diff(c[-20:]))))
        f["liquidity"]    = 0.0
        f["volume_trend"] = 1.0

    # ── EARNINGS SURPRISE PROXY ───────────────────────────────────────────────
    if len(c) >= 25:
        recent_ret = _safe_div(c[-1], c[-6]) - 1.0
        prior_vol = np.std(np.diff(c[-26:-6]) / (c[-26:-7] + 1e-10)) + 1e-6
        f["earnings_surprise"] = float(recent_ret / prior_vol)
    else:
        f["earnings_surprise"] = 0.0

    # ── BETA PROXY (vs market using rolling 63-day correlation) ───────────────
    # Requires external market return, so we store realized vol ratio as beta proxy.
    # True beta computed cross-sectionally in normalization when SPY returns available.
    f["beta_proxy"] = _safe_div(vol_30d, 15.0)  # normalized by avg market vol ~15%

    # rel_strength placeholder (overridden cross-sectionally)
    f["rel_strength"] = f["ret_3m"]

    return f
