"""
Elite Portfolio Optimizer.

Improvements:
  1. CVaR (Conditional Value-at-Risk) position sizing
  2. Fractional Kelly criterion for optimal bet sizing
  3. Black-Litterman views integration (signal as views)
  4. Ledoit-Wolf covariance (unchanged — already correct)
  5. Sector constraints inside optimizer
  6. Vol targeting
  7. Herfindahl concentration penalty
"""
import logging

import numpy as np
import pandas as pd
import yfinance as yf

from hedge_fund_ai.config import (
    CVAR_CONFIDENCE, KELLY_FRACTION,
    MAX_POSITION_WEIGHT, MAX_SECTOR_WEIGHT,
    MIN_POSITION_WEIGHT, TARGET_VOL,
)

logger = logging.getLogger(__name__)


# ── Covariance estimation ─────────────────────────────────────────────────────

def _ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        lw.fit(returns.values)
        return lw.covariance_
    except Exception:
        return np.cov(returns.values.T) if returns.shape[1] > 1 else returns.var().values.reshape(1, 1)


# ── CVaR position sizing ──────────────────────────────────────────────────────

def cvar_position_sizes(returns: pd.DataFrame, confidence: float = CVAR_CONFIDENCE) -> np.ndarray:
    """
    CVaR-based position sizing: allocate inversely to each stock's CVaR.
    CVaR = expected loss in the worst (1-confidence)% of days.
    Lower CVaR → higher weight.
    """
    n = returns.shape[1]
    cvars = []
    for col in returns.columns:
        r = returns[col].dropna().values
        if len(r) < 20:
            cvars.append(0.02)  # default 2% CVaR
            continue
        cutoff = np.percentile(r, (1 - confidence) * 100)
        tail   = r[r <= cutoff]
        cvar   = float(abs(np.mean(tail))) if len(tail) > 0 else 0.02
        cvars.append(max(cvar, 1e-4))

    inv_cvar = 1.0 / np.array(cvars)
    return inv_cvar / inv_cvar.sum()


# ── Fractional Kelly sizing ───────────────────────────────────────────────────

def kelly_weights(scores: np.ndarray, returns: pd.DataFrame,
                  fraction: float = KELLY_FRACTION) -> np.ndarray:
    """
    Fractional Kelly criterion applied cross-sectionally.
    Kelly weight ∝ expected_return / variance (signal / vol²).
    Fraction < 1.0 reduces drawdown at cost of some expected return.
    """
    if len(scores) != returns.shape[1]:
        return np.ones(len(scores)) / len(scores)

    vols = returns.std().values + 1e-8
    # Treat z-scored signal as expected return proxy (normalized to variance scale)
    signal_scaled = np.clip(scores, 0, None)  # only positive expected returns
    kelly = signal_scaled / (vols ** 2 + 1e-8)
    kelly = np.clip(kelly, 0, None)
    total = kelly.sum()
    if total < 1e-9:
        return np.ones(len(scores)) / len(scores)
    return (kelly / total) * fraction + (1 - fraction) * np.ones(len(scores)) / len(scores)


# ── Black-Litterman views ─────────────────────────────────────────────────────

def black_litterman_weights(
    cov: np.ndarray,
    market_weights: np.ndarray,
    views: np.ndarray,          # expected excess returns from signal
    tau: float = 0.05,          # uncertainty in prior
    omega_scale: float = 0.1,   # view uncertainty
) -> np.ndarray:
    """
    Black-Litterman: blend market equilibrium with signal views.
    views: signal z-scores used as expected return views.
    Returns posterior expected return vector (not weights directly).
    """
    n = len(market_weights)
    # Risk aversion from Sharpe ≈ 0.5 (conservative)
    lam = 2.5

    # Implied equilibrium returns
    pi = lam * cov @ market_weights

    # View matrix: each stock has its own absolute view
    P = np.eye(n)
    q = views.copy()

    # Omega: diagonal uncertainty matrix
    Omega = omega_scale * np.diag(np.diag(P @ (tau * cov) @ P.T))

    # BL posterior
    M1 = np.linalg.inv(tau * cov)
    M2 = P.T @ np.linalg.inv(Omega) @ P
    mu_bl = np.linalg.solve(M1 + M2, M1 @ pi + P.T @ np.linalg.inv(Omega) @ q)

    # Convert posterior returns to weights via mean-variance
    w_bl = np.linalg.solve(lam * cov, mu_bl)
    w_bl = np.clip(w_bl, 0, None)
    total = w_bl.sum()
    if total < 1e-9:
        return market_weights
    return w_bl / total


# ── Constraint application ────────────────────────────────────────────────────

def _herfindahl_smooth(w: np.ndarray, max_ratio: float = 2.5) -> np.ndarray:
    n = len(w)
    if n == 0:
        return w
    ew_hhi = 1.0 / n
    for _ in range(50):
        if np.sum(w ** 2) <= max_ratio * ew_hhi + 1e-8:
            break
        w = 0.80 * w + 0.20 * np.ones(n) / n
        w /= w.sum() + 1e-9
    return w


def _cap_positions(w: np.ndarray, cap: float = MAX_POSITION_WEIGHT) -> np.ndarray:
    for _ in range(50):
        w = np.minimum(w, cap)
        total = w.sum()
        if total < 1e-9:
            return np.ones(len(w)) / len(w)
        w /= total
        if w.max() <= cap + 1e-8:
            break
    return w


def _sector_constraints(weights: dict, sectors: dict) -> dict:
    for _ in range(20):
        sec_tot = {}
        for t, w in weights.items():
            s = sectors.get(t, "Unknown")
            sec_tot[s] = sec_tot.get(s, 0.0) + w
        over = {s: v for s, v in sec_tot.items() if v > MAX_SECTOR_WEIGHT + 1e-5}
        if not over:
            break
        for sector, tot in over.items():
            scale = MAX_SECTOR_WEIGHT / tot
            for t in list(weights):
                if sectors.get(t) == sector:
                    weights[t] *= scale
        total = sum(weights.values()) or 1e-9
        weights = {t: v / total for t, v in weights.items()}
    return weights


def _apply_position_limits(weights: dict, cap: float = MAX_POSITION_WEIGHT) -> dict:
    tickers = list(weights.keys())
    w_arr   = np.array([weights[t] for t in tickers], dtype=float)
    w_arr   = _cap_positions(w_arr, cap)
    return dict(zip(tickers, w_arr.tolist()))


def _vol_target(weights: dict, cov: np.ndarray, tickers: list) -> dict:
    w = np.array([weights.get(t, 0.0) for t in tickers])
    var = float(w @ cov @ w) * 252
    vol = np.sqrt(max(var, 1e-10))
    scale = min(TARGET_VOL / vol, 1.0)
    return {t: v * scale for t, v in weights.items()}


# ── Master optimizer ──────────────────────────────────────────────────────────

def optimize_portfolio(data: list, top_stocks: list) -> list:
    """
    Elite portfolio construction:
      1. Ledoit-Wolf covariance
      2. Black-Litterman blending (signal as views)
      3. CVaR position sizing
      4. Fractional Kelly overlay
      5. Blend all three methods: 40% BL, 30% CVaR, 30% Kelly
      6. Herfindahl penalty
      7. Position cap (20%)
      8. Sector constraints
      9. Vol targeting (12%)
    """
    if not top_stocks:
        return []

    sectors = {d.get("ticker"): d.get("sector", "Unknown") for d in data if d.get("ticker")}
    scores  = {d.get("ticker"): float(d.get("score", 0)) for d in data if d.get("ticker")}

    try:
        price_df = yf.download(top_stocks, period="6mo", progress=False)["Close"]
        if isinstance(price_df, pd.Series):
            price_df = price_df.to_frame(name=top_stocks[0])
        price_df = price_df.dropna(axis=1, how="all")

        valid = [t for t in top_stocks if t in price_df.columns]
        if not valid:
            raise ValueError("no price data")

        ret = price_df[valid].pct_change().dropna()
        if len(ret) < 20:
            raise ValueError("insufficient returns")

        cov       = _ledoit_wolf_cov(ret)
        n         = len(valid)
        sc_arr    = np.array([scores.get(t, 0.0) for t in valid])

        # Market weights: equal-weight as prior
        market_w  = np.ones(n) / n

        # Method 1: Black-Litterman
        try:
            bl_w = black_litterman_weights(cov, market_w, sc_arr)
        except Exception:
            bl_w = market_w.copy()

        # Method 2: CVaR-inverse sizing
        cvar_w = cvar_position_sizes(ret, confidence=CVAR_CONFIDENCE)

        # Method 3: Fractional Kelly
        sc_pos = np.clip(sc_arr, 0, None)
        kelly_w = kelly_weights(sc_pos, ret, fraction=KELLY_FRACTION)

        # Blend
        blended = 0.40 * bl_w + 0.30 * cvar_w + 0.30 * kelly_w
        blended = np.clip(blended, 0, None)
        blended /= blended.sum() + 1e-9

        # Concentration control
        blended = _herfindahl_smooth(blended)
        blended = _cap_positions(blended)

        weights = dict(zip(valid, blended.tolist()))

        # Remove sub-minimum positions
        weights = {t: w for t, w in weights.items() if w >= MIN_POSITION_WEIGHT}
        total   = sum(weights.values()) or 1e-9
        weights = {t: w / total for t, w in weights.items()}

        # Sector constraints
        weights = _sector_constraints(weights, sectors)

        # Vol targeting
        weights = _vol_target(weights, cov, list(weights.keys()))

        portfolio = sorted([
            {
                "ticker": t,
                "weight": round(float(w), 4),
                "sector": sectors.get(t, "Unknown"),
                "score":  round(scores.get(t, 0.0), 4),
            }
            for t, w in weights.items() if w >= MIN_POSITION_WEIGHT
        ], key=lambda x: -x["weight"])

        logger.info("Portfolio: %d pos | gross=%.1f%% | max=%.1f%%",
                    len(portfolio),
                    sum(p["weight"] for p in portfolio) * 100,
                    max((p["weight"] for p in portfolio), default=0) * 100)
        return portfolio

    except Exception as e:
        logger.warning("Optimizer failed: %s — equal-weight fallback", e)
        n = len(top_stocks)
        return [
            {"ticker": t, "weight": round(1.0/n, 4),
             "sector": sectors.get(t, "Unknown"), "score": 0}
            for t in top_stocks
        ]
