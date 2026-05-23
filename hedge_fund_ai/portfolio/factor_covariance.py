"""
Simplified Factor-Model Covariance (Barra-style, price-data only).

Decomposes returns into:
  1. Market factor (SPY beta)
  2. Momentum factor (12-1 cross-sectional)
  3. Quality factor (low-vol proxy for quality/BAB)
  4. Idiosyncratic residual (stock-specific)

This separation improves optimization by:
  - Correctly attributing systematic vs stock-specific risk
  - Preventing false diversification (stocks that look uncorrelated
    but share the same momentum/quality factor loading)
  - Enabling proper factor-level risk limits
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def estimate_factor_covariance(
    returns: pd.DataFrame,
    market_returns: pd.Series | None = None,
) -> dict:
    """
    Estimate factor-model covariance from a returns DataFrame.

    Args:
        returns:        T × N DataFrame of daily stock returns
        market_returns: T-length Series of market (SPY) daily returns

    Returns dict with:
        total_cov:      N×N total covariance matrix (for optimization)
        factor_cov:     K×K factor covariance matrix
        specific_var:   N-length diagonal of idiosyncratic variance
        factor_loadings: N×K loading matrix (beta, momentum, quality)
        factor_names:   list of factor names
        r_squared:      N-length array of R² per stock
    """
    T, N = returns.shape
    tickers = list(returns.columns)

    if T < 60:
        logger.warning("Too few observations (%d) for factor model — using full covariance", T)
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        lw.fit(returns.values)
        return {
            "total_cov":      lw.covariance_,
            "factor_cov":     None,
            "specific_var":   np.diag(lw.covariance_),
            "factor_loadings": None,
            "factor_names":   None,
            "r_squared":      np.zeros(N),
            "method":         "ledoit_wolf_fallback",
        }

    # ── Build factor return series ────────────────────────────────────────────

    # Factor 1: Market (SPY or equal-weight universe)
    if market_returns is not None and len(market_returns) >= T:
        mkt = market_returns.reindex(returns.index).fillna(0).values
    else:
        mkt = returns.mean(axis=1).values
    mkt = (mkt - mkt.mean()) / (mkt.std() + 1e-8)

    # Factor 2: Cross-sectional momentum (past 63-day return as of each day)
    # Approximate: use rolling 63-day cumulative return as a factor portfolio
    rolling_mom = returns.rolling(63).sum().shift(1)   # skip last month
    mom_factor  = rolling_mom.rank(axis=1, pct=True).subtract(0.5)  # long top, short bottom
    mom_factor  = mom_factor.mean(axis=1).fillna(0).values           # equal-weight long-short
    mom_factor  = (mom_factor - mom_factor.mean()) / (mom_factor.std() + 1e-8)

    # Factor 3: Quality/Low-vol (inverse realized vol)
    rolling_vol = returns.rolling(21).std()
    inv_vol     = 1.0 / (rolling_vol.shift(1) + 1e-8)
    # Normalize to portfolio weights
    inv_vol_norm = inv_vol.divide(inv_vol.sum(axis=1), axis=0).fillna(0)
    qual_factor  = (inv_vol_norm * returns).sum(axis=1).values
    qual_factor  = (qual_factor - qual_factor.mean()) / (qual_factor.std() + 1e-8)

    # Stack factors: T × K
    F = np.column_stack([mkt, mom_factor, qual_factor])
    factor_names = ["market", "momentum", "quality"]
    K = F.shape[1]

    # ── OLS regression per stock: R = B @ F + e ───────────────────────────────
    R = returns.values   # T × N

    # Add intercept
    X = np.column_stack([np.ones(T), F])   # T × (K+1)

    B_full = np.linalg.lstsq(X, R, rcond=None)[0]   # (K+1) × N
    B      = B_full[1:, :]                            # K × N  (drop intercept)
    B      = B.T                                       # N × K  (loadings)

    # Residuals
    fitted = (X @ B_full)   # T × N
    resid  = R - fitted       # T × N

    # R² per stock
    ss_res = np.var(resid, axis=0)
    ss_tot = np.var(R, axis=0) + 1e-10
    r2     = np.clip(1 - ss_res / ss_tot, 0, 1)

    # Factor covariance: K × K
    factor_cov = np.cov(F.T) if K > 1 else np.array([[np.var(F)]])

    # Specific variance: diagonal (idiosyncratic)
    specific_var = np.var(resid, axis=0)

    # Total covariance: B @ Σ_F @ B.T + diag(σ²_e)
    total_cov = B @ factor_cov @ B.T + np.diag(specific_var)

    # Shrinkage blend with sample covariance for stability
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf()
    lw.fit(R)
    alpha = 0.30  # 30% sample, 70% factor model
    blended_cov = (1 - alpha) * total_cov + alpha * lw.covariance_

    logger.debug("Factor model: R²=%.3f mean | %d stocks | %d factors",
                 float(np.mean(r2)), N, K)

    return {
        "total_cov":       blended_cov,
        "factor_cov":      factor_cov,
        "specific_var":    specific_var,
        "factor_loadings": B,           # N × K
        "factor_names":    factor_names,
        "tickers":         tickers,
        "r_squared":       r2,
        "method":          "3_factor_model",
    }


def compute_factor_risk_attribution(
    weights:        np.ndarray,
    factor_result:  dict,
) -> dict:
    """
    Decompose portfolio variance into factor vs idiosyncratic components.

    Returns {
      total_variance, factor_variance, specific_variance,
      factor_pct, specific_pct, factor_contributions: {name: pct}
    }
    """
    B   = factor_result.get("factor_loadings")
    Sf  = factor_result.get("factor_cov")
    sv  = factor_result.get("specific_var")
    cov = factor_result.get("total_cov")

    if B is None or Sf is None or sv is None:
        port_var = float(weights @ cov @ weights) if cov is not None else 0
        return {"total_variance": port_var, "factor_pct": 0.5, "specific_pct": 0.5}

    w = np.array(weights)

    factor_var   = float(w @ B @ Sf @ B.T @ w)
    specific_var = float(w @ np.diag(sv) @ w)
    total_var    = factor_var + specific_var + 1e-12

    factor_names = factor_result.get("factor_names", [])
    contributions = {}
    for k, fname in enumerate(factor_names):
        bk = B[:, k]
        var_k = float((w @ bk) ** 2 * Sf[k, k])
        contributions[fname] = round(var_k / total_var * 100, 2)

    return {
        "total_variance":       round(total_var, 8),
        "factor_variance":      round(factor_var, 8),
        "specific_variance":    round(specific_var, 8),
        "factor_pct":           round(factor_var / total_var * 100, 2),
        "specific_pct":         round(specific_var / total_var * 100, 2),
        "factor_contributions": contributions,
    }
