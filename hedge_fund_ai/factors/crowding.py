"""
Factor Crowding Monitor.

Measures how many institutional players hold the same positions.
High crowding → alpha decay risk, liquidity crunch on unwind, violent reversals.

Methods:
  1. 13F overlap: compare our portfolio to aggregate institutional holdings
     (via yfinance .institutional_holders)
  2. Crowding score: 0–1 (0 = unique, 1 = everyone owns it)
  3. Capacity-adjusted weight reduction for crowded factors
  4. Turnover-adjusted signal optimization (penalize high-churn factors)
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from hedge_fund_ai.data.cache import simple_cache

logger = logging.getLogger(__name__)

# Crowding thresholds
HIGH_CROWDING_THRESHOLD  = 0.60
CROWDING_WEIGHT_PENALTY  = 0.30  # reduce score by this fraction when crowded


@simple_cache(ttl=3600 * 6, disk=True)
def get_institutional_ownership_pct(ticker: str) -> float:
    """
    Return fraction of float held by institutions (0–1).
    Higher = more institutionally crowded.
    """
    try:
        info = yf.Ticker(ticker).info or {}
        held = info.get("heldPercentInstitutions")
        if held is not None:
            return float(np.clip(held, 0, 1))
    except Exception as e:
        logger.debug("Inst ownership %s: %s", ticker, e)
    return 0.5  # default: assume 50% institutional


@simple_cache(ttl=3600 * 24, disk=True)
def get_top_holders(ticker: str) -> list[dict]:
    """Return top institutional holders from yfinance."""
    try:
        holders = yf.Ticker(ticker).institutional_holders
        if holders is not None and not holders.empty:
            return holders.head(15).to_dict("records")
    except Exception:
        pass
    return []


def compute_crowding_score(
    portfolio_tickers: list[str],
    adv_map: dict | None = None,
) -> dict:
    """
    Compute crowding score per ticker.

    Components:
      1. Institutional ownership % (from 13F proxy)
      2. Concentration: how concentrated is ownership (top-5 / total)
      3. Liquidity relative to ownership (large positions in illiquid stocks = crowding risk)

    Returns {ticker: {"crowding_score": 0-1, "risk_level": str, "details": dict}}
    """
    results = {}

    for ticker in portfolio_tickers:
        try:
            inst_pct = get_institutional_ownership_pct(ticker)
            holders  = get_top_holders(ticker)

            # Concentration: if top holders hold >50% of institutional shares
            if holders:
                top_shares = sum(float(h.get("Shares", 0) or 0) for h in holders[:5])
                all_shares = sum(float(h.get("Shares", 0) or 0) for h in holders)
                concentration = top_shares / (all_shares + 1e-9)
            else:
                concentration = 0.5

            # Liquidity relative to institutional ownership
            adv = float((adv_map or {}).get(ticker, 50_000_000))
            # If institutions hold >$1B and ADV < $10M, crowding risk is high
            liquidity_risk = min(1.0, max(0.0, 1.0 - adv / 100_000_000))

            # Composite crowding score
            score = 0.50 * inst_pct + 0.30 * concentration + 0.20 * liquidity_risk
            score = float(np.clip(score, 0, 1))

            if score >= HIGH_CROWDING_THRESHOLD:
                risk_level = "high"
            elif score >= 0.40:
                risk_level = "medium"
            else:
                risk_level = "low"

            results[ticker] = {
                "crowding_score":     round(score, 3),
                "risk_level":         risk_level,
                "inst_ownership_pct": round(inst_pct, 3),
                "concentration":      round(concentration, 3),
                "liquidity_risk":     round(liquidity_risk, 3),
            }

            if risk_level == "high":
                logger.warning("HIGH CROWDING: %s score=%.2f (inst=%.0f%%)",
                               ticker, score, inst_pct * 100)

        except Exception as e:
            logger.debug("Crowding %s: %s", ticker, e)
            results[ticker] = {"crowding_score": 0.5, "risk_level": "unknown"}

    return results


def apply_crowding_penalty(
    scores: dict[str, float],
    crowding: dict[str, dict],
) -> dict[str, float]:
    """
    Reduce composite signal scores for highly crowded stocks.
    High crowding → alpha likely already priced in; reduce weight.
    """
    adjusted = {}
    for ticker, score in scores.items():
        cr = crowding.get(ticker, {})
        cs = float(cr.get("crowding_score", 0.5))

        if cs >= HIGH_CROWDING_THRESHOLD:
            penalty = CROWDING_WEIGHT_PENALTY * (cs - HIGH_CROWDING_THRESHOLD) / 0.40
            adjusted[ticker] = score * (1 - float(np.clip(penalty, 0, 0.50)))
        else:
            adjusted[ticker] = score

    return adjusted


# ── Turnover-adjusted signal optimization ─────────────────────────────────────

def compute_factor_turnover(
    current_scores:  dict[str, float],
    previous_scores: dict[str, float],
) -> float:
    """
    Measure how much a factor's cross-sectional ranks changed vs prior period.
    Higher = more turnover = higher transaction cost to follow this factor.

    Returns monthly rank-change turnover (0 = stable, 1 = completely reshuffled).
    """
    if not current_scores or not previous_scores:
        return 0.5  # unknown

    common = set(current_scores) & set(previous_scores)
    if len(common) < 5:
        return 0.5

    # Rank stability: Spearman correlation between period ranks
    from scipy.stats import spearmanr
    curr_vals = [current_scores[t] for t in common]
    prev_vals = [previous_scores[t] for t in common]

    try:
        import warnings
        if np.std(curr_vals) < 1e-10 or np.std(prev_vals) < 1e-10:
            return 0.5  # constant scores — no rank change information
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, _ = spearmanr(curr_vals, prev_vals)
        rho = float(rho) if not np.isnan(rho) else 0.0
        return float(np.clip(1 - rho, 0, 1))
    except Exception:
        return 0.5


def cost_adjusted_ic_hurdle(
    raw_ic_ir:       float,
    factor_turnover: float,
    cost_bps:        float = 10.0,
    annual_periods:  int   = 12,
) -> dict:
    """
    Compute the minimum IC-IR required for a factor to be cost-positive.

    Logic:
      - High-turnover factors incur more transaction costs
      - A factor must have IC-IR > (turnover × cost_rate) / signal_vol to be worth trading
      - If raw IC-IR < cost-adjusted hurdle → factor not worth trading at this turnover rate

    Returns {hurdle, passes, net_ic_ir}.
    """
    # Cost hurdle: roughly proportional to turnover × cost
    # At 100% monthly turnover and 10bps cost: hurdle ≈ 0.24 annualized IC-IR
    hurdle     = factor_turnover * (cost_bps / 10_000) * annual_periods * 2.0
    net_ic_ir  = raw_ic_ir - hurdle
    passes     = net_ic_ir > 0

    return {
        "raw_ic_ir":     round(raw_ic_ir, 3),
        "hurdle":        round(hurdle, 3),
        "net_ic_ir":     round(net_ic_ir, 3),
        "factor_turnover": round(factor_turnover, 3),
        "passes":        passes,
    }
