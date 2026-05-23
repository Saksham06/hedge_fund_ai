"""
Financial Modeling Prep (FMP) Integration.

FMP provides data yfinance and SimFin don't:
  - Actual EPS surprises (reported vs estimate)
  - Analyst estimate revisions (upgrade/downgrade momentum)
  - Institutional ownership changes (13F filings)
  - Earnings call sentiment (management tone)
  - Real-time price targets

All endpoints require FMP_API_KEY from config.
Free tier: 250 requests/day. Paid tier: unlimited.

If FMP_API_KEY is empty, all functions return empty dicts gracefully.
"""
import logging
import time
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import requests

from hedge_fund_ai.config import FMP_API_KEY, FMP_BASE
from hedge_fund_ai.data.cache import simple_cache

logger = logging.getLogger(__name__)


def _fmp_get(endpoint: str, params: dict = None, timeout: int = 10) -> list | dict | None:
    """Make a GET request to FMP API. Returns None on any error."""
    if not FMP_API_KEY:
        return None
    try:
        p = dict(params or {})
        p["apikey"] = FMP_API_KEY
        r = requests.get(f"{FMP_BASE}/{endpoint}", params=p, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "Error Message" in data:
            logger.debug("FMP error for %s: %s", endpoint, data["Error Message"])
            return None
        return data
    except Exception as e:
        logger.debug("FMP request failed %s: %s", endpoint, e)
        return None


# ── 1. Earnings Surprises ─────────────────────────────────────────────────────

@simple_cache(ttl=3600 * 6, disk=True)
def get_earnings_surprises(ticker: str, limit: int = 8) -> list[dict]:
    """
    Actual earnings surprises: reported EPS vs consensus estimate.
    Returns list of {date, actualEPS, estimatedEPS, surprise_pct} sorted newest first.
    Cached 6 hours.
    """
    data = _fmp_get(f"earnings-surprises/{ticker}", {"limit": limit})
    if not data or not isinstance(data, list):
        return []

    results = []
    for item in data:
        try:
            actual    = float(item.get("actualEarningResult") or 0)
            estimated = float(item.get("estimatedEarning") or 0)
            if estimated != 0:
                surprise_pct = (actual - estimated) / abs(estimated) * 100
            else:
                surprise_pct = 0.0
            results.append({
                "date":         item.get("date", ""),
                "actual_eps":   actual,
                "estimated_eps":estimated,
                "surprise_pct": round(surprise_pct, 2),
            })
        except Exception:
            continue
    return sorted(results, key=lambda x: x["date"], reverse=True)


def get_earnings_surprise_score(ticker: str, as_of_date: str = None) -> float:
    """
    Composite earnings surprise signal:
      - Recent surprise magnitude (most recent quarter)
      - Surprise consistency (fraction of last 4 quarters positive)
      - Surprise momentum (is recent surprise bigger than prior?)

    Returns a z-score-ready float in roughly [-3, +3].
    Uses as_of_date to avoid lookahead (only surprises before that date).
    """
    surprises = get_earnings_surprises(ticker, limit=8)
    if not surprises:
        return 0.0

    # Filter to only past data if as_of_date provided
    if as_of_date:
        surprises = [s for s in surprises if s["date"] <= str(as_of_date)]

    if not surprises:
        return 0.0

    recent = surprises[:4]  # last 4 quarters
    recent_pcts = [s["surprise_pct"] for s in recent]

    # Component 1: magnitude of most recent surprise (winsorized at ±50%)
    mag = float(np.clip(recent_pcts[0] if recent_pcts else 0, -50, 50)) / 50

    # Component 2: consistency (hit rate)
    hit_rate = float(np.mean([p > 0 for p in recent_pcts]))

    # Component 3: momentum (recent vs prior)
    if len(recent_pcts) >= 2:
        momentum = float(np.clip(recent_pcts[0] - recent_pcts[1], -30, 30)) / 30
    else:
        momentum = 0.0

    return round(0.5 * mag + 0.3 * (hit_rate - 0.5) * 2 + 0.2 * momentum, 4)


# ── 2. Analyst Estimate Revisions ─────────────────────────────────────────────

@simple_cache(ttl=3600 * 6, disk=True)
def get_analyst_revisions(ticker: str) -> dict:
    """
    Analyst estimate revisions over last 30 days.
    Up-revisions vs down-revisions = revision momentum.
    Returns {revision_score, upgrades, downgrades, net_revisions}.
    """
    data = _fmp_get(f"analyst-estimates/{ticker}", {"limit": 4})
    if not data or not isinstance(data, list):
        return {"revision_score": 0.0}

    # Count estimate changes: compare consecutive periods
    eps_ests = []
    for item in data[:4]:
        try:
            eps_ests.append(float(item.get("estimatedEpsAvg") or 0))
        except Exception:
            eps_ests.append(0.0)

    if len(eps_ests) < 2:
        return {"revision_score": 0.0}

    # Revision score: recent estimate vs older estimate (normalized)
    recent = eps_ests[0]
    older  = eps_ests[min(2, len(eps_ests)-1)]
    if older != 0:
        rev_score = float(np.clip((recent - older) / abs(older) * 10, -3, 3))
    else:
        rev_score = 0.0

    return {
        "revision_score":  round(rev_score, 4),
        "recent_est_eps":  recent,
        "prior_est_eps":   older,
    }


# ── 3. Institutional Ownership Changes ────────────────────────────────────────

@simple_cache(ttl=3600 * 24, disk=True)
def get_institutional_ownership(ticker: str) -> dict:
    """
    13F filing data: institutional ownership changes.
    Rising institutional ownership → smart money accumulation signal.
    Returns {ownership_change_score, total_holders, pct_held}.
    """
    data = _fmp_get(f"institutional-holder/{ticker}")
    if not data or not isinstance(data, list):
        return {"ownership_score": 0.0}

    try:
        total = len(data)
        buyers  = sum(1 for h in data if float(h.get("change") or 0) > 0)
        sellers = sum(1 for h in data if float(h.get("change") or 0) < 0)

        net = buyers - sellers
        score = float(np.clip(net / max(total, 1), -1, 1))

        shares_held = sum(float(h.get("shares") or 0) for h in data)
        return {
            "ownership_score": round(score, 4),
            "buyers":          buyers,
            "sellers":         sellers,
            "total_holders":   total,
            "shares_held":     shares_held,
        }
    except Exception:
        return {"ownership_score": 0.0}


# ── 4. Price Targets ──────────────────────────────────────────────────────────

@simple_cache(ttl=3600 * 6, disk=True)
def get_price_target_upside(ticker: str, current_price: float) -> float:
    """
    Consensus analyst price target vs current price.
    Positive = upside vs consensus, Negative = overvalued vs consensus.
    Returns float (e.g. 0.15 = 15% upside).
    """
    data = _fmp_get(f"price-target-consensus/{ticker}")
    if not data or not isinstance(data, (list, dict)):
        return 0.0

    if isinstance(data, list):
        data = data[0] if data else {}

    try:
        target = float(data.get("targetConsensus") or data.get("priceTarget") or 0)
        if target > 0 and current_price > 0:
            return round((target - current_price) / current_price, 4)
    except Exception:
        pass
    return 0.0


# ── 5. Composite FMP Signal ───────────────────────────────────────────────────

def get_fmp_alpha_signal(ticker: str, current_price: float = 0,
                         as_of_date: str = None) -> dict:
    """
    Master FMP signal: combines earnings surprise + analyst revisions
    + institutional ownership into a single alpha score.

    Returns dict with individual components and composite score.
    """
    if not FMP_API_KEY:
        return {"fmp_score": 0.0, "available": False}

    eps_score = get_earnings_surprise_score(ticker, as_of_date)
    rev_data  = get_analyst_revisions(ticker)
    inst_data = get_institutional_ownership(ticker)
    pt_upside = get_price_target_upside(ticker, current_price) if current_price > 0 else 0.0

    rev_score  = rev_data.get("revision_score", 0.0)
    inst_score = inst_data.get("ownership_score", 0.0)

    # Composite: earnings surprise drives most of alpha
    composite = (
        0.40 * float(np.clip(eps_score,  -2, 2)) +
        0.25 * float(np.clip(rev_score,  -2, 2)) +
        0.20 * float(np.clip(inst_score, -1, 1)) +
        0.15 * float(np.clip(pt_upside * 5, -2, 2))
    )

    return {
        "fmp_score":       round(composite, 4),
        "eps_surprise":    round(eps_score, 4),
        "revision_score":  round(rev_score, 4),
        "inst_score":      round(inst_score, 4),
        "pt_upside":       round(pt_upside, 4),
        "available":       True,
    }
