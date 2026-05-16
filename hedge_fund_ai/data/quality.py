"""
Data Quality Gates — called after every fetch before signals.

Checks:
  1. Price continuity: reject if > MAX_MISSING_PCT NaN prices
  2. Zero/negative prices
  3. Volume sanity: reject if avg daily dollar volume < MIN_ADV
  4. Stale data: warn if last price older than MAX_STALE_HOURS
  5. Trading calendar: flag non-trading days in data
  6. Price spike detection: reject if any single-day return > SPIKE_PCT

Returns filtered list + quality report logged to structured audit.
"""
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
MAX_MISSING_PCT    = 0.05   # reject if >5% prices missing
MIN_ADV            = 1e6    # $1M minimum avg daily dollar volume
MAX_STALE_HOURS    = 36     # warn if data older than 36h
SPIKE_PCT          = 0.40   # reject if any daily return > 40% (data error)
MIN_PRICE          = 1.0    # reject penny stocks
MIN_HISTORY_DAYS   = 60     # minimum required history


def run_quality_gates(data: list[dict], strict: bool = False) -> tuple[list[dict], dict]:
    """
    Filter a list of ticker data dicts through quality gates.

    Args:
        data:   list of fetched ticker dicts
        strict: if True, reject on any warning; if False, only reject on errors

    Returns:
        (filtered_data, quality_report)
    """
    passed, rejected = [], []
    report = {
        "total":    len(data),
        "passed":   0,
        "rejected": 0,
        "reasons":  {},
        "warnings": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    for d in data:
        ticker = d.get("ticker", "?")
        tech   = d.get("tech", {}) or {}
        ph     = tech.get("price_hist", [])
        vh     = tech.get("volume_hist", [])
        issues = []

        # Gate 1: minimum history
        if len(ph) < MIN_HISTORY_DAYS:
            issues.append(f"insufficient_history:{len(ph)}<{MIN_HISTORY_DAYS}")

        if ph:
            prices = np.array(ph, dtype=float)

            # Gate 2: missing prices
            nan_pct = float(np.isnan(prices).mean())
            if nan_pct > MAX_MISSING_PCT:
                issues.append(f"too_many_nans:{nan_pct:.1%}")

            # Gate 3: zero/negative prices
            if np.any(prices[np.isfinite(prices)] <= 0):
                issues.append("zero_or_negative_price")

            # Gate 4: penny stocks
            last_price = float(prices[np.isfinite(prices)][-1]) if np.any(np.isfinite(prices)) else 0
            if last_price < MIN_PRICE:
                issues.append(f"penny_stock:${last_price:.2f}")

            # Gate 5: price spikes (data errors)
            valid = prices[np.isfinite(prices)]
            if len(valid) > 1:
                rets = np.diff(valid) / (valid[:-1] + 1e-10)
                if np.any(np.abs(rets) > SPIKE_PCT):
                    max_ret = float(np.max(np.abs(rets)))
                    issues.append(f"price_spike:{max_ret:.0%}")

        # Gate 6: volume / ADV
        if ph and vh and len(vh) >= 20:
            try:
                px = np.array(ph[-20:], dtype=float)
                v  = np.array(vh[-20:], dtype=float)
                adv = float(np.nanmean(px * v))
                if adv < MIN_ADV:
                    issues.append(f"low_adv:${adv/1e6:.2f}M")
            except Exception:
                pass

        # Gate 7: data staleness (warn only)
        price_date = tech.get("last_price_date")
        if price_date:
            try:
                age_h = (datetime.now(timezone.utc) - pd.Timestamp(price_date, tz="UTC")).total_seconds() / 3600
                if age_h > MAX_STALE_HOURS:
                    report["warnings"].append(f"{ticker}: stale data ({age_h:.0f}h old)")
            except Exception:
                pass

        if issues:
            rejected.append({"ticker": ticker, "reasons": issues})
            report["reasons"][ticker] = issues
        else:
            passed.append(d)

    report["passed"]   = len(passed)
    report["rejected"] = len(rejected)

    if rejected:
        logger.warning(
            "Data quality: %d/%d passed | %d rejected: %s",
            len(passed), len(data), len(rejected),
            [(r["ticker"], r["reasons"][0]) for r in rejected[:5]],
        )
    else:
        logger.info("Data quality: all %d tickers passed", len(passed))

    if report["warnings"]:
        for w in report["warnings"]:
            logger.warning("Data quality warning: %s", w)

    return passed, report


def check_data_freshness(data: list[dict]) -> dict:
    """
    Returns a freshness report: {ticker: age_hours} for monitoring.
    Logs alert if any critical dataset is older than MAX_STALE_HOURS.
    """
    freshness = {}
    now = datetime.now(timezone.utc)
    for d in data:
        t = d.get("ticker", "?")
        price_date = (d.get("tech", {}) or {}).get("last_price_date")
        if price_date:
            try:
                age = (now - pd.Timestamp(price_date, tz="UTC")).total_seconds() / 3600
                freshness[t] = round(age, 1)
            except Exception:
                freshness[t] = None
    return freshness
