"""
Market Microstructure Awareness.

Builds a high-impact event calendar and provides:
  1. Blackout windows: restrict/scale-down execution around events
  2. Liquidity tier classification: ADV-based position caps
  3. VWAP/TWAP execution scheduling
  4. Smart order routing: avoid open/close auction extremes
"""
import logging
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── High-impact event calendar ────────────────────────────────────────────────
# These dates are when execution quality degrades significantly.
# Sources: Fed calendar, BLS CPI schedule, CBOE OpEx, S&P rebalance dates.

# FOMC meeting dates 2024-2025 (approximate — update quarterly)
_FOMC_DATES = {
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
}

# Monthly options expiration (3rd Friday of each month)
def _third_friday(year: int, month: int) -> str:
    d = date(year, month, 1)
    friday_count = 0
    while True:
        if d.weekday() == 4:  # Friday
            friday_count += 1
            if friday_count == 3:
                return d.isoformat()
        d += timedelta(days=1)

_OPEX_DATES = {_third_friday(y, m) for y in range(2022, 2027) for m in range(1, 13)}

# Quarterly options expiration (triple witching: March, June, Sept, Dec 3rd Fridays)
_TRIPLE_WITCHING = {_third_friday(y, m) for y in range(2022, 2027) for m in [3, 6, 9, 12]}

# Index rebalance windows (Russell, S&P — typically last Fridays of March/June/Sept/Dec)
_INDEX_REBALANCE = set()
for _y in range(2022, 2027):
    for _m in [3, 6, 9, 12]:
        _d = date(_y, _m, 28)
        while _d.weekday() != 4:
            _d -= timedelta(days=1)
        _INDEX_REBALANCE.add(_d.isoformat())


def is_blackout_window(
    check_date,
    blackout_types: set | None = None,
) -> dict:
    """
    Check if a date falls in any high-impact trading window.

    Returns:
        {
          "blackout":  bool,
          "reasons":   list of triggered event types,
          "scale":     float — suggested execution scale (0 = halt, 1 = normal)
        }
    """
    if blackout_types is None:
        blackout_types = {"fomc", "triple_witching", "index_rebalance"}

    date_str = pd.Timestamp(check_date).date().isoformat()
    reasons  = []
    scale    = 1.0

    if "fomc" in blackout_types and date_str in _FOMC_DATES:
        reasons.append("fomc_meeting")
        scale = min(scale, 0.50)   # halve position on FOMC day

    if "triple_witching" in blackout_types and date_str in _TRIPLE_WITCHING:
        reasons.append("triple_witching")
        scale = min(scale, 0.70)

    if "index_rebalance" in blackout_types and date_str in _INDEX_REBALANCE:
        reasons.append("index_rebalance")
        scale = min(scale, 0.75)

    if "opex" in blackout_types and date_str in _OPEX_DATES:
        reasons.append("options_expiration")
        scale = min(scale, 0.85)

    return {
        "blackout": len(reasons) > 0,
        "reasons":  reasons,
        "scale":    round(scale, 2),
    }


def get_upcoming_blackouts(days_ahead: int = 10) -> list[dict]:
    """Return all blackout dates in the next N days."""
    today = datetime.now().date()
    results = []
    for i in range(days_ahead):
        d = today + timedelta(days=i)
        check = is_blackout_window(d)
        if check["blackout"]:
            results.append({"date": d.isoformat(), **check})
    return results


# ── Liquidity tier classification ─────────────────────────────────────────────

def classify_liquidity_tier(avg_daily_dollar_vol: float) -> dict:
    """
    Classify a stock into a liquidity tier based on 20-day ADV.

    Tier 1 (mega-liquid): >$500M ADV  → max position 20%, turnover unrestricted
    Tier 2 (liquid):      $50–500M    → max position 15%, moderate turnover
    Tier 3 (mid):         $10–50M     → max position 10%, conservative turnover
    Tier 4 (illiquid):    <$10M       → max position 5%, strict turnover

    Returns:
        {tier, max_weight, max_turnover_per_rebalance, participation_rate}
    """
    adv = float(avg_daily_dollar_vol)

    if adv >= 500_000_000:
        return {"tier": 1, "max_weight": 0.20, "max_turnover": 0.10, "participation_rate": 0.10, "label": "mega"}
    elif adv >= 50_000_000:
        return {"tier": 2, "max_weight": 0.15, "max_turnover": 0.07, "participation_rate": 0.07, "label": "liquid"}
    elif adv >= 10_000_000:
        return {"tier": 3, "max_weight": 0.10, "max_turnover": 0.05, "participation_rate": 0.05, "label": "mid"}
    else:
        return {"tier": 4, "max_weight": 0.05, "max_turnover": 0.02, "participation_rate": 0.02, "label": "illiquid"}


def get_tier_caps(adv_map: dict) -> dict:
    """
    Return per-ticker position caps based on liquidity tier.
    adv_map: {ticker: avg_daily_dollar_vol}
    Returns: {ticker: max_weight}
    """
    caps = {}
    for ticker, adv in adv_map.items():
        tier_info = classify_liquidity_tier(adv)
        caps[ticker] = tier_info["max_weight"]
    return caps


# ── VWAP/TWAP execution scheduling ───────────────────────────────────────────

def build_vwap_schedule(
    ticker: str,
    target_notional: float,
    adv: float,
    execution_window_minutes: int = 390,  # full day default
    participation_rate: float | None = None,
) -> list[dict]:
    """
    Build a VWAP-style execution schedule.

    Splits the order across time buckets proportional to intraday volume
    distribution (approximated from standard NYSE volume profile).

    Returns list of {time_offset_minutes, notional, participation_pct}.
    """
    if target_notional <= 0:
        return []

    tier = classify_liquidity_tier(adv)
    if participation_rate is None:
        participation_rate = tier["participation_rate"]

    # Standard NYSE intraday volume profile (% of daily volume per 30-min bucket)
    # Based on empirical averages — higher at open/close
    volume_profile = [
        0.08, 0.06, 0.05, 0.05, 0.05, 0.05,   # 9:30-12:00
        0.05, 0.05, 0.05, 0.06, 0.07, 0.10,   # 12:00-15:00
        0.12, 0.11,                              # 15:00-16:00 (closing auction)
    ]
    # Normalize
    total = sum(volume_profile)
    volume_profile = [v / total for v in volume_profile]

    # Max single-bucket notional based on participation rate
    max_per_bucket = adv * participation_rate / (390 / 30)

    schedule = []
    remaining = target_notional

    for i, vol_frac in enumerate(volume_profile):
        if remaining <= 0:
            break
        # Skip first and last buckets (open/close — higher slippage)
        if i == 0 or i == len(volume_profile) - 1:
            bucket_notional = 0  # avoid open/close auctions
        else:
            target_bucket = target_notional * vol_frac
            bucket_notional = min(target_bucket, max_per_bucket, remaining)

        if bucket_notional >= 100:  # $100 minimum
            schedule.append({
                "time_offset_minutes": i * 30 + 15,  # center of bucket
                "notional":            round(bucket_notional, 2),
                "participation_pct":   round(bucket_notional / max(adv / (390/30), 1) * 100, 2),
            })
            remaining -= bucket_notional

    # If anything left (due to open/close exclusion), add to mid-day
    if remaining >= 100:
        schedule.append({
            "time_offset_minutes": 195,  # ~1pm
            "notional":            round(remaining, 2),
            "participation_pct":   round(remaining / max(adv / 13, 1) * 100, 2),
        })

    total_scheduled = sum(s["notional"] for s in schedule)
    logger.debug("VWAP schedule %s: $%.0f in %d buckets (of $%.0f target)",
                 ticker, total_scheduled, len(schedule), target_notional)

    return schedule


def avoid_auction_windows(order_time: datetime) -> bool:
    """
    Returns True if the current time is too close to open/close auction.
    Opening auction: 9:30-9:40 ET
    Closing auction: 15:50-16:00 ET
    """
    import pytz
    et = pytz.timezone("America/New_York")
    local = order_time.astimezone(et) if order_time.tzinfo else order_time
    h, m = local.hour, local.minute
    total_min = h * 60 + m

    open_start  = 9 * 60 + 30
    open_end    = 9 * 60 + 40
    close_start = 15 * 60 + 50
    close_end   = 16 * 60

    return (open_start <= total_min <= open_end) or (close_start <= total_min <= close_end)
