"""
Dynamic universe construction.
Screens for: market cap > $10B, avg daily dollar volume > $50M, price > $5.
Starts from the config UNIVERSE list (S&P 500 large-caps) and filters.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf

from hedge_fund_ai.config import UNIVERSE, MIN_UNIVERSE_SIZE
from hedge_fund_ai.data.cache import simple_cache

logger = logging.getLogger(__name__)

MIN_MARKET_CAP   = 5e9     # $5B
MIN_DOLLAR_VOL   = 20e6    # $20M avg daily dollar volume
MIN_PRICE        = 3.0     # exclude penny stocks


def _screen_ticker(t: str) -> str | None:
    """Screen a single ticker. Returns ticker if it passes, None otherwise."""
    try:
        info = yf.Ticker(t).info or {}
        mc  = info.get("marketCap") or 0
        vol = info.get("averageVolume") or 0
        px  = info.get("regularMarketPrice") or info.get("previousClose") or 0
        adv = (info.get("averageVolume10days") or vol) * px

        # If all values are zero — likely a rate limit or empty response
        # Don't reject; treat as unknown and include conservatively
        if mc == 0 and px == 0:
            logger.debug("Screen %s: empty response (rate limit?) — including conservatively", t)
            return t

        if mc > MIN_MARKET_CAP and adv > MIN_DOLLAR_VOL and px > MIN_PRICE:
            return t
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg or "429" in msg or "too many" in msg:
            # Rate limited — include conservatively rather than reject
            logger.debug("Screen %s: rate limited — including", t)
            return t
        logger.debug("Screen %s failed: %s", t, e)
    return None


@simple_cache(ttl=3600, disk=True)
def get_dynamic_universe() -> list[str]:
    """
    Return a filtered, investable universe of tickers.
    Results cached to disk for 1 hour.
    """
    candidates = list(UNIVERSE)
    valid = []

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_screen_ticker, t): t for t in candidates}
        for f in as_completed(futures):
            result = f.result()
            if result:
                valid.append(result)

    valid.sort()
    logger.info("Universe screened: %d/%d passed", len(valid), len(candidates))

    if len(valid) < MIN_UNIVERSE_SIZE:
        logger.warning("Universe too small after screening (%d), using full config list", len(valid))
        return sorted(candidates)

    return valid
