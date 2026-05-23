"""
OpenBB Free Fallback Router.

When yfinance rate-limits, this module tries OpenBB's free routing layer.
OpenBB aggregates multiple free providers (Yahoo Finance, FMP free, CBOE, etc.)
and handles rate limiting internally across providers.

Installation (one-time):
    pip install openbb

OpenBB free tier providers (no API key needed):
    - yfinance      (same source, different session = different rate limit)
    - intrinio      (delayed data, free tier)
    - cboe          (options + equity quotes)

Usage:
    from hedge_fund_ai.data.openbb_fallback import fetch_with_openbb
    close, volume = fetch_with_openbb("AAPL", period_days=252)

This module is imported lazily — if openbb is not installed, it returns None
gracefully and the caller falls back to cache or skips the ticker.
"""

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OBB_AVAILABLE: bool | None = None   # None = not yet checked


def _check_openbb() -> bool:
    """Lazy check — only import openbb once."""
    global _OBB_AVAILABLE
    if _OBB_AVAILABLE is not None:
        return _OBB_AVAILABLE
    try:
        import openbb  # noqa: F401
        _OBB_AVAILABLE = True
        logger.info("OpenBB available — will use as yfinance fallback")
    except ImportError:
        _OBB_AVAILABLE = False
        logger.info("OpenBB not installed — install with: pip install openbb")
    return _OBB_AVAILABLE


def fetch_with_openbb(
    ticker: str,
    period_days: int = 365,
) -> tuple[pd.Series | None, pd.Series | None]:
    """
    Fetch OHLCV data via OpenBB free router.

    Tries providers in order:
      1. yfinance  (different session from our main fetch)
      2. intrinio  (free delayed)
      3. cboe      (equity data)

    Returns (close_series, volume_series) or (None, None) on failure.
    """
    if not _check_openbb():
        return None, None

    try:
        from openbb import obb

        end_date   = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=period_days + 30)

        # Try each free provider in order
        providers = ["yfinance", "intrinio", "cboe"]
        for provider in providers:
            try:
                result = obb.equity.price.historical(
                    symbol     = ticker,
                    start_date = start_date.isoformat(),
                    end_date   = end_date.isoformat(),
                    provider   = provider,
                )

                if result is None or result.results is None:
                    continue

                df = result.to_df()
                if df is None or df.empty:
                    continue

                # Normalise column names across providers
                df.columns = [c.lower() for c in df.columns]
                if "close" not in df.columns:
                    continue

                close  = df["close"].dropna()
                volume = df["volume"].dropna() if "volume" in df.columns else None

                if len(close) < 30:
                    continue

                logger.info("OpenBB: %s fetched via %s (%d rows)", ticker, provider, len(close))
                return close, volume

            except Exception as e:
                logger.debug("OpenBB %s via %s: %s", ticker, provider, e)
                continue

        logger.warning("OpenBB: all providers failed for %s", ticker)
        return None, None

    except Exception as e:
        logger.warning("OpenBB fetch failed %s: %s", ticker, e)
        return None, None


def fetch_bulk_with_openbb(
    tickers: list[str],
    period_days: int = 365,
) -> dict[str, tuple[pd.Series, pd.Series | None]]:
    """
    Fetch multiple tickers via OpenBB.
    Returns {ticker: (close, volume)} for successful fetches only.
    """
    if not _check_openbb():
        return {}

    results = {}
    for t in tickers:
        close, volume = fetch_with_openbb(t, period_days)
        if close is not None:
            results[t] = (close, volume)

    logger.info("OpenBB bulk: %d/%d tickers fetched", len(results), len(tickers))
    return results


def is_available() -> bool:
    """Returns True if OpenBB is installed and can be used."""
    return _check_openbb()
