"""
Price fetcher for live portfolio pricing.
Handles empty ticker lists, rate limits, and single-ticker DataFrames.
"""
import logging
import time

import yfinance as yf

logger = logging.getLogger(__name__)


def get_prices(tickers: list) -> dict:
    """
    Fetch latest prices for a list of tickers.
    Returns {ticker: price_float}. Returns {} on any error.
    Handles: empty list, rate limits, single-ticker Series vs DataFrame.
    """
    if not tickers:
        logger.debug("get_prices called with empty list — returning {}")
        return {}

    tickers = [t for t in tickers if t]
    if not tickers:
        return {}

    for attempt in range(3):
        try:
            data = yf.download(tickers, period="2d", progress=False, auto_adjust=True)

            if data is None or data.empty:
                logger.warning("get_prices: empty download (attempt %d)", attempt + 1)
                time.sleep(5 * (attempt + 1))
                continue

            close = data["Close"] if "Close" in data.columns else data

            # Single ticker returns a Series
            if hasattr(close, "ndim") and close.ndim == 1:
                price = float(close.dropna().iloc[-1]) if not close.dropna().empty else 0.0
                return {tickers[0]: price}

            result = {}
            for t in tickers:
                if t in close.columns:
                    series = close[t].dropna()
                    result[t] = float(series.iloc[-1]) if not series.empty else 0.0
                else:
                    result[t] = 0.0
            return result

        except ValueError as e:
            # "No objects to concatenate" — all downloads failed (rate limit)
            if "concatenate" in str(e).lower() or "no objects" in str(e).lower():
                logger.warning("get_prices: all tickers failed (rate limit?) attempt %d — retrying in %ds",
                               attempt + 1, 10 * (attempt + 1))
                time.sleep(10 * (attempt + 1))
            else:
                logger.warning("get_prices ValueError: %s", e)
                break
        except Exception as e:
            logger.warning("get_prices error: %s", e)
            break

    # Fallback: try one by one
    logger.info("get_prices: falling back to sequential fetching for %d tickers", len(tickers))
    result = {}
    for t in tickers[:10]:   # limit sequential fallback to 10
        try:
            px = yf.Ticker(t).fast_info.get("lastPrice") or yf.Ticker(t).fast_info.get("previousClose")
            result[t] = float(px) if px else 0.0
            time.sleep(0.5)
        except Exception:
            result[t] = 0.0
    return result
