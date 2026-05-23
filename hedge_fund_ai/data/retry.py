"""
Retry decorator with exponential backoff.
Handles yfinance rate limits (429) with longer delays.
"""
import logging
import time
import functools

logger = logging.getLogger(__name__)

# yfinance rate limit detection strings
_RATE_LIMIT_SIGNALS = [
    "too many requests", "rate limit", "429", "ratelimit"
]


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in _RATE_LIMIT_SIGNALS)


def retry(max_attempts: int = 3, delay: float = 1.0, rate_limit_delay: float = 15.0):
    """
    Retry decorator with exponential backoff.

    On normal errors: delay * (attempt + 1)
    On rate-limit errors (429): rate_limit_delay seconds flat
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == max_attempts - 1:
                        raise
                    if _is_rate_limit(e):
                        wait = rate_limit_delay
                        logger.debug("Rate limit hit — waiting %.0fs before retry %d/%d",
                                     wait, i + 2, max_attempts)
                    else:
                        wait = delay * (i + 1)
                    time.sleep(wait)
        return wrapper
    return decorator
