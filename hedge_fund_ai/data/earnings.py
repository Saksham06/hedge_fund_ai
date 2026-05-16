import numpy as np

from hedge_fund_ai.data.cache import simple_cache
from hedge_fund_ai.data.retry import retry


def earnings_surprise_proxy_fast(price_hist):

    # proxy: abnormal return vs recent trend
    if len(price_hist) < 30:
        return 0

    recent = np.array(price_hist[-5:], dtype=float)
    prev = np.array(price_hist[-25:-5], dtype=float)

    if len(prev) == 0:
        return 0

    surprise = (recent.mean() - prev.mean()) / (np.std(prev) + 1e-6)

    return round(float(surprise), 3)


@simple_cache(ttl=3600)
@retry(max_attempts=3)
def earnings_surprise_yf(ticker):
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).info or {}
        eg = info.get("earningsGrowth") or 0
        rg = info.get("revenueGrowth") or 0
        return (eg - rg) * 100
    except Exception:
        return 0
