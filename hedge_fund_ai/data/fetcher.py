"""
Data fetcher — bulk download version.

Key change: replaces per-ticker yf.Ticker().history() (57 separate HTTP requests)
with a single yf.download(all_tickers) call. This is dramatically more
rate-limit friendly — one request instead of 57.

Falls back to sequential per-ticker fetching only if bulk download fails.
"""
import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from hedge_fund_ai.data.cache import simple_cache
from hedge_fund_ai.data.retry import retry

logger = logging.getLogger(__name__)

# Suppress cosmetic warnings that don't affect functionality
import urllib3
import warnings
urllib3.disable_warnings()  # suppresses InsecureRequest + version mismatch
warnings.filterwarnings("ignore", category=UserWarning, module="requests")
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*charset_normalizer.*")
try:
    from urllib3.connectionpool import log as _pool_log
    _pool_log.setLevel(logging.ERROR)
except Exception:
    pass

# Trading calendar (NYSE)
_calendar = None

def _get_calendar():
    global _calendar
    if _calendar is None:
        try:
            import pandas_market_calendars as mcal
            _calendar = mcal.get_calendar("NYSE")
        except Exception:
            _calendar = False
    return _calendar


def _is_trading_day(date: pd.Timestamp) -> bool:
    cal = _get_calendar()
    if cal is False:
        return date.dayofweek < 5
    try:
        schedule = cal.schedule(
            start_date=date.strftime("%Y-%m-%d"),
            end_date=date.strftime("%Y-%m-%d")
        )
        return not schedule.empty
    except Exception:
        return date.dayofweek < 5


_spy_returns: pd.Series | None = None

def _get_spy_returns() -> pd.Series:
    global _spy_returns
    if _spy_returns is None:
        try:
            spy = yf.download("SPY", period="1y", progress=False, auto_adjust=True)["Close"]
            if isinstance(spy, pd.DataFrame):
                spy = spy.iloc[:, 0]
            _spy_returns = spy.pct_change().dropna()
        except Exception:
            _spy_returns = pd.Series(dtype=float)
    return _spy_returns


def _compute_beta(close: pd.Series) -> float:
    try:
        spy = _get_spy_returns()
        ret = close.pct_change().dropna()
        common = ret.index.intersection(spy.index)[-60:]
        if len(common) < 20:
            return 1.0
        r, s = ret.loc[common].values, spy.loc[common].values
        return float(np.cov(r, s)[0, 1] / (np.var(s) + 1e-10))
    except Exception:
        return 1.0


def _to_float(x, default=0.0):
    try:
        if hasattr(x, "iloc"):
            x = x.iloc[-1]
        return float(x) if x is not None else default
    except Exception:
        return default


def _build_ticker_dict(t: str, close: pd.Series, volume: pd.Series | None,
                        info: dict) -> dict | None:
    """Build the data dict for one ticker from pre-fetched close/volume/info."""
    if close is None or len(close) < 30:
        return None

    price = _to_float(close.iloc[-1])
    if price <= 0:
        return None

    # Returns
    n = len(close)
    ret_1m = (_to_float(close.iloc[-1]) / _to_float(close.iloc[-22]) - 1) * 100 if n > 22 else 0.0
    ret_3m = (_to_float(close.iloc[-1]) / _to_float(close.iloc[-63]) - 1) * 100 if n > 63 else 0.0
    ret_6m = (_to_float(close.iloc[-1]) / _to_float(close.iloc[-126])- 1) * 100 if n > 126 else 0.0

    ret_12m_skip1m = 0.0
    if n >= 252:
        ret_12m_skip1m = (_to_float(close.iloc[-22]) / _to_float(close.iloc[-252]) - 1) * 100
    elif n >= 100:
        ret_12m_skip1m = (_to_float(close.iloc[-22]) / _to_float(close.iloc[0])   - 1) * 100

    # Technicals
    pct_chg     = close.pct_change()
    sma50       = float(close.rolling(50).mean().iloc[-1]) if n >= 50 else price
    sma200      = float(close.rolling(200).mean().iloc[-1]) if n >= 200 else price
    ann_vol_30d = float(pct_chg.rolling(30).std().iloc[-1] * np.sqrt(252) * 100) if n >= 31 else float(pct_chg.std() * np.sqrt(252) * 100)
    ann_vol_90d = float(pct_chg.rolling(90).std().iloc[-1] * np.sqrt(252) * 100) if n >= 91 else ann_vol_30d
    high_52w    = float(close.tail(252).max()) if n >= 50 else price
    high_52w_proximity = price / high_52w if high_52w > 0 else 0.5
    vol_regime  = ann_vol_30d / (ann_vol_90d + 1e-6)
    z_score     = (price - sma50) / (ann_vol_30d / 100 * sma50 + 1e-6)
    beta        = _compute_beta(close)
    last_date   = str(close.index[-1].date()) if n > 0 else None

    # Volume / liquidity
    vol_hist    = volume.tail(260).values.astype(float) if volume is not None and len(volume) > 0 else None
    liquidity   = 0.0
    flow_signal = 0.0

    if vol_hist is not None and len(vol_hist) >= 21:
        px = close.tail(len(vol_hist)).values.astype(float)
        ret_abs = np.abs(np.diff(px[-21:]) / (px[-21:-1] + 1e-10))
        dv = (px[-20:] * vol_hist[-20:]) + 1e-6
        amihud = float(np.mean(ret_abs / dv))
        liquidity = float(-np.log1p(amihud * 1e6))
        sign = np.sign(np.diff(px[-21:]))
        obv  = np.cumsum(sign * vol_hist[-20:])
        obv_std = np.std(obv) + 1e-6
        flow_signal = float((obv[-1] - obv[0]) / (obv_std * len(obv)))

    # Fundamentals from info
    roe_pct         = (info.get("returnOnEquity")  or 0) * 100
    op_margin_pct   = (info.get("operatingMargins") or 0) * 100
    net_margin_pct  = (info.get("profitMargins")    or 0) * 100
    revenue_growth  = (info.get("revenueGrowth")    or 0) * 100
    earnings_growth = (info.get("earningsGrowth")   or 0) * 100
    total_rev = info.get("totalRevenue") or 0
    fcf       = info.get("freeCashflow") or 0
    fcf_margin_pct  = (fcf / total_rev * 100) if total_rev else 0.0
    pe_ratio  = info.get("trailingPE")
    ps_ratio  = info.get("priceToSalesTrailing12Months")
    de_ratio  = info.get("debtToEquity") or 0
    quality   = 0.4 * roe_pct + 0.3 * op_margin_pct + 0.3 * fcf_margin_pct
    growth    = 0.6 * revenue_growth + 0.4 * earnings_growth
    value     = None
    if pe_ratio is not None or ps_ratio is not None:
        value = -(0.5 * (pe_ratio or 0) + 0.5 * (ps_ratio or 0))

    # PEAD proxy — better quality filter vs raw price change:
    # Only trigger if earnings-window return > 3% AND volume > 1.5x 20-day avg
    # This reduces false signals from normal daily moves
    earnings_surprise = 0.0
    if n >= 25:
        recent_ret = _to_float(close.iloc[-1]) / _to_float(close.iloc[-6]) - 1.0
        prior_vol  = float(pct_chg.iloc[-26:-6].std()) + 1e-6
        raw_signal = float(recent_ret / prior_vol)

        # Quality filter: require meaningful move + volume confirmation
        vol_confirm = False
        if vol_hist is not None and len(vol_hist) >= 20:
            avg_vol_20d = float(np.mean(vol_hist[-20:]))
            last_vol    = float(vol_hist[-1]) if len(vol_hist) > 0 else 0
            vol_confirm = last_vol > 1.5 * avg_vol_20d

        # Only use signal if: return > 3% AND volume confirms
        if abs(recent_ret) >= 0.03 and vol_confirm:
            earnings_surprise = raw_signal
        elif abs(recent_ret) >= 0.05:
            # Very large move (>5%) — accept even without volume confirmation
            earnings_surprise = raw_signal * 0.5  # half weight without volume

    return {
        "ticker": t,
        "tech": {
            "price_hist":         close.tail(260).values.tolist(),
            "volume_hist":        vol_hist.tolist() if vol_hist is not None else [],
            "ret_1m":             ret_1m,
            "ret_3m":             ret_3m,
            "ret_6m":             ret_6m,
            "ret_12m_skip1m":     ret_12m_skip1m,
            "sma50":              sma50,
            "sma200":             sma200,
            "z_score":            z_score,
            "ann_vol_30d":        ann_vol_30d,
            "ann_vol_90d":        ann_vol_90d,
            "vol_regime":         vol_regime,
            "high_52w_proximity": high_52w_proximity,
            "beta":               beta,
            "last_price_date":    last_date,
        },
        "fund": {
            "roe_pct":          roe_pct,
            "op_margin_pct":    op_margin_pct,
            "net_margin_pct":   net_margin_pct,
            "fcf_margin_pct":   fcf_margin_pct,
            "quality_score":    quality,
            "pe_ratio":         pe_ratio,
            "ps_ratio":         ps_ratio,
            "value_score":      value,
            "de_ratio":         de_ratio,
            "revenue_growth":   revenue_growth,
            "earnings_growth":  earnings_growth,
            "growth_score":     growth,
            "market_cap":       info.get("marketCap"),
            "total_assets":     info.get("totalAssets") or 0,
            "roic_pct":         0,
        },
        "sector":            info.get("sector", "Unknown"),
        "industry":          info.get("industry", "Unknown"),
        "beta":              beta,
        "liquidity":         liquidity,
        "flow_signal":       flow_signal,
        "earnings_surprise": earnings_surprise,
        "sentiment_score":   0.0,
        "news_volume":       0,
    }


@simple_cache(ttl=3600, disk=True)
def _fetch_info_single(t: str) -> dict:
    """Fetch .info for a single ticker (fundamentals only). Lightweight."""
    try:
        return yf.Ticker(t).info or {}
    except Exception:
        return {}


def fetch_universe(tickers: list[str]) -> list[dict]:
    """
    Fetch all tickers using a SINGLE bulk yf.download() call.
    This is the primary rate-limit fix: 1 request instead of N.

    Falls back to sequential per-ticker fetching if bulk fails.
    """
    from hedge_fund_ai.data.quality import run_quality_gates
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tickers = sorted(set(t.strip() for t in tickers if t.strip()))
    if not tickers:
        return []

    t0 = time.time()
    results = []

    # ── Primary path: single bulk download ────────────────────────────────────
    logger.info("Bulk downloading %d tickers (single request)...", len(tickers))
    try:
        raw = yf.download(
            tickers,
            period    = "1y",
            progress  = False,
            auto_adjust = True,
            group_by  = "ticker",
            threads   = False,    # prevents urllib3 pool exhaustion
        )

        if raw is None or raw.empty:
            raise ValueError("Empty bulk download")

        # Parse per-ticker slices — handle all yfinance MultiIndex layouts:
        # Layout A (old): columns = (field, ticker) → raw["Close"] gives ticker-columns
        # Layout B (new): columns = (ticker, field) → raw[ticker]["Close"] gives series
        # Layout C: single ticker flat columns
        close_all  = None
        volume_all = None

        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = raw.columns.get_level_values(0).unique().tolist()
            lvl1 = raw.columns.get_level_values(1).unique().tolist()

            if "Close" in lvl0:
                # Layout A: first level = field names
                close_all  = raw["Close"]
                volume_all = raw["Volume"] if "Volume" in lvl0 else None
            elif "Close" in lvl1:
                # Layout B: first level = ticker names
                close_cols  = {}
                volume_cols = {}
                for t in lvl0:
                    try:
                        if "Close" in raw[t].columns:
                            close_cols[t]  = raw[t]["Close"]
                        if "Volume" in raw[t].columns:
                            volume_cols[t] = raw[t]["Volume"]
                    except Exception:
                        pass
                if close_cols:
                    close_all  = pd.DataFrame(close_cols)
                    volume_all = pd.DataFrame(volume_cols) if volume_cols else None
            else:
                raise ValueError(f"Cannot parse MultiIndex: lvl0={lvl0[:3]} lvl1={lvl1[:3]}")
        else:
            # Single ticker — flat columns
            if "Close" in raw.columns:
                close_all  = raw[["Close"]].rename(columns={"Close": tickers[0]})
            if "Volume" in raw.columns:
                volume_all = raw[["Volume"]].rename(columns={"Volume": tickers[0]})

        if close_all is None or (hasattr(close_all, "empty") and close_all.empty):
            raise ValueError("No Close column in bulk download")

        logger.info("Bulk download: %d rows x %d tickers in %.1fs",
                    len(close_all), len(close_all.columns), time.time() - t0)

        # Fetch info for all tickers concurrently (lightweight — no history)
        logger.info("Fetching fundamentals for %d tickers...", len(tickers))
        info_map = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {ex.submit(_fetch_info_single, t): t for t in tickers}
            for f in as_completed(futures):
                t = futures[f]
                try:
                    info_map[t] = f.result()
                    time.sleep(0.1)  # small delay per info call
                except Exception:
                    info_map[t] = {}

        # Build result dicts
        for t in tickers:
            if t not in close_all.columns:
                logger.debug("Ticker %s not in bulk download result", t)
                continue
            close  = close_all[t].dropna()
            volume = volume_all[t].dropna() if volume_all is not None and t in volume_all.columns else None
            info   = info_map.get(t, {})

            d = _build_ticker_dict(t, close, volume, info)
            if d:
                results.append(d)

        logger.info("Built %d/%d ticker dicts from bulk download", len(results), len(tickers))

    except Exception as e:
        logger.warning("Bulk download failed: %s — trying sequential fallback", e)
        results = _fetch_sequential_fallback(tickers)

        # If sequential also got nothing, try OpenBB as 3rd tier
        if not results:
            logger.info("Sequential fallback empty — trying OpenBB free router...")
            try:
                from hedge_fund_ai.data.openbb_fallback import fetch_bulk_with_openbb, is_available
                if is_available():
                    obb_data = fetch_bulk_with_openbb(tickers)
                    for t, (close, volume) in obb_data.items():
                        info = _fetch_info_single(t)
                        d    = _build_ticker_dict(t, close, volume, info or {})
                        if d:
                            results.append(d)
                    if results:
                        logger.info("OpenBB fallback: recovered %d tickers", len(results))
                else:
                    logger.info("OpenBB not installed — skipping (pip install openbb to enable)")
            except Exception as obb_e:
                logger.debug("OpenBB fallback error: %s", obb_e)

    elapsed = time.time() - t0
    logger.info("fetch_universe: %d/%d tickers in %.1fs", len(results), len(tickers), elapsed)

    if not results:
        logger.warning("Live fetch returned 0 tickers — attempting cache fallback...")
        from hedge_fund_ai.data.price_cache import get_cache
        cache     = get_cache()
        cache_map = cache.get_bulk(tickers)
        if cache_map:
            logger.info("Cache fallback: loaded %d/%d tickers from yesterday's data",
                        len(cache_map), len(tickers))
            for t, (close, volume) in cache_map.items():
                try:
                    info = _fetch_info_single(t)
                    d    = _build_ticker_dict(t, close, volume, info or {})
                    if d:
                        results.append(d)
                except Exception:
                    pass
            if results:
                logger.info("Cache fallback succeeded: %d tickers available", len(results))
            else:
                logger.warning(
                    "Cache fallback also failed. Wait 2-3 minutes and retry. "
                    "Backtest will still run using prior state."
                )
        else:
            logger.warning(
                "No cache available. First run? Wait 2-3 minutes for rate limit to clear, "
                "then retry. Backtest will run on prior data."
            )
        if not results:
            return []

    results, qreport = run_quality_gates(results, strict=False)
    logger.info("Quality gates: %d/%d passed", qreport["passed"], qreport["total"])
    results.sort(key=lambda d: d["ticker"])
    return results


def _fetch_sequential_fallback(tickers: list[str]) -> list[dict]:
    """
    Sequential per-ticker fallback when bulk download fails.
    Uses longer delays and exponential backoff on rate limits.
    Also checks price cache first — if today's data is cached, skip fetch entirely.
    """
    from hedge_fund_ai.data.price_cache import get_cache
    cache   = get_cache()
    results = []

    for i, t in enumerate(tickers):
        # Check cache first — avoids any API call for cached tickers
        close_cached, vol_cached = cache.get(t)
        if close_cached is not None:
            try:
                info = _fetch_info_single(t)
                d    = _build_ticker_dict(t, close_cached, vol_cached, info)
                if d:
                    results.append(d)
                    logger.debug("Sequential: %s loaded from cache", t)
                    continue
            except Exception:
                pass

        # Not cached — fetch with throttled delay
        for attempt in range(3):
            try:
                ticker = yf.Ticker(t)
                hist   = ticker.history(period="1y", auto_adjust=True)
                if hist is None or hist.empty:
                    break
                close  = hist["Close"]
                volume = hist.get("Volume")
                info   = ticker.info or {}
                d      = _build_ticker_dict(t, close, volume, info)
                if d:
                    results.append(d)
                    # Save to cache for next run
                    cache.put(t, close, volume)
                break
            except Exception as e:
                msg = str(e).lower()
                if "rate" in msg or "429" in msg or "too many" in msg:
                    wait = 15 * (attempt + 1)   # 15s, 30s, 45s
                    logger.warning("Rate limit on %s (attempt %d) — waiting %ds", t, attempt+1, wait)
                    time.sleep(wait)
                else:
                    logger.warning("Sequential fetch failed %s: %s", t, e)
                    break

        # Throttle between tickers regardless
        delay = 2.0 if (i + 1) % 5 == 0 else 0.8
        time.sleep(delay)

    logger.info("Sequential fallback: %d/%d tickers", len(results), len(tickers))
    return results


@simple_cache(ttl=600, disk=True)
def fetch_macro() -> dict:
    """
    Fetch macro indicators via ONE bulk yf.download() call.
    This avoids triggering rate limits before equity fetch even starts.
    All 6 symbols fetched in a single HTTP request.
    """
    macro = {
        "vix": 20.0, "spy_3m": 0.0,
        "credit_spread_signal": 0.0, "yield_curve": 0.0,
        "vol_regime": 1.0, "regime": "neutral",
    }

    MACRO_SYMBOLS = ["^VIX", "SPY", "HYG", "IEF", "TLT", "SHY"]

    for attempt in range(3):
        try:
            raw = yf.download(
                MACRO_SYMBOLS, period="9mo", progress=False,
                auto_adjust=True, group_by="ticker", threads=False,
            )
            if raw is None or raw.empty:
                raise ValueError("empty macro download")

            def _get(sym):
                """Extract Close series for a symbol from bulk result."""
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        lvl0 = raw.columns.get_level_values(0).unique().tolist()
                        lvl1 = raw.columns.get_level_values(1).unique().tolist()
                        if sym in lvl0 and "Close" in lvl1:
                            return raw[sym]["Close"].dropna()
                        elif "Close" in lvl0 and sym in lvl1:
                            return raw["Close"][sym].dropna()
                    elif sym in raw.columns:
                        return raw[sym].dropna()
                except Exception:
                    pass
                return None

            # VIX + 5-day change for leading indicator
            vix_s = _get("^VIX")
            if vix_s is not None and len(vix_s):
                macro["vix"] = float(vix_s.iloc[-1])
                if len(vix_s) >= 5:
                    vix_5d_ago = float(vix_s.iloc[-5])
                    macro["vix_5d_change"] = float(
                        (vix_s.iloc[-1] - vix_5d_ago) / (vix_5d_ago + 1e-6) * 100
                    )

            # SPY — momentum + vol regime
            spy_s = _get("SPY")
            if spy_s is not None and len(spy_s) > 63:
                macro["spy_3m"] = float((spy_s.iloc[-1] / spy_s.iloc[-63] - 1) * 100)
                r   = spy_s.pct_change().dropna()
                v30 = float(r.tail(30).std() * np.sqrt(252))
                v90 = float(r.tail(90).std() * np.sqrt(252))
                macro["vol_regime"] = float(v30 / (v90 + 1e-6))

            # Credit spread: HYG vs IEF
            hyg_s = _get("HYG"); ief_s = _get("IEF")
            if hyg_s is not None and ief_s is not None and len(hyg_s) > 21 and len(ief_s) > 21:
                macro["credit_spread_signal"] = float(
                    hyg_s.iloc[-1] / hyg_s.iloc[-22] - ief_s.iloc[-1] / ief_s.iloc[-22]
                )

            # Yield curve: TLT vs SHY
            tlt_s = _get("TLT"); shy_s = _get("SHY")
            if tlt_s is not None and shy_s is not None and len(tlt_s) > 21 and len(shy_s) > 21:
                macro["yield_curve"] = float(
                    tlt_s.iloc[-1] / tlt_s.iloc[-22] - shy_s.iloc[-1] / shy_s.iloc[-22]
                )

            logger.info("Macro: VIX=%.1f SPY3m=%.1f%% credit=%.4f yc=%.4f vol_regime=%.2f",
                        macro["vix"], macro["spy_3m"],
                        macro["credit_spread_signal"], macro["yield_curve"],
                        macro["vol_regime"])
            break  # success

        except Exception as e:
            wait = 10 * (attempt + 1)
            if attempt < 2:
                logger.warning("Macro fetch attempt %d failed: %s — retrying in %ds", attempt+1, e, wait)
                time.sleep(wait)
            else:
                logger.warning("Macro fetch failed after 3 attempts: %s — using defaults", e)

    risk_on    = macro["spy_3m"] > 0 and macro["vix"] < 20
    credit_pos = macro["credit_spread_signal"] > 0
    macro["macro_score"] = (1 if risk_on else -1) + (1 if credit_pos else -1)
    macro["regime"]      = "risk_on" if (risk_on and credit_pos) else ("risk_off" if not risk_on else "neutral")
    return macro
