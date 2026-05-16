"""
Point-in-time fundamental data — unified interface.

Priority order (best to worst data quality):
  1. SimFin   — 10+ years, point-in-time, free, survivorship-bias-free
  2. FMP      — real earnings surprises, revisions, institutional flows
  3. yfinance — fallback for current fundamentals only

All lookups enforce strict point-in-time: only data published before
the query date is returned. Filing lag = 60 days (conservative).
"""
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from hedge_fund_ai.data.cache import simple_cache

logger = logging.getLogger(__name__)

FILING_LAG = 60  # days after period end before data is "available"


# ── Unified FundamentalStore (SimFin + yfinance fallback) ─────────────────────

class FundamentalStore:
    """
    Unified point-in-time fundamental lookup.
    Uses SimFin when available, falls back to yfinance quarterly statements.
    """

    def __init__(self, tickers: list[str], filing_lag_days: int = FILING_LAG):
        self.tickers     = list(tickers)
        self.filing_lag  = filing_lag_days
        self._yf_data:   dict[str, pd.DataFrame] = {}
        self._simfin:    object = None   # SimFinStore, loaded lazily
        self._simfin_ok: bool = False

    def load_all(self, max_workers: int = 6, try_simfin: bool = True):
        """Load all data sources. SimFin first, then yfinance for gaps."""
        # Try SimFin
        if try_simfin:
            try:
                from hedge_fund_ai.data.simfin_loader import SimFinStore
                store = SimFinStore()
                if store.load():
                    self._simfin    = store
                    self._simfin_ok = True
                    logger.info("SimFin loaded successfully — using 10-year fundamentals")
                else:
                    logger.info("SimFin unavailable — using yfinance fallback")
            except Exception as e:
                logger.info("SimFin init failed: %s — using yfinance", e)

        # yfinance fallback (always load for tickers SimFin might miss)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        logger.info("Loading yfinance fundamentals for %d tickers...", len(self.tickers))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self._load_yf, t): t for t in self.tickers}
            for f in as_completed(futures):
                t = futures[f]
                try:
                    df = f.result()
                    if df is not None and not df.empty:
                        self._yf_data[t] = df
                except Exception as e:
                    logger.debug("yfinance fundamentals %s: %s", t, e)

        covered = len(self._yf_data)
        logger.info("Fundamentals loaded: %d/%d via yfinance%s",
                    covered, len(self.tickers),
                    " + SimFin" if self._simfin_ok else "")

    def _load_yf(self, t: str) -> pd.DataFrame | None:
        """Pull yfinance quarterly statements and build PIT index."""
        try:
            tk  = yf.Ticker(t)
            inc = tk.quarterly_income_stmt
            bal = tk.quarterly_balance_sheet
            cf  = tk.quarterly_cashflow

            if inc is None or inc.empty:
                return None

            records = []
            for col in inc.columns:
                try:
                    rdate = pd.Timestamp(col)
                    avail = rdate + timedelta(days=self.filing_lag)

                    def _r(df, *keys):
                        if df is None or df.empty:
                            return None
                        for k in keys:
                            if k in df.index:
                                try:
                                    v = float(df.loc[k, col])
                                    return v if np.isfinite(v) else None
                                except Exception:
                                    continue
                        return None

                    rev  = _r(inc, "Total Revenue", "Revenue")
                    ni   = _r(inc, "Net Income", "Net Income Common Stockholders")
                    ebit = _r(inc, "Operating Income", "EBIT")
                    gp   = _r(inc, "Gross Profit")
                    ta   = _r(bal, "Total Assets")
                    eq   = _r(bal, "Stockholders Equity", "Total Equity Gross Minority Interest")
                    ltd  = _r(bal, "Total Debt", "Long Term Debt And Capital Lease Obligation")
                    std  = _r(bal, "Current Debt", "Short Term Debt")
                    ca   = _r(bal, "Total Current Assets", "Current Assets")
                    cl   = _r(bal, "Total Current Liabilities", "Current Liabilities")
                    cfo  = _r(cf,  "Operating Cash Flow", "Net Cash From Operating Activities")
                    cap  = _r(cf,  "Capital Expenditure", "Purchase Of Ppe")

                    total_debt = (ltd or 0) + (std or 0)
                    fcf = (cfo - abs(cap or 0)) if cfo is not None else None

                    records.append({
                        "available_date":  avail,
                        "report_date":     rdate,
                        "revenue":         rev,
                        "net_income":      ni,
                        "ebit":            ebit,
                        "gross_profit":    gp,
                        "total_assets":    ta,
                        "equity":          eq,
                        "total_debt":      total_debt,
                        "current_assets":  ca,
                        "current_liab":    cl,
                        "cfo":             cfo,
                        "fcf":             fcf,
                        "roe":    _pct(ni, eq),
                        "roa":    _pct(ni, ta),
                        "roic":   _pct(ebit, (eq or 0) + (total_debt or 0)),
                        "op_margin":  _pct(ebit, rev),
                        "net_margin": _pct(ni, rev),
                        "gp_margin":  _pct(gp, rev),
                        "fcf_margin": _pct(fcf, rev),
                        "de_ratio":   _div(total_debt, eq),
                        "curr_ratio": _div(ca, cl),
                    })
                except Exception:
                    continue

            if not records:
                return None

            df = pd.DataFrame(records).sort_values("available_date")

            # YoY growth
            for col_name, raw_col in [("rev_growth", "revenue"), ("ni_growth", "net_income")]:
                df[col_name] = df[raw_col].pct_change(4) * 100

            df = df.set_index("available_date").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            return df

        except Exception as e:
            logger.debug("yfinance load %s: %s", t, e)
            return None

    def as_of(self, ticker: str, date) -> dict:
        """
        Return fundamental snapshot available on or before `date`.
        Tries SimFin first, then yfinance.
        """
        date_ts = pd.Timestamp(date)

        # SimFin (preferred — longer history, better quality)
        if self._simfin_ok and self._simfin is not None:
            try:
                result = self._simfin.as_of(ticker, date_ts)
                if result:
                    return result
            except Exception:
                pass

        # yfinance fallback
        df = self._yf_data.get(ticker)
        if df is None or df.empty:
            return {}

        past = df[df.index <= date_ts]
        if past.empty:
            return {}

        row = past.iloc[-1]

        def _g(k): return float(row[k]) if k in row.index and pd.notna(row[k]) else 0.0

        # Build quality composite
        roe = _g("roe"); roic = _g("roic"); op = _g("op_margin"); fcf = _g("fcf_margin")
        quality = round(0.30*roe + 0.30*roic + 0.20*op + 0.20*fcf, 2)

        return {
            "roe_pct":         _g("roe"),
            "roa_pct":         _g("roa"),
            "roic_pct":        _g("roic"),
            "op_margin_pct":   _g("op_margin"),
            "net_margin_pct":  _g("net_margin"),
            "gross_margin_pct":_g("gp_margin"),
            "fcf_margin_pct":  _g("fcf_margin"),
            "de_ratio":        _g("de_ratio"),
            "current_ratio":   _g("curr_ratio"),
            "revenue_growth":  _g("rev_growth"),
            "ni_growth":       _g("ni_growth"),
            "quality_score":   quality,
            "cfo":             _g("cfo"),
            "fcf":             _g("fcf"),
        }

    def coverage(self) -> dict:
        total = len(self.tickers)
        covered_yf = len(self._yf_data)
        return {
            "total_tickers":   total,
            "yf_covered":      covered_yf,
            "simfin_available":self._simfin_ok,
            "per_ticker": {
                t: {
                    "n_quarters": len(self._yf_data[t]) if t in self._yf_data else 0,
                    "simfin": self._simfin_ok,
                }
                for t in self.tickers[:10]  # sample
            }
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(num, den):
    if num is None or den is None or den == 0:
        return None
    try:
        v = float(num) / float(den) * 100
        return round(v, 2) if np.isfinite(v) else None
    except Exception:
        return None


def _div(num, den):
    if num is None or den is None or den == 0:
        return None
    try:
        v = float(num) / float(den)
        return round(v, 4) if np.isfinite(v) else None
    except Exception:
        return None


def _safe_float(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0
