"""
SimFin Integration — 10+ years of point-in-time fundamentals.

SimFin provides:
  - Annual + quarterly income statements, balance sheets, cash flows
  - Point-in-time publish dates (no lookahead)
  - Free tier covers US equities 2012-present
  - Survivorship-bias-free (includes delisted companies)

Usage:
  store = SimFinStore()
  store.load()                      # downloads bulk CSVs (~200MB, cached)
  fund = store.as_of("AAPL", "2019-06-30")   # point-in-time lookup

Free tier limits: bulk download, no real-time, max 500 companies.
API key "free" works without registration for non-commercial use.
"""
import warnings
warnings.filterwarnings("ignore", message=".*date_parser.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*parse_dates.*", category=FutureWarning)
import logging
import os
from datetime import timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from hedge_fund_ai.config import SIMFIN_API_KEY, SIMFIN_DATA_DIR

logger = logging.getLogger(__name__)

FILING_LAG_DAYS = 60   # conservative: 60 days after period end


class SimFinStore:
    """
    Loads SimFin bulk data and provides point-in-time fundamental lookups.
    All lookups guarantee no lookahead: only uses data published <= query_date.
    """

    def __init__(self, data_dir: str = SIMFIN_DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._income:  dict[str, pd.DataFrame] = {}
        self._balance: dict[str, pd.DataFrame] = {}
        self._cashflow:dict[str, pd.DataFrame] = {}
        self._loaded = False

    # ── Loading ────────────────────────────────────────────────────────────────

    def load(self, use_cache: bool = True) -> bool:
        """
        Download SimFin bulk data and parse into per-ticker DataFrames.
        Returns True if data loaded successfully.
        """
        try:
            import simfin as sf
            sf.set_api_key(SIMFIN_API_KEY)
            sf.set_data_dir(self.data_dir)

            logger.info("Loading SimFin data (first run downloads ~200MB)...")

            # Load annual statements (wider history, better quality)
            income   = sf.load_income(variant="annual",   market="us")
            balance  = sf.load_balance(variant="annual",  market="us")
            cashflow = sf.load_cashflow(variant="annual", market="us")

            # Also load quarterly for recency
            income_q   = sf.load_income(variant="quarterly",   market="us")
            balance_q  = sf.load_balance(variant="quarterly",  market="us")
            cashflow_q = sf.load_cashflow(variant="quarterly", market="us")

            self._income   = self._build_pit_index(income,   income_q)
            self._balance  = self._build_pit_index(balance,  balance_q)
            self._cashflow = self._build_pit_index(cashflow, cashflow_q)

            n = len(self._income)
            logger.info("SimFin loaded: %d tickers, income=%d, balance=%d, cashflow=%d",
                        n, len(self._income), len(self._balance), len(self._cashflow))
            self._loaded = True
            return True

        except ImportError:
            logger.warning("simfin not installed. Run: pip install simfin")
            return False
        except Exception as e:
            err = str(e)
            if "401" in err or "Unauthorized" in err:
                logger.warning(
                    "SimFin 401 Unauthorized — free tier key 'free' only works for "
                    "bulk downloads on the legacy API. Set SIMFIN_API_KEY=free in .env "
                    "or register at simfin.com for a personal key. Using yfinance fallback."
                )
            elif "404" in err or "not found" in err.lower():
                logger.warning("SimFin dataset not found — API may have changed. Using yfinance fallback.")
            else:
                logger.warning("SimFin load failed: %s — using yfinance fallback", e)
            return False

    def _build_pit_index(self, annual: pd.DataFrame, quarterly: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Combine annual + quarterly into per-ticker DataFrames indexed by
        'available_date' = report_date + FILING_LAG_DAYS.
        """
        result = {}

        def _process(df: pd.DataFrame, freq: str):
            if df is None or df.empty:
                return
            # SimFin index: (Ticker, Report Date) or similar
            df = df.reset_index()

            # Normalize column names
            ticker_col   = next((c for c in df.columns if "ticker" in c.lower()), None)
            date_col     = next((c for c in df.columns if "report date" in c.lower()
                                 or "period" in c.lower()), None)
            publish_col  = next((c for c in df.columns if "publish" in c.lower()), None)

            if ticker_col is None or date_col is None:
                return

            for ticker, grp in df.groupby(ticker_col):
                ticker = str(ticker)
                grp = grp.copy()
                grp["report_date"] = pd.to_datetime(grp[date_col], errors="coerce")

                if publish_col and publish_col in grp.columns:
                    grp["available_date"] = pd.to_datetime(grp[publish_col], errors="coerce")
                    # Use max(publish_date, report_date + lag) to be conservative
                    fallback = grp["report_date"] + timedelta(days=FILING_LAG_DAYS)
                    grp["available_date"] = grp[["available_date"]].assign(
                        fb=fallback
                    ).max(axis=1)
                else:
                    grp["available_date"] = grp["report_date"] + timedelta(days=FILING_LAG_DAYS)

                grp = grp.dropna(subset=["available_date", "report_date"])
                grp = grp.set_index("available_date").sort_index()
                grp["_freq"] = freq

                if ticker in result:
                    result[ticker] = pd.concat([result[ticker], grp]).sort_index()
                else:
                    result[ticker] = grp

        _process(annual,    "annual")
        _process(quarterly, "quarterly")

        # Deduplicate by available_date (prefer quarterly for recency)
        for ticker in result:
            df = result[ticker]
            df = df[~df.index.duplicated(keep="last")]
            result[ticker] = df.sort_index()

        return result

    # ── Point-in-time lookup ───────────────────────────────────────────────────

    def as_of(self, ticker: str, date) -> dict:
        """
        Return all fundamental metrics available on or before `date`.
        Guaranteed no lookahead.
        """
        if not self._loaded:
            return {}

        date_ts = pd.Timestamp(date)

        inc  = self._get_row_as_of(self._income,   ticker, date_ts)
        bal  = self._get_row_as_of(self._balance,  ticker, date_ts)
        cf   = self._get_row_as_of(self._cashflow, ticker, date_ts)

        if inc is None and bal is None:
            return {}

        result = {}

        # ── Income statement ───────────────────────────────────────────────────
        rev  = _sf(inc, ["Revenue", "Total Revenue", "Net Revenue"])
        ni   = _sf(inc, ["Net Income", "Net Income (Common)"])
        ebit = _sf(inc, ["Operating Income (Loss)", "EBIT", "Operating Income"])
        gp   = _sf(inc, ["Gross Profit"])
        rd   = _sf(inc, ["Research & Development"])

        # ── Balance sheet ──────────────────────────────────────────────────────
        assets  = _sf(bal, ["Total Assets"])
        equity  = _sf(bal, ["Total Equity", "Stockholders Equity",
                             "Total Equity & Noncontrolling Interests"])
        debt_lt = _sf(bal, ["Long-Term Debt", "Long Term Debt"])
        debt_st = _sf(bal, ["Short-Term Debt", "Short Term Debt", "Current Portion of Long-Term Debt"])
        cash    = _sf(bal, ["Cash & Cash Equivalents", "Cash, Equivalents & Short-term Investments"])
        curr_a  = _sf(bal, ["Total Current Assets"])
        curr_l  = _sf(bal, ["Total Current Liabilities"])

        # ── Cash flow ──────────────────────────────────────────────────────────
        cfo     = _sf(cf, ["Net Cash from Operating Activities", "Cash from Operations"])
        capex   = _sf(cf, ["Capital Expenditures", "Purchase of Property, Plant & Equipment"])
        fcf     = (cfo - abs(capex or 0)) if (cfo is not None and capex is not None) else None

        # ── Derived metrics ────────────────────────────────────────────────────
        total_debt = (debt_lt or 0) + (debt_st or 0)

        result["revenue"]        = rev
        result["net_income"]     = ni
        result["gross_profit"]   = gp
        result["ebit"]           = ebit
        result["cfo"]            = cfo
        result["fcf"]            = fcf
        result["total_assets"]   = assets
        result["total_equity"]   = equity
        result["total_debt"]     = total_debt
        result["cash"]           = cash

        # Ratios
        result["roe_pct"]        = _pct(ni, equity)
        result["roa_pct"]        = _pct(ni, assets)
        result["roic_pct"]       = _pct(ebit, (equity or 0) + (total_debt or 0))
        result["op_margin_pct"]  = _pct(ebit, rev)
        result["net_margin_pct"] = _pct(ni, rev)
        result["gross_margin_pct"] = _pct(gp, rev)
        result["fcf_margin_pct"] = _pct(fcf, rev)
        result["de_ratio"]       = _div(total_debt, equity)
        result["current_ratio"]  = _div(curr_a, curr_l)
        result["rd_intensity"]   = _pct(rd, rev)   # R&D / revenue (innovation proxy)

        # YoY growth (requires prior year — look back 12-16 months for annual)
        prior_date = date_ts - timedelta(days=380)
        inc_prior  = self._get_row_as_of(self._income, ticker, prior_date)
        rev_prior  = _sf(inc_prior, ["Revenue", "Total Revenue", "Net Revenue"])
        ni_prior   = _sf(inc_prior, ["Net Income", "Net Income (Common)"])

        result["revenue_growth"] = _growth(rev, rev_prior)
        result["ni_growth"]      = _growth(ni, ni_prior)

        # Quality composite (Piotroski-inspired, computed below)
        result["quality_score"]  = _quality_composite(result)

        # Accruals (earnings quality): (NI - CFO) / Assets — lower is better
        result["accruals"] = (
            _div((ni or 0) - (cfo or 0), assets)
            if (ni is not None and cfo is not None and assets)
            else None
        )

        return {k: v for k, v in result.items() if v is not None}

    def _get_row_as_of(self, store: dict, ticker: str, date: pd.Timestamp):
        df = store.get(ticker)
        if df is None or df.empty:
            return None
        past = df[df.index <= date]
        return past.iloc[-1] if not past.empty else None

    def coverage(self) -> dict:
        tickers = set(self._income) | set(self._balance)
        return {
            t: {
                "income_quarters": len(self._income.get(t, [])),
                "balance_quarters": len(self._balance.get(t, [])),
                "income_start": str(self._income[t].index[0].date()) if t in self._income and len(self._income[t]) else None,
            }
            for t in sorted(tickers)
        }

    @property
    def loaded(self): return self._loaded


# ── Piotroski F-Score ──────────────────────────────────────────────────────────

def compute_piotroski(fund: dict, fund_prior: dict = None) -> int:
    """
    Compute Piotroski F-score (0-9). Higher = financially stronger.

    Profitability (4 points):
      F1: ROA > 0
      F2: CFO > 0
      F3: ROA improving YoY
      F4: CFO > Net Income (accruals quality)

    Leverage/Liquidity (3 points):
      F5: Long-term debt ratio decreasing
      F6: Current ratio improving
      F7: No new shares issued

    Operating efficiency (2 points):
      F8: Gross margin improving
      F9: Asset turnover improving
    """
    score = 0
    fp = fund_prior or {}

    roa   = fund.get("roa_pct", 0) or 0
    cfo   = fund.get("cfo") or 0
    ni    = fund.get("net_income") or 0
    gm    = fund.get("gross_margin_pct", 0) or 0
    de    = fund.get("de_ratio", 999) or 999
    cr    = fund.get("current_ratio", 0) or 0
    rev   = fund.get("revenue") or 0
    asset = fund.get("total_assets") or 1

    roa_p  = fp.get("roa_pct", 0) or 0
    gm_p   = fp.get("gross_margin_pct", 0) or 0
    de_p   = fp.get("de_ratio", 999) or 999
    cr_p   = fp.get("current_ratio", 0) or 0
    rev_p  = fp.get("revenue") or 0
    asset_p= fp.get("total_assets") or 1

    # Profitability
    if roa > 0:              score += 1  # F1
    if cfo > 0:              score += 1  # F2
    if roa > roa_p:          score += 1  # F3
    if cfo > ni:             score += 1  # F4 accruals

    # Leverage
    if de < de_p:            score += 1  # F5 debt decreasing
    if cr > cr_p:            score += 1  # F6 liquidity improving
    # F7: no dilution — skip (requires share count history)
    score += 1  # give benefit of doubt

    # Efficiency
    at  = rev / asset   if asset  else 0
    at_p = rev_p / asset_p if asset_p else 0
    if gm > gm_p:            score += 1  # F8
    if at > at_p:            score += 1  # F9

    return min(score, 9)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sf(row, keys: list):
    """Extract a float from a pandas Series by trying multiple column names."""
    if row is None:
        return None
    for k in keys:
        if k in row.index:
            try:
                v = float(row[k])
                return v if np.isfinite(v) else None
            except Exception:
                continue
    return None


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


def _growth(current, prior):
    if current is None or prior is None or prior == 0:
        return None
    try:
        v = (float(current) - float(prior)) / abs(float(prior)) * 100
        return round(v, 2) if np.isfinite(v) else None
    except Exception:
        return None


def _quality_composite(f: dict) -> float:
    """Weighted quality score from available metrics."""
    roe  = f.get("roe_pct",        0) or 0
    roic = f.get("roic_pct",       0) or 0
    op   = f.get("op_margin_pct",  0) or 0
    fcf  = f.get("fcf_margin_pct", 0) or 0
    roa  = f.get("roa_pct",        0) or 0
    return round(0.25*roe + 0.25*roic + 0.20*op + 0.20*fcf + 0.10*roa, 2)


# ── Singleton ─────────────────────────────────────────────────────────────────

_store: SimFinStore | None = None


def get_simfin_store() -> SimFinStore:
    global _store
    if _store is None:
        _store = SimFinStore()
    return _store
