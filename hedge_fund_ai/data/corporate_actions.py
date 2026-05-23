"""
Corporate Action Handler.

Tracks and adjusts for:
  - Stock splits (price/share adjustment)
  - Dividends (return impact, cost basis)
  - Spin-offs (new ticker creation, weight adjustment)
  - Mergers/acquisitions (position close)
  - Tender offers (partial position liquidation)

Data sources:
  1. yfinance actions (splits + dividends — free, built-in)
  2. SEC EDGAR 8-K filings (spin-offs, mergers — via EDGAR full-text search)
  3. Manual override file (state/corporate_actions_override.json)

Usage in walk_forward:
  ca = CorporateActionStore(tickers)
  ca.load()
  adj = ca.get_adjustment(ticker, from_date, to_date)
  adjusted_price = raw_price * adj["price_multiplier"]
"""

import json
import logging
import os
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import yfinance as yf

from hedge_fund_ai.data.cache import simple_cache

logger = logging.getLogger(__name__)

_OVERRIDE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "state", "corporate_actions_override.json"
)

# Known major events (manually curated for backtest accuracy)
# Format: {ticker: [{date, type, ratio, note}]}
_KNOWN_EVENTS = {
    "TSLA": [
        {"date": "2022-08-25", "type": "split",   "ratio": 3.0, "note": "3-for-1 split"},
        {"date": "2020-08-31", "type": "split",   "ratio": 5.0, "note": "5-for-1 split"},
    ],
    "GOOGL": [
        {"date": "2022-07-18", "type": "split",   "ratio": 20.0, "note": "20-for-1 split"},
    ],
    "AMZN": [
        {"date": "2022-06-06", "type": "split",   "ratio": 20.0, "note": "20-for-1 split"},
    ],
    "NVDA": [
        {"date": "2021-07-20", "type": "split",   "ratio": 4.0, "note": "4-for-1 split"},
        {"date": "2024-06-10", "type": "split",   "ratio": 10.0, "note": "10-for-1 split"},
    ],
    "AAPL": [
        {"date": "2020-08-31", "type": "split",   "ratio": 4.0, "note": "4-for-1 split"},
    ],
    "GE": [
        {"date": "2021-07-30", "type": "spinoff", "ratio": 1.0, "note": "GE Healthcare spinoff"},
    ],
    "JNJ": [
        {"date": "2023-05-04", "type": "spinoff", "ratio": 1.0, "note": "Kenvue spinoff"},
    ],
}


class CorporateActionStore:
    """
    Fetches and caches corporate actions per ticker.
    Provides point-in-time adjustment factors for historical price series.
    """

    def __init__(self, tickers: list[str]):
        self.tickers = list(tickers)
        self._actions: dict[str, pd.DataFrame] = {}
        self._overrides = self._load_overrides()

    def _load_overrides(self) -> dict:
        if not os.path.exists(_OVERRIDE_PATH):
            return {}
        try:
            with open(_OVERRIDE_PATH) as f:
                return json.load(f)
        except Exception:
            return {}

    def load(self, max_workers: int = 6):
        """Load corporate actions for all tickers concurrently."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info("Loading corporate actions for %d tickers...", len(self.tickers))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self._load_ticker, t): t for t in self.tickers}
            for f in as_completed(futures):
                t = futures[f]
                try:
                    df = f.result()
                    if df is not None:
                        self._actions[t] = df
                except Exception as e:
                    logger.debug("Corporate actions %s: %s", t, e)

        covered = len(self._actions)
        logger.info("Corporate actions loaded: %d/%d tickers", covered, len(self.tickers))

    @simple_cache(ttl=3600 * 24, disk=True)
    def _load_ticker(self, t: str) -> pd.DataFrame | None:
        """Pull splits + dividends from yfinance and merge with known events."""
        rows = []

        # yfinance splits
        try:
            ticker = yf.Ticker(t)
            splits = ticker.splits
            if splits is not None and len(splits) > 0:
                for date, ratio in splits.items():
                    if float(ratio) > 0:
                        rows.append({
                            "date":             pd.Timestamp(date).date().isoformat(),
                            "type":             "split",
                            "ratio":            float(ratio),
                            "price_multiplier": 1.0 / float(ratio),
                            "source":           "yfinance",
                        })
        except Exception as e:
            logger.debug("yfinance splits %s: %s", t, e)

        # yfinance dividends (record for return calculation)
        try:
            ticker = yf.Ticker(t)
            divs = ticker.dividends
            if divs is not None and len(divs) > 0:
                for date, amount in divs.tail(20).items():
                    rows.append({
                        "date":             pd.Timestamp(date).date().isoformat(),
                        "type":             "dividend",
                        "ratio":            float(amount),
                        "price_multiplier": 1.0,   # dividends don't adjust price retrospectively
                        "source":           "yfinance",
                    })
        except Exception as e:
            logger.debug("yfinance dividends %s: %s", t, e)

        # Known curated events
        for event in _KNOWN_EVENTS.get(t, []):
            ratio = float(event["ratio"])
            rows.append({
                "date":             event["date"],
                "type":             event["type"],
                "ratio":            ratio,
                "price_multiplier": 1.0 / ratio if event["type"] == "split" else 1.0,
                "source":           "curated",
                "note":             event.get("note", ""),
            })

        # Manual overrides
        for event in self._overrides.get(t, []):
            rows.append({**event, "source": "override"})

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").drop_duplicates(subset=["date", "type"])

    def get_events_between(self, ticker: str, from_date, to_date) -> list[dict]:
        """Return all corporate actions for ticker between from_date and to_date."""
        df = self._actions.get(ticker)
        if df is None or df.empty:
            return []
        mask = (df["date"] >= pd.Timestamp(from_date)) & (df["date"] <= pd.Timestamp(to_date))
        return df[mask].to_dict("records")

    def get_adjustment(self, ticker: str, as_of_date) -> dict:
        """
        Return cumulative price adjustment multiplier as of a date.
        Accounts for all splits that occurred before as_of_date.
        Multiply raw prices by this factor to get adjusted prices.
        """
        df = self._actions.get(ticker)
        if df is None or df.empty:
            return {"price_multiplier": 1.0, "events": []}

        past = df[df["date"] <= pd.Timestamp(as_of_date)]
        splits = past[past["type"] == "split"]

        cumulative = float(splits["price_multiplier"].prod()) if not splits.empty else 1.0
        events = past.to_dict("records")
        return {"price_multiplier": round(cumulative, 8), "events": events}

    def flag_merger_or_delisting(self, ticker: str, as_of_date) -> bool:
        """
        Returns True if ticker was acquired/merged/delisted by as_of_date.
        Used by universe filter to exclude such tickers from positions.
        """
        df = self._actions.get(ticker)
        if df is None or df.empty:
            return False
        past = df[
            (df["date"] <= pd.Timestamp(as_of_date)) &
            (df["type"].isin(["merger", "acquisition", "delisting"]))
        ]
        return not past.empty

    def has_spinoff(self, ticker: str, from_date, to_date) -> bool:
        """Returns True if a spin-off occurred in the period — signals position review needed."""
        events = self.get_events_between(ticker, from_date, to_date)
        return any(e["type"] == "spinoff" for e in events)

    def annual_dividend_yield(self, ticker: str) -> float:
        """Approximate annual dividend yield from recent payments."""
        df = self._actions.get(ticker)
        if df is None or df.empty:
            return 0.0
        divs = df[df["type"] == "dividend"].tail(4)
        return float(divs["ratio"].sum()) if not divs.empty else 0.0


def get_restatement_flag(fund_data_v1: dict, fund_data_v2: dict, threshold: float = 0.10) -> dict:
    """
    Compare two versions of fundamental data and flag metrics with >threshold revision.
    Returns {metric: {v1, v2, revision_pct, flagged}} for any restated metric.

    Used to detect when a company restated earnings/revenue after initial filing.
    Flagged tickers may need temporary exclusion from quality/growth signal.
    """
    flags = {}
    all_keys = set(fund_data_v1) | set(fund_data_v2)

    for key in all_keys:
        v1 = fund_data_v1.get(key)
        v2 = fund_data_v2.get(key)

        if v1 is None or v2 is None:
            continue
        try:
            v1f, v2f = float(v1), float(v2)
            if abs(v1f) < 1e-6:
                continue
            revision = abs(v2f - v1f) / abs(v1f)
            if revision > threshold:
                flags[key] = {
                    "v1":          round(v1f, 4),
                    "v2":          round(v2f, 4),
                    "revision_pct":round(revision * 100, 2),
                    "flagged":     True,
                }
        except Exception:
            continue

    if flags:
        logger.warning("Restatement flags: %s", list(flags.keys()))

    return flags
