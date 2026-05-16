"""
Live Signal Logger — Gap 3.

Logs daily model predictions vs realized forward returns.
After 60 days, computes realized IC to validate signal quality.

Usage:
  logger = SignalLogger()
  logger.log_signals(date, scores)          # call daily after scoring
  logger.fill_returns(date, fwd_returns)    # call 21 days later
  report = logger.ic_report()              # validate signal
  logger.save()
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "state", "signal_log.json")


class SignalLogger:
    """
    Records daily signal scores and fills in realized forward returns
    to compute live Information Coefficient.
    """

    def __init__(self, fwd_days: int = 21):
        self.fwd_days = fwd_days
        self._records: list[dict] = []
        self._load()

    def _load(self):
        if not os.path.exists(_LOG_PATH):
            return
        try:
            with open(_LOG_PATH) as f:
                self._records = json.load(f)
            logger.info("Signal log loaded: %d entries", len(self._records))
        except Exception as e:
            logger.warning("Signal log load failed: %s", e)
            self._records = []

    def save(self):
        os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
        with open(_LOG_PATH, "w") as f:
            json.dump(self._records, f, indent=2, default=str)

    def log_signals(self, date: str | datetime, scores: dict[str, float],
                    regime: str = "neutral", top_picks: list[str] | None = None):
        """
        Record model scores for all tickers on this date.

        Args:
            date:       today's date (YYYY-MM-DD or datetime)
            scores:     {ticker: composite_score}
            regime:     active regime label
            top_picks:  list of selected portfolio tickers
        """
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        fwd_due  = (pd.Timestamp(date) + timedelta(days=self.fwd_days)).strftime("%Y-%m-%d")

        record = {
            "signal_date":   date_str,
            "fwd_due_date":  fwd_due,
            "regime":        regime,
            "scores":        {t: round(float(s), 5) for t, s in scores.items()},
            "top_picks":     top_picks or [],
            "fwd_returns":   {},        # filled in later
            "ic":            None,      # filled in after fwd_returns
            "filled":        False,
            "logged_at":     datetime.now(timezone.utc).isoformat(),
        }
        self._records.append(record)
        self.save()
        logger.info("Signals logged: %d tickers, date=%s, due=%s",
                    len(scores), date_str, fwd_due)

    def fill_returns(self, fwd_prices: pd.DataFrame | None = None):
        """
        For all unfilled records whose fwd_due_date has passed,
        compute realized forward returns from price data and calculate IC.

        Call this daily. Pass in a DataFrame of recent prices.
        """
        if fwd_prices is None:
            try:
                import yfinance as yf
                tickers_needed = set()
                for r in self._records:
                    if not r["filled"]:
                        tickers_needed.update(r["scores"].keys())
                if not tickers_needed:
                    return
                fwd_prices = yf.download(list(tickers_needed), period="3mo", progress=False)["Close"]
                if isinstance(fwd_prices, pd.Series):
                    fwd_prices = fwd_prices.to_frame()
            except Exception as e:
                logger.warning("Price fetch for IC fill failed: %s", e)
                return

        today = pd.Timestamp.now().normalize()
        updated = False

        for record in self._records:
            if record.get("filled"):
                continue

            due = pd.Timestamp(record["fwd_due_date"])
            if today < due:
                continue  # not yet due

            signal_date = pd.Timestamp(record["signal_date"])

            fwd_rets: dict[str, float] = {}
            for t in record["scores"]:
                if t not in fwd_prices.columns:
                    continue
                try:
                    # Find prices on or after signal_date + fwd_days
                    px = fwd_prices[t].dropna()
                    start_px = px[px.index >= signal_date]
                    end_px   = px[px.index >= due]
                    if len(start_px) > 0 and len(end_px) > 0:
                        ret = float(end_px.iloc[0] / start_px.iloc[0] - 1)
                        fwd_rets[t] = round(ret, 6)
                except Exception:
                    continue

            if not fwd_rets:
                continue

            record["fwd_returns"] = fwd_rets

            # Compute IC: Spearman rank correlation of scores vs forward returns
            common = [t for t in record["scores"] if t in fwd_rets]
            if len(common) >= 5:
                import warnings
                from scipy.stats import spearmanr
                scores_v  = [record["scores"][t] for t in common]
                returns_v = [fwd_rets[t]          for t in common]
                if np.std(scores_v) < 1e-10 or np.std(returns_v) < 1e-10:
                    record["ic"] = None
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ic, _ = spearmanr(scores_v, returns_v)
                    record["ic"] = round(float(ic), 4) if not np.isnan(ic) else None

            record["filled"] = True
            updated = True
            logger.info("IC filled for %s: IC=%.3f (n=%d tickers)",
                        record["signal_date"], record.get("ic") or 0, len(fwd_rets))

        if updated:
            self.save()

    def ic_report(self) -> dict:
        """
        Summarize live signal IC across all filled records.
        This is the primary validation metric before deploying real capital.
        """
        filled = [r for r in self._records if r.get("filled") and r.get("ic") is not None]
        if not filled:
            return {
                "status":   "insufficient_data",
                "message":  f"Need filled records. {len(self._records)} total, 0 filled.",
                "n_filled": 0,
            }

        ics = [r["ic"] for r in filled]
        regimes = {}
        for r in filled:
            reg = r.get("regime", "unknown")
            if reg not in regimes:
                regimes[reg] = []
            regimes[reg].append(r["ic"])

        mean_ic   = float(np.mean(ics))
        std_ic    = float(np.std(ics)) + 1e-9
        ic_ir     = mean_ic / std_ic
        hit_rate  = float(np.mean([ic > 0 for ic in ics]))

        # Top-5 pick accuracy: did top picks outperform median?
        pick_accuracy = []
        for r in filled:
            if not r.get("top_picks") or not r.get("fwd_returns"):
                continue
            top_rets  = [r["fwd_returns"].get(t, 0) for t in r["top_picks"] if t in r["fwd_returns"]]
            all_rets  = list(r["fwd_returns"].values())
            if top_rets and all_rets:
                median_all = float(np.median(all_rets))
                pick_accuracy.append(float(np.mean(top_rets)) > median_all)

        report = {
            "status":       "valid" if len(filled) >= 12 else "early",
            "n_filled":     len(filled),
            "n_total":      len(self._records),
            "mean_ic":      round(mean_ic, 4),
            "ic_std":       round(std_ic, 4),
            "ic_ir":        round(ic_ir, 3),
            "ic_hit_rate":  round(hit_rate, 3),
            "latest_ic":    round(float(ics[-1]), 4) if ics else None,
            "top_pick_accuracy": round(float(np.mean(pick_accuracy)), 3) if pick_accuracy else None,
            "by_regime":    {
                reg: {
                    "mean_ic":  round(float(np.mean(vals)), 4),
                    "n":        len(vals),
                    "hit_rate": round(float(np.mean([v > 0 for v in vals])), 3),
                }
                for reg, vals in regimes.items()
            },
            "verdict":      _verdict(mean_ic, ic_ir, hit_rate, len(filled)),
        }

        logger.info("IC Report: mean=%.3f IC-IR=%.2f hit=%.1f%% n=%d — %s",
                    mean_ic, ic_ir, hit_rate * 100, len(filled), report["verdict"])
        return report

    def pending_fills(self) -> list[str]:
        """Dates that are due for filling but not yet filled."""
        today = pd.Timestamp.now().normalize()
        return [
            r["signal_date"] for r in self._records
            if not r.get("filled") and pd.Timestamp(r["fwd_due_date"]) <= today
        ]


def _verdict(mean_ic: float, ic_ir: float, hit_rate: float, n: int) -> str:
    """
    Professional assessment of signal quality.
    Based on Grinold & Kahn (Active Portfolio Management) thresholds.
    """
    if n < 12:
        return f"WAIT — need {12 - n} more periods before assessment"
    if mean_ic > 0.05 and ic_ir > 0.5 and hit_rate > 0.55:
        return "DEPLOY — strong signal quality, proceed to live capital"
    if mean_ic > 0.02 and ic_ir > 0.25:
        return "PAPER — signal exists but needs more history before capital"
    if mean_ic > 0.0 and hit_rate > 0.50:
        return "MONITOR — weak positive IC, watch for 3 more months"
    if mean_ic <= 0.0:
        return "NO ALPHA — signal has negative IC, do not deploy capital"
    return "UNCERTAIN — insufficient evidence either way"
