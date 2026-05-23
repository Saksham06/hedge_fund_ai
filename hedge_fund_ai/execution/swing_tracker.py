"""
Swing Position Tracker — per-position hold duration + exit signals.

Tracks:
  1. Entry date, entry price, current hold days
  2. Time-based exit: force close after SWING_MAX_HOLD_DAYS
  3. Profit-target exit: close when gain > SWING_PROFIT_TARGET
  4. Stop-loss exit: close when loss > SWING_STOP_LOSS
  5. Min hold enforcement: never exit before SWING_MIN_HOLD_DAYS

Swing metrics computed from closed trades:
  - Avg hold period
  - Win rate by bucket (<7d, 7-30d, 30-60d)
  - Profit factor
  - Max consecutive losses
  - Cost-adjusted net return
"""
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from hedge_fund_ai.config import (
    SWING_MAX_HOLD_DAYS, SWING_MIN_HOLD_DAYS,
    SWING_PROFIT_TARGET, SWING_STOP_LOSS,
)
import os as _os

# Gap risk thresholds
GAP_RISK_THRESHOLD   = float(_os.getenv("GAP_RISK_THRESHOLD",   "0.03"))  # 3% overnight gap
GAP_RISK_STOP_WIDEN  = float(_os.getenv("GAP_RISK_STOP_WIDEN",  "0.07"))  # widen stop to 7%
GAP_RISK_SIZE_REDUCE = float(_os.getenv("GAP_RISK_SIZE_REDUCE", "0.50"))  # reduce size 50%

# Trend filter thresholds (for whipsaw prevention)
TREND_VIX_THRESHOLD  = float(_os.getenv("TREND_VIX_THRESHOLD",  "22.0"))  # VIX < 22 = trending

logger = logging.getLogger(__name__)

_TRACKER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "state", "swing_positions.json"
)
_TRADES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "state", "swing_trades.json"
)


class SwingTracker:
    """
    Tracks open swing positions and records closed trades for metrics.
    """

    def __init__(self):
        self._open:   dict[str, dict] = {}   # {ticker: position_dict}
        self._closed: list[dict]      = []   # completed trades
        self._load()

    def _load(self):
        if os.path.exists(_TRACKER_PATH):
            try:
                with open(_TRACKER_PATH) as f:
                    self._open = json.load(f)
            except Exception:
                self._open = {}
        if os.path.exists(_TRADES_PATH):
            try:
                with open(_TRADES_PATH) as f:
                    self._closed = json.load(f)
            except Exception:
                self._closed = []

    def _save(self):
        os.makedirs(os.path.dirname(_TRACKER_PATH), exist_ok=True)
        with open(_TRACKER_PATH, "w") as f:
            json.dump(self._open, f, indent=2, default=str)
        with open(_TRADES_PATH, "w") as f:
            json.dump(self._closed[-500:], f, indent=2, default=str)  # keep last 500

    def open_position(self, ticker: str, entry_price: float, weight: float,
                      regime: str = "neutral", signal_score: float = 0.0):
        """Record a new swing position."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._open[ticker] = {
            "ticker":       ticker,
            "entry_date":   today,
            "entry_price":  round(float(entry_price), 4),
            "weight":       round(float(weight), 4),
            "regime":       regime,
            "signal_score": round(float(signal_score), 4),
            "peak_price":   round(float(entry_price), 4),
        }
        self._save()
        logger.info("SWING OPEN: %s @ $%.2f (score=%.2f)", ticker, entry_price, signal_score)

    def update_prices(self, current_prices: dict[str, float]):
        """Update peak prices for trailing stop logic."""
        for ticker, pos in self._open.items():
            px = current_prices.get(ticker, 0)
            if px > pos.get("peak_price", 0):
                pos["peak_price"] = round(float(px), 4)

    def detect_gap_risk(self, ticker: str, prev_close: float,
                         current_open: float) -> dict:
        """
        Detect overnight gap risk at market open.
        A gap > GAP_RISK_THRESHOLD triggers position size reduction or stop widening.

        Returns {has_gap, gap_pct, action}
        action: "widen_stop" | "reduce_size" | "none"
        """
        if prev_close <= 0 or current_open <= 0:
            return {"has_gap": False, "gap_pct": 0.0, "action": "none"}

        gap_pct = (current_open - prev_close) / prev_close
        if abs(gap_pct) > GAP_RISK_THRESHOLD:
            if gap_pct < 0:
                # Gap down > 3%: widen stop temporarily to avoid premature stop-out
                logger.warning("GAP RISK: %s gap=%.1f%% → widening stop to %.0f%%",
                               ticker, gap_pct*100, GAP_RISK_STOP_WIDEN*100)
                return {"has_gap": True, "gap_pct": round(gap_pct, 4),
                        "action": "widen_stop", "temp_stop": GAP_RISK_STOP_WIDEN}
            else:
                # Gap up > 3%: good for longs, no action needed
                return {"has_gap": True, "gap_pct": round(gap_pct, 4), "action": "none"}
        return {"has_gap": False, "gap_pct": round(gap_pct, 4), "action": "none"}

    def check_exits(self, current_prices: dict[str, float],
                    prev_closes: dict[str, float] | None = None) -> dict[str, str]:
        """
        Check every open position for exit conditions.

        Gap risk handling: if prev_closes provided, check for overnight gaps.
        A gap-down > 3% widens the stop to 7% temporarily (avoids stopping out
        on noise that will recover intraday).

        Exit reasons:
          "max_hold"        — held too long (> SWING_MAX_HOLD_DAYS)
          "profit_target"   — hit profit target
          "stop_loss"       — hit stop (may be widened for gap risk)
          "gap_down_exit"   — gapped below widened stop (severe gap)
        """
        today  = datetime.now(timezone.utc).date()
        exits  = {}

        for ticker, pos in self._open.items():
            entry_dt  = datetime.strptime(pos["entry_date"], "%Y-%m-%d").date()
            hold_days = (today - entry_dt).days
            entry_px  = float(pos["entry_price"])
            current   = float(current_prices.get(ticker, 0))

            if current <= 0:
                continue

            if hold_days < SWING_MIN_HOLD_DAYS:
                continue

            pnl_pct = (current - entry_px) / entry_px

            # Determine effective stop — may be widened due to gap risk
            effective_stop = SWING_STOP_LOSS
            if prev_closes and ticker in prev_closes:
                gap = self.detect_gap_risk(ticker, prev_closes[ticker], current)
                if gap["action"] == "widen_stop":
                    effective_stop = gap["temp_stop"]
                    pos["gap_widened_stop"] = True

            if hold_days >= SWING_MAX_HOLD_DAYS:
                exits[ticker] = "max_hold"
                logger.info("SWING EXIT (time): %s held %d days", ticker, hold_days)
            elif pnl_pct >= SWING_PROFIT_TARGET:
                exits[ticker] = "profit_target"
                logger.info("SWING EXIT (profit): %s +%.1f%%", ticker, pnl_pct*100)
            elif pnl_pct <= -effective_stop:
                reason = "stop_loss" if effective_stop == SWING_STOP_LOSS else "gap_down_exit"
                exits[ticker] = reason
                logger.warning("SWING EXIT (%s): %s %.1f%% (stop=%.0f%%)",
                               reason, ticker, pnl_pct*100, effective_stop*100)

        return exits

    def close_position(self, ticker: str, exit_price: float, exit_reason: str,
                        cost_bps: float = 15.0):
        """Record a closed trade and compute P&L."""
        pos = self._open.get(ticker)
        if pos is None:
            return

        today     = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        entry_px  = float(pos["entry_price"])
        entry_dt  = datetime.strptime(pos["entry_date"], "%Y-%m-%d").date()
        exit_dt   = datetime.now(timezone.utc).date()
        hold_days = (exit_dt - entry_dt).days

        gross_pnl = (exit_price - entry_px) / entry_px
        cost      = cost_bps / 10_000 * 2   # round-trip (entry + exit)
        net_pnl   = gross_pnl - cost

        trade = {
            "ticker":       ticker,
            "entry_date":   pos["entry_date"],
            "exit_date":    today,
            "hold_days":    hold_days,
            "entry_price":  entry_px,
            "exit_price":   round(float(exit_price), 4),
            "gross_pnl":    round(gross_pnl, 4),
            "net_pnl":      round(net_pnl, 4),
            "cost_bps":     cost_bps,
            "exit_reason":  exit_reason,
            "regime":       pos.get("regime", "unknown"),
            "signal_score": pos.get("signal_score", 0),
            "win":          net_pnl > 0,
        }
        self._closed.append(trade)
        del self._open[ticker]
        self._save()

        logger.info("SWING CLOSED: %s | hold=%dd | net=%.2f%% | reason=%s",
                    ticker, hold_days, net_pnl*100, exit_reason)

    def sync_with_portfolio(self, new_portfolio: list[dict],
                             current_prices: dict[str, float]):
        """
        Called each rebalance. Opens new positions, closes removed ones.
        Returns {ticker: exit_reason} for force-closed positions.
        """
        new_tickers = {p["ticker"] for p in new_portfolio}

        # Check time/price exits first
        exits = self.check_exits(current_prices)

        # Also close any position no longer in the new portfolio
        for ticker in list(self._open.keys()):
            if ticker not in new_tickers and ticker not in exits:
                exits[ticker] = "rebalance_removed"

        # Execute exits
        for ticker, reason in exits.items():
            price = float(current_prices.get(ticker, 0))
            if price > 0:
                self.close_position(ticker, price, reason)

        # Open new positions
        for p in new_portfolio:
            t = p["ticker"]
            if t not in self._open:
                price = float(current_prices.get(t, 0) or p.get("price", 0))
                if price > 0:
                    self.open_position(
                        t, price, p.get("weight", 0),
                        regime=p.get("regime", "neutral"),
                        signal_score=p.get("score", 0),
                    )

        return exits

    # ── Swing metrics ─────────────────────────────────────────────────────────

    def compute_metrics(self, last_n: int = 100) -> dict:
        """
        Compute swing-specific performance metrics from closed trades.
        """
        trades = self._closed[-last_n:] if self._closed else []
        if not trades:
            return {"n_trades": 0, "status": "no closed trades yet"}

        n         = len(trades)
        net_pnls  = [t["net_pnl"]  for t in trades]
        hold_days = [t["hold_days"] for t in trades]
        wins      = [t for t in trades if t["win"]]
        losses    = [t for t in trades if not t["win"]]

        # Win rate by hold bucket
        buckets = {
            "<7d":    [t for t in trades if t["hold_days"] < 7],
            "7-30d":  [t for t in trades if 7 <= t["hold_days"] < 30],
            "30-60d": [t for t in trades if t["hold_days"] >= 30],
        }
        win_rate_by_bucket = {
            b: {
                "n":        len(ts),
                "win_rate": round(float(np.mean([t["win"] for t in ts])), 3) if ts else 0,
                "avg_pnl":  round(float(np.mean([t["net_pnl"] for t in ts]))*100, 2) if ts else 0,
            }
            for b, ts in buckets.items()
        }

        # Profit factor
        gross_wins   = sum(t["gross_pnl"] for t in wins)
        gross_losses = abs(sum(t["gross_pnl"] for t in losses))
        profit_factor = gross_wins / (gross_losses + 1e-9)

        # Max consecutive losses
        result_seq = [1 if t["win"] else 0 for t in trades]
        max_consec_loss, cur = 0, 0
        for r in result_seq:
            if r == 0:
                cur += 1
                max_consec_loss = max(max_consec_loss, cur)
            else:
                cur = 0

        # Exit reason breakdown
        reasons = {}
        for t in trades:
            r = t.get("exit_reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1

        # Deployment verdict
        win_rate = float(np.mean([t["win"] for t in trades]))
        avg_hold = float(np.mean(hold_days))
        if win_rate >= 0.55 and profit_factor >= 1.3 and n >= 20:
            verdict = "STRONG — signal working, consider scaling capital"
        elif win_rate >= 0.52 and profit_factor >= 1.1 and n >= 10:
            verdict = "POSITIVE — early evidence of edge, continue paper trading"
        elif n < 10:
            verdict = f"EARLY — {n} trades, need ≥10 to assess"
        else:
            verdict = "WEAK — no clear edge yet, review signal weights"

        return {
            "n_trades":          n,
            "win_rate":          round(win_rate, 3),
            "avg_hold_days":     round(avg_hold, 1),
            "median_hold_days":  round(float(np.median(hold_days)), 1),
            "profit_factor":     round(profit_factor, 3),
            "avg_net_pnl_pct":   round(float(np.mean(net_pnls))*100, 3),
            "avg_win_pct":       round(float(np.mean([t["net_pnl"] for t in wins]))*100, 3) if wins else 0,
            "avg_loss_pct":      round(float(np.mean([t["net_pnl"] for t in losses]))*100, 3) if losses else 0,
            "max_consec_losses": max_consec_loss,
            "win_rate_by_bucket":win_rate_by_bucket,
            "exit_reasons":      reasons,
            "verdict":           verdict,
            "open_positions":    len(self._open),
        }

    def summary_log(self) -> str:
        """One-line summary for logging."""
        m = self.compute_metrics()
        if m["n_trades"] == 0:
            return "No swing trades closed yet"
        return (f"Swing: {m['n_trades']} trades | WR={m['win_rate']*100:.1f}% | "
                f"PF={m['profit_factor']:.2f} | AvgHold={m['avg_hold_days']:.1f}d | "
                f"AvgPnL={m['avg_net_pnl_pct']:+.2f}% | {m['verdict']}")
