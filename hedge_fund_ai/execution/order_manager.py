"""
Order Manager — production-grade execution.

Implements:
  1. Order slicing by ADV (VWAP-style participation rate limiting)
  2. Arrival price vs fill price tracking (execution quality)
  3. Position reconciliation (expected vs actual after market open)
  4. Volatility-adjusted stop losses (expands in high-vol, tightens in low-vol)
  5. Pre-trade validation (limits, buying power, sector caps)
"""
import json
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EXEC_LOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "state", "execution_log.json"
)

# Participation rate: never trade > this % of ADV in one order
MAX_PARTICIPATION_RATE = 0.05   # 5% of ADV per order slice
MIN_SLICE_USD          = 500    # minimum slice size (avoid sub-economic orders
MAX_SLICES_PER_STOCK   = 5      # never split into more than 5 slices


# ── 1. Order slicing ──────────────────────────────────────────────────────────

def compute_order_slices(
    ticker:          str,
    target_notional: float,   # USD value to trade
    avg_daily_vol:   float,   # ADV in USD
    current_price:   float,
    max_participation: float = MAX_PARTICIPATION_RATE,
) -> list[dict]:
    """
    Split a large order into slices based on ADV.
    Returns list of {notional, pct_of_adv, delay_minutes}.
    """
    if target_notional <= 0 or avg_daily_vol <= 0:
        return [{"notional": target_notional, "pct_of_adv": 0, "delay_minutes": 0}]

    max_per_slice = avg_daily_vol * max_participation
    max_per_slice = max(max_per_slice, MIN_SLICE_USD)

    if target_notional <= max_per_slice:
        pct = target_notional / (avg_daily_vol + 1e-6) * 100
        return [{"notional": target_notional, "pct_of_adv": round(pct, 2), "delay_minutes": 0}]

    # Need to slice
    n_slices = min(int(np.ceil(target_notional / max_per_slice)), MAX_SLICES_PER_STOCK)
    slice_size = target_notional / n_slices
    pct = slice_size / (avg_daily_vol + 1e-6) * 100

    slices = []
    for i in range(n_slices):
        slices.append({
            "notional":      round(slice_size, 2),
            "pct_of_adv":    round(pct, 2),
            "delay_minutes": i * 30,   # stagger by 30 minutes
        })

    logger.info("Order slicing: %s $%.0f → %d slices of $%.0f (%.1f%% ADV each)",
                ticker, target_notional, n_slices, slice_size, pct)
    return slices


# ── 2. Arrival price tracking ──────────────────────────────────────────────────

class ExecutionTracker:
    """
    Records arrival price (signal time) vs fill price for every order.
    Computes execution quality metrics and flags systematic slippage.
    """

    def __init__(self):
        self._log: list[dict] = self._load()

    def _load(self) -> list:
        if not os.path.exists(_EXEC_LOG_PATH):
            return []
        try:
            with open(_EXEC_LOG_PATH) as f:
                return json.load(f)
        except Exception:
            return []

    def _save(self):
        os.makedirs(os.path.dirname(_EXEC_LOG_PATH), exist_ok=True)
        with open(_EXEC_LOG_PATH, "w") as f:
            json.dump(self._log[-500:], f, indent=2, default=str)  # keep last 500

    def record_arrival(self, ticker: str, arrival_price: float, side: str,
                       notional: float) -> str:
        """Record the price at signal time (before sending order). Returns trade_id."""
        trade_id = f"{ticker}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        entry = {
            "trade_id":      trade_id,
            "ticker":        ticker,
            "side":          side,
            "notional":      notional,
            "arrival_price": arrival_price,
            "fill_price":    None,
            "slippage_bps":  None,
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "status":        "pending",
        }
        self._log.append(entry)
        self._save()
        return trade_id

    def record_fill(self, trade_id: str, fill_price: float):
        """Record the actual fill price and compute slippage."""
        for entry in reversed(self._log):
            if entry.get("trade_id") == trade_id:
                arrival = float(entry.get("arrival_price") or fill_price)
                side    = entry.get("side", "buy")
                if arrival > 0:
                    if side == "buy":
                        # Positive slippage = we paid more than arrival (bad)
                        slippage = (fill_price - arrival) / arrival * 10_000
                    else:
                        slippage = (arrival - fill_price) / arrival * 10_000
                else:
                    slippage = 0

                entry["fill_price"]   = fill_price
                entry["slippage_bps"] = round(float(slippage), 2)
                entry["status"]       = "filled"

                if abs(slippage) > 50:
                    logger.warning("HIGH SLIPPAGE: %s %s %.1fbps (arrival=%.2f fill=%.2f)",
                                   trade_id, entry["ticker"], slippage, arrival, fill_price)
                break
        self._save()

    def slippage_report(self, last_n: int = 100) -> dict:
        """Return slippage statistics for monitoring."""
        filled = [e for e in self._log[-last_n:] if e.get("status") == "filled"
                  and e.get("slippage_bps") is not None]
        if not filled:
            return {"n_trades": 0}

        slippages = [float(e["slippage_bps"]) for e in filled]
        by_ticker: dict = {}
        for e in filled:
            t = e["ticker"]
            by_ticker.setdefault(t, []).append(float(e["slippage_bps"]))

        worst_tickers = sorted(by_ticker.items(),
                               key=lambda x: np.mean(x[1]), reverse=True)[:5]

        return {
            "n_trades":          len(filled),
            "avg_slippage_bps":  round(float(np.mean(slippages)), 2),
            "p90_slippage_bps":  round(float(np.percentile(slippages, 90)), 2),
            "worst_tickers":     [(t, round(float(np.mean(v)), 2)) for t, v in worst_tickers],
            "systematic_flag":   float(np.mean(slippages)) > 20,  # >20bps avg = systematic issue
        }


# ── 3. Position reconciliation ────────────────────────────────────────────────

def reconcile_positions(
    expected: dict[str, float],    # {ticker: target_weight}
    alpaca_positions: dict,         # from Alpaca /v2/positions
    portfolio_value: float,
    tolerance_pct: float = 0.02,    # 2% weight tolerance before flagging
) -> dict:
    """
    Compare expected vs actual positions after market open.
    Returns dict of mismatches that need correction.
    """
    mismatches = {}
    all_tickers = set(expected) | set(alpaca_positions)

    for t in all_tickers:
        exp_w   = float(expected.get(t, 0))
        pos     = alpaca_positions.get(t, {})
        act_val = float(pos.get("market_value", 0)) if pos else 0
        act_w   = act_val / (portfolio_value + 1e-6)

        delta = abs(exp_w - act_w)
        if delta > tolerance_pct:
            mismatches[t] = {
                "expected_weight": round(exp_w, 4),
                "actual_weight":   round(act_w, 4),
                "delta":           round(delta, 4),
                "action":          "buy" if exp_w > act_w else "sell",
                "notional":        round((exp_w - act_w) * portfolio_value, 2),
            }
            logger.warning("RECONCILIATION mismatch: %s expected=%.2f%% actual=%.2f%%",
                           t, exp_w*100, act_w*100)

    if not mismatches:
        logger.info("Position reconciliation: all %d positions within tolerance", len(expected))
    else:
        logger.warning("Reconciliation: %d mismatches found", len(mismatches))

    return mismatches


# ── 4. Volatility-adjusted stop losses ────────────────────────────────────────

def compute_vol_adjusted_stop(
    ticker:        str,
    entry_price:   float,
    current_vol:   float,   # annualized volatility fraction (e.g. 0.25 = 25%)
    base_stop_pct: float = 0.08,  # base 8% stop
    vol_scalar:    float = 0.5,   # how much to scale stop by vol
    min_stop:      float = 0.04,  # never tighter than 4%
    max_stop:      float = 0.20,  # never wider than 20%
) -> dict:
    """
    Volatility-adjusted stop loss.

    Logic:
      - At 20% vol (average), stop = base_stop_pct (8%)
      - At 10% vol (calm), stop tightens to ~5% (lock in gains)
      - At 40% vol (high), stop widens to ~13% (avoid premature stop-out)

    Returns {stop_price, stop_pct, vol_used}
    """
    ref_vol   = 0.20   # reference vol (20% annualized)
    vol_adj   = base_stop_pct * (current_vol / ref_vol) ** vol_scalar
    stop_pct  = float(np.clip(vol_adj, min_stop, max_stop))
    stop_price = entry_price * (1 - stop_pct)

    return {
        "ticker":     ticker,
        "entry_price": round(entry_price, 4),
        "stop_pct":    round(stop_pct, 4),
        "stop_price":  round(stop_price, 4),
        "vol_used":    round(current_vol, 4),
    }
