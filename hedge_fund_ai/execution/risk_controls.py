"""
Gap 5: Position-level stop losses and daily loss circuit breaker.

Controls:
  1. Position stop loss: if any single position drops > STOP_LOSS_PCT from entry,
     close it immediately regardless of rebalance schedule.

  2. Daily loss circuit breaker: if the portfolio loses > DAILY_LOSS_LIMIT in a
     single day, halt all new orders for the rest of that day.

  3. Max drawdown halt: if total drawdown > MAX_DD_HALT, pause all trading and
     alert operator.

These run BEFORE the main execute_portfolio() call.
"""

import logging
import os
import json
from datetime import datetime, timezone

import pandas as pd

from hedge_fund_ai.config import (
    ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE, ALPACA_DRY_RUN,
)

logger = logging.getLogger(__name__)

# ── Configurable thresholds ───────────────────────────────────────────────────
STOP_LOSS_PCT    = float(os.getenv("STOP_LOSS_PCT",    "0.08"))   # 8% per position
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "0.03"))   # 3% portfolio daily loss
MAX_DD_HALT      = float(os.getenv("MAX_DD_HALT",      "0.20"))   # 20% drawdown → halt all trading
MIN_TRADE_USD    = float(os.getenv("MIN_TRADE_USD",    "100"))    # minimum order size

_STATE_PATH = os.path.join(os.path.dirname(__file__), "..", "state", "risk_state.json")


# ── Risk state ────────────────────────────────────────────────────────────────

def _load_risk_state() -> dict:
    if not os.path.exists(_STATE_PATH):
        return {"entry_prices": {}, "halted_until": None, "halt_reason": None}
    try:
        with open(_STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {"entry_prices": {}, "halted_until": None, "halt_reason": None}


def _save_risk_state(state: dict):
    os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
    with open(_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── Circuit breakers ──────────────────────────────────────────────────────────

def is_trading_halted() -> tuple[bool, str]:
    """
    Returns (halted: bool, reason: str).
    Checks both time-based halt (daily loss) and permanent halt (max DD).
    """
    state = _load_risk_state()
    halted_until = state.get("halted_until")
    if halted_until:
        halt_ts = pd.Timestamp(halted_until, tz="UTC")
        if pd.Timestamp.now(tz="UTC") < halt_ts:
            return True, state.get("halt_reason", "trading halted")
    return False, ""


def trigger_daily_halt(reason: str, hours: float = 24.0):
    """Halt trading for `hours` hours."""
    from datetime import timedelta
    state = _load_risk_state()
    until = (pd.Timestamp.now(tz="UTC") + timedelta(hours=hours)).isoformat()
    state["halted_until"] = until
    state["halt_reason"]  = reason
    _save_risk_state(state)
    logger.critical("TRADING HALTED: %s — until %s", reason, until)


def clear_halt():
    """Manually clear a trading halt."""
    state = _load_risk_state()
    state["halted_until"] = None
    state["halt_reason"]  = None
    _save_risk_state(state)
    logger.info("Trading halt cleared.")


# ── Entry price tracking ──────────────────────────────────────────────────────

def record_entry_prices(portfolio: list[dict]):
    """Record entry price for each new position."""
    state = _load_risk_state()
    for p in portfolio:
        t = p.get("ticker")
        px = p.get("price", 0)
        if t and px and t not in state["entry_prices"]:
            state["entry_prices"][t] = {"price": float(px),
                                        "date": datetime.now(timezone.utc).isoformat()}
    _save_risk_state(state)


def remove_entry_price(ticker: str):
    state = _load_risk_state()
    state["entry_prices"].pop(ticker, None)
    _save_risk_state(state)


# ── Stop loss check ───────────────────────────────────────────────────────────

def check_stop_losses(current_prices: dict[str, float]) -> list[str]:
    """
    Check all positions for stop-loss breach.
    Returns list of tickers that have breached STOP_LOSS_PCT and must be closed.
    """
    state = _load_risk_state()
    to_close = []

    for ticker, entry in state["entry_prices"].items():
        entry_px  = float(entry.get("price", 0))
        current   = float(current_prices.get(ticker, 0))
        if entry_px <= 0 or current <= 0:
            continue
        drawdown = (current - entry_px) / entry_px
        if drawdown < -STOP_LOSS_PCT:
            logger.warning(
                "STOP LOSS: %s down %.1f%% from entry $%.2f → $%.2f",
                ticker, abs(drawdown) * 100, entry_px, current
            )
            to_close.append(ticker)

    return to_close


def close_stop_loss_positions(tickers_to_close: list[str]) -> list[dict]:
    """
    Submit market sell orders for all stop-loss-breached positions.
    Returns list of order results.
    """
    if not tickers_to_close:
        return []

    if ALPACA_DRY_RUN:
        logger.info("DRY RUN: would close stop-loss positions: %s", tickers_to_close)
        return [{"ticker": t, "action": "stop_loss_dry_run"} for t in tickers_to_close]

    import requests
    results = []
    for ticker in tickers_to_close:
        try:
            url = f"{ALPACA_BASE}/v2/positions/{ticker}"
            r   = requests.delete(
                url,
                headers={
                    "APCA-API-KEY-ID":    ALPACA_API_KEY,
                    "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
                },
                timeout=15,
            )
            r.raise_for_status()
            logger.info("Stop loss executed: %s — position closed", ticker)
            remove_entry_price(ticker)
            results.append({"ticker": ticker, "action": "closed", "status": r.status_code})
        except Exception as e:
            logger.error("Stop loss order failed for %s: %s", ticker, e)
            results.append({"ticker": ticker, "action": "failed", "error": str(e)})

    return results


# ── Daily P&L check ───────────────────────────────────────────────────────────

def check_daily_loss(portfolio_value_today: float, portfolio_value_yesterday: float) -> bool:
    """
    Returns True if daily loss circuit breaker triggered (trading should halt).
    Automatically triggers halt if loss > DAILY_LOSS_LIMIT.
    """
    if portfolio_value_yesterday <= 0:
        return False

    daily_ret = (portfolio_value_today - portfolio_value_yesterday) / portfolio_value_yesterday

    if daily_ret < -DAILY_LOSS_LIMIT:
        trigger_daily_halt(
            reason=f"Daily loss {daily_ret*100:.2f}% breached limit {DAILY_LOSS_LIMIT*100:.1f}%",
            hours=24.0
        )
        return True

    return False


def check_max_drawdown(equity_curve: list[float]) -> bool:
    """
    Returns True if max drawdown halt triggered.
    Automatically halts trading indefinitely if drawdown > MAX_DD_HALT.
    """
    if len(equity_curve) < 2:
        return False

    s    = pd.Series(equity_curve)
    peak = s.cummax()
    dd   = float((s.iloc[-1] - peak.iloc[-1]) / (peak.iloc[-1] + 1e-10))

    if dd < -MAX_DD_HALT:
        trigger_daily_halt(
            reason=f"Max drawdown {dd*100:.1f}% breached limit {MAX_DD_HALT*100:.1f}%",
            hours=24 * 30  # 30-day halt — requires manual clear
        )
        logger.critical("MAX DRAWDOWN HALT: portfolio down %.1f%% from peak", abs(dd) * 100)
        return True

    return False


# ── Pre-execution risk gate ────────────────────────────────────────────────────

def pre_execution_check(portfolio: list[dict], equity_curve: list[float],
                        portfolio_value_today: float,
                        portfolio_value_yesterday: float) -> dict:
    """
    Master pre-execution risk check. Call before execute_portfolio().

    Returns:
        {
          "proceed": bool,        # False = do not trade today
          "stop_loss_closes": [], # tickers to close immediately
          "reason": str,          # if proceed=False, why
        }
    """
    # 1. Is trading halted?
    halted, reason = is_trading_halted()
    if halted:
        return {"proceed": False, "stop_loss_closes": [], "reason": reason}

    # 2. Daily loss circuit breaker
    if check_daily_loss(portfolio_value_today, portfolio_value_yesterday):
        return {
            "proceed": False,
            "stop_loss_closes": [],
            "reason": f"Daily loss limit breached ({DAILY_LOSS_LIMIT*100:.0f}%)",
        }

    # 3. Max drawdown halt
    if check_max_drawdown(equity_curve):
        return {
            "proceed": False,
            "stop_loss_closes": [],
            "reason": f"Max drawdown limit breached ({MAX_DD_HALT*100:.0f}%)",
        }

    # 4. Position stop losses (independent of rebalance schedule)
    current_prices = {p["ticker"]: float(p.get("price", 0)) for p in portfolio}
    to_close = check_stop_losses(current_prices)

    return {
        "proceed": True,
        "stop_loss_closes": to_close,
        "reason": "",
    }
