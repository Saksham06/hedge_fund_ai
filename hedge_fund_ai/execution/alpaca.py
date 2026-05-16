"""
Alpaca execution engine — production grade.
Integrates with risk_controls pre-execution gate.
"""
import logging
import time
from datetime import datetime, timezone

import requests

from hedge_fund_ai.config import (
    ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE,
    ALPACA_DRY_RUN, ALPACA_FRACTIONAL, ALPACA_MAX_ORDER_USD,
    ALPACA_PAPER, PORTFOLIO_NOTIONAL_USD,
)

logger = logging.getLogger(__name__)


def _headers() -> dict:
    return {
        "APCA-API-KEY-ID":     ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type":        "application/json",
    }


def _get(path: str) -> dict:
    r = requests.get(f"{ALPACA_BASE}{path}", headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def is_market_open() -> bool:
    try:
        return bool(_get("/v2/clock").get("is_open", False))
    except Exception as e:
        logger.warning("Market open check failed: %s", e)
        return True


def get_account() -> dict:
    return _get("/v2/account")


def get_positions() -> dict:
    try:
        return {p["symbol"]: p for p in _get("/v2/positions")}
    except Exception:
        return {}


def place_order(symbol: str, notional: float = None, qty: float = None,
                side: str = "buy", retry: int = 2) -> dict:
    url = f"{ALPACA_BASE}/v2/orders"
    order = (
        {"symbol": symbol, "notional": round(float(notional), 2),
         "side": side, "type": "market", "time_in_force": "day"}
        if (notional is not None and ALPACA_FRACTIONAL)
        else {"symbol": symbol, "qty": int(qty or 1),
              "side": side, "type": "market", "time_in_force": "day"}
    )
    last_exc = None
    for attempt in range(retry + 1):
        try:
            r = requests.post(url, json=order, headers=_headers(), timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                time.sleep(2 ** attempt)
                last_exc = e
            else:
                raise
        except Exception as e:
            last_exc = e
            time.sleep(1)
    raise RuntimeError(f"Order failed after {retry} retries: {last_exc}")


def execute_portfolio(portfolio: list, equity_curve: list = None) -> dict:
    """
    Full execution with pre-flight risk checks:
      1. Stop-loss checks (close breached positions first)
      2. Daily loss circuit breaker
      3. Max drawdown halt
      4. Market hours check
      5. Reconcile vs current positions
      6. Submit delta orders
    """
    if ALPACA_DRY_RUN:
        logger.info("ALPACA_DRY_RUN=true — skipping live execution")
        return {"dry_run": True, "orders": [], "stop_loss_closes": []}

    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        raise RuntimeError("Missing Alpaca credentials")

    # ── Pre-execution risk gate ────────────────────────────────────────────────
    from hedge_fund_ai.execution.risk_controls import (
        pre_execution_check, close_stop_loss_positions, record_entry_prices
    )

    account = get_account()
    portfolio_value = float(account.get("portfolio_value", PORTFOLIO_NOTIONAL_USD))
    prev_value = float(account.get("last_equity", portfolio_value))

    risk_check = pre_execution_check(
        portfolio            = portfolio,
        equity_curve         = equity_curve or [1.0],
        portfolio_value_today     = portfolio_value,
        portfolio_value_yesterday = prev_value,
    )

    # Close stop-loss positions regardless of whether we proceed
    sl_results = []
    if risk_check["stop_loss_closes"]:
        logger.warning("Stop losses triggered: %s", risk_check["stop_loss_closes"])
        sl_results = close_stop_loss_positions(risk_check["stop_loss_closes"])

    if not risk_check["proceed"]:
        logger.warning("Execution halted: %s", risk_check["reason"])
        return {
            "dry_run":          False,
            "halted":           True,
            "reason":           risk_check["reason"],
            "stop_loss_closes": sl_results,
            "orders":           [],
            "errors":           [],
        }

    if not is_market_open():
        logger.warning("Market closed — skipping execution")
        return {"dry_run": False, "skipped": True, "reason": "market_closed",
                "stop_loss_closes": sl_results, "orders": [], "errors": []}

    if not ALPACA_PAPER:
        confirm = input("\n⚠️  LIVE TRADING. Type 'YES' to confirm: ")
        if confirm != "YES":
            raise RuntimeError("Live trading not confirmed")

    mode = "PAPER" if ALPACA_PAPER else "LIVE"
    logger.info("Executing portfolio — %s mode (%d positions)", mode, len(portfolio))

    current_positions = get_positions()
    orders, errors = [], []

    for p in portfolio:
        symbol = p["ticker"]
        target  = portfolio_value * float(p["weight"])
        target  = min(target, ALPACA_MAX_ORDER_USD)
        cur_val = float(current_positions.get(symbol, {}).get("market_value", 0))
        delta   = target - cur_val

        if abs(delta) < 100:
            continue

        side = "buy" if delta > 0 else "sell"
        try:
            result = place_order(symbol, notional=abs(delta), side=side)
            orders.append(result)
            logger.info("Order %s %s $%.0f", side.upper(), symbol, abs(delta))
        except Exception as e:
            logger.error("Order failed %s: %s", symbol, e)
            errors.append({"symbol": symbol, "error": str(e)})

    # Record entry prices for new buys (for stop-loss tracking)
    record_entry_prices(portfolio)

    return {
        "dry_run":          False,
        "mode":             mode,
        "orders":           orders,
        "errors":           errors,
        "stop_loss_closes": sl_results,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }
