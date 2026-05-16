"""
FIFO Tax Lot Tracker.

Maintains a running ledger of all purchases (lots) per ticker.
Supports FIFO, LIFO, and specific-identification cost basis methods.

Tracks:
  - Cost basis per lot
  - Holding period (short-term vs long-term: >365 days)
  - Realized gains/losses
  - Wash sale detection (buy within 30 days of selling at a loss)
"""
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Literal

logger = logging.getLogger(__name__)

_LOTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "state", "tax_lots.json"
)

CostMethod = Literal["fifo", "lifo", "specific"]


class TaxLotLedger:
    """
    Append-only tax lot ledger.

    Each lot: {ticker, lot_id, open_date, shares, cost_basis, closed, close_date, proceeds}
    """

    def __init__(self, method: CostMethod = "fifo"):
        self.method = method
        self._lots: list[dict] = []
        self._realized: list[dict] = []
        self._load()

    def _load(self):
        if not os.path.exists(_LOTS_PATH):
            return
        try:
            with open(_LOTS_PATH) as f:
                data = json.load(f)
            self._lots     = data.get("lots", [])
            self._realized = data.get("realized", [])
        except Exception as e:
            logger.warning("Tax lot load: %s", e)

    def _save(self):
        os.makedirs(os.path.dirname(_LOTS_PATH), exist_ok=True)
        with open(_LOTS_PATH, "w") as f:
            json.dump({"lots": self._lots, "realized": self._realized},
                      f, indent=2, default=str)

    def add_lot(self, ticker: str, shares: float, cost_per_share: float,
                trade_date: str | None = None) -> str:
        """Record a purchase. Returns lot_id."""
        lot_id = f"{ticker}_{len(self._lots):06d}"
        self._lots.append({
            "lot_id":         lot_id,
            "ticker":         ticker,
            "shares":         float(shares),
            "cost_per_share": float(cost_per_share),
            "cost_basis":     round(float(shares) * float(cost_per_share), 4),
            "open_date":      trade_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "closed":         False,
            "close_date":     None,
            "proceeds":       None,
        })
        self._save()
        logger.debug("Lot opened: %s %s %.2f sh @ $%.2f",
                     lot_id, ticker, shares, cost_per_share)
        return lot_id

    def close_lots(self, ticker: str, shares_to_sell: float,
                   price_per_share: float, trade_date: str | None = None) -> dict:
        """
        Close lots for a sell order using the configured cost method.
        Returns summary of realized gains/losses.
        """
        date_str = trade_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        open_lots = [l for l in self._lots
                     if l["ticker"] == ticker and not l["closed"] and l["shares"] > 0]

        if not open_lots:
            return {"realized_gain": 0, "short_term": 0, "long_term": 0, "lots_closed": []}

        # Sort by method
        if self.method == "fifo":
            open_lots.sort(key=lambda l: l["open_date"])
        elif self.method == "lifo":
            open_lots.sort(key=lambda l: l["open_date"], reverse=True)

        total_realized   = 0.0
        short_term_gain  = 0.0
        long_term_gain   = 0.0
        remaining        = float(shares_to_sell)
        closed_lot_ids   = []
        wash_sale_risk   = False

        for lot in open_lots:
            if remaining <= 0:
                break

            sell_shares = min(lot["shares"], remaining)
            proceeds    = sell_shares * price_per_share
            cost        = sell_shares * lot["cost_per_share"]
            gain        = proceeds - cost

            # Holding period classification
            open_dt  = datetime.strptime(lot["open_date"], "%Y-%m-%d")
            close_dt = datetime.strptime(date_str, "%Y-%m-%d")
            held_days = (close_dt - open_dt).days
            is_long_term = held_days > 365

            if is_long_term:
                long_term_gain  += gain
            else:
                short_term_gain += gain

                # Wash sale risk: selling at a loss within 30 days of buying
                if gain < 0 and held_days < 30:
                    wash_sale_risk = True
                    logger.warning("WASH SALE RISK: %s lot %s (held %d days, loss=$%.2f)",
                                   ticker, lot["lot_id"], held_days, gain)

            total_realized += gain

            # Update lot
            lot["shares"] -= sell_shares
            if lot["shares"] < 0.001:
                lot["closed"]     = True
                lot["close_date"] = date_str
                lot["proceeds"]   = round(proceeds, 4)
                closed_lot_ids.append(lot["lot_id"])

            remaining -= sell_shares

            self._realized.append({
                "lot_id":         lot["lot_id"],
                "ticker":         ticker,
                "shares_sold":    round(sell_shares, 6),
                "cost_per_share": lot["cost_per_share"],
                "proceeds_ps":    round(price_per_share, 4),
                "gain":           round(gain, 4),
                "held_days":      held_days,
                "is_long_term":   is_long_term,
                "close_date":     date_str,
                "wash_sale_risk": wash_sale_risk,
            })

        self._save()

        return {
            "realized_gain":  round(total_realized, 4),
            "short_term":     round(short_term_gain, 4),
            "long_term":      round(long_term_gain, 4),
            "lots_closed":    closed_lot_ids,
            "wash_sale_risk": wash_sale_risk,
        }

    def tax_summary(self, tax_year: int | None = None) -> dict:
        """
        Generate annual realized gain/loss summary for tax reporting.
        If tax_year is None, uses current year.
        """
        year = tax_year or datetime.now().year
        year_realized = [
            r for r in self._realized
            if r.get("close_date", "").startswith(str(year))
        ]

        st = sum(r["gain"] for r in year_realized if not r["is_long_term"])
        lt = sum(r["gain"] for r in year_realized if r["is_long_term"])
        wash = sum(1 for r in year_realized if r.get("wash_sale_risk"))

        return {
            "tax_year":           year,
            "short_term_gain":    round(st, 2),
            "long_term_gain":     round(lt, 2),
            "total_realized":     round(st + lt, 2),
            "wash_sale_count":    wash,
            "n_transactions":     len(year_realized),
        }

    def open_positions(self) -> dict:
        """Return current open lots summary per ticker."""
        by_ticker: dict[str, dict] = {}
        for lot in self._lots:
            if lot["closed"] or lot["shares"] < 0.001:
                continue
            t = lot["ticker"]
            if t not in by_ticker:
                by_ticker[t] = {"shares": 0.0, "cost_basis": 0.0, "avg_cost": 0.0, "lots": 0}
            by_ticker[t]["shares"]     += lot["shares"]
            by_ticker[t]["cost_basis"] += lot["shares"] * lot["cost_per_share"]
            by_ticker[t]["lots"]       += 1

        for t, v in by_ticker.items():
            v["avg_cost"] = round(v["cost_basis"] / max(v["shares"], 1e-6), 4)
            v["shares"]   = round(v["shares"], 6)

        return by_ticker
