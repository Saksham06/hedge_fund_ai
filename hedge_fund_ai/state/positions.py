import json
import os
from datetime import datetime, timezone

STATE_FILE = os.path.join(os.path.dirname(__file__), "positions.json")


def load_prev_portfolio():
    if not os.path.exists(STATE_FILE):
        return []
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
            positions = payload.get("positions")
            if isinstance(positions, list):
                return positions
            if isinstance(payload, list):
                return payload
            return []
    except Exception:
        return []


def save_positions(weights, prices, sectors):
    data = {"timestamp": datetime.now(timezone.utc).isoformat(), "positions": []}

    for t, w in (weights or {}).items():
        data["positions"].append(
            {
                "ticker": t,
                "weight": float(w),
                "price": (prices or {}).get(t),
                "sector": (sectors or {}).get(t),
            }
        )

    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_portfolio(portfolio):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        weights = {p["ticker"]: float(p["weight"]) for p in (portfolio or []) if p.get("ticker")}
        prices = {p["ticker"]: p.get("price") for p in (portfolio or []) if p.get("ticker")}
        sectors = {p["ticker"]: p.get("sector") for p in (portfolio or []) if p.get("ticker")}
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "positions": [
                    {"ticker": t, "weight": weights[t], "price": prices.get(t), "sector": sectors.get(t)}
                    for t in weights
                ],
            },
            f,
            indent=2,
        )
