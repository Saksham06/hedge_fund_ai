import json
import os
from datetime import datetime
from hedge_fund_ai.config import PORTFOLIO_NOTIONAL_USD

STATE_FILE = os.path.join(os.path.dirname(__file__), "performance.json")


def load_performance():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return []


def save_performance(data):
    json.dump(data, open(STATE_FILE, "w"), indent=2)


def log_portfolio(portfolio, prices):

    perf = load_performance()

    value = 0.0
    for p in portfolio:
        ticker = p["ticker"]
        weight = float(p["weight"])
        price = float(prices.get(ticker, 0) or 0)
        target_dollars = PORTFOLIO_NOTIONAL_USD * weight
        shares = (target_dollars / price) if price > 0 else 0.0
        value += shares * price

    entry = {
        "timestamp": datetime.now().isoformat(),
        "portfolio_value": round(value, 2),
    }

    perf.append(entry)
    save_performance(perf)

    return entry
