import json
import os
import pandas as pd

STATE_DIR = os.path.dirname(__file__)
EQUITY_PATH = os.path.join(STATE_DIR, "backtest_equity.json")


def save_equity_curve(equity: pd.Series):
    if equity is None or equity.empty:
        return
    data = {
        "dates":  [str(d) for d in equity.index],
        "values": list(equity.values),
    }
    with open(EQUITY_PATH, "w") as f:
        json.dump(data, f)


def load_equity_curve() -> list | None:
    if not os.path.exists(EQUITY_PATH):
        return None
    try:
        with open(EQUITY_PATH) as f:
            data = json.load(f)
        return data.get("values")
    except Exception:
        return None
