import json
import os

STATE_FILE = os.path.join(os.path.dirname(__file__), "macro_history.json")


def load_macro_history(limit=260):
    if not os.path.exists(STATE_FILE):
        return []
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f) or []
            return data[-limit:]
    except Exception:
        return []


def append_macro_snapshot(macro_snapshot, limit=260):
    history = load_macro_history(limit=limit)
    history.append(macro_snapshot or {})
    history = history[-limit:]
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return history

