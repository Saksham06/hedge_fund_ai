def select_top(data, n=10, key="score"):
    ranked = sorted(data, key=lambda x: x.get(key, x.get("signal", 0)), reverse=True)
    return [d["ticker"] for d in ranked[:n]]
