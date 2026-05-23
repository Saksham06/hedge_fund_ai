def compute_turnover(prev_portfolio, new_portfolio):

    prev = {p["ticker"]: float(p["weight"]) for p in (prev_portfolio or [])}
    new = {p["ticker"]: float(p["weight"]) for p in (new_portfolio or [])}

    tickers = set(prev) | set(new)

    turnover = 0.0

    for t in tickers:
        turnover += abs(new.get(t, 0.0) - prev.get(t, 0.0))

    return turnover / 2.0  # standard definition

