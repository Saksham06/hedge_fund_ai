def slippage(price, volatility):
    # higher vol -> higher slippage
    slip_pct = min(0.002 + (volatility or 0) / 1000, 0.01)
    return price * slip_pct

