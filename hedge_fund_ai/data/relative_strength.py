def relative_strength(stock_ret, market_ret):
    if market_ret is None:
        return 0
    return (stock_ret or 0) - market_ret

