import numpy as np
import pandas as pd
import yfinance as yf


def apply_costs(returns, cost_per_trade=0.001):
    return returns - cost_per_trade


def run_backtest(tickers, rebalance_days=21, start="2022-01-01"):

    prices = yf.download(tickers, start=start, progress=False)["Close"]

    returns = prices.pct_change().dropna()

    dates = returns.index

    portfolio_returns = []

    if isinstance(returns, pd.Series):
        n_assets = 1
    else:
        n_assets = int(returns.shape[1])

    weights = np.ones(n_assets) / n_assets

    for i in range(len(dates)):

        rebalanced = False
        if i % rebalance_days == 0:
            weights = np.ones(n_assets) / n_assets
            rebalanced = i != 0

        row = returns.iloc[i]
        row_values = row.values if hasattr(row, "values") and getattr(row, "ndim", 0) else np.array([float(row)])

        daily_ret = float(np.dot(row_values, weights))

        if rebalanced:
            daily_ret = float(apply_costs(daily_ret))

        portfolio_returns.append(daily_ret)

    equity = pd.Series(portfolio_returns, index=dates)
    equity = (1 + equity).cumprod()

    return equity
