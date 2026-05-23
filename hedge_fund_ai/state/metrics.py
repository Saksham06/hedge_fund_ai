import numpy as np


def compute_metrics(perf):

    if len(perf) < 2:
        return {}

    values = [p["portfolio_value"] for p in perf]

    returns = np.diff(values) / values[:-1]

    sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)

    peak = values[0]
    drawdowns = []

    for v in values:
        peak = max(peak, v)
        dd = (v - peak) / peak
        drawdowns.append(dd)

    max_dd = min(drawdowns)

    return {
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(float(max_dd), 3),
        "total_return": round((values[-1] / values[0] - 1), 3),
    }
