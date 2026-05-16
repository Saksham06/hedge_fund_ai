import numpy as np


def risk_parity_weights(returns):

    if getattr(returns, "ndim", 0) == 1:
        return np.array([1.0], dtype=float)

    cov = returns.cov()

    inv_vol = 1 / np.sqrt(np.diag(cov))

    weights = inv_vol / inv_vol.sum()

    return weights
