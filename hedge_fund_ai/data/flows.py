import numpy as np


def volume_flow_signal(price_hist, volume_hist):
    if len(price_hist) < 20 or len(volume_hist) < 20:
        return 0

    price = np.array(price_hist[-20:], dtype=float)
    volume = np.array(volume_hist[-20:], dtype=float)

    # price change weighted by volume
    flow = np.sum(np.diff(price) * volume[1:])
    return float(flow / (np.sum(volume) + 1e-6))

