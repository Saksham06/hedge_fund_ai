"""
Realistic transaction cost model.

Components applied ONLY on rebalance days:
  1. Fixed commission: FIXED_BPS (e.g. 5 bps)
  2. Bid-ask spread proxy: SPREAD_BPS (e.g. 3 bps for liquid names)
  3. Market impact: Almgren-Chriss linear approximation
       impact_bps = IMPACT_COEFF * sqrt(trade_size_pct / adv_pct)
     where adv_pct = trade_notional / avg_daily_dollar_volume

All costs are one-way (applied per side). Round-trip = 2x.

Returns total cost as a fraction of portfolio value (not bps).
"""
import math
import logging

logger = logging.getLogger(__name__)

# Tuneable constants (realistic for US large-cap equities)
FIXED_BPS   = 5.0    # fixed commission per trade (one-way)
SPREAD_BPS  = 3.0    # half bid-ask spread (one-way)
IMPACT_COEFF = 0.1   # market impact coefficient (Almgren-Chriss linear)


def compute_trade_cost(
    weight_change: float,         # |new_weight - old_weight|
    avg_daily_dollar_vol: float,  # ADV in dollars (past 21 days)
    portfolio_notional: float,    # total portfolio $ value
) -> float:
    """
    Returns cost as a fraction of portfolio value for one side of a trade.

    Args:
        weight_change:           |Δweight| (e.g. 0.05 for 5% change)
        avg_daily_dollar_vol:    ADV in dollars (used to compute impact)
        portfolio_notional:      portfolio value in dollars

    Returns:
        cost as fraction of portfolio (e.g. 0.0008 = 8bps of portfolio)
    """
    if weight_change < 1e-6:
        return 0.0

    trade_dollars = abs(weight_change) * portfolio_notional

    # Fixed + spread (one-way)
    fixed_cost = (FIXED_BPS + SPREAD_BPS) / 10_000

    # Market impact: sqrt model
    if avg_daily_dollar_vol > 0:
        participation = trade_dollars / avg_daily_dollar_vol
        impact_fraction = IMPACT_COEFF * math.sqrt(participation)
        # Cap at 50bps per side (extreme impact)
        impact = min(impact_fraction, 0.005)
    else:
        impact = 0.001  # fallback: 10bps if no volume data

    cost_per_dollar = fixed_cost + impact
    return float(cost_per_dollar * trade_dollars / portfolio_notional)


def compute_rebalance_cost(
    old_weights: dict,
    new_weights: dict,
    adv_map: dict,              # {ticker: avg_daily_dollar_volume}
    portfolio_notional: float,
) -> tuple[float, dict]:
    """
    Compute total one-way transaction cost for a rebalance.

    Returns:
        (total_cost_fraction, per_ticker_cost_dict)

    total_cost_fraction: fraction of portfolio to deduct from equity
    per_ticker_cost_dict: breakdown for audit log
    """
    all_tickers = set(old_weights) | set(new_weights)
    per_ticker = {}
    total_cost = 0.0

    for t in all_tickers:
        old_w = float(old_weights.get(t, 0.0))
        new_w = float(new_weights.get(t, 0.0))
        delta = abs(new_w - old_w)
        if delta < 1e-6:
            continue

        adv = float(adv_map.get(t, 0))
        cost = compute_trade_cost(delta, adv, portfolio_notional)
        per_ticker[t] = {
            "weight_change": round(delta, 5),
            "adv_m":         round(adv / 1e6, 2),
            "cost_bps":      round(cost * portfolio_notional / (delta * portfolio_notional + 1e-6) * 10_000, 2),
            "cost_fraction": round(cost, 7),
        }
        total_cost += cost

    return float(total_cost), per_ticker
