"""
P&L Attribution & Factor Explainability.

Decomposes daily/monthly returns into:
  1. Factor contribution (market beta, momentum, quality tilt)
  2. Security selection (stock-specific alpha vs factor prediction)
  3. Timing contribution (rebalance timing vs passive hold)
  4. Execution cost drag
  5. Interaction effects

Generates standardized attribution reports for stakeholders.
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_brinson_attribution(
    portfolio_weights:  dict[str, float],
    benchmark_weights:  dict[str, float],
    stock_returns:      dict[str, float],
    benchmark_return:   float,
) -> dict:
    """
    Brinson-Hood-Beebower (BHB) attribution decomposition.

    Components:
      Allocation effect:  over/underweight sectors vs benchmark
      Selection effect:   stock picks within each sector
      Interaction effect: combined over/underweight + stock selection

    Returns {allocation, selection, interaction, total_active} in fraction.
    """
    all_tickers = set(portfolio_weights) | set(benchmark_weights)

    allocation   = 0.0
    selection    = 0.0
    interaction  = 0.0

    for t in all_tickers:
        wp  = float(portfolio_weights.get(t, 0))
        wb  = float(benchmark_weights.get(t, 0))
        rp  = float(stock_returns.get(t, 0))
        rb  = benchmark_return  # simplified: use portfolio benchmark return

        # Allocation: (wp - wb) × (rb - R_b_total)
        allocation  += (wp - wb) * (rb - benchmark_return)
        # Selection: wb × (rp - rb)
        selection   += wb * (rp - rb)
        # Interaction: (wp - wb) × (rp - rb)
        interaction += (wp - wb) * (rp - rb)

    portfolio_return = sum(
        portfolio_weights.get(t, 0) * stock_returns.get(t, 0)
        for t in all_tickers
    )
    total_active = portfolio_return - benchmark_return

    return {
        "portfolio_return":  round(float(portfolio_return), 6),
        "benchmark_return":  round(float(benchmark_return), 6),
        "total_active":      round(float(total_active), 6),
        "allocation_effect": round(float(allocation), 6),
        "selection_effect":  round(float(selection), 6),
        "interaction_effect":round(float(interaction), 6),
        "sum_check":         round(float(allocation + selection + interaction), 6),
    }


def compute_factor_attribution(
    portfolio_weights: dict[str, float],
    factor_loadings:   dict[str, dict],   # {ticker: {factor: loading}}
    factor_returns:    dict[str, float],  # {factor: period_return}
    stock_returns:     dict[str, float],
) -> dict:
    """
    Decompose portfolio return into factor contributions + residual alpha.

    R_p = Σ_k (w_p · b_k) × F_k + α_p
    where b_k = weighted average factor loading, F_k = factor return.
    """
    factor_contribs = {}
    total_factor_return = 0.0

    # Portfolio-weighted factor loadings
    for factor, f_ret in factor_returns.items():
        port_loading = sum(
            portfolio_weights.get(t, 0) * factor_loadings.get(t, {}).get(factor, 0)
            for t in portfolio_weights
        )
        contrib = float(port_loading * f_ret)
        factor_contribs[factor] = {
            "loading":      round(float(port_loading), 4),
            "factor_return":round(float(f_ret), 6),
            "contribution": round(contrib, 6),
        }
        total_factor_return += contrib

    portfolio_return = sum(
        portfolio_weights.get(t, 0) * stock_returns.get(t, 0)
        for t in portfolio_weights
    )
    alpha_residual = portfolio_return - total_factor_return

    return {
        "portfolio_return":   round(float(portfolio_return), 6),
        "factor_return_total":round(float(total_factor_return), 6),
        "alpha_residual":     round(float(alpha_residual), 6),
        "factor_breakdown":   factor_contribs,
        "alpha_pct_of_total": round(float(alpha_residual / (abs(portfolio_return) + 1e-8) * 100), 2),
    }


def compute_timing_contribution(
    rebalance_dates:     list[str],
    portfolio_returns:   pd.Series,
    buyhold_returns:     pd.Series,
) -> dict:
    """
    Timing contribution: did rebalancing add or destroy value vs a passive hold?
    Positive timing = rebalances were well-timed (bought low, sold high).
    """
    if portfolio_returns.empty or buyhold_returns.empty:
        return {"timing_contribution": 0.0}

    try:
        port_total  = float(portfolio_returns.add(1).prod() - 1)
        bh_total    = float(buyhold_returns.add(1).prod() - 1)
        timing_contrib = port_total - bh_total
        return {
            "portfolio_total":    round(port_total, 6),
            "buyhold_total":      round(bh_total, 6),
            "timing_contribution":round(timing_contrib, 6),
            "n_rebalances":       len(rebalance_dates),
        }
    except Exception:
        return {"timing_contribution": 0.0}


def generate_attribution_report(
    equity:             pd.Series,
    audit_log,
    benchmark:          pd.Series | None = None,
    period:             str = "monthly",
) -> pd.DataFrame:
    """
    Generate a standardized attribution report as a DataFrame.
    One row per period (monthly or quarterly).

    Columns: date, portfolio_return, benchmark_return, active_return,
             factor_contrib, selection_contrib, cost_drag
    """
    if equity is None or equity.empty:
        return pd.DataFrame()

    records = audit_log._records if audit_log else []
    rows    = []

    rets = equity.pct_change().dropna()
    if period == "monthly":
        grouped = rets.resample("ME")
    else:
        grouped = rets.resample("QE")

    bench_rets = benchmark.pct_change().dropna() if benchmark is not None else None

    for period_end, period_rets in grouped:
        if period_rets.empty:
            continue

        port_ret  = float(period_rets.add(1).prod() - 1)
        bench_ret = 0.0
        if bench_rets is not None:
            period_bench = bench_rets.loc[period_rets.index]
            bench_ret    = float(period_bench.add(1).prod() - 1) if len(period_bench) > 0 else 0.0

        # Find audit records in this period
        period_records = [
            r for r in records
            if r.date <= str(period_end.date())
        ]

        avg_cost = float(np.mean([r.total_cost_frac for r in period_records])) if period_records else 0.0
        avg_dd   = float(np.mean([r.dd_scale for r in period_records])) if period_records else 1.0
        n_reb    = len(period_records)

        rows.append({
            "period_end":       str(period_end.date()),
            "portfolio_return": round(port_ret, 6),
            "benchmark_return": round(bench_ret, 6),
            "active_return":    round(port_ret - bench_ret, 6),
            "avg_cost_drag":    round(avg_cost, 6),
            "avg_dd_scale":     round(avg_dd, 3),
            "n_rebalances":     n_reb,
            "outperformed":     port_ret > bench_ret,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        hit_rate = float(df["outperformed"].mean())
        avg_active = float(df["active_return"].mean())
        logger.info("Attribution: hit_rate=%.1f%% | avg_active=%.3f%%",
                    hit_rate * 100, avg_active * 100)
    return df
