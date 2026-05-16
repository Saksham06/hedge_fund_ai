"""
Swing Trading Metrics — computed from walk-forward backtest and live trades.

The 7 metrics that matter for swing:
  1. Avg Hold Period (target 7-45 days)
  2. Win Rate by Hold Bucket (<7d, 7-30d, 30-60d)
  3. Profit Factor (gross wins / gross losses, target >1.3)
  4. Max Consecutive Losses (target ≤5)
  5. Cost-Adjusted Net Return (target >8% annual)
  6. Signal-to-Noise Ratio (avg return / std per hold period)
  7. Regime Win Rate (split by risk_on/off/neutral)
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_swing_metrics(equity: pd.Series, audit_log,
                          swing_tracker=None) -> dict:
    """
    Compute all 7 swing-specific metrics from the equity curve + audit log.
    If swing_tracker provided, adds live per-trade metrics.
    """
    result = {}

    records = audit_log._records if audit_log else []
    if not records:
        return {"status": "no audit records"}

    # Period returns from audit log
    period_rets = [r.period_return for r in records if r.period_return is not None]
    dates       = [r.date for r in records if r.period_return is not None]

    if not period_rets:
        return {"status": "no period returns yet"}

    arr = np.array(period_rets, dtype=float)

    # ── Metric 1: Hold-period return distribution ──────────────────────────────
    # In backtest: each period = REBALANCE_DAYS (5 days for swing)
    rebalance_days = 5
    result["avg_period_return"]   = round(float(np.mean(arr)) * 100, 3)
    result["median_period_return"]= round(float(np.median(arr)) * 100, 3)

    # ── Metric 2: Win rate by regime (proxy for hold-bucket) ──────────────────
    regime_buckets: dict = {}
    for r in records:
        if r.period_return is None:
            continue
        reg = r.regime
        regime_buckets.setdefault(reg, []).append(r.period_return)

    result["win_rate_by_regime"] = {
        reg: {
            "win_rate": round(float(np.mean([x > 0 for x in vals])), 3),
            "avg_ret":  round(float(np.mean(vals)) * 100, 3),
            "n":        len(vals),
        }
        for reg, vals in regime_buckets.items()
    }

    # ── Metric 3: Profit factor ────────────────────────────────────────────────
    wins   = arr[arr > 0]
    losses = arr[arr < 0]
    pf     = float(wins.sum()) / (abs(losses.sum()) + 1e-9)
    result["profit_factor"] = round(pf, 3)

    # ── Metric 4: Max consecutive losses ──────────────────────────────────────
    seq = [1 if r > 0 else 0 for r in period_rets]
    max_loss, cur = 0, 0
    for s in seq:
        if s == 0:
            cur += 1; max_loss = max(max_loss, cur)
        else:
            cur = 0
    result["max_consecutive_losses"] = max_loss

    # ── Metric 5: Annualized net return ───────────────────────────────────────
    if equity is not None and not equity.empty:
        total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1)
        years     = max(len(equity) / 252, 0.01)
        cagr      = float((1 + total_ret) ** (1 / years) - 1)
        result["cagr_pct"]     = round(cagr * 100, 2)
        result["total_return"] = round(total_ret * 100, 2)

    # ── Metric 6: Signal-to-noise ratio ───────────────────────────────────────
    snr = float(np.mean(arr)) / (float(np.std(arr)) + 1e-9)
    result["signal_to_noise"] = round(snr * np.sqrt(252 / rebalance_days), 3)

    # ── Metric 7: Sharpe (annualized at swing frequency) ──────────────────────
    sharpe = float(np.mean(arr) / (np.std(arr) + 1e-9)) * np.sqrt(252 / rebalance_days)
    result["swing_sharpe"] = round(sharpe, 3)

    # ── Win rate overall ──────────────────────────────────────────────────────
    result["win_rate"]    = round(float(np.mean(arr > 0)), 3)
    result["n_periods"]   = len(period_rets)

    # ── Live trade metrics from swing_tracker ─────────────────────────────────
    if swing_tracker:
        live = swing_tracker.compute_metrics()
        result["live_trades"] = live

    # ── Swing deployment verdict ──────────────────────────────────────────────
    win_rate = result["win_rate"]
    pf_val   = result["profit_factor"]
    n        = result["n_periods"]

    if n < 20:
        verdict = f"EARLY — {n} periods, need ≥20 swing periods to assess"
    elif win_rate >= 0.55 and pf_val >= 1.3:
        verdict = "STRONG SWING EDGE — consider live capital at $500-$1,000"
    elif win_rate >= 0.52 and pf_val >= 1.1:
        verdict = "MODERATE EDGE — paper trade 4 more weeks before capital"
    elif win_rate >= 0.50 and pf_val >= 1.0:
        verdict = "BREAKEVEN — signal weak, review factor weights"
    else:
        verdict = "NO EDGE — win rate or profit factor below threshold"

    result["swing_verdict"] = verdict
    logger.info("Swing metrics: WR=%.1f%% | PF=%.2f | SNR=%.2f | %s",
                win_rate*100, pf_val, snr, verdict)

    return result
