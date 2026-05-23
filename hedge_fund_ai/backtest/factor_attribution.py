"""
Phase 5 factor attribution: which factor actually made money?

Methods:
  1. IC-weighted return attribution: sum(IC * factor_exposure * period_return)
  2. Per-factor average IC across all rebalance periods
  3. Stability: IC consistency across time periods and regimes
"""
import numpy as np
from collections import defaultdict


def factor_contribution(trade_log: list) -> dict:
    """Legacy wrapper — aggregates factor scores from trade_log."""
    contributions = {}
    for entry in (trade_log or []):
        for _, factors in (entry.get("factor_scores") or {}).items():
            for k, v in (factors or {}).items():
                contributions[k] = contributions.get(k, 0) + v
    return contributions


def attribution_from_audit(audit_log) -> dict:
    """
    Full factor attribution from AuditLog.

    Returns per-factor:
      - mean_ic: average Spearman IC vs forward returns
      - ic_ir:   IC / std(IC) — information ratio of the factor
      - hit_rate: fraction of periods with positive IC
      - avg_exposure: average weighted exposure in portfolio
      - est_contribution_bps: estimated return contribution (IC × exposure × period_ret)
    """
    records = audit_log._records if audit_log else []
    if not records:
        return {}

    ic_by_factor: dict[str, list] = defaultdict(list)
    exposure_by_factor: dict[str, list] = defaultdict(list)

    for r in records:
        # IC realized this period
        for factor, ic_val in r.realized_ic.items():
            ic_by_factor[factor].append(float(ic_val))

        # Weighted factor exposure
        total_w = sum(r.final_weights.values()) or 1.0
        for t, w in r.final_weights.items():
            fe = r.factor_exposures.get(t, {})
            for factor, val in fe.items():
                exposure_by_factor[factor].append(float(val) * float(w) / total_w)

    result = {}
    all_factors = set(ic_by_factor) | set(exposure_by_factor)

    for factor in sorted(all_factors):
        ics  = np.array(ic_by_factor.get(factor, []), dtype=float)
        exps = np.array(exposure_by_factor.get(factor, []), dtype=float)

        mean_ic  = float(np.mean(ics))  if len(ics)  else 0.0
        std_ic   = float(np.std(ics))   if len(ics)  else 1.0
        ic_ir    = float(mean_ic / (std_ic + 1e-6))
        hit_rate = float((ics > 0).mean()) if len(ics) else 0.5
        avg_exp  = float(np.mean(exps))  if len(exps) else 0.0

        # Estimated contribution: IC * exposure * sqrt(252) annualized signal
        est_contrib_bps = round(mean_ic * avg_exp * 100 * 10_000, 1)

        result[factor] = {
            "mean_ic":           round(mean_ic, 4),
            "ic_ir":             round(ic_ir, 3),
            "hit_rate":          round(hit_rate, 3),
            "avg_exposure":      round(avg_exp, 4),
            "est_contrib_bps":   est_contrib_bps,
            "n_observations":    len(ics),
        }

    return result


def stability_report(audit_log) -> dict:
    """
    Phase 5 stability: does performance hold across time periods and regimes?

    Splits the backtest into thirds and computes per-third metrics.
    """
    records = [r for r in (audit_log._records if audit_log else [])
               if r.period_return is not None]
    if len(records) < 6:
        return {"error": "insufficient records for stability analysis"}

    n = len(records)
    thirds = [records[:n//3], records[n//3:2*n//3], records[2*n//3:]]
    labels = ["early", "mid", "late"]

    period_stats = {}
    for label, chunk in zip(labels, thirds):
        rets = [r.period_return for r in chunk]
        period_stats[label] = {
            "n": len(rets),
            "avg_return": round(float(np.mean(rets)), 5),
            "hit_rate":   round(float(np.mean([r > 0 for r in rets])), 3),
            "start_date": chunk[0].date,
            "end_date":   chunk[-1].date,
        }

    # Regime stability
    regime_stats = {}
    regime_buckets: dict[str, list] = defaultdict(list)
    for r in records:
        regime_buckets[r.regime].append(r.period_return)
    for regime, rets in regime_buckets.items():
        regime_stats[regime] = {
            "n":          len(rets),
            "avg_return": round(float(np.mean(rets)), 5),
            "hit_rate":   round(float(np.mean([r > 0 for r in rets])), 3),
        }

    return {"by_period": period_stats, "by_regime": regime_stats}
