"""
Monte Carlo Simulation & Multiple-Testing Correction.

Methods:
  1. Multiple-testing correction (Benjamini-Hochberg FDR) on factor ICs
  2. Monte Carlo path simulation: shuffles historical returns preserving
     autocorrelation and volatility clustering (block bootstrap)
  3. Capacity modeling: scales costs with AUM to find return decay point

Reference:
  - Harvey, Liu, Zhu (2016) "...and the Cross-Section of Expected Returns"
  - Politis & Romano (1994) "The Stationary Bootstrap"
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── 1. Multiple-testing correction ────────────────────────────────────────────

def benjamini_hochberg(p_values: dict[str, float], fdr: float = 0.05) -> dict[str, bool]:
    """
    Benjamini-Hochberg False Discovery Rate correction.
    Returns {factor: significant} where significant = survives correction.

    More appropriate than Bonferroni for correlated financial factors.
    """
    if not p_values:
        return {}

    factors  = list(p_values.keys())
    p_arr    = np.array([p_values[f] for f in factors])
    n        = len(p_arr)
    ranks    = np.argsort(p_arr) + 1  # 1-indexed ranks
    thresholds = (ranks / n) * fdr

    sorted_p   = np.sort(p_arr)
    sorted_thr = np.sort(thresholds)

    # Find largest k where p(k) <= k/n * fdr
    k_max = 0
    for k in range(n - 1, -1, -1):
        if sorted_p[k] <= sorted_thr[k]:
            k_max = k + 1  # +1 because 0-indexed
            break

    cutoff = sorted_thr[k_max - 1] if k_max > 0 else 0.0
    result = {f: float(p_values[f]) <= cutoff for f in factors}

    n_sig = sum(result.values())
    logger.info("BH correction: %d/%d factors significant at FDR=%.2f (cutoff p=%.4f)",
                n_sig, n, fdr, cutoff)
    return result


def ic_significance_test(ic_series: list[float]) -> dict:
    """
    Test whether a factor's IC is statistically significant.
    Uses t-test: H0: mean IC = 0.
    Returns {t_stat, p_value, significant, ic_ir}.
    """
    from scipy import stats
    arr = np.array(ic_series, dtype=float)
    if len(arr) < 4:
        return {"t_stat": 0, "p_value": 1.0, "significant": False, "ic_ir": 0}
    t, p = stats.ttest_1samp(arr, 0)
    ic_ir = float(np.mean(arr) / (np.std(arr) + 1e-6))
    return {
        "t_stat":      round(float(t), 3),
        "p_value":     round(float(p), 4),
        "significant": bool(p < 0.05),
        "ic_ir":       round(ic_ir, 3),
    }


def correct_factor_significance(ic_history: dict[str, list]) -> dict:
    """
    Run BH correction across all factors and return adjusted significance.
    Returns {factor: {"significant": bool, "p_value": float, ...}}
    """
    p_values = {}
    details  = {}
    for factor, history in ic_history.items():
        if len(history) >= 6:
            test = ic_significance_test(history)
            p_values[factor] = test["p_value"]
            details[factor]  = test

    sig = benjamini_hochberg(p_values, fdr=0.10)   # 10% FDR for finance
    for factor in details:
        details[factor]["bh_significant"] = sig.get(factor, False)

    return details


# ── 2. Monte Carlo block bootstrap ────────────────────────────────────────────

def block_bootstrap_equity(
    equity: pd.Series,
    n_simulations: int = 500,
    block_size: int    = 21,     # monthly blocks preserve autocorrelation
    seed: int          = 42,
) -> dict:
    """
    Block bootstrap Monte Carlo: resample blocks of returns (not individual days)
    to preserve autocorrelation and volatility clustering.

    Returns:
      {
        "sharpe_dist":     array of simulated Sharpes
        "cagr_dist":       array of simulated CAGRs
        "maxdd_dist":      array of simulated max DDs
        "model_sharpe":    the real model Sharpe
        "p_sharpe_gt_0":   fraction of simulations with positive Sharpe
        "p_sharpe_gt_1":   fraction of simulations with Sharpe > 1.0
        "var_95":          5th percentile simulated total return
        "verdict":         interpretation string
      }
    """
    rng = np.random.default_rng(seed)
    rets = equity.pct_change().dropna().values.astype(float)
    n    = len(rets)

    if n < 60:
        return {"error": "insufficient history for Monte Carlo (need ≥60 days)"}

    sharpes, cagrs, maxdds = [], [], []

    for _ in range(n_simulations):
        # Block bootstrap: draw random starting points for each block
        sim_rets = []
        while len(sim_rets) < n:
            start = int(rng.integers(0, n - block_size))
            sim_rets.extend(rets[start:start + block_size].tolist())
        sim_rets = np.array(sim_rets[:n])

        # Compute metrics
        mu  = float(np.mean(sim_rets))
        sig = float(np.std(sim_rets)) + 1e-9
        sharpes.append(mu / sig * np.sqrt(252))

        eq = np.cumprod(1 + sim_rets)
        peak = np.maximum.accumulate(eq)
        dd   = (eq - peak) / (peak + 1e-9)
        maxdds.append(float(dd.min()))
        total = float(eq[-1] - 1)
        years = n / 252
        cagrs.append(float((1 + total) ** (1 / max(years, 0.01)) - 1))

    sharpes = np.array(sharpes)
    cagrs   = np.array(cagrs)
    maxdds  = np.array(maxdds)

    real_ret = rets
    real_mu  = float(np.mean(real_ret))
    real_sig = float(np.std(real_ret)) + 1e-9
    model_sharpe = real_mu / real_sig * np.sqrt(252)

    p_gt_0 = float(np.mean(sharpes > 0))
    p_gt_1 = float(np.mean(sharpes > 1.0))
    var_95  = float(np.percentile(cagrs, 5))

    if p_gt_1 > 0.50:
        verdict = f"ROBUST: >50% of simulations produce Sharpe>1 ({p_gt_1:.0%})"
    elif p_gt_0 > 0.70:
        verdict = f"ACCEPTABLE: {p_gt_0:.0%} of simulations positive Sharpe"
    else:
        verdict = f"FRAGILE: only {p_gt_0:.0%} of simulations positive Sharpe"

    logger.info("Monte Carlo (%d sims): Sharpe %.2f±%.2f | CAGR %.1f±%.1f%% | MaxDD %.1f±%.1f%% | %s",
                n_simulations,
                float(np.mean(sharpes)), float(np.std(sharpes)),
                float(np.mean(cagrs))*100, float(np.std(cagrs))*100,
                float(np.mean(maxdds))*100, float(np.std(maxdds))*100,
                verdict)

    return {
        "sharpe_mean":   round(float(np.mean(sharpes)), 3),
        "sharpe_std":    round(float(np.std(sharpes)), 3),
        "sharpe_p5":     round(float(np.percentile(sharpes, 5)), 3),
        "sharpe_p95":    round(float(np.percentile(sharpes, 95)), 3),
        "cagr_mean":     round(float(np.mean(cagrs)), 4),
        "cagr_p5":       round(var_95, 4),
        "maxdd_mean":    round(float(np.mean(maxdds)), 4),
        "model_sharpe":  round(model_sharpe, 3),
        "p_sharpe_gt_0": round(p_gt_0, 3),
        "p_sharpe_gt_1": round(p_gt_1, 3),
        "n_simulations": n_simulations,
        "verdict":       verdict,
    }


# ── 3. Capacity modeling ───────────────────────────────────────────────────────

def model_capacity(
    equity: pd.Series,
    audit_log,
    aum_levels: list[float] | None = None,
) -> dict:
    """
    Estimate strategy capacity: at what AUM do transaction costs erode alpha?

    Model:
      - Market impact scales as sqrt(trade_size / ADV)
      - At current AUM, costs are C0
      - At AUM_x, costs are C0 * sqrt(AUM_x / AUM_0)
      - Find AUM where alpha = 0

    Returns capacity estimates at each AUM level.
    """
    if aum_levels is None:
        aum_levels = [100_000, 500_000, 1_000_000, 5_000_000,
                      10_000_000, 50_000_000, 100_000_000]

    records = audit_log._records if audit_log else []
    if not records or equity is None or equity.empty:
        return {"error": "insufficient data"}

    # Current cost rate from audit log
    avg_cost_frac = float(np.mean([r.total_cost_frac for r in records
                                   if r.total_cost_frac > 0])) if records else 0.001

    # Estimated alpha (daily)
    rets = equity.pct_change().dropna()
    alpha_daily = float(rets.mean())

    # Reference AUM
    ref_aum = 100_000.0

    results = {}
    for aum in aum_levels:
        # Costs scale with sqrt of AUM ratio (Almgren-Chriss linear model)
        scale = np.sqrt(aum / ref_aum)
        cost_frac = avg_cost_frac * scale
        # Net alpha at this AUM (daily)
        net_alpha = alpha_daily - cost_frac / 252  # spread cost daily
        net_sharpe_adj = (net_alpha / (float(rets.std()) + 1e-9)) * np.sqrt(252)

        results[f"${aum/1e6:.1f}M"] = {
            "aum":           aum,
            "cost_frac_pct": round(cost_frac * 100, 4),
            "net_alpha_ann": round(net_alpha * 252, 4),
            "adj_sharpe":    round(net_sharpe_adj, 3),
            "viable":        net_alpha > 0,
        }

    # Find capacity limit (last viable AUM)
    viable = [v for v in results.values() if v["viable"]]
    capacity_limit = viable[-1]["aum"] if viable else 0

    logger.info("Capacity model: viable up to $%.1fM AUM", capacity_limit / 1e6)

    return {
        "by_aum":           results,
        "capacity_limit_usd": capacity_limit,
        "ref_cost_frac":    round(avg_cost_frac, 6),
        "alpha_daily":      round(alpha_daily, 6),
    }
