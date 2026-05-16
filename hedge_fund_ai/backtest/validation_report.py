"""
Automated Validation Report.

Generates a comprehensive PDF/text report covering:
  1. Backtest performance vs SPY and equal-weight
  2. Placebo test result with p-value
  3. Factor IC attribution (which factors worked)
  4. Stability across time periods and regimes
  5. Live signal IC report (paper trading validation)
  6. Deployment verdict
"""
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REPORT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "state", "validation_report.txt"
)


def _bar(value: float, min_v: float, max_v: float, width: int = 20) -> str:
    """ASCII progress bar."""
    pct = (value - min_v) / (max_v - min_v + 1e-9)
    filled = int(round(pct * width))
    filled = max(0, min(filled, width))
    return "█" * filled + "░" * (width - filled)


def _pass_fail(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def generate_validation_report(
    metrics: dict,
    placebo: dict | None,
    audit_log,
    signal_ic_report: dict | None,
    equity: pd.Series | None = None,
) -> str:
    """
    Generate a full text validation report.
    Returns the report as a string and saves to state/.
    """
    from hedge_fund_ai.backtest.factor_attribution import attribution_from_audit, stability_report

    lines = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def h1(title): lines.append("\n" + "=" * 70); lines.append(f"  {title}"); lines.append("=" * 70)
    def h2(title): lines.append(f"\n── {title} " + "─" * max(0, 50 - len(title)))
    def row(label, value, note=""): lines.append(f"  {label:<35} {value:<18} {note}")

    lines.append(f"\n{'AI HEDGE FUND — VALIDATION REPORT':^70}")
    lines.append(f"{'Generated: ' + now:^70}\n")

    # ── 1. BACKTEST PERFORMANCE ────────────────────────────────────────────────
    h1("1. BACKTEST PERFORMANCE")
    m = metrics or {}

    sharpe   = m.get("sharpe", 0)
    sortino  = m.get("sortino", 0)
    calmar   = m.get("calmar", 0)
    max_dd   = m.get("max_drawdown", 0)
    cagr     = m.get("cagr", 0)
    win_rate = m.get("win_rate", 0)
    ann_vol  = m.get("ann_vol", 0)
    alpha    = m.get("alpha_ann", 0)
    beta     = m.get("beta", 1)
    vs_spy   = m.get("excess_vs_spy", 0)
    vs_ew    = m.get("excess_vs_ew",  0)

    h2("Core Metrics")
    row("Sharpe Ratio",        f"{sharpe:.3f}",           _pass_fail(sharpe > 1.0))
    row("Sortino Ratio",       f"{sortino:.3f}",          _pass_fail(sortino > 1.5))
    row("Calmar Ratio",        f"{calmar:.3f}",           _pass_fail(calmar > 0.5))
    row("Max Drawdown",        f"{max_dd*100:.1f}%",      _pass_fail(max_dd > -0.20))
    row("CAGR",                f"{cagr*100:.1f}%",        _pass_fail(cagr > 0.10))
    row("Win Rate",            f"{win_rate*100:.1f}%",    _pass_fail(win_rate > 0.50))
    row("Annualized Vol",      f"{ann_vol*100:.1f}%",     "target 12%")

    h2("Benchmark Comparison")
    row("Alpha (annualized)",  f"{alpha*100:.2f}%",       _pass_fail(alpha > 0.02))
    row("Beta vs SPY",         f"{beta:.3f}",             "target < 0.90")
    row("Excess vs SPY",       f"{vs_spy*100:.2f}%",      _pass_fail(vs_spy > 0))
    row("Excess vs EW",        f"{vs_ew*100:.2f}%",       _pass_fail(vs_ew > 0))

    if m.get("avg_cost_bps"):
        h2("Execution")
        row("Avg Cost (bps)",  f"{m['avg_cost_bps']:.1f}",    "per rebalance")
        row("Avg Raw Turnover",f"{m.get('avg_raw_turnover',0):.3f}", "")
        row("Avg Filt Turnover",f"{m.get('avg_filtered_turnover',0):.3f}","")

    # ── 2. PLACEBO TEST ────────────────────────────────────────────────────────
    h1("2. PLACEBO TEST (Is this alpha or beta?)")
    if placebo:
        p_val   = placebo.get("p_value", 1.0)
        verdict = placebo.get("verdict", "")
        row("Model Sharpe",    f"{placebo.get('model_sharpe',0):.3f}", "")
        row("Random Mean",     f"{placebo.get('random_mean',0):.3f}",  "")
        row("Random Std",      f"{placebo.get('random_std',0):.3f}",   "")
        row("p-value",         f"{p_val:.3f}",                         _pass_fail(p_val < 0.05))
        lines.append(f"\n  VERDICT: {verdict}")
    else:
        lines.append("  Placebo test not yet run. Will execute on next backtest.")

    # ── 3. FACTOR ATTRIBUTION ──────────────────────────────────────────────────
    h1("3. FACTOR IC ATTRIBUTION")
    if audit_log and audit_log._records:
        attr = attribution_from_audit(audit_log)
        if attr:
            lines.append(f"\n  {'Factor':<22} {'Mean IC':>8} {'IC-IR':>7} {'Hit%':>7} {'Est Contrib':>11} {'n':>4}")
            lines.append("  " + "-" * 62)
            sorted_factors = sorted(attr.items(), key=lambda x: -x[1]["ic_ir"])
            for factor, stats in sorted_factors:
                lines.append(
                    f"  {factor:<22} {stats['mean_ic']:>8.4f} {stats['ic_ir']:>7.3f} "
                    f"{stats['hit_rate']*100:>6.0f}% {stats['est_contrib_bps']:>10.1f}bps"
                    f"  {stats['n_observations']:>4}"
                )
    else:
        lines.append("  No audit data available yet.")

    # ── 4. STABILITY ANALYSIS ──────────────────────────────────────────────────
    h1("4. STABILITY ACROSS PERIODS AND REGIMES")
    if audit_log and audit_log._records:
        stab = stability_report(audit_log)

        h2("By Time Period (early / mid / late thirds)")
        by_period = stab.get("by_period", {})
        for label, stats in by_period.items():
            lines.append(
                f"  {label:<8} {stats['start_date']}→{stats['end_date']}  "
                f"avg={stats['avg_return']*100:+.3f}%  hit={stats['hit_rate']*100:.0f}%  "
                f"n={stats['n']}"
            )

        h2("By Regime")
        by_regime = stab.get("by_regime", {})
        for regime, stats in by_regime.items():
            lines.append(
                f"  {regime:<12} avg={stats['avg_return']*100:+.3f}%  "
                f"hit={stats['hit_rate']*100:.0f}%  n={stats['n']}"
            )
    else:
        lines.append("  No audit data available.")

    # ── 5. LIVE SIGNAL IC (paper trading) ─────────────────────────────────────
    h1("5. LIVE SIGNAL VALIDATION (Paper Trading IC)")
    if signal_ic_report:
        n_filled = signal_ic_report.get("n_filled", 0)
        mean_ic  = signal_ic_report.get("mean_ic", 0)
        ic_ir    = signal_ic_report.get("ic_ir", 0)
        hit_rate = signal_ic_report.get("ic_hit_rate", 0)
        verdict  = signal_ic_report.get("verdict", "N/A")

        row("Periods Filled",  f"{n_filled}",             f"(need ≥12)")
        row("Mean IC",         f"{mean_ic:.4f}",          _pass_fail(mean_ic > 0.02))
        row("IC-IR",           f"{ic_ir:.3f}",            _pass_fail(ic_ir > 0.25))
        row("Hit Rate",        f"{hit_rate*100:.1f}%",    _pass_fail(hit_rate > 0.52))

        if signal_ic_report.get("by_regime"):
            h2("IC by Regime")
            for reg, stats in signal_ic_report["by_regime"].items():
                lines.append(f"  {reg:<14} IC={stats['mean_ic']:>7.4f}  hit={stats['hit_rate']*100:.0f}%  n={stats['n']}")

        lines.append(f"\n  VERDICT: {verdict}")
    else:
        lines.append("  No live signal data yet. Run pipeline daily to accumulate.")
        lines.append("  IC will fill after 21 trading days per signal logged.")

    # ── 6. DEPLOYMENT VERDICT ─────────────────────────────────────────────────
    h1("6. DEPLOYMENT VERDICT")

    checks = {
        "Sharpe > 1.0":            sharpe > 1.0,
        "Beats SPY":               vs_spy > 0,
        "Beats equal-weight":      vs_ew > 0,
        "Max DD < 20%":            max_dd > -0.20,
        "CAGR > 10%":              cagr > 0.10,
        "Placebo p < 0.10":        (placebo or {}).get("p_value", 1.0) < 0.10,
        "Live IC > 0 (≥12 periods)": (
            (signal_ic_report or {}).get("n_filled", 0) >= 12 and
            (signal_ic_report or {}).get("mean_ic", 0) > 0
        ),
    }

    passed = sum(checks.values())
    total  = len(checks)
    lines.append("")
    for check, result in checks.items():
        lines.append(f"  {'✅' if result else '❌'}  {check}")

    lines.append(f"\n  Score: {passed}/{total} checks passed")

    if passed >= 6 and checks.get("Live IC > 0 (≥12 periods)"):
        deploy_verdict = "[DEPLOY] — Start with $500-$1,000 and monitor execution slippage"
    elif passed >= 5:
        deploy_verdict = "[PAPER] — Strong backtest but live IC not yet validated"
    elif passed >= 3:
        deploy_verdict = "[NOT READY] — Address failing checks before deploying capital"
    else:
        deploy_verdict = "[DO NOT DEPLOY] — Multiple critical failures"

    lines.append(f"\n  {deploy_verdict}")

    lines.append(f"\n{'─'*70}")
    lines.append(f"  Report saved: {_REPORT_PATH}")
    lines.append(f"  Generated:    {now}")
    lines.append(f"{'─'*70}\n")

    report = "\n".join(lines)

    # Save
    try:
        os.makedirs(os.path.dirname(_REPORT_PATH), exist_ok=True)
        with open(_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
    except Exception as e:
        logger.warning("Report save failed: %s", e)

    return report
