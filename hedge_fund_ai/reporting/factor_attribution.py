"""
Factor attribution: weighted exposure per factor across portfolio.
Also provides time-series IC summary from stored IC history.
"""
import json
import os

from hedge_fund_ai.reporting.llm import ask_llm

_IC_FILE = os.path.join(os.path.dirname(__file__), "..", "state", "ic_history.json")


def compute_factor_attribution(portfolio_data: list[dict]) -> dict:
    """Compute weighted factor exposures across portfolio."""
    factors = {
        "momentum_12_1": 0, "momentum": 0, "quality": 0, "growth": 0,
        "value": 0, "sentiment": 0, "flow_signal": 0, "earnings_surprise": 0,
        "liquidity": 0, "volatility": 0,
    }
    total_w = sum(p.get("weight", 0) for p in portfolio_data) or 1.0

    for p in portfolio_data:
        w = float(p.get("weight", 0)) / total_w
        feats = p.get("features", {})
        factors["momentum_12_1"]    += w * float(feats.get("momentum_12_1", 0) or (p.get("ret_3m") or 0))
        factors["momentum"]         += w * float(feats.get("momentum",      0) or (p.get("ret_3m") or 0))
        factors["quality"]          += w * float(p.get("roe_pct")          or 0)
        factors["growth"]           += w * float(p.get("revenue_growth")   or 0)
        factors["sentiment"]        += w * float(p.get("sentiment_score")  or 0)
        factors["flow_signal"]      += w * float(feats.get("flow_signal",  0))
        factors["earnings_surprise"]+= w * float(p.get("earnings_surprise")or 0)
        factors["liquidity"]        += w * float(feats.get("liquidity",    0))
        factors["volatility"]       += w * float(p.get("volatility")       or 0)

    return {k: round(v, 4) for k, v in factors.items()}


def get_ic_summary() -> str:
    """Return recent IC stats per factor for inclusion in reports."""
    if not os.path.exists(_IC_FILE):
        return "No IC history available yet."
    try:
        with open(_IC_FILE) as f:
            history = json.load(f)
        lines = []
        for factor, vals in sorted(history.items()):
            if not vals:
                continue
            import numpy as np
            arr  = np.array(vals[-12:])
            mean = float(np.mean(arr))
            std  = float(np.std(arr)) + 1e-6
            ir   = mean / std
            lines.append(f"  {factor:<22}: IC={mean:+.3f}  IR={ir:+.2f}  n={len(arr)}")
        return "\n".join(lines) if lines else "IC history empty."
    except Exception as e:
        return f"IC history read error: {e}"


def explain_factor_attribution(factors: dict) -> str:
    ic_summary = get_ic_summary()

    prompt = f"""You are a quantitative portfolio manager explaining factor attribution.

WEIGHTED FACTOR EXPOSURES (z-scored, portfolio-weighted)
{chr(10).join(f'  {k}: {v:+.3f}' for k, v in factors.items())}

ROLLING IC HISTORY (last 12 periods, Spearman rank correlation with forward returns)
{ic_summary}

Provide a 3-paragraph factor attribution analysis:
1. Which factors are driving alpha (positive IC + positive exposure)
2. Which factors are a drag or risk (negative IC or unfavorable regime tilts)
3. Recommended factor weight adjustments for next period

Be precise. Reference actual IC values. Do not pad."""

    return ask_llm(prompt, max_tokens=500)
