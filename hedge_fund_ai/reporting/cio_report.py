"""
CIO daily report with rich structured prompt.
"""
from hedge_fund_ai.reporting.llm import ask_llm


def generate_cio_report(metrics: dict, top_stocks: list, macro: dict) -> str:
    m = metrics or {}
    regime = macro.get("regime", "NEUTRAL")
    vix    = macro.get("vix", "N/A")
    spy3m  = macro.get("spy_3m", "N/A")
    credit = macro.get("credit_spread_signal", "N/A")

    positions_str = ", ".join(
        f"{t['ticker']} ({t['weight']*100:.1f}%, score={t.get('score', 0):.2f})"
        if isinstance(t, dict) else str(t)
        for t in (top_stocks or [])[:8]
    )

    prompt = f"""You are the CIO of a quantitative hedge fund writing today's investor brief.

PERFORMANCE METRICS
  Sharpe:        {m.get('sharpe', 'N/A')}
  Sortino:       {m.get('sortino', 'N/A')}
  Calmar:        {m.get('calmar', 'N/A')}
  Max Drawdown:  {m.get('max_drawdown', 'N/A')}
  CAGR:          {m.get('cagr', 'N/A')}
  Info Ratio:    {m.get('info_ratio', 'N/A')}
  Alpha (ann):   {m.get('alpha_ann', 'N/A')}
  Beta vs SPY:   {m.get('beta', 'N/A')}

MACRO ENVIRONMENT
  Regime:        {regime}
  VIX:           {vix}
  SPY 3m ret:    {spy3m}%
  Credit signal: {credit}

TOP POSITIONS
  {positions_str}

Write a concise, precise CIO brief (4-5 paragraphs) covering:
1. Market regime and macro context
2. Portfolio performance vs benchmark (use the numbers above)
3. Factor attribution: which signals are working
4. Key risks and tail scenarios
5. Tactical outlook and any planned adjustments

Tone: institutional, direct, data-driven. No hype. Cite specific numbers."""

    return ask_llm(prompt, max_tokens=700)
