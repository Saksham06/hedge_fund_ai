from hedge_fund_ai.reporting.llm import ask_llm


def analyze_strategy(metrics: dict) -> str:
    m = metrics or {}
    prompt = f"""You are a hedge fund risk committee evaluating a quantitative equity strategy.

STRATEGY STATISTICS
  Sharpe Ratio:      {m.get('sharpe', 'N/A')} (target: > 1.0)
  Sortino Ratio:     {m.get('sortino', 'N/A')} (target: > 1.5)
  Calmar Ratio:      {m.get('calmar', 'N/A')} (target: > 0.5)
  Max Drawdown:      {m.get('max_drawdown', 'N/A')} (limit: -20%)
  Max DD Duration:   {m.get('max_dd_duration', 'N/A')} trading days
  CAGR:              {m.get('cagr', 'N/A')}
  Annualized Vol:    {m.get('ann_vol', 'N/A')}
  Win Rate:          {m.get('win_rate', 'N/A')}
  Profit Factor:     {m.get('profit_factor', 'N/A')}
  Alpha (annualized):{m.get('alpha_ann', 'N/A')}
  Beta:              {m.get('beta', 'N/A')}
  Info Ratio:        {m.get('info_ratio', 'N/A')}

Provide a structured risk committee assessment:
1. VERDICT: Allocate / Monitor / Reduce / Exit — with brief rationale
2. RISK FLAGS: List any metrics that breach acceptable limits
3. IMPROVEMENTS: 3 specific, actionable suggestions to improve risk-adjusted returns
4. SCENARIO ANALYSIS: How would this strategy perform in (a) 2008-style crash (b) 2022 rate shock

Be precise, numerical, and professional. No boilerplate."""

    return ask_llm(prompt, max_tokens=600)
