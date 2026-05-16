from hedge_fund_ai.reporting.llm import ask_llm


def generate_explanation(result: dict) -> str:
    portfolio = result.get("portfolio", [])
    macro     = result.get("macro_summary", {})
    regime    = result.get("regime", macro.get("regime", "NEUTRAL"))

    top5 = [
        f"{p['ticker']} {p['weight']*100:.1f}%"
        for p in (portfolio or [])[:5]
        if isinstance(p, dict)
    ]

    prompt = f"""You are a hedge fund CIO. Summarize today's portfolio decision in 2 paragraphs.

Regime: {regime}
Top 5 positions: {', '.join(top5)}
Macro: VIX={macro.get('vix', 'N/A')}, SPY 3m={macro.get('spy_3m', 'N/A')}%

Paragraph 1: What the portfolio is positioned for (market thesis)
Paragraph 2: Key risks to monitor

Be direct and quantitative."""
    return ask_llm(prompt, max_tokens=300)
