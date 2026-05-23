from hedge_fund_ai.reporting.llm import ask_llm


def explain_trade(position: dict) -> str:
    t = position
    prompt = f"""You are a portfolio manager explaining a position to an investor.

POSITION: {t.get('ticker')}
  Weight:           {t.get('weight', 0)*100:.1f}%
  Composite Score:  {t.get('score', 'N/A')}
  Sector:           {t.get('sector', 'N/A')}
  3m Return:        {t.get('ret_3m', 'N/A')}%
  Annual Vol:       {t.get('volatility', 'N/A')}%
  ROE:              {t.get('roe_pct', 'N/A')}%
  Revenue Growth:   {t.get('revenue_growth', 'N/A')}%
  Sentiment Score:  {t.get('sentiment_score', 'N/A')}
  News Volume:      {t.get('news_volume', 'N/A')} articles

Explain in 3-4 sentences:
- Primary reason for selection (which factor drove the score)
- Key upside catalyst
- Main downside risk
- Sizing rationale (why {t.get('weight', 0)*100:.1f}% weight)"""

    return ask_llm(prompt, max_tokens=250)
