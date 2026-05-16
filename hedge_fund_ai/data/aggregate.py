"""
Aggregate news sentiment for a ticker.
Uses LLM batch scoring when available, keyword fallback otherwise.
"""
from hedge_fund_ai.data.decay import time_decay
from hedge_fund_ai.data.sentiment import analyze_batch, analyze_text


def aggregate_advanced(news_items: list[dict], ticker: str = "") -> float:
    """
    Returns a weighted sentiment score in [-1, +1].
    - Uses LLM to score all headlines in a single API call
    - Applies exponential time-decay (half-life=24h)
    - Weights by recency and confidence
    """
    if not news_items:
        return 0.0

    # LLM-score all headlines in one call
    enriched = analyze_batch(news_items, ticker=ticker)

    total_score = 0.0
    total_weight = 0.0

    for item in enriched:
        text = item.get("text", "")
        ts = item.get("published_at")

        # Prefer LLM score; fallback to keyword
        if "llm_sentiment" in item:
            sentiment = item["llm_sentiment"]
            confidence = 0.85  # LLM is more reliable
        else:
            res = analyze_text(text)
            sentiment = res["sentiment"]
            confidence = res["confidence"]

        decay = time_decay(ts)
        weight = max(0.01, confidence * decay)
        total_score += sentiment * weight
        total_weight += weight

    return float(total_score / total_weight) if total_weight else 0.0
