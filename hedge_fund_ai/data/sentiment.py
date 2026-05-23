"""
Sentiment analysis for news headlines.
Uses Groq LLM when available; falls back to keyword model.

LLM approach: batch headlines per ticker into one prompt → single API call.
Returns a score in [-1, +1] with confidence and event_weight.
"""
import json
import logging

import requests

logger = logging.getLogger(__name__)

# ── Keyword fallback ──────────────────────────────────────────────────────────
POSITIVE = {
    "beat", "beats", "surge", "surges", "rally", "rallies", "upgrade",
    "upgraded", "strong", "growth", "record", "profit", "profits", "bullish",
    "raised", "guidance", "buyback", "dividend", "outperform", "overweight",
    "partnership", "acquisition", "launch", "approved",
}
NEGATIVE = {
    "miss", "misses", "plunge", "plunges", "selloff", "downgrade", "downgraded",
    "weak", "decline", "loss", "losses", "bearish", "lawsuit", "probe", "fraud",
    "recall", "investigation", "cut", "lowered", "warning", "default", "layoff",
    "restructuring", "fine", "penalty",
}


def _keyword_score(text: str) -> dict:
    s = (text or "").lower()
    tokens = {t.strip(".,:;!?()[]{}\"'") for t in s.split() if t}
    pos = len(tokens & POSITIVE)
    neg = len(tokens & NEGATIVE)
    raw = pos - neg
    sentiment = max(-1.0, min(1.0, raw / 3.0)) if raw != 0 else 0.0
    confidence = 0.5 + min(0.5, (pos + neg) / 6.0)
    return {"sentiment": float(sentiment), "confidence": float(confidence), "event_weight": 1.0}


# ── LLM-powered scoring ───────────────────────────────────────────────────────
def _llm_score_batch(headlines: list[str], ticker: str) -> list[float]:
    """
    Score a list of headlines for a single ticker in one LLM call.
    Returns a list of floats in [-1, +1], same length as headlines.
    Falls back to keyword model on any error.
    """
    from hedge_fund_ai.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL

    if not GROQ_API_KEY or not headlines:
        return [_keyword_score(h)["sentiment"] for h in headlines]

    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    prompt = f"""You are a financial sentiment analyst.
Score each headline below for {ticker} stock on a scale from -1.0 (very bearish) to +1.0 (very bullish).
0.0 = neutral/irrelevant. Consider earnings beats/misses, guidance, analyst actions, macro risk, legal issues.

Headlines:
{numbered}

Respond ONLY with a JSON array of numbers, e.g. [0.8, -0.3, 0.0].
One number per headline, in the same order. No explanation."""

    try:
        r = requests.post(
            GROQ_BASE_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 200,
            },
            timeout=10,
        )
        content = r.json()["choices"][0]["message"]["content"].strip()
        scores = json.loads(content)
        if isinstance(scores, list) and len(scores) == len(headlines):
            return [max(-1.0, min(1.0, float(s))) for s in scores]
    except Exception as e:
        logger.debug("LLM sentiment failed for %s: %s — using keyword fallback", ticker, e)

    return [_keyword_score(h)["sentiment"] for h in headlines]


def analyze_text(text: str) -> dict:
    """Single-item interface kept for backward compatibility."""
    return _keyword_score(text)


def analyze_batch(news_items: list[dict], ticker: str = "") -> list[dict]:
    """
    Score all news items for a ticker in one LLM call.
    Returns enriched items with 'llm_sentiment' field added.
    """
    if not news_items:
        return news_items

    headlines = [item.get("text", "") for item in news_items]
    scores = _llm_score_batch(headlines, ticker)

    enriched = []
    for item, score in zip(news_items, scores):
        enriched.append({**item, "llm_sentiment": score})
    return enriched
