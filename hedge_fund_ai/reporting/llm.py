"""
LLM client for Groq API with structured prompts.
All prompts inject actual numbers — no vague placeholders.
"""
import logging

import requests
from requests import HTTPError

from hedge_fund_ai.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL

logger = logging.getLogger(__name__)


def _extract_text(payload: dict) -> str:
    if "error" in payload:
        return f"LLM error: {payload['error']}"
    choices = payload.get("choices") or []
    if choices:
        msg = choices[0].get("message", {})
        return str(msg.get("content", ""))
    return "LLM returned unexpected format."


def ask_llm(prompt: str, max_tokens: int = 600, temperature: float = 0.3) -> str:
    if not GROQ_API_KEY:
        return "[LLM not configured — set GROQ_API_KEY]"
    try:
        r = requests.post(
            GROQ_BASE_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=15,
        )
        r.raise_for_status()
        return _extract_text(r.json())
    except HTTPError as e:
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                return _extract_text(resp.json())
            except Exception:
                return f"LLM HTTP error: {resp.text[:200]}"
        return f"LLM HTTP error: {e}"
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
        return f"LLM error: {e}"
