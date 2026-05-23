"""
Phase 4 & 5: Q&A Engine — orchestrates the full question-answering pipeline.

Pipeline per query:
  1. route_query()        → ParsedQuery (type + entities + confidence)
  2. Load context         → metrics_cache snapshot + audit + state files
  3. Cache check          → return cached response if hash matches
  4. build_prompt()       → inject grounded data into template
  5. ask_llm()            → Groq API call with token limits
  6. validate_response()  → cross-reference numeric claims
  7. log_query_audit()    → append-only compliance log
  8. cache result         → keyed by (question_type + entities + date_range)

Conversation memory:
  - Last 5 exchanges stored per session in memory (not persisted — no PII)
  - Auto-expires after 30 minutes
  - Injected as conversation history for context continuity

Cost controls:
  - Per-type token limits
  - Query result cache (hash of type + entities + date)
  - Async batch for daily summaries
"""

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_RUN_ID = os.getenv("RUN_ID", "SYSTEM")

# Per-type token limits
_TOKEN_LIMITS = {
    "explain_pnl":     1_200,
    "explain_factor":  1_000,
    "explain_risk":    1_000,
    "answer_audit":    1_200,
    "strategy_health": 1_500,
    "meta_help":       500,
}

# Query result cache: {hash: (timestamp, response)}
_QUERY_CACHE: dict[str, tuple[float, str]] = {}
_CACHE_TTL = 300   # 5-minute cache per identical query


def _query_hash(question_type: str, entities: dict, date_range: str) -> str:
    key = json.dumps({
        "type":       question_type,
        "entities":   entities,
        "date_range": date_range,
    }, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _load_context() -> dict:
    """Load all metric sources into a single context dict."""
    from hedge_fund_ai.llm.metrics_cache import get_metrics_snapshot

    ctx = {}
    try:
        snap = get_metrics_snapshot()
        ctx["pnl_last_30d"] = snap.get("pnl_last_30d", [])
        ctx["factor_ic"]    = snap.get("factor_ic", {})
        ctx["regime"]       = snap.get("regime", {})
    except Exception as e:
        logger.warning("Metrics cache unavailable: %s", e)

    state_dir = os.path.join(os.path.dirname(__file__), "..", "state")

    for key, filename in [
        ("metrics",    "last_metrics.json"),
        ("signal_ic",  "ic_report.json"),
        ("risk_state", "risk_state.json"),
    ]:
        path = os.path.join(state_dir, filename)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    ctx[key] = json.load(f)
            except Exception:
                ctx[key] = {}
        else:
            ctx[key] = {}

    metrics = ctx.get("metrics", {})
    ctx["placebo"]     = metrics.get("placebo_test", {})
    ctx["monte_carlo"] = metrics.get("monte_carlo", {})

    # Audit log (last 20 records)
    audit_path = os.path.join(state_dir, "audit_log.json")
    if os.path.exists(audit_path):
        try:
            with open(audit_path) as f:
                audit_full = json.load(f)
            ctx["audit"] = audit_full[-20:]
        except Exception:
            ctx["audit"] = []
    else:
        ctx["audit"] = []

    return ctx


# ── Conversation Memory ────────────────────────────────────────────────────────

class ConversationMemory:
    """
    Stores last N exchanges per session. In-memory only — no PII persisted.
    Auto-expires after EXPIRE_SECONDS of inactivity.
    """
    EXPIRE_SECONDS = 1800   # 30 minutes
    MAX_TURNS      = 5

    def __init__(self):
        self._sessions: dict[str, dict] = {}

    def _clean(self, session_id: str):
        sess = self._sessions.get(session_id)
        if sess and time.time() - sess["last_active"] > self.EXPIRE_SECONDS:
            del self._sessions[session_id]

    def add(self, session_id: str, question: str, answer: str):
        self._clean(session_id)
        if session_id not in self._sessions:
            self._sessions[session_id] = {"turns": [], "last_active": time.time()}
        sess = self._sessions[session_id]
        sess["turns"].append({"q": question[:300], "a": answer[:500]})
        sess["turns"] = sess["turns"][-self.MAX_TURNS:]
        sess["last_active"] = time.time()

    def get_history(self, session_id: str) -> list[dict]:
        self._clean(session_id)
        sess = self._sessions.get(session_id)
        return sess["turns"] if sess else []

    def clear(self, session_id: str):
        self._sessions.pop(session_id, None)


_memory = ConversationMemory()


# ── Main Q&A function ─────────────────────────────────────────────────────────

def answer_question(
    question:    str,
    session_id:  str = "default",
    run_id:      str = _RUN_ID,
) -> dict:
    """
    End-to-end question answering with validation, caching, and audit logging.

    Returns:
        {
          question_type, confidence, validation_passed,
          response, disclaimer, cache_hit,
          claim_checks: [...],
          session_history: [...]
        }
    """
    from hedge_fund_ai.llm.query_router import route_query
    from hedge_fund_ai.llm.prompt_templates import build_prompt
    from hedge_fund_ai.llm.response_validator import validate_response
    from hedge_fund_ai.llm.metrics_cache import log_query_audit
    from hedge_fund_ai.reporting.llm import ask_llm

    t_start = time.time()

    # Step 1: Route
    parsed = route_query(question)
    logger.info("Query routed: type=%s conf=%.2f fallback=%s entities=%s",
                parsed.question_type, parsed.confidence,
                parsed.fallback, parsed.entities)

    # Step 2: Load context
    ctx = _load_context()
    ctx["entities"] = parsed.entities

    # Step 3: Cache check
    date_range = parsed.entities.get("date_from", "")
    q_hash     = _query_hash(parsed.question_type, parsed.entities, date_range)
    cached     = _QUERY_CACHE.get(q_hash)
    cache_hit  = False

    if cached and (time.time() - cached[0]) < _CACHE_TTL:
        logger.debug("Cache hit: %s", q_hash)
        cached_response = cached[1]
        cache_hit = True
        return {
            "question_type":    parsed.question_type,
            "confidence":       parsed.confidence,
            "validation_passed":True,
            "response":         cached_response,
            "disclaimer":       None,
            "cache_hit":        True,
            "claim_checks":     [],
            "session_history":  _memory.get_history(session_id),
            "elapsed_ms":       round((time.time() - t_start) * 1000, 1),
        }

    # Step 4: Build prompt with conversation history
    history = _memory.get_history(session_id)
    prompt  = build_prompt(parsed.question_type, ctx)

    if history:
        history_block = "\n".join(
            f"Q: {t['q']}\nA: {t['a']}" for t in history[-2:]
        )
        prompt = f"CONVERSATION HISTORY (last 2 turns):\n{history_block}\n\n---\n\n{prompt}"

    # Step 5: LLM call with per-type token limit
    max_tokens = _TOKEN_LIMITS.get(parsed.question_type, 800)
    raw_resp   = ask_llm(prompt, max_tokens=max_tokens, temperature=0.2)

    # Step 6: Validate
    validated = validate_response(raw_resp, ctx)

    # Step 7: Cache result
    _QUERY_CACHE[q_hash] = (time.time(), validated.final_response)

    # Step 8: Audit log
    try:
        log_query_audit(
            query_hash      = q_hash,
            question_type   = parsed.question_type,
            question_text   = question,
            confidence      = validated.confidence,
            validation_pass = validated.all_passed,
            response_length = len(validated.final_response),
            run_id          = run_id,
        )
    except Exception as e:
        logger.debug("Audit log: %s", e)

    # Step 9: Update conversation memory
    _memory.add(session_id, question, validated.plain_text[:400])

    elapsed = round((time.time() - t_start) * 1000, 1)
    logger.info("Q&A complete: type=%s conf=%.2f valid=%s elapsed=%dms",
                parsed.question_type, validated.confidence,
                validated.all_passed, elapsed)

    return {
        "question_type":    parsed.question_type,
        "confidence":       validated.confidence,
        "validation_passed":validated.all_passed,
        "response":         validated.final_response,
        "disclaimer":       validated.disclaimer,
        "cache_hit":        cache_hit,
        "claim_checks":     [
            {"claim": c.claim_text, "passed": c.passed,
             "claimed": c.claimed_val, "actual": c.actual_val,
             "delta_pct": c.delta_pct}
            for c in validated.claim_checks
        ],
        "session_history":  _memory.get_history(session_id),
        "elapsed_ms":       elapsed,
    }
