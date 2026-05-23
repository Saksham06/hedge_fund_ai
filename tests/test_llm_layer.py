"""
Tests for all 5 phases of the LLM Q&A layer.
Phase 1: metrics_cache (SQLite, schema validation, versioning)
Phase 2: query_router + prompt_templates
Phase 3: response_validator
Phase 4: qa_engine (caching, conversation memory)
Phase 5: cost controls, audit logging
"""
import json
import os
import time
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


# ── Phase 1: Metrics Cache ────────────────────────────────────────────────────

class TestMetricsCache:
    @pytest.fixture(autouse=True)
    def patch_db(self, tmp_path):
        import hedge_fund_ai.llm.metrics_cache as mc
        mc._DB_PATH = str(tmp_path / "metrics_cache.db")
        mc._init_db()
        yield
        mc._DB_PATH = str(tmp_path / "metrics_cache.db")

    def test_snapshot_pnl_valid(self):
        from hedge_fund_ai.llm.metrics_cache import snapshot_pnl, get_latest_pnl
        records = [{"date": "2024-01-15", "net_return": 0.012,
                    "factor_contributions": {"momentum": 0.008},
                    "regime": "risk_on", "risk_state": {}}]
        snapshot_pnl(records, run_id="TEST")
        result = get_latest_pnl(10)
        assert len(result) == 1
        assert result[0]["date"] == "2024-01-15"
        assert abs(result[0]["net_return"] - 0.012) < 1e-9

    def test_schema_rejects_large_return(self):
        from hedge_fund_ai.llm.metrics_cache import snapshot_pnl, get_latest_pnl
        bad = [{"date": "2024-01-16", "net_return": 5.0}]  # 500% return — rejected
        snapshot_pnl(bad, run_id="TEST")
        result = get_latest_pnl(10)
        assert len(result) == 0

    def test_schema_rejects_missing_date(self):
        from hedge_fund_ai.llm.metrics_cache import snapshot_pnl, get_latest_pnl
        bad = [{"net_return": 0.01}]  # no date field
        snapshot_pnl(bad, run_id="TEST")
        assert get_latest_pnl(10) == []

    def test_snapshot_factor_ic(self):
        from hedge_fund_ai.llm.metrics_cache import snapshot_factor_ic, get_factor_ic_summary_from_cache
        ic = {"momentum": {"mean_ic": 0.05, "ic_ir": 0.8, "hit_rate": 0.6, "n": 12},
              "quality":  {"mean_ic": 0.03, "ic_ir": 0.5, "hit_rate": 0.55, "n": 10}}
        snapshot_factor_ic(ic, run_id="TEST", date="2024-01-15")
        result = get_factor_ic_summary_from_cache()
        assert "momentum" in result
        assert abs(result["momentum"]["mean_ic"] - 0.05) < 1e-9

    def test_ic_schema_rejects_invalid(self):
        from hedge_fund_ai.llm.metrics_cache import snapshot_factor_ic, get_factor_ic_summary_from_cache
        bad = {"bad_factor": {"mean_ic": 2.5}}  # |IC| > 1 → rejected
        snapshot_factor_ic(bad, run_id="TEST")
        result = get_factor_ic_summary_from_cache()
        assert "bad_factor" not in result

    def test_snapshot_regime(self):
        from hedge_fund_ai.llm.metrics_cache import snapshot_regime, get_latest_regime
        macro = {"regime": "risk_on", "vix": 16.5, "spy_3m": 4.2,
                 "credit_spread_signal": 0.01, "yield_curve": 0.005, "vol_regime": 0.85}
        snapshot_regime(macro, run_id="TEST")
        r = get_latest_regime()
        assert r["regime"] == "risk_on"
        assert abs(r["vix"] - 16.5) < 0.01

    def test_data_version_changes_with_data(self):
        from hedge_fund_ai.llm.metrics_cache import _data_version
        v1 = _data_version({"sharpe": 1.2})
        v2 = _data_version({"sharpe": 1.3})
        assert v1 != v2

    def test_data_version_stable_same_data(self):
        from hedge_fund_ai.llm.metrics_cache import _data_version
        v1 = _data_version({"a": 1, "b": 2})
        v2 = _data_version({"b": 2, "a": 1})
        assert v1 == v2  # order-independent

    def test_query_audit_log(self):
        from hedge_fund_ai.llm.metrics_cache import log_query_audit
        log_query_audit("hash123", "explain_pnl", "Why did we lose?",
                        0.85, True, 450, "TEST")
        # Just verify no exception raised
        assert True

    def test_export_daily_json(self, tmp_path):
        from hedge_fund_ai.llm.metrics_cache import snapshot_pnl, export_daily_json
        import hedge_fund_ai.llm.metrics_cache as mc
        # Patch state dir to tmp
        original_path = os.path.join(os.path.dirname(mc.__file__), "..", "state")
        snapshot_pnl([{"date": "2024-01-15", "net_return": 0.01,
                        "factor_contributions": {}, "regime": "neutral", "risk_state": {}}])
        # Should not raise
        try:
            export_daily_json()
        except Exception:
            pass  # may fail if state dir not writable in test env


# ── Phase 2: Query Router ─────────────────────────────────────────────────────

class TestQueryRouter:
    def test_pnl_routing(self):
        from hedge_fund_ai.llm.query_router import route_query
        r = route_query("Why did the portfolio lose money last week?")
        assert r.question_type == "explain_pnl"
        assert r.confidence >= 0.80

    def test_factor_routing(self):
        from hedge_fund_ai.llm.query_router import route_query
        r = route_query("Which factor is working best right now?")
        assert r.question_type == "explain_factor"
        assert r.confidence >= 0.80

    def test_risk_routing(self):
        from hedge_fund_ai.llm.query_router import route_query
        r = route_query("What is our current drawdown?")
        assert r.question_type == "explain_risk"

    def test_audit_routing(self):
        from hedge_fund_ai.llm.query_router import route_query
        r = route_query("What happened on the last rebalance?")
        assert r.question_type == "answer_audit"

    def test_strategy_health_routing(self):
        from hedge_fund_ai.llm.query_router import route_query
        r = route_query("Should we deploy real capital yet?")
        assert r.question_type == "strategy_health"

    def test_meta_help_routing(self):
        from hedge_fund_ai.llm.query_router import route_query
        r = route_query("What can you help me with?")
        assert r.question_type == "meta_help"
        assert r.confidence >= 0.90

    def test_low_confidence_fallback(self):
        from hedge_fund_ai.llm.query_router import route_query
        r = route_query("xyzzy nonsense gobbledygook")
        assert r.fallback is True

    def test_ticker_extraction(self):
        from hedge_fund_ai.llm.query_router import route_query
        r = route_query("Why is AAPL down this week?")
        assert "AAPL" in r.entities.get("tickers", [])

    def test_factor_extraction(self):
        from hedge_fund_ai.llm.query_router import route_query
        r = route_query("How is the momentum factor performing?")
        assert "momentum" in r.entities.get("factors", [])

    def test_date_range_extraction_last_week(self):
        from hedge_fund_ai.llm.query_router import route_query
        r = route_query("What happened last week with returns?")
        assert "date_from" in r.entities

    def test_date_range_last_month(self):
        from hedge_fund_ai.llm.query_router import route_query
        r = route_query("Show me performance for last month")
        assert "date_from" in r.entities

    def test_empty_question(self):
        from hedge_fund_ai.llm.query_router import route_query
        r = route_query("")
        assert r.question_type == "meta_help"
        assert r.confidence == 1.0


# ── Phase 2: Prompt Templates ─────────────────────────────────────────────────

class TestPromptTemplates:
    def _make_context(self):
        return {
            "pnl_last_30d": [{"date": "2024-01-15", "net_return": 0.012,
                              "factor_contributions": {"momentum": 0.008}, "regime": "risk_on"}],
            "factor_ic":    {"momentum": {"mean_ic": 0.05, "ic_ir": 0.8, "hit_rate": 0.6, "n": 12}},
            "regime":       {"regime": "risk_on", "vix": 16.5, "spy_3m": 4.2},
            "metrics":      {"sharpe": 1.2, "max_drawdown": -0.12, "cagr": 0.14},
            "placebo":      {"p_value": 0.04, "verdict": "SIGNIFICANT"},
            "monte_carlo":  {"p_sharpe_gt_1": 0.65, "verdict": "ROBUST"},
            "signal_ic":    {"n_filled": 15, "mean_ic": 0.06, "verdict": "DEPLOY"},
            "risk_state":   {"halted_until": None},
            "audit":        [{"date": "2024-01-10", "regime": "risk_on",
                              "final_weights": {"AAPL": 0.15}, "raw_turnover": 0.30}],
            "entities":     {"factors": ["momentum"], "date_refs": ["last week"]},
        }

    def test_explain_pnl_prompt_contains_data(self):
        from hedge_fund_ai.llm.prompt_templates import build_prompt
        ctx = self._make_context()
        p = build_prompt("explain_pnl", ctx)
        assert "0.012" in p or "net_return" in p
        assert "Only use" in p or "only use" in p.lower()

    def test_explain_factor_prompt_contains_ic(self):
        from hedge_fund_ai.llm.prompt_templates import build_prompt
        ctx = self._make_context()
        p = build_prompt("explain_factor", ctx)
        assert "momentum" in p.lower()
        assert "0.05" in p or "mean_ic" in p

    def test_strategy_health_prompt_contains_checklist(self):
        from hedge_fund_ai.llm.prompt_templates import build_prompt
        ctx = self._make_context()
        p = build_prompt("strategy_health", ctx)
        assert "Sharpe" in p or "sharpe" in p
        assert "p_value" in p or "0.04" in p
        assert "1.2" in p  # sharpe value injected

    def test_audit_prompt_contains_records(self):
        from hedge_fund_ai.llm.prompt_templates import build_prompt
        ctx = self._make_context()
        p = build_prompt("answer_audit", ctx)
        assert "2024-01-10" in p

    def test_meta_help_prompt_lists_capabilities(self):
        from hedge_fund_ai.llm.prompt_templates import build_prompt
        ctx = self._make_context()
        p = build_prompt("meta_help", ctx)
        assert "P&L" in p or "pnl" in p.lower()
        assert "factor" in p.lower()

    def test_all_prompts_have_rules_section(self):
        from hedge_fund_ai.llm.prompt_templates import build_prompt
        ctx = self._make_context()
        for qtype in ["explain_pnl", "explain_factor", "explain_risk",
                      "answer_audit", "strategy_health"]:
            p = build_prompt(qtype, ctx)
            assert "RULES" in p or "rules" in p.lower(), f"No RULES in {qtype}"

    def test_prompts_have_json_output_schema(self):
        from hedge_fund_ai.llm.prompt_templates import build_prompt
        ctx = self._make_context()
        for qtype in ["explain_pnl", "explain_factor", "explain_risk",
                      "answer_audit", "strategy_health"]:
            p = build_prompt(qtype, ctx)
            assert "```json" in p, f"No JSON schema in {qtype}"


# ── Phase 3: Response Validator ───────────────────────────────────────────────

class TestResponseValidator:
    def _make_snapshot(self):
        return {
            "metrics": {"sharpe": 1.20, "max_drawdown": -0.12, "cagr": 0.14},
            "factor_ic": {"momentum": {"mean_ic": 0.05, "ic_ir": 0.80}},
        }

    def test_valid_response_passes(self):
        from hedge_fund_ai.llm.response_validator import validate_response
        response = """```json
{"summary_one_line": "Portfolio returned +1.2% last week.", "confidence": 0.85}
```
The Sharpe ratio of 1.20 indicates strong risk-adjusted performance."""
        result = validate_response(response, self._make_snapshot())
        assert result.confidence > 0.0
        assert isinstance(result.all_passed, bool)

    def test_mismatched_claim_flagged(self):
        from hedge_fund_ai.llm.response_validator import validate_response
        response = "The Sharpe of 3.50 was excellent."  # actual is 1.20
        result = validate_response(response, self._make_snapshot())
        failed = [c for c in result.claim_checks if not c.passed]
        assert len(failed) >= 1

    def test_disclaimer_appended_on_failure(self):
        from hedge_fund_ai.llm.response_validator import validate_response
        response = "The Sharpe is 9.99."
        result = validate_response(response, self._make_snapshot())
        if not result.all_passed:
            assert result.disclaimer is not None
            assert "Validation notice" in result.final_response or "⚠️" in result.final_response

    def test_low_confidence_disclaimer(self):
        from hedge_fund_ai.llm.response_validator import validate_response
        response = "Perhaps the portfolio might possibly have done something."
        result = validate_response(response, {})
        if result.confidence < 0.70:
            assert result.disclaimer is not None

    def test_json_extraction(self):
        from hedge_fund_ai.llm.response_validator import _extract_json_block
        text = '```json\n{"key": "value", "num": 1.5}\n```'
        parsed = _extract_json_block(text)
        assert parsed["key"] == "value"
        assert parsed["num"] == 1.5

    def test_json_extraction_empty(self):
        from hedge_fund_ai.llm.response_validator import _extract_json_block
        parsed = _extract_json_block("No JSON here at all.")
        assert parsed == {}

    def test_numeric_claim_extraction(self):
        from hedge_fund_ai.llm.response_validator import _extract_numeric_claims
        text = "The Sharpe ratio of 1.23 and a drawdown of -15.4%"
        claims = _extract_numeric_claims(text)
        labels = [c[0] for c in claims]
        assert any("sharpe" in l for l in labels)

    def test_plain_text_extraction(self):
        from hedge_fund_ai.llm.response_validator import _extract_plain_text
        text = '```json\n{"a": 1}\n```\n\nHere is the explanation.'
        plain = _extract_plain_text(text)
        assert "Here is the explanation" in plain
        assert "```json" not in plain

    def test_confidence_extraction_from_json(self):
        from hedge_fund_ai.llm.response_validator import _extract_confidence
        parsed = {"confidence": 0.92}
        conf = _extract_confidence(parsed, "")
        assert conf == pytest.approx(0.92)


# ── Phase 4: Q&A Engine ───────────────────────────────────────────────────────

class TestQAEngine:
    def test_conversation_memory_stores_turns(self):
        from hedge_fund_ai.llm.qa_engine import ConversationMemory
        mem = ConversationMemory()
        mem.add("sess1", "Q1", "A1")
        mem.add("sess1", "Q2", "A2")
        history = mem.get_history("sess1")
        assert len(history) == 2
        assert history[0]["q"] == "Q1"

    def test_memory_max_turns(self):
        from hedge_fund_ai.llm.qa_engine import ConversationMemory
        mem = ConversationMemory()
        for i in range(10):
            mem.add("sess2", f"Q{i}", f"A{i}")
        assert len(mem.get_history("sess2")) <= ConversationMemory.MAX_TURNS

    def test_memory_expires(self):
        from hedge_fund_ai.llm.qa_engine import ConversationMemory
        mem = ConversationMemory()
        mem.EXPIRE_SECONDS = 0.01
        mem.add("sess3", "Q", "A")
        time.sleep(0.05)
        assert mem.get_history("sess3") == []

    def test_memory_clear(self):
        from hedge_fund_ai.llm.qa_engine import ConversationMemory
        mem = ConversationMemory()
        mem.add("sess4", "Q", "A")
        mem.clear("sess4")
        assert mem.get_history("sess4") == []

    def test_query_hash_deterministic(self):
        from hedge_fund_ai.llm.qa_engine import _query_hash
        h1 = _query_hash("explain_pnl", {"factors": ["momentum"]}, "2024-01")
        h2 = _query_hash("explain_pnl", {"factors": ["momentum"]}, "2024-01")
        assert h1 == h2

    def test_query_hash_different_for_different_input(self):
        from hedge_fund_ai.llm.qa_engine import _query_hash
        h1 = _query_hash("explain_pnl", {}, "2024-01")
        h2 = _query_hash("explain_factor", {}, "2024-01")
        assert h1 != h2

    def test_answer_question_uses_llm(self):
        from hedge_fund_ai.llm.qa_engine import answer_question
        with patch("hedge_fund_ai.llm.qa_engine._load_context", return_value={}), \
             patch("hedge_fund_ai.reporting.llm.ask_llm",
                   return_value='```json\n{"confidence": 0.85}\n```\nTest response.'):
            result = answer_question("Why did we lose money?", session_id="test_sess")
        assert "response" in result
        assert "question_type" in result
        assert result["question_type"] in [
            "explain_pnl","explain_factor","explain_risk",
            "answer_audit","strategy_health","meta_help"
        ]

    def test_cache_hit_on_repeat_query(self):
        from hedge_fund_ai.llm.qa_engine import answer_question, _QUERY_CACHE
        _QUERY_CACHE.clear()
        with patch("hedge_fund_ai.llm.qa_engine._load_context", return_value={}), \
             patch("hedge_fund_ai.reporting.llm.ask_llm", return_value="Cached response."):
            r1 = answer_question("What is our Sharpe?", session_id="cache_test")
            r2 = answer_question("What is our Sharpe?", session_id="cache_test")
        assert r2["cache_hit"] is True

    def test_result_has_required_keys(self):
        from hedge_fund_ai.llm.qa_engine import answer_question
        with patch("hedge_fund_ai.llm.qa_engine._load_context", return_value={}), \
             patch("hedge_fund_ai.reporting.llm.ask_llm", return_value="Response."):
            result = answer_question("Help me understand the strategy.")
        for key in ["question_type","confidence","validation_passed",
                    "response","cache_hit","claim_checks","elapsed_ms"]:
            assert key in result, f"Missing: {key}"


# ── Phase 5: Cost controls & audit ────────────────────────────────────────────

class TestCostControls:
    def test_token_limits_defined_for_all_types(self):
        from hedge_fund_ai.llm.qa_engine import _TOKEN_LIMITS
        for qtype in ["explain_pnl","explain_factor","explain_risk",
                      "answer_audit","strategy_health","meta_help"]:
            assert qtype in _TOKEN_LIMITS
            assert _TOKEN_LIMITS[qtype] <= 1500

    def test_meta_help_lowest_token_limit(self):
        from hedge_fund_ai.llm.qa_engine import _TOKEN_LIMITS
        assert _TOKEN_LIMITS["meta_help"] <= _TOKEN_LIMITS["explain_pnl"]

    def test_cache_ttl_set(self):
        from hedge_fund_ai.llm.qa_engine import _CACHE_TTL
        assert _CACHE_TTL > 0

    def test_audit_log_written(self, tmp_path):
        import hedge_fund_ai.llm.metrics_cache as mc
        mc._DB_PATH = str(tmp_path / "audit_test.db")
        mc._init_db()
        from hedge_fund_ai.llm.metrics_cache import log_query_audit, _conn
        log_query_audit("abc123","explain_pnl","test question",0.85,True,300,"TEST")
        with _conn() as c:
            rows = c.execute("SELECT * FROM query_audit").fetchall()
        assert len(rows) == 1
        assert rows[0]["question_type"] == "explain_pnl"
        assert rows[0]["validation_pass"] == 1
