"""
Phase 2: Prompt Templates — 5 structured templates with strict output schema.

Design principles:
  1. Every template injects ONLY actual metrics from metrics_cache — no hallucination
  2. Explicit "Only use provided data" instruction in every prompt
  3. Structured output: JSON block + plain English summary
  4. Few-shot examples calibrate response format
  5. Source citation: every number must name its source field
"""

import json
from datetime import datetime, timezone


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _safe_json(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


# ── Template 1: Explain P&L ───────────────────────────────────────────────────

def explain_pnl_prompt(
    pnl_records: list[dict],
    factor_ic:   dict,
    regime:      dict,
    date_range:  str = "last 30 days",
) -> str:
    return f"""You are a quantitative portfolio analyst explaining P&L to a fund manager.

DATA SNAPSHOT (as of {_ts()})
Only use the numbers below. Do not invent any figures.

DAILY P&L RECORDS (last 30 entries):
{_safe_json(pnl_records[-10:])}

ACTIVE FACTOR IC (which factors predicted returns correctly):
{_safe_json(factor_ic)}

MARKET REGIME:
{_safe_json(regime)}

QUESTION: Explain the portfolio's P&L for the {date_range}.

OUTPUT FORMAT — respond in this exact structure:

```json
{{
  "summary_one_line": "<one sentence>",
  "net_return_cited": "<value from pnl_records, cite date>",
  "top_contributors": ["<factor>: <direction> <magnitude>", ...],
  "top_detractors":   ["<factor>: <direction> <magnitude>", ...],
  "regime_impact":    "<how regime affected factor weights>",
  "confidence":       <0.0-1.0, based on data completeness>
}}
```

Plain English (2-3 paragraphs after JSON):
- Start with net return figure (cite source field and date)
- Name the 2-3 factors that drove most of the return, with IC values
- Explain what the regime was and how it shifted weights
- If data is incomplete, state it explicitly — do not estimate

RULES:
- Never quote a number not present in the data above
- If a field is null or missing, say "data not available"
- Cite source field name for every number (e.g., "from pnl_records.net_return")
"""


# ── Template 2: Explain Factor ────────────────────────────────────────────────

def explain_factor_prompt(
    factor_ic:   dict,
    pnl_records: list[dict],
    regime:      dict,
    factor_name: str | None = None,
) -> str:
    focus = f"Focus specifically on the '{factor_name}' factor." if factor_name else \
            "Rank all factors by IC-IR and explain the top 3."

    return f"""You are a factor researcher explaining signal performance to a portfolio manager.

DATA SNAPSHOT (as of {_ts()})
Only use the numbers below. Do not invent any figures.

FACTOR IC STATISTICS:
{_safe_json(factor_ic)}

RECENT P&L (for context):
{_safe_json(pnl_records[-5:])}

MARKET REGIME:
{_safe_json(regime)}

QUESTION: Which factors are working? {focus}

OUTPUT FORMAT:

```json
{{
  "top_factor": "<name>",
  "top_ic_ir":  <number from factor_ic>,
  "bottom_factor": "<name>",
  "bottom_ic_ir":  <number from factor_ic>,
  "regime_favored_factors": ["<factor>", ...],
  "auto_disabled": ["<factor if ic_ir < 0>", ...],
  "confidence": <0.0-1.0>
}}
```

Plain English (2-3 paragraphs):
- State the best and worst performing factors with their IC and IC-IR values (cite field names)
- Explain why the regime ({regime.get("regime", "unknown")}) makes certain factors stronger or weaker
- Flag any factors with negative IC-IR that may be auto-disabled
- State clearly if IC data is insufficient (n < 6 observations)

RULES:
- Only cite numbers present in the FACTOR IC STATISTICS above
- IC-IR = mean_ic / std_ic — if not provided, do not compute it from other fields
- Do not speculate about future factor performance
"""


# ── Template 3: Explain Risk ──────────────────────────────────────────────────

def explain_risk_prompt(
    pnl_records: list[dict],
    regime:      dict,
    risk_state:  dict,
    metrics:     dict,
) -> str:
    return f"""You are a risk officer explaining current portfolio risk to a fund manager.

DATA SNAPSHOT (as of {_ts()})
Only use the numbers below. Do not invent any figures.

PORTFOLIO METRICS (from backtest/live):
{_safe_json(metrics)}

RECENT P&L (for drawdown context):
{_safe_json(pnl_records[-10:])}

MARKET REGIME:
{_safe_json(regime)}

RISK CONTROLS STATE:
{_safe_json(risk_state)}

QUESTION: What is the current risk posture of the portfolio?

OUTPUT FORMAT:

```json
{{
  "max_drawdown":      "<value from metrics, cite field>",
  "current_regime":    "<from regime.regime>",
  "exposure_scale":    "<from risk_state or metrics>",
  "active_halts":      "<from risk_state.halted_until or none>",
  "stop_losses_at_risk": ["<ticker>: <% from peak>", ...],
  "vol_target_pct":    "<from metrics or config>",
  "risk_level":        "low | medium | high | critical",
  "confidence":        <0.0-1.0>
}}
```

Plain English (2-3 paragraphs):
- State the current drawdown and compare to the -20% halt threshold
- Describe the regime and what it implies for exposure scaling
- Note any active halts, stop-loss positions at risk, or elevated concentration
- If no live data exists yet (paper trading), say so explicitly

RULES:
- Never invent risk numbers — only use what is in the data above
- "risk_level" must be logically derived from the provided numbers, not assumed
"""


# ── Template 4: Answer Audit ──────────────────────────────────────────────────

def answer_audit_prompt(
    audit_records: list[dict],
    date_ref:      str | None = None,
) -> str:
    # Filter to relevant records if date provided
    if date_ref:
        relevant = [r for r in audit_records if r.get("date", "") >= date_ref[:7]][:5]
        if not relevant:
            relevant = audit_records[-5:]
    else:
        relevant = audit_records[-5:]

    return f"""You are a compliance officer explaining a specific rebalance decision from the audit log.

DATA SNAPSHOT (as of {_ts()})
Only use the audit records below. Do not invent decisions.

AUDIT RECORDS:
{_safe_json(relevant)}

QUESTION: Explain what happened on {date_ref or "the most recent rebalance"}.

OUTPUT FORMAT:

```json
{{
  "rebalance_date":    "<from audit.date>",
  "regime":            "<from audit.regime>",
  "positions_opened":  ["<ticker> @ <weight>", ...],
  "positions_closed":  ["<ticker>", ...],
  "raw_turnover":      "<from audit.raw_turnover>",
  "filtered_turnover": "<from audit.filtered_turnover>",
  "cost_bps":          "<audit.total_cost_frac * 10000>",
  "dd_scale":          "<from audit.dd_scale>",
  "universe_size":     "<from audit.universe_size>",
  "confidence":        <0.0-1.0>
}}
```

Plain English (2-3 paragraphs):
- State exactly what was bought and sold on that date, with weights (cite audit record)
- Explain the regime active that day and how it influenced factor weights
- Report the cost and turnover, noting if drawdown control reduced exposure

RULES:
- Only describe decisions in the provided audit records
- If the date is not in the records, say "no audit record found for that date"
- Do not infer or reconstruct decisions not in the data
"""


# ── Template 5: Strategy Health ───────────────────────────────────────────────

def strategy_health_prompt(
    metrics:      dict,
    factor_ic:    dict,
    signal_ic:    dict,
    placebo:      dict,
    monte_carlo:  dict,
) -> str:
    return f"""You are a quantitative researcher giving an honest deployment assessment.

DATA SNAPSHOT (as of {_ts()})
Only use the numbers below. Do not speculate.

BACKTEST METRICS:
{_safe_json(metrics)}

FACTOR IC SUMMARY (out-of-sample):
{_safe_json(factor_ic)}

LIVE SIGNAL IC (paper trading validation):
{_safe_json(signal_ic)}

PLACEBO TEST RESULT:
{_safe_json(placebo)}

MONTE CARLO SIMULATION:
{_safe_json(monte_carlo)}

QUESTION: Is this strategy ready to deploy with real capital?

OUTPUT FORMAT:

```json
{{
  "deployment_verdict": "DEPLOY | PAPER | NOT_READY | DO_NOT_DEPLOY",
  "checks_passed": <integer>,
  "checks_total":  7,
  "critical_failures": ["<specific check that failed>", ...],
  "live_ic_verdict":   "<from signal_ic.verdict>",
  "placebo_p_value":   "<from placebo.p_value>",
  "monte_carlo_verdict":"<from monte_carlo.verdict>",
  "recommended_capital":"<$500-$1000 | $0 | scale up>",
  "confidence":        <0.0-1.0>
}}
```

Deployment checklist (evaluate each explicitly):
✅/❌ Sharpe > 1.0: {metrics.get("sharpe", "N/A")} (need > 1.0)
✅/❌ Beats SPY: excess_vs_spy = {metrics.get("excess_vs_spy", "N/A")}
✅/❌ Beats equal-weight: excess_vs_ew = {metrics.get("excess_vs_ew", "N/A")}
✅/❌ Max DD < 20%: max_drawdown = {metrics.get("max_drawdown", "N/A")}
✅/❌ Placebo p < 0.10: p_value = {placebo.get("p_value", "N/A")}
✅/❌ Live IC > 0 (≥12 periods): {signal_ic.get("n_filled", 0)} periods filled
✅/❌ Monte Carlo robust: {monte_carlo.get("p_sharpe_gt_0", "N/A")} simulations positive

Plain English (3 paragraphs):
- State the verdict and the specific number of checks passed
- Name each failing check with the exact value and what threshold was missed
- Give a concrete next step: either deploy amount, or what to fix first

RULES:
- Never recommend deployment if live IC periods < 12
- Never recommend deployment if placebo p-value > 0.10 and live IC not validated
- State exactly which data is missing rather than making optimistic assumptions
"""


# ── Meta / Help ───────────────────────────────────────────────────────────────

def meta_help_prompt() -> str:
    return f"""You are the AI assistant for an algorithmic hedge fund system.

Current time: {_ts()}

Explain what questions you can answer based on the following data sources:

1. Daily P&L attribution (last 30 days): net returns, factor contributions
2. Factor IC history: rolling information coefficient per factor
3. Market regime: VIX, SPY momentum, credit spread, yield curve
4. Rebalance audit log: every decision with weights, turnover, costs
5. Strategy health: backtest metrics, placebo test, live IC validation

You can answer questions like:
- "Why did the portfolio underperform last week?"
- "Which factor is working best right now?"
- "What is our current drawdown and risk exposure?"
- "What did we buy/sell on the last rebalance?"
- "Should we deploy real capital yet?"

You CANNOT answer:
- Questions about individual stock price targets or earnings forecasts
- Questions about future market direction
- Questions not grounded in the above data sources

Be concise. Cite data sources. Say clearly when you don't have the data.
"""


# ── Template dispatcher ────────────────────────────────────────────────────────

def build_prompt(question_type: str, context: dict) -> str:
    """
    Dispatch to the correct template based on question type.
    context: dict of metric dicts loaded from metrics_cache.
    """
    pnl     = context.get("pnl_last_30d", [])
    ic      = context.get("factor_ic", {})
    regime  = context.get("regime", {})
    metrics = context.get("metrics", {})
    audit   = context.get("audit", [])
    sig_ic  = context.get("signal_ic", {})
    placebo = context.get("placebo", {})
    mc      = context.get("monte_carlo", {})
    risk_st = context.get("risk_state", {})
    entities= context.get("entities", {})

    date_ref = entities.get("date_refs", [None])[0] if entities.get("date_refs") else None
    factor   = entities.get("factors", [None])[0]   if entities.get("factors")   else None

    if question_type == "explain_pnl":
        return explain_pnl_prompt(pnl, ic, regime,
                                  date_range=entities.get("date_from", "last 30 days"))
    elif question_type == "explain_factor":
        return explain_factor_prompt(ic, pnl, regime, factor_name=factor)
    elif question_type == "explain_risk":
        return explain_risk_prompt(pnl, regime, risk_st, metrics)
    elif question_type == "answer_audit":
        return answer_audit_prompt(audit, date_ref=date_ref)
    elif question_type == "strategy_health":
        return strategy_health_prompt(metrics, ic, sig_ic, placebo, mc)
    else:
        return meta_help_prompt()
