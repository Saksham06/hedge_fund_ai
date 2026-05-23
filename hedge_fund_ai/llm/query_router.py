"""
Phase 2: Query Router — NLP classifier mapping questions to structured types.

Architecture:
  1. Keyword + regex pattern matching (fast, no model needed)
  2. Entity extraction: ticker, date_range, metric_name, factor_name
  3. Confidence scoring: 0.0–1.0 based on match quality
  4. Fallback routing: confidence < 0.8 → structured_fallback type

Question types:
  explain_pnl       "Why did the portfolio lose money last week?"
  explain_factor     "Which factor is working best right now?"
  explain_risk       "What's our current drawdown?"
  answer_audit       "What happened on the rebalance on Jan 15?"
  strategy_health    "Is the signal still valid? Should we deploy?"
  meta_help          "What can you answer?" / "How do you work?"
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


@dataclass
class ParsedQuery:
    raw_text:      str
    question_type: str
    confidence:    float
    entities:      dict = field(default_factory=dict)
    fallback:      bool = False


# ── Pattern definitions ────────────────────────────────────────────────────────

_TICKER_RE = re.compile(r'\b([A-Z]{1,5})\b')
_DATE_RE   = re.compile(
    r'\b(\d{4}-\d{2}-\d{2}|'
    r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}(?:,\s*\d{4})?|'
    r'(last\s+(?:week|month|quarter|year)|yesterday|today|'
    r'(?:past|last)\s+\d+\s+(?:days?|weeks?|months?)))\b',
    re.IGNORECASE
)

_KNOWN_TICKERS = {
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","UNH","XOM",
    "AVGO","LLY","V","MA","PG","COST","HD","KO","PEP","ADBE","CRM",
    "CSCO","ORCL","AMD","QCOM","NFLX","DIS","NKE","BAC","WMT","CVX",
    "SPY","QQQ","IWM","VIX","TLT","HYG","IEF",
}

_KNOWN_FACTORS = {
    "momentum","momentum_12_1","quality","value","growth","sentiment",
    "flow_signal","earnings_surprise","liquidity","piotroski","roic",
    "volatility","skewness","accruals","fmp_alpha","rel_strength",
    "sector_momentum","high_52w","price_accel","reversal","trend_slope",
}

_KNOWN_METRICS = {
    "sharpe","sortino","calmar","drawdown","max_dd","cagr","return",
    "alpha","beta","information_ratio","tracking_error","win_rate",
    "turnover","cost","slippage","ic","ic_ir","hit_rate","pnl",
    "exposure","weight","position","portfolio_value",
}

# (pattern_list, question_type, base_confidence)
_RULES = [
    # P&L explanation
    (["why.*(?:lost|lose|down|negative|underperform|drop|fell|decline)",
      ".*(?:lose|lost).*money",
      "(?:pnl|p&l|profit|loss|return|performance).*(?:why|explain|what caused)",
      "what.*(?:caused|drove|happened to).*(?:return|performance|portfolio)",
      "how did.*(?:portfolio|fund|strategy).*(?:do|perform)"],
     "explain_pnl", 0.88),

    # Factor explanation
    (["(?:which|what).*factor.*(?:work|best|top|worst|lead|driv|contribut)",
      "factor.*(?:attribution|breakdown|decompos|contribution|explai)",
      "(?:momentum|quality|value|sentiment|flow|liquidity|piotroski).*(?:score|signal|ic|work)",
      "how.*(?:factor|signal).*perform",
      "(?:ic|information coefficient).*(?:factor|signal)"],
     "explain_factor", 0.90),

    # Risk / drawdown
    (["(?:drawdown|risk|volatility|var|cvar|exposure)",
      "(?:how much|what is).*(?:risk|loss|drawdown|down from)",
      "stop.?loss|circuit.?breaker|halt",
      "(?:current|today).*(?:risk|exposure|position)",
      "(?:portfolio|strategy).*(?:risk|safe|dangerous)"],
     "explain_risk", 0.87),

    # Audit / historical decision
    (["(?:what|why).*(?:rebalanc|trade|buy|sell|decision|chose)",
      "(?:on|at|during).*(?:rebalanc|trade)",
      "audit.*(?:log|trail|record|history)",
      "(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\\d{4}-\\d{2}-\\d{2}).*(?:rebalanc|trade|decision)",
      "(?:last|previous).*(?:rebalanc|trade)"],
     "answer_audit", 0.85),

    # Strategy health / deployment
    (["(?:should|can|ready).*(?:deploy|invest|trade|live)",
      "(?:is|are).*(?:signal|strategy|factor).*(?:valid|working|good|real|alpha)",
      "(?:ic|information coefficient).*(?:positive|negative|valid|significant)",
      "(?:placebo|monte carlo|backtest).*(?:result|show|say|indicate)",
      "(?:deploy|invest|trade).*(?:capital|money|real)"],
     "strategy_health", 0.88),

    # Meta / help
    (["(?:what can you|what do you|how do you|help|assist)",
      "(?:what question|what topic|what can i ask)",
      "(?:explain|describe).*(?:yourself|how you work|your capabilities)",
      "hello|hi |hey |help me"],
     "meta_help", 0.95),
]


def _extract_entities(text: str) -> dict:
    entities = {}

    # Tickers
    candidates = _TICKER_RE.findall(text.upper())
    tickers = [t for t in candidates if t in _KNOWN_TICKERS]
    if tickers:
        entities["tickers"] = tickers

    # Factors
    words = set(re.split(r'\W+', text.lower()))
    factors = [f for f in _KNOWN_FACTORS if f in words or f.replace("_", " ") in text.lower()]
    if factors:
        entities["factors"] = factors

    # Metrics
    metrics = [m for m in _KNOWN_METRICS if m in words]
    if metrics:
        entities["metrics"] = metrics

    # Dates / date ranges
    date_matches = _DATE_RE.findall(text)
    if date_matches:
        entities["date_refs"] = [m[0] for m in date_matches if m[0]]

    # Specific date ranges
    now = datetime.now(timezone.utc)
    if re.search(r'last\s+week', text, re.I):
        entities["date_from"] = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        entities["date_to"]   = now.strftime("%Y-%m-%d")
    elif re.search(r'last\s+month', text, re.I):
        entities["date_from"] = (now - timedelta(days=30)).strftime("%Y-%m-%d")
        entities["date_to"]   = now.strftime("%Y-%m-%d")
    elif re.search(r'last\s+quarter', text, re.I):
        entities["date_from"] = (now - timedelta(days=90)).strftime("%Y-%m-%d")
        entities["date_to"]   = now.strftime("%Y-%m-%d")

    return entities


def _match_rules(text: str) -> tuple[str, float]:
    """Return (question_type, confidence) for the best-matching rule."""
    text_lower = text.lower()
    best_type  = "meta_help"
    best_conf  = 0.0

    for patterns, q_type, base_conf in _RULES:
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                # Boost confidence if multiple patterns match
                score = base_conf + 0.05 * (
                    sum(1 for p in patterns if re.search(p, text_lower, re.I)) - 1
                )
                score = min(score, 0.99)
                if score > best_conf:
                    best_conf = score
                    best_type = q_type
                break

    return best_type, best_conf


def route_query(question: str) -> ParsedQuery:
    """
    Parse and route a natural language question.
    Returns a ParsedQuery with type, confidence, and extracted entities.
    """
    q = question.strip()
    if not q:
        return ParsedQuery(raw_text=q, question_type="meta_help",
                           confidence=1.0, fallback=False)

    q_type, conf = _match_rules(q)
    entities     = _extract_entities(q)
    fallback     = conf < 0.80

    if fallback:
        q_type = "meta_help"

    return ParsedQuery(
        raw_text      = q,
        question_type = q_type,
        confidence    = round(conf, 3),
        entities      = entities,
        fallback      = fallback,
    )
