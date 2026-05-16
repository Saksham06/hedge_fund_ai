"""
Phase 3: Response Validator — cross-references LLM claims against metrics cache.

Pipeline:
  1. Extract numeric claims from LLM response (JSON block + inline numbers)
  2. Cross-reference each claim against metrics_cache values
  3. Flag mismatches beyond tolerance threshold
  4. Auto-append disclaimer if validation fails or confidence < 0.7
  5. Return ValidatedResponse with pass/fail status per claim
"""

import json
import logging
import re
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

TOLERANCE = 0.05          # 5% relative tolerance for numeric claims
LOW_CONFIDENCE = 0.70     # below this → append disclaimer


@dataclass
class ClaimCheck:
    claim_text:   str
    claimed_val:  float | None
    source_field: str | None
    actual_val:   float | None
    passed:       bool
    delta_pct:    float | None = None


@dataclass
class ValidatedResponse:
    raw_response:   str
    parsed_json:    dict
    plain_text:     str
    claim_checks:   list[ClaimCheck]
    all_passed:     bool
    confidence:     float
    final_response: str          # response shown to user
    disclaimer:     str | None = None


def _extract_json_block(text: str) -> dict:
    """Extract the first ```json ... ``` block from a response."""
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try parsing the whole response as JSON
    try:
        return json.loads(text)
    except Exception:
        return {}


def _extract_plain_text(text: str) -> str:
    """Return everything after the JSON block."""
    cleaned = re.sub(r"```json[\s\S]*?```", "", text, flags=re.IGNORECASE).strip()
    return cleaned


def _extract_numeric_claims(text: str) -> list[tuple[str, float]]:
    """
    Extract (context_phrase, numeric_value) pairs from plain text.
    Looks for patterns like "Sharpe of 1.23" or "drawdown -15.4%".
    """
    claims = []
    pattern = re.compile(
        r'((?:sharpe\s*ratio|sharpe|sortino|calmar|drawdown|cagr|return|ic|ic-ir|alpha|beta|'
        r'turnover|cost|p-value|hit.?rate|win.?rate)\s*(?:of|=|:)?\s*)'
        r'(-?\d+\.?\d*%?)',
        re.IGNORECASE
    )
    for match in pattern.finditer(text):
        label = match.group(1).strip().rstrip("=: ").lower()
        raw   = match.group(2).replace("%", "")
        try:
            val = float(raw)
            if "%" in match.group(2):
                val /= 100.0
            claims.append((label, val))
        except ValueError:
            continue
    return claims


def _field_lookup(field_hint: str, metrics_snapshot: dict) -> float | None:
    """
    Try to find the actual value of a field in the metrics snapshot.
    Handles nested dicts with fuzzy key matching.
    """
    import re as _re
    # Normalize: strip trailing filler words, collapse spaces to underscores
    hint = _re.sub(r"\s+", "_", field_hint.lower().strip())
    hint = hint.replace("-", "_")
    for suffix in ("_of", "_is", "_ratio", "_score", "_rate"):
        if hint.endswith(suffix):
            hint = hint[:-len(suffix)]
    hint = hint.strip("_")

    # Aliases
    aliases = {
        "sharpe":      ["sharpe", "sharpe_ratio"],
        "sortino":     ["sortino", "sortino_ratio"],
        "drawdown":    ["max_drawdown", "drawdown"],
        "calmar":      ["calmar", "calmar_ratio"],
        "cagr":        ["cagr", "annual_return"],
        "return":      ["total_return", "cagr", "net_return"],
        "alpha":       ["alpha_ann", "alpha"],
        "beta":        ["beta"],
        "turnover":    ["avg_filtered_turnover", "avg_raw_turnover"],
        "ic":          ["mean_ic"],
        "ic_ir":       ["ic_ir"],
        "hit_rate":    ["hit_rate", "win_rate"],
        "p_value":     ["p_value"],
    }

    search_keys = [hint] + aliases.get(hint, [])

    def _search(obj, keys):
        if not isinstance(obj, dict):
            return None
        for k in keys:
            if k in obj:
                try:
                    return float(obj[k])
                except (TypeError, ValueError):
                    pass
        # Recurse
        for v in obj.values():
            if isinstance(v, dict):
                result = _search(v, keys)
                if result is not None:
                    return result
        return None

    return _search(metrics_snapshot, search_keys)


def _extract_confidence(parsed_json: dict, raw_response: str) -> float:
    """Extract confidence from JSON block or estimate from response quality."""
    if "confidence" in parsed_json:
        try:
            return float(parsed_json["confidence"])
        except (TypeError, ValueError):
            pass

    # Estimate: penalize vague language, reward citations
    vague_phrases = ["might", "could be", "perhaps", "possibly", "unclear",
                     "not sure", "data not available", "insufficient"]
    cite_phrases  = ["from pnl", "from factor_ic", "from audit", "from metrics",
                     "from regime", "cite", "source"]

    text_lower = raw_response.lower()
    vague_count = sum(1 for p in vague_phrases if p in text_lower)
    cite_count  = sum(1 for p in cite_phrases if p in text_lower)

    base = 0.80
    base -= 0.05 * min(vague_count, 4)
    base += 0.03 * min(cite_count, 3)
    return round(float(np.clip(base, 0.30, 0.95)), 3)


def validate_response(
    raw_response:     str,
    metrics_snapshot: dict,
    confidence:       float | None = None,
) -> ValidatedResponse:
    """
    Validate an LLM response against the metrics cache.

    Returns ValidatedResponse with claim-by-claim check results
    and a final_response safe to show to the user.
    """
    parsed_json = _extract_json_block(raw_response)
    plain_text  = _extract_plain_text(raw_response)

    # Extract confidence
    conf = confidence if confidence is not None else _extract_confidence(parsed_json, raw_response)

    # Extract and check numeric claims
    claims_in_text = _extract_numeric_claims(plain_text)
    claim_checks: list[ClaimCheck] = []
    all_passed = True

    for label, claimed_val in claims_in_text:
        actual = _field_lookup(label, metrics_snapshot)
        if actual is None:
            # Can't verify — not a failure, just unverifiable
            claim_checks.append(ClaimCheck(
                claim_text=label, claimed_val=claimed_val,
                source_field=None, actual_val=None, passed=True,
            ))
            continue

        # Relative tolerance check
        if abs(actual) > 1e-6:
            delta = abs(claimed_val - actual) / abs(actual)
        else:
            delta = abs(claimed_val - actual)

        passed = delta <= TOLERANCE
        if not passed:
            all_passed = False
            logger.warning(
                "Claim mismatch: '%s' LLM=%.4f actual=%.4f (delta=%.1f%%)",
                label, claimed_val, actual, delta * 100
            )

        claim_checks.append(ClaimCheck(
            claim_text=label, claimed_val=claimed_val,
            source_field=label, actual_val=actual,
            passed=passed, delta_pct=round(delta * 100, 2),
        ))

    # Also check key JSON fields
    for key, claimed in parsed_json.items():
        if isinstance(claimed, (int, float)) and not isinstance(claimed, bool):
            actual = _field_lookup(key, metrics_snapshot)
            if actual is not None:
                delta = abs(float(claimed) - actual) / (abs(actual) + 1e-10)
                passed = delta <= TOLERANCE
                if not passed:
                    all_passed = False
                    logger.warning("JSON field mismatch: %s LLM=%.4f actual=%.4f", key, claimed, actual)
                claim_checks.append(ClaimCheck(
                    claim_text=f"json:{key}", claimed_val=float(claimed),
                    source_field=key, actual_val=actual,
                    passed=passed, delta_pct=round(delta * 100, 2),
                ))

    # Build final response
    disclaimer = None
    if not all_passed:
        disclaimer = (
            "\n\n⚠️ **Validation notice**: One or more numeric values in this response "
            "could not be verified against the metrics cache. Please cross-check with "
            "the dashboard before acting on specific figures."
        )
    elif conf < LOW_CONFIDENCE:
        disclaimer = (
            f"\n\n💡 **Low confidence** ({conf:.0%}): Insufficient data to answer fully. "
            "Please check the dashboard for complete metrics."
        )

    final = raw_response
    if disclaimer:
        final = raw_response + disclaimer

    return ValidatedResponse(
        raw_response   = raw_response,
        parsed_json    = parsed_json,
        plain_text     = plain_text,
        claim_checks   = claim_checks,
        all_passed     = all_passed,
        confidence     = conf,
        final_response = final,
        disclaimer     = disclaimer,
    )
