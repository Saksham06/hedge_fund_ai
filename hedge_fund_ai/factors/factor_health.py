"""
Factor Health Monitor.

Implements:
  1. Redundancy filter: remove factors with cross-correlation > threshold
     (keep the one with higher rolling IC)
  2. 3-month statistical significance gate: auto-disable factors that
     fall below minimum IC-IR for 3 consecutive evaluation periods
  3. Monthly turnover cap: smooth low-conviction changes to avoid whipsaw
  4. Factor stability score: daily metric logged for monitoring
"""
import json
import logging
import os
from collections import deque, defaultdict
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

_HEALTH_PATH = os.path.join(
    os.path.dirname(__file__), "..", "state", "factor_health.json"
)

# Thresholds
REDUNDANCY_CORR_THRESHOLD = 0.75     # disable if any pair > this correlation
MIN_IC_IR_FOR_ACTIVATION  = 0.10     # factor must clear this IC-IR to stay active
CONSECUTIVE_FAIL_LIMIT    = 3        # periods below min before auto-disable
MONTHLY_TURNOVER_CAP      = 0.30     # max factor weight change per month


class FactorHealthMonitor:
    """
    Tracks factor IC history, detects redundancy, manages auto-disable logic.
    State persisted to disk across runs.
    """

    def __init__(self):
        self._state = self._load()

    def _load(self) -> dict:
        if not os.path.exists(_HEALTH_PATH):
            return {
                "disabled_factors": [],
                "consecutive_fails": {},
                "last_eval_date":    None,
                "ic_matrix_cache":   {},
            }
        try:
            with open(_HEALTH_PATH) as f:
                return json.load(f)
        except Exception:
            return {"disabled_factors": [], "consecutive_fails": {}, "last_eval_date": None, "ic_matrix_cache": {}}

    def _save(self):
        os.makedirs(os.path.dirname(_HEALTH_PATH), exist_ok=True)
        with open(_HEALTH_PATH, "w") as f:
            json.dump(self._state, f, indent=2, default=str)

    def get_disabled_factors(self) -> list[str]:
        return list(self._state.get("disabled_factors", []))

    def evaluate(self, ic_summary: dict, factor_return_matrix: dict | None = None):
        """
        Called after each backtest rebalance or weekly on live system.

        ic_summary: {factor: {"mean_ic": ..., "ic_ir": ..., "n": ...}}
        factor_return_matrix: {factor: [period_returns]} for redundancy check
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._state["last_eval_date"] = today

        # ── Significance gate ──────────────────────────────────────────────────
        for factor, stats in ic_summary.items():
            ic_ir = stats.get("ic_ir", 0) or 0
            n     = stats.get("n", 0) or 0

            if n < 6:
                continue  # too early to judge

            fails = self._state["consecutive_fails"].get(factor, 0)

            if ic_ir < MIN_IC_IR_FOR_ACTIVATION:
                fails += 1
                logger.info("Factor %s IC-IR=%.3f below threshold (fail %d/%d)",
                            factor, ic_ir, fails, CONSECUTIVE_FAIL_LIMIT)
            else:
                fails = max(0, fails - 1)  # recovery: credit back one pass

            self._state["consecutive_fails"][factor] = fails

            if fails >= CONSECUTIVE_FAIL_LIMIT:
                if factor not in self._state["disabled_factors"]:
                    self._state["disabled_factors"].append(factor)
                    logger.warning("Factor AUTO-DISABLED: %s (IC-IR=%.3f for %d periods)",
                                   factor, ic_ir, fails)
            elif factor in self._state["disabled_factors"] and ic_ir > MIN_IC_IR_FOR_ACTIVATION * 1.5:
                self._state["disabled_factors"].remove(factor)
                logger.info("Factor RE-ENABLED: %s (IC-IR=%.3f recovered)", factor, ic_ir)

        # ── Redundancy filter ─────────────────────────────────────────────────
        if factor_return_matrix and len(factor_return_matrix) >= 2:
            self._check_redundancy(factor_return_matrix, ic_summary)

        self._save()

    def _check_redundancy(self, return_matrix: dict, ic_summary: dict):
        """
        Remove redundant factors: if two factors correlate > REDUNDANCY_CORR_THRESHOLD,
        disable the one with lower IC-IR.
        """
        factors  = list(return_matrix.keys())
        n_f      = len(factors)
        if n_f < 2:
            return

        # Build correlation matrix from factor IC time series
        ic_series = {}
        for f, hist in return_matrix.items():
            if len(hist) >= 6:
                ic_series[f] = np.array(hist, dtype=float)

        checked = set()
        for i, f1 in enumerate(factors):
            for j, f2 in enumerate(factors):
                if j <= i or (f1, f2) in checked:
                    continue
                checked.add((f1, f2))
                s1 = ic_series.get(f1)
                s2 = ic_series.get(f2)
                if s1 is None or s2 is None:
                    continue
                n = min(len(s1), len(s2))
                if n < 6:
                    continue
                corr = float(np.corrcoef(s1[-n:], s2[-n:])[0, 1])
                if abs(corr) > REDUNDANCY_CORR_THRESHOLD:
                    ir1 = ic_summary.get(f1, {}).get("ic_ir", 0) or 0
                    ir2 = ic_summary.get(f2, {}).get("ic_ir", 0) or 0
                    loser = f1 if ir1 < ir2 else f2
                    if loser not in self._state["disabled_factors"]:
                        self._state["disabled_factors"].append(loser)
                        logger.warning(
                            "Factor REDUNDANCY: %s vs %s (corr=%.2f) → disabling %s (IC-IR=%.3f)",
                            f1, f2, corr, loser, min(ir1, ir2)
                        )


def apply_monthly_turnover_cap(
    new_weights: dict[str, float],
    old_weights: dict[str, float],
    max_delta: float = MONTHLY_TURNOVER_CAP,
) -> dict[str, float]:
    """
    Smooth factor/position weight changes so no single factor
    moves more than max_delta in one period.

    Preserves direction but limits speed of change.
    High-conviction signals (large delta) still move faster.
    """
    smoothed = {}
    for factor in set(new_weights) | set(old_weights):
        nw = float(new_weights.get(factor, 0))
        ow = float(old_weights.get(factor, 0))
        delta = nw - ow
        if abs(delta) > max_delta:
            # Move at most max_delta per period toward new weight
            smoothed[factor] = ow + np.sign(delta) * max_delta
        else:
            smoothed[factor] = nw
    return smoothed


def compute_factor_stability(ic_history: dict[str, list]) -> dict:
    """
    Compute a stability score per factor: how consistent is the IC?
    Returns {factor: {"stability": 0-1, "trend": "improving"|"stable"|"degrading"}}
    """
    result = {}
    for factor, history in ic_history.items():
        if len(history) < 4:
            result[factor] = {"stability": 0.5, "trend": "unknown"}
            continue
        arr = np.array(history[-12:], dtype=float)
        # Stability = 1 - (std / (|mean| + std)), higher = more consistent
        mean_abs = abs(float(np.mean(arr)))
        std      = float(np.std(arr))
        stability = float(mean_abs / (mean_abs + std + 1e-6))

        # Trend: compare last half vs first half
        mid = len(arr) // 2
        early = float(np.mean(arr[:mid]))
        late  = float(np.mean(arr[mid:]))
        if late > early + 0.01:
            trend = "improving"
        elif late < early - 0.01:
            trend = "degrading"
        else:
            trend = "stable"

        result[factor] = {"stability": round(stability, 3), "trend": trend}
    return result
