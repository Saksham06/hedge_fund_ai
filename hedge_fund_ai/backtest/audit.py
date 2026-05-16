"""
Audit trail for the walk-forward backtester.

Every rebalance produces a structured AuditRecord that is:
  - Stored in memory for the run
  - Serialized to JSON after completion

The audit log lets you explain every single result:
  - Which universe was tradeable on that date
  - Which regime was active
  - What each factor score was (cross-sectionally z-scored)
  - What weights were selected (pre and post turnover filter)
  - Raw turnover vs filtered turnover
  - Exact cost breakdown per ticker
  - Forward return (filled in on next rebalance)
"""
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import date as Date

logger = logging.getLogger(__name__)

_AUDIT_PATH = os.path.join(os.path.dirname(__file__), "..", "state", "audit_log.json")


@dataclass
class AuditRecord:
    rebalance_idx:    int
    date:             str
    regime:           str
    universe_size:    int
    dd_scale:         float

    # Signal
    raw_scores:       dict = field(default_factory=dict)    # {ticker: raw_composite}
    cs_scores:        dict = field(default_factory=dict)    # {ticker: cross-sect z-scored}
    factor_exposures: dict = field(default_factory=dict)    # {ticker: {factor: z_score}}

    # Weights
    target_weights:   dict = field(default_factory=dict)    # after optimizer
    prev_weights:     dict = field(default_factory=dict)    # weights entering this period
    final_weights:    dict = field(default_factory=dict)    # after turnover filter

    # Turnover
    raw_turnover:      float = 0.0   # sum |target - prev|
    filtered_turnover: float = 0.0   # sum |final - prev|

    # Costs
    total_cost_frac:  float = 0.0   # fraction of equity deducted
    per_ticker_cost:  dict = field(default_factory=dict)
    adv_map:          dict = field(default_factory=dict)

    # Performance (filled in retrospectively)
    period_return:    float | None = None   # portfolio return over hold period
    benchmark_return: float | None = None   # SPY return over same period

    # Factor IC realized (filled next period)
    realized_ic:      dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class AuditLog:
    def __init__(self):
        self._records: list[AuditRecord] = []

    def append(self, record: AuditRecord):
        self._records.append(record)

    def fill_returns(self, idx: int, period_return: float, bench_return: float | None = None):
        """Fill in realized return for record at list index idx."""
        if 0 <= idx < len(self._records):
            self._records[idx].period_return = round(float(period_return), 6)
            if bench_return is not None:
                self._records[idx].benchmark_return = round(float(bench_return), 6)

    def fill_ic(self, idx: int, ic_dict: dict):
        """Fill in realized factor IC for record at list index idx."""
        if 0 <= idx < len(self._records):
            self._records[idx].realized_ic = {k: round(v, 4) for k, v in ic_dict.items()}

    def to_list(self) -> list[dict]:
        return [r.to_dict() for r in self._records]

    def save(self, path: str = _AUDIT_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_list(), f, indent=2, default=str)
        logger.info("Audit log saved: %d records → %s", len(self._records), path)

    def summary(self) -> str:
        if not self._records:
            return "Empty audit log."
        lines = [f"{'Date':<12} {'Regime':<10} {'Univ':>4} {'Pos':>3} "
                 f"{'RawTurn':>8} {'FiltTurn':>9} {'Cost bps':>9} {'PeriodRet':>10}"]
        lines.append("-" * 75)
        for r in self._records:
            cost_bps = r.total_cost_frac * 10_000
            ret_str  = f"{r.period_return*100:+.2f}%" if r.period_return is not None else "   N/A"
            lines.append(
                f"{r.date:<12} {r.regime:<10} {r.universe_size:>4} "
                f"{len(r.final_weights):>3} {r.raw_turnover:>8.3f} "
                f"{r.filtered_turnover:>9.3f} {cost_bps:>9.2f} {ret_str:>10}"
            )
        return "\n".join(lines)

    def regime_breakdown(self) -> dict:
        """Return per-regime average metrics."""
        from collections import defaultdict
        buckets: dict[str, list] = defaultdict(list)
        for r in self._records:
            if r.period_return is not None:
                buckets[r.regime].append(r.period_return)
        return {
            regime: {
                "n_periods": len(rets),
                "avg_return": round(float(sum(rets) / len(rets)), 5),
                "hit_ratio":  round(sum(1 for x in rets if x > 0) / len(rets), 3),
            }
            for regime, rets in buckets.items()
        }
