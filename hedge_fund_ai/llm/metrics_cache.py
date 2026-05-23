"""
Phase 1: Metrics Cache — query-optimized SQLite storage.

Exports and caches:
  - daily_pnl_attribution: date, net_return, factor_contributions, regime, risk_state
  - factor_ic_history: rolling IC, t-stat, hit_rate per factor per date
  - regime_probabilities: current regime distribution

Design:
  - SQLite for structured, versioned snapshots (no heavy deps)
  - Each snapshot tagged with run_id + data_version + calc_timestamp
  - Schema validation rejects malformed metrics before insert
  - Read path returns typed dicts, never raw SQL rows
"""

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "state", "metrics_cache.db"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pnl_attribution (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL,
    data_version     TEXT NOT NULL,
    calc_timestamp   TEXT NOT NULL,
    date             TEXT NOT NULL,
    net_return       REAL,
    factor_contributions TEXT,   -- JSON
    regime           TEXT,
    risk_state       TEXT,       -- JSON
    UNIQUE(run_id, date)
);

CREATE TABLE IF NOT EXISTS factor_ic (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL,
    data_version     TEXT NOT NULL,
    calc_timestamp   TEXT NOT NULL,
    date             TEXT NOT NULL,
    factor           TEXT NOT NULL,
    mean_ic          REAL,
    t_stat           REAL,
    hit_rate         REAL,
    ic_ir            REAL,
    n_observations   INTEGER,
    UNIQUE(run_id, date, factor)
);

CREATE TABLE IF NOT EXISTS regime_state (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL,
    data_version     TEXT NOT NULL,
    calc_timestamp   TEXT NOT NULL,
    date             TEXT NOT NULL,
    regime           TEXT NOT NULL,
    vix              REAL,
    spy_3m           REAL,
    credit_signal    REAL,
    yield_curve      REAL,
    vol_regime       REAL,
    UNIQUE(run_id, date)
);

CREATE TABLE IF NOT EXISTS query_audit (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    run_id           TEXT,
    query_hash       TEXT NOT NULL,
    question_type    TEXT,
    question_text    TEXT,
    confidence       REAL,
    validation_pass  INTEGER,
    response_length  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_pnl_date     ON pnl_attribution(date);
CREATE INDEX IF NOT EXISTS idx_ic_factor    ON factor_ic(factor, date);
CREATE INDEX IF NOT EXISTS idx_regime_date  ON regime_state(date);
CREATE INDEX IF NOT EXISTS idx_query_hash   ON query_audit(query_hash);
"""


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _conn() as c:
        c.executescript(_SCHEMA)


_init_db()


def _data_version(obj: Any) -> str:
    """SHA-256 hash of serialized data — detects staleness."""
    raw = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


# ── SCHEMA VALIDATORS ─────────────────────────────────────────────────────────

def _validate_pnl(row: dict) -> list[str]:
    errors = []
    if "date" not in row:
        errors.append("missing field: date")
    if "net_return" in row:
        v = row["net_return"]
        if v is not None and (not isinstance(v, (int, float)) or not np.isfinite(v)):
            errors.append(f"net_return not finite: {v}")
        if v is not None and abs(v) > 0.50:
            errors.append(f"net_return implausibly large: {v:.2%}")
    return errors


def _validate_ic(row: dict) -> list[str]:
    errors = []
    for req in ["factor", "mean_ic"]:
        if req not in row:
            errors.append(f"missing field: {req}")
    if "mean_ic" in row and row["mean_ic"] is not None:
        if abs(row["mean_ic"]) > 1.0:
            errors.append(f"mean_ic out of [-1,1]: {row['mean_ic']}")
    return errors


# ── WRITE PATH ────────────────────────────────────────────────────────────────

def snapshot_pnl(records: list[dict], run_id: str = "SYSTEM"):
    """
    Persist daily P&L attribution records.
    Rejects records that fail schema validation.
    """
    ts  = datetime.now(timezone.utc).isoformat()
    ver = _data_version(records)
    ok, bad = 0, 0

    with _conn() as c:
        for row in records:
            errs = _validate_pnl(row)
            if errs:
                logger.warning("PnL schema rejected %s: %s", row.get("date"), errs)
                bad += 1
                continue
            try:
                c.execute("""
                    INSERT OR REPLACE INTO pnl_attribution
                    (run_id, data_version, calc_timestamp, date, net_return,
                     factor_contributions, regime, risk_state)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (
                    run_id, ver, ts,
                    str(row.get("date", "")),
                    float(row["net_return"]) if row.get("net_return") is not None else None,
                    json.dumps(row.get("factor_contributions", {}), default=str),
                    str(row.get("regime", "")),
                    json.dumps(row.get("risk_state", {}), default=str),
                ))
                ok += 1
            except Exception as e:
                logger.debug("PnL insert: %s", e)
                bad += 1

    logger.info("PnL snapshot: %d ok, %d rejected", ok, bad)


def snapshot_factor_ic(ic_summary: dict, run_id: str = "SYSTEM",
                       date: str | None = None):
    """
    Persist rolling IC stats per factor.
    ic_summary: {factor: {"mean_ic", "ic_ir", "hit_rate", "n", ...}}
    """
    ts   = datetime.now(timezone.utc).isoformat()
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ver  = _data_version(ic_summary)

    with _conn() as c:
        for factor, stats in ic_summary.items():
            errs = _validate_ic({"factor": factor, "mean_ic": stats.get("mean_ic")})
            if errs:
                logger.warning("IC schema rejected %s: %s", factor, errs)
                continue
            try:
                # Compute t-stat if not present
                mean_ic = float(stats.get("mean_ic", 0) or 0)
                ic_ir   = float(stats.get("ic_ir",   0) or 0)
                n       = int(stats.get("n",         0) or 0)
                # t-stat approximation: IC-IR * sqrt(n)
                t_stat  = ic_ir * (n ** 0.5) if n > 0 else 0.0

                c.execute("""
                    INSERT OR REPLACE INTO factor_ic
                    (run_id, data_version, calc_timestamp, date, factor,
                     mean_ic, t_stat, hit_rate, ic_ir, n_observations)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (
                    run_id, ver, ts, date, factor,
                    mean_ic, round(t_stat, 4),
                    float(stats.get("hit_rate", 0) or 0),
                    ic_ir, n,
                ))
            except Exception as e:
                logger.debug("IC insert %s: %s", factor, e)


def snapshot_regime(macro: dict, run_id: str = "SYSTEM"):
    ts   = datetime.now(timezone.utc).isoformat()
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ver  = _data_version(macro)
    with _conn() as c:
        try:
            c.execute("""
                INSERT OR REPLACE INTO regime_state
                (run_id, data_version, calc_timestamp, date, regime,
                 vix, spy_3m, credit_signal, yield_curve, vol_regime)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                run_id, ver, ts, date,
                str(macro.get("regime", "neutral")),
                float(macro.get("vix",                    20) or 20),
                float(macro.get("spy_3m",                  0) or 0),
                float(macro.get("credit_spread_signal",    0) or 0),
                float(macro.get("yield_curve",             0) or 0),
                float(macro.get("vol_regime",              1) or 1),
            ))
        except Exception as e:
            logger.debug("Regime insert: %s", e)


def log_query_audit(query_hash: str, question_type: str, question_text: str,
                    confidence: float, validation_pass: bool,
                    response_length: int, run_id: str = "SYSTEM"):
    """Append-only query audit log for compliance."""
    ts = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute("""
            INSERT INTO query_audit
            (timestamp, run_id, query_hash, question_type, question_text,
             confidence, validation_pass, response_length)
            VALUES (?,?,?,?,?,?,?,?)
        """, (ts, run_id, query_hash, question_type, question_text[:500],
              confidence, int(validation_pass), response_length))


# ── READ PATH ─────────────────────────────────────────────────────────────────

def get_latest_pnl(n_days: int = 30) -> list[dict]:
    with _conn() as c:
        rows = c.execute("""
            SELECT * FROM pnl_attribution
            ORDER BY date DESC LIMIT ?
        """, (n_days,)).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        for k in ("factor_contributions", "risk_state"):
            try:
                d[k] = json.loads(d[k]) if d.get(k) else {}
            except Exception:
                d[k] = {}
        result.append(d)
    return result


def get_factor_ic_history(factor: str | None = None, n_rows: int = 50) -> list[dict]:
    with _conn() as c:
        if factor:
            rows = c.execute("""
                SELECT * FROM factor_ic WHERE factor=?
                ORDER BY date DESC LIMIT ?
            """, (factor, n_rows)).fetchall()
        else:
            rows = c.execute("""
                SELECT * FROM factor_ic ORDER BY date DESC LIMIT ?
            """, (n_rows,)).fetchall()
    return [dict(r) for r in rows]


def get_factor_ic_summary_from_cache() -> dict:
    """Return latest IC stats per factor as {factor: {mean_ic, t_stat, hit_rate, ic_ir}}."""
    with _conn() as c:
        rows = c.execute("""
            SELECT factor, mean_ic, t_stat, hit_rate, ic_ir, n_observations
            FROM factor_ic
            WHERE (factor, date) IN (
                SELECT factor, MAX(date) FROM factor_ic GROUP BY factor
            )
        """).fetchall()
    return {r["factor"]: dict(r) for r in rows}


def get_latest_regime() -> dict:
    with _conn() as c:
        row = c.execute("""
            SELECT * FROM regime_state ORDER BY date DESC LIMIT 1
        """).fetchone()
    return dict(row) if row else {}


def get_metrics_snapshot() -> dict:
    """Return a complete snapshot of all latest metrics for LLM context."""
    return {
        "pnl_last_30d":   get_latest_pnl(30),
        "factor_ic":      get_factor_ic_summary_from_cache(),
        "regime":         get_latest_regime(),
        "as_of":          datetime.now(timezone.utc).isoformat(),
    }


def export_daily_json():
    """Export canonical JSON files read by the LLM layer."""
    state_dir = os.path.join(os.path.dirname(__file__), "..", "state")
    os.makedirs(state_dir, exist_ok=True)

    snapshot = get_metrics_snapshot()

    with open(os.path.join(state_dir, "daily_pnl_attribution.json"), "w") as f:
        json.dump(snapshot["pnl_last_30d"], f, indent=2, default=str)

    with open(os.path.join(state_dir, "factor_ic_history.json"), "w") as f:
        json.dump(snapshot["factor_ic"], f, indent=2, default=str)

    with open(os.path.join(state_dir, "regime_probabilities.json"), "w") as f:
        json.dump(snapshot["regime"], f, indent=2, default=str)

    logger.info("Daily JSON exports written to state/")
