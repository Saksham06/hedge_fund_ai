"""
Structured JSON logging with correlation IDs and metrics collection.

Replaces basic console logging with:
  1. JSON-formatted log records (parseable by log aggregators)
  2. Correlation ID injection into every log line
  3. Prometheus-compatible metrics counters
  4. Alert routing: WARNING+ → Telegram, CRITICAL → immediate
  5. File rotation: daily logs, 30-day retention
"""
import json
import logging
import logging.handlers
import os
import threading
import time
from datetime import datetime, timezone

_METRICS: dict[str, float] = {}
_METRICS_LOCK = threading.Lock()
_RUN_ID = "UNKNOWN"


def set_run_id(run_id: str):
    global _RUN_ID
    _RUN_ID = run_id


def increment_metric(name: str, value: float = 1.0):
    with _METRICS_LOCK:
        _METRICS[name] = _METRICS.get(name, 0.0) + value


def get_metrics() -> dict:
    with _METRICS_LOCK:
        return dict(_METRICS)


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "run_id":  _RUN_ID,
            "logger":  record.name,
            "msg":     record.getMessage(),
            "module":  record.module,
            "line":    record.lineno,
        }
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)


class MetricsFilter(logging.Filter):
    """Counts log events by level for Prometheus-style scraping."""

    def filter(self, record: logging.LogRecord) -> bool:
        increment_metric(f"log_count.{record.levelname.lower()}")
        if record.levelname in ("WARNING", "ERROR", "CRITICAL"):
            increment_metric("alert_count")
        return True


class TelegramAlertHandler(logging.Handler):
    """
    Sends WARNING+ messages to Telegram (non-blocking, best-effort).
    Uses a thread to avoid blocking the main pipeline.
    """

    def __init__(self, min_level: int = logging.WARNING):
        super().__init__(min_level)
        self._queue: list[str] = []
        self._lock  = threading.Lock()
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()

    def emit(self, record: logging.LogRecord):
        if record.levelno < self.level:
            return
        try:
            msg = f"[{record.levelname}] [{_RUN_ID}] {record.getMessage()}"
            with self._lock:
                self._queue.append(msg)
        except Exception:
            pass

    def _flush_loop(self):
        while True:
            time.sleep(5)
            with self._lock:
                msgs = list(self._queue)
                self._queue.clear()
            if msgs:
                try:
                    from hedge_fund_ai.reporting.telegram import send_telegram
                    send_telegram("\n".join(msgs[:5]))  # max 5 per batch
                except Exception:
                    pass


def setup_logging(
    run_id: str,
    log_dir: str | None = None,
    json_format: bool = True,
    alert_level: int = logging.WARNING,
) -> logging.Logger:
    """
    Configure structured logging for a pipeline run.

    Args:
        run_id:      Correlation ID from startup validator
        log_dir:     Directory for rotating log files (default: state/logs/)
        json_format: Use JSON formatter (True) or human-readable (False)
        alert_level: Minimum level for Telegram alerts

    Returns root logger configured for this run.
    """
    set_run_id(run_id)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    formatter = JSONFormatter() if json_format else logging.Formatter(
        f"%(asctime)s [{run_id}] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (human-readable for development)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        f"%(asctime)s [{run_id}] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    console.addFilter(MetricsFilter())
    root.addHandler(console)

    # File handler — rotating daily, 30-day retention
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), "..", "state", "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"pipeline.log")
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file, when="midnight", backupCount=30, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Telegram alert handler (WARNING+)
    try:
        from hedge_fund_ai.config import TELEGRAM_TOKEN
        if TELEGRAM_TOKEN:
            tg = TelegramAlertHandler(min_level=alert_level)
            tg.setLevel(alert_level)
            root.addHandler(tg)
    except Exception:
        pass

    logger = logging.getLogger("pipeline")
    logger.info("Logging initialized | run_id=%s | log_file=%s", run_id, log_file)
    return logger


def write_metrics_snapshot(path: str | None = None):
    """Write current metrics to a JSON file for Prometheus scraping."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "state", "metrics.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id":    _RUN_ID,
        "metrics":   get_metrics(),
    }
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
