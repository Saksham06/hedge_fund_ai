"""
Startup Validator — run before any pipeline stage.
Checks env vars, API connectivity, disk access, Python deps.
Assigns a unique correlation ID to each run.
"""
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_OPTIONAL_WITH_IMPACT = {
    "ALPACA_API_KEY":    "Live execution disabled",
    "ALPACA_API_SECRET": "Live execution disabled",
    "GROQ_API_KEY":      "LLM features use keyword fallback",
    "FMP_API_KEY":       "Earnings surprise / revision signals disabled",
    "NEWS_API_KEY":      "News sentiment disabled",
    "SIMFIN_API_KEY":    "SimFin disabled — yfinance fallback only",
    "TELEGRAM_TOKEN":    "Telegram alerts disabled",
}


@dataclass
class StartupReport:
    run_id:      str
    start_time:  str
    warnings:    list = field(default_factory=list)
    errors:      list = field(default_factory=list)
    api_status:  dict = field(default_factory=dict)
    can_proceed: bool = True

    def log(self):
        logger.info("=" * 70)
        logger.info("RUN ID: %s  START: %s", self.run_id, self.start_time)
        for e in self.errors:
            logger.error("STARTUP ERROR: %s", e)
        for w in self.warnings:
            logger.warning("STARTUP WARN:  %s", w)
        for api, s in self.api_status.items():
            logger.info("%s %s: %s", "✅" if s["ok"] else "⚠️ ", api, s["msg"])
        if not self.can_proceed:
            logger.critical("Pipeline CANNOT proceed — fix errors above.")
        logger.info("=" * 70)


def validate_startup(dry_run: bool = False) -> StartupReport:
    run_id = str(uuid.uuid4())[:8].upper()
    report = StartupReport(
        run_id=run_id,
        start_time=datetime.now(timezone.utc).isoformat(),
    )

    # Optional env vars (deduplicated — each var checked once)
    seen = set()
    for var, impact in _OPTIONAL_WITH_IMPACT.items():
        if var in seen:
            continue
        seen.add(var)
        if not os.getenv(var):
            report.warnings.append(f"{var} not set → {impact}")

    # State dir writeable
    state_dir = os.path.join(os.path.dirname(__file__), "..", "state")
    try:
        os.makedirs(state_dir, exist_ok=True)
        tp = os.path.join(state_dir, ".write_test")
        open(tp, "w").close()
        os.remove(tp)
        report.api_status["state_dir"] = {"ok": True, "msg": state_dir}
    except Exception as e:
        report.errors.append(f"State dir not writeable: {e}")
        report.can_proceed = False

    # API checks
    report.api_status.update(_check_apis(dry_run))

    # Python version
    if sys.version_info < (3, 10):
        report.warnings.append(f"Python {sys.version} < 3.10")

    # Key packages
    for pkg in ["yfinance", "sklearn", "scipy", "pandas", "numpy"]:
        try:
            mod = __import__(pkg if pkg != "sklearn" else "sklearn")
            report.api_status[pkg] = {"ok": True, "msg": getattr(mod, "__version__", "ok")}
        except ImportError:
            report.errors.append(f"{pkg} not installed")
            report.can_proceed = False

    report.log()
    return report


def _check_apis(dry_run: bool) -> dict:
    import requests
    status = {}

    # yfinance
    try:
        import yfinance as yf
        px = yf.Ticker("SPY").fast_info.get("lastPrice")
        status["yfinance"] = {"ok": bool(px), "msg": f"SPY=${px:.2f}" if px else "no price"}
    except Exception as e:
        status["yfinance"] = {"ok": False, "msg": str(e)[:60]}

    # Alpaca
    from hedge_fund_ai.config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE
    if ALPACA_API_KEY and ALPACA_API_SECRET and not dry_run:
        try:
            r = requests.get(f"{ALPACA_BASE}/v2/account",
                             headers={"APCA-API-KEY-ID": ALPACA_API_KEY,
                                      "APCA-API-SECRET-KEY": ALPACA_API_SECRET}, timeout=5)
            acc = r.json() if r.status_code == 200 else {}
            status["alpaca"] = {"ok": r.status_code == 200,
                                "msg": f"equity=${float(acc.get('equity',0)):,.0f}" if acc else f"HTTP {r.status_code}"}
        except Exception as e:
            status["alpaca"] = {"ok": False, "msg": str(e)[:60]}
    else:
        status["alpaca"] = {"ok": True, "msg": "dry-run / no key"}

    # Groq
    from hedge_fund_ai.config import GROQ_API_KEY
    if GROQ_API_KEY:
        try:
            r = requests.get("https://api.groq.com/openai/v1/models",
                             headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, timeout=5)
            status["groq"] = {"ok": r.status_code == 200,
                              "msg": "ok" if r.status_code == 200 else f"HTTP {r.status_code}"}
        except Exception as e:
            status["groq"] = {"ok": False, "msg": str(e)[:60]}
    else:
        status["groq"] = {"ok": True, "msg": "no key — fallback active"}

    # FMP
    from hedge_fund_ai.config import FMP_API_KEY, FMP_BASE
    if FMP_API_KEY:
        try:
            r = requests.get(f"{FMP_BASE}/profile/AAPL",
                             params={"apikey": FMP_API_KEY}, timeout=5)
            status["fmp"] = {"ok": r.status_code == 200,
                             "msg": "ok" if r.status_code == 200 else f"HTTP {r.status_code}"}
        except Exception as e:
            status["fmp"] = {"ok": False, "msg": str(e)[:60]}
    else:
        status["fmp"] = {"ok": True, "msg": "no key"}

    return status
