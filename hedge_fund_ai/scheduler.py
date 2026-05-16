"""
Auto-scheduler — runs the pipeline every weekday at 7PM IST (13:30 UTC).

7PM IST = 13:30 UTC = 30 min after US market close (4PM ET).
Runs only on weekdays (Mon-Fri) — no trades on weekends.

Usage:
    python -m hedge_fund_ai.scheduler

Env vars:
    SCHEDULE_TIME_UTC=13:30   # 7PM IST
    SCHEDULE_MODE=paper
    SCHEDULE_HORIZON=swing
    SCHEDULE_NO_BACKTEST=true  # skip backtest on daily runs (faster)
"""
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SCHEDULER] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scheduler")

SCHEDULE_TIME_UTC    = os.getenv("SCHEDULE_TIME_UTC",    "13:30")
SCHEDULE_MODE        = os.getenv("SCHEDULE_MODE",        "paper")
SCHEDULE_HORIZON     = os.getenv("SCHEDULE_HORIZON",     "swing")
SCHEDULE_UNIVERSE    = os.getenv("SCHEDULE_UNIVERSE",    "")
SCHEDULE_NO_BACKTEST = os.getenv("SCHEDULE_NO_BACKTEST", "true").lower() in ("1","true","yes")


def _next_run_dt() -> datetime:
    """Next scheduled run datetime (UTC). Skips weekends."""
    now  = datetime.now(timezone.utc)
    h, m = map(int, SCHEDULE_TIME_UTC.split(":"))
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    # Skip Saturday (5) and Sunday (6)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return target


def _build_cmd() -> list[str]:
    cmd = [sys.executable, "-m", "hedge_fund_ai.run",
           "--mode", SCHEDULE_MODE, "--horizon", SCHEDULE_HORIZON]
    if SCHEDULE_UNIVERSE:
        cmd += ["--universe", SCHEDULE_UNIVERSE]
    if SCHEDULE_NO_BACKTEST:
        cmd += ["--no-backtest"]
    return cmd


def _ist(utc_dt: datetime) -> str:
    ist = utc_dt + timedelta(hours=5, minutes=30)
    return ist.strftime("%Y-%m-%d %H:%M IST")


def main():
    h, m = map(int, SCHEDULE_TIME_UTC.split(":"))
    ist_h = (h * 60 + m + 330) // 60 % 24
    ist_m = (h * 60 + m + 330) % 60

    logger.info("=" * 60)
    logger.info("Hedge Fund Swing Scheduler")
    logger.info("Run time : %02d:%02d UTC = %02d:%02d IST (weekdays only)",
                h, m, ist_h, ist_m)
    logger.info("Mode     : %s | Horizon: %s | No-BT: %s",
                SCHEDULE_MODE, SCHEDULE_HORIZON, SCHEDULE_NO_BACKTEST)
    logger.info("Command  : %s", " ".join(_build_cmd()))
    logger.info("=" * 60)

    run_count = 0
    while True:
        next_dt    = _next_run_dt()
        wait_secs  = (next_dt - datetime.now(timezone.utc)).total_seconds()

        logger.info("Next run: %s  (in %.0f min)",
                    _ist(next_dt), wait_secs / 60)

        # Sleep in 60s ticks — log every 30 minutes
        slept = 0
        while slept < wait_secs:
            chunk  = min(60, wait_secs - slept)
            time.sleep(chunk)
            slept += chunk
            remaining = wait_secs - slept
            if remaining > 0 and int(slept) % 1800 < 65:
                logger.info("Next run in %.0f min (%s)",
                            remaining / 60, _ist(next_dt))

        run_count += 1
        now_ist = _ist(datetime.now(timezone.utc))
        logger.info("=" * 60)
        logger.info("SCHEDULED RUN #%d — %s", run_count, now_ist)
        logger.info("=" * 60)

        t0 = time.time()
        try:
            result = subprocess.run(_build_cmd(), timeout=3600)
            elapsed = time.time() - t0
            if result.returncode == 0:
                logger.info("Run #%d complete in %.0fs ✅", run_count, elapsed)
            else:
                logger.error("Run #%d failed (code %d) after %.0fs ❌",
                             run_count, result.returncode, elapsed)
        except subprocess.TimeoutExpired:
            logger.error("Run #%d timed out after 3600s", run_count)
        except Exception as e:
            logger.error("Run #%d error: %s", run_count, e)


if __name__ == "__main__":
    main()
