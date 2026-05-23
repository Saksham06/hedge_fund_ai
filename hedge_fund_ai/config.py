"""
Central configuration — all tuneable parameters, all data sources.
Override any value via environment variable or .env file.
"""
import os
import warnings
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*charset_normalizer.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="requests")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    pass


def _bool(k, d="false"):  return (os.getenv(k, d) or d).strip().lower() in ("1","true","yes","y","on")
def _float(k, d):
    try: return float(os.getenv(k, str(d)) or d)
    except: return d
def _int(k, d):
    try: return int(os.getenv(k, str(d)) or d)
    except: return d


# ── Data providers ────────────────────────────────────────────────────────────
SIMFIN_API_KEY   = os.getenv("SIMFIN_API_KEY", "free")        # "free" uses SimFin free tier
SIMFIN_DATA_DIR  = os.getenv("SIMFIN_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "state", "simfin"))
FMP_API_KEY      = os.getenv("FMP_API_KEY", "")               # financialmodelingprep.com
FMP_BASE         = "https://financialmodelingprep.com/api/v3"
NEWS_API_KEY     = os.getenv("NEWS_API_KEY", "")

# ── LLM ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL  = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")

# ── Notifications ─────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Execution ─────────────────────────────────────────────────────────────────
ALPACA_API_KEY      = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET   = os.getenv("ALPACA_API_SECRET", "") or os.getenv("ALPACA_SECRET", "")
ALPACA_SECRET       = ALPACA_API_SECRET
ALPACA_PAPER        = _bool("ALPACA_PAPER", "true")
ALPACA_BASE         = os.getenv("ALPACA_BASE",
    "https://paper-api.alpaca.markets" if _bool("ALPACA_PAPER","true")
    else "https://api.alpaca.markets")
ALPACA_DRY_RUN         = _bool("ALPACA_DRY_RUN", "true")
ALPACA_MAX_ORDER_USD   = _float("ALPACA_MAX_ORDER_USD", 50_000)
ALPACA_FRACTIONAL      = _bool("ALPACA_FRACTIONAL", "true")
PORTFOLIO_NOTIONAL_USD = _float("PORTFOLIO_NOTIONAL_USD", 100_000)

# ── Pipeline ──────────────────────────────────────────────────────────────────
ENABLE_TRADE_EXPLANATIONS = _bool("ENABLE_TRADE_EXPLANATIONS", "false")

# ── Signal / Factor ───────────────────────────────────────────────────────────
TURNOVER_THRESHOLD = _float("TURNOVER_THRESHOLD", 0.02)
RANDOM_SEED        = _int("RANDOM_SEED", 42)
MIN_UNIVERSE_SIZE  = _int("MIN_UNIVERSE_SIZE", 15)
TOP_N_POSITIONS    = _int("TOP_N_POSITIONS", 10)
IC_WINDOW          = _int("IC_WINDOW", 24)          # rolling IC window (periods)
IC_DECAY_HALF_LIFE = _int("IC_DECAY_HALF_LIFE", 12) # exponential decay in periods

# ── Portfolio ─────────────────────────────────────────────────────────────────
MAX_POSITION_WEIGHT  = _float("MAX_POSITION_WEIGHT", 0.20)
MIN_POSITION_WEIGHT  = _float("MIN_POSITION_WEIGHT", 0.03)
MAX_SECTOR_WEIGHT    = _float("MAX_SECTOR_WEIGHT", 0.35)
TARGET_VOL           = _float("TARGET_VOL", 0.12)
KELLY_FRACTION       = _float("KELLY_FRACTION", 0.25)   # fractional Kelly (conservative)
CVAR_CONFIDENCE      = _float("CVAR_CONFIDENCE", 0.95)  # CVaR confidence level

# ── Backtest ──────────────────────────────────────────────────────────────────
BACKTEST_STALE_DAYS  = _int("BACKTEST_STALE_DAYS", 7)
BACKTEST_PERIOD      = os.getenv("BACKTEST_PERIOD", "10y")   # extended to 10 years
REBALANCE_DAYS       = _int("REBALANCE_DAYS", 21)
COST_BPS             = _int("COST_BPS", 10)

# ── Universe ──────────────────────────────────────────────────────────────────
UNIVERSE = os.getenv("UNIVERSE",
    "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,BRK-B,JPM,UNH,XOM,AVGO,LLY,V,MA,"
    "PG,COST,HD,KO,PEP,ADBE,CRM,CSCO,ORCL,AMD,QCOM,NFLX,DIS,NKE,BAC,"
    "WMT,CVX,MRK,PFE,ABBV,TMO,ACN,TXN,NEE,DHR,LIN,HON,RTX,AMGN,SBUX,"
    "GILD,MDT,BLK,SPGI,MMM,GS,MS,C,WFC,USB,BK,SCHW,ICE,CME,SPGI,"
    "BMY,REGN,VRTX,ISRG,SYK,BSX,MDT,EW,A,DHR,"
    "LRCX,KLAC,AMAT,ASML,MU,STX,WDC,"
    "UNP,CSX,NSC,FDX,UPS,DAL,UAL,"
    "EMR,ETN,ITW,ROK,PH,DOV,GWW,"
    "SHW,APD,ECL,LIN,PPG,DD,"
    "AMT,CCI,EQIX,PSA,PLD,SPG"
).split(",")

# ── Horizon mode ──────────────────────────────────────────────────────────────
HORIZON = os.getenv("HORIZON", "swing")   # "swing" | "longterm"

# ── Swing trading parameters ──────────────────────────────────────────────────
SWING_REBALANCE_DAYS   = _int("SWING_REBALANCE_DAYS",   5)    # every 5 trading days (~1 week)
SWING_MIN_HOLD_DAYS    = _int("SWING_MIN_HOLD_DAYS",    3)    # never exit before 3 days
SWING_MAX_HOLD_DAYS    = _int("SWING_MAX_HOLD_DAYS",   60)    # force-exit after 60 days
SWING_TOP_N            = _int("SWING_TOP_N",           12)    # 12 positions
SWING_TARGET_VOL       = _float("SWING_TARGET_VOL",   0.18)   # allow more vol
SWING_MAX_POSITION     = _float("SWING_MAX_POSITION",  0.12)  # 12% cap per position
SWING_STOP_LOSS        = _float("SWING_STOP_LOSS",     0.05)  # 5% hard stop
SWING_PROFIT_TARGET    = _float("SWING_PROFIT_TARGET", 0.12)  # 12% take-profit
SWING_COST_BPS         = _int("SWING_COST_BPS",        15)    # 15bps round-trip
SWING_BACKTEST_PERIOD  = os.getenv("SWING_BACKTEST_PERIOD", "3y")

# ── Price cache ───────────────────────────────────────────────────────────────
PRICE_CACHE_DIR      = os.getenv("PRICE_CACHE_DIR",
    os.path.join(os.path.dirname(__file__), "state", "price_cache"))
PRICE_CACHE_TTL_HOURS = _int("PRICE_CACHE_TTL_HOURS", 24)  # 24h = always EOD data

# ── Long-term parameters ──────────────────────────────────────────────────────
LT_REBALANCE_DAYS    = _int("LT_REBALANCE_DAYS",    63)
LT_MIN_HOLD_DAYS     = _int("LT_MIN_HOLD_DAYS",    126)
LT_MAX_HOLD_DAYS     = _int("LT_MAX_HOLD_DAYS",   1825)
LT_TOP_N             = _int("LT_TOP_N",             15)
LT_TARGET_VOL        = _float("LT_TARGET_VOL",     0.12)
LT_MAX_POSITION      = _float("LT_MAX_POSITION",   0.20)
LT_STOP_LOSS         = _float("LT_STOP_LOSS",      0.15)
LT_PROFIT_TARGET     = _float("LT_PROFIT_TARGET",  0.50)
LT_COST_BPS          = _int("LT_COST_BPS",           8)
LT_BACKTEST_PERIOD   = os.getenv("LT_BACKTEST_PERIOD", "10y")
