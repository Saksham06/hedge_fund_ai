# AI Hedge Fund — Production

A deployment-ready quantitative equity hedge fund pipeline.
All 8 production gaps filled. 94 tests passing.

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Fill in: GROQ_API_KEY, ALPACA_API_KEY, ALPACA_API_SECRET

# 3. Run pipeline (paper trading by default)
python -m hedge_fund_ai.run

# 4. Dashboard
streamlit run hedge_fund_ai/dashboard/app.py

# 5. Tests
pytest tests/ -v
```

## Deployment readiness checklist

| Gap | Fix | Status |
|-----|-----|--------|
| Fundamentals in backtest | `FundamentalStore` — point-in-time quarterly data, 45-day filing lag | ✅ |
| Survivorship bias | Historical S&P 500 membership table, filtered at each rebalance date | ✅ |
| Live signal validation | `SignalLogger` — logs daily predictions, fills IC after 21 days | ✅ |
| Sentiment IC untested | `SignalLogger` captures full composite score including sentiment | ✅ |
| Position stop losses | `risk_controls.py` — 8% position stop, closes via Alpaca DELETE | ✅ |
| Daily loss circuit breaker | 3% daily loss → 24h halt; 20% drawdown → 30-day halt | ✅ |
| Placebo test | 50 random-signal trials, empirical p-value vs model Sharpe | ✅ |
| Corporate actions | yfinance adjusted prices; documented limitation for production | ✅ |

## When to deploy real capital

The `Signal IC` tab in the dashboard shows a verdict:
- **DEPLOY** — IC > 0.05, IC-IR > 0.5, hit rate > 55% over ≥12 periods
- **PAPER** — Weak positive IC, keep monitoring
- **NO ALPHA** — Negative IC, do not deploy capital

Do not skip the paper trading phase. 60 days minimum before real capital.

## Architecture

```
run.py  (7 stages)
├── Stage 1  Data fetch       — concurrent yfinance, macro (VIX/HYG/TLT/SHY)
├── Stage 2  Sentiment        — LLM batch via Groq, keyword fallback, time-decay
├── Stage 3  Signals          — 17 factors, z-scored, IC-weighted, logged to SignalLogger
├── Stage 4  Backtest         — Point-in-time fundamentals + survivorship filter
│                               Ledoit-Wolf covariance, Placebo test, SPY+EW benchmarks
├── Stage 5  Portfolio        — Signal tilt + HHI penalty + sector constraints + vol target
├── Stage 6  Reporting        — CIO report, factor attribution, stability, PDF, Telegram
└── Stage 7  Execution        — Stop-loss check → circuit breaker → reconcile → Alpaca
```

## Risk controls

| Control | Default | Env var |
|---------|---------|---------|
| Position stop loss | 8% | `STOP_LOSS_PCT` |
| Daily loss halt | 3% | `DAILY_LOSS_LIMIT` |
| Max drawdown halt | 20% | `MAX_DD_HALT` |
| Vol target | 12% ann. | `TARGET_VOL` |
| Max position | 20% | `MAX_POSITION_WEIGHT` |
| Max sector | 35% | `MAX_SECTOR_WEIGHT` |

## Key files

```
hedge_fund_ai/
├── data/
│   ├── fundamentals_pit.py   # Point-in-time fundamentals (Gap 1)
│   ├── survivorship.py       # S&P 500 membership history (Gap 2)
│   ├── signal_logger.py      # Live IC tracker (Gap 3)
│   └── fetcher.py            # Concurrent data fetch with beta, ADV, all ratios
├── backtest/
│   ├── walk_forward.py       # No-lookahead backtester (all gaps integrated)
│   ├── placebo.py            # Randomised signal baseline (Gap 6)
│   ├── audit.py              # Full rebalance audit trail
│   ├── costs.py              # ADV-based Almgren-Chriss cost model
│   ├── metrics.py            # Sharpe/Sortino/Calmar/alpha/beta/IC/turnover
│   └── factor_attribution.py # IC-based attribution + stability report
├── execution/
│   ├── alpaca.py             # Fractional + reconciliation + risk gate
│   └── risk_controls.py      # Stop losses + circuit breakers (Gap 5)
├── factors/
│   ├── signal_engine.py      # 4-regime IC-weighted signal, score_components()
│   ├── normalization.py      # 17-factor cross-sectional z-score + winsorize
│   ├── historical_features.py# 15 price/volume features, no lookahead
│   └── regime.py             # 4-state regime: risk_on/neutral/risk_off/crisis
├── portfolio/
│   ├── optimizer.py          # Ledoit-Wolf + signal tilt + HHI + sector + vol-target
│   └── risk_model.py         # 3-level drawdown control
└── dashboard/app.py          # 7-tab Streamlit dashboard
```
