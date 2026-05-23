"""
Elite AI Hedge Fund Pipeline — Swing Trading Mode.

Default mode: SWING (5-day rebalance, momentum signals, 60-day max hold).
Use --horizon longterm for quarterly rebalance with quality factors.

CLI:
  python -m hedge_fund_ai.run --mode paper
  python -m hedge_fund_ai.run --mode paper --horizon swing --no-backtest
"""
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from hedge_fund_ai.config import (
    ALPACA_DRY_RUN, BACKTEST_PERIOD,
    BACKTEST_STALE_DAYS, MIN_UNIVERSE_SIZE,
    PORTFOLIO_NOTIONAL_USD, UNIVERSE,
    # Swing params
    SWING_REBALANCE_DAYS, SWING_TOP_N, SWING_TARGET_VOL,
    SWING_MAX_POSITION, SWING_BACKTEST_PERIOD,
    SWING_STOP_LOSS, SWING_PROFIT_TARGET, SWING_COST_BPS,
    # LT params
    LT_REBALANCE_DAYS, LT_TOP_N, LT_TARGET_VOL,
    LT_BACKTEST_PERIOD,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_STATE = os.path.join(os.path.dirname(__file__), "state")


def _save_state(filename: str, obj):
    os.makedirs(_STATE, exist_ok=True)
    with open(os.path.join(_STATE, filename), "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _horizon_params(horizon: str) -> dict:
    """Return horizon-specific parameters."""
    if horizon == "swing":
        return {
            "rebalance_days":  SWING_REBALANCE_DAYS,
            "top_n":           SWING_TOP_N,
            "target_vol":      SWING_TARGET_VOL,
            "max_position":    SWING_MAX_POSITION,
            "backtest_period": SWING_BACKTEST_PERIOD,
            "stop_loss":       SWING_STOP_LOSS,
            "profit_target":   SWING_PROFIT_TARGET,
            "cost_bps":        SWING_COST_BPS,
        }
    else:  # longterm
        return {
            "rebalance_days":  LT_REBALANCE_DAYS,
            "top_n":           LT_TOP_N,
            "target_vol":      LT_TARGET_VOL,
            "max_position":    0.20,
            "backtest_period": LT_BACKTEST_PERIOD,
            "stop_loss":       0.15,
            "profit_target":   0.50,
            "cost_bps":        8,
        }


# ── Stage 0: Startup ──────────────────────────────────────────────────────────
def stage_startup(dry_run: bool = False, horizon: str = "swing"):
    from hedge_fund_ai.core.startup import validate_startup
    report = validate_startup(dry_run=dry_run)
    if not report.can_proceed:
        raise RuntimeError(f"Startup failed: {report.errors}")
    logger.info("Horizon: %s | Rebalance: %dd | Stop: %.0f%% | Target: %.0f%%",
                horizon.upper(),
                _horizon_params(horizon)["rebalance_days"],
                _horizon_params(horizon)["stop_loss"] * 100,
                _horizon_params(horizon)["profit_target"] * 100)
    return report


# ── Stage 1: Data fetch ───────────────────────────────────────────────────────
def stage_fetch_data(universe, run_id: str = ""):
    from hedge_fund_ai.data.fetcher import fetch_macro, fetch_universe
    from hedge_fund_ai.data.quality import check_data_freshness
    from hedge_fund_ai.factors.regime import detect_regime
    from hedge_fund_ai.state.macro import append_macro_snapshot
    from hedge_fund_ai.data.price_cache import get_cache

    logger.info("[%s] Stage 1/7: Fetching %d tickers", run_id, len(universe))

    # Show cache stats
    cache = get_cache()
    stats = cache.stats()
    if stats["fresh"] > 0:
        logger.info("[%s] Price cache: %d fresh / %d total (%.1fMB)",
                    run_id, stats["fresh"], stats["total"], stats["size_mb"])

    macro         = fetch_macro()
    macro_history = append_macro_snapshot(macro)
    regime        = detect_regime(macro_history)
    logger.info("[%s] Regime: %s | VIX=%.1f | SPY3m=%.1f%% | Credit=%.4f",
                run_id, regime.upper(), macro.get("vix", 0),
                macro.get("spy_3m", 0), macro.get("credit_spread_signal", 0))

    data = fetch_universe(universe[:60])

    # Save to cache
    if data:
        for d in data:
            try:
                ph = d.get("tech", {}).get("price_hist", [])
                vh = d.get("tech", {}).get("volume_hist", [])
                if ph:
                    cache.put(d["ticker"],
                              pd.Series(ph), pd.Series(vh) if vh else None)
            except Exception:
                pass

    freshness = check_data_freshness(data)
    stale = {t: h for t, h in freshness.items() if h and h > 36}
    if stale:
        logger.warning("[%s] Stale data: %s", run_id, list(stale.keys())[:5])

    logger.info("[%s] Fetched: %d/%d tickers", run_id, len(data), len(universe))
    return data, macro, macro_history, regime


# ── Stage 2: Sentiment ────────────────────────────────────────────────────────
def stage_sentiment(data, run_id: str = ""):
    from hedge_fund_ai.data.news import fetch_news_batch
    from hedge_fund_ai.data.aggregate import aggregate_advanced
    logger.info("[%s] Stage 2/7: Sentiment", run_id)
    news_map = fetch_news_batch([d["ticker"] for d in data])
    for d in data:
        items = news_map.get(d["ticker"], [])
        d["sentiment_score"]    = aggregate_advanced(items, ticker=d["ticker"])
        d["news_volume"]        = len(items)
        d["sentiment_strength"] = float(d["sentiment_score"]) * float(np.log1p(len(items)))
    return data


# ── Stage 3: Signals ──────────────────────────────────────────────────────────
def stage_signals(data, macro, regime, run_id: str = "", horizon: str = "swing"):
    from hedge_fund_ai.data.earnings import earnings_surprise_proxy_fast
    from hedge_fund_ai.data.flows import volume_flow_signal
    from hedge_fund_ai.data.fmp_loader import get_fmp_alpha_signal
    from hedge_fund_ai.data.signal_logger import SignalLogger
    from hedge_fund_ai.data.simfin_loader import compute_piotroski
    from hedge_fund_ai.factors.decision import select_top
    from hedge_fund_ai.factors.factor_health import FactorHealthMonitor, compute_factor_stability
    from hedge_fund_ai.factors.normalization import normalize_features
    from hedge_fund_ai.state.positions import load_prev_portfolio

    logger.info("[%s] Stage 3/7: Signals (regime=%s, horizon=%s)", run_id, regime, horizon)

    fhm      = FactorHealthMonitor()
    disabled = fhm.get_disabled_factors()
    if disabled:
        logger.warning("[%s] Disabled factors: %s", run_id, disabled)

    for d in data:
        ph = d.get("tech", {}).get("price_hist", [])
        vh = d.get("tech", {}).get("volume_hist", [])
        d["earnings_surprise"] = earnings_surprise_proxy_fast(ph)
        d["flow_signal"]       = volume_flow_signal(ph, vh)
        try:
            price = float((ph or [0])[-1] or 0)
            fmp   = get_fmp_alpha_signal(d["ticker"], current_price=price)
            d["fmp_score"] = fmp.get("fmp_score", 0)
        except Exception:
            d["fmp_score"] = 0.0
        try:
            d["piotroski_score"] = compute_piotroski(d.get("fund", {}) or {})
        except Exception:
            d["piotroski_score"] = 0
        ni     = float((d.get("fund", {}) or {}).get("net_income",   0) or 0)
        cfo    = float((d.get("fund", {}) or {}).get("cfo",          0) or 0)
        assets = float((d.get("fund", {}) or {}).get("total_assets", 1) or 1)
        d["accruals"] = float(np.clip((ni - cfo) / max(assets, 1), -0.5, 0.5))

    data = normalize_features(data)

    # Choose signal engine based on horizon
    if horizon == "swing":
        from hedge_fund_ai.factors.swing_signals import (
            build_swing_signal, swing_score_components, update_swing_ic, flush_swing_ic
        )
        build_signal_fn    = build_swing_signal
        score_components_fn= swing_score_components
    else:
        from hedge_fund_ai.factors.signal_engine import (
            build_signal as build_signal_fn,
            score_components as score_components_fn,
        )

    # IC summary
    if horizon == "swing":
        from hedge_fund_ai.factors.swing_signals import _ic_history as ic_hist
    else:
        from hedge_fund_ai.factors.signal_engine import _ic_history as ic_hist

    from hedge_fund_ai.factors.signal_engine import get_factor_ic_summary
    ic_summary = get_factor_ic_summary()
    if ic_summary:
        top_f = sorted(ic_summary.items(), key=lambda x: -x[1]["ic_ir"])[:5]
        logger.info("[%s] Top IC-IR: %s", run_id,
                    " | ".join(f"{f}={v['ic_ir']:.2f}" for f, v in top_f))
        fhm.evaluate(ic_summary)

    prev = load_prev_portfolio()

    top_n = _horizon_params(horizon)["top_n"]
    for d in data:
        raw = build_signal_fn(d, regime)
        d["score"]      = raw
        d["components"] = score_components_fn(d, regime)

    top_stocks = select_top(data, n=top_n, key="score")

    # Trend filter — reduce entries in choppy markets (swing mode only)
    if horizon == "swing":
        from hedge_fund_ai.factors.trend_filter import assess_market_trend, apply_trend_filter
        trend = assess_market_trend(macro)
        scores_map_raw = {d["ticker"]: float(d.get("score", 0)) for d in data}
        top_stocks = apply_trend_filter(top_stocks, scores_map_raw, trend)
        logger.info("[%s] Trend: %s | entries=%d/%d",
                    run_id, trend["regime_label"].upper(),
                    len(top_stocks), top_n)

    # Log signals for IC validation
    try:
        sl = SignalLogger()
        sl.fill_returns()
        scores_map  = {d["ticker"]: d.get("score", 0) for d in data}
        top_tickers = [t if isinstance(t, str) else t.get("ticker", "") for t in top_stocks]
        sl.log_signals(datetime.now(timezone.utc), scores_map,
                       regime=regime, top_picks=top_tickers)
        ic_rep = sl.ic_report()
        logger.info("[%s] Live IC: mean=%.4f IC-IR=%.2f n=%d — %s",
                    run_id, ic_rep.get("mean_ic", 0), ic_rep.get("ic_ir", 0),
                    ic_rep.get("n_filled", 0), ic_rep.get("verdict", "N/A"))
        _save_state("ic_report.json", ic_rep)
    except Exception as e:
        logger.warning("[%s] Signal logger: %s", run_id, e)

    return data, top_stocks


# ── Stage 4: Backtest ─────────────────────────────────────────────────────────
def stage_backtest(universe, run_id: str = "", horizon: str = "swing"):
    from hedge_fund_ai.backtest.factor_attribution import attribution_from_audit, stability_report
    from hedge_fund_ai.backtest.metrics import compute_metrics
    from hedge_fund_ai.backtest.monte_carlo import block_bootstrap_equity, correct_factor_significance, model_capacity
    from hedge_fund_ai.backtest.placebo import run_placebo_test
    from hedge_fund_ai.backtest.validation_report import generate_validation_report
    from hedge_fund_ai.backtest.walk_forward import walk_forward_backtest
    from hedge_fund_ai.backtest.swing_metrics import compute_swing_metrics
    from hedge_fund_ai.data.fundamentals_pit import FundamentalStore
    from hedge_fund_ai.data.signal_logger import SignalLogger
    from hedge_fund_ai.factors.regime import compute_regime_series
    from hedge_fund_ai.state.backtest import load_equity_curve, save_equity_curve

    hparams = _horizon_params(horizon)
    bt_period = hparams["backtest_period"]
    rebal_days = hparams["rebalance_days"]

    equity_path = os.path.join(_STATE, "backtest_equity.json")
    if os.path.exists(equity_path):
        age = (datetime.now().timestamp() - os.path.getmtime(equity_path)) / 86400
        if age < BACKTEST_STALE_DAYS:
            logger.info("[%s] Stage 4/7: Backtest skipped (%.1f days old)", run_id, age)
            ec = load_equity_curve()
            return compute_metrics(ec) if ec else {}, None

    logger.info("[%s] Stage 4/7: Backtest (%s, rebal=%dd, horizon=%s)",
                run_id, bt_period, rebal_days, horizon)
    bt_tickers = [t for t in universe[:40] if t]

    market  = yf.download(bt_tickers + ["SPY"], period=bt_period, progress=False)
    prices  = market["Close"]
    volumes = market.get("Volume")
    spy = yf.download("SPY",  period=bt_period, progress=False)["Close"]
    vix = yf.download("^VIX", period=bt_period, progress=False)["Close"]
    if isinstance(spy, pd.DataFrame): spy = spy.iloc[:, 0]
    if isinstance(vix, pd.DataFrame): vix = vix.iloc[:, 0]

    regime_series = compute_regime_series(spy, vix)

    logger.info("[%s] Loading fundamentals...", run_id)
    fund_store = FundamentalStore(bt_tickers)
    try:
        fund_store.load_all(max_workers=6, try_simfin=True)
    except Exception as e:
        logger.warning("[%s] Fundamentals: %s", run_id, e)
        fund_store = None

    equity, spy_equity, ew_equity, audit_log = walk_forward_backtest(
        prices, volumes, regime_series, bt_tickers,
        fundamental_store=fund_store,
        use_survivorship_filter=True,
        rebalance_step=rebal_days,
    )

    try:
        audit_log.save()
    except Exception as e:
        logger.warning("[%s] Audit save: %s", run_id, e)

    save_equity_curve(equity)

    spy_aligned = spy_equity.reindex(equity.index, method="ffill") if not spy_equity.empty else None
    ew_aligned  = ew_equity.reindex(equity.index, method="ffill")  if not ew_equity.empty else None
    metrics = compute_metrics(equity, benchmark=spy_aligned, ew_equity=ew_aligned, audit_log=audit_log)
    metrics["factor_attribution"] = attribution_from_audit(audit_log)
    metrics["stability"]          = stability_report(audit_log)

    # Swing-specific metrics
    if horizon == "swing":
        swing_m = compute_swing_metrics(equity, audit_log)
        metrics["swing_metrics"] = swing_m
        logger.info("[%s] Swing verdict: %s", run_id, swing_m.get("swing_verdict", "N/A"))

    # Placebo
    placebo = None
    if not equity.empty and len(equity) > 50:
        logger.info("[%s] Placebo test (50 trials)...", run_id)
        try:
            placebo = run_placebo_test(prices, volumes, regime_series, bt_tickers,
                                       model_equity=equity, n_trials=50)
            metrics["placebo_test"] = placebo
            logger.info("[%s] Placebo: %s", run_id, placebo.get("verdict", ""))
        except Exception as e:
            logger.warning("[%s] Placebo: %s", run_id, e)

    # Monte Carlo
    if not equity.empty and len(equity) > 50:
        try:
            mc = block_bootstrap_equity(equity, n_simulations=200)
            metrics["monte_carlo"] = mc
        except Exception as e:
            logger.warning("[%s] Monte Carlo: %s", run_id, e)

    # BH correction
    try:
        from hedge_fund_ai.factors.signal_engine import _ic_history
        ic_hist_raw = {f: list(v) for f, v in _ic_history.items() if len(v) >= 6}
        if ic_hist_raw:
            sig_tests = correct_factor_significance(ic_hist_raw)
            metrics["factor_significance"] = sig_tests
    except Exception:
        pass

    # Capacity
    if not equity.empty:
        try:
            cap = model_capacity(equity, audit_log)
            metrics["capacity"] = cap
        except Exception:
            pass

    # Live IC
    signal_ic = None
    try:
        sl = SignalLogger()
        signal_ic = sl.ic_report()
    except Exception:
        pass

    # Validation report
    try:
        report_txt = generate_validation_report(metrics, placebo, audit_log, signal_ic, equity)
        logger.info("\n%s", report_txt)
    except Exception as e:
        logger.warning("[%s] Validation report: %s", run_id, e)

    _save_state("last_metrics.json", metrics)

    if metrics:
        sm = metrics.get("swing_metrics", {})
        logger.info(
            "[%s] Backtest | Sharpe=%.2f | CAGR=%.1f%% | vs_SPY=%.1f%% | "
            "SwingWR=%.1f%% | PF=%.2f",
            run_id, metrics.get("sharpe", 0), metrics.get("cagr", 0)*100,
            metrics.get("excess_vs_spy", 0)*100,
            sm.get("win_rate", 0)*100, sm.get("profit_factor", 0),
        )

    return metrics, audit_log


# ── Stage 5: Portfolio ────────────────────────────────────────────────────────
def stage_portfolio(data, top_stocks, regime="neutral",
                    run_id: str = "", horizon: str = "swing"):
    from hedge_fund_ai.execution.prices import get_prices
    from hedge_fund_ai.portfolio.optimizer import optimize_portfolio
    from hedge_fund_ai.portfolio.risk_model import drawdown_control, dynamic_vol_target_scale
    from hedge_fund_ai.state.backtest import load_equity_curve
    from hedge_fund_ai.state.performance import log_portfolio
    from hedge_fund_ai.state.positions import save_positions
    from hedge_fund_ai.execution.swing_tracker import SwingTracker

    logger.info("[%s] Stage 5/7: Portfolio (%d candidates, horizon=%s)",
                run_id, len(top_stocks), horizon)

    if not top_stocks or not data:
        logger.warning("[%s] No candidates — skipping portfolio", run_id)
        return [], {}

    hparams = _horizon_params(horizon)

    portfolio = optimize_portfolio(data, top_stocks)
    if not portfolio:
        return [], {}

    tickers = [p["ticker"] for p in portfolio]
    prices  = get_prices(tickers)

    for p in portfolio:
        p["price"]    = float(prices.get(p["ticker"], 0) or 0)
        p["notional"] = round(PORTFOLIO_NOTIONAL_USD * float(p["weight"]), 2)
        p["regime"]   = regime

    # Dynamic vol + drawdown scaling
    try:
        ec = load_equity_curve()
        if ec and len(ec) > 1:
            eq_series = pd.Series(ec)
            dd_scale  = drawdown_control(eq_series)
            vol_scale = dynamic_vol_target_scale(
                eq_series, target_vol=hparams["target_vol"], regime=regime)
            scale = min(dd_scale, vol_scale)
            if scale < 0.99:
                logger.warning("[%s] Scaling exposure: %.2f", run_id, scale)
                for p in portfolio:
                    p["weight"] = round(float(p["weight"]) * scale, 4)
    except Exception:
        pass

    # Swing tracker — sync positions, apply exits
    if horizon == "swing":
        tracker = SwingTracker()
        tracker.update_prices(prices)
        exits = tracker.sync_with_portfolio(portfolio, prices)
        if exits:
            logger.info("[%s] Swing exits: %s", run_id, exits)
        logger.info("[%s] %s", run_id, tracker.summary_log())
        _save_state("swing_metrics.json", tracker.compute_metrics())

    log_portfolio(portfolio, prices)
    save_positions(
        {p["ticker"]: float(p["weight"]) for p in portfolio},
        prices,
        {p["ticker"]: p.get("sector") for p in portfolio},
    )
    logger.info("[%s] Portfolio: %s", run_id,
                " | ".join(f"{p['ticker']} {p['weight']*100:.1f}%" for p in portfolio[:6]))
    return portfolio, prices


# ── Stage 6: Reporting ────────────────────────────────────────────────────────
def stage_reporting(portfolio, data, metrics, macro, top_stocks, audit_log,
                    run_id: str = "", horizon: str = "swing"):
    from hedge_fund_ai.reporting.cio_report import generate_cio_report
    from hedge_fund_ai.reporting.explainer import generate_explanation
    from hedge_fund_ai.reporting.factor_attribution import compute_factor_attribution, explain_factor_attribution
    from hedge_fund_ai.reporting.pdf_report import generate_pdf
    from hedge_fund_ai.reporting.strategy_review import analyze_strategy
    from hedge_fund_ai.reporting.telegram import send_telegram
    from hedge_fund_ai.execution.report import generate_execution_report

    logger.info("[%s] Stage 6/7: Reporting (horizon=%s)", run_id, horizon)

    data_by_t = {d.get("ticker"): d for d in data}
    port_data = []
    for p in (portfolio or []):
        d    = data_by_t.get(p["ticker"], {})
        tech = d.get("tech", {}) or {}
        fund = d.get("fund", {}) or {}
        port_data.append({
            **p,
            "ret_3m":          tech.get("ret_3m"),
            "volatility":      tech.get("ann_vol_30d"),
            "roe_pct":         fund.get("roe_pct"),
            "revenue_growth":  fund.get("revenue_growth"),
            "piotroski_score": d.get("piotroski_score", 0),
            "fmp_score":       d.get("fmp_score", 0),
            "sentiment_score": d.get("sentiment_score"),
            "features":        d.get("features", {}),
            "components":      d.get("components", {}),
        })

    cio = generate_cio_report(metrics or {}, port_data, macro)
    if metrics:
        analyze_strategy(metrics)

    result = generate_execution_report(portfolio or [], macro)
    result["explanation"] = generate_explanation(result)
    result["cio_report"]  = cio

    # Build Telegram message
    sm      = (metrics or {}).get("swing_metrics", {})
    placebo = (metrics or {}).get("placebo_test", {})
    msg     = (
        f"📈 [{run_id}] {horizon.upper()} | {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"
        f"Sharpe={metrics.get('sharpe','?')} | CAGR={metrics.get('cagr','?')} | "
        f"vs SPY={metrics.get('excess_vs_spy','?')}\n"
    )
    if sm:
        msg += (f"Swing: WR={sm.get('win_rate',0)*100:.1f}% | "
                f"PF={sm.get('profit_factor',0):.2f} | "
                f"AvgHold={sm.get('avg_period_return',0):.1f}d\n"
                f"Verdict: {sm.get('swing_verdict','N/A')}\n")
    msg += f"{' | '.join(p['ticker'] for p in (portfolio or [])[:6])}"

    send_telegram(msg)
    generate_pdf(f"[{run_id}] {horizon.upper()} Report\n{cio}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main(dry_run: bool = False, horizon: str = "swing"):
    try:
        report = stage_startup(dry_run=dry_run, horizon=horizon)
        run_id = report.run_id
    except Exception as e:
        logging.critical("Startup failed: %s", e)
        return {}

    logger.info("=" * 70)
    logger.info("Elite AI Hedge Fund [%s] | %s | horizon=%s",
                run_id, datetime.now(timezone.utc).isoformat(), horizon.upper())
    logger.info("=" * 70)

    from hedge_fund_ai.data.universe import get_dynamic_universe
    from hedge_fund_ai.data.price_cache import get_cache
    universe = get_dynamic_universe()
    if len(universe) < MIN_UNIVERSE_SIZE:
        universe = UNIVERSE

    # Clear stale cache on startup
    try:
        get_cache().clear_stale()
    except Exception:
        pass

    data, macro, macro_history, regime = stage_fetch_data(universe, run_id=run_id)

    try:
        data = stage_sentiment(data, run_id=run_id)
    except Exception as e:
        logger.error("[%s] Sentiment: %s", run_id, e)

    data, top_stocks = stage_signals(data, macro, regime,
                                     run_id=run_id, horizon=horizon)

    metrics, audit_log = stage_backtest(universe, run_id=run_id, horizon=horizon)

    portfolio, prices = stage_portfolio(data, top_stocks, regime=regime,
                                        run_id=run_id, horizon=horizon)

    result = {}
    try:
        result = stage_reporting(portfolio, data, metrics, macro, top_stocks,
                                 audit_log, run_id=run_id, horizon=horizon)
    except Exception as e:
        logger.error("[%s] Reporting: %s", run_id, e)

    if not ALPACA_DRY_RUN and not dry_run:
        logger.info("[%s] Stage 7/7: Execution (Alpaca)", run_id)
        try:
            from hedge_fund_ai.execution.alpaca import execute_portfolio
            from hedge_fund_ai.state.backtest import load_equity_curve
            ec = load_equity_curve() or [1.0]
            exec_result = execute_portfolio(portfolio or [], equity_curve=ec)
            logger.info("[%s] Execution: %d orders | %d errors | %d stop-closes",
                        run_id,
                        len(exec_result.get("orders", [])),
                        len(exec_result.get("errors", [])),
                        len(exec_result.get("stop_loss_closes", [])))
        except Exception as e:
            logger.error("[%s] Execution: %s", run_id, e)
    else:
        logger.info("[%s] Stage 7/7: Skipped (dry_run)", run_id)

    logger.info("[%s] Pipeline complete.", run_id)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Elite AI Hedge Fund — Swing Trading")
    parser.add_argument("--mode",     default="paper", choices=["paper","live","dry"])
    parser.add_argument("--horizon",  default="swing", choices=["swing","longterm"])
    parser.add_argument("--universe", default=None)
    parser.add_argument("--lookback", default=None, type=int)
    parser.add_argument("--no-backtest", action="store_true")
    args = parser.parse_args()

    import hedge_fund_ai.config as _cfg
    if args.mode == "dry":
        os.environ["ALPACA_DRY_RUN"] = "true"
    elif args.mode == "live":
        os.environ["ALPACA_DRY_RUN"] = "false"
        os.environ["ALPACA_PAPER"]   = "false"
    else:
        os.environ["ALPACA_DRY_RUN"] = "true"
        os.environ["ALPACA_PAPER"]   = "true"

    if args.universe and args.universe.lower() != "sp500":
        os.environ["UNIVERSE"] = args.universe

    if args.lookback:
        for d, p in [(252,"1y"),(504,"2y"),(756,"3y"),(1260,"5y"),(2520,"10y")]:
            if args.lookback <= d:
                os.environ["SWING_BACKTEST_PERIOD"] = p
                os.environ["BACKTEST_PERIOD"] = p
                break

    if args.no_backtest:
        os.environ["BACKTEST_STALE_DAYS"] = "99999"

    os.environ["HORIZON"] = args.horizon
    main(dry_run=(args.mode == "dry"), horizon=args.horizon)
