"""
Gap 6: Placebo test (randomised signal baseline).

Motivation:
  If you replace your signal with random scores and run the backtest,
  you should get ~0 Sharpe. If random scores give positive Sharpe in
  your universe over your test period, you have beta exposure, not alpha.

Usage:
  results = run_placebo_test(prices, volumes, regime_series, tickers, n_trials=50)
  print(results["verdict"])

What it tests:
  1. Random permutation: shuffle scores each period — destroys cross-sectional signal
  2. Null hypothesis: H0 = "random selection has same Sharpe as model"
  3. p-value: fraction of random trials beating your model Sharpe
  4. If p < 0.05, your signal is statistically significant vs random
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _random_walk_forward(prices_df, volume_df, tickers, regime_series,
                         rebalance_step: int = 21, seed: int = 0) -> pd.Series:
    """
    Walk-forward backtest with RANDOM scores each period.
    Everything else (costs, weights, constraints) is identical to the real backtest.
    """
    from hedge_fund_ai.backtest.walk_forward import (
        _pit_universe, _adv_map, _cs_zscore, _ledoit_wolf_cov,
        _vol_target_scale, _drawdown_scale, _to_df,
    )
    from hedge_fund_ai.backtest.costs import compute_rebalance_cost
    from hedge_fund_ai.config import MIN_UNIVERSE_SIZE, TURNOVER_THRESHOLD

    rng = np.random.default_rng(seed)
    prices_df = _to_df(prices_df, "price")
    volume_df = _to_df(volume_df, "vol") if volume_df is not None else pd.DataFrame(index=prices_df.index)
    returns_df = prices_df.pct_change()

    equity: list[float] = [1.0]
    dates:  list        = []
    prev_weights: dict  = {}

    for i in range(100, len(prices_df) - rebalance_step, rebalance_step):
        date  = prices_df.index[i]
        valid = [t for t in _pit_universe(prices_df, volume_df, i) if t in tickers]
        if len(valid) < MIN_UNIVERSE_SIZE:
            continue

        # ── RANDOM SCORES (this is the placebo) ───────────────────────────────
        random_scores = {t: float(rng.standard_normal()) for t in valid}
        cs_scores     = _cs_zscore(random_scores)

        top = sorted(cs_scores, key=cs_scores.get, reverse=True)[:10]
        if not top:
            continue

        # Same portfolio construction as real backtest
        try:
            ret_w = pd.DataFrame({
                t: prices_df[t].iloc[max(0, i-126):i]
                for t in top if t in prices_df.columns
            }).pct_change().dropna()
            if ret_w.shape[0] >= 30 and ret_w.shape[1] >= 2:
                cov  = _ledoit_wolf_cov(ret_w)
                vols = np.sqrt(np.diag(cov)) + 1e-8
                rp   = (1 / vols) / (1 / vols).sum()
                sc   = np.array([max(cs_scores.get(t, 0), 0) for t in ret_w.columns])
                sc  /= sc.sum() + 1e-9
                w    = 0.60 * rp + 0.40 * sc
                w    = np.clip(w, 0, None)
                w   /= w.sum() + 1e-9
                for _ in range(20):
                    w = np.minimum(w, 0.20)
                    w /= w.sum() + 1e-9
                    if w.max() <= 0.20 + 1e-6:
                        break
                scale = _vol_target_scale(w, cov)
                weights = {t: float(w[j]) * scale for j, t in enumerate(ret_w.columns)}
            else:
                raise ValueError("no data")
        except Exception:
            weights = {t: 1.0 / len(top) for t in top}

        dd_scale = _drawdown_scale(equity)
        if dd_scale < 1.0:
            weights = {t: v * dd_scale for t, v in weights.items()}

        # Turnover filter
        filtered = {}
        for t in set(weights) | set(prev_weights):
            nw, ow = float(weights.get(t, 0)), float(prev_weights.get(t, 0))
            filtered[t] = nw if abs(nw - ow) > TURNOVER_THRESHOLD else ow
        filtered = {t: v for t, v in filtered.items() if v > 1e-6}
        tot = sum(filtered.values()) or 1e-9
        filtered = {t: v / tot for t, v in filtered.items()}

        # Costs
        adv = _adv_map(prices_df, volume_df, i, list(filtered.keys()))
        cost, _ = compute_rebalance_cost(prev_weights, filtered, adv, 100_000)
        current = equity[-1] * (1 - cost)

        for j in range(rebalance_step):
            idx = i + j
            if idx >= len(prices_df):
                break
            dr = sum(
                float(w) * float(returns_df[t].iloc[idx])
                for t, w in filtered.items()
                if t in returns_df.columns and not pd.isna(returns_df[t].iloc[idx])
            )
            current *= (1 + dr)
            equity.append(current)
            dates.append(prices_df.index[idx])

        prev_weights = filtered

    if not dates:
        return pd.Series(dtype=float)
    n = len(dates)
    return pd.Series(equity[1:n+1], index=pd.DatetimeIndex(dates))


def _sharpe(equity: pd.Series) -> float:
    if equity is None or len(equity) < 10:
        return 0.0
    r = equity.pct_change().dropna()
    return float(r.mean() / (r.std() + 1e-10) * np.sqrt(252))


def run_placebo_test(
    prices, volumes, regime_series, tickers,
    model_equity: pd.Series,
    n_trials: int = 50,
    rebalance_step: int = 21,
) -> dict:
    """
    Run n_trials random-signal backtests and compare against the model.

    Args:
        model_equity:  the REAL model's equity curve (pd.Series)
        n_trials:      number of random trials (50 is sufficient)

    Returns dict with:
        model_sharpe:   Sharpe of the real model
        random_mean:    mean Sharpe across random trials
        random_std:     std of random Sharpe
        p_value:        fraction of random trials beating model (empirical p-value)
        verdict:        plain-English assessment
    """
    from hedge_fund_ai.backtest.walk_forward import _to_df

    prices_df = _to_df(prices, "price")
    volume_df = _to_df(volumes, "vol") if volumes is not None else pd.DataFrame(index=prices_df.index)
    tickers   = [t for t in tickers if t]

    model_sharpe  = _sharpe(model_equity)
    random_sharpes = []

    logger.info("Placebo test: %d random trials...", n_trials)
    for seed in range(n_trials):
        eq = _random_walk_forward(
            prices_df, volume_df, tickers, regime_series,
            rebalance_step=rebalance_step, seed=seed,
        )
        random_sharpes.append(_sharpe(eq))
        if (seed + 1) % 10 == 0:
            logger.info("  Placebo %d/%d done", seed + 1, n_trials)

    random_sharpes = np.array(random_sharpes)
    p_value = float(np.mean(random_sharpes >= model_sharpe))

    result = {
        "model_sharpe":  round(model_sharpe, 3),
        "random_mean":   round(float(np.mean(random_sharpes)), 3),
        "random_std":    round(float(np.std(random_sharpes)), 3),
        "random_p5":     round(float(np.percentile(random_sharpes, 5)), 3),
        "random_p95":    round(float(np.percentile(random_sharpes, 95)), 3),
        "p_value":       round(p_value, 3),
        "n_trials":      n_trials,
        "verdict":       _placebo_verdict(model_sharpe, p_value,
                                          float(np.mean(random_sharpes))),
    }

    logger.info(
        "Placebo result: model=%.2f | random mean=%.2f ± %.2f | p=%.3f | %s",
        model_sharpe, result["random_mean"], result["random_std"],
        p_value, result["verdict"],
    )
    return result


def _placebo_verdict(model_sharpe: float, p_value: float, random_mean: float) -> str:
    if p_value < 0.05 and model_sharpe > random_mean:
        return f"SIGNIFICANT: model beats random at p={p_value:.3f} — alpha likely real"
    if p_value < 0.10:
        return f"BORDERLINE: p={p_value:.3f} — need more data to confirm alpha"
    if model_sharpe <= random_mean:
        return f"NO ALPHA: model ({model_sharpe:.2f}) does not beat random ({random_mean:.2f})"
    return f"NOT SIGNIFICANT: p={p_value:.3f} — cannot distinguish from luck"
