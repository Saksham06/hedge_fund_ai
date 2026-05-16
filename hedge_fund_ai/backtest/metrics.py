"""
Phase 4 metrics: Sharpe, Sortino, Calmar, CAGR, hit ratio, turnover,
benchmark comparison vs SPY and equal-weight.
"""
import numpy as np
import pandas as pd


def compute_metrics(equity, benchmark: pd.Series = None, ew_equity: pd.Series = None,
                    audit_log=None) -> dict:
    if equity is None:
        return {}

    if isinstance(equity, list):
        if not equity:
            return {}
        if isinstance(equity[0], dict):
            equity = pd.Series([e.get("portfolio_value", 1.0) for e in equity])
        else:
            equity = pd.Series([float(x) for x in equity])

    if not isinstance(equity, pd.Series) or len(equity) < 10:
        return {}

    ret = equity.pct_change().dropna()
    if ret.empty:
        return {}

    mu       = float(np.mean(ret))
    sig      = float(np.std(ret)) + 1e-10
    down_std = float(ret[ret < 0].std()) + 1e-10

    sharpe   = float(mu / sig * np.sqrt(252))
    sortino  = float(mu / down_std * np.sqrt(252))

    peak   = equity.cummax()
    dd     = (equity - peak) / (peak + 1e-10)
    max_dd = float(dd.min())

    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1)
    years     = max(len(equity) / 252, 0.01)
    cagr      = float((1 + total_ret) ** (1.0 / years) - 1)
    calmar    = float(cagr / (abs(max_dd) + 1e-10))
    ann_vol   = float(sig * np.sqrt(252))
    win_rate  = float((ret > 0).mean())

    avg_win  = float(ret[ret > 0].mean()) if (ret > 0).any() else 0.0
    avg_loss = float(ret[ret < 0].mean()) if (ret < 0).any() else 0.0
    profit_f = float(abs(avg_win) / (abs(avg_loss) + 1e-10))

    # DD duration
    in_dd, dd_dur, cur = (dd < 0), 0, 0
    for v in in_dd:
        cur = cur + 1 if v else 0
        dd_dur = max(dd_dur, cur)

    result = {
        "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
        "calmar": round(calmar, 3), "max_drawdown": round(max_dd, 4),
        "total_return": round(total_ret, 4), "cagr": round(cagr, 4),
        "ann_vol": round(ann_vol, 4), "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 5), "avg_loss": round(avg_loss, 5),
        "profit_factor": round(profit_f, 3), "max_dd_duration": int(dd_dur),
    }

    # Turnover from audit log
    if audit_log is not None:
        records = audit_log._records
        if records:
            avg_raw   = float(np.mean([r.raw_turnover      for r in records]))
            avg_filt  = float(np.mean([r.filtered_turnover for r in records]))
            avg_cost  = float(np.mean([r.total_cost_frac   for r in records])) * 10_000
            result["avg_raw_turnover"]      = round(avg_raw, 4)
            result["avg_filtered_turnover"] = round(avg_filt, 4)
            result["avg_cost_bps"]          = round(avg_cost, 2)
            result["n_rebalances"]          = len(records)
            result["regime_breakdown"]      = audit_log.regime_breakdown()

    # SPY benchmark
    if benchmark is not None and len(benchmark) > 10:
        try:
            br    = benchmark.pct_change().dropna()
            common = ret.index.intersection(br.index)
            if len(common) > 20:
                p, b  = ret.loc[common], br.loc[common]
                beta  = float(np.cov(p, b)[0, 1] / (np.var(b) + 1e-10))
                alpha = float((np.mean(p) - beta * np.mean(b)) * 252)
                te    = float(np.std(p - b) * np.sqrt(252))
                ir    = float((np.mean(p) - np.mean(b)) / (np.std(p - b) + 1e-10) * np.sqrt(252))
                spy_ret = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1)
                result.update({
                    "beta": round(beta, 3), "alpha_ann": round(alpha, 4),
                    "tracking_error": round(te, 4), "info_ratio": round(ir, 3),
                    "spy_total_return": round(spy_ret, 4),
                    "excess_vs_spy": round(total_ret - spy_ret, 4),
                })
        except Exception:
            pass

    # Equal-weight benchmark
    if ew_equity is not None and len(ew_equity) > 10:
        try:
            ew_ret = float(ew_equity.iloc[-1] / ew_equity.iloc[0] - 1)
            ew_sharpe_r = ew_equity.pct_change().dropna()
            ew_sharpe   = float(ew_sharpe_r.mean() / (ew_sharpe_r.std() + 1e-10) * np.sqrt(252))
            result["ew_total_return"] = round(ew_ret, 4)
            result["ew_sharpe"]       = round(ew_sharpe, 3)
            result["excess_vs_ew"]    = round(total_ret - ew_ret, 4)
        except Exception:
            pass

    return result
