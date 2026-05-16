"""
Elite normalization pipeline.

Improvements:
  1. Sector-neutral z-scoring: z-score within sector first, then globally
     (prevents technology sector from dominating the universe)
  2. All new factors: Piotroski, ROIC, accruals, FMP alpha, sector momentum
  3. Composite quality using SimFin-derived ROIC + Piotroski
  4. Earnings quality via accruals ratio
"""
import numpy as np


def winsorize(arr: np.ndarray, lower: float = 2.5, upper: float = 97.5) -> np.ndarray:
    lo = np.nanpercentile(arr, lower)
    hi = np.nanpercentile(arr, upper)
    return np.clip(arr, lo, hi)


def zscore(values, winsor: bool = True) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if winsor:
        arr = winsorize(arr)
    mu  = np.nanmean(arr)
    std = np.nanstd(arr) + 1e-6
    return (arr - mu) / std


def sector_neutral_zscore(values: list[float], sectors: list[str]) -> np.ndarray:
    """
    Z-score within sector first, then globally.
    This prevents one sector from monopolizing the top/bottom of rankings.

    Algorithm:
      1. For each sector: demean and scale by that sector's std
      2. Apply global winsorize + zscore on the sector-demeaned values
    """
    arr     = np.array(values, dtype=float)
    sectors = list(sectors)
    result  = arr.copy()

    unique_sectors = set(sectors)
    for sec in unique_sectors:
        idx = [i for i, s in enumerate(sectors) if s == sec]
        if len(idx) < 2:
            continue
        sub = arr[idx]
        mu  = float(np.nanmean(sub))
        std = float(np.nanstd(sub)) + 1e-6
        for i in idx:
            result[i] = (arr[i] - mu) / std

    # Global normalization on sector-adjusted values
    return zscore(result.tolist(), winsor=True)


def normalize_features(data: list[dict]) -> list[dict]:
    """
    Build cross-sectionally z-scored (and sector-neutral) features for each stock.
    Attaches 'features' dict to each entry in data.
    """
    if not data:
        return data

    def _f(d, *path, default=0.0):
        """Safe nested get."""
        cur = d
        for k in path:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k)
            if cur is None:
                return default
        try:
            return float(cur) if cur is not None else default
        except Exception:
            return default

    sectors = [d.get("sector", "Unknown") or "Unknown" for d in data]
    tech    = [d.get("tech", {}) or {} for d in data]
    fund    = [d.get("fund", {}) or {} for d in data]

    # ── Momentum factors ────────────────────────────────────────────────────────
    mom12_1 = [_f(d, "tech", "ret_12m_skip1m") for d in data]
    mom3m   = [_f(d, "tech", "ret_3m")         for d in data]
    mom6m   = [_f(d, "tech", "ret_6m")         for d in data]
    ret_1m  = [_f(d, "tech", "ret_1m")         for d in data]
    accel   = [m3 - m6 for m3, m6 in zip(mom3m, mom6m)]
    reversal= [-r for r in ret_1m]

    # Cross-sectional relative strength (vs universe median)
    mom3_arr  = np.array(mom3m, dtype=float)
    rel_str   = (mom3_arr - float(np.nanmedian(mom3_arr))).tolist()

    # Sector-relative momentum (stock vs its sector median)
    sector_medians = {}
    for s in set(sectors):
        idx = [i for i, sec in enumerate(sectors) if sec == s]
        vals = [mom3m[i] for i in idx]
        sector_medians[s] = float(np.nanmedian(vals)) if vals else 0.0
    sect_mom = [mom3m[i] - sector_medians.get(sectors[i], 0.0) for i in range(len(data))]

    # ── Quality / profitability factors ─────────────────────────────────────────
    roe       = [_f(d, "fund", "roe_pct")         for d in data]
    roic      = [_f(d, "fund", "roic_pct")        for d in data]
    op_margin = [_f(d, "fund", "op_margin_pct")   for d in data]
    fcf_margin= [_f(d, "fund", "fcf_margin_pct")  for d in data]
    gm        = [_f(d, "fund", "gross_margin_pct")for d in data]
    piotroski = [float(d.get("piotroski_score", 0) or 0) for d in data]

    # Composite quality: ROIC-anchored (most predictive)
    quality   = [0.30*r + 0.25*ri + 0.20*o + 0.15*f + 0.10*g
                 for r, ri, o, f, g in zip(roe, roic, op_margin, fcf_margin, gm)]

    # ── Growth ─────────────────────────────────────────────────────────────────
    rev_growth = [_f(d, "fund", "revenue_growth")  for d in data]
    ni_growth  = [_f(d, "fund", "ni_growth")       for d in data]
    growth     = [0.6*rg + 0.4*ng for rg, ng in zip(rev_growth, ni_growth)]

    # ── Value ──────────────────────────────────────────────────────────────────
    value_raw = []
    for d in data:
        pe = d.get("fund", {}).get("pe_ratio")
        ps = d.get("fund", {}).get("ps_ratio")
        if pe is None and ps is None:
            value_raw.append(np.nan)
        else:
            value_raw.append(-(0.5 * (pe or 0) + 0.5 * (ps or 0)))
    va = np.array(value_raw, dtype=float)
    va = np.where(np.isnan(va), np.nanmedian(va[np.isfinite(va)]) if np.any(np.isfinite(va)) else 0.0, va)

    # ── Alternative alpha (FMP) ────────────────────────────────────────────────
    fmp_alpha  = [float(d.get("fmp_score", 0) or 0) for d in data]
    eps_surp   = [float(d.get("earnings_surprise", 0) or 0) for d in data]
    sentiment  = [float(d.get("sentiment_score", 0) or 0) for d in data]

    # ── Risk / structural ──────────────────────────────────────────────────────
    volatility = [_f(d, "tech", "ann_vol_30d", default=20) for d in data]
    flow       = [float(d.get("flow_signal", 0) or 0)       for d in data]
    liquidity  = [float(d.get("liquidity", 0) or 0)         for d in data]
    vol_reg    = [_f(d, "tech", "vol_regime", default=1)     for d in data]
    high52     = [_f(d, "tech", "high_52w_proximity", default=0.5) for d in data]
    trend_sl   = [_f(d, "tech", "trend_slope")               for d in data]
    skewness   = [float(d.get("skewness", 0) or 0)           for d in data]
    accruals   = [float(d.get("accruals", 0) or 0)           for d in data]

    # ── Build z-scored feature dict ────────────────────────────────────────────
    # Use sector-neutral z-score for momentum and quality (biggest sector tilts)
    z = {
        "momentum_12_1":  sector_neutral_zscore(mom12_1, sectors),
        "momentum":       sector_neutral_zscore(mom3m,   sectors),
        "price_accel":    zscore(accel),
        "reversal":       zscore(reversal),
        "rel_strength":   zscore(rel_str),
        "sector_momentum":zscore(sect_mom),
        "quality":        sector_neutral_zscore(quality,    sectors),
        "piotroski":      zscore(piotroski),
        "roic":           sector_neutral_zscore(roic,       sectors),
        "growth":         sector_neutral_zscore(rev_growth, sectors),
        "value":          zscore(va.tolist(), winsor=False),
        "fmp_alpha":      zscore(fmp_alpha),
        "earnings_surprise": zscore(eps_surp),
        "sentiment":      zscore(sentiment),
        "flow_signal":    zscore(flow),
        "liquidity":      zscore(liquidity),
        "volatility":     zscore(volatility),
        "vol_regime":     zscore(vol_reg),
        "high_52w":       zscore(high52),
        "trend_slope":    zscore(trend_sl),
        "skewness":       zscore(skewness),
        "accruals":       zscore(accruals),
        "mean_reversion": zscore([_f(d, "tech", "z_score") for d in data]),
    }

    for i, d in enumerate(data):
        d["features"] = {k: float(v[i]) for k, v in z.items()}

    return data
