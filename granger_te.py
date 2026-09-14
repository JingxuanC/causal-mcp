#!/usr/bin/env python3
"""granger_te.py — 两序列 lead-lag 因果：Granger 因果检验 + 传递熵。

- Granger：statsmodels grangercausalitytests，双向 F 检验（x→y / y→x），
  取各滞后阶 p 值最小者为 best_lag（Granger 1969）。
- Transfer Entropy（Schreiber 2000, PRL 85(2)）：分位数分箱离散化，
  TE(x→y) = Σ p(y_{t+1}, y_t, x_t) log2 [ p(y_{t+1}|y_t,x_t) / p(y_{t+1}|y_t) ]，
  纯 numpy 3D 直方图实现，双向计算。
lead_lag 判定：仅单向 Granger 显著（α=0.05）→ 该方向 leads；双向显著时
取 p 更小一侧；均不显著看 TE 相对强弱（>20% 差）否则 "none"。
"""

from __future__ import annotations

import numpy as np


def _granger_direction(dst: np.ndarray, src: np.ndarray, max_lag: int) -> dict:
    """检验 src → dst（statsmodels 约定：列 [dst, src]）。

    多重比较校正（P1 修复点）：在 max_lag 个滞后阶里取最小 p 本身就是选择效应
    （5 次检验取最小值），p 会系统性偏小、把噪声报成因果。这里同时输出
    Bonferroni 校正 p（p_value_adjusted，天然满足 >= 原 p）与 BH-FDR 校正 p
    （p_value_fdr），并给出 n_lags_tested；p_value 保留原始最小 p 以兼容旧调用。
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    data = np.column_stack([dst, src])
    try:
        # statsmodels >= 0.15 移除了 verbose 参数；旧版本 verbose 默认 False，
        # 所以先不传，失败再退回旧签名（否则 0.15 下直接 TypeError 崩溃）
        res = grangercausalitytests(data, maxlag=max_lag)
    except TypeError:
        res = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    pvals = {lag: float(res[lag][0]["ssr_ftest"][1]) for lag in range(1, max_lag + 1)}
    n_lags = len(pvals)
    best_lag = min(pvals, key=pvals.get)
    p_raw = float(pvals[best_lag])

    p_bonf = float(min(1.0, p_raw * n_lags))
    # Benjamini-Hochberg step-up：p_adj(i) = min_{j>=i} p_(j)*m/j
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    p_fdr = {}
    prev = 1.0
    for rank in range(n_lags, 0, -1):
        lag, p = ordered[rank - 1]
        prev = min(prev, p * n_lags / rank)
        p_fdr[lag] = float(min(1.0, prev))

    return {
        "best_lag": int(best_lag),
        "p_value": round(p_raw, 6),
        "p_value_raw": round(p_raw, 6),
        "p_value_adjusted": round(p_bonf, 6),
        "p_value_fdr": round(float(p_fdr[best_lag]), 6),
        "n_lags_tested": int(n_lags),
        "correction": "bonferroni",
        "p_by_lag": {str(k): round(float(v), 6) for k, v in pvals.items()},
    }


def _quantile_bins(a: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.quantile(a, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    # 重复分位（常数段）去重，保证 digitize 可用
    qs = np.unique(qs)
    return np.digitize(a, qs[1:-1])


def _transfer_entropy(src: np.ndarray, dst: np.ndarray, n_bins: int = 8) -> float:
    x = _quantile_bins(src, n_bins)
    y = _quantile_bins(dst, n_bins)
    nb = max(x.max(), y.max()) + 1
    joint = np.zeros((nb, nb, nb))  # (y_next, y_now, x_now)
    for t in range(len(x) - 1):
        joint[y[t + 1], y[t], x[t]] += 1
    total = joint.sum()
    if total == 0:
        return 0.0
    p = joint / total
    p_yx = p.sum(axis=0)                       # p(y_now, x_now)
    p_yy = p.sum(axis=2)                       # p(y_next, y_now)
    p_y = p.sum(axis=(0, 2))                   # p(y_now)
    te = 0.0
    for yn in range(nb):
        for yo in range(nb):
            for xo in range(nb):
                if p[yn, yo, xo] > 0 and p_yx[yo, xo] > 0 and p_yy[yn, yo] > 0 and p_y[yo] > 0:
                    te += p[yn, yo, xo] * np.log2(
                        (p[yn, yo, xo] / p_yx[yo, xo]) / (p_yy[yn, yo] / p_y[yo]))
    return float(max(te, 0.0))


def granger_te(x: list, y: list, max_lag: int = 5) -> dict:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    n = min(len(xa), len(ya))
    if n < max_lag * 4 + 10:
        return {"error": f"insufficient data (need >={max_lag * 4 + 10} points, got {n})"}
    xa, ya = xa[-n:], ya[-n:]

    out = {"n": int(n), "max_lag": int(max_lag)}

    try:
        out["granger"] = {
            "x_causes_y": _granger_direction(ya, xa, max_lag),
            "y_causes_x": _granger_direction(xa, ya, max_lag),
        }
    except ImportError:
        return {"error": "statsmodels not installed (pip install statsmodels)"}

    te_xy = _transfer_entropy(xa, ya)
    te_yx = _transfer_entropy(ya, xa)
    out["transfer_entropy"] = {"x_to_y": round(te_xy, 5), "y_to_x": round(te_yx, 5)}

    p_xy = out["granger"]["x_causes_y"]["p_value"]
    p_yx = out["granger"]["y_causes_x"]["p_value"]
    sig_xy, sig_yx = p_xy < 0.05, p_yx < 0.05
    if sig_xy and not sig_yx:
        lead = "x_leads"
    elif sig_yx and not sig_xy:
        lead = "y_leads"
    elif sig_xy and sig_yx:
        lead = "x_leads" if p_xy < p_yx else "y_leads"
    elif te_xy > 1.2 * te_yx and te_xy > 0.001:
        lead = "x_leads"
    elif te_yx > 1.2 * te_xy and te_yx > 0.001:
        lead = "y_leads"
    else:
        lead = "none"
    out["lead_lag"] = lead
    return out
