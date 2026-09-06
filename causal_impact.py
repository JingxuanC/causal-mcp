#!/usr/bin/env python3
"""causal_impact.py — 事件反事实影响评估（对标 Google CausalImpact 的 BSTS）。

fallback（默认，纯 numpy）：市场模型 OLS 对冲回归 —— 用事件前窗口把股票收益
对基准收益做 OLS，外推事件后反事实收益，实际 - 反事实 = 影响；经验 p 值为
事件前残差同长度滚动窗口累计值中 |cum| >= 观测值的比例（双侧置换式）。
可选增强：装了 pycausalimpact（import 名 causalimpact）时走真 BSTS
（Brodersen et al., Google, Annals of Applied Statistics 2015），惰性 import，
失败自动降级回 ols_hedge。method 字段标注实际走的实现。
"""

from __future__ import annotations

import numpy as np


def _returns(klines: list) -> tuple[list, np.ndarray]:
    dates, closes = [], []
    for k in klines:
        dates.append(k["date"])
        closes.append(float(k["close"]))
    closes = np.asarray(closes, dtype=float)
    rets = closes[1:] / closes[:-1] - 1.0
    return dates[1:], rets


def _ols_hedge(pre_stock: np.ndarray, pre_bench: np.ndarray,
               post_stock: np.ndarray, post_bench: np.ndarray) -> dict:
    n_pre, n_post = len(pre_stock), len(post_stock)
    X = np.column_stack([np.ones(n_pre), pre_bench])
    coef, *_ = np.linalg.lstsq(X, pre_stock, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])

    pre_pred = X @ coef
    pre_resid = pre_stock - pre_pred
    sigma = float(np.std(pre_resid, ddof=2)) if n_pre > 2 else float(np.std(pre_resid))

    ss_res = float(np.sum(pre_resid ** 2))
    ss_tot = float(np.sum((pre_stock - pre_stock.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # 事件后反事实：用对冲模型外推
    cf = alpha + beta * post_bench
    impact_daily = post_stock - cf
    cum_impact = float(np.sum(impact_daily))

    # 经验 p 值：事件前残差同长度滚动窗口累计的双侧分位
    if n_pre > n_post:
        cums = np.array([np.sum(pre_resid[i:i + n_post]) for i in range(n_pre - n_post + 1)])
        p_value = float((1 + np.sum(np.abs(cums) >= abs(cum_impact))) / (1 + len(cums)))
    else:
        p_value = 1.0

    # 95% 区间：σ·√N·√(1+1/n_pre)，含外推不确定性
    half = 1.96 * sigma * np.sqrt(n_post * (1.0 + 1.0 / max(n_pre, 1)))
    return {
        "cumulative_impact": round(cum_impact, 6),
        "avg_daily_impact": round(cum_impact / max(n_post, 1), 6),
        "p_value": round(p_value, 4),
        "ci_95": [round(cum_impact - half, 6), round(cum_impact + half, 6)],
        "significant": bool(p_value < 0.05),
        "alpha": round(alpha, 6),
        "beta": round(beta, 4),
        "r_squared": round(r2, 4),
        "sigma": round(sigma, 6),
        "method": "ols_hedge",
    }


def _bsts(pre_stock: np.ndarray, pre_bench: np.ndarray,
          post_stock: np.ndarray, post_bench: np.ndarray,
          pre_dates: list, post_dates: list) -> dict:
    import pandas as pd
    from causalimpact import CausalImpact  # pycausalimpact，惰性导入

    idx = pd.to_datetime(pre_dates + post_dates)
    df = pd.DataFrame({"y": np.concatenate([pre_stock, post_stock]),
                       "x1": np.concatenate([pre_bench, post_bench])}, index=idx)
    pre_period = [idx[0], idx[len(pre_stock) - 1]]
    post_period = [idx[len(pre_stock)], idx[-1]]
    ci = CausalImpact(df, pre_period, post_period)
    inf = ci.inferences
    cum = float(inf["post_cum_effects"].iloc[-1])
    n_post = len(post_stock)
    lower = float(inf["post_cum_effects_lower"].iloc[-1])
    upper = float(inf["post_cum_effects_upper"].iloc[-1])
    p_value = float(ci.p_value) if hasattr(ci, "p_value") else np.nan
    return {
        "cumulative_impact": round(cum, 6),
        "avg_daily_impact": round(cum / max(n_post, 1), 6),
        "p_value": round(p_value, 4),
        "ci_95": [round(lower, 6), round(upper, 6)],
        "significant": bool(p_value < 0.05),
        "method": "bsts",
    }


def causal_impact(symbol: str, event_date: str, klines: list,
                  benchmark: list) -> dict:
    stock_dates, stock_rets = _returns(klines)
    if benchmark:
        _, bench_rets = _returns(benchmark)
        n = min(len(stock_rets), len(bench_rets))
        stock_rets, bench_rets = stock_rets[-n:], bench_rets[-n:]
        stock_dates = stock_dates[-n:]
    else:
        bench_rets = np.zeros(len(stock_rets))

    pre_idx = [i for i, d in enumerate(stock_dates) if d < event_date]
    post_idx = [i for i, d in enumerate(stock_dates) if d >= event_date]
    if len(pre_idx) < 20 or len(post_idx) < 1:
        return {"error": "insufficient data (need >=20 pre-event returns and >=1 post)",
                "symbol": symbol, "event_date": event_date, "status": "insufficient data"}

    pre_stock, pre_bench = stock_rets[pre_idx], bench_rets[pre_idx]
    post_stock, post_bench = stock_rets[post_idx], bench_rets[post_idx]

    base = {"symbol": symbol, "event_date": event_date,
            "n_pre": len(pre_idx), "n_post": len(post_idx),
            "post_window": [stock_dates[post_idx[0]], stock_dates[post_idx[-1]]],
            "status": "ok"}

    # 可选增强：装了 pycausalimpact 走真 BSTS，失败降级回 OLS 对冲
    try:
        out = _bsts(pre_stock, pre_bench, post_stock, post_bench,
                    [stock_dates[i] for i in pre_idx], [stock_dates[i] for i in post_idx])
        out.update(base)
        return out
    except ImportError:
        pass
    except Exception:
        pass  # BSTS 数值失败时同样降级，保证可用性

    out = _ols_hedge(pre_stock, pre_bench, post_stock, post_bench)
    out.update(base)
    return out
