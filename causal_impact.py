#!/usr/bin/env python3
"""causal_impact.py — 事件反事实影响评估（对标 Google CausalImpact 的 BSTS）。

fallback（默认，纯 numpy）：市场模型 OLS 对冲回归 —— 用事件前窗口把股票收益
对基准收益做 OLS，外推事件后反事实收益，实际 - 反事实 = 影响。
p 值有两套，字段不混用：
  * p_value（主）= 累计预测误差的连续双侧 t 检验，方差含估计窗参数不确定度
    Var(ΣAR)=σ²(L + L²/n_pre + (Σ_post(x-x̄))²/S_xx)，分辨率不受块数限制；
  * p_value_empirical = 事件前残差长度 L 的窗口累计和双侧经验分位（+1 修正），
    同时输出 n_blocks / p_value_floor（经验 p 的分辨率下界）与 block_mode；
    n_blocks < 20 时标 empirical_status="insufficient_pre_window" 并降
    blocks_reliable=false —— 不再硬编码 p=1.0，也不再让粗分辨率冒充结论。
benchmark 缺失/长度不匹配/零方差时降级为均值调整超额收益（method="mean_adjusted"，
r_squared=None），绝不拿零列硬跑 OLS。
可选增强：装了 pycausalimpact（import 名 causalimpact）时走真 BSTS
（Brodersen et al., Google, Annals of Applied Statistics 2015），惰性 import，
失败自动降级回 ols_hedge。method 字段标注实际走的实现。
"""

from __future__ import annotations

import numpy as np

# 经验置换块数下限：低于此值经验 p 的分辨率过粗（下界 1/(1+n_blocks) 过大），
# 必须显式标注为不可靠，不能当作显著/不显著的结论依据（见 _ols_hedge）。
MIN_PERMUTATION_BLOCKS = 20


def _two_sided_t_p(t_stat: float, df: int):
    """双侧 t 检验 p 值；无 scipy 时退化为正态近似（大样本下差异可忽略）。"""
    if df <= 0 or not np.isfinite(t_stat):
        return None
    try:
        from scipy import stats as _st  # 惰性导入，不把 scipy 变成硬依赖
        return float(2.0 * (1.0 - _st.t.cdf(abs(t_stat), df)))
    except Exception:
        import math
        return float(math.erfc(abs(t_stat) / math.sqrt(2.0)))


def _fmt_p(p, nd: int = 4):
    """p 值格式化：极小 p 用科学计数法表示，避免 round 之后显示成 0.0。"""
    if p is None:
        return None
    p = float(p)
    if p < 10 ** (-nd):
        return float(f"{p:.2e}")
    return round(p, nd)


def _block_sums(pre_resid: np.ndarray, block_len: int) -> tuple[np.ndarray, str]:
    """事件前残差中长度 block_len 的窗口累计和 → 经验零分布样本。

    只使用线性滚动窗口（重叠），窗口数 = n_pre - block_len + 1。
    当 block_len >= n_pre 时线性窗口数为 0 —— 旧代码正是在这里硬编码 p=1.0（P1 修复点），
    现在改为诚实返回空数组（p_value_empirical=None + empirical_status=
    "insufficient_pre_window"），而不是编造一个 p，也不是用循环窗口凑数：
    循环窗口在 block_len ≈ n_pre 时每个窗口几乎覆盖全部 pre 残差、彼此仅差一项，
    零分布过窄 → 纯噪声下实测假阳性率 29.2%（名义 α=0.05，3000 次重复），
    属于典型的误导性结论，故不采用。
    """
    n_pre = len(pre_resid)
    block_len = max(int(block_len), 1)
    if n_pre <= block_len:
        return np.array([], dtype=float), "none"
    cums = np.array([np.sum(pre_resid[i:i + block_len])
                     for i in range(n_pre - block_len + 1)], dtype=float)
    return cums, "rolling_overlapping"


def _returns(klines: list) -> tuple[list, np.ndarray]:
    dates, closes = [], []
    for k in klines:
        dates.append(k["date"])
        closes.append(float(k["close"]))
    closes = np.asarray(closes, dtype=float)
    rets = closes[1:] / closes[:-1] - 1.0
    return dates[1:], rets


def _ols_hedge(pre_stock: np.ndarray, pre_bench, post_stock: np.ndarray,
               post_bench, alpha: float = 0.05,
               benchmark_status: str = "ok") -> dict:
    n_pre, n_post = len(pre_stock), len(post_stock)

    # ── 市场模型（对冲回归）──
    # benchmark 缺失 / 长度不匹配 / 零方差时**不得**拿零列硬跑 OLS：
    # lstsq 对含零列的 X 不报错，会静默退化成常数均值模型却仍报 r_squared
    # （P1 修复点）。此时明确降级为 market-model-free 的均值调整超额收益。
    bench_ok = (pre_bench is not None and post_bench is not None
                and len(pre_bench) == n_pre and len(post_bench) == n_post
                and float(np.std(pre_bench)) > 1e-12)
    if bench_ok:
        X = np.column_stack([np.ones(n_pre), pre_bench])
        if np.linalg.matrix_rank(X) < X.shape[1]:
            bench_ok = False
            benchmark_status = "rank_deficient"
        else:
            coef, *_ = np.linalg.lstsq(X, pre_stock, rcond=None)
            alpha_hat, beta = float(coef[0]), float(coef[1])
            pre_resid = pre_stock - (X @ coef)
            cf = alpha_hat + beta * post_bench
            ss_res = float(np.sum(pre_resid ** 2))
            ss_tot = float(np.sum((pre_stock - pre_stock.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            method, beta_identified = "ols_hedge", True
            x_bar = float(np.mean(pre_bench))
            s_xx = float(np.sum((pre_bench - x_bar) ** 2))
            dev_post = float(np.sum(post_bench - x_bar))
            quad = (n_post ** 2) / n_pre + (dev_post ** 2 / s_xx if s_xx > 0 else 0.0)

    if not bench_ok:
        if benchmark_status == "ok":
            benchmark_status = "missing_or_degenerate"
        alpha_hat, beta, r2 = float(np.mean(pre_stock)), 0.0, None
        pre_resid = pre_stock - alpha_hat
        cf = np.full(n_post, alpha_hat)
        method, beta_identified = "mean_adjusted", False
        quad = (n_post ** 2) / n_pre  # 无回归量：只有估计窗均值的不确定度

    sigma = float(np.std(pre_resid, ddof=2)) if n_pre > 2 else float(np.std(pre_resid))

    impact_daily = post_stock - cf
    cum_impact = float(np.sum(impact_daily))

    # ── 主 p 值：累计预测误差的连续检验 ──
    # Var(Σ AR) = σ²·(L + L²/n_est + (Σ_post(x_t-x̄))²/S_xx)
    # 第一项是事件窗噪声，后两项是估计窗参数（α̂、β̂）的不确定度 —— 旧的
    # “经验置换 p” 分辨率被块数锁死（下界 = 1/(1+n_blocks)），效应再强也卡在
    # 0.0175（60/5）或干脆硬编码 1.0（n_pre<=n_post）。故主 p 用连续检验，
    # 经验置换 p 作为稳健性对照一并输出（含 n_blocks 与下界）。
    var_cum = sigma ** 2 * (n_post + quad)
    se_cum = float(np.sqrt(max(var_cum, 0.0)))
    t_stat = cum_impact / se_cum if se_cum > 0 else 0.0
    p_predictive = _two_sided_t_p(t_stat, n_pre - 2)

    # ── 经验置换 p（双侧，+1 修正），块数与下界一并输出 ──
    cums, block_mode = _block_sums(pre_resid, n_post)
    n_blocks = int(len(cums))
    if n_blocks >= 1:
        n_ge = int(np.sum(np.abs(cums) >= abs(cum_impact)))
        p_empirical = float((1 + n_ge) / (1 + n_blocks))
        p_floor = 1.0 / (1 + n_blocks)
        empirical_status = "ok"
    else:  # block_len >= n_pre：无法构造窗口 → 诚实报无（旧代码在此硬编码 p=1.0）
        p_empirical, p_floor = None, None
        empirical_status = "insufficient_pre_window"

    blocks_reliable = n_blocks >= MIN_PERMUTATION_BLOCKS
    if n_blocks >= 1 and not blocks_reliable:
        empirical_status = "few_blocks_unreliable"
    p_value = p_predictive if p_predictive is not None else p_empirical
    significant = bool(p_value is not None and p_value < alpha)

    note = None
    if n_blocks == 0:
        note = ("no pre-event window of length n_post (n_pre <= n_post): empirical "
                "permutation p unavailable; p_value is the predictive-t p")
    elif not blocks_reliable:
        note = (f"few permutation blocks (n_blocks={n_blocks} < {MIN_PERMUTATION_BLOCKS}): "
                f"empirical p resolution >= {round(p_floor, 4)}; "
                f"significance uses predictive-t p")
    elif p_empirical is not None and p_predictive is not None:
        if (p_empirical < alpha) != (p_predictive < alpha):
            note = ("empirical block p and predictive-t p disagree at alpha="
                    f"{alpha} (p_empirical={_fmt_p(p_empirical)}, "
                    f"p_predictive={_fmt_p(p_predictive)}): treat as inconclusive")

    half = 1.96 * se_cum  # 95% 区间用同一套预测误差标准误
    return {
        "cumulative_impact": round(cum_impact, 6),
        "avg_daily_impact": round(cum_impact / max(n_post, 1), 6),
        "p_value": _fmt_p(p_value),
        "p_value_predictive_t": _fmt_p(p_predictive),
        "p_value_empirical": _fmt_p(p_empirical),
        "p_value_floor": None if p_floor is None else round(p_floor, 6),
        "p_value_source": "predictive_t",
        "n_blocks": n_blocks,
        "block_mode": block_mode,
        "blocks_reliable": bool(blocks_reliable),
        "empirical_status": empirical_status,
        "ci_95": [round(cum_impact - half, 6), round(cum_impact + half, 6)],
        "significant": significant,
        "significant_empirical": (None if p_empirical is None else bool(p_empirical < alpha)),
        "alpha_threshold": round(float(alpha), 4),
        "t_stat": round(float(t_stat), 4),
        "se_cumulative": round(se_cum, 6),
        "se_formula": "sqrt(sigma^2*(L + L^2/n_pre + (sum_post(x-xbar))^2/S_xx))",
        "alpha": round(alpha_hat, 6),
        "beta": round(beta, 4),
        "beta_identified": bool(beta_identified),
        "r_squared": None if r2 is None else round(r2, 4),
        "sigma": round(sigma, 6),
        "benchmark_status": benchmark_status,
        "note": note,
        "method": method,
        "status": "ok" if p_value is not None else "insufficient_pre_window",
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
        "p_value_predictive_t": None,
        "p_value_empirical": None,
        "p_value_floor": None,
        "p_value_source": "bsts_posterior",
        "n_blocks": None,
        "block_mode": None,
        "blocks_reliable": None,
        "ci_95": [round(lower, 6), round(upper, 6)],
        "significant": bool(p_value < 0.05),
        "method": "bsts",
    }


def causal_impact(symbol: str, event_date: str, klines: list,
                  benchmark: list) -> dict:
    stock_dates, stock_rets = _returns(klines)
    benchmark_status = "ok"
    if benchmark and len(benchmark) >= 2:
        _, bench_rets = _returns(benchmark)
        n = min(len(stock_rets), len(bench_rets))
        if len(stock_rets) != len(bench_rets):
            # 长度不一致：按尾部对齐（旧行为保留），但必须标注，不再静默
            benchmark_status = "length_mismatch_tail_aligned"
        stock_rets, bench_rets = stock_rets[-n:], bench_rets[-n:]
        stock_dates = stock_dates[-n:]
    else:
        # 不给 benchmark 时不再用零收益列硬跑 OLS（会静默退化成均值模型）
        bench_rets = None
        benchmark_status = "missing"

    pre_idx = [i for i, d in enumerate(stock_dates) if d < event_date]
    post_idx = [i for i, d in enumerate(stock_dates) if d >= event_date]
    if len(pre_idx) < MIN_PERMUTATION_BLOCKS or len(post_idx) < 1:
        return {"error": "insufficient data (need >=20 pre-event returns and >=1 post)",
                "symbol": symbol, "event_date": event_date, "status": "insufficient data"}

    pre_stock = stock_rets[pre_idx]
    post_stock = stock_rets[post_idx]
    pre_bench = None if bench_rets is None else bench_rets[pre_idx]
    post_bench = None if bench_rets is None else bench_rets[post_idx]

    base = {"symbol": symbol, "event_date": event_date,
            "n_pre": len(pre_idx), "n_post": len(post_idx),
            "post_window": [stock_dates[post_idx[0]], stock_dates[post_idx[-1]]],
            "benchmark_status": benchmark_status,
            "status": "ok"}

    # 可选增强：装了 pycausalimpact 走真 BSTS，失败降级回 OLS 对冲
    try:
        out = _bsts(pre_stock, pre_bench, post_stock, post_bench,
                    [stock_dates[i] for i in pre_idx], [stock_dates[i] for i in post_idx])
        out["benchmark_status"] = benchmark_status
        return {**base, **out}
    except ImportError:
        pass
    except Exception:
        pass  # BSTS 数值失败时同样降级，保证可用性

    out = _ols_hedge(pre_stock, pre_bench, post_stock, post_bench,
                     benchmark_status=benchmark_status)
    # {**base, **out}：_ols_hedge 的 status 可以覆盖 base 的 "ok"
    # （n_blocks 不足时返回 insufficient_pre_window），base 其余字段保留
    return {**base, **out}
