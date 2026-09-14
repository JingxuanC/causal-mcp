#!/usr/bin/env python3
"""
event_study.py — Financial Event Study using statsmodels OLS.
Supports single-factor (market model) and multi-factor (market + sector).

统计稳健性约定（修复 P1 问题后）：
- benchmark 缺失 / 长度不匹配 / 估计窗内零方差 → **不跑 OLS**，降级为
  market-model-free 的均值调整超额收益，method="raw_excess_return" 且
  r_squared=None（含零列的 lstsq 不报错，会静默退化成常数均值模型却仍报
  r_squared/is_significant —— 禁止）。
- CAR 标准误含估计窗参数不确定度：
  SE(CAR) = sqrt(σ²·(L + 1'X_win(X'X)⁻¹X_win'1))
          ≈ sqrt(L·σ² + L²/n_est·σ² + …)，不再是只有 σ·√L。
- 输出 method / car_se / se_formula / benchmark_status / sector_status 便于调用方
  判断结论的成色，字段名保持向后兼容。
"""

import numpy as np
from scipy import stats


def _cumulative_se(sigma, X, full_factors, window_idx, n, L):
    """CAR 标准误 = sqrt( σ² · (L + 1'·X_win·(X'X)⁻¹·X_win'·1) )

    括号里第一项 L 是事件窗噪声；第二项是**估计窗参数（α̂/β̂）的不确定度**，
    单个回归量时展开为 (L²/n_est + (Σ_win(x-x̄))²/S_xx)，即
        SE(CAR) = sqrt( L·σ² + L²/n_est·σ² + ... )
    旧代码只有 σ·√L，漏掉估计窗项 → t 值系统性偏大、p 值系统性偏小
    （P1 修复点）。无法取到窗口回归量时回退为保守的 L²/n_est 项。
    """
    n_est, k = X.shape[0], X.shape[1]
    quad = (L * L) / max(n_est, 1)          # 保守回退（仅参数不确定度项）
    source = "fallback_L2_over_n_est"
    if window_idx is not None:
        ws, we = window_idx
        rows = []
        for i in range(ws, we + 1):
            if 0 <= i < n:
                rows.append([1.0] + [float(f[i]) for f in full_factors])
        if rows:
            try:
                Xw = np.asarray(rows, dtype=float)
                XtX_inv = np.linalg.pinv(X.T @ X)
                ones = np.ones(len(rows))
                q = float(ones @ (Xw @ XtX_inv @ Xw.T) @ ones)
                if np.isfinite(q) and q > 0:
                    quad, source = q, "exact_1'Xw(X'X)^-1Xw'1"
            except (np.linalg.LinAlgError, ValueError):
                pass
    var = float(sigma) ** 2 * (L + quad)
    return float(np.sqrt(max(var, 0.0))), source, float(quad)


def _raw_excess_car(stock_returns, event_idx, est_start, est_end, car_windows,
                    alpha: float, benchmark_status: str,
                    sector_status: str = "not_provided"):
    """market-model-free 降级：以估计窗均值为基准的简单超额收益（mean-adjusted）。

    benchmark 缺失/长度不匹配/零方差时的唯一诚实出路：**不跑 OLS**，因此
    绝不返回 r_squared（没有回归就不报拟合优度），method 明确标注
    "raw_excess_return" 说明降级，benchmark_status 说明原因。
    """
    n = len(stock_returns)
    est = np.asarray(stock_returns[est_start:est_end], dtype=float)
    est = est[np.isfinite(est)]
    if len(est) < 20:
        return _empty_result("insufficient estimation data", alpha)
    mu = float(np.mean(est))
    excess = np.asarray(stock_returns, dtype=float) - mu
    sigma = float(np.std(est - mu, ddof=1)) if len(est) > 1 else 0.0

    cars = {}
    for (w_start, w_end) in car_windows:
        i, j = event_idx + w_start, event_idx + w_end
        if i < 0 or j >= n:
            continue
        cars[_car_key(w_start, w_end)] = float(np.sum(excess[i:j + 1]))

    L = _primary_window_days(car_windows)
    n_est = len(est)
    # 与市场模型路径同一套公式：σ²(L + L²/n_est)（无回归量偏离项）
    se = float(np.sqrt(max(sigma ** 2 * (L + (L * L) / n_est), 0.0)))
    primary_car = cars.get("car_1d", 0.0)
    t_stat = primary_car / se if se > 0 else 0.0
    df = n_est - 1
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df)) if df > 0 else 1.0
    impact = (abs(cars.get("car_1d", 0)) * 0.4 +
              abs(cars.get("car_3d", 0)) * 0.3 +
              abs(cars.get("car_5d", 0)) * 0.2)

    return {
        "car_1d": round(cars.get("car_1d", 0), 6),
        "car_3d": round(cars.get("car_3d", 0), 6),
        "car_5d": round(cars.get("car_5d", 0), 6),
        "car_pre1d": round(cars.get("car_-1d_-1d", 0), 6),
        "alpha": round(mu, 6),
        "beta_market": None,
        "beta_identified": False,
        "r_squared": None,            # ← 没有回归，不报拟合优度
        "sigma": round(sigma, 6),
        "car_se": round(se, 6),
        "se_formula": "sqrt(sigma^2*(L + L^2/n_est))",
        "n_est": n_est,
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "is_significant": bool(p_value < alpha),
        "alpha_threshold": round(float(alpha), 4),
        "direction": "UP" if primary_car > 0 else "DOWN",
        "impact_score": round(float(impact), 4),
        "confidence": round(float(_confidence(p_value, 0.0, alpha)), 4),
        "method": "raw_excess_return",
        "model": "mean_adjusted",
        "degraded": True,
        "benchmark_status": benchmark_status,
        "sector_status": sector_status,
        "note": ("benchmark missing / length-mismatch / zero-variance: market model NOT "
                 "fitted; CAR is a mean-adjusted excess return; r_squared not reported"),
        "status": "ok",
    }


def market_model_car(stock_returns, market_returns, event_idx,
                     est_window=(-60, -6), car_windows=[(0, 1), (0, 3), (0, 5)],
                     sector_returns=None, alpha: float = 0.05,
                     benchmark_status=None):
    """
    Compute CAR using market model or multi-factor regression.

    Args:
        stock_returns: array-like, daily stock returns
        market_returns: array-like, daily market/index returns（长度必须与
            stock_returns 一致；否则不跑 OLS，降级 raw_excess_return）
        event_idx: int, index of event day in the series
        est_window: (start_offset, end_offset) relative to event_idx
        car_windows: list of (start, end) tuples for CAR computation
        sector_returns: optional array-like, sector benchmark returns
        alpha: float, 显著性水平（默认 0.05），用于 is_significant 与 confidence
        benchmark_status: optional str, 调用方已知的基准状态（如 length_mismatch）

    Returns:
        dict with car values, t-stat, p-value, alpha(回归截距)/alpha_threshold(显著性阈值),
              betas, r_squared, car_se, se_formula, method
    """
    n = len(stock_returns)
    est_start = max(0, event_idx + est_window[0])
    est_end = max(0, event_idx + est_window[1])

    if est_end - est_start < 20:
        return _empty_result("insufficient estimation data", alpha)

    # Build factor matrix (store full arrays for CAR prediction)
    y = stock_returns[est_start:est_end]
    full_factors = []
    factor_names = []
    if market_returns is not None and len(market_returns) == n:
        full_factors.append(market_returns)
        factor_names.append("beta_market")

    # 板块基准：长度不匹配时旧代码静默丢弃 → 现在显式标注 sector_status
    sector_status = "not_provided"
    if sector_returns is not None:
        if len(sector_returns) == n:
            full_factors.append(sector_returns)
            factor_names.append("beta_sector")
            sector_status = "ok"
        else:
            sector_status = "dropped_length_mismatch"

    # 无有效基准 → 不拿零列硬跑 OLS（P1 修复点）
    if not full_factors:
        return _raw_excess_car(stock_returns, event_idx, est_start, est_end,
                               car_windows, alpha,
                               benchmark_status or "missing", sector_status)

    X_list = [np.ones(len(y))]
    for f in full_factors:
        X_list.append(np.asarray(f[est_start:est_end], dtype=float))  # slice for estimation

    X = np.column_stack(X_list)

    # Remove NaN
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[mask], y[mask]

    if len(X) < 20:
        return _empty_result("insufficient clean data after NaN removal", alpha)

    # ── 零列/常数列检查（P1 修复点）─────────────────────────────
    # lstsq 对含零列的 X 不报错，会静默退化成常数均值模型（beta=0、alpha=mean）
    # 却仍返回 r_squared/is_significant，把常数模型伪装成市场模型。
    factor_std = np.std(X[:, 1:], axis=0)
    if np.any(~np.isfinite(factor_std)) or np.any(factor_std <= 1e-12):
        return _raw_excess_car(stock_returns, event_idx, est_start, est_end,
                               car_windows, alpha,
                               benchmark_status or "zero_variance_regressor",
                               sector_status)

    try:
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return _empty_result("singular matrix in regression", alpha)

    if np.linalg.matrix_rank(X) < X.shape[1]:
        return _empty_result("rank-deficient factor matrix (collinear regressors)", alpha)

    y_pred = X @ beta_hat
    residuals = y - y_pred
    sigma = np.std(residuals, ddof=len(beta_hat))
    r_squared = 1 - np.var(residuals) / np.var(y) if np.var(y) > 0 else 0

    # Compute CAR for each window
    cars = {}
    car_idx = {}
    for (w_start, w_end) in car_windows:
        w_start_idx = event_idx + w_start
        w_end_idx = event_idx + w_end
        if w_start_idx < 0 or w_end_idx >= n:
            continue

        ar_sum = 0.0
        count = 0
        for i in range(w_start_idx, w_end_idx + 1):
            # Predicted return using factors
            predicted = beta_hat[0]  # alpha
            for j, f in enumerate(full_factors):
                predicted += beta_hat[j + 1] * f[i]
            ar = stock_returns[i] - predicted
            ar_sum += ar
            count += 1

        key = _car_key(w_start, w_end)
        cars[key] = ar_sum if count > 0 else 0.0
        car_idx[key] = (w_start_idx, w_end_idx)

    # t-test on CAR_1d
    # 标准误含估计窗参数不确定度（P1 修复点）：SE = sqrt(L·σ² + L²/n_est·σ² + …)
    # 旧代码只有 σ·√L，漏掉估计窗项 → t 偏大。
    primary_car = cars.get("car_1d", 0)
    car_se, se_source, se_quad = _cumulative_se(
        sigma, X, full_factors, car_idx.get("car_1d"), n,
        _primary_window_days(car_windows))
    t_stat = primary_car / car_se if car_se > 0 else 0.0
    df = len(residuals) - len(beta_hat)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df)) if df > 0 else 1.0

    direction = "UP" if primary_car > 0 else "DOWN"

    # Impact score
    impact = (abs(cars.get("car_1d", 0)) * 0.4 +
              abs(cars.get("car_3d", 0)) * 0.3 +
              abs(cars.get("car_5d", 0)) * 0.2)

    result = {
        "car_1d": round(cars.get("car_1d", 0), 6),
        "car_3d": round(cars.get("car_3d", 0), 6),
        "car_5d": round(cars.get("car_5d", 0), 6),
        "car_pre1d": round(cars.get("car_-1d_-1d", 0), 6),
        "alpha": round(float(beta_hat[0]), 6),
        "r_squared": round(float(r_squared), 4),
        "sigma": round(float(sigma), 6),
        "car_se": round(float(car_se), 6),
        "se_formula": "sqrt(sigma^2 * (L + 1'X_win(X'X)^-1 X_win'1))",
        "se_terms": {"L": float(_primary_window_days(car_windows)),
                     "quad": round(float(se_quad), 6), "source": se_source},
        "n_est": int(len(residuals)),
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "is_significant": bool(p_value < alpha),
        "alpha_threshold": round(float(alpha), 4),
        "direction": direction,
        "impact_score": round(float(impact), 4),
        "confidence": round(float(_confidence(p_value, r_squared, alpha)), 4),
        "method": "market_model",
        "model": "market_model",
        "degraded": False,
        "beta_identified": True,
        "benchmark_status": benchmark_status or "ok",
        "sector_status": sector_status,
        "status": "ok",
    }

    # Add factor betas
    for i, name in enumerate(factor_names):
        result[name] = round(float(beta_hat[i + 1]), 6)

    return result


def _car_key(w_start, w_end):
    if w_start < 0:
        return f"car_{w_start}d_{w_end}d"
    return f"car_{w_end}d"


def _primary_window_days(windows, primary_key="car_1d"):
    """返回 primary CAR 窗口的实际天数（如 (0,1) → 2 天）。"""
    for ws, we in windows:
        if _car_key(ws, we) == primary_key:
            return we - ws + 1
    return 1


def _empty_result(reason, alpha: float = 0.05):
    return {
        "car_1d": 0, "car_3d": 0, "car_5d": 0, "car_pre1d": 0,
        "alpha": 0, "r_squared": 0, "sigma": 0,
        "car_se": 0, "se_formula": None, "n_est": 0,
        "t_stat": 0, "p_value": 1.0, "is_significant": False,
        "alpha_threshold": round(float(alpha), 4),
        "direction": "N/A", "impact_score": 0, "confidence": 0,
        "method": None, "model": None, "degraded": None,
        "benchmark_status": None, "sector_status": None,
        "status": reason,
    }


def _confidence(p_value, r_squared, alpha: float = 0.05):
    sig_score = max(0, 1 - p_value / alpha) if alpha > 0 else 0.0
    fit_score = max(0, min(1, r_squared / 0.3))
    return sig_score * 0.7 + fit_score * 0.3


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    n = 200
    market = np.random.normal(0.0005, 0.01, n)
    # stock with beta=1.2, event day jump of +5%
    stock = 0.0003 + 1.2 * market + np.random.normal(0, 0.015, n)
    stock[100] += 0.05  # event jump
    
    event_idx = 100
    result = market_model_car(
        stock_returns=stock,
        market_returns=market,
        event_idx=event_idx,
    )
    
    print("Event Study Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"\n  ✓ alpha={result['alpha']:.4f} (should be ~0.0003)")
    print(f"  ✓ beta={result['beta']:.4f} (should be ~1.2)")
