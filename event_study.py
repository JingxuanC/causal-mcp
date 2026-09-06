#!/usr/bin/env python3
"""
event_study.py — Financial Event Study using statsmodels OLS.
Supports single-factor (market model) and multi-factor (market + sector).
"""

import numpy as np
from scipy import stats


def market_model_car(stock_returns, market_returns, event_idx,
                     est_window=(-60, -6), car_windows=[(0, 1), (0, 3), (0, 5)],
                     sector_returns=None):
    """
    Compute CAR using market model or multi-factor regression.

    Args:
        stock_returns: array-like, daily stock returns
        market_returns: array-like, daily market/index returns
        event_idx: int, index of event day in the series
        est_window: (start_offset, end_offset) relative to event_idx
        car_windows: list of (start, end) tuples for CAR computation
        sector_returns: optional array-like, sector benchmark returns

    Returns:
        dict with car values, t-stat, p-value, alpha, betas, r_squared
    """
    n = len(stock_returns)
    est_start = max(0, event_idx + est_window[0])
    est_end = max(0, event_idx + est_window[1])

    if est_end - est_start < 20:
        return _empty_result("insufficient estimation data")

    # Build factor matrix (store full arrays for CAR prediction)
    y = stock_returns[est_start:est_end]
    full_factors = [market_returns]
    factor_names = ["beta_market"]

    if sector_returns is not None and len(sector_returns) == n:
        full_factors.append(sector_returns)
        factor_names.append("beta_sector")

    X_list = [np.ones(len(y))]
    for f in full_factors:
        X_list.append(f[est_start:est_end])  # slice for estimation

    X = np.column_stack(X_list)

    # Remove NaN
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[mask], y[mask]

    if len(X) < 20:
        return _empty_result("insufficient clean data after NaN removal")

    try:
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return _empty_result("singular matrix in regression")

    y_pred = X @ beta_hat
    residuals = y - y_pred
    sigma = np.std(residuals, ddof=len(beta_hat))
    r_squared = 1 - np.var(residuals) / np.var(y) if np.var(y) > 0 else 0

    # Compute CAR for each window
    cars = {}
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

    # t-test on CAR_1d
    # 标准误按 primary CAR 对应窗口的实际天数（σ·√窗口天数），
    # 不能把所有 CAR 窗口天数求和——否则 t 被压低，p 恒不显著
    primary_car = cars.get("car_1d", 0)
    car_se = sigma * np.sqrt(max(1, _primary_window_days(car_windows)))
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
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "is_significant": bool(p_value < 0.20),
        "direction": direction,
        "impact_score": round(float(impact), 4),
        "confidence": round(float(_confidence(p_value, r_squared)), 4),
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


def _empty_result(reason):
    return {
        "car_1d": 0, "car_3d": 0, "car_5d": 0, "car_pre1d": 0,
        "alpha": 0, "r_squared": 0, "sigma": 0,
        "t_stat": 0, "p_value": 1.0, "is_significant": False,
        "direction": "N/A", "impact_score": 0, "confidence": 0,
        "status": reason,
    }


def _confidence(p_value, r_squared):
    sig_score = max(0, 1 - p_value / 0.20)
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
