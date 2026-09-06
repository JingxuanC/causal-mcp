#!/usr/bin/env python3
"""pcmci_discover.py — 多变量时序因果发现（带滞后边）。

fallback（默认，纯 numpy/scipy）：lagged 条件偏相关筛选 —— 对每个有序对
(i→j) 和每个滞后 τ，检验 corr(X_i(t-τ), X_j(t)) 在条件集 Z = {X_j 自身滞后
1..max_lag} ∪ {其他所有变量的滞后 1..max_lag} 下的偏相关（残差化 + Fisher z
检验），思想对齐 PCMCI 的 MCI 检验但条件集取全集（保守版）。
可选增强：装了 tigramite 时走真 PCMCI（Runge et al., Science Advances 2019,
ParCorr 条件独立检验），惰性 import，失败自动降级。method 字段标注实际实现。
"""

from __future__ import annotations

import numpy as np


def _partial_corr_pvalue(a: np.ndarray, b: np.ndarray, Z: np.ndarray) -> tuple[float, float]:
    """a,b 给定 Z（列矩阵，可能为空）的偏相关 + Fisher z 双侧 p 值。"""
    from scipy import stats
    if Z.size:
        Za = np.column_stack([np.ones(len(a)), Z])
        ra = a - Za @ np.linalg.lstsq(Za, a, rcond=None)[0]
        rb = b - Za @ np.linalg.lstsq(Za, b, rcond=None)[0]
    else:
        ra, rb = a - a.mean(), b - b.mean()
    sa, sb = np.std(ra), np.std(rb)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0, 1.0
    r = float(np.dot(ra, rb) / (len(a) * sa * sb))
    r = max(min(r, 0.999999), -0.999999)
    n_eff = len(a) - Z.shape[1] - 3
    if n_eff <= 0:
        return r, 1.0
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(n_eff)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return r, float(p)


def _granger_fallback(arr: np.ndarray, columns: list, max_lag: int, alpha: float) -> list:
    n, k = arr.shape
    # 标准化，常数列置零方差保护
    std = arr.std(axis=0)
    arr = (arr - arr.mean(axis=0)) / np.where(std < 1e-12, 1.0, std)
    edges = []
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            for tau in range(1, max_lag + 1):
                t0 = max_lag
                a = arr[t0 - tau:n - tau, i]
                b = arr[t0:n, j]
                # 条件集：j 的自身滞后 + 其他变量（含 i 的其他滞后除外当前项）的全部滞后
                zcols = [arr[t0 - l:n - l, j] for l in range(1, max_lag + 1)]
                for m in range(k):
                    if m == j:
                        continue
                    for l in range(1, max_lag + 1):
                        if m == i and l == tau:
                            continue
                        zcols.append(arr[t0 - l:n - l, m])
                Z = np.column_stack(zcols) if zcols else np.empty((len(a), 0))
                r, p = _partial_corr_pvalue(a, b, Z)
                if p < alpha:
                    edges.append({"from": columns[i], "to": columns[j], "lag": int(tau),
                                  "strength": round(r, 4), "p_value": round(p, 6)})
    edges.sort(key=lambda e: e["p_value"])
    return edges


def _tigramite_pcmci(arr: np.ndarray, columns: list, max_lag: int, alpha: float) -> list:
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    dataframe = pp.DataFrame(arr, var_names=columns)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(verbosity=0), verbosity=0)
    res = pcmci.run_pcmci(tau_min=1, tau_max=max_lag, pc_alpha=alpha)
    p_mat, val_mat = res["p_matrix"], res["val_matrix"]
    edges = []
    # tigramite 矩阵约定：p_matrix[i, j, tau] = 变量 i 在滞后 tau → 变量 j
    for i in range(len(columns)):
        for j in range(len(columns)):
            if i == j:
                continue
            for tau in range(1, max_lag + 1):
                if p_mat[i, j, tau] < alpha:
                    edges.append({"from": columns[i], "to": columns[j], "lag": int(tau),
                                  "strength": round(float(val_mat[i, j, tau]), 4),
                                  "p_value": round(float(p_mat[i, j, tau]), 6)})
    edges.sort(key=lambda e: e["p_value"])
    return edges


def pcmci_discover(data: dict, max_lag: int = 3, alpha: float = 0.05) -> dict:
    columns = list(data.get("columns", []))
    rows = data.get("rows", [])
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(columns) or len(columns) < 2:
        return {"error": "data must be {columns: [...], rows: [[...]]} with >=2 columns"}
    n, k = arr.shape
    if n < max_lag * (k + 2) + 10:
        return {"error": f"insufficient rows ({n}) for k={k}, max_lag={max_lag}"}

    try:
        edges = _tigramite_pcmci(arr, columns, max_lag, alpha)
        return {"edges": edges, "method": "pcmci", "n": int(n), "max_lag": int(max_lag),
                "alpha": alpha}
    except ImportError:
        pass
    except Exception:
        pass  # 数值失败同样降级，保证可用性

    edges = _granger_fallback(arr, columns, max_lag, alpha)
    return {"edges": edges, "method": "granger_fallback", "n": int(n),
            "max_lag": int(max_lag), "alpha": alpha}
