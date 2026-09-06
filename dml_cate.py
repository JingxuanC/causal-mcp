#!/usr/bin/env python3
"""dml_cate.py — 异质处理效应估计（哪类标的对事件/因子反应更强）。

fallback（默认，sklearn）：T-learner —— 对 treatment=1 / treatment=0 两组分别
拟合 GradientBoostingRegressor，同一样本上两模型预测差即 CATE；特征重要性取
两模型 feature_importances_ 均值。处理变量非二元时按中位数二分并在输出标注
treatment_binarized。
可选增强：装了 econml 时走 LinearDML（Chernozhukov et al. 2018, Econometrics
Journal 双重机器学习），惰性 import，失败自动降级。method 字段标注实际实现。
"""

from __future__ import annotations

import numpy as np


def _prepare(treatment: list, outcome: list, features: dict) -> tuple:
    names = sorted(features.keys())
    if not names:
        raise ValueError("features must be a non-empty {name: [...]} dict")
    T = np.asarray(treatment, dtype=float)
    Y = np.asarray(outcome, dtype=float)
    X = np.column_stack([np.asarray(features[f], dtype=float) for f in names])
    n = min(len(T), len(Y), len(X))
    return T[:n], Y[:n], X[:n], names, n


def _t_learner(T: np.ndarray, Y: np.ndarray, X: np.ndarray, names: list) -> dict:
    from sklearn.ensemble import GradientBoostingRegressor
    m1 = GradientBoostingRegressor(random_state=42)
    m0 = GradientBoostingRegressor(random_state=42)
    m1.fit(X[T == 1], Y[T == 1])
    m0.fit(X[T == 0], Y[T == 0])
    cate = m1.predict(X) - m0.predict(X)
    imp = (m1.feature_importances_ + m0.feature_importances_) / 2.0
    return {"cate": cate, "importance": imp, "method": "t_learner"}


def _dml(T: np.ndarray, Y: np.ndarray, X: np.ndarray, names: list,
         binary: bool) -> dict:
    from econml.dml import LinearDML
    est = LinearDML(discrete_treatment=binary, random_state=42)
    est.fit(Y, T, X=X)
    cate = np.asarray(est.effect(X)).ravel()
    coef = np.asarray(est.coef_).ravel()[:len(names)]
    imp = np.abs(coef)
    if imp.sum() > 0:
        imp = imp / imp.sum()
    return {"cate": cate, "importance": imp, "method": "dml"}


def dml_cate(treatment: list, outcome: list, features: dict) -> dict:
    T, Y, X, names, n = _prepare(treatment, outcome, features)

    uniq = np.unique(T)
    binarized = False
    if len(uniq) != 2:
        med = float(np.median(T))
        T = (T > med).astype(float)
        binarized = True
    n1, n0 = int(np.sum(T == 1)), int(np.sum(T == 0))
    if min(n1, n0) < 20:
        return {"error": f"insufficient treated/control samples (treated={n1}, control={n0}, need >=20 each)"}

    try:
        res = _dml(T, Y, X, names, binary=not binarized)
    except ImportError:
        res = _t_learner(T, Y, X, names)
    except Exception:
        res = _t_learner(T, Y, X, names)

    cate = res["cate"]
    q90, q10 = np.quantile(cate, 0.9), np.quantile(cate, 0.1)
    importance = sorted(
        ({"feature": f, "importance": round(float(w), 4)} for f, w in zip(names, res["importance"])),
        key=lambda d: -d["importance"])
    return {
        "cate_summary": {
            "mean": round(float(cate.mean()), 6),
            "std": round(float(cate.std()), 6),
            "top_decile_mean": round(float(cate[cate >= q90].mean()), 6),
            "bottom_decile_mean": round(float(cate[cate <= q10].mean()), 6),
        },
        "feature_importance": importance,
        "method": res["method"],
        "n": int(n),
        "n_treated": n1,
        "n_control": n0,
        "treatment_binarized": binarized,
    }
