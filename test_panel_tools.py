"""causal 组形状容差验收：同一份数据用 5 种形状喂进去，结果必须一致。

2026-09-15 修复前：klines 只吃扁平 bar 数组、data 只吃 {columns,rows}、
features 只吃 {name:[...]}，形状错时抛 pandas 内部异常而不是可自诊断 JSON。
"""

import json
from datetime import date, timedelta

import numpy as np
import pytest

import tools

N = 260
SYMS = ["sh600519", "sz300750", "sh601318"]

rng = np.random.default_rng(20260915)
DATES = [(date(2025, 6, 2) + timedelta(days=i)).isoformat() for i in range(N)]
CLOSES = {
    s: list(100.0 + np.cumsum(rng.normal(0, 1.1, N)) + i * 3.0)
    for i, s in enumerate(SYMS)
}
# 让第二个标的有领先关系，便于 granger 有实际输出
CLOSES[SYMS[1]] = list(np.roll(CLOSES[SYMS[0]], 1) + rng.normal(0, 0.2, N))


def _bars(sym):
    return [
        {"date": d, "open": c * 0.999, "high": c * 1.012,
         "low": c * 0.988, "close": c, "volume": 1_000_000 + i}
        for i, (d, c) in enumerate(zip(DATES, CLOSES[sym]))
    ]


ONE = SYMS[0]
SINGLE_SHAPES = {
    "flat_bars": _bars(ONE),
    "columnar": {
        "columns": ["date", "open", "high", "low", "close"],
        "rows": [[b["date"], b["open"], b["high"], b["low"], b["close"]] for b in _bars(ONE)],
    },
    "keyed_bars": {ONE: _bars(ONE)},
    "wide_symbol_outer": {ONE: dict(zip(DATES, CLOSES[ONE]))},
    "long_frame": [
        {"date": d, "symbol": ONE, "close": c} for d, c in zip(DATES, CLOSES[ONE])
    ],
}

MULTI_SHAPES = {
    "keyed_bars": {s: _bars(s) for s in SYMS},
    "columnar": {
        "columns": ["date", "symbol", "close"],
        "rows": [[d, s, c] for s in SYMS for d, c in zip(DATES, CLOSES[s])],
    },
    "wide_symbol_outer": {s: dict(zip(DATES, CLOSES[s])) for s in SYMS},
    "long_frame": [
        {"date": d, "symbol": s, "close": c}
        for s in SYMS for d, c in zip(DATES, CLOSES[s])
    ],
}

EVENT_DATE = DATES[200]


def _j(s):
    return json.loads(s)


@pytest.mark.parametrize("name", sorted(SINGLE_SHAPES))
def test_event_study_shape_tolerant(name):
    ref = _j(tools.event_study(ONE, EVENT_DATE, SINGLE_SHAPES["flat_bars"]))
    assert "error" not in ref, ref
    got = _j(tools.event_study(ONE, EVENT_DATE, SINGLE_SHAPES[name]))
    assert "error" not in got, (name, got.get("error"))
    assert got == ref, name


@pytest.mark.parametrize("name", sorted(SINGLE_SHAPES))
def test_causal_impact_shape_tolerant(name):
    ref = _j(tools.causal_impact(ONE, EVENT_DATE, SINGLE_SHAPES["flat_bars"]))
    assert "error" not in ref, ref
    got = _j(tools.causal_impact(ONE, EVENT_DATE, SINGLE_SHAPES[name]))
    assert "error" not in got, (name, got.get("error"))
    assert got == ref, name


@pytest.mark.parametrize("name", sorted(MULTI_SHAPES))
def test_learn_graph_shape_tolerant(name):
    ref = _j(tools.learn_graph(MULTI_SHAPES["keyed_bars"], method="pc"))
    assert "error" not in ref, ref
    got = _j(tools.learn_graph(MULTI_SHAPES[name], method="pc"))
    assert "error" not in got, (name, got.get("error"))
    assert got["nodes"] == ref["nodes"], name
    assert got["edges"] == ref["edges"], name


@pytest.mark.parametrize("name", sorted(MULTI_SHAPES))
def test_pcmci_shape_tolerant(name):
    ref = _j(tools.pcmci_discover(MULTI_SHAPES["keyed_bars"]))
    assert "error" not in ref, ref
    got = _j(tools.pcmci_discover(MULTI_SHAPES[name]))
    assert "error" not in got, (name, got.get("error"))
    assert got["edges"] == ref["edges"], name


def test_learn_graph_still_accepts_native_matrix():
    """原生 {columns, rows}（无 date/close 列）必须继续可用 ——
    不能因为引入 as_dataframe 就把这种输入拒掉。"""
    m = {
        "columns": ["x1", "x2", "x3"],
        "rows": [[float(v) for v in r] for r in rng.normal(0, 1, (60, 3))],
    }
    r = _j(tools.learn_graph(m))
    assert "error" not in r, r
    assert set(r["nodes"]) == {"x1", "x2", "x3"}


def test_granger_accepts_plain_lists_and_panel():
    a = _j(tools.granger_te(list(CLOSES[SYMS[0]]), list(CLOSES[SYMS[1]])))
    assert "error" not in a, a
    b = _j(tools.granger_te({SYMS[0]: _bars(SYMS[0]), SYMS[1]: _bars(SYMS[1])}))
    assert "error" not in b, b
    c = _j(tools.granger_te(
        [{"date": d, "value": v} for d, v in zip(DATES, CLOSES[SYMS[0]])],
        [{"date": d, "value": v} for d, v in zip(DATES, CLOSES[SYMS[1]])],
    ))
    assert "error" not in c, c
    assert a["lead_lag"] == b["lead_lag"] == c["lead_lag"]


def test_dml_cate_feature_shapes():
    n = 80
    treatment = [0.0, 1.0] * (n // 2)
    outcome = [float(v) for v in rng.normal(0, 1, n)]
    native = {"f1": [float(v) for v in rng.normal(0, 1, n)],
              "f2": [float(v) for v in rng.normal(0, 1, n)]}
    a = _j(tools.dml_cate(treatment, outcome, native))
    assert "error" not in a, a
    col = {"columns": ["f1", "f2"], "rows": [[native["f1"][i], native["f2"][i]] for i in range(n)]}
    b = _j(tools.dml_cate(treatment, outcome, col))
    assert "error" not in b, b
    assert a["cate_summary"] == b["cate_summary"]
    rows = [{"date": DATES[i], "symbol": "s1", "value": native["f1"][i]} for i in range(n)]
    c = _j(tools.dml_cate(treatment, outcome, {"f1": [r["value"] for r in rows]}))
    assert "error" not in c, c


def test_bad_shape_returns_diagnosable_json():
    r = _j(tools.event_study(ONE, EVENT_DATE, {"nonsense": 1}))
    assert "error" in r
    assert "klines 无法解析" in r["error"]
    r2 = _j(tools.learn_graph("nonsense"))
    assert "error" in r2


def test_multi_series_rejected_for_single_series_tools():
    r = _j(tools.event_study(ONE, EVENT_DATE, MULTI_SHAPES["keyed_bars"]))
    assert "error" in r
    assert "单序列" in r["error"]
