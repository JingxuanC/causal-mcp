#!/usr/bin/env python3
"""
event_study_batch.py — Batch event study called from Go causal engine.
Takes JSON via stdin, outputs JSON to stdout.

Input JSON format:
{
  "events": [
    {"event_id": 1, "symbol": "002371", "event_date": "2026-04-17",
     "klines": [{"date":"2026-01-01","close":100},...],
     "benchmark": [{"date":"2026-01-01","close":3200},...]}
  ]
}

Output JSON format:
{
  "results": [
    {"event_id": 1, "symbol": "002371", "car_1d": 0.032, ...}
  ]
}
"""

import sys, json
from datetime import datetime
import numpy as np
from event_study import market_model_car, _empty_result


def process_batch(data):
    results = []
    for evt in data.get("events", []):
        sym = evt["symbol"]
        event_date = evt["event_date"]

        # Convert klines to returns
        closes = np.array([k["close"] for k in evt["klines"]], dtype=float)
        dates = [k["date"] for k in evt["klines"]]
        stock_ret = np.diff(np.log(closes))  # log returns
        # 收益序列只有 N-1 个元素：stock_ret[i] 是 dates[i] → dates[i+1] 的收益，
        # 即"落在 dates[i+1] 当天"的收益。所以日期序列必须右移一位（dates[1:]）
        # 才能与 stock_ret 逐位对齐（与 causal_impact.py::_returns 的写法一致）。
        # 直接用完整 dates 去索引 stock_ret 会让 event_idx 偏早一天，
        # 事件窗口整体落到事件日之前。
        ret_dates = dates[1:]

        # Market benchmark returns
        # benchmark 缺失/长度不匹配时**不再**填零跑 OLS（P1 修复点）：
        # 含零列的 lstsq 不报错，会静默退化成常数均值模型却仍报 r_squared。
        # 这里传 None 并带上状态，由 event_study 降级为 raw_excess_return 并标注。
        raw_bench = evt.get("benchmark") or []
        bench_closes = np.array([b["close"] for b in raw_bench], dtype=float)
        benchmark_status = "ok"
        market_ret = None
        if len(bench_closes) == 0:
            benchmark_status = "missing"
        elif len(bench_closes) != len(closes):
            benchmark_status = "length_mismatch"
        elif len(bench_closes) < 2:
            benchmark_status = "too_short"
        else:
            market_ret = np.diff(np.log(bench_closes))

        # Sector benchmark returns (if provided): 长度不匹配时明确标注，不再静默丢弃
        sector_ret = None
        sector_status = "not_provided"
        if evt.get("sector_benchmark"):
            sb = np.array([b["close"] for b in evt["sector_benchmark"]], dtype=float)
            if len(sb) == len(closes) and len(sb) >= 2:
                sector_ret = np.diff(np.log(sb))
                sector_status = "ok"
            else:
                sector_status = "dropped_length_mismatch"

        # 显著性水平（默认 0.05，可被单事件覆盖）
        alpha = float(evt.get("alpha", 0.05))

        # Find event index（在收益序列的日期轴 ret_dates 上定位）
        try:
            event_idx = ret_dates.index(event_date)
        except ValueError:
            # Find nearest trading date by calendar distance (YYYY-MM-DD strings sort lexically = chronologically)
            _evt_ord = datetime.strptime(event_date, "%Y-%m-%d").toordinal()
            event_idx = min(range(len(ret_dates)),
                            key=lambda i: abs(datetime.strptime(ret_dates[i], "%Y-%m-%d").toordinal() - _evt_ord))
        
        # event_idx 必须给 car_5d 窗口 (0,5) 留足 6 个交易日；
        # 后置数据不足时诚实返回 insufficient，不静默回移事件索引
        # （回移会把 CAR 算到事件之前的交易日上，结果全错）
        if event_idx > len(stock_ret) - 6:
            result = _empty_result("insufficient post-event data", alpha)
            result["event_id"] = evt["event_id"]
            result["symbol"] = sym
            result["event_date"] = event_date
            result["benchmark_status"] = benchmark_status
            result["sector_status"] = sector_status
            results.append(result)
            continue

        result = market_model_car(
            stock_returns=stock_ret,
            market_returns=market_ret,
            event_idx=event_idx,
            sector_returns=sector_ret,
            alpha=alpha,
            benchmark_status=benchmark_status,
        )
        # 批量层掌握的 sector 状态更权威（长度不匹配在此被拦下，未传入函数）
        if sector_status != "not_provided":
            result["sector_status"] = sector_status
        result["event_id"] = evt["event_id"]
        result["symbol"] = sym
        result["event_date"] = event_date
        results.append(result)

    return {"results": results}


if __name__ == "__main__":
    raw = sys.stdin.read()
    data = json.loads(raw)
    output = process_batch(data)
    print(json.dumps(output, ensure_ascii=False))
