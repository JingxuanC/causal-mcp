"""tools.py — A股因果分析 MCP 的工具注册表（4 个工具）。

从 Athena py-sidecar 抽取 causal 域子集：
- event_study / event_study_batch：事件研究（市场模型 OLS + CAR + t 检验）
- refute：反事实验证（placebo / random common cause / data subset 三件套，
  DoWhy 方法论的纯 numpy 自包含实现，不依赖 dowhy 包）
- learn_graph：因果图结构学习（PC 算法纯 numpy/scipy；NOTEARS 路径
  惰性 import sklearn，未装时自动降级为相关性边）

重依赖全部惰性导入：未装 sklearn 也可启动服务并正常调用全部工具。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent


# ── Toolkit interface ──
class ToolDef:
    def __init__(self, name: str, description: str, inputSchema: dict):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema

    def to_dict(self):
        return {"name": self.name, "description": self.description, "inputSchema": self.inputSchema}


TOOLS: dict[str, ToolDef] = {}
HANDLERS: dict[str, callable] = {}


def tool(name: str, description: str, properties: dict, required: Optional[list] = None):
    """Decorator to register a tool."""
    def deco(fn):
        TOOLS[name] = ToolDef(name, description, {
            "type": "object",
            "properties": properties,
            "required": required or list(properties.keys()),
        })
        HANDLERS[name] = fn
        return fn
    return deco


_KLINES_SCHEMA = {
    "type": "array",
    "description": "日K线（按日期升序）: [{date: 'YYYY-MM-DD', close: number}, ...]",
    "items": {"type": "object",
              "properties": {"date": {"type": "string"}, "close": {"type": "number"}},
              "required": ["date", "close"]},
}

_EVENT_PROPS = {
    "symbol": {"type": "string", "description": "股票代码，如 002371 / sh600519"},
    "event_date": {"type": "string", "description": "事件日期 YYYY-MM-DD（非交易日自动取最近交易日）"},
    "klines": _KLINES_SCHEMA,
    "benchmark": {**_KLINES_SCHEMA,
                  "description": "市场基准（指数）日K线，与 klines 等长；缺失/不等长时按零收益处理"},
    "sector_benchmark": {**_KLINES_SCHEMA,
                         "description": "可选：板块基准日K线，提供后启用多因子模型（市场+板块）"},
}

_EVENT_RESULT_DOC = (
    "返回 JSON: {car_1d, car_3d, car_5d, car_pre1d, alpha, beta_market[, beta_sector], "
    "r_squared, sigma, t_stat, p_value, is_significant, direction, impact_score, "
    "confidence, status}。status != 'ok' 时其余字段为零值占位。"
)


# ═══════════════════════════════════════════════════════════════
# 事件研究（市场模型 / 多因子 OLS）
# ═══════════════════════════════════════════════════════════════

@tool("event_study", "单事件研究：市场模型 OLS 估计正常收益，计算事件窗口 CAR "
      "(cumulative abnormal return) + t 检验。估计窗 (-60,-6)，CAR 窗口 (0,1)/(0,3)/(0,5)。"
      + _EVENT_RESULT_DOC,
      _EVENT_PROPS,
      required=["symbol", "event_date", "klines"])
def event_study(symbol: str, event_date: str, klines: list,
                benchmark: Optional[list] = None,
                sector_benchmark: Optional[list] = None) -> str:
    from event_study_batch import process_batch
    evt = {"event_id": 1, "symbol": symbol, "event_date": event_date,
           "klines": klines, "benchmark": benchmark or []}
    if sector_benchmark:
        evt["sector_benchmark"] = sector_benchmark
    result = process_batch({"events": [evt]})["results"][0]
    result.pop("event_id", None)
    return json.dumps(result, ensure_ascii=False)


@tool("event_study_batch", "批量事件研究：一次提交多个事件，子进程执行（stdin JSON → "
      "stdout JSON），与 Athena Go causal engine 的调用方式一致。"
      "每个事件返回字段同 event_study，另带 event_id/symbol/event_date 回显。",
      {"events": {"type": "array",
                  "description": "事件列表: [{event_id, symbol, event_date, klines, benchmark?, sector_benchmark?}, ...]",
                  "items": {"type": "object"}},
       "timeout": {"type": "integer", "description": "子进程超时秒数（默认 120）", "default": 120}},
      required=["events"])
def event_study_batch(events: list, timeout: int = 120) -> str:
    payload = json.dumps({"events": events}, ensure_ascii=False)
    try:
        proc = subprocess.run(
            [sys.executable, str(_HERE / "event_study_batch.py")],
            input=payload, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"batch event study timed out after {timeout}s"})
    if proc.returncode != 0:
        return json.dumps({"error": "batch event study failed",
                           "stderr": proc.stderr[-2000:]})
    return proc.stdout.strip()


# ═══════════════════════════════════════════════════════════════
# 反事实验证（DoWhy 方法论三件套，纯 numpy 自包含实现）
# ═══════════════════════════════════════════════════════════════

@tool("refute", "反事实验证：对一条事件研究因果链（含 CAR/p_value）跑 3 个 refuter —— "
      "placebo_treatment（安慰剂处理）、random_common_cause（随机共因）、"
      "data_subset（数据子集 bootstrap），输出逐项 robust/refuted 判定 + 总体评级 A-F。"
      "返回 JSON: {refutations: [...], passed, failed, grade, recommendation}。"
      "纯 numpy 实现，无需安装 dowhy。",
      {"chain": {"type": "object",
                 "description": "因果链: {event_id?, symbol?, event_title?, car_1d, car_3d, "
                                "car_5d, p_value?, direction?}（car_* 来自 event_study 结果）",
                 "properties": {
                     "event_id": {"type": "integer"},
                     "symbol": {"type": "string"},
                     "car_1d": {"type": "number"},
                     "car_3d": {"type": "number"},
                     "car_5d": {"type": "number"},
                     "p_value": {"type": "number"},
                     "direction": {"type": "string"},
                 },
                 "required": ["car_1d", "car_3d", "car_5d"]}},
      required=["chain"])
def refute(chain: dict) -> str:
    try:
        from causal_refuter import refute_causal_chain
    except ImportError as e:
        return json.dumps({"error": f"refute dependency missing: {e}",
                           "hint": "pip install numpy pandas"})
    return json.dumps(refute_causal_chain(chain), ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# 因果图结构学习（PC / NOTEARS）
# ═══════════════════════════════════════════════════════════════

@tool("learn_graph", "因果图结构学习：从多标的价格/因子矩阵自动发现因果边。"
      "method='pc'（默认）：PC 算法，偏相关 + Fisher z 检验迭代删边，纯 numpy/scipy；"
      "method='notears'：简化 NOTEARS（Ridge 回归识别父节点，未装 sklearn 时降级为相关性）。"
      "返回 JSON: {edges: [{from, to, weight, direction}], nodes, method, "
      "metrics: {n_edges, sparsity, avg_weight}}。",
      {"data": {"type": "object",
                "description": "表格数据: {columns: ['sh600519', ...], rows: [[1700, ...], ...], dates?: [...]}",
                "properties": {
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "rows": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                    "dates": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["columns", "rows"]},
       "method": {"type": "string", "enum": ["pc", "notears"], "default": "pc",
                  "description": "结构学习算法（默认 pc）"},
       "alpha": {"type": "number", "default": 0.05,
                 "description": "PC 算法条件独立性检验显著性水平（默认 0.05）"}},
      required=["data"])
def learn_graph(data: dict, method: str = "pc", alpha: float = 0.05) -> str:
    try:
        from causal_graph import learn_graph as _impl
    except ImportError as e:
        return json.dumps({"error": f"learn_graph dependency missing: {e}",
                           "hint": "pip install numpy scipy pandas"})
    return json.dumps(_impl({"data": data, "method": method, "alpha": alpha}),
                      ensure_ascii=False)


EXTRA_SCHEMAS: dict = {}
