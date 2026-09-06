"""tools.py — A股因果分析 MCP 的工具注册表（8 个工具）。

从 Athena py-sidecar 抽取 causal 域子集：
- event_study / event_study_batch：事件研究（市场模型 OLS + CAR + t 检验）
- refute：反事实验证（placebo / random common cause / data subset 三件套，
  DoWhy 方法论的纯 numpy 自包含实现，不依赖 dowhy 包）
- learn_graph：因果图结构学习（PC 算法纯 numpy/scipy；NOTEARS 路径
  惰性 import sklearn，未装时自动降级为相关性边）
- causal_impact：事件反事实影响评估（默认 OLS 对冲回归纯 numpy；装了
  pycausalimpact 时惰性 import 走真 BSTS，method 字段标注实际实现）
- granger_te：两序列 lead-lag（statsmodels Granger F 检验 + 纯 numpy
  传递熵，Schreiber 2000）
- pcmci_discover：多变量时序因果发现（默认纯 numpy lagged 条件偏相关
  筛选；装了 tigramite 时走真 PCMCI，Runge et al. 2019）
- dml_cate：异质处理效应（默认 sklearn T-learner；装了 econml 时走
  LinearDML，Chernozhukov et al. 2018）

重依赖全部惰性导入：未装 statsmodels/sklearn/tigramite/econml/pycausalimpact
也可启动服务；有 fallback 的工具自动降级并在 method 字段标注。
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


# ═══════════════════════════════════════════════════════════════
# 事件反事实影响评估（OLS 对冲 fallback / BSTS 增强）
# ═══════════════════════════════════════════════════════════════

@tool("causal_impact", "事件反事实影响评估（对标 Google CausalImpact 的 BSTS）："
      "用事件前窗口对基准收益做 OLS 对冲回归（市场模型），外推事件后反事实收益，"
      "实际-反事实残差 → 累计影响 + 经验 p 值（事件前残差同长度滚动窗口的双侧置换分位）"
      "+ 95% 区间。装了 pycausalimpact 时自动走真 BSTS（Brodersen et al. 2015，惰性 import，"
      "失败降级）。返回 JSON: {cumulative_impact, avg_daily_impact, p_value, ci_95, "
      "significant, method: 'ols_hedge'|'bsts', alpha, beta, r_squared, ...}。",
      {"symbol": {"type": "string", "description": "股票代码，如 002371 / sh600519"},
       "event_date": {"type": "string", "description": "事件日期 YYYY-MM-DD（该日起为事件后窗口）"},
       "klines": _KLINES_SCHEMA,
       "benchmark": {**_KLINES_SCHEMA,
                     "description": "市场基准（指数）日K线，与 klines 对齐；缺失按零收益处理"}},
      required=["symbol", "event_date", "klines"])
def causal_impact(symbol: str, event_date: str, klines: list,
                  benchmark: Optional[list] = None) -> str:
    try:
        from causal_impact import causal_impact as _impl
    except ImportError as e:
        return json.dumps({"error": f"causal_impact dependency missing: {e}",
                           "hint": "pip install numpy pandas"})
    return json.dumps(_impl(symbol, event_date, klines, benchmark or []), ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# 两序列 lead-lag 因果（Granger + 传递熵）
# ═══════════════════════════════════════════════════════════════

@tool("granger_te", "两序列 lead-lag 因果分析：statsmodels Granger 因果 F 检验"
      "（双向，逐滞后阶 p 值）+ 纯 numpy 传递熵（Schreiber 2000，分位数分箱，双向）。"
      "输入收益率序列。返回 JSON: {granger: {x_causes_y: {best_lag, p_value, p_by_lag}, "
      "y_causes_x: {...}}, transfer_entropy: {x_to_y, y_to_x}, lead_lag: "
      "'x_leads'|'y_leads'|'none'}。依赖 statsmodels（requirements 必需项）。",
      {"x": {"type": "array", "items": {"type": "number"},
             "description": "序列 X（如基准/领先者收益率，按时间升序）"},
       "y": {"type": "array", "items": {"type": "number"},
             "description": "序列 Y（与 X 等长）"},
       "max_lag": {"type": "integer", "default": 5,
                   "description": "最大滞后阶（默认 5）"}},
      required=["x", "y"])
def granger_te(x: list, y: list, max_lag: int = 5) -> str:
    try:
        from granger_te import granger_te as _impl
    except ImportError as e:
        return json.dumps({"error": f"granger_te dependency missing: {e}",
                           "hint": "pip install numpy scipy statsmodels"})
    return json.dumps(_impl(x, y, max_lag), ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# 多变量时序因果发现（lagged 偏相关 fallback / PCMCI 增强）
# ═══════════════════════════════════════════════════════════════

@tool("pcmci_discover", "多变量时序因果发现（带滞后边）：装了 tigramite 时走真 PCMCI"
      "（Runge et al., Science Advances 2019，ParCorr 条件独立检验，惰性 import）；"
      "未装时自动降级为纯 numpy 的 lagged 条件偏相关筛选（对每对变量、每个滞后做 "
      "MCI 风格偏相关 + Fisher z 检验，method='granger_fallback'）。"
      "返回 JSON: {edges: [{from, to, lag, strength, p_value}], method, n, max_lag, alpha}。",
      {"data": {"type": "object",
                "description": "表格数据: {columns: ['x1', 'x2', ...], rows: [[...], ...]}，"
                               "行按时间升序（收益率或标准化序列）",
                "properties": {
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "rows": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                },
                "required": ["columns", "rows"]},
       "max_lag": {"type": "integer", "default": 3, "description": "最大滞后阶（默认 3）"},
       "alpha": {"type": "number", "default": 0.05,
                 "description": "条件独立检验显著性水平（默认 0.05）"}},
      required=["data"])
def pcmci_discover(data: dict, max_lag: int = 3, alpha: float = 0.05) -> str:
    try:
        from pcmci_discover import pcmci_discover as _impl
    except ImportError as e:
        return json.dumps({"error": f"pcmci_discover dependency missing: {e}",
                           "hint": "pip install numpy scipy"})
    return json.dumps(_impl(data, max_lag, alpha), ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# 异质处理效应（T-learner fallback / DML 增强）
# ═══════════════════════════════════════════════════════════════

@tool("dml_cate", "异质处理效应估计（哪类标的对事件/因子反应更强）："
      "装了 econml 时走 LinearDML（Chernozhukov et al. 2018 双重机器学习，惰性 import）；"
      "未装时自动降级为 sklearn T-learner（处理/对照组各拟合 GradientBoostingRegressor，"
      "预测差即 CATE，method='t_learner'）。处理变量非二元时按中位数二分并标注。"
      "返回 JSON: {cate_summary: {mean, std, top_decile_mean, bottom_decile_mean}, "
      "feature_importance: [{feature, importance}], method: 't_learner'|'dml', n, ...}。",
      {"treatment": {"type": "array", "items": {"type": "number"},
                     "description": "处理变量（事件哑变量 0/1，或因子暴露——非二元自动按中位数二分）"},
       "outcome": {"type": "array", "items": {"type": "number"},
                   "description": "结果变量（如事件后前瞻收益）"},
       "features": {"type": "object",
                    "description": "特征面板: {feature_name: [数值序列], ...}，长度与 treatment 一致",
                    "additionalProperties": {"type": "array", "items": {"type": "number"}}}},
      required=["treatment", "outcome", "features"])
def dml_cate(treatment: list, outcome: list, features: dict) -> str:
    try:
        from dml_cate import dml_cate as _impl
    except ImportError as e:
        return json.dumps({"error": f"dml_cate dependency missing: {e}",
                           "hint": "pip install numpy scikit-learn"})
    return json.dumps(_impl(treatment, outcome, features), ensure_ascii=False)


EXTRA_SCHEMAS: dict = {}
