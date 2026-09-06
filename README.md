# Causal MCP

A 股因果分析工具集的独立 MCP（Model Context Protocol）服务。从
[Athena](https://github.com/JingxuanC/Athena) 的 py-sidecar 中抽取
causal 域，让任何 MCP 客户端（Claude Desktop、Kimi Code、Cursor、自研
Agent）都能直接做「事件研究 → 反事实验证 → 因果图学习」的完整因果推断链路。

与 [causal-memory.com](https://causal-memory.com) 公网服务的关系：本项目是
其同源方法论的本地/自托管版本，数据不出本机，无需账号即可使用；
公网服务提供托管的多租户 + license 额度体系，本项目也内置了同一套
`mcp_gateway.py` 鉴权/额度模块供私有化部署选用。

## 工具清单（4 个）

| 工具 | 说明 | 负载 |
|------|------|------|
| `event_study` | 单事件研究：市场模型 OLS（可选板块因子）估计正常收益，事件窗口 CAR (0,1)/(0,3)/(0,5) + t 检验 + 显著性/方向/impact_score | 轻（同步） |
| `event_study_batch` | 批量事件研究：多事件一次提交，子进程 stdin/stdout JSON 执行（与 Athena Go causal engine 调用方式一致） | 中（同步，120s 超时） |
| `refute` | 反事实验证三件套：placebo_treatment / random_common_cause / data_subset，输出逐项 robust/refuted + 总体评级 A-F。DoWhy 方法论的纯 numpy 自包含实现，**无需安装 dowhy** | 轻（同步） |
| `learn_graph` | 因果图结构学习：`pc`（偏相关 + Fisher z 检验，纯 numpy/scipy）或 `notears`（Ridge 简化版，未装 sklearn 时自动降级为相关性边） | 轻（同步） |

4 个工具均为秒级同步调用，直接返回结果。server 内置异步任务队列
（`mcp_gateway.py` JobQueue），后续接入重负载工具时无需改架构。

## 快速开始

```bash
pip install -r requirements.txt
python3 server.py --port 50057
```

验证：

```bash
curl http://127.0.0.1:50057/health
curl http://127.0.0.1:50057/tools
```

接入 MCP 客户端（以 Claude Desktop / Kimi Code 为例）：

```yaml
# mcp 配置
causal:
  url: http://127.0.0.1:50057/mcp
```

## 调用示例

### event_study — 单事件研究

```bash
curl -s http://127.0.0.1:50057/mcp -d '{
  "jsonrpc": "2.0", "id": 1, "method": "tools/call",
  "params": {"name": "event_study", "arguments": {
    "symbol": "002371",
    "event_date": "2026-04-17",
    "klines": [{"date": "2026-01-05", "close": 100.0}, ...],
    "benchmark": [{"date": "2026-01-05", "close": 3200.0}, ...]
  }}}'
```

返回（JSON 字符串）：

```json
{"car_1d": 0.032, "car_3d": 0.041, "car_5d": 0.038, "car_pre1d": -0.002,
 "alpha": 0.0003, "beta_market": 1.21, "r_squared": 0.42, "sigma": 0.014,
 "t_stat": 2.31, "p_value": 0.024, "is_significant": true,
 "direction": "UP", "impact_score": 0.0327, "confidence": 0.81, "status": "ok"}
```

约定：klines/benchmark 按日期升序；事件日要求之后至少还有 6 个交易日
（CAR (0,5) 窗口），不足时诚实返回 `status: "insufficient post-event data"`；
提供 `sector_benchmark` 时自动切换为市场+板块双因子模型（多返回 `beta_sector`）。

### event_study_batch — 批量

```json
{"events": [{"event_id": 1, "symbol": "002371", "event_date": "2026-04-17",
             "klines": [...], "benchmark": [...]},
            {"event_id": 2, "symbol": "sh600519", "event_date": "2026-04-18",
             "klines": [...], "benchmark": [...], "sector_benchmark": [...]}]}
```

返回 `{"results": [{event_id, symbol, event_date, car_1d, ...}, ...]}`，
单事件失败不影响其他事件。

### refute — 反事实验证

入参为 event_study 的产出（CAR 三窗口 + p_value）：

```json
{"chain": {"symbol": "sh600519", "car_1d": 0.019, "car_3d": 0.025,
           "car_5d": 0.031, "p_value": 0.042, "direction": "up"}}
```

返回：

```json
{"refutations": [
   {"method": "placebo_treatment", "result": "robust", "p_value": 0.03, "interpretation": "..."},
   {"method": "random_common_cause", "result": "robust", "interpretation": "..."},
   {"method": "data_subset", "result": "robust", "interpretation": "..."}],
 "passed": 3, "failed": 0, "grade": "A",
 "recommendation": "Strong evidence: all refutation tests passed"}
```

评级语义：A 全部通过 / B 多数通过 / C 喜忧参半 / D 多数失败 / F 疑似伪相关。
CAR 全窗口绝对值 < 0.005 时直接跳过（事件影响可忽略，无因果主张可验）。

### learn_graph — 因果图学习

```json
{"data": {"columns": ["sh600519", "sz000858", "sh600036"],
          "rows": [[1700.0, 185.0, 35.5], [1715.0, 187.0, 36.0], ...]},
 "method": "pc", "alpha": 0.05}
```

返回 `{"edges": [{"from", "to", "weight", "direction"}], "nodes": [...],
"metrics": {"n_edges", "sparsity", "avg_weight"}}`。至少 2 列 5 行数据。

## 依赖说明

必需：`numpy` `scipy` `pandas`（纯科学计算栈，无 qlib/redis/DB）。

可选（全部惰性导入，未装不影响服务启动）：

- `scikit-learn` — `learn_graph` 的 `method="notears"` 路径；未装自动降级
- `dowhy` / `networkx` — 完整 DoWhy 反事实管线。本项目的 `refute` 是
  DoWhy 三种 refuter 方法论的纯 numpy 自包含实现，不依赖 dowhy 包
- `statsmodels` — OLS 诊断参考；当前事件研究用 numpy lstsq + scipy t 检验

## 鉴权与额度（可选）

默认开放模式（本地/内网）。设置环境变量后强制 license key 鉴权：

```bash
export MCP_LICENSE_FILE=/path/to/licenses.json
python3 server.py --port 50057
# 客户端请求头：X-License-Key: <key>
```

license JSON 格式与额度语义见 `mcp_gateway.py` docstring（与
[factor-miner-mcp](https://github.com/JingxuanC/factor-miner-mcp) 相同）。
`GET /quota` 查余量，`GET /queue-stats` 看队列。

## 端点一览

```
GET  /health        健康检查
GET  /tools         工具 JSON schema 列表
POST /mcp           MCP JSON-RPC（initialize / tools/list / tools/call）
GET  /jobs/<id>     异步任务状态/结果（预留给未来的重负载工具）
GET  /quota         license 额度余量（鉴权模式）
GET  /queue-stats   队列概况
```

## 方法学说明

- **事件研究**：估计窗 (-60,-6) 内做市场模型（或多因子）OLS，事件窗内
  预测正常收益，AR = 实际 - 预测，CAR 为窗口 AR 求和；t 检验标准误按
  primary 窗口实际天数缩放（σ·√N），不跨窗口求和
- **反事实验证**：CAR 三窗口反推日度 AR 序列后跑三种 refuter——
  安慰剂（随机事件日 100 次置换）、随机共因（50 次噪声扰动稳定性）、
  数据子集（50 次 70% bootstrap 方向一致性）
- **因果图**：PC 算法从全连接图出发，用偏相关 + Fisher z 检验按条件集
  大小（≤3）迭代删除条件独立边；NOTEARS 为 Ridge 回归简化版

## 致谢

- 本项目主体来自 [Athena](https://github.com/JingxuanC/Athena)
  （multi-agent 量化交易系统）的 py-sidecar causal 域
- `mcp_gateway.py` 与 [factor-miner-mcp](https://github.com/JingxuanC/factor-miner-mcp) 共用
- refuter 设计参考 [DoWhy](https://github.com/py-why/dowhy) 的
  refutation 方法论（MIT）

## License

MIT
