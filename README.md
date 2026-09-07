# Causal MCP

A 股因果分析工具集的独立 MCP（Model Context Protocol）服务。从
[Athena](https://github.com/JingxuanC/Athena) 的 py-sidecar 中抽取
causal 域，让任何 MCP 客户端（Claude Desktop、Kimi Code、Cursor、自研
Agent）都能直接做「事件研究 → 反事实验证 → 因果图学习」的完整因果推断链路。

与 [causal-memory.com](https://causal-memory.com) 公网服务的关系：本项目是
其同源方法论的本地/自托管版本，数据不出本机，无需账号即可使用；
公网服务提供托管的多租户 + license 额度体系，本项目也内置了同一套
`mcp_gateway.py` 鉴权/额度模块供私有化部署选用。

## 工具清单（8 个）

| 工具 | 说明 | 负载 |
|------|------|------|
| `event_study` | 单事件研究：市场模型 OLS（可选板块因子）估计正常收益，事件窗口 CAR (0,1)/(0,3)/(0,5) + t 检验 + 显著性/方向/impact_score | 轻（同步） |
| `event_study_batch` | 批量事件研究：多事件一次提交，子进程 stdin/stdout JSON 执行（与 Athena Go causal engine 调用方式一致） | 中（同步，120s 超时） |
| `refute` | 反事实验证三件套：placebo_treatment / random_common_cause / data_subset，输出逐项 robust/refuted + 总体评级 A-F。DoWhy 方法论的纯 numpy 自包含实现，**无需安装 dowhy** | 轻（同步） |
| `learn_graph` | 因果图结构学习：`pc`（偏相关 + Fisher z 检验，纯 numpy/scipy）或 `notears`（Ridge 简化版，未装 sklearn 时自动降级为相关性边） | 轻（同步） |
| `causal_impact` | 事件反事实影响评估（对标 Google CausalImpact 的 BSTS）：默认纯 numpy OLS 对冲回归外推反事实 + 经验 p 值 + 95% 区间；装了 pycausalimpact 自动走真 BSTS | 轻（同步） |
| `granger_te` | 两序列 lead-lag 因果：statsmodels Granger F 检验（双向逐滞后）+ 纯 numpy 传递熵（分位数分箱，双向） | 轻（同步） |
| `pcmci_discover` | 多变量时序因果发现（带滞后边）：装了 tigramite 走真 PCMCI（ParCorr）；未装自动降级为纯 numpy lagged 条件偏相关筛选 | 轻（同步） |
| `dml_cate` | 异质处理效应（CATE）：装了 econml 走 LinearDML；未装自动降级为 sklearn T-learner（GBRT 双模型预测差）+ 特征重要性 | 轻（同步） |

8 个工具均为秒级同步调用，直接返回结果。server 内置异步任务队列
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

## Docker 部署

无需本地 Python 环境，一条命令起服务：

```bash
docker compose up -d        # 构建镜像 + 启动容器（首次构建约 2-4 分钟）
docker compose ps           # 查看状态
docker compose logs -f      # 跟踪日志
```

验证：

```bash
curl http://127.0.0.1:50057/health
curl http://127.0.0.1:50057/tools   # 应返回 8 个工具
```

license 鉴权（可选）：在 `docker-compose.yml` 中取消注释，把宿主机
`licenses.json` 挂进容器并设置 `MCP_LICENSE_FILE`：

```yaml
environment:
  MCP_LICENSE_FILE: /app/licenses/licenses.json
volumes:
  - ./licenses.json:/app/licenses/licenses.json:ro
```

可选重库（FULL 镜像）：默认镜像不含 `pycausalimpact` / `tigramite` /
`econml`，对应工具走内置 fallback（输出 `method` 字段标注）。需要真
BSTS / PCMCI / LinearDML 路径时：

```bash
docker compose build --build-arg FULL=true   # 或编辑 compose 中 args.FULL
docker compose up -d
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

### causal_impact — 事件反事实影响评估

```json
{"symbol": "002371", "event_date": "2026-04-17",
 "klines": [{"date": "2026-01-05", "close": 100.0}, ...],
 "benchmark": [{"date": "2026-01-05", "close": 3200.0}, ...]}
```

事件日前为估计窗、事件日起为影响窗。返回：

```json
{"cumulative_impact": 0.319, "avg_daily_impact": 0.029, "p_value": 0.0,
 "ci_95": [0.284, 0.367], "significant": true, "method": "bsts",
 "alpha": -0.0009, "beta": 1.13, "r_squared": 0.66, "n_pre": 59, "n_post": 11}
```

`method` 标注实际实现：`"bsts"`（装了 pycausalimpact，走 Google CausalImpact
同款贝叶斯结构时序）或 `"ols_hedge"`（fallback：事件前窗口 OLS 对冲回归
外推反事实，经验 p 值为事件前残差同长度滚动窗口累计的双侧置换分位）。

### granger_te — 两序列 lead-lag 因果

```json
{"x": [0.001, -0.002, ...], "y": [0.002, 0.001, ...], "max_lag": 5}
```

输入收益率序列。返回：

```json
{"granger": {"x_causes_y": {"best_lag": 2, "p_value": 0.0, "p_by_lag": {...}},
             "y_causes_x": {"best_lag": 1, "p_value": 0.72, "p_by_lag": {...}}},
 "transfer_entropy": {"x_to_y": 0.76, "y_to_x": 0.84},
 "lead_lag": "x_leads"}
```

Granger 用 statsmodels `grangercausalitytests` 双向 F 检验；传递熵为纯 numpy
分位数分箱实现。`lead_lag`：单向显著取该侧，双向显著取 p 更小侧，均不显著
时看 TE 相对强弱（>20% 差），否则 `"none"`。

### pcmci_discover — 多变量时序因果发现

```json
{"data": {"columns": ["X1", "X2", "X3"], "rows": [[...], ...]},
 "max_lag": 3, "alpha": 0.05}
```

返回 `{"edges": [{"from", "to", "lag", "strength", "p_value"}], "method": ...}`。
`method` 为 `"pcmci"`（装了 tigramite，ParCorr 条件独立检验）或
`"granger_fallback"`（纯 numpy：对每对变量、每个滞后做 MCI 风格条件偏相关
+ Fisher z 检验，条件集 = 目标自身滞后 + 其他变量全部滞后）。

### dml_cate — 异质处理效应

```json
{"treatment": [1, 0, 1, ...], "outcome": [0.03, -0.01, ...],
 "features": {"feature_A": [...], "feature_B": [...]}}
```

面板数据：treatment 为事件哑变量或因子暴露（非二元自动按中位数二分，
输出标注 `treatment_binarized`），outcome 如事件后前瞻收益，features 为
标的属性。返回：

```json
{"cate_summary": {"mean": 0.014, "std": 0.014, "top_decile_mean": 0.040,
                  "bottom_decile_mean": -0.010},
 "feature_importance": [{"feature": "feature_A", "importance": 0.95}, ...],
 "method": "dml", "n": 600, "n_treated": 293, "n_control": 307}
```

`method` 为 `"dml"`（装了 econml，LinearDML 双重机器学习）或 `"t_learner"`
（sklearn GradientBoosting 双模型预测差）。top/bottom 十分位 CATE 差异 +
特征重要性排序用于识别"哪类标的对处理反应更强"。

## 依赖说明

必需：`numpy` `scipy` `pandas` `statsmodels` `scikit-learn`
（纯科学计算栈，无 qlib/redis/DB）。

可选（全部惰性导入，未装不影响服务启动，有 fallback 的工具自动降级并在
返回的 `method` 字段标注实际实现）：

- `pycausalimpact` — `causal_impact` 的真 BSTS 路径（Google CausalImpact
  的 Python 移植）；未装走 OLS 对冲回归 fallback（`method="ols_hedge"`）
- `tigramite` — `pcmci_discover` 的真 PCMCI 路径；未装走纯 numpy lagged
  条件偏相关 fallback（`method="granger_fallback"`）
- `econml` — `dml_cate` 的 LinearDML 路径；未装走 sklearn T-learner
  fallback（`method="t_learner"`）
- `dowhy` / `networkx` — 完整 DoWhy 反事实管线。本项目的 `refute` 是
  DoWhy 三种 refuter 方法论的纯 numpy 自包含实现，不依赖 dowhy 包

## 方法学出处

- **事件研究 / OLS 对冲**：市场模型（Sharpe 1964）；CausalImpact fallback
  沿用同框架外推反事实
- **CausalImpact (BSTS)**：Brodersen et al., "Inferring causal impact using
  Bayesian structural time-series models", *Annals of Applied Statistics*,
  2015（Google）；Python 移植 [pycausalimpact](https://github.com/WillianFuks/tfcausalimpact)
- **Granger 因果**：Granger, "Investigating causal relations by econometric
  models and cross-spectral methods", *Econometrica*, 1969（statsmodels 实现）
- **传递熵**：Schreiber, "Measuring information transfer", *Physical Review
  Letters* 85(2), 2000（本项目为分位数分箱的纯 numpy 实现）
- **PCMCI**：Runge et al., "Detecting and quantifying causal associations in
  large nonlinear time series datasets", *Science Advances* 5(11), 2019
  （[tigramite](https://github.com/jakobrunge/tigramite) 实现；fallback 为
  同思想的条件偏相关保守版）
- **DML / CATE**：Chernozhukov et al., "Double/debiased machine learning for
  treatment and structural parameters", *Econometrics Journal* 21(1), 2018
  （[EconML](https://github.com/py-why/EconML) LinearDML；fallback 为
  sklearn T-learner）

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
GET  /metrics       Prometheus 指标（免鉴权）
```

## 可观察性 / Observability

`GET /metrics` 输出 Prometheus text exposition 格式（免鉴权，仅工具名级聚合）：

| 指标 | 类型 | 说明 |
|------|------|------|
| `mcp_tool_calls_total{tool,status}` | counter | 调用次数，status ∈ ok/error/rejected_license/rejected_quota/queued |
| `mcp_tool_latency_seconds_sum{tool}` / `mcp_tool_latency_seconds_count{tool}` | counter | 延迟累计/次数，相除得平均延迟 |
| `mcp_uptime_seconds` | gauge | 进程启动至今秒数 |
| `mcp_queue_depth` | gauge | 当前排队中的异步任务数 |
| `mcp_queue_jobs_total{status}` | counter | 已完成的异步任务数，status ∈ done/error |

scrape 配置示例：

```yaml
scrape_configs:
  - job_name: causal-mcp
    metrics_path: /metrics
    static_configs:
      - targets: ["127.0.0.1:50057"]
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
- **因果影响**：fallback 用事件前窗口对基准收益做 OLS 对冲回归，外推
  事件后反事实，经验 p 值为事件前残差同长度滚动窗口累计的双侧置换分位，
  95% 区间按 σ·√N·√(1+1/n_pre) 含外推不确定性；BSTS 路径直接取
  pycausalimpact 的 post_cum_effects 与 p_value
- **lead-lag**：Granger 双向 F 检验取逐滞后最小 p；传递熵按分位数 8 分箱
  离散化后计算条件概率比的对数和
- **时序因果发现**：fallback 对每对变量每个滞后做 MCI 风格条件偏相关
  （条件集 = 目标自身滞后 + 其他变量全部滞后）+ Fisher z；PCMCI 路径用
  tigramite ParCorr
- **异质效应**：T-learner 对处理/对照组各拟合 GradientBoosting，同一样本
  预测差为 CATE，特征重要性取两模型均值；DML 路径用 econml LinearDML，
  重要性取 |coef| 归一化

## 致谢

- 本项目主体来自 [Athena](https://github.com/JingxuanC/Athena)
  （multi-agent 量化交易系统）的 py-sidecar causal 域
- `mcp_gateway.py` 与 [factor-miner-mcp](https://github.com/JingxuanC/factor-miner-mcp) 共用
- refuter 设计参考 [DoWhy](https://github.com/py-why/dowhy) 的
  refutation 方法论（MIT）

## License

MIT
