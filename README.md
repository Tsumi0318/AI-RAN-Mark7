# AI-RAN 语义校准卸载博弈实验

本目录是在方案 B 拥塞模型基础上，对原 PDF 代价结构进行恢复后的完整实验。实验使用阿里巴巴公开的 GenTD26 匿名生产轨迹、已记录的真实 DeepSeek 语义预测，以及新的 DeepSeek Game Master 协调过程。

本实验验证的是：在固定数据、参数和仿真假设下，异步最佳响应是否收敛到一个经过检查的纯策略纳什均衡（PSNE）。PSNE 不等于系统总代价的全局最优，也不代表模型已经具备生产部署能力。

## 1. 本版本解决的问题

此前实现把任务计算影响写成独立的中心计算代价。该写法不属于 PDF 的原始代价结构。当前版本按以下原则修正：

1. 保持 PDF 的节点能耗、中心拥塞、显存惩罚和最佳响应结构不变；
2. DeepSeek 根据任务语义特征预测计算倍率和显存倍率；
3. 计算倍率只校准 `Dcomp` 的有效服务率，显存倍率只校准 `M` 的单任务显存参数；
4. Game Master 返回 `Dcomp + M` 的中心代价与语义警告；
5. 不再向节点卸载代价额外加入独立的 `C_compute`。

方案 B 是此前已经选定的拥塞函数更新，不是 PDF 原公式。除 `Dcomp` 的函数形式外，博弈参与者、二元策略、代价组成、势函数、异步最佳响应和停止条件均按 PDF 的逻辑实现。

## 2. 目录结构

```text
Mark7/
  00_原始数据/
    GenTD26/                       实际使用的公开匿名轨迹
    官方压缩包/                    官方压缩文件
  01_源码与说明/
    mark7_pdf_cost_experiment.py   主实验、方案 B 拟合、实时 LLM 协调
    base_ai_ran_components.py      数据读取、Intent、DeepSeek 与验证组件
    postprocess_mark7_full.py      基线、压力测试、公式审计
    plot_mark7_queue_fit.py        队列拟合图
    plot_mark7_full_results.py     其余四张结果图
  02_表格数据/                     参数、逐轮记录、反馈与结果
  03_结果图/
    PNG/ PDF/ SVG/ TIFF/           按图片格式分类
```

## 3. 数据来源

数据来源：[Alibaba Cluster Trace - GenTD26](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI)。`data_manifest.csv` 记录每个数据文件的大小、SHA-256、来源和使用状态。

| 数据文件 | 实验用途 |
|---|---|
| `lora_request_trace.csv` | 构造任务 Intent：Prompt 字符长度、Steps、图片数量、LoRA、执行时间 |
| `queue_size_raw_anon.csv` | 构造等效拥塞代理 K |
| `queue_rt_raw_anon.csv` | 拟合 Dcomp 的网关时延目标 |
| `model_predict_data_anon.csv` | 估计单服务容器处理率 |
| `pipeline_update_latency_anon.csv` | 记录调度器/流水线更新延迟统计 |
| `pod_gpu_duty_cycle_anon.csv` | 估计同时出现的 GPU 服务容器数量 c |
| `pod_gpu_memory_used_bytes_anon.csv` | 拟合指数显存障碍函数的形状 |

主实验固定随机种子，从指定高峰小时的成功请求中抽取 200 条任务，前 100 条用于主博弈，因此 `N=100`。Prompt 特征是数据集给出的字符长度，不是真实 tokenizer Token 数。数据集未提供可靠的图片分辨率，因此本实验不使用分辨率特征，只使用图片数量。

## 4. PDF 原始博弈结构

节点 i 的策略为：

```text
s_i = 0：本地执行
s_i = 1：卸载到中心
K(s) = 所有 s_i 的和
```

本地与传输能耗代价为：

```text
E_i(s_i) = (1 - s_i) * e_i,loc + s_i * e_i,tx
```

按照 PDF 的 Algorithm 1 与势函数证明，活动实现采用：

```text
C_i(local)   = e_i,loc
C_i(offload) = e_i,tx + Dcomp(K) + M(K)
```

当暂时把节点 i 设为本地、其余节点共有 K_-i 个卸载任务时：

```text
Delta_center = Dcomp(K_-i + 1) + M(K_-i + 1)
```

最佳响应规则：

```text
若 e_i,tx + Delta_center < e_i,loc，则 s_i = 1；
否则 s_i = 0。
```

PDF 公式 (1) 的排版会让 `M(K)` 是否乘以 `s_i` 存在表面歧义，但 Algorithm 1、单边偏离证明和势函数均把 `M(K)` 作为卸载中心代价处理。本实现遵循这三处相互一致的逻辑。

## 5. 节点代价参数

任务执行时间先按样本中位数归一化：

```text
x_i = T_i / median(T)
```

本地代价与传输代价：

```text
e_i,loc = 1.8 * x_i + 0.4

e_i,tx  = 0.15 + 0.15 * min(L_i / L_95, 1.5)
```

其中 `T_i` 是数据中的执行时间，`L_i` 是 Prompt 字符长度，`L_95` 是字符长度的 95 分位数。`1.8`、`0.4`、`0.15` 和 `0.15` 都是显式仿真假设，不是数据集直接给出的物理参数。总代价是归一化量，不能直接解释成焦耳、人民币或端到端真实时延。

## 6. LLM 语义预测如何进入原公式

每个任务的 Intent 包括：

```json
{
  "prompt_length_chars": 150,
  "steps": 50,
  "num_images": 1,
  "lora": true,
  "lora_count": 1,
  "observed_exec_time_seconds": 35.2,
  "local_energy_cost_normalized": 2.2,
  "tx_energy_cost_normalized": 0.3
}
```

DeepSeek 语义解析器返回：

```text
q_i：任务相对计算倍率
m_i：任务相对显存倍率
risk_level：风险等级
semantic_warning：语义风险提示
```

为了保留一个所有节点共享的 `Dcomp(K)` 和 `M(K)`，并继续满足 PDF 的精确势博弈结构，本实验对 100 条主任务预测取算术平均：

```text
q_bar = 100 个 q_i 的平均值
m_bar = 100 个 m_i 的平均值
```

然后校准原公式参数：

```text
mu_card_eff = mu_card_data / q_bar
v_req_eff / Vmax = m_bar / K_cap
```

含义是：平均计算倍率越高，有效服务率越低；平均显存倍率越高，单个卸载任务占用的等效显存越大。

这里没有新增第三个中心代价项。语义预测只改变 `Dcomp` 与 `M` 的参数。

逐任务预测保存在 `semantic_resource_predictions_reused.csv`，聚合结果保存在 `semantic_parameter_calibration.csv`。这些预测来自此前对同一批 200 条任务的真实 DeepSeek API 调用；任务没有变化，因此复用预测记录。新的博弈状态与中心反馈使用新的缓存重新调用 DeepSeek。

## 7. 方案 B 的 Dcomp

方案 B 是本实验相对 PDF 的唯一核心公式更新：

```text
Dcomp(K) = D_overhead
           + [1 / (c * mu_card_eff)] * a / [1 - rho(K)]

rho(K) = lambda * K / (c * mu_card_eff)
```

参数含义：

| 参数 | 含义 | 来源 |
|---|---|---|
| `D_overhead` | 固定网关/调度代理开销 | 数据拟合 |
| `c` | 同时出现的服务容器数量中位数 | 数据派生，不等同于确认的物理 GPU 数 |
| `mu_card_data` | 单服务容器基础处理率 | 推理时延数据派生 |
| `q_bar` | LLM 预测的平均计算倍率 | DeepSeek 语义预测 |
| `mu_card_eff` | 语义校准后的有效处理率 | `mu_card_data / q_bar` |
| `a` | 利用率压力缩放 | 数据拟合 |
| `lambda` | K 到利用率的映射系数 | 数据拟合 |

拟合采用鲁棒对数残差最小二乘。该模型的拟合质量必须由 `selected_model_B_metrics.csv` 中的 R-squared、MAE 和 RMSE 判断；即使方案 B 优于先前候选，也不能据此声称它已经准确描述一般真实网络。

本次实际校准与拟合结果：

| 参数或指标 | 数值 |
|---|---:|
| `q_bar` | 1.235 |
| `m_bar` | 1.150 |
| `c` | 109 |
| `mu_card_data` | 0.044483 task/s |
| `mu_card_eff` | 0.036019 task/s |
| `c * mu_card_eff` | 3.926030 task/s |
| `D_overhead` | 0.212791 s |
| `a` | 2.5230e-6 |
| `lambda` | 0.039260 |
| R-squared | 0.02038 |
| MAE | 35.154 ms |
| RMSE | 52.764 ms |

R-squared 接近 0，说明方案 B 对这批网关时延散点的解释力仍然很弱。该结果必须作为限制保留。

## 8. PDF 结构下的 M(K)

使用归一化显存容量后：

```text
M(K) = alpha * exp[ beta * (K * v_req_eff / Vmax - 1) ]
```

其中：

```text
Vmax = 16 GB                         显式假设
K_cap = 80                           显式等效槽位假设
v_req_eff / Vmax = m_bar / K_cap     LLM 语义校准
```

`alpha` 与 `beta` 从匿名 VRAM 尾部波动拟合。容量尺度和 16 GB 仍是仿真假设，所以 Memory Violation 是软件容量代理，不是物理 GPU OOM 实测。

## 9. Game Master 流程

每轮随机选择一个节点 i：

```text
1. 节点发送包含语义特征、e_i,loc 和 e_i,tx 的 Intent。
2. 程序根据当前 K 计算候选 K+1。
3. 已校准的 Dcomp 计算排队代价。
4. 已校准的 M 计算显存惩罚。
5. DeepSeek 返回中心代价、计算影响说明、显存影响说明和警告。
6. Python 核验中心代价必须等于 Dcomp + M。
7. 节点根据 e_i,tx + Delta_center 与 e_i,loc 比较后更新策略。
```

LLM 不直接决定 `s_i`，也不能修改公式核验值。这样既保留语义协调，又保证最佳响应的数值可复现。

最大更新次数为 1500。连续 `N=100` 轮没有策略变化后，程序再对全部节点检查是否存在有利单边偏离；只有全部通过才标记为 PSNE。

## 10. 势函数与数学审计

当前实现的势函数与 PDF 证明同构：

```text
Phi(s) = 所有节点的 E_i(s_i)
         + 从 k=1 到 K(s) 累加 [Dcomp(k) + M(k)]
```

程序随机生成 200 次单节点策略翻转，检查：

```text
Delta C_i = Delta Phi
```

结果保存在 `formula_audit.csv`。该审计只证明代码中的代价变化和势函数变化一致，不证明公式具备物理真实性，也不证明收敛点是全局最优。

## 11. PDF 第 3 节的全部验证输出

| PDF 验证项目 | 实现与输出 |
|---|---|
| 最终策略 s* | `model_equilibrium_strategies.csv` |
| 收敛与稳定性 | `model_convergence_traces.csv`、图 02 |
| 系统总代价与基线 | `algorithm_comparison.csv`、图 03 |
| 能耗-时延候选前沿 | `pareto_front.csv` |
| Memory Violation Rate，N=30 到 200 | `memory_violation_sweep.csv`、图 04 |
| LLM Token 与真实 API 时延 | `llm_coordination_overhead.csv`、图 05 |
| 每轮中心反馈与语义警告 | `llm_feedback_events.csv` |
| 精确势恒等式审计 | `formula_audit.csv` |

对比基线包括：All-Local、All-Offload、Random p=0.5 和只按 `e_i,tx < e_i,loc` 决策的 Greedy。

PDF 中“势函数每一轮严格下降”“N 到 3N 轮收敛”“违规率严格为 0”“总代价显著更低”属于预期结果，不是代码预设结论。最终是否满足必须以实际 CSV 为准。抽到已处于最佳响应的节点时，势函数会保持不变，因此实际曲线应表述为不增，而非每轮严格下降。

## 12. 本次实际结果

主实验：

| 指标 | 结果 |
|---|---:|
| 节点数 N | 100 |
| 最终卸载数 K* | 62 |
| 卸载比例 | 62% |
| 最佳响应更新次数 | 796 |
| 实际策略变化次数 | 71 |
| 最终势函数 | 111.3383 |
| 系统总归一化代价 | 208.0966 |
| 卸载任务的 Dcomp | 212.792 ms/task |
| 模拟显存负载 | 14.26 GB / 16 GB |
| 主实验容量违规 | 0 |
| 全节点 PSNE 检查 | 通过 |
| 200 次势函数恒等式审计最大误差 | 5.28e-14 |

基线对比：

| 策略 | K | 系统总归一化代价 | 容量违规 |
|---|---:|---:|---:|
| LLM 协调最佳响应 | 62 | 208.10 | 0 |
| All-Local | 0 | 280.60 | 0 |
| All-Offload | 100 | 71857.53 | 1 |
| Greedy | 100 | 71857.53 | 1 |
| Random p=0.5，500 次均值 | 49.882 | 183.93 | 0 |

随机基线的平均系统总代价低于本次 PSNE。因此，结果支持“协调策略在本次压力测试中避免容量违规并收敛到 PSNE”，但不支持“协调策略的系统总代价优于所有基线”。这正是 PSNE 与全局最优必须区分的原因。

实时 DeepSeek 协调开销：

| 指标 | 结果 |
|---|---:|
| 逻辑协调次数 | 796 |
| 新的真实 API 调用 | 248 |
| 缓存复用 | 548 |
| 记录 Tokens | 98,478 |
| 平均 API 时延 | 1397.05 ms |
| P95 API 时延 | 1861.96 ms |
| 请求模型 | `deepseek-chat` |
| 服务端返回模型 | `deepseek-v4-flash` |
| LLM 数值与公式不一致 | 0 |

API 时延约为秒级，不是 PDF 预期描述中的微秒级。因此当前结果不能用于证明协调开销已经满足 Non-RT RIC 的工程时延要求。

## 13. 主要输出文件

| 文件 | 内容 |
|---|---|
| `selected_model_B_metrics.csv` | 方案 B 参数与拟合质量 |
| `selected_model_B_fit_points.csv` | 每个拟合点的观测、预测与残差 |
| `semantic_intents.csv` | 200 条结构化任务 Intent |
| `semantic_resource_predictions_reused.csv` | 真实 LLM 语义资源预测记录 |
| `semantic_parameter_calibration.csv` | q_bar、m_bar、mu_eff 和 v_req_eff |
| `model_equilibrium_strategies.csv` | 100 个节点的最终策略 s* |
| `model_convergence_traces.csv` | 每轮 K、势函数、总代价和容量状态 |
| `model_game_metrics.csv` | 主实验汇总指标与 PSNE 检查 |
| `llm_feedback_events.csv` | 每轮 LLM 回复、Token、时延和公式核验 |
| `llm_coordination_overhead.csv` | 调用量、Token、平均时延和 P95 时延 |
| `algorithm_comparison.csv` | 主算法与四类基线 |
| `memory_violation_sweep.csv` | N=30 到 200 的容量压力测试 |
| `formula_audit.csv` | Delta C_i = Delta Phi 数值审计 |
| `assumptions_and_parameters.csv` | 数据派生参数、拟合参数和显式假设 |
| `run_summary.json` | 实时 LLM 实验总汇总 |

## 14. 运行方法

安装依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

完整实验：

```bash
export DEEPSEEK_API_KEY=YOUR_KEY
.venv/bin/python 01_源码与说明/mark7_pdf_cost_experiment.py --full
.venv/bin/python 01_源码与说明/postprocess_mark7_full.py
.venv/bin/python 01_源码与说明/plot_mark7_queue_fit.py
.venv/bin/python 01_源码与说明/plot_mark7_full_results.py
```

不调用 API 的公式与流程核验：

```bash
.venv/bin/python 01_源码与说明/mark7_pdf_cost_experiment.py --deterministic
.venv/bin/python 01_源码与说明/postprocess_mark7_full.py
```

API 密钥只从环境变量读取，不能写入代码、README、CSV、JSON 或缓存。

## 15. 结论边界

- 方案 B 的拟合质量有限时，不能宣称得到准确的一般拥塞规律；
- `c` 是服务容器代理，不是确认的物理 GPU 卡数；
- Prompt 长度是字符数，不是 tokenizer Token 数；
- 16 GB、80 个等效槽位、本地和传输代价系数是显式假设；
- 语义倍率取均值是为了保留共享拥塞函数与精确势结构的建模选择；
- Memory Violation Rate 是仿真容量代理；
- 收敛到 PSNE 不等于达到全局最优；
- 本实验是公开匿名轨迹驱动的机制验证，不是生产部署验证。
