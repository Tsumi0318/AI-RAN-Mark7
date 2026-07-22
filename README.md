# AI-RAN 语义校准卸载博弈实验

本仓库包含一个可复现的 AI-RAN 边缘节点卸载博弈实验。实验使用阿里巴巴公开的 GenTD26 匿名生产轨迹构造任务和资源状态，通过 DeepSeek 完成任务语义影响预测和中心 Game Master 协调，并由确定性 Python 公式核验中心代价。

实验验证的是：在本次固定数据、参数与仿真假设下，异步最佳响应能否达到经过全节点检查的纯策略纳什均衡（PSNE）。PSNE 不等于系统总代价的全局最优，也不代表系统已经能够投入生产环境。

## 1. 实验目标

实验回答以下问题：

1. Prompt 长度、推理步数、图片数量和 LoRA 等语义特征能否形成节点 Intent；
2. LLM 预测的计算与显存影响能否用于校准拥塞函数参数；
3. Game Master 协调下的异步最佳响应是否达到 PSNE；
4. 协调策略与全本地、全卸载、随机和贪心策略相比表现如何；
5. Memory Violation Rate、Token 消耗和真实 API 时延是多少。

## 2. 目录结构

```text
Mark7/
  00_原始数据/
    GenTD26/                          实际使用的公开匿名轨迹
    官方压缩包/                       官方压缩文件
  01_源码与说明/
    mark7_pdf_cost_experiment.py      主实验、参数拟合、实时 LLM 协调
    base_ai_ran_components.py         数据、Intent、DeepSeek 与博弈组件
    postprocess_mark7_full.py         基线、压力测试和公式审计
    plot_mark7_queue_fit.py           拥塞模型拟合图
    plot_mark7_full_results.py        其余四张验证图
  02_表格数据/                        参数、逐轮记录、反馈与结果
  03_结果图/
    PNG/ PDF/ SVG/ TIFF/              按图片格式分类
```

## 3. 数据来源与使用方式

数据来源：[Alibaba Cluster Trace - GenTD26](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI)。`data_manifest.csv` 记录文件大小、SHA-256、来源和使用状态。

| 数据文件 | 实验用途 |
|---|---|
| `lora_request_trace.csv` | Prompt 字符长度、Steps、图片数量、LoRA、执行时间和任务类型 |
| `queue_size_raw_anon.csv` | 构造等效拥塞代理 K |
| `queue_rt_raw_anon.csv` | 拟合网关排队时延 |
| `model_predict_data_anon.csv` | 估计单服务容器处理率 |
| `pipeline_update_latency_anon.csv` | 统计流水线更新和调度开销 |
| `pod_gpu_duty_cycle_anon.csv` | 估计同时出现的 GPU 服务容器数 c |
| `pod_gpu_memory_used_bytes_anon.csv` | 拟合显存指数障碍函数 |

主实验固定随机种子，从指定高峰小时的成功请求中抽取 200 条任务，前 100 条用于主博弈，因此 `N=100`。

Prompt 特征是数据集提供的字符长度，不是真实 tokenizer Token 数。数据集中没有可靠的图片分辨率字段，因此实验使用图片数量，不使用图片分辨率。

## 4. 策略与代价函数

节点 i 只有本地执行和卸载两种策略：

```math
s_i\in\{0,1\},\qquad
s_i=0\;\text{表示本地执行},\quad
s_i=1\;\text{表示卸载到中心池}
```

当前卸载节点数为：

```math
K(\mathbf{s})=\sum_{i=1}^{N}s_i
```

节点的本地与传输能耗代价为：

```math
E_i(s_i)=(1-s_i)e_{i,\mathrm{loc}}+s_i e_{i,\mathrm{tx}}
```

两种策略对应的个体代价为：

```math
C_i^{\mathrm{loc}}=e_{i,\mathrm{loc}}
```

```math
C_i^{\mathrm{off}}
=e_{i,\mathrm{tx}}+D_{\mathrm{comp}}(K)+M(K)
```

其中 `D_comp(K)` 是中心计算拥塞代价，`M(K)` 是显存压力惩罚。本实验不额外加入独立的中心计算代价项。

## 5. 节点参数

任务执行时间按样本中位数归一化：

```math
x_i=\frac{T_i}{T_{\mathrm{med}}}
```

本地代价为：

```math
e_{i,\mathrm{loc}}=1.8x_i+0.4
```

传输代价为：

```math
e_{i,\mathrm{tx}}
=0.15+0.15\min\left(\frac{L_i}{L_{95}},1.5\right)
```

其中 `T_i` 是任务执行时间，`T_med` 是样本执行时间中位数，`L_i` 是 Prompt 字符长度，`L_95` 是字符长度的 95 分位数。

系数 1.8、0.4、0.15 和 0.15 是显式仿真假设，不是 GenTD26 直接给出的物理参数。因此系统总代价是归一化指标，不能直接解释为焦耳、人民币或完整端到端时延。

结果表还记录两个辅助模拟能耗指标：

```math
E_i^{\mathrm{loc,sim}}
=P_{\mathrm{edge}}\times1.8T_i\times1000
```

```math
E_i^{\mathrm{tx,sim}}
=e_{\mathrm{radio}}
\left[1+\frac{2(L_i+L_i^-)}{1024}\right]
```

其中 `P_edge=10 W`、`e_radio=0.2 mJ/KB`、固定元数据为 1 KB、每字符按 2 字节估算。这些也是模拟量，不是设备实测能耗。

## 6. LLM 语义影响预测

每个节点的结构化 Intent 包括：

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

DeepSeek 语义解析器为每个任务返回：

- `q_i`：相对计算倍率；
- `m_i`：相对显存倍率；
- `risk_level`：风险等级；
- `semantic_warning`：语义风险提示。

为了保持所有节点共享同一个拥塞函数，本实验对 100 条主任务的预测取算术平均：

```math
\bar q=\frac{1}{N}\sum_{i=1}^{N}q_i
```

```math
\bar m=\frac{1}{N}\sum_{i=1}^{N}m_i
```

语义预测通过以下方式校准原有参数：

```math
\mu_{\mathrm{card,eff}}
=\frac{\mu_{\mathrm{card,data}}}{\bar q}
```

```math
\frac{v_{\mathrm{req,eff}}}{V_{\max}}
=\frac{\bar m}{K_{\mathrm{cap}}}
```

平均计算倍率越高，有效服务率越低；平均显存倍率越高，单个卸载任务的等效显存需求越大。语义预测只调整 `D_comp` 和 `M` 的参数，不作为第三个独立代价项。

逐任务预测保存在 `semantic_resource_predictions_reused.csv`，聚合参数保存在 `semantic_parameter_calibration.csv`。

## 7. 计算拥塞函数

计算拥塞代价为：

```math
D_{\mathrm{comp}}(K)
=D_{\mathrm{overhead}}
+\frac{1}{c\mu_{\mathrm{card,eff}}}
\frac{a}{1-\rho(K)}
```

```math
\rho(K)
=\frac{\lambda K}{c\mu_{\mathrm{card,eff}}}
```

参数含义：

| 参数 | 含义 | 来源 |
|---|---|---|
| `D_overhead` | 固定网关和调度代理开销 | 数据拟合 |
| `c` | 同时出现的服务容器数量中位数 | 数据派生，不等同于物理 GPU 卡数 |
| `mu_card,data` | 单服务容器基础处理率 | 推理时延数据派生 |
| `q_bar` | LLM 预测的平均计算倍率 | DeepSeek 语义预测 |
| `mu_card,eff` | 语义校准后的有效处理率 | `mu_card,data / q_bar` |
| `a` | 利用率压力缩放参数 | 数据拟合 |
| `lambda` | K 到利用率的映射系数 | 数据拟合 |

本次参数与拟合结果：

| 参数或指标 | 数值 |
|---|---:|
| `q_bar` | 1.235 |
| `m_bar` | 1.150 |
| `c` | 109 |
| `mu_card,data` | 0.044483 task/s |
| `mu_card,eff` | 0.036019 task/s |
| `c * mu_card,eff` | 3.926030 task/s |
| `D_overhead` | 0.212791 s |
| `a` | 2.5230e-6 |
| `lambda` | 0.039260 |
| R-squared | 0.02038 |
| MAE | 35.154 ms |
| RMSE | 52.764 ms |

拟合采用鲁棒对数残差最小二乘。`R-squared=0.02038` 表明该函数对当前网关时延散点的解释力很弱，不能将其表述为已经验证的一般拥塞规律或高精度时延预测器。

## 8. 显存障碍函数

显存压力惩罚为：

```math
M(K)
=\alpha\exp\left[
\beta\left(
K\frac{v_{\mathrm{req,eff}}}{V_{\max}}-1
\right)
\right]
```

参数关系为：

```math
V_{\max}=16\,\mathrm{GB},\qquad
K_{\mathrm{cap}}=80,\qquad
\frac{v_{\mathrm{req,eff}}}{V_{\max}}
=\frac{\bar m}{K_{\mathrm{cap}}}
```

`V_max=16 GB` 和 `K_cap=80` 是显式仿真假设。`alpha` 与 `beta` 根据匿名 VRAM 尾部波动拟合。

模拟显存负载比例为：

```math
V(\mathbf{s})
=K(\mathbf{s})\frac{v_{\mathrm{req,eff}}}{V_{\max}}
```

当 `V(s)>1` 时记为模拟内存违规。该指标是软件容量代理，不是物理 GPU OOM 实测。

## 9. 中心反馈与最佳响应

假设节点 i 暂时设为本地执行，其余节点共有 `K_-i` 个卸载任务。节点 i 改为卸载时，中心代价为：

```math
\Delta_i^{\mathrm{center}}
=D_{\mathrm{comp}}(K_{-i}+1)
+M(K_{-i}+1)
```

节点卸载总代价为：

```math
C_i^{\mathrm{off}}
=e_{i,\mathrm{tx}}+\Delta_i^{\mathrm{center}}
```

若卸载代价更低，节点选择卸载：

```math
e_{i,\mathrm{tx}}+\Delta_i^{\mathrm{center}}
<e_{i,\mathrm{loc}}
\quad\Longrightarrow\quad s_i\leftarrow1
```

否则节点选择本地执行：

```math
e_{i,\mathrm{tx}}+\Delta_i^{\mathrm{center}}
\ge e_{i,\mathrm{loc}}
\quad\Longrightarrow\quad s_i\leftarrow0
```

每轮随机选择一个节点更新。最大更新次数为 1500；连续 `N=100` 轮没有策略改变后，程序对全部节点检查是否还存在有利单边偏离，全部通过才标记为 PSNE。

## 10. Game Master 的作用

每次更新时，Game Master 接收当前 `K`、候选 `K+1`、当前与候选显存比例、任务 Intent 以及公式分项，返回：

```json
{
  "center_cost_increment": 1.23,
  "compute_impact": "...",
  "vram_impact": "...",
  "semantic_warning": "..."
}
```

LLM 不直接决定 `s_i`。Python 重新计算并核验中心代价必须满足：

```math
\Delta_i^{\mathrm{center}}
=D_{\mathrm{comp}}(K_{-i}+1)+M(K_{-i}+1)
```

节点再根据核验后的数值执行最佳响应。这样 LLM 负责语义理解和反馈，确定性公式负责数值约束与可复现性。

## 11. 势函数与 PSNE 审计

本实验使用势函数：

```math
\Phi(\mathbf{s})
=\sum_{i=1}^{N}E_i(s_i)
+\sum_{k=1}^{K(\mathbf{s})}
\left[D_{\mathrm{comp}}(k)+M(k)\right]
```

对任意节点单边策略翻转，程序检查：

```math
\Delta C_i=\Delta\Phi
```

200 次随机翻转审计的最大绝对误差为：

```math
\max\left|\Delta C_i-\Delta\Phi\right|
=5.28\times10^{-14}
```

PSNE 判定条件为：

```math
s_i^*\in\mathrm{BR}_i(\mathbf{s}_{-i}^*),
\qquad \forall i\in\mathcal{N}
```

该审计证明代码中的个体代价变化与势函数变化一致，但不证明公式具有物理真实性，也不证明收敛点是全局最优。

## 12. 验证输出

| 验证项目 | 输出 |
|---|---|
| 最终策略 `s*` | `model_equilibrium_strategies.csv` |
| 收敛与稳定性 | `model_convergence_traces.csv`、图 02 |
| 系统总代价与基线 | `algorithm_comparison.csv`、图 03 |
| 能耗与时延候选前沿 | `pareto_front.csv` |
| N=30 到 200 的 Memory Violation Rate | `memory_violation_sweep.csv`、图 04 |
| LLM Token 与真实 API 时延 | `llm_coordination_overhead.csv`、图 05 |
| 每轮中心反馈与语义警告 | `llm_feedback_events.csv` |
| 势函数恒等式审计 | `formula_audit.csv` |

对比基线包括 All-Local、All-Offload、Random `p=0.5` 和只根据 `e_i,tx < e_i,loc` 决策的 Greedy。

势函数曲线在抽到已处于最佳响应的节点时会保持不变，因此实际结果应表述为势函数不增，而不是每轮严格下降。

## 13. 实验结果

系统总代价按全部节点个体代价求和：

```math
C_{\mathrm{sys}}(\mathbf{s})
=\sum_{i=1}^{N}C_i(s_i,\mathbf{s}_{-i})
```

主实验结果：

| 指标 | 结果 |
|---|---:|
| 节点数 N | 100 |
| 最终卸载数 K* | 62 |
| 卸载比例 | 62% |
| 最佳响应更新次数 | 796 |
| 实际策略变化次数 | 71 |
| 最终势函数 | 111.3383 |
| 系统总归一化代价 | 208.0966 |
| 卸载任务的拥塞时延 | 212.792 ms/task |
| 模拟显存负载 | 14.26 GB / 16 GB |
| 主实验容量违规 | 0 |
| 全节点 PSNE 检查 | 通过 |

基线对比：

| 策略 | K | 系统总归一化代价 | 容量违规 |
|---|---:|---:|---:|
| LLM 协调最佳响应 | 62 | 208.10 | 0 |
| All-Local | 0 | 280.60 | 0 |
| All-Offload | 100 | 71857.53 | 1 |
| Greedy | 100 | 71857.53 | 1 |
| Random `p=0.5`，500 次均值 | 49.882 | 183.93 | 0 |

随机基线的平均系统总代价低于本次 PSNE。因此结果支持“协调策略在本次压力测试中避免容量违规并达到 PSNE”，但不支持“协调策略的系统总代价优于所有基线”。

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

API 时延约为秒级，因此当前结果不能证明协调开销已经满足实时控制要求。

## 14. 主要输出文件

| 文件 | 内容 |
|---|---|
| `selected_model_B_metrics.csv` | 拥塞函数参数与拟合质量 |
| `selected_model_B_fit_points.csv` | 每个拟合点的观测、预测与残差 |
| `semantic_intents.csv` | 200 条结构化任务 Intent |
| `semantic_resource_predictions_reused.csv` | 真实 LLM 语义影响预测记录 |
| `semantic_parameter_calibration.csv` | `q_bar`、`m_bar`、`mu_eff` 和 `v_req_eff` |
| `model_equilibrium_strategies.csv` | 100 个节点的最终策略 `s*` |
| `model_convergence_traces.csv` | 每轮 K、势函数、总代价和容量状态 |
| `model_game_metrics.csv` | 主实验汇总指标与 PSNE 检查 |
| `llm_feedback_events.csv` | 每轮 LLM 回复、Token、时延和公式核验 |
| `llm_coordination_overhead.csv` | 调用量、Token、平均时延和 P95 时延 |
| `algorithm_comparison.csv` | 主算法与四类基线 |
| `memory_violation_sweep.csv` | N=30 到 200 的容量压力测试 |
| `formula_audit.csv` | `Delta C_i = Delta Phi` 数值审计 |
| `assumptions_and_parameters.csv` | 数据派生参数、拟合参数和显式假设 |
| `run_summary.json` | 实时 LLM 实验总汇总 |

五张结果图分别以 PNG、PDF、SVG 和 TIFF 四种格式保存在 `03_结果图` 对应目录中。

## 15. 运行方法

安装依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

完整 LLM 实验：

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

## 16. 结论边界

- 当前拥塞函数的拟合解释力较弱，不能宣称得到了一般拥塞规律；
- `c=109` 是服务容器数量代理，不是确认的物理 GPU 卡数；
- Prompt 长度是字符数，不是真实 tokenizer Token 数；
- 16 GB、80 个等效槽位、本地代价和传输代价系数是显式假设；
- 语义倍率取均值是为了保留共享拥塞函数与精确势结构的建模选择；
- Memory Violation Rate 是软件容量代理，不是物理 OOM 实测；
- 收敛到 PSNE 不等于达到全局最优；
- 本实验是公开匿名轨迹驱动的机制验证，不是生产部署验证。
