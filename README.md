# 基于拥塞势博弈与大语言模型协同的 AI-RAN 计算卸载实验

本仓库实现一个可复现的 AI-RAN 二元计算卸载实验。完整运行从阿里巴巴公开的 GenTD26 匿名生产轨迹抽取固定的 200 条任务，重新调用 DeepSeek 生成每条任务的语义资源预测，再以其中前 100 条任务执行中心 Game Master 协调、异步最佳响应和 PSNE 检查。中心拥塞与显存公式分项由确定性 Python 提供并复算，DeepSeek 返回语义资源倍率、中心代价、影响说明和语义警告。因此，本实验验证的是“LLM 语义反馈 + Python 公式约束”的协调流程，不是由 LLM 独立推导中心代价。

本实验验证的是：在本次固定数据、参数和仿真假设下，异步最佳响应是否到达一个经过全节点检查的 PSNE。PSNE 不等于系统总代价的全局最优，也不代表该系统已经能够投入生产环境。

在阅读结果前需要明确三项边界：主实验总代价是归一化无量纲指标；`V_max=16 GB` 与 `K_cap=80` 是模拟器假设，不是 GenTD26 提供的硬件规格；Memory Violation Rate 是软件容量代理，不是真实 CUDA OOM。DeepSeek Game Master 的本次记录平均时延为 `1455.19 ms`、P95 为 `1933.32 ms`，当前结果只支持离线机制验证，不支持实时 RAN 部署结论。

## 1 系统模型与理论建模

### 1.1 网络场景与变量定义

系统由 1 个中心计算池和 `N` 个边缘节点组成，边缘节点集合为：

$$
\mathcal{N}=\{1,2,\ldots,N\}
$$

节点 `i` 的卸载策略为：

$$
s_i\in\{0,1\},\qquad
s_i=0\ \text{表示本地执行},\quad
s_i=1\ \text{表示卸载到中心池}
$$

全局策略组合与当前卸载节点数分别为：

$$
\mathbf{s}=(s_1,s_2,\ldots,s_N)
$$

$$
K(\mathbf{s})=\sum_{j=1}^{N}s_j
$$

主实验设置 `N=100`。程序使用固定随机种子 `20260717`，从指定高峰小时的成功请求中抽取 200 条任务，前 100 条用于主博弈，其余任务用于节点规模扩展实验。

#### 数据来源

数据来自 [Alibaba Cluster Trace - GenTD26](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI)。`data_manifest.csv` 记录文件大小、SHA-256、来源和使用状态。

| 数据文件 | 在实验中的用途 |
|---|---|
| `lora_request_trace.csv` | 构造 Prompt 字符长度、Steps、图片数量、LoRA、执行时间和任务类型 |
| `queue_size_raw_anon.csv` | 构造等效拥塞状态 `K` |
| `queue_rt_raw_anon.csv` | 拟合网关排队时延 |
| `model_predict_data_anon.csv` | 估计单服务容器处理率 |
| `pipeline_update_latency_anon.csv` | 统计流水线更新和调度开销 |
| `pod_gpu_duty_cycle_anon.csv` | 估计同时出现的 GPU 服务容器数量 `c` |
| `pod_gpu_memory_used_bytes_anon.csv` | 拟合显存指数障碍函数 |

Prompt 特征是数据集给出的字符长度，不是真实 tokenizer 生成的 Prompt Token 数。数据集没有可靠的图片分辨率字段，因此实验使用图片数量，不使用图片分辨率。

### 1.2 极简效用/代价函数

节点 `i` 的总代价由本地或传输代价、中心算力拥塞代价和显存压力惩罚组成：

$$
C_i(s_i,\mathbf{s}_{-i})
=E_i(s_i)
+s_i\left[D_{\mathrm{comp}}(K(\mathbf{s}))+M(K(\mathbf{s}))\right]
$$

该结构表示只有选择卸载的节点承担中心拥塞和显存压力代价。本实验没有增加独立的第三项任务计算代价；LLM 的语义预测只校准 `D_comp` 和 `M` 的参数。

#### 1. 能耗代价

理论模型中的本地与传输代价为：

$$
E_i(s_i)
=(1-s_i)e_{i,\mathrm{loc}}+s_i e_{i,\mathrm{tx}}
$$

因此两种策略的个体代价分别为：

$$
C_i^{\mathrm{loc}}=e_{i,\mathrm{loc}}
$$

$$
C_i^{\mathrm{off}}
=e_{i,\mathrm{tx}}+D_{\mathrm{comp}}(K)+M(K)
$$

由于公开轨迹没有直接提供统一的边缘能耗和无线传输能耗，Mark7 先把任务执行时间按样本中位数归一化：

$$
x_i=\frac{T_i}{T_{\mathrm{med}}}
$$

然后实例化本地代价与传输代价：

$$
e_{i,\mathrm{loc}}=1.8x_i+0.4
$$

$$
e_{i,\mathrm{tx}}
=0.15+0.15\min\left(\frac{L_i}{L_{95}},1.5\right)
$$

其中，`T_i` 是任务执行时间，`T_med` 是样本执行时间中位数，`L_i` 是 Prompt 字符长度，`L_95` 是字符长度的 95 分位数。系数 `1.8`、`0.4`、`0.15` 和 `0.15` 是显式仿真假设，不是 GenTD26 直接给出的物理参数。因此主博弈的总代价是归一化指标，不能解释为焦耳、人民币或完整端到端时延。

结果表另行记录两个辅助模拟能耗指标，它们不参与最佳响应决策：

$$
E_i^{\mathrm{loc,sim}}
=P_{\mathrm{edge}}\times1.8T_i\times1000
$$

$$
E_i^{\mathrm{tx,sim}}
=e_{\mathrm{radio}}
\left[1+\frac{2(L_i+L_i^-)}{1024}\right]
$$

其中 `P_edge=10 W`、`e_radio=0.2 mJ/KB`、固定元数据为 `1 KB`，每字符按 `2 byte` 估算。这些同样是模拟量，不是设备实测能耗。

#### 2. 算力拥塞代价

理论分析采用以下单服务池拥塞代理：

$$
D_{\mathrm{comp}}(K)=\frac{1}{\mu-\lambda K}
$$

这个基础式用于表达“负载接近服务能力时，拥塞代价非线性上升”的理论结构，不是本次数据拟合所直接采用的标准排队模型。

实际仿真根据多服务容器轨迹，将中心视为理想负载均衡的大算力池，并采用带固定开销与压力缩放的工程拟合模型：

$$
D_{\mathrm{comp}}(K)
=D_{\mathrm{overhead}}
+\frac{1}{c\mu_{\mathrm{card,eff}}}
\frac{a}{1-\rho(K)}
$$

$$
\rho(K)=\frac{\lambda K}{c\mu_{\mathrm{card,eff}}}
$$

上式可等价写为：

$$
D_{\mathrm{comp}}(K)
=D_{\mathrm{overhead}}
+\frac{a}{c\mu_{\mathrm{card,eff}}-\lambda K}
$$

因此它保留了 `1/(处理能力-负载)` 的非线性结构，但增加了固定开销 `D_overhead`、池化处理能力 `c*mu_card,eff` 和缩放参数 `a`。当 `D_overhead=0`、`a=1` 且 `mu=c*mu_card,eff` 时，该式在代数形式上退化为基础拥塞代理。它是轨迹驱动的利用率代理模型，不应称为标准 M/M/1 或标准 M/M/c 模型。

数据与语义校准关系为：

$$
\mu_{\mathrm{card,data}}
=\frac{1}{R}\sum_{r=1}^{R}\frac{1000}{T_r^{\mathrm{infer,ms}}}
$$

$$
\bar q=\frac{1}{N}\sum_{i=1}^{N}q_i,\qquad
\mu_{\mathrm{card,eff}}
=\frac{\mu_{\mathrm{card,data}}}{\bar q}
$$

其中 `q_i` 是 LLM 给出的任务相对计算倍率。`c` 取匹配时间点中服务容器数量的中位数；`D_overhead`、`a` 和 `lambda` 通过网关排队观测拟合。

| 参数或拟合指标 | 本次取值 | 来源 |
|---|---:|---|
| `q_bar` | 1.199 | 100 条主任务的 LLM 预测均值 |
| `c` | 109 | 数据派生的服务容器数量代理，不等同于物理 GPU 卡数 |
| `mu_card,data` | 0.044483 task/s | 推理时延数据派生 |
| `mu_card,eff` | 0.037100 task/s | `mu_card,data/q_bar` |
| `c*mu_card,eff` | 4.043909 task/s | 池化有效服务率 |
| `D_overhead` | 0.212794 s | 数据拟合 |
| `a` | 2.3966e-6 | 数据拟合 |
| `lambda` | 0.040439 | 数据拟合 |
| R-squared | 0.02038 | 拟合质量 |
| MAE | 35.154 ms | 拟合质量 |
| RMSE | 52.764 ms | 拟合质量 |

拟合使用 1002 个完成时间戳匹配的观测点，其中 863 个点的队列长度为 0，占 `86.13%`。这种明显的零膨胀意味着同一个 `K` 下仍混有网关固定开销、调度、模型加载和任务差异等未观测因素。

拟合使用鲁棒对数残差最小二乘。`R-squared=0.02038` 表明 `K` 在当前样本内只能解释很少一部分网关时延变化；`MAE=35.154 ms`、`RMSE=52.764 ms` 的单位均为毫秒，但本实验没有定义业务或 SLA 可接受阈值。因此这些误差不能被直接判定为“可部署”，该模型也不能被当作一般场景下的高精度真实时延预测器。它在本实验中的作用是提供一个数据校准、随拥塞状态变化且可进入势博弈的代价代理。

#### 3. 内存溢出惩罚

显存硬约束采用指数障碍函数。使用物理显存量表示时，其结构为：

$$
M(K)=\alpha\exp\left[\beta_{\mathrm{phys}}(Kv_{\mathrm{req}}-V_{\max})\right]
$$

Mark7 使用无量纲显存负载比例进行计算：

$$
M(K)
=\alpha\exp\left[
\beta\left(
K\frac{v_{\mathrm{req,eff}}}{V_{\max}}-1
\right)
\right]
$$

两种写法的结构等价，对应关系为：

$$
\beta=\beta_{\mathrm{phys}}V_{\max}
$$

LLM 显存倍率通过以下关系进入原有障碍函数参数：

$$
\bar m=\frac{1}{N}\sum_{i=1}^{N}m_i
$$

$$
\frac{v_{\mathrm{req,eff}}}{V_{\max}}
=\frac{\bar m}{K_{\mathrm{cap}}}
$$

本次 `m_bar=1.150`、`V_max=16 GB`、`K_cap=80`，因此单个卸载任务的等效显存比例为 `0.014375`。`V_max` 和 `K_cap` 是显式仿真假设；`alpha=5.808043`、`beta=11.010905` 根据匿名 VRAM 尾部代理拟合。

全局显存负载比例为：

$$
V(\mathbf{s})
=K(\mathbf{s})\frac{v_{\mathrm{req,eff}}}{V_{\max}}
$$

容量违规指示量为：

$$
I_{\mathrm{mem}}(\mathbf{s})
=\mathbb{I}\left[V(\mathbf{s})>1\right]
$$

这是一项软件容量代理，不是物理 GPU OOM 实测。

### 1.3 拥塞势博弈证明

#### 定义 1：精确势博弈

如果存在映射：

$$
\Phi:\{0,1\}^{N}\rightarrow\mathbb{R}
$$

使任意节点 `i` 在固定其他节点策略 `s_-i` 时，任意两种策略 `a_i,b_i` 都满足：

$$
C_i(a_i,\mathbf{s}_{-i})-C_i(b_i,\mathbf{s}_{-i})
=\Phi(a_i,\mathbf{s}_{-i})-\Phi(b_i,\mathbf{s}_{-i})
$$

则该博弈是精确势博弈。

#### 定理 1：本卸载博弈是精确势博弈

定义共享中心代价：

$$
G(k)=D_{\mathrm{comp}}(k)+M(k)
$$

候选势函数为：

$$
\Phi(\mathbf{s})
=\sum_{j=1}^{N}E_j(s_j)
+\sum_{k=1}^{K(\mathbf{s})}G(k)
$$

也就是：

$$
\Phi(\mathbf{s})
=\sum_{j=1}^{N}E_j(s_j)
+\sum_{k=1}^{K(\mathbf{s})}
\left[D_{\mathrm{comp}}(k)+M(k)\right]
$$

下面分别证明节点从本地切换到卸载，以及从卸载切换到本地时，个体代价变化都等于势函数变化。

#### 证明一：节点 `i` 从 `0` 切换为 `1`

固定其他节点策略，并定义其他节点的卸载数：

$$
K_{-i}=\sum_{j\ne i}s_j
$$

切换前 `s_i=0`，全局卸载数为 `K_-i`，节点 `i` 的代价为：

$$
C_i(0,\mathbf{s}_{-i})=e_{i,\mathrm{loc}}
$$

切换后 `s_i=1`，全局卸载数为 `K_-i+1`，节点 `i` 的代价为：

$$
C_i(1,\mathbf{s}_{-i})
=e_{i,\mathrm{tx}}+G(K_{-i}+1)
$$

因此节点个体代价变化为：

$$
\begin{aligned}
\Delta C_i^{0\rightarrow1}
&=C_i(1,\mathbf{s}_{-i})-C_i(0,\mathbf{s}_{-i})\\
&=e_{i,\mathrm{tx}}-e_{i,\mathrm{loc}}+G(K_{-i}+1)\\
&=e_{i,\mathrm{tx}}-e_{i,\mathrm{loc}}
+D_{\mathrm{comp}}(K_{-i}+1)+M(K_{-i}+1).
\end{aligned}
$$

势函数变化为：

$$
\begin{aligned}
\Delta\Phi^{0\rightarrow1}
&=\Phi(1,\mathbf{s}_{-i})-\Phi(0,\mathbf{s}_{-i})\\
&=(e_{i,\mathrm{tx}}-e_{i,\mathrm{loc}})
+\sum_{k=1}^{K_{-i}+1}G(k)-\sum_{k=1}^{K_{-i}}G(k)\\
&=e_{i,\mathrm{tx}}-e_{i,\mathrm{loc}}+G(K_{-i}+1)\\
&=\Delta C_i^{0\rightarrow1}.
\end{aligned}
$$

#### 证明二：节点 `i` 从 `1` 切换为 `0`

切换前 `s_i=1`，全局卸载数为 `K_-i+1`；切换后 `s_i=0`，全局卸载数为 `K_-i`。节点个体代价变化为：

$$
\begin{aligned}
\Delta C_i^{1\rightarrow0}
&=C_i(0,\mathbf{s}_{-i})-C_i(1,\mathbf{s}_{-i})\\
&=e_{i,\mathrm{loc}}-e_{i,\mathrm{tx}}-G(K_{-i}+1).
\end{aligned}
$$

势函数变化为：

$$
\begin{aligned}
\Delta\Phi^{1\rightarrow0}
&=\Phi(0,\mathbf{s}_{-i})-\Phi(1,\mathbf{s}_{-i})\\
&=(e_{i,\mathrm{loc}}-e_{i,\mathrm{tx}})
+\sum_{k=1}^{K_{-i}}G(k)-\sum_{k=1}^{K_{-i}+1}G(k)\\
&=e_{i,\mathrm{loc}}-e_{i,\mathrm{tx}}-G(K_{-i}+1)\\
&=\Delta C_i^{1\rightarrow0}.
\end{aligned}
$$

两个方向都满足：

$$
\Delta C_i=\Delta\Phi
$$

因此，本实验定义的二元卸载博弈是精确势博弈。

#### 有限步收敛结论

当节点执行严格改善自身代价的最佳响应时：

$$
\Delta C_i<0
\quad\Longrightarrow\quad
\Delta\Phi<0
$$

系统只有 `2^N` 个有限策略状态，所以势函数不可能沿严格改善路径无限下降或循环。严格改善过程最终会停在不存在有利单边偏离的状态，即纯策略纳什均衡：

$$
s_i^{\star}\in\mathrm{BR}_{i}(\mathbf{s}_{-i}^{\star}),
\qquad \forall i\in\mathcal{N}
$$

程序采用确定性平局规则：仅当卸载代价严格更低时选择卸载，否则选择本地。因此实际轨迹中的势函数是单调不增；抽到已经处于最佳响应的节点时，势函数会保持不变，并非每一轮都严格下降。

为核验数学式与代码实现一致，程序随机生成 200 个状态和单节点翻转，检查两个方向的 `Delta C_i=Delta Phi`。本次最大绝对误差为：

$$
\max|\Delta C_i-\Delta\Phi|
=5.28\times10^{-14}<10^{-9}
$$

该结果验证的是公式实现一致性，不证明参数具有物理真实性，也不证明得到的 PSNE 是系统总代价的全局最小值。

## 2 算法实现：基于 LLM 的语义协调机制

中心 LLM 作为 Game Master，通过“节点提议 - 中心反馈 - 节点决策”机制协调最佳响应。LLM 不直接决定节点策略；确定性 Python 公式负责最终数值核验和决策，从而保留可复现性。

### 2.1 携带语义特征的任务 Intent

每个边缘节点向中心发送结构化 Intent：

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

DeepSeek 语义解析器针对每个任务返回：

- `q_i`：相对计算倍率；
- `m_i`：相对显存倍率；
- `risk_level`：风险等级；
- `semantic_warning`：语义风险提示。

为了使全部节点共享同一个 `D_comp(K)` 和 `M(K)`，从而保持上面的精确势结构，主实验对 100 条任务预测取算术平均：

$$
\bar q=\frac{1}{N}\sum_{i=1}^{N}q_i,\qquad
\bar m=\frac{1}{N}\sum_{i=1}^{N}m_i
$$

完整运行时，程序用固定种子从阿里任务池构造 200 条 Intent，并以带 UTC 运行标识的缓存文件调用 `deepseek-v4-flash`。由于本实验只需要结构化数值反馈，API 请求显式设置 `thinking=disabled`，并使用 `max_tokens=512`，避免无关推理耗尽输出预算。每条语义预测同时保存任务 `intent_hash`、请求模型、服务端实际模型、时延、Token 数、解析状态和缓存状态到 `semantic_resource_predictions.csv`；`semantic_generation_manifest.json` 记录任务数、API 调用数、模型、温度、思考模式、Token 上限、Intent 哈希和缓存文件名。

聚合校准参数保存在 `semantic_parameter_calibration.csv`。使用全局均值使所有节点面对同一个只依赖卸载总数 `K` 的共享中心代价 `G(K)`，从而保留精确势函数证明所需的结构；代价是不同长 Prompt、Steps、LoRA 和图片数量任务之间的细粒度差异不会直接进入每次最佳响应。

### 2.2 中心 Game Master 反馈

每次更新时，Game Master 接收当前拥塞状态、候选拥塞状态、当前与候选显存比例、任务 Intent 及公式分项，并返回：

```json
{
  "center_cost_increment": 1.23,
  "compute_impact": "...",
  "vram_impact": "...",
  "semantic_warning": "..."
}
```

假设节点 `i` 暂时设为本地执行，其他节点形成 `s_-i`。若节点 `i` 改为卸载，中心代价为：

$$
\Delta_i^{\mathrm{center}}
=D_{\mathrm{comp}}(K_{-i}+1)+M(K_{-i}+1)
$$

Python 预先提供 `D_comp` 与 `M` 的公式分项，并始终重新计算上述结果。LLM 根据拥塞状态、Intent 和这些分项返回中心代价、算力影响、显存影响和语义警告；返回数值只有在容差内与确定性结果一致时才被标记为一致，节点实际使用的是 Python 核验值，不是未经约束的自然语言输出。因此这里实现的是受公式约束的 LLM Game Master，而不是让 LLM 独立估算未知的中心代价函数。

### 2.3 LLM 协同最佳响应算法

```text
输入：节点集合 N、中心 LLM 协调器、最大更新次数 T_max
输出：候选均衡策略 s*

1. 使用固定随机种子生成初始二元策略 s(0)
2. 对 t = 1, 2, ..., T_max：
3.     随机抽取一个边缘节点 i
4.     把节点 i 暂时设为本地，计算 K_-i
5.     节点 i 向 Game Master 发送结构化 Intent
6.     Game Master 感知 K_-i，并评估候选状态 K_-i + 1
7.     计算并返回：
           Delta_center = D_comp(K_-i + 1) + M(K_-i + 1)
       同时返回 compute impact、VRAM impact 和 semantic warning
8.     Python 重新计算并核验 Delta_center
9.     如果 e_i,tx + Delta_center < e_i,loc：
10.        s_i <- 1
11.    否则：
12.        s_i <- 0
13.    如果连续 N 次更新都没有策略变化：
14.        对全部节点执行一次最佳响应检查
15.        如果所有节点都不存在有利单边偏离，则停止
16. 返回最终策略 s*
```

主实验参数为 `T_max=1500`、连续无变化阈值 `N=100`。增加全节点检查是为了防止“连续随机抽到无需改变的节点”被误判为均衡。

## 3 仿真测试与验证指标设计

### 3.1 基准算法对比

实验复现以下四类基准：

1. **All-Local**：所有节点本地执行，`s_i=0, forall i`。
2. **All-Offload**：所有节点卸载，`s_i=1, forall i`。
3. **Random Allocation**：每个节点独立按 `Bernoulli(0.5)` 选择策略；主结果报告 500 次随机策略的均值。
4. **Greedy Heuristic**：忽略中心拥塞，仅在 `e_i,tx < e_i,loc` 时卸载。

### 3.2 核心评估指标

#### 指标一：博弈收敛性分析

**测试方法：** 每轮记录 `K(s)`、势函数 `Phi(s)`、系统总代价、显存负载和策略是否变化，并在停止后检查全部节点是否满足最佳响应条件。

**输出：** `model_convergence_traces.csv`、`model_equilibrium_strategies.csv`、`formula_audit.csv` 和图 `02_convergence_stability`。

**本次结果：**

| 指标 | 结果 |
|---|---:|
| 节点数 `N` | 100 |
| 最终卸载数 `K*` | 63 |
| 最佳响应更新次数 | 858 |
| 实际策略变化次数 | 70 |
| 最终势函数 | 110.0005 |
| 全节点 PSNE 检查 | 通过 |
| 势函数恒等式最大误差 | `5.40e-14` |

结果支持“本次有限仿真达到一个 PSNE”。势函数只在发生严格改善时下降，在无策略变化的轮次保持不变，因此应表述为“单调不增并最终稳定”，不能表述为每一轮都严格下降。

#### 指标二：帕累托候选前沿与系统总代价

系统总代价按全部节点个体代价求和：

$$
C_{\mathrm{sys}}(\mathbf{s})
=\sum_{i=1}^{N}C_i(s_i,\mathbf{s}_{-i})
$$

对当前二元共享拥塞模型，它也可以写成：

$$
C_{\mathrm{sys}}(\mathbf{s})
=\sum_{i=1}^{N}E_i(s_i)
+K(\mathbf{s})\left[D_{\mathrm{comp}}(K(\mathbf{s}))+M(K(\mathbf{s}))\right]
$$

注意 `C_sys` 与势函数 `Phi` 不是同一个量。势函数用于证明单边最佳响应收敛，系统总代价用于比较策略整体表现。

平均模拟能耗与全节点平均排队时延分别为：

$$
\bar E_{\mathrm{sim}}
=\frac{1}{N}\sum_{i=1}^{N}
\left[(1-s_i)E_i^{\mathrm{loc,sim}}+s_iE_i^{\mathrm{tx,sim}}\right]
$$

$$
\bar D_{\mathrm{all}}
=\frac{K(\mathbf{s})}{N}D_{\mathrm{comp}}(K(\mathbf{s}))
$$

**测试方法：** 比较主算法与四类基准的 `C_sys`、平均模拟能耗、排队时延和容量状态；另按本地与传输代价差构造 `K=0...N` 的候选策略，报告能耗 - 时延候选前沿。该前沿是结构化候选集上的前沿，不是对全部 `2^N` 个策略的穷举。

**输出：** `algorithm_comparison.csv`、`pareto_front.csv` 和图 `03_system_cost_comparison`。

**本次结果：**

| 策略 | `K` | 系统总归一化代价 | 容量违规 |
|---|---:|---:|---:|
| LLM 协调最佳响应 | 63 | 211.18 | 0 |
| All-Local | 0 | 280.60 | 0 |
| All-Offload | 100 | 58463.59 | 1 |
| Greedy | 100 | 58463.59 | 1 |
| Random `p=0.5`，500 次均值 | 49.882 | 181.77 | 0 |

随机基线的平均系统总代价 `181.77` 低于本次 PSNE 的 `211.18`，同时随机基线的平均卸载数约为 `49.882`，低于 PSNE 的 `63`。随机分配只是 500 个静态 Bernoulli 策略的数值基线：它不检查节点是否愿意单边改变策略，也不是分布式最佳响应均衡或具有个体理性的调度算法。因此结果支持“协调策略达到 PSNE 并避免主实验容量违规”，但不支持“协调策略的系统总代价优于所有基准”，更不支持“达到全局最优”。

#### 指标三：内存硬约束违规率

对每种节点规模与算法重复 `R=30` 次，模拟内存违规率定义为：

$$
\mathrm{MVR}(N)
=\frac{1}{R}\sum_{r=1}^{R}
\mathbb{I}\left[V(\mathbf{s}^{(r)})>1\right]
$$

**测试方法：** 将节点数从 `N=30` 增加到 `N=200`，步长为 10，分别统计主算法和四类基准的容量违规率。

**输出：** `memory_violation_sweep.csv` 和图 `04_memory_violation_rate`。

**本次结果：** 在本次 30 次重复的软件容量测试中，最佳响应策略在 `N=30...200` 的违规率均为 0。All-Offload 和 Greedy 从 `N=80` 起违规率为 1；Random 的违规率从 `N=120` 开始上升，并在 `N>=170` 时达到 1。该结果只适用于当前 `16 GB`、`80` 个等效基础槽位、语义倍率和任务样本，不能解释为真实硬件 OOM 保证。

#### 指标四：LLM 协调开销分析

**测试方法：** 记录 Game Master 的逻辑协调次数、真实 API 调用数、缓存复用、Token 数、平均 API 时延、P95 API 时延和数值核验结果。

**输出：** `llm_feedback_events.csv`、`llm_coordination_overhead.csv` 和图 `05_llm_coordination_overhead`。

**本次结果：**

| 指标 | 结果 |
|---|---:|
| 新生成的任务语义预测 | 200 |
| 本次语义预测真实 API 调用 | 200 |
| 逻辑协调次数 | 858 |
| 本次 Game Master 真实 API 调用 | 267 |
| 缓存复用 | 591 |
| Game Master 记录 Tokens | 105,911 |
| 任务语义预测 Tokens | 61,718 |
| 平均 API 时延 | 1455.19 ms |
| P95 API 时延 | 1933.32 ms |
| 请求模型 | `deepseek-v4-flash` |
| 服务端返回模型 | `deepseek-v4-flash` |
| LLM 数值与公式不一致 | 0 |

实测 API 时延约为秒级，不是微秒至毫秒级。因此当前结果不能证明 LLM 协调已经满足实时控制要求；它只说明该机制能够在离线软件仿真中完成语义反馈与数值核验。请求使用 `temperature=0` 和 `thinking=disabled`，但这本身不构成模型输出数学确定性的保证；固定输入、持久化缓存和 Python 数值复核共同提供本实验所需的可重复性。

## 4 代码、输出与复现

### 4.1 目录结构

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

### 4.2 主要输出文件

| 文件 | 内容 |
|---|---|
| `selected_model_B_metrics.csv` | 拥塞函数参数与拟合质量 |
| `selected_model_B_fit_points.csv` | 每个拟合点的观测、预测与残差 |
| `semantic_intents.csv` | 200 条结构化任务 Intent |
| `semantic_resource_predictions.csv` | 本次完整运行新生成的 LLM 语义影响预测记录 |
| `semantic_generation_manifest.json` | 语义生成模型、API 调用、温度、Intent 哈希和缓存溯源 |
| `semantic_parameter_calibration.csv` | `q_bar`、`m_bar`、`mu_eff` 和 `v_req_eff` |
| `model_equilibrium_strategies.csv` | 100 个节点的最终策略 `s*` |
| `model_convergence_traces.csv` | 每轮 `K`、势函数、总代价和容量状态 |
| `model_game_metrics.csv` | 主实验汇总指标与 PSNE 检查 |
| `llm_feedback_events.csv` | 每轮 LLM 回复、Token、时延和公式核验 |
| `llm_coordination_overhead.csv` | 调用量、Token、平均时延和 P95 时延 |
| `algorithm_comparison.csv` | 主算法与四类基准 |
| `pareto_front.csv` | 能耗 - 时延候选前沿 |
| `memory_violation_sweep.csv` | `N=30...200` 的容量压力测试 |
| `formula_audit.csv` | `Delta C_i=Delta Phi` 数值审计 |
| `assumptions_and_parameters.csv` | 数据派生参数、拟合参数和显式假设 |
| `run_summary.json` | 实时 LLM 实验总汇总 |

五张结果图分别以 PNG、PDF、SVG 和 TIFF 四种格式保存在 `03_结果图` 对应目录中。

### 4.3 运行方法

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

`--full` 会创建带 UTC 运行标识的新缓存文件，生成 200 条语义预测，再完成中心协调与博弈计算。实际 API 数、Token 数和时延写入 `semantic_generation_manifest.json`、`llm_coordination_overhead.csv` 和 `run_summary.json`。

不调用 API 的公式与流程核验：

```bash
.venv/bin/python 01_源码与说明/mark7_pdf_cost_experiment.py --deterministic
.venv/bin/python 01_源码与说明/postprocess_mark7_full.py
```

API 密钥只从环境变量读取，不能写入代码、README、CSV、JSON 或缓存。

## 5 结论边界

- 本实验验证了代码中的精确势恒等式，并在本次运行中得到经过全节点检查的 PSNE；
- 精确势博弈保证严格改善路径到达某个 PSNE，不保证该均衡使系统总代价全局最小；
- 当前拥塞函数的 `R-squared=0.02038`，拟合解释力很弱，不能宣称得到了一般拥塞规律；
- `c=109` 是服务容器数量代理，不是确认的物理 GPU 卡数；
- Prompt 长度是字符数，不是真实 tokenizer Token 数；
- `16 GB`、`80` 个等效槽位、本地代价系数、传输代价系数和模拟能耗参数包含显式假设；
- 将任务级语义倍率取均值是为了保留共享拥塞函数和精确势结构，但会损失任务异构性；
- 只有在执行 `--full` 并生成新的 `semantic_resource_predictions.csv` 后，结果才能称为从阿里任务样本到 LLM 协调的完整运行；
- Memory Violation Rate 是软件容量代理，不是物理 OOM 实测；
- LLM API 时延为秒级，当前实验不支持实时部署结论；
- 本实验是公开匿名轨迹驱动的机制验证，不是生产部署验证。
