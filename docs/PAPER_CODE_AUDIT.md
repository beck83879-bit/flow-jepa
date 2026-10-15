# Flow-JEPA 论文、公式与实现复核

审计日期：2026-10-15。论文版本：[2608.29029v3](https://arxiv.org/abs/2608.29029v3)，共 17 页；涉及继承设置时同时核对 [LeWM v3](https://arxiv.org/abs/2603.19312v3)。
被审计代码：上游 `ab73e7c50435e3a1fffb4d3f6d175445e57a2f4e`，复现提交 `b9fbe1ed7121d1c19f6f444ff4abcbb13d2afdfd`，以及服务器训练种子 3073 的四个第 20 轮权重。

**结论：主要算法和实际实验的主参数对齐，但不能称为公式、描述、全部附录及参数完全一致。**
新增的关键证据是：TwoRoom/Cube 的双向未来 self-attention 允许后续动作间接影响较早预测，这与附录“动作条件始终保持时间因果”的严格含义存在张力。
另有损失归约的数学记号歧义、通用默认配置与论文任务配置不一致，以及作者未公开的视觉初始化和评估细节。

本次仅增加审计脚本、证据和文档；没有修改模型、训练损失、评估逻辑、已有权重，也没有启动新训练或重新计算正式成功率。

## 1. 最重要的发现

### 1.1 前缀 cross-attention 不等于端到端动作因果性

附录 A，第 13 页原文：

> In both variants, action conditioning remains temporally causal: the token at position i receives only the action prefix a_t:t+i-1.

实现同时包含以下行为：

- [module.py](../module.py#L344) 的 cross-attention mask 确实只让第 i 个未来 token 直接读取对应的动作前缀。
- TwoRoom/Cube 的 future self-attention 是双向的；PushT/Reacher 是因果的，任务选择与附录一致。
- [FlowTrajectoryBlock](../module.py#L215) 每层按 self-attention、cross-attention、MLP 执行，预测器共 6 层。
- 因此，第 j 个未来 token 在一层 cross-attention 中接收较晚动作，下一层的双向 self-attention 就可能把该信息传给更早的 token。迭代 Euler 推理也会传递这种依赖。

本次使用实际最终权重、CPU float32、`eval()`、固定历史和初始噪声，只给最后一个动作块的各维加 0.25。输入为合成潜变量和归一化尺度的动作，用于检验函数依赖；没有使用环境 rollout 来计算新的成功率。

| 任务 | 未来 self-attention | 首个速度 token 最大绝对变化 | 8 步 Euler 后首个预测 token 最大绝对变化 | 首个速度投影对最后动作的最大绝对梯度 |
|---|---|---:|---:|---:|
| TwoRoom | 双向 | 0.0272923 | 0.0994850 | 0.0246381 |
| Cube | 双向 | 0.0480962 | 0.113173 | 0.0258472 |
| PushT | 因果 | 0 | 0 | 0 |
| Reacher | 因果 | 0 | 0 | 0 |

前四个动作的编码在干预前后逐元素完全相同。四个任务在第一层后的首个 token 变化均为 0；TwoRoom/Cube 在后续层出现变化，符合跨层传播路径。四个任务的最后一个输出 token 均对最后动作有反应，因此零变化并非模型完全忽略动作。

**如何定性：**如果论文只是在描述直接 cross-attention 的可见范围，实现符合；如果它要求最终早期预测完全独立于未来动作，TwoRoom/Cube 实现不满足。需要作者澄清其“temporally causal”的含义，不能简单地把两种表述等同。

这也是对 2026-10-14 审计表中“动作条件一致”的修正：此前只检查了直接 mask，结论范围过宽。该行为来自未改动的上游模型，不是本复现新增的改动。
本诊断证明的是函数依赖，不能证明它导致了成功率下降。当前规划只使用最后的预测状态，而最后状态本来就应依赖所有动作；影响究竟有利还是有害需要另做受控实验。不能直接把双向配置改成因果并声称仍在复现同一论文设置。

### 1.2 式 (13) 的平方范数与代码 MSE 存在归约歧义

论文第 6 页式 (13) 写为对轨迹误差平方范数取期望；[jepa.py](../jepa.py#L174) 实际执行：

```python
F.mse_loss(pred_velocity, target_velocity)
```

默认 `reduction="mean"`，对 batch、预测时间、token、特征维全部平均。当前形状为 `B x 5 x 1 x 192`。
如果把论文范数理解为每个样本内的元素平方和，再对样本取平均，则：

```text
L_literal_norm = 5 * 1 * 192 * L_code_MSE = 960 * L_code_MSE
```

四个实际权重的数值检查给出比值约 960（浮点误差范围 959.999817–960.000061）。
有 SIGReg 项时，这不是对整个目标函数统一乘常数：固定 lambda 后，两个损失的相对尺度会改变。
PushT 的视觉模块冻结，SIGReg 本身不向可训练的 Flow 模块提供梯度；其余任务则需要特别注意相对尺度。

**如何定性：**这是明确可测的实现归约和公式字面归约差异，但论文可能省略了均值约定；上游也是同样的 MSE。尚不能判断作者真实实验采用了另一种归约，更不能据此直接把损失乘 960 或调整 lambda。

### 1.3 通用默认配置不是四个任务的论文配置

| 入口 | 当前通用默认值 | 论文主实验 / 本次正式实验 |
|---|---|---|
| `config/train/fjepa.yaml` + `model/fjepa.yaml` | standard noise、双向、无视觉初始化、dropout 0.1、SIGReg 0.09 | 应按任务选择；尤其不等于论文 PushT/Reacher |
| `config/eval/solver/cem.yaml` | CEM 30 轮 | Flow 四任务均 10 轮 |
| `config/eval/pusht.yaml` | noisy 半径 35 | PushT 主比较为半径 10 |

所以仅替换 `data=pusht` 或 `data=dmc`，不会自动得到论文所选配置。只启用 PushT 的噪声开关而不覆盖半径，也不是论文主比较。

**已完成的 3072/3073 实验不受这些默认值影响：**训练通过 `paper_*.yaml` 显式设定任务差异；`run_flow_evaluation.sh` 明确覆盖 CEM 为 10、PushT 半径为 10。本次重新读取了第二轮 24 份正式评估 manifest，均为正确值，且对应权重 SHA-256 与本次数值诊断一致。

### 1.4 初始化和扰动协议仍未达到逐项可核验

PushT/Reacher 使用本地训练的 LeWM 视觉权重，冻结 / 继续训练的行为符合附录，但作者所选初始化 checkpoint 的唯一标识尚未得到对应。
论文式 (17) 与代码的单幅图像加噪公式一致，但论文没有明确完整 epsilon 图是否逐帧重采样、是否也污染目标图。代码固定每条 rollout 的整张噪声图，并污染目标与当前图像。
这些会影响复现边界，不能用“s、r 相同”替代完整协议一致性证明。

## 2. 逐公式与架构对应

| 论文位置 | 实现对应 | 结论 |
|---|---|---|
| 式 (1)–(2)，第 3 页：随机单位方向投影、Epps–Pulley SIGReg | `module.SIGReg`，Gaussian 方向后归一化，特征函数与标准正态比较 | 结构一致；数值积分、按时间分开统计见下文 |
| 式 (3)–(4)，第 4 页 | 描述 LeWM 自回归基线 | 不是要求 Flow 模型也实现同样的递推 |
| 式 (5)–(7)，第 4–5 页：线性路径、常速度监督 | `JEPA.flow_loss` 的线性插值、`target - source` | 一致；损失归约见 1.2 |
| 式 (8)–(9)，第 5 页：h 个未来潜变量及动作嵌入 | `train.flow_jepa_forward` 取 1 帧历史、5 帧目标、对应 5 个动作块 | 当前维度 `B x 5 x 1 x 192`，单 CLS token 对应论文 `h x d` |
| 式 (10)、附录 A：独立 Gaussian 源 | `torch.randn_like`；standard 或 current + 0.5 epsilon | 0.5 为标准差，方差为 0.25；任务配置一致 |
| 式 (11)：`Z_tau = (1-tau)Z0 + tau Z` | 每个 batch 样本一个均匀 tau，广播到整条轨迹 | 一致；没有对每个未来 token 另采一个 tau |
| 式 (12)、图 3：时间、历史、动作条件的向量场 | sinusoidal time/position、AdaLN、self/cross attention、MLP | 主结构一致；动作间接依赖例外见 1.1 |
| 式 (13)：Flow Matching 回归 | 预测 velocity，监督 `Z-Z0` | 插值、监督一致；均值与范数记号有歧义 |
| 式 (14)：Flow + lambda SIGReg | `pred_loss + lambd * sigreg_loss`；当前与未来 6 帧均正则化 | 相加和覆盖范围一致 |
| 式 (15)–(16)：显式 Euler | `tau=k/N`、`dt=1/N`、`x += dt*v`，N=8 | 一致，无错误使用末端时间、额外步长或自回归替代 |
| §4.2：ViT CLS 后 MLP、独立动作编码 | `JEPA.encode`、`module.Embedder` | 一致；动作编码 kernel=1，不跨时间混合 |
| §5.1：只使用末端 latent 的规划代价 | `JEPA.criterion` | 仅最后预测状态对目标 latent 的平方距离，未把所有时间步相加 |
| 式 (17)，第 7 页：RGB 空间局部 Gaussian 扰动 | `make_noise_patches`、`add_noise` | 空间核、独立 RGB epsilon、clip 一致；uint8 量化及协议边界见第 5 节 |

Flow 训练中没有 `detach()` 目标潜变量；未冻结时，梯度也会经过目标和 state-centered 源中的当前 latent。
这与联合训练视觉表征的总体描述相容，但论文没有逐一规定这些梯度路径，不能宣称 stop-gradient 细节已经由公式唯一确定。

## 3. 主实验参数核对

### 3.1 所有任务的共同配置

“明确”表示 Flow 论文直接给出；“继承”表示论文指向 LeWM，后者给出；“代码设置”表示可从作者实现和实际运行确认，但不能独立从这两篇论文查到完整数值约定。

| 项目 | 论文依据 | 第二轮实际值 | 判断 |
|---|---|---|---|
| epochs / batch | 附录 A，明确 | 20 / 128 | 一致 |
| history / prediction horizon | 附录 A，明确 | 1 / 5 | 一致 |
| 帧间隔 / 每动作块底层动作数 | §4.1、附录 B.1，明确 | 5 / 5，按顺序拼接 | 一致 |
| 输入观察数 | 1 历史 + 5 未来 | 6 | 一致 |
| 图像 / backbone / patch | 附录 A，明确 | 224x224 / ViT-Tiny / 14x14 | 一致 |
| latent / action embedding | 附录 A，明确 | 192 / 192 | 一致 |
| predictor depth / heads / head dim / MLP | 附录 A，明确 | 6 / 16 / 64 / 2048 | 一致，attention 内部维度 16x64 不必等于 latent 192 |
| Euler 积分步数 | §5.3、Table 5/9，主实验 8 | 8 | 一致 |
| CEM iterations / candidates / elites | §5.1 + LeWM planning，明确及继承 | 10 / 300 / 30 | 主参数一致 |
| 初始 CEM variance | LeWM planning，继承 | 1 | 一致 |
| horizon / receding horizon / action block | §5.1，明确 | 5 / 5 / 5 | 执行全部 25 个底层动作后再规划 |
| 目标偏移 / 单例动作预算 | LeWM 附录 F.1，继承 | 25 / 50 | 一致 |
| 单组评估例数 | §5.1，明确 | 每种子每条件 50 | 一致；额外汇总 3 组不是论文同一组 50 例 |
| 硬件 | 附录 A，明确 | 每任务 1 张 RTX 6000 Ada | 型号一致；多任务并行的计时不等同作者隔离计时 |
| optimizer / LR / weight decay | 代码设置 | AdamW / 5e-5 / 1e-3 | 与作者代码一致，论文没有完整独立数值表 |
| LR schedule | 代码及依赖设置 | optimizer step 更新；1% warmup 后 cosine | 既有日志核验通过，不能只按 `interval=epoch` 推断实际行为 |
| precision / gradient clip | 代码设置 | bf16-mixed / 1.0 | 代码与运行一致，论文未明确这两项 |
| train split / seed | 代码设置 | 0.9 / 3073 | 作者所选划分和训练种子未公开对应清单 |
| SIGReg projections / knots | LeWM 及代码设置 | 1024 / 17 | 1024 可由 LeWM 正文确认；积分离散细节见第 4 节 |

### 3.2 分任务设置

| 任务 | Flow 源 | future self-attention | 视觉 encoder + projector | predictor dropout | lambda SIGReg |
|---|---|---|---|---:|---:|
| TwoRoom | N(0,I) | 双向 | 从头联合训练 | 0.1 | 0.1 |
| Cube | N(0,I) | 双向 | 从头联合训练 | 0.1 | 0.1 |
| PushT | N(z_current,0.5²I) | 因果 | LeWM 初始化，冻结 | 0 | 0.09 |
| Reacher | N(z_current,0.5²I) | 因果 | LeWM 初始化，继续训练 | 0.1 | 0.1 |

上述字段与附录 A 一致，但不能据此消除第 1 节列出的歧义和权重来源限制。
实际视觉初始化：PushT 使用本地 LeWM seed3072 epoch10，Reacher 使用本地 LeWM seed3073 epoch10。Flow 的训练种子 3073 与初始化模型的训练种子是不同概念。

## 4. SIGReg、数据与 CEM 的实现细节

SIGReg 先将 embedding 排列为 `T x B x D`，对每个时间位置分别在 batch 维估计特征函数，然后平均时间位置与 1024 个投影；不是把 `T*B` 样本合成一个总体再检验。
当前只保留 CLS，因此 token 数为 1；这种按时间统计方式与 LeWM 伪代码及其“每个时间步独立施加 SIGReg”的描述相符。

代码在 `[0,3]` 使用 17 个节点，以梯形法、Gaussian 权重和偶对称因子近似双边积分，并乘 batch size。
用独立的 `torch.trapezoid` 表达重算，当前 CPU 测试绝对误差为 0。LeWM 附录对积分范围给了例如 `[0.2,4]` 的示意，含 “e.g.”，不能把示例范围当成强制参数而断言当前代码错误。

数据与预处理的既有核验结论仍适用：

- 每任务首、中、末窗口的观察帧与动作块曾与 H5 原始数据逐元素对照；此次未重做完整数据扫描。
- 图像预处理抽查误差为 0；动作训练归一化使用样本标准差，评估使用总体标准差，其相对尺度差约 2e-7 至 6e-7，不支持解释几十个百分点的差距。
- 官方 PushT H5 为 18,685 条轨迹、平均约 125 帧，LeWM 文字描述为 20,000 条、平均 196 步。官方压缩包与训练文件的完整哈希已核验，不能将差异归因于下载不完整，也不能自行补数据后称为严格复现。
- 训练随机划分的是数据窗口；评估从公开离线数据抽起点。没有作者提供的独立固定测试清单，不应宣称对完全独立测试轨迹的泛化结论。

本次另核对了安装版本的 `stable_worldmodel/solver/cem.py` 与 `policy.py`：

- 300 个候选中，第一个候选固定为当前均值，另外 299 个随机采样；这比论文“300 randomly sampled”更具体，是继承求解器的实现细节。
- 精英更新使用 `std`，尽管变量名是 `var`，并未把方差误作标准差；初始尺度为 1 时也不存在平方根差异。
- 5 个 block 的计划被 reshape 成 25 个独立底层动作依次执行，并非同一个动作重复 5 次。依赖中的 `action_block` docstring 用了 “repeated”，应以实际 reshape/数据组织为准。
- 整个计划执行完再规划，故在当前 horizon=receding_horizon 设置下没有剩余计划用于 warm start。
- 起点选择沿用上游 `choice(len(valid_indices)-1)`，排除了最后一个有效索引。影响范围很小，但严格而言不等于从所有有效起点均匀抽样；本次为保留既有实验配对没有改动。

## 5. 图像噪声公式与语义边界

式 (17) 的实际计算为：

```text
epsilon[p,c] ~ N(0,s²)
weight[p] = exp(-||p-center||² / (2*r²))
output = uint8(clip(image + weight * epsilon, 0, 255))
```

主实验 s=100；TwoRoom/Reacher/Cube r=35，PushT r=10。噪声在原始 0–255 RGB 空间添加，再做模型归一化。
本次固定随机数并捕获 epsilon，对两个半径逐像素独立重算：空间核和 float32 噪声数组误差为 0；clip 后转 uint8 的参考输出误差为 0。与不量化的实数公式相比，uint8 截断误差严格小于 1 灰度级，不能将其误称为更改了 s 或 r。

仍需明确以下区别：

| 细节 | 论文能确认什么 | 当前公开代码行为 |
|---|---|---|
| 时间一致性 | 中心在每条 rollout 内固定 | 中心和整张 epsilon 图均固定 |
| 目标图 | 未明确是否也污染 | 同一噪声图用于目标与当前观察 |
| 背景采样 | 远离 agent 和任务物体 | 使用起点/目标点集、分割或颜色启发式 |
| 保护范围 | 定性描述 | 中心距保护点大于 r+14；尝试 1000 次后最远网格点回退，未必仍满足距离 |
| 动态物体 | 未给出完整跟踪协议 | 固定中心不保证运动后的物体仍远离噪声 |
| Gaussian 支撑 | 式 (17) 为 Gaussian 衰减 | 核不是硬圆形 mask，中心远离物体不代表物体像素完全无噪声 |

因此，“公式一致”和“作者每条评估轨迹的扰动完全一致”是不同结论。当前没有足够证据认定这些未指定细节就是性能差距的原因，也不能为了提高分数直接缩小或移动噪声后继续使用主实验标签。

## 6. 附录覆盖范围

| 内容 | 论文设定 | 当前复现状态 |
|---|---|---|
| 附录 A | 模型、源分布、注意力、初始化、训练、硬件 | 本报告逐项核对；保留上述因果性与归约问题 |
| 附录 B.1 | 四环境、frameskip=5、5 动作块 | 主协议采用；环境精确版本和作者原始权重不可完全对应 |
| Fig.7–8 | Reacher/PushT 的定性成功案例 | 本地有案例与视频，但不是作者同一测试样本，不能逐帧对齐 |
| Table 3/7、Fig.9 | r=10,15,20,25,30,35,40，s=100 | 仅主设定已有正式结果，完整半径 sweep 未复现 |
| Table 4/8、Fig.10 | s=50,100,150,200；PushT r10、其余 r35 | 完整强度 sweep 未复现 |
| Table 5/9、Fig.11 | Euler 4,8,16,32 | 只完成主设定 8 步；当前 `eval.py` 明确限制为 8 |
| Table 6 | PushT CEM 10,20,30 | 只完成主设定 10；当前正式验证器要求 10 |
| Table 10 | TwoRoom 推理延迟与 50 rollouts 计时 | 未完成同口径基准；现有带渲染/视频/审计保存的运行时间不可直接比较 |

这意味着已复现工作的范围是主实验尝试，而不是完成论文全部附录实验。附录 sweep 的额外点不是主实验必须同时采用的参数，不能把 sweep 的 32 步误当成所有正式评估应改用 32 步。

## 7. 结果解释与下一步优先级

第二轮训练种子 3073 的已有正式结果（评估种子 42/43/44 各 50 例合计）：

| 任务 | 本地 clean / noisy | 论文 clean / noisy |
|---|---:|---:|
| TwoRoom | 96.00% / 60.00% | 100% / 96% |
| PushT | 82.67% / 45.33% | 96% / 90% |
| Reacher | 78.00% / 75.33% | 90% / 90% |
| Cube | 79.33% / 49.33% | 82% / 66% |

这些仍是此前完成的成绩，本次公式诊断没有改变它们。论文各条件 50 例，本地汇总 150 例；作者的测试起点不公开，所以只能数值比较，不能宣称相同测试样本上的严格复测。

优先需要澄清或获取的内容：

1. 作者所谓动作因果性是否仅指直接 cross-attention mask，以及所选模型是否就是当前公开的双向堆叠实现。
2. 式 (13) 实际采用 mean 还是 trajectory sum，以及 SIGReg 的准确归约约定。
3. PushT/Reacher 所选视觉初始化权重、训练配置和对应 checkpoint 哈希。
4. 作者评估起点、环境依赖版本、目标图加噪与 epsilon 时间相关性协议。

在这些问题澄清前，保留现有公开实现作为基准是可复查的选择。后续若研究严格动作因果版本或其他噪声语义，应使用新的实验名、独立结果目录与配对评估，归类为诊断/消融，不覆盖当前复现结论。

## 8. 数值证据与复查

- [数值诊断脚本](../tools/audit_paper_equations.py)：四任务 checkpoint、固定动作干预、层间 hook、梯度、Flow 损失、Euler、式 (17)、SIGReg。
- [数值结果与解析训练配置](paper-equation-checks-20261015.json)：含脚本哈希、模型哈希、被加载核心代码哈希。全部使用 CPU float32，所有模型处于 eval 模式。
- [实验源码哈希](experiment-source-sha256.json)：此前实际实验的来源记录。
- [复现总说明](../REPRODUCTION.md)：运行方式、逐种子正式结果、依赖和已知限制。

当前权重身份：

| 任务 | epoch20 权重 SHA-256 |
|---|---|
| TwoRoom | `2f7b7eb19be4cc9195890e85cb3f04cdef80fe9e907361468c56d0c8383000f7` |
| Cube | `6c94fe4e887c19f2fa3599d68839c090f7bc51ba93b3b7d5d20ccc406db5da74` |
| PushT | `e97916062d0386c7f21a073266eb3391043c044c96118920dd6b93e7dc9e5d43` |
| Reacher | `6a7930f370850a3ff7cd1a51e41cedf3df0f06a7e9fc0095c0d060baed666d5c` |

在已有依赖和四个权重的环境中，从仓库根目录复查；使用新的输出路径：

```bash
python tools/audit_paper_equations.py \
  --repo "$PWD" \
  --data-root "$STABLEWM_HOME" \
  --output "$STABLEWM_HOME/evaluations/flow_equation_audit_recheck/numerical_checks.json"
```

脚本拒绝覆盖已有输出。数值探针使用合成输入，作用是验证公式和依赖关系，不是测试集导航评估；一个非零反例可以否定严格函数独立性，但不能由该数值推算闭环成功率损失。

本次数值重算中，四模型源分布、Flow MSE 和 8 步 Euler 的参考误差均为 0；SIGReg 与独立梯形积分表达的误差为 0；RGB 扰动核及量化后输出参考误差为 0。这些测试不等价于证明未公开的作者实验配置与当前实现完全相同。
