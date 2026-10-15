# Flow-JEPA 复现说明

本分支保存截至 **2026-10-15** 的 Flow-JEPA 复现代码与实验说明。目标仓库为
`beck83879-bit/flow-jepa`，分支为 `reproduction/seed3073`。

**结果摘要：训练正常完成，但尚未复现论文的整体成功率，尤其是视觉噪声下的鲁棒性。**
第二轮仅更换训练种子，没有带来整体改善。本文保留实际结果和已知限制，不能将
“参数核验通过”理解为“完整复现论文”。

## 1. 来源与改动范围

- 上游：[HuoYanchen/Flow-JEPA](https://github.com/HuoYanchen/Flow-JEPA)。
- 基础提交：`ab73e7c50435e3a1fffb4d3f6d175445e57a2f4e`。
- 论文：[Flow-JEPA](https://arxiv.org/abs/2608.29029)，实验对照使用 v3 的 Table 1/2。
- 数据与基础环境：[LeWM](https://github.com/lucas-maes/le-wm)。
- 保留上游 Git 历史、README 展示素材及 [MIT 许可证](LICENSE)。原 README 的演示不是本轮复现产生的结果。

`jepa.py`、`module.py`、`utils.py` 中的核心模型实现保持与基础提交一致。主要新增或修改：

| 文件 | 作用 |
|---|---|
| `train.py` | 提前调用 `pl.seed_everything`，并显式向训练 Manager 传递种子 |
| `config/train/paper_*.yaml` | 四任务固定训练配置，显式定义数据路径、Flow 源、视觉初始化及冻结策略 |
| `run_flow_training.sh` | 空闲 GPU 检查、配置核验、短测、源码快照、正式训练和退出码记录 |
| `verify_flow_run.py` | 核验解析配置、实际模型源码路径、Flow 预测器、有限权重和冻结视觉模块 |
| `eval.py`、`eval_utils.py` | 统一 clean/noisy 闭环执行、固定种子、恢复数据初始状态、停止成功环境、保存可核验轨迹 |
| `run_flow_evaluation.sh` | 论文评估参数、零噪声对照、3 组评估种子、独立输出目录 |
| `verify_flow_evaluation.py` | 检查参数、样本数、有效动作、成功标记、clean/noisy 配对和零噪声轨迹一致性 |

本分支没有修改模型架构、Flow Matching 损失、噪声算法来追求更高分数。
评估入口相对上游有修改，具体差异可用 `git diff ab73e7c -- eval.py eval_utils.py` 查看，
不能把它描述成完全未改动的官方评估脚本。

## 2. 资源与环境

实测硬件为 **NVIDIA RTX 6000 Ada，48 GB**，每任务使用 1 张卡。
服务器 SSH 别名虽含 A6000，实际 GPU 型号以 `nvidia-smi` 为准。
四个任务可用 4 张空闲卡并行，也可以逐个运行。

本轮训练约耗时：TwoRoom 8 小时、PushT 19 小时、Reacher 23.5 小时、Cube 24.3 小时。
这些是在四任务并行的本机记录，不是通用性能保证。
batch=128 时，PushT 因冻结视觉模块显存约 2.7 GB，其余任务约 20 GB。
评估比训练省显存，但仍须保留环境渲染、数据加载和视频输出所需的 CPU、内存及存储资源。

实际使用已有的 Linux Conda 环境，名称为 `le-wm`。**环境名字不代表使用的是 LeWM 动力学模型。**
所有最终权重均核验为本仓库 `module.FlowTrajectoryPredictor`。

关键已安装版本如下，仅用于描述经过实验验证的环境，**不是经过全新安装验证的锁文件**：

| 组件 | 版本 |
|---|---|
| Python | 3.10 |
| PyTorch / torchvision | 2.6.0+cu124 / 0.21.0+cu124 |
| stable-worldmodel / stable-pretraining | 0.1.1 / 0.1.8 |
| lightning | 2.6.6 |
| numpy | 2.2.6 |
| h5py / hdf5plugin | 3.16.0 / 7.1.0 |
| gymnasium | 1.2.3 |
| mujoco / ogbench | 3.14.0 / 1.2.1 |
| pymunk | 7.3.0 |
| hydra-core / omegaconf | 1.3.7 / 2.3.1 |

从零搭建环境时应另外验证依赖兼容性，避免不加版本约束地升级整个训练环境。
尤其要检查 `stable-baselines3` 对 PyTorch 的要求，不能忽略 pip 的依赖冲突。
已有环境可以先做以下检查，再使用短测确认实际执行路径：

```bash
conda activate le-wm
python -m pip check
python -c 'import torch, stable_worldmodel, stable_pretraining, hdf5plugin; print(torch.__version__); print(torch.cuda.is_available())'
nvidia-smi
```

本轮默认关闭 W&B，使用本地日志及 CSV，不需要外网认证才能开始训练或评估。
原服务器还安装了环境级的原子 checkpoint 保存插件；该插件不属于本仓库，
本次未将整个 Conda 环境或机器级补丁打包上传。

## 3. 数据和视觉初始化

数据来源：[LeWM Hugging Face collection](https://huggingface.co/collections/quentinll/lewm)。
沿用四个官方 HDF5 数据集，没有重新生成训练数据。

**本分支的 `paper_*` 配置直接使用 `$STABLEWM_HOME` 下的文件**，与上游 README
举例的 `$STABLEWM_HOME/datasets/` 布局不同。按本分支脚本运行时，目录应为：

```text
$STABLEWM_HOME/
  tworoom.h5
  pusht_expert_train.h5
  reacher.h5
  cube_single_expert.h5
  checkpoints/
    pusht_paper_seed3072/weights_epoch_10.pt
    reacher_paper_seed3073/weights_epoch_10.pt
```

```bash
export STABLEWM_HOME="$HOME/.stable-wm"
export LOCAL_DATASET_DIR="$STABLEWM_HOME"
mkdir -p "$STABLEWM_HOME/checkpoints" "$STABLEWM_HOME/logs/flow-jepa"
ls -lh "$STABLEWM_HOME"/{tworoom,pusht_expert_train,reacher,cube_single_expert}.h5
```

TwoRoom/Cube 从头训练视觉模块。PushT/Reacher 需要先准备上述 LeWM 视觉初始化权重：

- PushT：LeWM 训练种子 3072、第 10 轮的 encoder 和 projector，加载后冻结。
- Reacher：LeWM 训练种子 3073、第 10 轮的 encoder 和 projector，加载后继续训练。
- Flow 预测器仍是本项目的模型，不能用完整 LeWM 动力学模型代替。
- 这些是本地复现的初始化权重，并不是作者发布且唯一标识的 checkpoint；数据与权重均未包含在 Git 仓库。

实测数据规模：TwoRoom、Reacher、Cube 各 10,000 条轨迹；PushT 为 18,685 条，平均约 125 帧。
PushT 与旧 LeWM 论文中 20,000 条、平均 196 步的描述不一致，但压缩包与官方发布哈希、解压后 H5 哈希已核对一致。
不能为凑齐论文描述自行补数据，或把这一差异当成全部任务失败的唯一原因。

## 4. 训练配置

共同设置：20 epochs、batch size 128、history 1、预测长度 5、frameskip 5、输入序列 6 帧、
224×224 图像、ViT-Tiny patch14、latent 192、AdamW lr=5e-5、weight decay=1e-3、`bf16-mixed`。
预测器使用 6 个 block、16 个 attention head、head dimension 64、MLP 2048。

| 任务 | 潜变量源 | future self-attention | 视觉初始化 | predictor dropout | SIGReg 权重 |
|---|---|---|---|---:|---:|
| TwoRoom | N(0,I) | 双向 | 从头训练 | 0.1 | 0.1 |
| Cube | N(0,I) | 双向 | 从头训练 | 0.1 | 0.1 |
| PushT | N(z_current,0.5²I) | 因果 | LeWM，冻结 | 0 | 0.09 |
| Reacher | N(z_current,0.5²I) | 因果 | LeWM，继续训练 | 0.1 | 0.1 |

调度器实际执行行为在原实验环境中已审计：每 optimizer step 更新，前 1% step 预热后余弦衰减。
不能仅凭 YAML 的 `interval` 字面值推断整个依赖框架中的实际步频。

### 原实验服务器运行方式

**两个 shell 启动器保留了实际实验路径，目前不是可直接用于任意机器的通用启动器。**
它们内部明确设置：

```text
STABLEWM_HOME=/home/wangtz/.stable-wm
python=/home/wangtz/anaconda3/envs/le-wm/bin/python
repository=/home/wangtz/Flow-JEPA
```

迁移机器时，需在 `run_flow_training.sh` 和 `run_flow_evaluation.sh` 中修改这三个位置；
只在外部 export 同名环境变量不会覆盖脚本内部的固定设置。GPU 编号检查当前接受 0–3。

在原服务器、新输出目录且卡空闲时，示例命令如下：

```bash
cd /home/wangtz/Flow-JEPA
bash run_flow_training.sh pusht   0 3073
bash run_flow_training.sh tworoom 1 3073
bash run_flow_training.sh reacher 2 3073
bash run_flow_training.sh cube    3 3073
```

上面的四行在同一个终端中会顺序执行。要并行，应分别放入独立终端或 tmux 会话，例如：

```bash
tmux new-session -d -s flow-cube-3073 'bash /home/wangtz/Flow-JEPA/run_flow_training.sh cube 3 3073'
tmux attach -t flow-cube-3073
```

本轮 3072/3073 输出已存在。脚本会拒绝已有实验目录、已有权重目录和忙碌 GPU，
防止覆盖或重复启动。新实验请选新的训练种子或先明确设计新的运行目录，不要删除原结果来绕过检查。

每次先执行配置核验，再用 3 个训练 batch、1 个验证 batch 做短测，确认模型重载和冻结行为，
然后才进入正式 20 轮。第 20 轮权重和配置位于：

```text
$STABLEWM_HOME/checkpoints/flow_pusht_seed3073_init3072/
$STABLEWM_HOME/checkpoints/flow_tworoom_seed3073/
$STABLEWM_HOME/checkpoints/flow_reacher_seed3073_init3073/
$STABLEWM_HOME/checkpoints/flow_cube_seed3073/
```

模型目录中的 `config.yaml` / `config.json` 应与权重一起保留，不要只移动一个 `.pt` 文件后假设能独立加载。

### 其他机器的直接 Python 示例

在准备好同版本依赖、数据和初始化权重后，可以不使用固定路径启动器：

```bash
export STABLEWM_HOME="$HOME/.stable-wm"
export LOCAL_DATASET_DIR="$STABLEWM_HOME"
export CUDA_VISIBLE_DEVICES=0
python verify_flow_run.py --task tworoom --seed 3074 --output /tmp/tworoom3074_profile.json
python train.py --config-name=paper_tworoom seed=3074
```

这条直接训练命令不会自动执行启动器的空闲 GPU 检查、源码快照和短测流程；正式实验建议适配并使用启动器。

## 5. 闭环评估

评估使用训练完成的第 20 轮权重，**训练种子与评估种子分开管理**：
训练种子 3072/3073 影响初始化、数据划分及样本顺序；评估种子 42/43/44 影响起点抽样、规划和噪声。

| 项目 | 设置 |
|---|---|
| CEM | 10 iterations，300 samples，30 elites，初始方差 1 |
| 规划 | 5 个动作块，每块 5 个底层动作，执行完 25 步再重新规划 |
| 代价 | 仅终点预测潜变量与目标潜变量的代价 |
| Flow | 8 步 Euler |
| 目标 / 预算 | 数据同轨迹 25 步后的目标，最多执行 50 个底层动作 |
| 样本 | 每评估种子 50 例，分别 clean/noisy |
| RGB 扰动 | 标准差 s=100；PushT r=10，其余 r=35；clip 到 [0,255] |

按作者公开代码，每条 rollout 固定同一噪声图，作用于当前观测和目标图。
图像噪声不是训练潜变量源的噪声，两者的标准差不可混用。

```bash
cd /home/wangtz/Flow-JEPA
# 参数依次为：任务、GPU、输出版本、训练种子
bash run_flow_evaluation.sh pusht   1 v2 3073
bash run_flow_evaluation.sh reacher 2 v2 3073
bash run_flow_evaluation.sh cube    3 v2 3073
bash run_flow_evaluation.sh tworoom 1 v2 3073
```

示例同样按顺序执行；并行需独立会话及不同空闲 GPU。默认训练种子仍是 3072，评估新模型时必须显式传第四个参数。
现有 v2/3073 结果不可覆盖。如确需独立重复评估，可使用新的版本号（例如 v3），并说明重复运行原因。

脚本先用独立评估种子 9001、每组 2 例执行 clean、zero-noise、noisy 短测。
zero-noise 必须与 clean 的成功轨迹和动作一致；这些短测不计入正式成功率。
随后运行 42/43/44 的 6 组正式评估，每任务 300 次 rollout。

Reacher 保留本次数据集对应的棋盘格渲染设置 `reacher_grid_texuniform=false`；
Reacher/Cube 启动时核验 qpos/qvel 与数据一致。成功后停止相应环境，
检查有效环境动作有限，不将已停止环境的 NaN 占位动作误判成策略异常。

默认输出：

```text
$STABLEWM_HOME/evaluations/flow_<task>_train3073_epoch20_paper_v2/
  source/
  environment.txt
  source_sha256.txt
  smoke_check.json
  seed42_check.json
  seed43_check.json
  seed44_check.json
  seed42_clean/
    evaluation_manifest.json
    metrics.json
    initialization.npz
    rollout.npz
  seed42_noisy/
    perturbation.npz
    ...
```

`metrics.json` 包含逐例成功状态和首次成功步数；manifest 保存解析配置、起点和权重哈希；
`rollout.npz` 保存动作、有效环境及成功轨迹。seed42 保存每例视频，seed43/44 默认不保存视频。

## 6. 实测结果

两个训练种子的四任务均正常完成 20 轮，最终权重有限，日志未发现损失 NaN/Inf。
每个模型每种条件合并 3×50 例；论文是每种条件 50 例，不是相同的原始测试集。

| 任务 | seed3072 clean | seed3073 clean | seed3072 noisy | seed3073 noisy | 论文 clean / noisy |
|---|---:|---:|---:|---:|---:|
| TwoRoom | 96.00% | 96.00% | 60.67% | 60.00% | 100% / 96% |
| PushT | 83.33% | 82.67% | 45.33% | 45.33% | 96% / 90% |
| Reacher | 78.67% | 78.00% | 75.33% | 75.33% | 90% / 90% |
| Cube | 76.67% | 79.33% | 54.67% | 49.33% | 82% / 66% |

第二轮逐评估种子成绩，每项 clean/noisy 各 50 例：

| 任务 | eval42 clean / noisy | eval43 clean / noisy | eval44 clean / noisy |
|---|---:|---:|---:|
| TwoRoom | 96% / 60% | 96% / 62% | 96% / 58% |
| PushT | 90% / 44% | 82% / 48% | 76% / 44% |
| Reacher | 80% / 80% | 70% / 68% | 84% / 78% |
| Cube | 82% / 48% | 86% / 56% | 70% / 44% |

逐起点配对后，第二轮相比第一轮：

- PushT：clean 净少成功 1 例；noisy 有 8 例改善、8 例退步，74 例两轮都失败。
- Reacher：clean 净少成功 1 例；noisy 有 25 例改善、25 例退步，总分持平不等于行为完全相同。
- Cube：clean 净多成功 4 例；noisy 有 8 例改善、16 例退步，净少成功 8 例。
- TwoRoom：clean 持平；noisy 净少成功 1 例。

这只比较两个训练种子，不是充分的训练方差研究，也不据此宣称统计显著性。
验证集随训练种子改变，验证损失不能当成同一 held-out 集上的严格比较；
Cube 预测损失略低却 noisy 更差，也说明训练损失不等同于闭环鲁棒性。

## 7. 已核验内容与未解决问题

已核验：

1. 两轮运行的是本仓库 Flow 预测器，非 LeWM 动力学模型；源码与训练快照对应。
2. 依赖版本、解析后的评估配置一致；新旧轮的起点、目标和环境 reset seeds 一致。
3. 两轮 noisy 数组和中心逐元素一致；clean 与 zero-noise 短测一致。
4. 训练权重有限，正式评估的数量、成功标记和有效动作检查通过。
5. TwoRoom 的 150 个起点渲染与 H5 逐像素相等；各重放 25 步专家动作，位置误差为 0。
   最后一项没有推广声称为其余三个环境的完整动力学重放核验。

尚未解决：

- 作者没有公开完整环境锁定版本、精确测试起点及所选视觉初始化权重的唯一身份。
- PushT/Reacher 使用本地 LeWM 初始化；PushT 视觉模块冻结，单换 Flow 种子不会改变它的编码器。
- 原始数据描述差异、视觉表征对噪声的敏感性、预测与规划误差仍需分别研究。
- 已有诊断发现部分噪声失败案例的潜空间目标方向变化更大，但这是相关性证据，不能确定唯一因果机制。
- 公开噪声中心采样包含启发式和回退逻辑，不保证移动物体全程远离固定扰动中心。本次沿用公开行为，没有为提高分数调整它。

因此不应把这份分支或 `paper_*` 命名理解成已经达到论文结果。

## 8. 版本与产物管理

本仓库上传源码、配置、启动器、检查脚本和本说明；不上传数据、checkpoint、原始运行日志、
视频结果、噪声数组、API key、SSH 私钥或 W&B 凭据。`.gitignore` 已增加常见实验产物规则。

实际实验仍保留在服务器的 `experiments/`、`checkpoints/`、`evaluations/` 目录。
训练和评估脚本会保存源码副本、Git 基础提交、工作区 diff、SHA-256 和 `pip freeze`。
本分支也保留一份 [已完成实验源码哈希](docs/experiment-source-sha256.json)，可用于确认上传的核心文件与运行版本对应。

重新运行后，应保留配置、权重、源码快照和原始指标，再汇总成绩；
不能只记录 Git 提交号，因为旧实验运行时包含未提交的复现修改。
