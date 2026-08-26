# A3 与右手 EMG/IMU 补充实验协议

## 1. 实验范围

本轮只补充两部分：

1. 运行原 Phase A 中尚未完成的 A3（双相机 gated residual/cross-view）；
2. 新增 S1–S12，比较右手 EMG/IMU 的 ResNet10-1D 与 dilated Conv1d，并分别评价 Direct Node、Direct Tier3 和 sensor-only M2-Direct。

不新增、也不运行“两个单摄像头 scratch M2-Direct”。A0 仍是已有的主相机 M2-Direct 基线；A1 仍是已有第二相机结果。原 A0–A7 的模型、数据和训练实现不修改。

所有实验沿用同一 `all_runs` LOSO manifest、A/D/J/M 四折和 seed 1/2/42。信号只使用右手：EMG 为 8×512，IMU 为右手 accelerometer 3 通道加 gyroscope 3 通道，即 6×256。

## 2. 条件总表

| ID | 输入 | 编码器 | 训练目标 | 是否使用 M2 | 初始化 |
|---|---|---|---|---|---|
| A3 | 两个摄像头 | 原两路 RGB backbone 特征 | 35 node | 是，cross-view gated residual | 沿用原 A3 协议 |
| S1 | 右手 EMG | S9 的冻结 ResNet10-1D | 35 node | 是 | M2 与 node head 从头训练 |
| S2 | 右手 EMG | S10 的冻结 Dilated-1D | 35 node | 是 | M2 与 node head 从头训练 |
| S3 | 右手 IMU | S11 的冻结 ResNet10-1D | 35 node | 是 | M2 与 node head 从头训练 |
| S4 | 右手 IMU | S12 的冻结 Dilated-1D | 35 node | 是 | M2 与 node head 从头训练 |
| S5 | 右手 EMG | ResNet10-1D | 35 node | 否 | 编码器与 head 独立从头训练 |
| S6 | 右手 EMG | Dilated-1D | 35 node | 否 | 编码器与 head 独立从头训练 |
| S7 | 右手 IMU | ResNet10-1D | 35 node | 否 | 编码器与 head 独立从头训练 |
| S8 | 右手 IMU | Dilated-1D | 35 node | 否 | 编码器与 head 独立从头训练 |
| S9 | 右手 EMG | ResNet10-1D | 31 Tier3 | 否 | 编码器与 head 独立从头训练 |
| S10 | 右手 EMG | Dilated-1D | 31 Tier3 | 否 | 编码器与 head 独立从头训练 |
| S11 | 右手 IMU | ResNet10-1D | 31 Tier3 | 否 | 编码器与 head 独立从头训练 |
| S12 | 右手 IMU | Dilated-1D | 31 Tier3 | 否 | 编码器与 head 独立从头训练 |

S5–S8 与 S9–S12 是相互独立的完整训练。比如 S5 不加载 S9 的编码器；这样 Direct Node 与 Direct Tier3 的比较不会混入预训练迁移效应。

## 3. 编码器

### 3.1 ResNet10-1D

实现复用目标二 `renet1d_my.py` 的 ResNet10 拓扑：stem Conv1d、四个 residual stage，block 数为 `[1,1,1,1]`，stage 通道为 64/128/256/512；每次下采样由 stride=2 完成，最后全局平均池化得到 512-D clip feature。本实验把输出分类器移到外部，使同一 backbone 可接 31-Tier3 或 35-node head。

### 3.2 Dilated Conv1d

实现调用本 Phase A 已有的 `TemporalSignalEncoder`，其三个 temporal block 使用 dilation 1/2/4，在较少下采样的情况下扩大时间感受野。原输出为 256-D，再经 Linear(256,512) 与 LayerNorm 对齐到 512-D。这样两种 backbone 进入分类头或 M2 时拥有相同特征维度。

## 4. 各实验的 forward 流程

### A3：双相机 gated residual/cross-view

1. 主相机与第二相机各自生成当前 clip 的 512-D RGB feature；
2. 原 M2 用主相机当前特征和同一 run 内之前 clip 的主相机特征形成 baseline fused representation；
3. cross-view adapter 对两路当前特征做投影和交互；
4. gate 根据当前跨视角信息预测 residual 权重；
5. `baseline representation + gated cross-view residual` 送入 35-node head；
6. node 概率按固定映射求和得到 Tier3 概率。

A3 的准确实现仍位于原 `phase_a/models.py` 和 `tools/train_condition.py`，本轮只通过统一脚本调度，不改其数值路径。

### S5–S8：Direct Node

1. 从 leakage-free cache 读取并按 train-fold 统计量标准化右手信号；
2. 信号进入 ResNet10-1D 或 Dilated-1D，得到 512-D 当前 clip feature；
3. 单一 Linear(512,35) head 输出 node logits；
4. 以真实 node 做交叉熵训练；
5. 评估时将 35-node softmax 概率按固定 node→Tier3 映射求和，附带报告 31-Tier3 指标。

Direct Node 的编码器和 head 一起从随机初始化训练，不读取 A0、S1–S4 或 S9–S12 权重。

### S9–S12：Direct Tier3

1. 标准化右手信号进入对应 1D encoder；
2. encoder 输出 512-D feature；
3. 独立 Linear(512,31) head 输出 Tier3 logits；
4. 以真实 Tier3 做交叉熵训练。

这些模型没有 node head，因此只能产生 Tier3 结果，不能从 31 类唯一还原 35 node。它们同时充当 S1–S4 的上游特征学习任务。

### S1–S4：sensor-only M2-Direct

这是严格的两阶段流程：

**阶段一：Tier3 signal encoder**

1. 先独立训练 S9/S10/S11/S12；
2. 训练目标仅为 31-Tier3；
3. 取训练完成的 encoder，冻结其全部参数；
4. 对 train/test clip 提取干净的 512-D 特征 cache，cache 记录源 checkpoint 和 manifest hash。

**阶段二：M2 + node head**

1. 当前 clip 的冻结 512-D signal feature 作为 `current`；
2. 只在同一 run 内收集之前 clip 的冻结 signal features 作为 `history`，不跨 run、不读取未来 clip；
3. current projection 生成 query，history projection 生成 key/value；
4. masked attention 对可用历史做加权池化；首个 clip 没有历史时使用零历史向量；
5. `[current, attended_history]` 经 fusion MLP，残差加回 current projection；
6. Linear head 输出 35-node logits；
7. M2、投影、attention、fusion 和 node head 全部随机初始化并从头训练，冻结的 signal encoder 不更新；
8. 训练目标为真实 node 的交叉熵，Tier3 由 node 概率聚合得到。

这里“sensor-only”表示整个 forward 不读取任何摄像头特征。“不加载 A0”不仅指不加载 A0 head，也包括不加载 A0 的 M2、history attention 或 fusion 参数。

### 逐条件解释

- **S1**：右手 EMG 经 S9 ResNet10-1D Tier3 训练得到 encoder；冻结 512-D EMG 特征后，从头训练历史 M2 与 35-node head。它与 S5 的差异主要是加入历史建模，与 S2 的差异主要是 1D encoder。
- **S2**：右手 EMG 经 S10 Dilated-1D Tier3 训练得到 encoder，再进行冻结特征 M2/node 训练。它检验较密集、扩张卷积时间建模是否改善 EMG 的跨 clip 历史表示。
- **S3**：右手 IMU 经 S11 ResNet10-1D Tier3 训练得到 encoder，再以 IMU 历史训练 scratch M2/node。它检验动作方向与相位信息在历史级是否足以识别 node。
- **S4**：右手 IMU 经 S12 Dilated-1D Tier3 训练得到 encoder，再训练 scratch M2/node。它是 IMU sensor-only M2 的主要 dilated 候选。
- **S5**：右手 EMG ResNet10-1D 与 35-node head 端到端从头训练，只看当前 clip。它不加载 S9，是 S1 的 current-only 对照。
- **S6**：右手 EMG Dilated-1D 与 35-node head 端到端从头训练，只看当前 clip。它同时与 S5 比较 encoder，与 S2 比较历史 M2 的增量。
- **S7**：右手 IMU ResNet10-1D 与 35-node head 端到端从头训练，只看当前 clip。它不加载 S11，是 S3 的 current-only 对照。
- **S8**：右手 IMU Dilated-1D 与 35-node head 端到端从头训练，只看当前 clip。它同时与 S7 比较 encoder，与 S4 比较历史 M2 的增量。
- **S9**：右手 EMG ResNet10-1D 与 31-Tier3 head 端到端从头训练；既报告独立 Tier3 能力，也是 S1 唯一允许的上游 encoder。
- **S10**：右手 EMG Dilated-1D 与 31-Tier3 head 端到端从头训练；既与 S9 比较 dilated，也为 S2 提供冻结 encoder。
- **S11**：右手 IMU ResNet10-1D 与 31-Tier3 head 端到端从头训练；既报告独立 Tier3 能力，也为 S3 提供冻结 encoder。
- **S12**：右手 IMU Dilated-1D 与 31-Tier3 head 端到端从头训练；既与 S11 比较 dilated，也为 S4 提供冻结 encoder。

## 5. 训练与公平性约束

- 所有信号 normalization 只使用该 LOSO fold 的 train manifest；
- test_normal/test_fault 只用于报告；
- 默认 direct encoder 训练 50 epochs、batch 16、effective batch 64、AdamW、学习率 1e-3、weight decay 1e-4；具体值只由 `config/supplementary_experiments.json` 控制；
- S1–S4 的 M2 训练超参数沿用 `config/phase_a.json` 中现有 M2 训练设置；
- train 可做预注册的轻微时间平移增强，test 不增强；
- 不使用 test fold 选择 backbone、epoch、gate 或阈值；
- 每个 condition/fold/seed 都写独立 checkpoint、逐样本 predictions、metrics 和完成标记。

## 6. 计划比较

| 比较 | 回答的问题 |
|---|---|
| S1 vs S5；S2 vs S6；S3 vs S7；S4 vs S8 | 同一模态和 backbone 下，历史 M2 是否优于当前 clip direct node |
| S2 vs S1；S4 vs S3 | M2 路径中 dilated 是否优于 ResNet10 |
| S6 vs S5；S8 vs S7 | Direct Node 路径中 dilated 是否优于 ResNet10 |
| S10 vs S9；S12 vs S11 | Direct Tier3 路径中 dilated 是否优于 ResNet10 |

paired bootstrap 以同一 participant/clip 为配对单位，在 participant 内重采样 clip，然后把同一 clip 的三个 seed 作为重复测量展开；默认 10,000 次。只有完整四折三 seed 时统一脚本才会启动 bootstrap。

同时生成 12 个 fold×seed 的逐次 delta 和门槛文件：多数正增益至少为 7/12；Macro-F1 与最弱类别 Recall 分开计数并报告共同改善次数；Fault Macro-F1 使用原 Phase A 的 0.5 percentage-point 非劣界限。

## 7. 输出与判定

S1–S8 输出总体、Normal/Fault、Stage、31 Tier3、35 node、当前前 12 个混淆对、低 Recall 类别及其误分类样本名。S9–S12 输出总体、Normal/Fault、Stage 和 31 Tier3；其 35-node 列明确标记为不适用，而不是伪造 node 结果。

压力测试包括 clean、信号置零和配置中的正/负时间偏移。对 sensor-only 模型，置零并不能“回退到 A0”，因为 forward 中没有摄像头分支；这里测量的是性能退化和对齐敏感度。A3 等摄像头融合模型仍按原 Phase A 的 A0 回退标准判断。

延迟文件 `latency_end_to_end_signal_scope.json` 覆盖 signal encoder 与 classifier；S1–S4 还包含历史 M2/node head，历史特征视为已缓存。它不包含传感器硬件采样等待时间。

## 8. 代码定位

- 统一配置：`config/supplementary_experiments.json`
- 统一入口：`scripts/run_all_phase_a.ps1`
- ResNet10/Dilated backbone：`phase_a/signal_backbones.py`
- Direct 与 sensor-only M2：`phase_a/supplementary_models.py`
- signal/feature-history dataset：`phase_a/sensor_data.py`
- Direct 训练：`tools/train_signal_direct.py`
- 冻结特征提取：`tools/extract_signal_features.py`
- sensor-only M2 训练：`tools/train_sensor_m2.py`
- 压力、延迟、bootstrap、汇总：`tools/run_supplementary_stress.py`、`tools/benchmark_supplementary_latency.py`、`tools/paired_bootstrap_supplementary.py`、`tools/summarize_supplementary.py`
