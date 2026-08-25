# Phase A 实验说明与预注册协议

## 1. 固定问题

训练范围固定为 `all_runs`；外层评估固定为 A/D/J/M 四折 leave-one-subject-out；每折固定 seed 1、2、42。A0 直接引用已有 `001484412812/M2-Direct` 结果。所有条件必须使用相同 sample 集、相同 run 内实际历史顺序、相同 node/Tier3/stage 映射。禁止 graph-valid shuffle，禁止 validation/test 驱动的 best epoch 或超参数选择。

参与者优先汇总：每个 fold×seed 先独立算指标，再对 4 participant × 3 seed 等权平均。因此样本较多的 J 不会在主表中获得额外权重。另保留逐 clip 预测用于配对统计。

## 2. 统一输入与 M2 主干

每个摄像头先独立训练一个 31-Tier3 ResNet3D-18 backbone，并从最终第 100 epoch checkpoint 提取确定性的 512-D clip 特征。backbone 的 train/test manifest、训练超参和 seed 与 A0 一致，且每个 camera/fold/seed 独立训练，避免把第二视角建立在不公平的预训练来源上。

右手 `mindrove.pt` 的原始字段为 `right_emg [L,8]`、`right_acc [L,3]`、`right_gyro [L,3]`。IMU 在通道维拼成 `[L,6]`。每个 clip 先线性重采样为 EMG `[8,512]`、IMU `[6,256]`，再用该 LOSO 折 train clip 的逐通道 mean/std 标准化。这个长度归一化适用于 Phase A 的已分割 clip 分类；后续实时流不能按整段拉伸，需改为固定采样率滑窗。

融合后，每个当前 clip 和历史 clip 都变成一个 512-D observation。其后完全沿用 M2-Direct：

1. 当前 observation 经 `Linear(512→256)+LayerNorm` 成为单 query。
2. 每个历史 observation 经独立 `Linear(512→256)+LayerNorm`，加“距当前 clip 的反向距离”position embedding；1 表示最近历史。
3. 在历史前添加一个可学习 null token，单 query 对 null+全部历史做 4-head attention。
4. 将当前 512-D observation 与 256-D history context 拼接，经 `Linear(768→512)`；该层初始化为 `[I, 0]`，初始时忽略历史。
5. 512-D fused feature 进入可训练 35-node linear head。
6. 35-node softmax 概率按固定 node→Tier3 映射相加，得到 31-Tier3 概率。

## 3. A0-A7 forward

### A0：主相机 M2-Direct（冻结基线）

`001484412812 clip → 已有该 fold/seed 的 RGB backbone → 512-D → M2 历史注意力 → 35-node logits → 31-Tier3`

A0 不重训。直接使用已有 12 次运行的 checkpoint、三分区指标、逐 clip CSV 和 node probability `.pt`，防止基线漂移。

### A1：第二相机单独 M2-Direct

`001528512812 clip → 独立同协议 RGB backbone → 512-D → 同结构 M2 → node/Tier3`

A1 同时回答第二视角自身质量，并为 A2 提供独立概率。默认第二相机是 `001528512812`，但这是正式开跑前仍需确认的一项预注册选择。

### A2：双相机 late probability fusion

`A0 node softmax` 与 `A1 node softmax` 按 sample_name 严格对齐，然后固定：

`p_node = 0.5 × p_A0 + 0.5 × p_A1`

不新增参数、不在 test 上寻找权重。先对 node 概率融合，再聚合 Tier3；不能平均两个 argmax，也不能先聚合 Tier3 后反推 node。

### A3：双相机 gated residual/cross-view

1. 主/第二相机的 512-D 特征分别投影到 256-D，加 view embedding。
2. 主视角 token 作为 query，对“主+第二视角”两个 token 做 4-head cross-view attention；第二相机缺失时其 token 被 mask。
3. context 投影回 512-D，门控网络读取 `[主特征, context, available_flag]`，生成逐维 gate。
4. `observation = primary + gate × delta(context)`；delta 末层零初始化、gate bias=-3，故初始 observation 严格等于 primary。
5. 当前和所有历史 clip 都做相同 cross-view 融合，再进入统一 M2。

A3 的 M2 attention、history fusion 和 node head 从同 fold×seed 的 A0 checkpoint 精确加载并冻结；只训练 cross-view adapter。这样“第二相机缺失”路径在数值上严格等于 A0，而不是仅依赖正则化学会回退。

### A4：主相机 + 右手 IMU

`right_acc+right_gyro [6,256] → dilated Conv1d encoder → 256-D IMU context → gated residual into primary 512-D → M2`

IMU branch 主要表达运动方向、速度变化和相位。available flag 参与 gate；缺失时残差精确为零。

### A5：主相机 + 右手 EMG

`right_emg [8,512] → dilated Conv1d encoder → 256-D EMG context → gated residual into primary 512-D → M2`

EMG branch 主要表达肌肉激活与接触前后的边界信号。结构与 A4 独立，不共享 normalization 或时序编码器。

### A6：主相机 + 右手 EMG + IMU

`primary → EMG gated residual → IMU gated residual → M2`

两支 sensor encoder 和 gate 独立训练，用于测试可穿戴内部互补性。训练时各新模态独立以 0.2 概率 dropout，避免模型把某一传感器当作不可缺失条件。

### A7：双相机 + 右手 EMG + IMU

`primary → cross-view gated residual → EMG gated residual → IMU gated residual → M2`

融合顺序固定，不做排列搜索。所有新 branch 采用 zero-residual 初始化；训练模态 dropout 后，任一或全部新增模态缺失时仍有显式 primary-only 路径。

A4-A7 与 A3 相同，均精确加载并冻结对应 A0 的 M2/history/head，只训练新增的 observation adapters 和 signal encoders。这是 Phase A 的“增量信息隔离”设计：新增模态必须通过改变 512-D observation 给既有 A0 决策器带来收益。若 Phase A 通过，下一阶段才比较联合微调是否还有额外收益。

## 4. 训练公平性

- A1/A3-A7 均训练 50 epochs，不用 validation、不 early stop、无学习率 scheduler；AdamW，恒定 lr=1e-3，weight decay=1e-4，与 A0 的 M2 训练设置一致。
- 物理 batch=16，梯度累积 4 次，effective batch=64；这是因为 history 中每个 clip 都需运行信号 encoder。
- 主相机与第二相机 backbone 均冻结。A1 从头训练第二相机自己的 M2/head；A3-A7 冻结从同 fold×seed A0 精确加载的 M2/history/head，只训练 observation fusion 与 sensor encoder。
- `action_loss_weight=0`，与 A0 一致，仅以 35-node CE 训练。
- 同一 fold 的 train-only signal stats 在三个 seed 间共享；不同 fold 绝不共享 stats。
- 训练时 EMG 与 IMU 各自以 0.3 概率施加均匀分布于 ±10% 归一化 clip 长度的零填充时间位移；测试集不随机增强。这是预注册的 offset 鲁棒性训练，不根据压力测试结果调参。
- 每次运行保留最终 epoch checkpoint，不选最好 seed。
- A3-A7 在训练前后都做 primary-only logit 等价性断言（max absolute error ≤1e-5）；断言失败即停止该任务。

## 5. 强制输出

每个 condition/fold/seed 的 `test_results` 必须包含：

- `test_all|normal|fault_metrics.json`：node/Tier3 accuracy、macro-F1、macro-recall、逐类 precision/recall/F1/confusion matrix，以及 Stage 1/2/3 分层。
- `test_all|normal|fault_predictions.csv`：逐 clip node/Tier3 truth、prediction、confidence。
- `test_all|normal|fault_probabilities.pt`：完整 35-node 概率与行元数据。

汇总工具生成总体、Normal/Fault、Stage、31 Tier3、35 node、每条件前 12 个 node 与 Tier3 混淆对和 paired bootstrap CI。

## 6. Paired clip bootstrap

对每个 A1-A7 与 A0 比较。bootstrap 的抽样单位是唯一 clip，并在 participant 内分层有放回采样；同一被抽中的 clip 同时展开其 seed 1/2/42 的配对预测，从而不把三个 seed 当作三个独立 clip。默认 10,000 次，报告总体 Node accuracy/macro-F1/最弱支持类别 Recall、Tier3 accuracy/macro-F1、Normal/Fault Node 指标和 Stage 1/2/3 Node macro-F1 的 delta、95% percentile CI 与 `P(delta>0)`。

## 7. 增量价值门槛

以下规则需同时满足；任何单一最好 seed 都不构成证据：

1. 12 个 fold×seed 中至少 7 个 Node Macro-F1 为正增益。
2. 至少 7 个 fold×seed 的“该折有支持的最弱 node Recall”改善，并且至少 7 个 fold×seed 中这两项在同一次运行里同时改善。
3. Fault Node Macro-F1 非劣：预注册暂定允许均值最多下降 0.5 个百分点，并结合 bootstrap CI 判断。
4. 缺失模态时 Node Macro-F1 回退到距离 A0 不超过 1.0 个百分点；A7 必查全部新增模态同时缺失。
5. 时间偏差压力：EMG/IMU 分别按 clip 归一化长度作 ±5%、±10%、±20% 零填充位移；性能退化曲线必须报告。
6. 延迟和吞吐达到目标硬件预算。

第 3/4 条的 0.5/1.0 pp 是包内暂定 margin；如果要修改，必须在训练前改配置并记录。第 6 条目前不能判定，因为目标硬件和预算尚未提供。

## 8. 对后续实时动作分割的接口

Phase A 只回答“新增模态是否具有 clip 级增量信息”。进入实时分割时，建议保持两层结构：

1. 各模态独立按真实时间戳形成固定时长滑窗并编码，保留 availability、age、clock offset 和质量分数；不要把不同时长动作统一拉伸。
2. causal gated residual 融合产生每个时刻 observation；M2 只读取已经发生的历史，不看未来。
3. 同时输出 node posterior 与 boundary/contact evidence。EMG 对 onset/contact，IMU 对运动相位，第二视角对遮挡互补；边界 head 与 node head共享 observation，但分别校准。
4. 在线 decoder 使用持续时间、hysteresis 和 task constraints，且在模态失联时明确退回 A0 路径。

只有通过 Phase A 增量价值、压力与完整端到端延迟门槛的模态，才进入实时分割版本。

## 9. 当前阻塞/待确认信息

1. **第二相机最终选择**：当前默认 `001528512812`；数据还包含完整的 `001431512812`。正式大规模启动前应确认，或先只跑两个候选 A1 的低成本预筛，但预筛规则必须固定且不能看 A3-A7。
2. **目标硬件与预算**：需要 GPU/CPU 型号、单 clip/滑窗 p95 延迟上限、最低吞吐、允许 batch size；否则延迟只能测量，不能 pass/fail。
3. **RGB 与 MindRove 的独立时间戳证据**：manifest 有左右 MindRove board start/end timestamp，但没有三相机 clip 的 frame timestamp。现有分割 clip 可按 sample 对齐，无法仅凭 manifest 复核跨设备绝对时偏。实时融合前需要相机时间戳/同步日志或可复现的 offset 标定。
4. **全张量数值审计**：当前环境没有带 PyTorch 的训练解释器，因此本次只完成文件和 shape 元数据审计。训练环境必须先运行 `audit_dataset.py --load-tensors`，检查所有右手 tensor 的 key、shape、NaN/Inf。
5. **实时 ground truth/评价容差**：Phase A 不需要；后续边界 F1 需要明确 onset/offset 容差和在线 latency 的计时边界。
