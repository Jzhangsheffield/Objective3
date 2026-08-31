# B0–B5 详细实验协议

## 1. 科学问题与固定范围

主要问题是：在 A0 主相机 M2 已很强的条件下，cam1 与右手 IMU 的互补信息能否通过同级融合稳定提升跨参与者泛化？本阶段不再测试 EMG、双手信号或 participant-specific test calibration，以控制变量。

外层协议固定为 A/D/J/M LOSO；每个 fold 使用 seed 1/2/42。训练使用 `all_runs`。正式评价 split 为 `test_all`、`test_normal`、`test_fault`。任何模型、温度、权重、门控阈值或 early selection 都不得根据外层测试标签选择。

## 2. 数据与编码器

### 2.1 摄像头

- cam0：`001484412812`，复用已完成的外层 3D ResNet-18 与 A0 M2。
- cam1：`001528512812`，B0 为每个外层 fold/seed 独立训练 3D ResNet-18。
- 输入：16 帧，224×224。
- B3–B5 缓存 `layer2` 空间平均后的 `[4,128]` temporal tokens，并同时缓存 `[512]` global feature。
- 两路摄像头不共享 backbone 参数。

### 2.2 IMU

- 仅右手 `right_acc + right_gyro`，6 通道。
- 每个 clip 线性重采样到长度 256。
- normalization 只使用对应训练 partition 的 channel mean/std；内层与外层各自重新统计。
- encoder 为 scratch 1D ResNet10；长度 256 输出 `[8,512]` temporal tokens 与 `[512]` global feature。
- 训练时以 0.3 概率施加最多 ±10% 的 zero-padded time shift。

## 3. 条件定义

### B0：强两视角参考线

补齐 Phase A A1 与 A2。cam0 与 cam1 各自保留独立 M2-direct；B0 的正式预测为两者 35-node 概率的 0.5/0.5 平均。B0 既验证小范围 A2 增益是否跨 fold/seed 稳定，也为 B1/B2 提供 cam1 外层专家。

### B1：温度校准静态三专家融合

三个专家为 cam0 M2、cam1 M2、IMU Direct Node。对每个专家学习一个正温度 `T_m`，再学习 simplex 权重 `w_m≥0, Σw_m=1`：

`p = softmax(Σ_m w_m log(clamp(p_m))/T_m)`。

优化目标是 OOF node cross-entropy，并对权重偏离均匀分布施加 0.001 L2。Adam，1500 steps，lr 0.03。

### B2：质量感知动态三专家融合

先使用 B1 的温度。每个 clip 的 gate 输入共 15 维：每个专家的归一化 entropy、top-1 probability、top1-top2 margin（9维），三对 JS divergence（3维），三路 availability（3维）。两层 MLP hidden=16，dropout=0.1，输出 masked softmax 权重。AdamW，固定 1200 个 optimizer steps，batch 128，lr 0.001，weight decay 0.001；用轻量权重熵正则减少单专家坍缩。固定 step 数使不同 outer fold 在样本量略有差异时保持相同的优化强度，并避免原 300 epochs 配置带来的过训练风险。

### B3：当前 clip 的对称三流 token fusion

三路 global token 与 temporal tokens 先独立投影到 `d_model=256` 并加入 modality embedding。4 个可学习 bottleneck tokens 通过 2 层 multi-head attention 从全部模态读取信息；没有任何一路被定义为 primary。分类损失为 joint node CE 加 0.2×三个辅助 unimodal node CE 的平均。训练期间每路以 0.2 概率 modality dropout，且至少保留一路。

### B4：B3 + actual M2 history

每个当前/历史 clip 先经过同一三流 observation fusion 得到 512-D 表示。当前表示作为 query，按实际 run 顺序读取此前 clip 的融合表示；最大历史长度 35。历史不跨 run，不使用未来 clip。

### B5：B4 + soft alignment + contrastive

在 bottleneck fusion 前，8 个双相机 temporal tokens（cam0 4 + cam1 4）与 8 个 IMU tokens做双向 multi-head cross-attention；这允许网络吸收小幅不同步，而不采用固定硬对齐。另加 0.05×对称 InfoNCE，temperature 0.1，使同 clip 的 camera/IMU embedding 接近。其余训练项与 B4 一致。

## 4. B1/B2 的严格 inner-LOSO

对外层 held-out participant `o`，外层训练参与者还有 3 人。依次选其中一人为 inner held-out `i`，在剩余 2 人上从头训练 cam0、cam1 和 IMU 专家，再只对 `i` 生成预测。合并三个 `i` 的预测后，每个外层训练 clip 恰好有一次真正 OOF 预测。

禁止将外层 cam0/cam1 backbone 用于 cross-fit，因为这些 backbone 训练时已看到 inner held-out participant。即使只重新训练融合 head，也仍会通过 backbone 表示泄漏 inner 标签分布。因此配置固定 `forbid_outer_backbone_reuse=true`。

每个 outer×seed 只拟合一个 B1 和一个 B2；随后冻结并一次性应用于真正外层测试集。外层测试集不参与温度、权重、gate 或模型选择。

## 5. B3–B5 训练配置

- encoder：外层训练 partition 上已训练的 cam0/cam1/IMU encoder；通过 cache 冻结。
- epochs 50；直接 batch 32；effective batch 32；不使用 gradient accumulation。
- AdamW，lr 3e-4，weight decay 1e-4；gradient clip 1.0。
- 不使用 validation/early stopping，不选择最佳 epoch，保存 final epoch。
- B3/B4/B5 参数量不同属于方法差异；共同的编码器输入和训练 split 保持一致。

## 6. 评价与比较

主指标为 35-node Macro-F1；同时报告 accuracy、balanced accuracy、weakest present-class recall、31-Tier3 Macro-F1/accuracy、Stage 1/2/3、Normal 和 Fault。12 个 fold×seed 作为预注册重复单位，不将 431 个 clip 当作独立实验重复。

核心配对比较：

1. B0 vs A0：确认两摄像头后融合能否稳定复现。
2. B1 vs B0：IMU 加入静态校准融合是否有增量。
3. B2 vs B1：动态质量 gate 是否优于静态权重。
4. B3 vs B0：直接三流融合是否优于独立专家概率融合。
5. B4 vs B3：M2 history 的增量。
6. B5 vs B4：soft alignment + contrastive 的增量。

验收建议沿用 Phase A：12 次中至少 7 次主指标正增益；平均 Node Macro-F1 与 weakest recall 同升；Fault Macro-F1 不低于参考超过 0.5 pp。完整矩阵前不做结论性 best-condition 宣称。

## 7. 预期风险

- 严格 cross-fit 需 72 个内层 RGB backbone，是 B1/B2 最大计算成本，但这是避免 representation leakage 的必要代价。
- 只有 4 位参与者，内层 backbone 每次仅用 2 人，OOF 专家可能比最终外层专家弱；它仍是当前最干净的 meta-fusion 估计。
- B5 的 soft alignment 只能处理 token 层面的相对错位，不能修复原始数据中严重的错误配对或不同 sample_name。
- B3–B5 使用冻结 encoder cache，回答“融合层是否有效”；若有效，再单独预注册小学习率 end-to-end fine-tuning，不应在本矩阵中临时追加。
