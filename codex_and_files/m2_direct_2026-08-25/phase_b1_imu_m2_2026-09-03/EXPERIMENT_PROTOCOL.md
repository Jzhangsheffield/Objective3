# B1_IMU_M2 实验协议与实现说明

## 1. 研究问题与假设

原 B1 的三个专家为 `cam0 M2 + cam1 M2 + IMU Direct Node`。本追加实验把第三个专家改成 `IMU M2`：

| 条件 | cam0 | cam1 | IMU | 融合器 |
|---|---|---|---|---|
| 原 B1 | M2 actual history | M2 actual history | Direct Node | 三温度 + 静态 simplex 权重 |
| B1_IMU_M2 | M2 actual history | M2 actual history | M2 actual history | 同上，重新用 OOF 拟合 |

假设是 IMU 的瞬时波形可能在相邻工序间相似，而同一 run 已发生的步骤能提供工艺进度先验。M2 可用历史区分“当前动作外观/运动相近、但工序位置不同”的 Node，因此可能提升 IMU 专家本身和最终融合。

这个假设值得验证，但不能直接用 Phase A 的 `S3 > S7` 作为因果结论。S3 与 S7 不仅历史结构不同，上游 encoder 的训练方式也不同。本实验固定 Phase B encoder，减少该混杂因素。

## 2. 数据划分与防泄漏

### 2.1 外层 LOSO

每次选一个参与者 `O ∈ {A,D,J,M}` 作为最终测试集，其余三人为外层训练 partition。整个实验共 `4 outer × 3 seeds = 12` 个最终运行。

### 2.2 inner-LOSO 融合训练

融合参数不能在 outer test 上拟合。对每个 outer `O`：

1. 在剩余三名参与者中依次选一名 inner held-out `I`；
2. 读取原 Phase B 中只用另外两人训练的 IMU encoder；
3. 用这两人的冻结 IMU 特征训练新的 IMU M2 头；
4. 预测 `I`，获得该 inner participant 的 OOF 概率；
5. 与原 Phase B 同一 inner split 的 cam0/cam1 M2 OOF 概率对齐；
6. 合并三名 inner held-out 的预测，拟合一个 B1 静态融合器。

因此每条用于拟合融合器的样本，都来自未见过该参与者的三个专家模型。outer test 既不参与 IMU M2 训练，也不参与温度或融合权重学习。

### 2.3 历史边界

历史只允许使用：

- 同一 participant；
- 同一 run；
- `annotation_row_index` 严格小于当前 clip 的真实既往 clip。

不跨 run 拼接，不使用当前之后的 clip，不把预测类别反馈为历史，也不使用 outer test 标签。测试时这是“实际已发生历史”（actual history），不是 oracle label history，因为输入是既往 clip 的冻结 IMU feature，而非其真值类别。

## 3. 固定 IMU encoder 与特征

复用 Phase B 的 `IMUResNet10`。其输入为 Phase B 已缓存并标准化的 IMU 信号；`forward_features` 输出每个 clip 的 512 维 global feature。

- inner：读取 `crossfit/.../imu_direct_node/last.pth`，对对应 inner train/test signal cache 离线提取 512-D feature；
- outer：直接读取 Phase B 为 B3–B5 已生成的 `temporal_caches/.../imu_train.pt` 和 `imu_test.pt` 中 `global_features`；
- 新 M2 训练时 encoder 不在计算图中，也不会反向更新；
- feature cache 保存来源 checkpoint、SHA-256、manifest 和冻结标记，便于追溯。

固定的是 encoder 表示，不复用原 Direct Node 的线性分类头。新的 M2 历史头从头训练，因此比较应称为“Direct head 与 M2 history head 的比较”，而不是仅增加一个完全无参数的历史操作。

## 4. IMU M2 模型

设当前 clip 的冻结特征为 `x_t ∈ R^512`，同一 run 的既往特征为 `x_1...x_(t-1)`。

### 4.1 投影和位置编码

当前特征和历史特征使用相互独立的投影：

```text
q_t = LayerNorm(W_current x_t + b_current)          ∈ R^256
h_i = LayerNorm(W_history x_i + b_history) + e_pos ∈ R^256
```

`e_pos` 是可学习的相对位置 embedding。最近一个历史 clip 的 position id 为 1，再往前依次为 2、3……，超过 35 的位置统一截到 id 35；实际历史 token 本身不会被删除。

### 4.2 null-history token

每个样本的历史序列最前面加入一个可学习的 `null_history ∈ R^256`。它有两个作用：

- run 的第一个 clip 没有既往历史时，attention 仍有合法 key/value；
- 有历史时，模型也可以把权重分给 null token，相当于主动忽略不可靠历史。

padding token 通过 mask 排除，不参与 attention。

### 4.3 四头注意力

当前特征是唯一 query，`null + 所有既往 clip` 是 key/value：

```text
context_t = MultiHeadAttention(
    query=q_t,
    key=[null, h_1, ..., h_(t-1)],
    value=[null, h_1, ..., h_(t-1)],
    heads=4
) ∈ R^256
```

这里不是在 IMU 时间采样点之间做 attention，而是在 clip 级历史之间做 attention。每个历史 token 已经是 IMUResNet10 对一个完整 clip 的 512-D 摘要。

### 4.4 当前—历史融合和 Node 分类

```text
u_t      = Linear([x_t ; context_t]) ∈ R^512
logits_t = Linear(LayerNorm(u_t))    ∈ R^35
```

融合层输入是 `512 + 256 = 768` 维。初始化时，融合层的前 512 列为单位矩阵，其余权重和 bias 为 0，所以初始状态近似 `u_t=x_t`。随后训练再逐渐学习是否以及如何引入历史，降低随机历史分支在训练初期破坏当前观察的风险。

模型结构与 Phase A 的 `SensorM2Direct` 保持一致，仅输入换成固定的 Phase B IMU encoder 表示。

## 5. IMU M2 训练配置

| 配置 | 值 |
|---|---:|
| feature dim | 512 |
| attention dim | 256 |
| heads | 4 |
| max position id | 35 |
| dropout | 0.1 |
| Node classes | 35 |
| epochs | 50 |
| physical/effective batch | 64 / 64 |
| optimizer | AdamW |
| learning rate | 0.001 |
| weight decay | 0.0001 |
| gradient clipping | 1.0 |
| loss | 35 类 cross entropy |

冻结后的 512-D feature 很小，batch 64 可直接放入显存，因此不使用 gradient accumulation。与“physical batch 16、累积到 64”相比，优化器更新所见的有效样本数相同；直接 batch 64 更简单且没有 encoder 激活的显存压力。

训练不使用 validation、early stopping 或 best epoch；每个 outer/inner/seed 都保存第 50 轮 final epoch。seed 控制模型初始化、DataLoader shuffle、PyTorch CPU/CUDA 随机数。

## 6. B1_IMU_M2 决策融合

三个输入概率按相同 `sample_name` 严格对齐：

```text
p_cam0, p_cam1, p_imu_m2 ∈ R^35
```

每个专家学习一个温度 `T_m > 0`，三个融合权重由 softmax 参数化，因此 `w_m ≥ 0` 且总和为 1：

```text
z_m,c = log(max(p_m,c, 1e-8)) / T_m
s_c   = Σ_m w_m z_m,c
q_c   = softmax(s)_c
```

优化目标是 inner OOF 样本上的 Node cross entropy，另加很轻的权重均匀 L2：

```text
L = CE(q, y) + 0.001 × Σ_m (w_m - 1/3)^2
```

使用 Adam，1500 个 full-batch step，学习率 0.03。温度限制在 `[0.05, 20]`。这些设置与原 B1 相同，但参数必须重新拟合，因为 IMU 概率分布已经改变。

最终融合仍是加权几何平均式的静态 late fusion，并不是 B2 的逐 clip gate；每个 outer×seed 只学习一组三个温度和三个全局权重。

## 7. 主要与次要比较

### 7.1 主要比较

`B1_IMU_M2` 对 `B1`，在相同 12 个 outer×seed 单元上配对：

- 主指标：`test_all Node Macro-F1`；
- 辅助指标：Node accuracy、weakest-class recall、Tier3 Macro-F1；
- 同时检查 `test_normal` 与 `test_fault`，防止总体提升只来自一个子集。

### 7.2 机制比较

`IMU_M2` 对 `IMU_Direct`：确认新融合的变化是否确实伴随 IMU 专家本身改善。若 IMU M2 单模态改善但最终融合不升，可能是新错误与 camera 高度重合，或静态融合权重/温度已把 IMU 边际信息压低。

### 7.3 统计输出

所有比较都保留 12 个配对运行，不挑最佳 seed。自动汇总报告：

- 两方法的均值和标准差；
- 每个配对单元的差值；
- 平均差及 t 分布 95% CI；
- paired t-test、Wilcoxon signed-rank p 值；
- 12 个单元中的胜/平/负计数；
- 新融合中三专家权重和温度的均值。

样本规模只有 12 个配对单元，p 值应与效应量、置信区间、参与者一致性和 fault 子集共同解释，不能只用单一显著性阈值判断。

## 8. 结果解释边界

以下结论在本设计中可以支持：

- 固定 Phase B IMU encoder 表示时，clip 级实际历史头是否改善 IMU 专家；
- 将该专家放入与 B1 相同的三专家融合框架后，整体性能是否提高；
- 融合权重是否因 IMU M2 变强而重新分配。

以下结论不能仅由本实验支持：

- 历史对所有 IMU encoder 或其他传感器都必然有效；
- end-to-end 联合更新 encoder 会有相同收益；
- Phase A S3 与 S7 的全部差距都是历史造成的；
- 静态权重已是使用 IMU M2 的最优融合方式。

若新 IMU M2 明显改善但 B1_IMU_M2 仍无提升，下一步更有针对性的实验应是固定这三个 M2 专家，比较静态权重与受正则约束的质量 gate，而不是再次更换 encoder。
