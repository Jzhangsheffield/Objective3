# Atomic-Tail A0–A4 DualPos 实验结果详细分析报告

**更新日期：** 2026-08-20  
**结果目录：** `outputs/A0`、`A1`、`A2`、`A3`、`A3-DualPos`、`A4-DualPos`  
**训练范围：** `all_runs`  
**测试协议：** A、D、J、M 四折 LOSO；seeds 1、2、42；测试统一采用 actual chronological history  
**模型类型：** 所有实验均为 Direct Feature–History Fusion，不是 frozen-logit delta correction

---

## 1. 核心结论

1. **A3-DualPos 从头训练没有优于 A3。**在 `test_all` 上，A3-DualPos node accuracy 为 **90.52 ± 2.84%**，比 A3 的 **91.09 ± 2.66%**低 **0.57 pp**；12 个 fold×seed 配对为 **2 胜、0 平、10 负**，配对 95% CI 为 **[-1.73, +0.60] pp**。
2. A3-DualPos 的主要问题集中在 **Stage 3**：相对 A3 为 **-3.36 pp**，配对 95% CI 为 **[-6.62, -0.09] pp**。Stage 1 反而为 `+0.91 pp`，说明 displacement 编码可能在长历史/流程后段引入了额外扰动。
3. **A4-DualPos 是目前最有价值的新结果。**它相对 A0 的 `test_all` node accuracy 提升 **+0.45 pp**，达到 **91.02 ± 2.87%**；12 个配对为 **8 胜、1 平、3 负**，配对 95% CI 为 **[+0.01, +0.89] pp**。node macro-F1 提升 `+0.41 pp`，为 9 胜、3 负，但区间仍轻微跨过 0。
4. A4-DualPos 相对 A0 的增益比原 A3 更稳定。三个 seed 的平均增益分别为 **+0.07、+0.02、+1.27 pp**，没有出现 A3 那样 seed 1/2 为负、seed 42 大幅为正的明显反转；四个 participant 的三-seed 均值也全部为正。
5. **A4-DualPos 与 A3 总体处于同一水准。**A4-DualPos 相对 A3 的 node accuracy 为 `-0.07 pp`，node macro-F1 为 `+0.06 pp`。但 A4-DualPos 相对 A3 的 Stage 3 低 **1.55 pp**，说明它提高了稳定性，却没有保留 A3 在后段流程上的全部优势。
6. 预测级补充分析与 fold×seed 主分析方向一致：A4-DualPos 相对 A0 净增加 26/5,685 个正确 clip-seed 预测，cluster bootstrap 95% CI 为 **[+0.10, +0.79] pp**；正向信号主要来自 normal split，fault split 仍不确定。
7. **现阶段不能把 A4-DualPos 的全部 +0.45 pp 归因于 DualPos 本身。**A4-DualPos 同时引入 A0 warm-start、shift-only warmup、paired actual/augmented training、动态 shuffle 和 actual-only calibration；而单独加入 DualPos 的 A3-DualPos 反而下降。因此论文投稿前需要一个完全同训练日程的 `A4-NoShift` 控制实验。
8. 项目配置中建议的“明确有效”门槛是相对 A0 至少 `+1 pp`、12 个配对至少 9 胜且最差 participant 不明显退化。A4-DualPos 当前为 `+0.45 pp`、8胜/1平/3负，属于**小幅、较稳定的正向结果**，尚未达到“明显提升”的预设门槛。

综合判断：**A3-DualPos 证明了“让 shuffle 位移可见”本身并不足以改善从头训练；A4-DualPos 则表明，在强 A0 起点上将 DualPos 作为保守的 paired training regularizer，可以得到小而稳定的总体增益。当前最合适的论文表述是“稳定的小幅改善”，而不是“显著的大幅提升”。**

---

## 2. 数据完整性与分析口径

### 2.1 完成情况

| 实验 | 完成任务 | participant | seeds | 状态 |
|---|---:|---|---|---|
| A0 | 12/12 | A、D、J、M | 1、2、42 | 完整 |
| A1 | 12/12 | A、D、J、M | 1、2、42 | 完整 |
| A2 | 12/12 | A、D、J、M | 1、2、42 | 完整 |
| A3 | 12/12 | A、D、J、M | 1、2、42 | 完整 |
| A3-DualPos | 12/12 | A、D、J、M | 1、2、42 | 完整 |
| A4-DualPos | 12/12 | A、D、J、M | 1、2、42 | 完整 |

共分析 72 个完成任务。A3-DualPos 和 A4-DualPos 的 12 个任务均具有 `test_normal`、`test_fault`、`test_all`、predictions、augmentation audit 和 train log。

### 2.2 统计口径

- `均值 ± SD`：12 个 participant×seed 运行的算术均值和样本标准差；每个 fold×seed 等权。
- paired delta：在完全相同 participant、seed 和 split 之间计算。
- paired 95% CI：12 个配对差值的 t 区间，用于探索性不确定性描述；没有进行多重比较校正。
- 预测级补充分析：以同一 participant/run 下的 clips 和三个 seed 组成 cluster，进行 20,000 次 participant-run cluster bootstrap。
- `pp` 表示百分点，例如 90% 到 91% 为 `+1 pp`。
- LOSO fold×seed 配对是主要统计单位；预测级 cluster bootstrap 只作补充。

### 2.3 模型和训练公平性

- A0 复用共享旧 M2-Direct checkpoint。
- A1、A2、A3、A3-DualPos 均从头训练 50 epochs。
- A4-DualPos 从与 participant、seed、scope 完全匹配的 A0 checkpoint 热启动，随后训练 2 epochs shift-only、8 epochs paired joint fine-tuning 和 3 epochs actual-only calibration。
- A4-DualPos 测试时所有 displacement 均为 0；`shift_embedding` 不直接参与 actual-order 推理，其作用是训练期正则化。

因此，`A3-DualPos − A3` 是相对干净的 DualPos 编码消融；`A4-DualPos − A0` 是有效的工程性能对比，但不是单一机制消融。若用于论文中的严格方法归因，仍需要同训练日程的 no-shift 控制和当前代码路径下的 `A0-fresh`。

---

## 3. 总体性能

### 3.1 Test-all

| 实验 | Node accuracy | Node macro-F1 | Tier-3 accuracy | Tier-3 macro-F1 |
|---|---:|---:|---:|---:|
| A0 | 90.57 ± 3.06 | 87.81 ± 3.03 | 90.64 ± 3.03 | 87.06 ± 3.21 |
| A1 | 89.35 ± 3.66 | 86.78 ± 3.48 | 89.80 ± 3.35 | 86.35 ± 3.22 |
| A2 | 89.92 ± 3.56 | 87.16 ± 3.42 | 90.00 ± 3.48 | 86.44 ± 3.43 |
| **A3** | **91.09 ± 2.66** | 88.16 ± 2.67 | **91.28 ± 2.69** | **87.49 ± 2.80** |
| A3-DualPos | 90.52 ± 2.84 | 87.46 ± 3.00 | 90.72 ± 2.83 | 86.75 ± 3.15 |
| **A4-DualPos** | 91.02 ± 2.87 | **88.22 ± 3.09** | 91.04 ± 2.84 | 87.42 ± 3.22 |

观察：

- A3 保持最高 node accuracy 和 Tier-3 accuracy；A4-DualPos 仅低 `0.07 pp` 和 `0.24 pp`。
- A4-DualPos 的 node macro-F1 最高，但只比 A3 高 `0.06 pp`，不能视为实质差异。
- A3-DualPos 几乎回到 A0 水平，而没有延续 A3 的均值优势。
- A4-DualPos 的价值不在于刷新最高均值，而在于相对 A0 的 fold×seed 方向更一致。

### 3.2 Normal 与 fault

| Split | 实验 | Node accuracy | Node macro-F1 | Tier-3 accuracy | Tier-3 macro-F1 |
|---|---|---:|---:|---:|---:|
| normal | A0 | 91.26 ± 3.68 | 88.19 ± 4.56 | 91.30 ± 3.62 | 87.44 ± 4.99 |
| normal | A3 | **91.85 ± 3.36** | **88.90 ± 4.26** | **91.98 ± 3.33** | **88.29 ± 4.51** |
| normal | A3-DualPos | 91.16 ± 3.48 | 87.71 ± 4.35 | 91.34 ± 3.42 | 87.01 ± 4.60 |
| normal | A4-DualPos | 91.70 ± 3.40 | 88.67 ± 4.54 | 91.72 ± 3.35 | 87.86 ± 4.90 |
| fault | A0 | 89.75 ± 4.25 | 86.34 ± 5.37 | 89.86 ± 4.17 | 85.38 ± 5.89 |
| fault | A3 | 89.78 ± 3.25 | 85.48 ± 5.28 | **90.13 ± 3.11** | 84.37 ± 5.98 |
| fault | A3-DualPos | 89.87 ± 4.21 | 86.61 ± 4.62 | 90.03 ± 4.19 | 85.60 ± 5.02 |
| fault | **A4-DualPos** | **90.07 ± 3.57** | **86.80 ± 3.95** | 90.07 ± 3.57 | **85.83 ± 4.40** |

- A3-DualPos 相对 A3：normal accuracy `-0.69 pp`、normal macro-F1 `-1.19 pp`；fault accuracy `+0.09 pp`、fault macro-F1 `+1.12 pp`。DualPos 从头训练产生了 normal/fault 之间的权衡。
- A4-DualPos 相对 A0：normal accuracy `+0.44 pp`，fault accuracy `+0.33 pp`；两者数值都为正，但 fault 的不确定性更大。
- A4-DualPos 相对 A3：normal accuracy `-0.14 pp`，fault accuracy `+0.29 pp`，总体上仍属于同一水准。

---

## 4. 关键配对比较

### 4.1 Test-all 四项指标

| 对比 | 指标 | 平均差值 | 95% CI | W/T/L |
|---|---|---:|---:|---:|
| A3-DualPos − A3 | Node accuracy | **-0.57 pp** | [-1.73, +0.60] | **2/0/10** |
|  | Node macro-F1 | -0.71 pp | [-1.99, +0.58] | 3/0/9 |
|  | Tier-3 accuracy | -0.56 pp | [-1.76, +0.64] | 2/0/10 |
|  | Tier-3 macro-F1 | -0.74 pp | [-2.12, +0.63] | 4/0/8 |
| A3-DualPos − A0 | Node accuracy | -0.04 pp | [-1.77, +1.68] | 5/1/6 |
|  | Node macro-F1 | -0.35 pp | [-1.84, +1.13] | 4/0/8 |
| A4-DualPos − A0 | Node accuracy | **+0.45 pp** | **[+0.01, +0.89]** | **8/1/3** |
|  | Node macro-F1 | +0.41 pp | [-0.08, +0.90] | 9/0/3 |
|  | Tier-3 accuracy | +0.40 pp | [-0.04, +0.83] | 8/1/3 |
|  | Tier-3 macro-F1 | +0.35 pp | [-0.19, +0.89] | 9/0/3 |
| A4-DualPos − A3-DualPos | Node accuracy | +0.49 pp | [-1.00, +1.99] | 7/0/5 |
|  | Node macro-F1 | +0.76 pp | [-0.49, +2.02] | 9/0/3 |
| A4-DualPos − A3 | Node accuracy | -0.07 pp | [-1.58, +1.43] | 6/1/5 |
|  | Node macro-F1 | +0.06 pp | [-1.33, +1.45] | 7/0/5 |

A3-DualPos 对 A3 的 10/12 负向配对，比只看 `-0.57 pp` 的均值更值得重视。A4-DualPos 对 A0 的 node accuracy 区间刚好高于 0；效应很小，应表述为 consistent positive trend，而不是强显著性结论。

### 4.2 Split-specific paired delta

| Split | 对比 | Node accuracy delta | 95% CI | W/T/L | Macro-F1 delta |
|---|---|---:|---:|---:|---:|
| test-all | A3-DualPos − A3 | -0.57 | [-1.73, +0.60] | 2/0/10 | -0.71 |
| normal | A3-DualPos − A3 | -0.69 | [-2.04, +0.66] | 3/0/9 | -1.19 |
| fault | A3-DualPos − A3 | +0.09 | [-1.23, +1.41] | 6/1/5 | +1.12 |
| test-all | A4-DualPos − A0 | **+0.45** | **[+0.01, +0.89]** | **8/1/3** | +0.41 |
| normal | A4-DualPos − A0 | +0.44 | [-0.08, +0.96] | 8/1/3 | +0.48 |
| fault | A4-DualPos − A0 | +0.33 | [-0.62, +1.27] | 6/1/5 | +0.46 |

fault 样本较少且 participant-run clusters 只有 27 个，因此区间明显更宽。当前不能主张 A4-DualPos 已显著提升 fault robustness。

---

## 5. Participant 与 seed 稳定性

### 5.1 Test-all participant 均值

| 实验 | A | D | J | M |
|---|---:|---:|---:|---:|
| A0 | 89.56 | 88.53 | 94.47 | 89.71 |
| A3 | **90.80** | 88.82 | 94.89 | 89.86 |
| A3-DualPos | 89.64 | **89.39** | 94.17 | 88.89 |
| A4-DualPos | 89.95 | 89.18 | **95.02** | **89.93** |

| 对比 | A | D | J | M |
|---|---:|---:|---:|---:|
| A3-DualPos − A3 | -1.16 | +0.58 | -0.72 | -0.97 |
| **A4-DualPos − A0** | **+0.39** | **+0.65** | **+0.54** | **+0.22** |

A4-DualPos 对四个 participant 的三-seed 平均值全部为正，这是其稳定性最积极的证据。但 fault-only 中 M 相对 A0 下降 `-1.53 pp`，不能只报告总体 participant 均值。

### 5.2 Normal/fault participant delta

| Split 与对比 | A | D | J | M |
|---|---:|---:|---:|---:|
| A3-DualPos − A3, normal | -0.68 | +0.33 | -1.21 | -1.20 |
| A3-DualPos − A3, fault | -2.19 | +2.15 | +0.40 | 0.00 |
| A4-DualPos − A0, normal | +0.11 | +0.58 | +0.43 | +0.65 |
| A4-DualPos − A0, fault | +0.97 | +1.08 | +0.79 | **-1.53** |

### 5.3 Seed 均值与交互

| 实验 | seed 1 | seed 2 | seed 42 |
|---|---:|---:|---:|
| A0 | 92.28 | 90.85 | 88.57 |
| A3 | 90.77 | 90.64 | 91.87 |
| A3-DualPos | 90.72 | 89.81 | 91.05 |
| A4-DualPos | **92.35** | **90.86** | 89.84 |
| A3-DualPos − A3 | -0.05 | -0.83 | -0.82 |
| **A4-DualPos − A0** | **+0.07** | **+0.02** | **+1.27** |

A4-DualPos 的增益仍以 seed 42 最大，但与 A3 不同，seed 1 和 seed 2 的平均值没有下降。其目标“减少 seed-dependent reversal”已经部分实现。

### 5.4 A4-DualPos 相对 A0 的完整 fold×seed 配对

| Participant | Seed | A0 | A4-DualPos | Delta |
|---|---:|---:|---:|---:|
| A | 1 | 91.65 | 91.88 | +0.23 |
| A | 2 | 88.63 | 88.17 | -0.46 |
| A | 42 | 88.40 | 89.79 | +1.39 |
| D | 1 | 91.56 | 91.56 | 0.00 |
| D | 2 | 89.61 | 90.04 | +0.43 |
| D | 42 | 84.42 | 85.93 | +1.52 |
| J | 1 | 95.32 | 95.14 | -0.18 |
| J | 2 | 94.77 | 95.32 | +0.54 |
| J | 42 | 93.33 | 94.59 | +1.26 |
| M | 1 | 90.60 | 90.83 | +0.22 |
| M | 2 | 90.38 | 89.93 | -0.45 |
| M | 42 | 88.14 | 89.04 | +0.89 |

三次负向配对的下降均不超过 `0.46 pp`，而正向配对最高为 `+1.52 pp`。这比 A3−A0 曾出现的 `-4.11/+6.49 pp` 摆动温和得多。

---

## 6. Stage-level 结果

| 对比 | Stage 1 | Stage 2 | Stage 3 |
|---|---:|---:|---:|
| A3 − A0 | -0.58 | +0.51 | **+1.47** |
| A3-DualPos − A3 | +0.91 | -0.28 | **-3.36** |
| A3-DualPos − A0 | +0.33 | +0.22 | -1.89 |
| A4-DualPos − A0 | +0.71 | +0.50 | -0.09 |
| A4-DualPos − A3-DualPos | +0.38 | +0.28 | +1.80 |
| A4-DualPos − A3 | +1.29 | -0.01 | **-1.55** |

最值得注意的两项：

- A3-DualPos − A3，Stage 3：`-3.36 pp`，95% CI `[-6.62, -0.09]`，2/3/7。
- A4-DualPos − A3，Stage 3：`-1.55 pp`，95% CI `[-2.84, -0.27]`，1/4/7。

A3 的 true-recency 设计原本主要在 Stage 2/3 发挥作用。显式位移加入后，A3-DualPos 的 Stage 1 有数值改善，但 Stage 3 明显下降，说明 displacement 对更长历史可能过强或更难优化。A4-DualPos 的 warm-start 和保守训练恢复了部分 Stage 3 损失，但相对 A3 仍低 1.55 pp。

---

## 7. 预测级分析

### 7.1 A3-DualPos vs A3

在 5,685 个 test-all clip-seed 预测中：

- 两者预测相同：5,309，`93.39%`；
- A3-DualPos-only correct：126；A3-only correct：158；
- 净减少 32 个正确预测，clip-weighted delta 为 `-0.56 pp`；
- participant-run cluster bootstrap 95% CI：`[-1.13, +0.02] pp`。

normal split 的 cluster bootstrap CI 为 `[-1.29, -0.05] pp`，方向更明确；fault 为 `[-1.57, +1.06] pp`，不确定。

| 方向 | Node | Support | Recall delta |
|---|---|---:|---:|
| 提升 | 9 `move_pedal_to_safe_location` | 66 | +9.09 |
| 提升 | 19 `put_sample_on_machine_table_2` | 297 | +5.05 |
| 提升 | 18 `reverse_sample` | 297 | +2.69 |
| 下降 | 20 `grip_sample_from_machine_table_3` | 297 | **-9.43** |
| 下降 | 31 `move_pedal_to_original_place` | 72 | -8.33 |
| 下降 | 1 `unlock_crimper` | 66 | -6.06 |
| 下降 | 34 `take_lock_from_table` | 75 | -5.33 |

### 7.2 A4-DualPos vs A0

在 5,685 个 test-all clip-seed 预测中：

- 两者预测相同：5,545，`97.54%`；
- A4-DualPos-only correct：66；A0-only correct：40；
- 净增加 26 个正确预测，clip-weighted delta 为 `+0.46 pp`；
- participant-run cluster bootstrap 95% CI：`[+0.10, +0.79] pp`。

| Split | A4-only correct | A0-only correct | Clip-weighted delta | Cluster bootstrap 95% CI |
|---|---:|---:|---:|---:|
| test-all | 66 | 40 | +0.46 | `[+0.10, +0.79]` |
| normal | 50 | 30 | +0.46 | `[+0.05, +0.85]` |
| fault | 16 | 10 | +0.44 | `[-0.30, +1.18]` |

| 方向 | Node | Support | Recall delta |
|---|---|---:|---:|
| 提升 | 34 `take_lock_from_table` | 75 | +5.33 |
| 提升 | 1 `unlock_crimper` | 66 | +3.03 |
| 提升 | 9 `move_pedal_to_safe_location` | 66 | +3.03 |
| 提升 | 20 `grip_sample_from_machine_table_3` | 297 | +2.36 |
| 提升 | 17 `grip_sample_from_machine_table_2` | 297 | +1.68 |
| 下降 | 33 `turn_off_main_switch` | 78 | **-5.13** |
| 下降 | 7 `turn_on_water_pump` | 66 | -1.52 |
| 下降 | 8 `turn_on_extractor_fan` | 66 | -1.52 |
| 下降 | 31 `move_pedal_to_original_place` | 72 | -1.39 |

预测一致率达到 97.54%，说明 A4-DualPos 没有大范围改变 A0 的决策边界，而是在少量困难样本上产生净正向修正。这与“小学习率 warm-start regularization”的设计目标一致。类别统计把同一 clip 在多个 seed 下重复计数，只能作探索性诊断。

---

## 8. DualPos 实际作用范围

| 实验 | Active-tail 比例 | 真正改变顺序比例 | Shifted history token 比例 | Mean absolute position shift | Kendall distance |
|---|---:|---:|---:|---:|---:|
| A3 | 69.39% | 17.79% | 0 | 0 | 0.0291 |
| A3-DualPos | 69.39% | 17.79% | 15.90% | 0.556 | 0.0291 |
| A4-DualPos | 69.39% | 17.79% | 15.90% | 0.556 | 0.0291 |

- 三者使用相同 active-tail-only graph-valid shuffle；顺序扰动强度相同。
- 约 17.79% 的训练样本真正改变顺序，约 15.90% 的 history tokens 具有非零 displacement。
- `mean absolute position shift=0.556` 是对全部 history tokens 的平均；按 shifted tokens 条件化后，平均绝对位移约为 `3.49` 个位置。
- DualPos 确实让 shuffle 对 attention 可见，但可见范围只覆盖约 16% 的历史 token。A3-DualPos 的下降不是“代码没有产生位移”，而更可能是位移信号强度、长历史交互或从头优化方式不合适。

---

## 9. 训练动态

| 实验 | 初始化 | Epochs | 首次 ≥99% train acc | 首次 ≥99.9% | 最终 train acc | 平均训练时间/任务 |
|---|---|---:|---:|---:|---:|---:|
| A3 | 从头训练 | 50 | 2.0 | 3.25 | 99.95% | — |
| A3-DualPos | 从头训练 | 50 | 2.0 | 3.00 | 99.93% | 344.8 s |
| A4-DualPos | A0 warm-start | 13 | 1.0 | 1.00 | 100.00% | 90.7 s |

| A4-DualPos phase | Epochs/任务 | 可训练参数 | 阶段末平均 train acc |
|---|---:|---|---:|
| `dualpos_shift_warmup` | 2 | shift embedding | 99.969% |
| `dualpos_mixed_finetune` | 8 | 全模型；base/shift 分组 LR | 99.988% |
| `actual_calibration` | 3 | base；shift frozen | 100.000% |

A3-DualPos 与 A3 一样在第 2–4 epoch 已几乎拟合训练集，继续训练到 50 epoch 没有转化为更好泛化。A4-DualPos 的训练准确率从第一阶段就接近 100%，证明 A0 warm-start 主导了模型能力；新增分支只在很小的参数邻域内调整。最终结果可能同时来自 paired augmented view 和额外 actual calibration，不能只根据 final checkpoint 判断哪个 phase 产生增益。

---

## 10. 对 DualPos 假设的判断

### 得到支持的部分

- 位移编码已被正确施加：shifted token fraction 和 position shift audit 均非零。
- 在 A0 warm-start、paired actual/augmented 和 actual calibration 条件下，最终模型相对 A0 得到小幅、跨 participant 更一致的增益。
- A4-DualPos 对 A0 的 fold×seed paired CI 和预测级 cluster bootstrap CI 都给出正向信号，两种加权口径方向一致。

### 未得到支持的部分

- “只要模型同时知道真实时间位置和增强呈现顺序，从头训练就会明显提升”没有得到支持；A3-DualPos 对 A3 为 2胜10负。
- “DualPos 对长历史最有帮助”没有得到支持；当前最明显的退化恰好在 Stage 3。
- “A4-DualPos 的提升完全来自 displacement”尚未被证明，因为缺少相同 schedule 的 no-shift 控制。

### 当前最合理的机制解释

DualPos 更适合作为**强基线附近的训练期扰动提示/正则化信号**，而不是作为从头训练时的额外位置表征。A0 warm-start 保留了 actual-order 决策边界，paired view 让非零 shift 只在局部困难样本上施加约束，actual calibration 再把模型拉回测试分布。这个解释与 A4-DualPos 97.54% 的 A0 预测一致率相符。

---

## 11. 投稿前最需要补充的实验

### 11.1 最高优先级：A4-NoShift 严格控制

建议新增一个不改变主编号的补充实验，例如 `A4-DualPos-NoShift`：

- 使用与 A4-DualPos 完全相同的 A0 checkpoint、batch、seed、2+8+3 epoch 日程、actual/augmented CE 权重、shuffle refresh 和优化器设置；
- 唯一差异是强制 `history_shift_ids=0`，使 `E_shift` 始终不参与 token；
- Phase 1 保留同样的数据遍历和 epoch 数，即使它对 base 参数不更新，以保证训练预算可比；
- 核心报告 `A4-DualPos − A4-DualPos-NoShift` 的 12 个 paired deltas。

这一实验回答：`+0.45 pp` 是来自 DualPos，还是来自对 A0 的额外 paired fine-tuning/calibration。没有它，A4-DualPos 可以作为完整方法结果，但不适合把增益全部归因于 DualPos 分支。

### 11.2 无需重训的阶段 checkpoint 评估

A4-DualPos 已保存 `after_dualpos_shift_warmup.pth`、`after_dualpos_mixed_finetune.pth` 和 `after_actual_calibration.pth`。建议直接对三个 checkpoint 做 actual-order test，无需重新训练。若提升只在 final calibration 后出现，说明增益主要来自 actual calibration；若 mixed checkpoint 已超过 A0，则更支持 paired DualPos training 的作用。

### 11.3 Stage 3 定向改进

如果继续优化方法，优先级建议为：

1. 对 `E_shift` 增加可学习标量 gate，并初始化为较小值；
2. gate 随 history length 增长而衰减，或只对小位移保留完整幅度；
3. 将 `shift_embedding_init_std` 从 `0.02` 向更小范围扫描，例如 `{0.005, 0.01, 0.02}`；
4. 单独报告 Stage 3 和长 history subset，而不是只看 test-all；
5. 保持 A4 的 warm-start/paired/calibration 框架，不建议再次用 50 epochs 从头训练搜索。

### 11.4 论文级稳健性

- 补跑当前代码路径下的 `A0-fresh`，排除旧 checkpoint 与新训练代码路径差异。
- 若算力允许，在确定最终方法后把 seeds 从 3 扩展到至少 5；不要先看结果再选择性增加有利 seed。
- 保留 normal/fault、participant、stage、node macro-F1 和 Tier-3 macro-F1，不能只报告 test-all accuracy。

---

## 12. 论文表述建议

### 当前可以使用

> Explicitly encoding shuffle displacement did not improve the scratch-trained model: A3-DualPos reduced test-all node accuracy by 0.57 percentage points relative to A3 and produced losses in 10 of 12 fold-seed pairs, with the largest degradation occurring in Stage 3. In contrast, the warm-started paired formulation, A4-DualPos, preserved the strong actual-order baseline and achieved a modest but more consistent improvement over A0 (+0.45 percentage points; 8 wins, 1 tie, and 3 losses). These results suggest that DualPos is more effective as a conservative training-time regularizer around a pretrained actual-order model than as an additional positional representation learned from scratch.

### 只有完成 no-shift 控制后才建议使用

> The gain of A4-DualPos over an otherwise identical no-shift training control demonstrates that explicit shuffle displacement, rather than additional fine-tuning or calibration alone, contributes to the observed improvement.

### 当前不建议使用

- “DualPos significantly outperforms A3.”
- “DualPos substantially improves recognition accuracy.”
- “The displacement branch alone causes the A4 improvement.”
- “A4-DualPos improves fault robustness across all participants.”
- “The method is uniformly effective for long histories.”

---

## 13. 最终判断

| 问题 | 结论 |
|---|---|
| A3-DualPos 是否优于 A3？ | 否；平均 `-0.57 pp`，2胜10负，Stage 3 明显下降。 |
| A3-DualPos 是否至少保持 A0？ | 基本持平；相对 A0 `-0.04 pp`，区间很宽。 |
| A4-DualPos 是否优于 A0？ | 数值上是，而且方向较稳定；`+0.45 pp`，8胜1平3负。 |
| A4-DualPos 是否优于 A3？ | 总体基本持平；accuracy `-0.07 pp`、macro-F1 `+0.06 pp`。 |
| A4-DualPos 是否达到“明显提升”门槛？ | 尚未；低于预设 `+1 pp/至少9胜` 门槛。 |
| 能否把 A4 增益归因于 DualPos？ | 尚不能；需要相同 schedule 的 no-shift 控制和阶段 checkpoint 评估。 |
| 当前最值得保留的方案？ | A4-DualPos，作为稳定的小幅改进方案；A3 仍是最高 accuracy 的简单 scratch 方案。 |

最重要的下一步不是继续扩展 A5–A8，而是先完成 **A4-DualPos-NoShift + 三阶段 checkpoint evaluation**。这两个补充可以用较小计算成本判断 A4-DualPos 是否具有可发表的机制证据。

---

## 14. 可复现分析入口

本报告的机器统计由更新后的分析脚本生成：

```powershell
python tools\analyze_a0_a3_detailed.py `
  --package-root . `
  --bootstrap-repetitions 20000 `
  --brief
```

分析脚本当前读取 `outputs/A0`、`A1`、`A2`、`A3`、`A3-DualPos` 和 `A4-DualPos`；不会修改训练结果。脚本文件名为兼容旧入口暂未改名。
