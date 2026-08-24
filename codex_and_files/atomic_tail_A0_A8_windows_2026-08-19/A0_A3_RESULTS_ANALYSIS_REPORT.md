# Atomic-Tail 实验结果分析报告（A0–A3 历史版；已更新 DualPos）

> **2026-08-20 更新：** A3-DualPos 与 A4-DualPos 的完整分析已写入
> [A0_A4_DUALPOS_RESULTS_ANALYSIS_REPORT.md](A0_A4_DUALPOS_RESULTS_ANALYSIS_REPORT.md)。
> 新结论：A3-DualPos 相对 A3 为 `-0.57 pp`（2胜10负），A4-DualPos 相对 A0 为
> `+0.45 pp`（8胜1平3负）。以下内容保留为 A0–A3 的历史分析，避免覆盖原始比较记录。

**分析日期：** 2026-08-20  
**结果目录：** `outputs/A0`–`outputs/A3`  
**训练范围：** `all_runs`  
**测试协议：** A、D、J、M 四折 LOSO；seeds 1、2、42；测试统一采用 actual chronological history  
**模型：** A0–A3 均为 Direct Feature–History Fusion，不是 frozen-logit delta correction

---

## 1. 核心结论

1. **A3 是 A0–A3 中总体均值最高、跨运行标准差最小的方案。**在 `test_all` 上，A3 node accuracy 为 **91.09 ± 2.66%**，A0 为 **90.57 ± 3.06%**；A3 数值上高 **0.52 个百分点**。
2. **“A3 与 A0 基本处于同一水准”是当前最稳妥的结论。**12 个 participant×seed 配对中，A3 对 A0 为 **7 胜、0 平、5 负**；平均差为 **+0.52 pp**，配对 95% CI 为 **[-1.22, +2.27] pp**，跨过 0。
3. A3 的优势主要来自 **normal runs**：`test_normal` 相对 A0 为 **+0.59 pp**；`test_fault` 仅 **+0.04 pp**，可以视为基本持平。
4. A3 在四个 participant 的三 seed 均值上都高于 A0，但提升很小：A `+1.24 pp`、D `+0.29 pp`、J `+0.42 pp`、M `+0.15 pp`。
5. **seed 依赖非常明显。**A3−A0 在 seed 1、2、42 上分别为 **−1.51、−0.21、+3.30 pp**。因此当前平均增益很大程度由 seed 42 驱动，尚不能称为稳定提升。
6. A1 的旧式 atomic-tail once 明显偏弱；A2 的 active-tail-only gating 恢复了一部分性能；A3 的 true-recency position 又进一步恢复并超过 A0 的均值。A3 相对 A1 的 test-all node accuracy 提升 **+1.74 pp**，配对 95% CI **[+0.40, +3.08] pp**。
7. A1–A3 在第 2 个 epoch 已达到约 99% 训练准确率，第 3–4 个 epoch 达到约 99.9%，但仍训练到 50 epoch。训练集极快饱和，说明后续 A4–A8 应重点依赖 paired regularization、低学习率和实际顺序校准，而不是增加训练 epoch。

综合判断：**A3 已经成功消除了旧 atomic-tail A1 的主要性能损失，并在均值和方差上略优于 A0，但现阶段证据不足以表述为“稳定超过 A0”。**

---

## 2. 数据完整性与比较口径

### 2.1 完成情况

| 实验 | 完成任务 | participant | seeds | 缺失任务 |
|---|---:|---|---|---|
| A0 | 12/12 | A、D、J、M | 1、2、42 | 无 |
| A1 | 12/12 | A、D、J、M | 1、2、42 | 无 |
| A2 | 12/12 | A、D、J、M | 1、2、42 | 无 |
| A3 | 12/12 | A、D、J、M | 1、2、42 | 无 |

共分析 48 个完成任务。每个任务均包含 `test_normal`、`test_fault`、`test_all`、augmentation audit 和 train log。

### 2.2 指标汇总方法

- 表中的 `均值 ± SD` 是 12 个 participant×seed 运行的算术均值和样本标准差。
- paired delta 按完全相同的 participant、seed、split 配对。
- paired 95% CI 使用 12 个配对差值的 t 区间；这里只作为探索性不确定性描述，没有进行多重比较校正。
- 预测级补充分析将同一 participant/run 的所有 clips 和三个 seed 视作一个 cluster，进行了 20,000 次 participant-run cluster bootstrap。
- `pp` 表示百分点，例如 90% 到 91% 为 `+1 pp`。

### 2.3 一个重要的公平性说明

A0 默认复用了旧实验包的 `m2_direct` checkpoint，而 A1–A3 由新包从头训练。它们使用相同 Direct Fusion 架构、特征缓存、测试协议和 actual-order evaluation，但 **A0 与 A1–A3 不是完全同一训练代码路径下重新训练得到的模型**。

因此：

- A2−A1 和 A3−A2 是较干净的机制消融；
- A3−A0 可以用于工程效果比较，但若要形成论文中的严格因果消融，建议补跑一次由当前代码从头训练的 `A0-fresh`。

---

## 3. 总体性能

### 3.1 Test-all

| 实验 | Node accuracy | Node macro-F1 | Tier-3 accuracy | Tier-3 macro-F1 |
|---|---:|---:|---:|---:|
| A0 | 90.57 ± 3.06 | 87.81 ± 3.03 | 90.64 ± 3.03 | 87.06 ± 3.21 |
| A1 | 89.35 ± 3.66 | 86.78 ± 3.48 | 89.80 ± 3.35 | 86.35 ± 3.22 |
| A2 | 89.92 ± 3.56 | 87.16 ± 3.42 | 90.00 ± 3.48 | 86.44 ± 3.43 |
| **A3** | **91.09 ± 2.66** | **88.16 ± 2.67** | **91.28 ± 2.69** | **87.49 ± 2.80** |

观察：

- A3 在四项总体指标上均为最高均值。
- A3 的 node accuracy SD 为 2.66%，低于 A0 的 3.06%、A1 的 3.66% 和 A2 的 3.56%。这说明 A3 的离散程度有所下降，但还不能仅凭 SD 判定统计稳定性显著改善。
- A1 相对 A0 损失 1.22 pp，说明旧式“无 active tail 也做 graph-valid shuffle + 重排后伪 position”整体上伤害实际顺序测试性能。
- A2 将 A1 的损失从 −1.22 pp 缩小到 −0.64 pp。
- A3 在 A2 基础上再提升 1.17 pp，最终略高于 A0。

### 3.2 Normal 与 fault

| Split | 实验 | Node accuracy | Node macro-F1 | Tier-3 accuracy | Tier-3 macro-F1 |
|---|---|---:|---:|---:|---:|
| normal | A0 | 91.26 ± 3.68 | 88.19 ± 4.56 | 91.30 ± 3.62 | 87.44 ± 4.99 |
| normal | A1 | 90.01 ± 4.52 | 87.17 ± 5.22 | 90.40 ± 4.17 | 86.73 ± 5.12 |
| normal | A2 | 90.54 ± 4.14 | 87.57 ± 4.74 | 90.60 ± 4.05 | 86.86 ± 4.85 |
| normal | **A3** | **91.85 ± 3.36** | **88.90 ± 4.26** | **91.98 ± 3.33** | **88.29 ± 4.51** |
| fault | A0 | 89.75 ± 4.25 | **86.34 ± 5.37** | 89.86 ± 4.17 | **85.38 ± 5.89** |
| fault | A1 | 88.72 ± 3.85 | 85.30 ± 5.60 | 89.24 ± 3.63 | 84.53 ± 6.10 |
| fault | A2 | 89.01 ± 3.99 | 85.56 ± 5.59 | 89.07 ± 3.87 | 84.49 ± 6.27 |
| fault | **A3** | **89.78 ± 3.25** | 85.48 ± 5.28 | **90.13 ± 3.11** | 84.37 ± 5.98 |

关键区别：

- A3 对 A0 的 node accuracy：normal `+0.59 pp`，fault 仅 `+0.04 pp`。
- A3 的 fault node macro-F1 比 A0 低 `0.85 pp`，Tier-3 macro-F1 低 `1.01 pp`；accuracy 持平并不等于少数类表现也持平。
- 当前改善主要是正常流程识别的轻微提高，而不是对故障流程的鲁棒性提高。

---

## 4. 配对消融分析

### 4.1 Test-all node accuracy

| 对比 | 平均差值 | 95% CI | Win/Tie/Loss | 解释 |
|---|---:|---:|---:|---|
| A1 − A0 | −1.22 pp | [−2.89, +0.45] | 5/1/6 | 旧 atomic-tail 整体偏弱 |
| A2 − A0 | −0.64 pp | [−2.03, +0.75] | 5/1/6 | active-tail-only 缩小损失 |
| A3 − A0 | **+0.52 pp** | **[−1.22, +2.27]** | **7/0/5** | 数值略优，但不确定性较大 |
| A2 − A1 | +0.57 pp | [−0.91, +2.06] | 7/0/5 | gating 有正向趋势 |
| A3 − A2 | +1.17 pp | [−0.67, +3.00] | 7/0/5 | true recency 有正向趋势 |
| A3 − A1 | **+1.74 pp** | **[+0.40, +3.08]** | **8/1/3** | 完整改进明显优于旧 A1 |

### 4.2 A3 相对 A0 的其他指标

| 指标 | Test-all paired delta | 95% CI | Win/Tie/Loss |
|---|---:|---:|---:|
| Node accuracy | +0.52 pp | [−1.22, +2.27] | 7/0/5 |
| Node macro-F1 | +0.35 pp | [−1.21, +1.91] | 7/0/5 |
| Tier-3 accuracy | +0.64 pp | [−1.15, +2.43] | 7/0/5 |
| Tier-3 macro-F1 | +0.43 pp | [−1.19, +2.04] | 6/0/6 |

四项区间全部跨过 0。因此，当前正确表述是“performance parity with a small positive numerical trend”，而不是“consistent improvement over A0”。

### 4.3 Split-specific paired delta

| Split | A3−A0 node accuracy | 95% CI | W/T/L | Node macro-F1 delta |
|---|---:|---:|---:|---:|
| test-all | +0.52 pp | [−1.22, +2.27] | 7/0/5 | +0.35 pp |
| normal | +0.59 pp | [−1.22, +2.39] | 6/1/5 | +0.71 pp |
| fault | +0.04 pp | [−1.86, +1.93] | 6/1/5 | −0.85 pp |

---

## 5. Participant 与 seed 稳定性

### 5.1 Test-all participant 均值

| 实验 | A | D | J | M |
|---|---:|---:|---:|---:|
| A0 | 89.56 ± 1.81 | 88.53 ± 3.69 | 94.47 ± 1.02 | 89.71 ± 1.36 |
| A1 | 88.86 ± 2.78 | 86.36 ± 1.98 | 94.47 ± 0.55 | 87.70 ± 2.16 |
| A2 | 88.79 ± 0.35 | 88.02 ± 4.10 | 94.59 ± 1.36 | 88.29 ± 2.58 |
| **A3** | **90.80 ± 1.10** | **88.82 ± 1.84** | **94.89 ± 0.99** | **89.86 ± 1.23** |

A3−A0 的 participant 平均差：

- A：`+1.24 pp`
- D：`+0.29 pp`
- J：`+0.42 pp`
- M：`+0.15 pp`

三 seed 均值上，A3 对四个 participant 都没有退化，这是积极信号。但 D 和 M 的平均改善非常小。

### 5.2 Normal/fault 的 participant 差异

| Split | A | D | J | M |
|---|---:|---:|---:|---:|
| A3−A0 normal | +0.79 | +0.50 | +0.69 | +0.37 |
| A3−A0 fault | +2.19 | −1.08 | −0.20 | −0.77 |

normal split 在四个 participant 上均为正；fault split 只有 A 为正。这进一步证明 A3 的当前优势主要来自正常流程，而不是对 fault 的普适改进。

### 5.3 Seed 交互

| 实验 | seed 1 | seed 2 | seed 42 |
|---|---:|---:|---:|
| A0 | 92.28 | 90.85 | 88.57 |
| A1 | 88.85 | 90.23 | 88.97 |
| A2 | 89.71 | 90.71 | 89.36 |
| A3 | 90.77 | 90.64 | 91.87 |
| **A3−A0** | **−1.51** | **−0.21** | **+3.30** |

完整 A3−A0 配对如下：

| Participant | Seed | A0 | A3 | Delta |
|---|---:|---:|---:|---:|
| A | 1 | 91.65 | 91.18 | −0.46 |
| A | 2 | 88.63 | 89.56 | +0.93 |
| A | 42 | 88.40 | 91.65 | +3.25 |
| D | 1 | 91.56 | 87.45 | −4.11 |
| D | 2 | 89.61 | 88.10 | −1.52 |
| D | 42 | 84.42 | 90.91 | +6.49 |
| J | 1 | 95.32 | 95.86 | +0.54 |
| J | 2 | 94.77 | 94.95 | +0.18 |
| J | 42 | 93.33 | 93.87 | +0.54 |
| M | 1 | 90.60 | 88.59 | −2.01 |
| M | 2 | 90.38 | 89.93 | −0.45 |
| M | 42 | 88.14 | 91.05 | +2.91 |

风险点：

- D fold 从 `−4.11 pp` 到 `+6.49 pp`，表明增强与 feature/model seed 存在强交互。
- A3 的总体均值提升不能被解读为每个 seed 都稳定获益。
- 后续 A4–A8 的重要目标应当是降低这种 seed-dependent reversal，而不只是提高平均值。

---

## 6. Stage-level 结果

Test-all node accuracy paired delta：

| 对比 | Stage 1 | Stage 2 | Stage 3 |
|---|---:|---:|---:|
| A1−A0 | +0.33 pp | −1.38 pp | −2.26 pp |
| A2−A1 | −0.73 pp | +0.59 pp | **+1.96 pp** |
| A3−A2 | −0.18 pp | +1.30 pp | +1.76 pp |
| A3−A0 | −0.58 pp | +0.51 pp | +1.47 pp |

其中 A2−A1 的 Stage 3 配对 95% CI 为 `[+0.66, +3.26] pp`；其他上述 stage 区间大多跨过 0。

解释：active-tail-only gating 对流程后段更有帮助；A3 的优势也主要出现在 Stage 2/3，而不是 Stage 1。这与“历史顺序和真实 recency 在长历史中更重要”的机制预期一致。

---

## 7. 预测级 A3 vs A0 分析

### 7.1 Test-all

在所有 participant、seed、clip 展开后的 5,685 个 clip-seed 预测中：

- A0 与 A3 预测相同：5,325，**93.67%**；
- 两者都正确：5,040；
- A3 正确、A0 错误：150；
- A0 正确、A3 错误：121；
- 两者都错误：374；
- A3 净增加 29 个正确预测，对应 clip-weighted `+0.51 pp`。

以 103 个 participant-run cluster 进行 20,000 次 bootstrap，A3−A0 的 clip-weighted 95% CI 为约 **[+0.05, +0.97] pp**。这个结果显示一个小的正向信号，但与 12 个 fold-seed 均值的 t 区间结论不同，因为两者的加权单位不同：前者更偏向样本数较多的 run，后者让每个 fold-seed 等权。

论文中应以预先确定的 LOSO fold/seed 配对为主要分析，cluster bootstrap 作为补充，不应只选择更有利的统计口径。

### 7.2 Normal 与 fault

| Split | A3-only correct | A0-only correct | 净正确数 | Cluster bootstrap 95% CI |
|---|---:|---:|---:|---:|
| normal | 111 | 86 | +25 / 4,323 | `[+0.07, +1.07] pp` |
| fault | 39 | 35 | +4 / 1,362 | `[−0.74, +1.45] pp` |

这再次确认：小幅正向信号集中在 normal，fault 基本持平。

### 7.3 类别层面的主要变化

Test-all 中相对 A0 提升较大的节点：

| Node | Support（clip-seed） | A0 recall | A3 recall | Delta |
|---|---:|---:|---:|---:|
| 20 `grip_sample_from_machine_table_3` | 297 | 82.83 | 91.25 | +8.42 |
| 34 `take_lock_from_table` | 75 | 58.67 | 65.33 | +6.67 |
| 35 `lock_crimper` | 75 | 78.67 | 82.67 | +4.00 |
| 18 `reverse_sample` | 297 | 86.87 | 89.90 | +3.03 |
| 24 `put_sample_on_table` | 309 | 78.64 | 81.55 | +2.91 |

下降较大的节点：

| Node | Support（clip-seed） | A0 recall | A3 recall | Delta |
|---|---:|---:|---:|---:|
| 9 `move_pedal_to_safe_location` | 66 | 86.36 | 81.82 | −4.55 |
| 19 `put_sample_on_machine_table_2` | 297 | 97.64 | 93.27 | −4.38 |
| 4 `turn_on_crimper` | 66 | 77.27 | 74.24 | −3.03 |
| 11 `put_protection_cover_on_ground` | 66 | 92.42 | 89.39 | −3.03 |
| 26 `take_protection_cover_from_ground` | 75 | 94.67 | 92.00 | −2.67 |

这些是跨 seed 重复计数的探索性类别统计，不应当作独立样本显著性结果。但 node 20 的提升和 node 19 的下降值得在 A4–A8 后继续追踪。

---

## 8. Augmentation 实际作用范围

| 实验 | Active-tail 样本比例 | 真正改变顺序的比例 | 平均 normalized Kendall distance |
|---|---:|---:|---:|
| A0 | 69.39% | 0.00% | 0.0000 |
| A1 | 69.39% | 32.76% | 0.0683 |
| A2 | 69.39% | 17.79% | 0.0291 |
| A3 | 69.39% | 17.79% | 0.0291 |

解释：

1. 约 69.4% 的训练样本被识别为 active incomplete atomic prefix，但受到图约束后，A2/A3 只有约 17.8% 的样本真正改变顺序。
2. A1 改变约 32.8% 的样本，因为它在无 active tail 时也执行普通 graph-valid shuffle。
3. A2 将扰动覆盖率约减半、Kendall 距离从 0.068 降到 0.029，结果比 A1 恢复约 0.57 pp。
4. A2 与 A3 的 shuffle order 完全相同；二者差异只来自 position ID 语义。因此 A3−A2 的 +1.17 pp 是支持 true-recency 的最直接证据，尽管 95% CI 仍跨过 0。
5. 实际增强比例不到 20%，意味着 A4 之后的 paired/consistency 机制不能只依赖“changed samples 数量”，还应记录 changed-mask 子集上的一致性损失和准确率。

---

## 9. 训练动态

| 实验 | 训练任务 | Epochs | 首次达到 99% train acc | 首次达到 99.9% train acc | 最终 train acc |
|---|---:|---:|---:|---:|---:|
| A0 | 0（复用 checkpoint） | — | — | — | — |
| A1 | 12 | 50 | 2.0 | 3.33 | 99.92% |
| A2 | 12 | 50 | 2.0 | 3.08 | 99.95% |
| A3 | 12 | 50 | 2.0 | 3.25 | 99.95% |

训练集在极早期就几乎被完全拟合。由此得到三个判断：

- 50 epoch 对 A1–A3 很可能超过了学习主要模式所需的预算；
- 没有训练内 validation，使用最后 epoch checkpoint 无法判断更早 checkpoint 是否具有更好泛化；
- A4–A8 采用 A0 warm-start、较小 LR、20 epoch mixed fine-tuning + 5 epoch actual calibration 是合理方向，但仍建议记录每个 epoch 的训练内验证或固定少量候选 epoch。

---

## 10. 对 A4–A8 的直接建议

### 10.1 是否继续运行

建议继续运行 A4–A8。A3 已经达到理想的起点：它没有像 A1/A2 一样明显低于 A0，并且具有小幅正向趋势。后续实验应回答“能否把不稳定的小趋势转化为跨 seed 的稳定增益”。

### 10.2 首要观察指标

除总体均值外，必须同步观察：

1. seed 1/2/42 的 A4–A8−A0 delta；
2. D fold 是否还出现 `−4 pp / +6 pp` 这种翻转；
3. normal 与 fault 的分离结果；
4. worst-participant 和 worst-seed；
5. node 20、19、24、34、35 的 recall；
6. Stage 3 的增益是否保留；
7. changed history 子集和 unchanged history 子集各自的 consistency loss。

### 10.3 对各后续实验的预期

- **A4（paired + warm-start + calibration）**：最有希望降低 seed 依赖。重点不是追求很大的平均提升，而是让 seed 1/2 不再退化。
- **A5（consistency）**：应改善 actual/augmented view 的预测稳定性。若平均值上升但 fault macro-F1 继续下降，应降低 consistency weight 或提高 confidence threshold。
- **A6（plausibility sampling）**：当前 A2/A3 Kendall distance 已较小，A6 可能进一步降低有效 changed fraction。需要防止“采样约束过强导致几乎不增强”。
- **A7（tail-order auxiliary）**：可能对 Stage 2/3 和长 history 节点更有帮助；应重点检查 node 20、18、24。
- **A8（Tier-3 auxiliary）**：应同时检查 node accuracy 与 Tier-3 macro-F1，防止只提高粗粒度 Tier-3 而损害 35-node 区分。

### 10.4 推荐的继续运行顺序

1. 先完成 A4、A5；
2. 检查三 seed 是否都至少不低于 A0；
3. 再运行 A6；
4. A7、A8 最后加入辅助损失；
5. 若计算资源允许，补充一个当前代码路径下从头训练的 `A0-fresh`，只保存 metrics/predictions，权重可以在确认结果后归档到共享 artifact 目录。

---

## 11. 当前可用于论文的表述

### 11.1 推荐表述

> Restricting graph-valid reordering to active atomic tails partially recovered the degradation caused by unrestricted history shuffling. Preserving the original temporal recency of reordered events further improved mean test-all node accuracy from 89.92% to 91.09%, reaching performance parity with the actual-order Direct Fusion baseline (90.57%). The gain over the baseline was small and seed-dependent (+0.52 percentage points on average; 7 wins and 5 losses across 12 fold-seed pairs), motivating the paired consistency and calibration stages evaluated in A4–A8.

### 11.2 当前不建议的表述

- “A3 significantly outperforms A0.”
- “Atomic-tail augmentation consistently improves every participant and seed.”
- “A3 improves fault robustness.”

这些结论目前都不受结果支持。

---

## 12. 最终判断

用户提出的“A3 和 A0 基本达到同一水准”是正确的。进一步细化为：

- **总体均值：** A3 略优；
- **跨 participant：** 三 seed 均值均略优；
- **跨 seed：** 不稳定，提升主要由 seed 42 驱动；
- **normal：** 小幅正向；
- **fault：** accuracy 持平，macro-F1 略降；
- **相对旧 A1：** A3 有较明确改善；
- **论文级结论：** A3 是一个不损害基线性能、并具有小幅正向趋势的合格基础版本，但真正的亮点仍需 A4–A8 将其转化为稳定、可重复的提升。

---

## 13. 可复现分析入口

本报告的机器统计由以下脚本生成：

```powershell
python tools\analyze_a0_a3_detailed.py `
  --package-root . `
  --bootstrap-repetitions 20000
```

脚本只读取 `outputs/A0`–`outputs/A3`，不会修改训练结果。
