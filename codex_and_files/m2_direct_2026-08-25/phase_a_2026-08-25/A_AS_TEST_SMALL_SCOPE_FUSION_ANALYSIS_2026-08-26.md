# A_as_test 小范围 Phase A 融合实验整合分析

> 更新日期：2026-08-27；测试范围：`A_as_test`、`seed_1`、`all_runs`。摄像头/融合比较 A0–A6（不含未运行的 A7），并联合分析 S1–S12 右手 EMG/IMU 实验。

## 1. 结论摘要

- 在这一次单 fold、单 seed 测试中，按总体 Node Macro-F1 排名最高的是 **A2（双相机 0.5/0.5 概率后融合）**：93.75%，相对 A0 的 90.06% 为 **+3.70 pp**；其 Fault Node Macro-F1 变化为 **+8.02 pp**。
- **A1**：Node Macro-F1 +1.81 pp，Node accuracy +2.78 pp；Tier3 Macro-F1 +1.87 pp。
- **A2**：Node Macro-F1 +3.70 pp，Node accuracy +3.48 pp；Tier3 Macro-F1 +3.97 pp。
- **A3**：Node Macro-F1 -0.43 pp，Node accuracy -0.70 pp；Tier3 Macro-F1 -0.28 pp。
- **A4**：Node Macro-F1 +0.50 pp，Node accuracy +0.23 pp；Tier3 Macro-F1 +0.64 pp。
- **A5**：Node Macro-F1 +1.09 pp，Node accuracy +0.70 pp；Tier3 Macro-F1 +1.18 pp。
- **A6**：Node Macro-F1 +0.32 pp，Node accuracy +0.46 pp；Tier3 Macro-F1 +0.24 pp。
- **A2 的增益来源是一个本身就很强且与主视角互补的第二视角**：A1 单独已达到 94.43% accuracy / 91.87% Macro-F1，分别比 A0 +2.78 / +1.81 pp；A2 又比 A1 高 +0.70 / +1.89 pp。
- **A3 没有复现 A2 的双视角增益**：总体 Node Macro-F1/accuracy 相对 A0 分别为 -0.43 / -0.70 pp，相对 A2 分别为 -4.12 / -4.18 pp。这说明当前 gated residual/cross-view 训练并未把第二视角的互补性转化成更好的总体预测。
- **A1 单独视角的提升也具有子集差异**：Normal/Fault Node Macro-F1 相对 A0 分别为 +0.03 / +6.23 pp；Stage 1 为 -2.54 pp，Stage 2 为 +4.56 pp，Stage 3 为 +1.87 pp；最弱类 Recall 为 66.67%。
- **A5 是当前最有希望的可穿戴条件**：总体/Fault Node Macro-F1 分别比 A0 +1.09 / +3.53 pp，最弱类 Recall 从 33.33% 提到 41.67%；但 Stage 1 Macro-F1 下降 -0.83 pp。
- **A4 的信号较弱且存在指标分歧**：Fault Macro-F1 为 +1.13 pp，满足当前以 Macro-F1 定义的单次非劣方向；但 Fault accuracy 为 -0.73 pp，不能概括为全面改善。
- **A6 没有表现出 EMG+IMU 的简单叠加收益**：总体 Macro-F1 只比 A0 +0.32 pp，低于 A5，最弱类 Recall 仍为 33.33%；Stage 3 Macro-F1 还下降 -0.91 pp。
- **S1–S12 中 IMU 明显强于 EMG，最佳 sensor-only Node 为 S3**：Node Macro-F1 79.95%，相对 A0 仍低 -10.10 pp；但它修正了 22 个 A0 错误，A0+S3 oracle accuracy 可到 96.75%，说明 IMU 仍有可利用的非重叠信息。
- **1D encoder 的优劣随模态改变**：Dilated 在 EMG 的 M2/Direct Node/Direct Tier3 三种比较中均优于 ResNet10，但在 IMU 三种比较中均低于 ResNet10；不能为 EMG 与 IMU 固定同一个 backbone。
- **训练日志显示主要问题是跨参与者泛化而非训练不足**：S1–S3 末轮训练准确率均为 100%，而测试准确率分别为 25.29%、50.81%、84.69%；A3 从首轮起训练准确率就是 100%，可学习的稳健修错信号非常有限。
- 这些数字只能回答“在 A 被留作测试者且 seed=1 时有没有迹象”，尚不能回答“传感器是否稳定有价值”。原验收规则要求 12 个 fold×seed 多数正增益、最弱类 Recall 与 Node Macro-F1 同升、Fault 不退化、压力测试和硬件预算均通过。

## 2. 数据完整性与可比性

A0–A6 七组预测以及 S1–S12 均逐 `sample_name` 对齐到同一组 **431 clips**：Normal 294、Fault 137；Stage 1 66、Stage 2 308、Stage 3 57。所有条件保存的 node/Tier3 真值完全一致。

| 条件 | 训练前无传感器回退最大误差 | 训练后无传感器回退最大误差 |
| --- | --- | --- |
| A3 | 2.861e-06 | 2.861e-06 |
| A4 | 1.907e-06 | 1.907e-06 |
| A5 | 1.907e-06 | 1.907e-06 |
| A6 | 9.775e-06 | 9.775e-06 |

A3–A6 的误差量级约为浮点计算误差，支持“新增模态缺失时回到 A0 路径”的实现正确性；但缺失/失步情形仍需结合正式压力测试判断。

## 3. 总体结果

| 条件 | 输入/融合 | Node Acc | ΔAcc pp | Node Macro-F1 | ΔF1 pp | 最弱 Node Recall | Tier3 Acc | Tier3 Macro-F1 | ΔF1 pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 主相机 M2-Direct | 91.65% | — | 90.06% | — | 33.33% | 91.65% | 89.02% | — |
| A1 | 第二相机单独 M2-Direct | 94.43% | +2.78 | 91.87% | +1.81 | 66.67% | 94.43% | 90.89% | +1.87 |
| A2 | 双相机 0.5/0.5 概率后融合 | 95.13% | +3.48 | 93.75% | +3.70 | 66.67% | 95.13% | 92.99% | +3.97 |
| A3 | 双相机 gated residual/cross-view | 90.95% | -0.70 | 89.63% | -0.43 | 41.67% | 90.95% | 88.74% | -0.28 |
| A4 | 主相机 + 右手 IMU | 91.88% | +0.23 | 90.55% | +0.50 | 41.67% | 91.88% | 89.66% | +0.64 |
| A5 | 主相机 + 右手 EMG | 92.34% | +0.70 | 91.15% | +1.09 | 41.67% | 92.34% | 90.20% | +1.18 |
| A6 | 主相机 + 右手 EMG + IMU | 92.11% | +0.46 | 90.38% | +0.32 | 33.33% | 92.11% | 89.26% | +0.24 |

A1 已作为完整候选纳入后续所有子集、类别、混淆和 bootstrap 表；A2 的增益需要同时相对 A0 与 A1 判断，才能区分“第二相机本身更强”和“双视角互补”两种来源。

### 3.1 A0 错误修正与新引入错误

| 条件 | 相对 A0 改变预测 | 修正 A0 错误 | 破坏 A0 正确 | 净正确数 | 两者都错 |
| --- | --- | --- | --- | --- | --- |
| A1 | 47 | 26 | 14 | +12 | 10 |
| A2 | 27 | 19 | 4 | +15 | 17 |
| A3 | 8 | 2 | 5 | -3 | 34 |
| A4 | 7 | 4 | 3 | +1 | 32 |
| A5 | 8 | 5 | 2 | +3 | 31 |
| A6 | 4 | 3 | 1 | +2 | 33 |

“净正确数”直接对应 accuracy 的净变化；Macro-F1 还会受到这些修正/损害落在哪些类别以及预测精度变化的影响。

## 4. Normal / Fault 与 Stage 分解

| 子集 | 条件 | N | Node Acc | ΔAcc pp | Node Macro-F1 | ΔF1 pp | Tier3 Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Normal | A0 | 294 | 93.20% | — | 91.35% | — | 90.42% |
| Normal | A1 | 294 | 94.56% | +1.36 | 91.38% | +0.03 | 90.32% |
| Normal | A2 | 294 | 95.58% | +2.38 | 93.12% | +1.77 | 92.29% |
| Normal | A3 | 294 | 92.52% | -0.68 | 89.94% | -1.41 | 88.87% |
| Normal | A4 | 294 | 93.88% | +0.68 | 91.71% | +0.36 | 90.69% |
| Normal | A5 | 294 | 93.54% | +0.34 | 91.57% | +0.22 | 90.71% |
| Normal | A6 | 294 | 93.54% | +0.34 | 91.54% | +0.19 | 90.57% |
| Fault | A0 | 137 | 88.32% | — | 86.72% | — | 85.48% |
| Fault | A1 | 137 | 94.16% | +5.84 | 92.95% | +6.23 | 92.16% |
| Fault | A2 | 137 | 94.16% | +5.84 | 94.73% | +8.02 | 94.05% |
| Fault | A3 | 137 | 87.59% | -0.73 | 88.73% | +2.01 | 88.49% |
| Fault | A4 | 137 | 87.59% | -0.73 | 87.85% | +1.13 | 87.50% |
| Fault | A5 | 137 | 89.78% | +1.46 | 90.25% | +3.53 | 89.14% |
| Fault | A6 | 137 | 89.05% | +0.73 | 87.20% | +0.48 | 85.70% |

| Stage | 条件 | N | Node Acc | ΔAcc pp | Node Macro-F1 | ΔF1 pp |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | A0 | 66 | 89.39% | — | 92.51% | — |
| 1 | A1 | 66 | 87.88% | -1.52 | 89.97% | -2.54 |
| 1 | A2 | 66 | 92.42% | +3.03 | 93.72% | +1.21 |
| 1 | A3 | 66 | 87.88% | -1.52 | 91.68% | -0.83 |
| 1 | A4 | 66 | 89.39% | +0.00 | 92.51% | +0.00 |
| 1 | A5 | 66 | 87.88% | -1.52 | 91.68% | -0.83 |
| 1 | A6 | 66 | 90.91% | +1.52 | 93.50% | +0.99 |
| 2 | A0 | 308 | 92.86% | — | 92.99% | — |
| 2 | A1 | 308 | 96.75% | +3.90 | 97.55% | +4.56 |
| 2 | A2 | 308 | 96.10% | +3.25 | 96.73% | +3.74 |
| 2 | A3 | 308 | 92.21% | -0.65 | 92.21% | -0.78 |
| 2 | A4 | 308 | 92.86% | +0.00 | 92.96% | -0.03 |
| 2 | A5 | 308 | 93.51% | +0.65 | 93.60% | +0.61 |
| 2 | A6 | 308 | 93.51% | +0.65 | 93.66% | +0.67 |
| 3 | A0 | 57 | 87.72% | — | 92.27% | — |
| 3 | A1 | 57 | 89.47% | +1.75 | 94.14% | +1.87 |
| 3 | A2 | 57 | 92.98% | +5.26 | 96.36% | +4.09 |
| 3 | A3 | 57 | 87.72% | +0.00 | 93.03% | +0.76 |
| 3 | A4 | 57 | 89.47% | +1.75 | 92.97% | +0.70 |
| 3 | A5 | 57 | 91.23% | +3.51 | 95.27% | +3.00 |
| 3 | A6 | 57 | 85.96% | -1.75 | 91.36% | -0.91 |

## 5. 类别影响总览与 35 Node 详细分析

### 5.1 类别影响总览图

#### 5.1.1 A0–A6 摄像头与融合：35 Node

![35 Node Recall 与 F1 类别影响热图](analysis/a_as_test_seed_1/node_class_impact_heatmap.png)

上半图是各方法的绝对 Recall，类别按 **Node 编号 N1→N35** 排列，粗框表示 Recall 低于 80%。下半图是候选方法相对 A0 的 F1 变化：蓝色为提高、红色为下降。如果上半图两个方法的 Recall 数字相同，而下半图 F1 仍有颜色变化，表示该类别正确数没有变，但其他类别误报进来的数量发生了变化，从而改变了 Precision 和 F1。

#### 5.1.2 S1–S8 Sensor-only：35 Node

![S1-S8 35 Node 类别影响热图](analysis/a_as_test_seed_1/sensor_node_class_impact_heatmap.png)

上半图给出 A0 与 S1–S8 的绝对 Node Recall，下半图给出 sensor-only 相对 A0 的 Node F1 变化。横向同样按 **N1→N35** 排列，并与 5.1.1 使用相同的画布宽度、左右边距和节点列宽，因此两图可以按列逐节点比较。S9–S12 是 Direct Tier3，不产生 Node 预测，因此不应强行放入 35 Node 图。

#### 5.1.3 S1–S12 Sensor-only：31 Tier3

![S1-S12 31 Tier3 类别影响热图](analysis/a_as_test_seed_1/sensor_tier3_class_impact_heatmap.png)

该图在所有 S1–S12 都具备的 Tier3 输出口径下比较类别影响，因此补齐了 S9–S12。类别按 **Tier3 编号 T0→T30** 排列；图仍分为绝对 Recall 和相对 A0 的 F1 变化两部分，便于把模态差异与训练目标差异并列检查。

### 5.2 逐类别数值表

下表以真实类别为行。每个候选单元格为 `Recall变化 / F1变化 / 正确数净变化`；前两项单位均为百分点。小支持度类别的一两个 clip 就会造成很大的百分点波动，应同时看 support 与正确数。

| ID | Node | 支持 | A0 R | A0 F1 | A1 ΔR/ΔF1/Δ正确 | A2 ΔR/ΔF1/Δ正确 | A3 ΔR/ΔF1/Δ正确 | A4 ΔR/ΔF1/Δ正确 | A5 ΔR/ΔF1/Δ正确 | A6 ΔR/ΔF1/Δ正确 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | node_1_unlock_crimper | 6 | 66.7% | 72.7% | +0.0 / +0.0 / +0 | +0.0 / +7.3 / +0 | +0.0 / +0.0 / +0 | +0.0 / +7.3 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 2 | node_2_put_lock_on_table | 6 | 100.0% | 100.0% | -16.7 / -28.6 / -1 | +0.0 / -14.3 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 3 | node_3_turn_on_main_switch | 6 | 100.0% | 100.0% | +0.0 / -7.7 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 4 | node_4_turn_on_crimper | 6 | 83.3% | 83.3% | +0.0 / -11.9 / +0 | +0.0 / -6.4 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 5 | node_5_adjust_parameters | 6 | 100.0% | 100.0% | -33.3 / -20.0 / -2 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 6 | node_6_turn_on_air_compressor | 6 | 83.3% | 62.5% | +16.7 / +12.5 / +1 | +16.7 / +17.5 / +1 | +0.0 / +4.2 / +0 | +0.0 / +4.2 / +0 | +0.0 / +8.9 / +0 | +0.0 / +0.0 / +0 |
| 7 | node_7_turn_on_water_pump | 6 | 100.0% | 92.3% | -16.7 / -9.0 / -1 | -16.7 / -9.0 / -1 | -16.7 / -9.0 / -1 | +0.0 / +0.0 / +0 | -16.7 / -9.0 / -1 | +0.0 / +0.0 / +0 |
| 8 | node_8_turn_on_extractor_fan | 6 | 66.7% | 80.0% | +16.7 / +3.3 / +1 | +16.7 / +3.3 / +1 | +0.0 / -7.3 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +16.7 / +3.3 / +1 |
| 9 | node_9_move_pedal_to_safe_location | 6 | 100.0% | 100.0% | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 10 | node_10_remove_protection_cover_from_crimper | 6 | 83.3% | 90.9% | +16.7 / +9.1 / +1 | +16.7 / +9.1 / +1 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 11 | node_11_put_protection_cover_on_ground | 6 | 100.0% | 92.3% | +0.0 / +7.7 / +0 | +0.0 / +7.7 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 12 | node_12_take_plier_from_table | 24 | 95.8% | 75.4% | +4.2 / +13.5 / +1 | +0.0 / +9.8 / +0 | +0.0 / -1.2 / +0 | +0.0 / +0.0 / +0 | +0.0 / -1.2 / +0 | +0.0 / +0.0 / +0 |
| 13 | node_13_grip_sample_from_table_1 | 24 | 95.8% | 97.9% | +4.2 / +2.1 / +1 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 14 | node_14_put_sample_under_electrodes_1 | 24 | 100.0% | 100.0% | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 15 | node_15_press_pedal_1 | 24 | 100.0% | 100.0% | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 16 | node_16_put_sample_on_machine_table_1 | 21 | 100.0% | 100.0% | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 17 | node_17_grip_sample_from_machine_table_2 | 21 | 85.7% | 92.3% | +14.3 / +3.1 / +3 | +9.5 / +5.3 / +2 | +4.8 / +2.7 / +1 | +9.5 / +5.3 / +2 | +4.8 / +2.7 / +1 | +4.8 / +2.7 / +1 |
| 18 | node_18_reverse_sample | 21 | 100.0% | 89.4% | +0.0 / +10.6 / +0 | +0.0 / +8.3 / +0 | +0.0 / +1.9 / +0 | +0.0 / +4.0 / +0 | +0.0 / +4.0 / +0 | +0.0 / +1.9 / +0 |
| 19 | node_19_put_sample_on_machine_table_2 | 21 | 100.0% | 100.0% | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | -4.8 / -2.4 / -1 | +0.0 / +0.0 / +0 | -4.8 / -2.4 / -1 | +0.0 / +0.0 / +0 |
| 20 | node_20_grip_sample_from_machine_table_3 | 21 | 95.2% | 97.6% | +4.8 / +2.4 / +1 | +4.8 / +2.4 / +1 | +0.0 / -2.3 / +0 | +0.0 / +0.0 / +0 | +0.0 / -2.3 / +0 | +0.0 / +0.0 / +0 |
| 21 | node_21_put_sample_under_electrodes_2 | 21 | 100.0% | 97.7% | +0.0 / +2.3 / +0 | +0.0 / +2.3 / +0 | +0.0 / -4.3 / +0 | +0.0 / -4.3 / +0 | +0.0 / +2.3 / +0 | +0.0 / +2.3 / +0 |
| 22 | node_22_press_pedal_2 | 21 | 95.2% | 97.6% | +4.8 / +2.4 / +1 | +4.8 / +2.4 / +1 | -9.5 / -5.3 / -2 | -9.5 / -5.3 / -2 | +4.8 / +2.4 / +1 | +4.8 / +2.4 / +1 |
| 23 | node_23_inspect_sample | 17 | 94.1% | 97.0% | +5.9 / +3.0 / +1 | +5.9 / +3.0 / +1 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +5.9 / +3.0 / +1 | +0.0 / +0.0 / +0 |
| 24 | node_24_put_sample_on_table | 24 | 41.7% | 57.1% | +37.5 / +31.2 / +9 | +25.0 / +20.9 / +6 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 25 | node_25_put_plier_on_table | 24 | 100.0% | 100.0% | -20.8 / -11.6 / -5 | -4.2 / -2.1 / -1 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 26 | node_26_take_protection_cover_from_ground | 6 | 83.3% | 90.9% | +16.7 / +9.1 / +1 | +16.7 / +9.1 / +1 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 27 | node_27_put_protection_cover_on_crimper | 6 | 100.0% | 92.3% | +0.0 / +7.7 / +0 | +0.0 / +7.7 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 28 | node_28_turn_off_extractor_fan | 6 | 100.0% | 92.3% | -16.7 / -1.4 / -1 | -16.7 / -1.4 / -1 | -16.7 / -9.0 / -1 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | -16.7 / -1.4 / -1 |
| 29 | node_29_turn_off_water_pump | 6 | 83.3% | 90.9% | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / -7.6 / +0 | +0.0 / +0.0 / +0 | +0.0 / -7.6 / +0 | +0.0 / +0.0 / +0 |
| 30 | node_30_turn_off_air_compressor | 6 | 33.3% | 44.4% | +50.0 / +46.5 / +3 | +50.0 / +46.5 / +3 | +16.7 / +15.6 / +1 | +16.7 / +15.6 / +1 | +33.3 / +28.3 / +2 | +0.0 / +0.0 / +0 |
| 31 | node_31_move_pedal_to_original_place | 6 | 100.0% | 100.0% | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / -7.7 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 32 | node_32_turn_off_crimper | 6 | 83.3% | 90.9% | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +16.7 / +9.1 / +1 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 33 | node_33_turn_off_main_switch | 5 | 100.0% | 100.0% | -20.0 / -11.1 / -1 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 34 | node_34_take_lock_from_table | 5 | 100.0% | 90.9% | +0.0 / -7.6 / +0 | +0.0 / +0.0 / +0 | +0.0 / +9.1 / +0 | +0.0 / +0.0 / +0 | +0.0 / +9.1 / +0 | +0.0 / +0.0 / +0 |
| 35 | node_35_lock_crimper | 5 | 100.0% | 83.3% | -20.0 / +5.6 / -1 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | -20.0 / -10.6 / -1 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |

### 5.3 各融合方法最明显的 Node 增益与退化

**A1（第二相机单独 M2-Direct）**

- Recall 增益最大：`node_30_turn_off_air_compressor` (n=6, +50.0 pp, Δ正确=+3)；`node_24_put_sample_on_table` (n=24, +37.5 pp, Δ正确=+9)；`node_8_turn_on_extractor_fan` (n=6, +16.7 pp, Δ正确=+1)；`node_6_turn_on_air_compressor` (n=6, +16.7 pp, Δ正确=+1)；`node_10_remove_protection_cover_from_crimper` (n=6, +16.7 pp, Δ正确=+1)；`node_26_take_protection_cover_from_ground` (n=6, +16.7 pp, Δ正确=+1)
- Recall 退化最大：`node_5_adjust_parameters` (n=6, -33.3 pp, Δ正确=-2)；`node_25_put_plier_on_table` (n=24, -20.8 pp, Δ正确=-5)；`node_35_lock_crimper` (n=5, -20.0 pp, Δ正确=-1)；`node_33_turn_off_main_switch` (n=5, -20.0 pp, Δ正确=-1)；`node_28_turn_off_extractor_fan` (n=6, -16.7 pp, Δ正确=-1)；`node_7_turn_on_water_pump` (n=6, -16.7 pp, Δ正确=-1)
- 35 个受支持 Node 中：Recall 改善 12 类、退化 7 类、不变 16 类。

**A2（双相机 0.5/0.5 概率后融合）**

- Recall 增益最大：`node_30_turn_off_air_compressor` (n=6, +50.0 pp, Δ正确=+3)；`node_24_put_sample_on_table` (n=24, +25.0 pp, Δ正确=+6)；`node_8_turn_on_extractor_fan` (n=6, +16.7 pp, Δ正确=+1)；`node_6_turn_on_air_compressor` (n=6, +16.7 pp, Δ正确=+1)；`node_10_remove_protection_cover_from_crimper` (n=6, +16.7 pp, Δ正确=+1)；`node_26_take_protection_cover_from_ground` (n=6, +16.7 pp, Δ正确=+1)
- Recall 退化最大：`node_28_turn_off_extractor_fan` (n=6, -16.7 pp, Δ正确=-1)；`node_7_turn_on_water_pump` (n=6, -16.7 pp, Δ正确=-1)；`node_25_put_plier_on_table` (n=24, -4.2 pp, Δ正确=-1)
- 35 个受支持 Node 中：Recall 改善 10 类、退化 3 类、不变 22 类。

**A3（双相机 gated residual/cross-view）**

- Recall 增益最大：`node_30_turn_off_air_compressor` (n=6, +16.7 pp, Δ正确=+1)；`node_17_grip_sample_from_machine_table_2` (n=21, +4.8 pp, Δ正确=+1)
- Recall 退化最大：`node_28_turn_off_extractor_fan` (n=6, -16.7 pp, Δ正确=-1)；`node_7_turn_on_water_pump` (n=6, -16.7 pp, Δ正确=-1)；`node_22_press_pedal_2` (n=21, -9.5 pp, Δ正确=-2)；`node_19_put_sample_on_machine_table_2` (n=21, -4.8 pp, Δ正确=-1)
- 35 个受支持 Node 中：Recall 改善 2 类、退化 4 类、不变 29 类。

**A4（主相机 + 右手 IMU）**

- Recall 增益最大：`node_30_turn_off_air_compressor` (n=6, +16.7 pp, Δ正确=+1)；`node_32_turn_off_crimper` (n=6, +16.7 pp, Δ正确=+1)；`node_17_grip_sample_from_machine_table_2` (n=21, +9.5 pp, Δ正确=+2)
- Recall 退化最大：`node_35_lock_crimper` (n=5, -20.0 pp, Δ正确=-1)；`node_22_press_pedal_2` (n=21, -9.5 pp, Δ正确=-2)
- 35 个受支持 Node 中：Recall 改善 3 类、退化 2 类、不变 30 类。

**A5（主相机 + 右手 EMG）**

- Recall 增益最大：`node_30_turn_off_air_compressor` (n=6, +33.3 pp, Δ正确=+2)；`node_23_inspect_sample` (n=17, +5.9 pp, Δ正确=+1)；`node_17_grip_sample_from_machine_table_2` (n=21, +4.8 pp, Δ正确=+1)；`node_22_press_pedal_2` (n=21, +4.8 pp, Δ正确=+1)
- Recall 退化最大：`node_7_turn_on_water_pump` (n=6, -16.7 pp, Δ正确=-1)；`node_19_put_sample_on_machine_table_2` (n=21, -4.8 pp, Δ正确=-1)
- 35 个受支持 Node 中：Recall 改善 4 类、退化 2 类、不变 29 类。

**A6（主相机 + 右手 EMG + IMU）**

- Recall 增益最大：`node_8_turn_on_extractor_fan` (n=6, +16.7 pp, Δ正确=+1)；`node_17_grip_sample_from_machine_table_2` (n=21, +4.8 pp, Δ正确=+1)；`node_22_press_pedal_2` (n=21, +4.8 pp, Δ正确=+1)
- Recall 退化最大：`node_28_turn_off_extractor_fan` (n=6, -16.7 pp, Δ正确=-1)
- 35 个受支持 Node 中：Recall 改善 3 类、退化 1 类、不变 31 类。

### 5.4 各方法低 Recall Node 与误分类样本名称

这里将低 Recall 预定义为 **Recall < 80%**。A0–A6 表中列出造成低 Recall 的全部误分类样本；S1–S8 因错误较多，每个低 Recall Node 最多用固定随机种子抽取 10 个误分类样本。原备份报告的 A0、A1、A2、A4、A5、A6 备注已按方法、真实 Node 和样本名逐条导入；A3 与 S1–S8 的备注栏保持空白，供后续人工检查。如需同时查看这些类别中预测正确的样本，可打开 `analysis/a_as_test_seed_1/LOW_RECALL_NODE_SAMPLE_INDEX.md`；S1–S8 的完整列表见 `SENSOR_LOW_RECALL_NODE_SAMPLE_INDEX.md`；便于筛选的逐样本表分别为 `low_recall_node_samples.csv` 和 `sensor_low_recall_node_samples.csv`。

#### 5.4.1 A0–A6 摄像头与融合

##### A0 — 主相机 M2-Direct

| 低 Recall Node | 支持 | 正确 | Recall | 误分类样本 → 预测 Node | 备注 |
| --- | --- | --- | --- | --- | --- |
| node_1_unlock_crimper | 6 | 4/6 | 66.7% | `sample_000051` → `node_4_turn_on_crimper`<br>`sample_000130` → `node_35_lock_crimper` | 样本质量良好<br>样本质量良 |
| node_8_turn_on_extractor_fan | 6 | 4/6 | 66.7% | `sample_000134` → `node_28_turn_off_extractor_fan`<br>`sample_000278` → `node_6_turn_on_air_compressor` | 样本质量良好<br>有严重遮挡 |
| node_24_put_sample_on_table | 24 | 10/24 | 41.7% | `sample_000038` → `node_12_take_plier_from_table`<br>`sample_000073` → `node_12_take_plier_from_table`<br>`sample_000087` → `node_12_take_plier_from_table`<br>`sample_000150` → `node_12_take_plier_from_table`<br>`sample_000178` → `node_12_take_plier_from_table`<br>`sample_000226` → `node_34_take_lock_from_table`<br>`sample_000232` → `node_12_take_plier_from_table`<br>`sample_000238` → `node_12_take_plier_from_table`<br>`sample_000259` → `node_12_take_plier_from_table`<br>`sample_000265` → `node_12_take_plier_from_table`<br>`sample_000309` → `node_12_take_plier_from_table`<br>`sample_000315` → `node_12_take_plier_from_table`<br>`sample_000329` → `node_12_take_plier_from_table`<br>`sample_000430` → `node_12_take_plier_from_table` | 样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好 |
| node_30_turn_off_air_compressor | 6 | 2/6 | 33.3% | `sample_000041` → `node_6_turn_on_air_compressor`<br>`sample_000118` → `node_6_turn_on_air_compressor`<br>`sample_000269` → `node_6_turn_on_air_compressor`<br>`sample_000336` → `node_6_turn_on_air_compressor` | 有遮挡<br>样本质量良好<br>样本质量良好<br>样本质量良好 |

##### A1 — 第二相机单独 M2-Direct

| 低 Recall Node | 支持 | 正确 | Recall | 误分类样本 → 预测 Node | 备注 |
| --- | --- | --- | --- | --- | --- |
| node_1_unlock_crimper | 6 | 4/6 | 66.7% | `sample_000051` → `node_4_turn_on_crimper`<br>`sample_000130` → `node_4_turn_on_crimper` | 样本质量良好<br>样本质量良好 |
| node_5_adjust_parameters | 6 | 4/6 | 66.7% | `sample_000137` → `node_6_turn_on_air_compressor`<br>`sample_000214` → `node_17_grip_sample_from_machine_table_2` | 有遮挡<br>有遮挡 |
| node_24_put_sample_on_table | 24 | 19/24 | 79.2% | `sample_000150` → `node_2_put_lock_on_table`<br>`sample_000202` → `node_12_take_plier_from_table`<br>`sample_000226` → `node_2_put_lock_on_table`<br>`sample_000232` → `node_2_put_lock_on_table`<br>`sample_000315` → `node_34_take_lock_from_table` | 样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好 |
| node_25_put_plier_on_table | 24 | 19/24 | 79.2% | `sample_000203` → `node_12_take_plier_from_table`<br>`sample_000227` → `node_12_take_plier_from_table`<br>`sample_000266` → `node_12_take_plier_from_table`<br>`sample_000368` → `node_12_take_plier_from_table`<br>`sample_000431` → `node_12_take_plier_from_table` | 样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好 |

##### A2 — 双相机 0.5/0.5 概率后融合

| 低 Recall Node | 支持 | 正确 | Recall | 误分类样本 → 预测 Node | 备注 |
| --- | --- | --- | --- | --- | --- |
| node_1_unlock_crimper | 6 | 4/6 | 66.7% | `sample_000051` → `node_4_turn_on_crimper`<br>`sample_000130` → `node_35_lock_crimper` | 样本质量良好<br>样本质量良好 |
| node_24_put_sample_on_table | 24 | 16/24 | 66.7% | `sample_000073` → `node_12_take_plier_from_table`<br>`sample_000150` → `node_2_put_lock_on_table`<br>`sample_000226` → `node_2_put_lock_on_table`<br>`sample_000232` → `node_12_take_plier_from_table`<br>`sample_000238` → `node_12_take_plier_from_table`<br>`sample_000259` → `node_12_take_plier_from_table`<br>`sample_000309` → `node_12_take_plier_from_table`<br>`sample_000315` → `node_34_take_lock_from_table` | 样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好 |

##### A3 — 双相机 gated residual/cross-view

| 低 Recall Node | 支持 | 正确 | Recall | 误分类样本 → 预测 Node | 备注 |
| --- | --- | --- | --- | --- | --- |
| node_1_unlock_crimper | 6 | 4/6 | 66.7% | `sample_000051` → `node_4_turn_on_crimper`<br>`sample_000130` → `node_35_lock_crimper` |  |
| node_8_turn_on_extractor_fan | 6 | 4/6 | 66.7% | `sample_000134` → `node_28_turn_off_extractor_fan`<br>`sample_000278` → `node_6_turn_on_air_compressor` |  |
| node_24_put_sample_on_table | 24 | 10/24 | 41.7% | `sample_000038` → `node_12_take_plier_from_table`<br>`sample_000073` → `node_12_take_plier_from_table`<br>`sample_000087` → `node_12_take_plier_from_table`<br>`sample_000150` → `node_12_take_plier_from_table`<br>`sample_000178` → `node_12_take_plier_from_table`<br>`sample_000226` → `node_12_take_plier_from_table`<br>`sample_000232` → `node_12_take_plier_from_table`<br>`sample_000238` → `node_12_take_plier_from_table`<br>`sample_000259` → `node_12_take_plier_from_table`<br>`sample_000265` → `node_12_take_plier_from_table`<br>`sample_000309` → `node_12_take_plier_from_table`<br>`sample_000315` → `node_12_take_plier_from_table`<br>`sample_000329` → `node_12_take_plier_from_table`<br>`sample_000430` → `node_12_take_plier_from_table` |  |
| node_30_turn_off_air_compressor | 6 | 3/6 | 50.0% | `sample_000041` → `node_6_turn_on_air_compressor`<br>`sample_000118` → `node_6_turn_on_air_compressor`<br>`sample_000336` → `node_6_turn_on_air_compressor` |  |

##### A4 — 主相机 + 右手 IMU

| 低 Recall Node | 支持 | 正确 | Recall | 误分类样本 → 预测 Node | 备注 |
| --- | --- | --- | --- | --- | --- |
| node_1_unlock_crimper | 6 | 4/6 | 66.7% | `sample_000051` → `node_4_turn_on_crimper`<br>`sample_000130` → `node_35_lock_crimper` | 样本质量良好<br>样本质量良好 |
| node_8_turn_on_extractor_fan | 6 | 4/6 | 66.7% | `sample_000134` → `node_28_turn_off_extractor_fan`<br>`sample_000278` → `node_6_turn_on_air_compressor` | 样本质量良好<br>有严重遮挡 |
| node_24_put_sample_on_table | 24 | 10/24 | 41.7% | `sample_000038` → `node_12_take_plier_from_table`<br>`sample_000073` → `node_12_take_plier_from_table`<br>`sample_000087` → `node_12_take_plier_from_table`<br>`sample_000150` → `node_12_take_plier_from_table`<br>`sample_000178` → `node_12_take_plier_from_table`<br>`sample_000226` → `node_34_take_lock_from_table`<br>`sample_000232` → `node_12_take_plier_from_table`<br>`sample_000238` → `node_12_take_plier_from_table`<br>`sample_000259` → `node_12_take_plier_from_table`<br>`sample_000265` → `node_12_take_plier_from_table`<br>`sample_000309` → `node_12_take_plier_from_table`<br>`sample_000315` → `node_12_take_plier_from_table`<br>`sample_000329` → `node_12_take_plier_from_table`<br>`sample_000430` → `node_12_take_plier_from_table` | 样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好 |
| node_30_turn_off_air_compressor | 6 | 3/6 | 50.0% | `sample_000041` → `node_6_turn_on_air_compressor`<br>`sample_000118` → `node_6_turn_on_air_compressor`<br>`sample_000336` → `node_6_turn_on_air_compressor` | 有遮挡<br>样本质量良好<br>样本质量良好 |

##### A5 — 主相机 + 右手 EMG

| 低 Recall Node | 支持 | 正确 | Recall | 误分类样本 → 预测 Node | 备注 |
| --- | --- | --- | --- | --- | --- |
| node_1_unlock_crimper | 6 | 4/6 | 66.7% | `sample_000051` → `node_4_turn_on_crimper`<br>`sample_000130` → `node_35_lock_crimper` | 样本质量良好<br>样本质量良好 |
| node_8_turn_on_extractor_fan | 6 | 4/6 | 66.7% | `sample_000134` → `node_28_turn_off_extractor_fan`<br>`sample_000278` → `node_6_turn_on_air_compressor` | 样本质量良好<br>有严重遮挡 |
| node_24_put_sample_on_table | 24 | 10/24 | 41.7% | `sample_000038` → `node_12_take_plier_from_table`<br>`sample_000073` → `node_12_take_plier_from_table`<br>`sample_000087` → `node_12_take_plier_from_table`<br>`sample_000150` → `node_12_take_plier_from_table`<br>`sample_000178` → `node_12_take_plier_from_table`<br>`sample_000226` → `node_12_take_plier_from_table`<br>`sample_000232` → `node_12_take_plier_from_table`<br>`sample_000238` → `node_12_take_plier_from_table`<br>`sample_000259` → `node_12_take_plier_from_table`<br>`sample_000265` → `node_12_take_plier_from_table`<br>`sample_000309` → `node_12_take_plier_from_table`<br>`sample_000315` → `node_12_take_plier_from_table`<br>`sample_000329` → `node_12_take_plier_from_table`<br>`sample_000430` → `node_12_take_plier_from_table` | 样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好 |
| node_30_turn_off_air_compressor | 6 | 4/6 | 66.7% | `sample_000118` → `node_6_turn_on_air_compressor`<br>`sample_000336` → `node_6_turn_on_air_compressor` | 样本质量良好<br>样本质量良好 |

##### A6 — 主相机 + 右手 EMG + IMU

| 低 Recall Node | 支持 | 正确 | Recall | 误分类样本 → 预测 Node | 备注 |
| --- | --- | --- | --- | --- | --- |
| node_1_unlock_crimper | 6 | 4/6 | 66.7% | `sample_000051` → `node_4_turn_on_crimper`<br>`sample_000130` → `node_35_lock_crimper` | 样本质量良好<br>样本质量良好 |
| node_24_put_sample_on_table | 24 | 10/24 | 41.7% | `sample_000038` → `node_12_take_plier_from_table`<br>`sample_000073` → `node_12_take_plier_from_table`<br>`sample_000087` → `node_12_take_plier_from_table`<br>`sample_000150` → `node_12_take_plier_from_table`<br>`sample_000178` → `node_12_take_plier_from_table`<br>`sample_000226` → `node_34_take_lock_from_table`<br>`sample_000232` → `node_12_take_plier_from_table`<br>`sample_000238` → `node_12_take_plier_from_table`<br>`sample_000259` → `node_12_take_plier_from_table`<br>`sample_000265` → `node_12_take_plier_from_table`<br>`sample_000309` → `node_12_take_plier_from_table`<br>`sample_000315` → `node_12_take_plier_from_table`<br>`sample_000329` → `node_12_take_plier_from_table`<br>`sample_000430` → `node_12_take_plier_from_table` | 样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好<br>样本质量良好 |
| node_30_turn_off_air_compressor | 6 | 2/6 | 33.3% | `sample_000041` → `node_6_turn_on_air_compressor`<br>`sample_000118` → `node_6_turn_on_air_compressor`<br>`sample_000269` → `node_6_turn_on_air_compressor`<br>`sample_000336` → `node_6_turn_on_air_compressor` | 有遮挡<br>样本质量良好<br>样本质量良好<br>样本质量良好 |

#### 5.4.2 S1–S8 Sensor-only

以下每个低 Recall Node 最多展示 10 个误分类样本；抽样是确定性的，重复生成报告不会无故换样本。括号中的 `显示 x/y` 表示本表展示数/该类别全部误分类数。完整错误清单仍保存在上述 Sensor 索引与 CSV 中。

##### S1 — EMG ResNet10 Tier3→M2 Node

| 低 Recall Node | 支持 | 正确 | Recall | 随机抽取误分类样本 → 预测 Node | 备注 |
| --- | --- | --- | --- | --- | --- |
| node_1_unlock_crimper | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000005` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000051` → `node_9_move_pedal_to_safe_location`<br>`sample_000130` → `node_9_move_pedal_to_safe_location`<br>`sample_000204` → `node_9_move_pedal_to_safe_location`<br>`sample_000274` → `node_9_move_pedal_to_safe_location` |  |
| node_2_put_lock_on_table | 6 | 2/6 | 33.3% | （显示 4/4）<br>`sample_000131` → `node_34_take_lock_from_table`<br>`sample_000205` → `node_25_put_plier_on_table`<br>`sample_000275` → `node_7_turn_on_water_pump`<br>`sample_000380` → `node_25_put_plier_on_table` |  |
| node_3_turn_on_main_switch | 6 | 4/6 | 66.7% | （显示 2/2）<br>`sample_000276` → `node_5_adjust_parameters`<br>`sample_000381` → `node_18_reverse_sample` |  |
| node_4_turn_on_crimper | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000007` → `node_32_turn_off_crimper`<br>`sample_000053` → `node_16_put_sample_on_machine_table_1`<br>`sample_000132` → `node_6_turn_on_air_compressor`<br>`sample_000210` → `node_3_turn_on_main_switch`<br>`sample_000280` → `node_3_turn_on_main_switch`<br>`sample_000382` → `node_3_turn_on_main_switch` |  |
| node_5_adjust_parameters | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000011` → `node_23_inspect_sample`<br>`sample_000060` → `node_23_inspect_sample`<br>`sample_000137` → `node_23_inspect_sample`<br>`sample_000214` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000284` → `node_23_inspect_sample`<br>`sample_000389` → `node_20_grip_sample_from_machine_table_3` |  |
| node_6_turn_on_air_compressor | 6 | 2/6 | 33.3% | （显示 4/4）<br>`sample_000003` → `node_19_put_sample_on_machine_table_2`<br>`sample_000209` → `node_3_turn_on_main_switch`<br>`sample_000279` → `node_25_put_plier_on_table`<br>`sample_000385` → `node_3_turn_on_main_switch` |  |
| node_7_turn_on_water_pump | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000002` → `node_33_turn_off_main_switch`<br>`sample_000055` → `node_3_turn_on_main_switch`<br>`sample_000135` → `node_3_turn_on_main_switch`<br>`sample_000207` → `node_2_put_lock_on_table`<br>`sample_000277` → `node_3_turn_on_main_switch`<br>`sample_000384` → `node_31_move_pedal_to_original_place` |  |
| node_8_turn_on_extractor_fan | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000004` → `node_33_turn_off_main_switch`<br>`sample_000054` → `node_2_put_lock_on_table`<br>`sample_000134` → `node_3_turn_on_main_switch`<br>`sample_000208` → `node_3_turn_on_main_switch`<br>`sample_000278` → `node_31_move_pedal_to_original_place`<br>`sample_000383` → `node_3_turn_on_main_switch` |  |
| node_9_move_pedal_to_safe_location | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000010` → `node_3_turn_on_main_switch`<br>`sample_000059` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000136` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000211` → `node_31_move_pedal_to_original_place`<br>`sample_000281` → `node_32_turn_off_crimper`<br>`sample_000388` → `node_32_turn_off_crimper` |  |
| node_10_remove_protection_cover_from_crimper | 6 | 4/6 | 66.7% | （显示 2/2）<br>`sample_000128` → `node_6_turn_on_air_compressor`<br>`sample_000386` → `node_20_grip_sample_from_machine_table_3` |  |
| node_11_put_protection_cover_on_ground | 6 | 4/6 | 66.7% | （显示 2/2）<br>`sample_000009` → `node_6_turn_on_air_compressor`<br>`sample_000387` → `node_5_adjust_parameters` |  |
| node_12_take_plier_from_table | 24 | 6/24 | 25.0% | （显示 10/18）<br>`sample_000012` → `node_2_put_lock_on_table`<br>`sample_000103` → `node_2_put_lock_on_table`<br>`sample_000152` → `node_2_put_lock_on_table`<br>`sample_000228` → `node_6_turn_on_air_compressor`<br>`sample_000240` → `node_2_put_lock_on_table`<br>`sample_000261` → `node_2_put_lock_on_table`<br>`sample_000285` → `node_25_put_plier_on_table`<br>`sample_000298` → `node_2_put_lock_on_table`<br>`sample_000311` → `node_2_put_lock_on_table`<br>`sample_000317` → `node_25_put_plier_on_table` |  |
| node_13_grip_sample_from_table_1 | 24 | 6/24 | 25.0% | （显示 10/18）<br>`sample_000090` → `node_25_put_plier_on_table`<br>`sample_000167` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000191` → `node_18_reverse_sample`<br>`sample_000241` → `node_12_take_plier_from_table`<br>`sample_000262` → `node_18_reverse_sample`<br>`sample_000299` → `node_2_put_lock_on_table`<br>`sample_000318` → `node_18_reverse_sample`<br>`sample_000342` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000356` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000419` → `node_18_reverse_sample` |  |
| node_14_put_sample_under_electrodes_1 | 24 | 0/24 | 0.0% | （显示 10/24）<br>`sample_000063` → `node_13_grip_sample_from_table_1`<br>`sample_000105` → `node_12_take_plier_from_table`<br>`sample_000140` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000154` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000168` → `node_18_reverse_sample`<br>`sample_000300` → `node_25_put_plier_on_table`<br>`sample_000313` → `node_18_reverse_sample`<br>`sample_000319` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000406` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000420` → `node_10_remove_protection_cover_from_crimper` |  |
| node_15_press_pedal_1 | 24 | 2/24 | 8.3% | （显示 10/22）<br>`sample_000015` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000029` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000064` → `node_1_unlock_crimper`<br>`sample_000092` → `node_23_inspect_sample`<br>`sample_000141` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000169` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000193` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000218` → `node_23_inspect_sample`<br>`sample_000237` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000288` → `node_23_inspect_sample` |  |
| node_16_put_sample_on_machine_table_1 | 21 | 0/21 | 0.0% | （显示 10/21）<br>`sample_000030` → `node_2_put_lock_on_table`<br>`sample_000065` → `node_2_put_lock_on_table`<br>`sample_000156` → `node_2_put_lock_on_table`<br>`sample_000170` → `node_2_put_lock_on_table`<br>`sample_000194` → `node_2_put_lock_on_table`<br>`sample_000289` → `node_5_adjust_parameters`<br>`sample_000302` → `node_2_put_lock_on_table`<br>`sample_000394` → `node_2_put_lock_on_table`<br>`sample_000408` → `node_5_adjust_parameters`<br>`sample_000422` → `node_2_put_lock_on_table` |  |
| node_17_grip_sample_from_machine_table_2 | 21 | 11/21 | 52.4% | （显示 10/10）<br>`sample_000017` → `node_33_turn_off_main_switch`<br>`sample_000031` → `node_18_reverse_sample`<br>`sample_000080` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000108` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000143` → `node_26_take_protection_cover_from_ground`<br>`sample_000220` → `node_26_take_protection_cover_from_ground`<br>`sample_000252` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000290` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000346` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000360` → `node_10_remove_protection_cover_from_crimper` |  |
| node_18_reverse_sample | 21 | 9/21 | 42.9% | （显示 10/12）<br>`sample_000032` → `node_17_grip_sample_from_machine_table_2`<br>`sample_000081` → `node_28_turn_off_extractor_fan`<br>`sample_000109` → `node_19_put_sample_on_machine_table_2`<br>`sample_000144` → `node_28_turn_off_extractor_fan`<br>`sample_000158` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000347` → `node_19_put_sample_on_machine_table_2`<br>`sample_000361` → `node_19_put_sample_on_machine_table_2`<br>`sample_000396` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000410` → `node_28_turn_off_extractor_fan`<br>`sample_000424` → `node_10_remove_protection_cover_from_crimper` |  |
| node_19_put_sample_on_machine_table_2 | 21 | 16/21 | 76.2% | （显示 5/5）<br>`sample_000019` → `node_8_turn_on_extractor_fan`<br>`sample_000033` → `node_30_turn_off_air_compressor`<br>`sample_000254` → `node_25_put_plier_on_table`<br>`sample_000292` → `node_3_turn_on_main_switch`<br>`sample_000425` → `node_31_move_pedal_to_original_place` |  |
| node_20_grip_sample_from_machine_table_3 | 21 | 3/21 | 14.3% | （显示 10/18）<br>`sample_000020` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000069` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000111` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000160` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000198` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000223` → `node_12_take_plier_from_table`<br>`sample_000248` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000293` → `node_28_turn_off_extractor_fan`<br>`sample_000349` → `node_18_reverse_sample`<br>`sample_000363` → `node_10_remove_protection_cover_from_crimper` |  |
| node_21_put_sample_under_electrodes_2 | 21 | 0/21 | 0.0% | （显示 10/21）<br>`sample_000035` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000112` → `node_34_take_lock_from_table`<br>`sample_000161` → `node_18_reverse_sample`<br>`sample_000175` → `node_32_turn_off_crimper`<br>`sample_000199` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000256` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000294` → `node_28_turn_off_extractor_fan`<br>`sample_000364` → `node_12_take_plier_from_table`<br>`sample_000399` → `node_25_put_plier_on_table`<br>`sample_000413` → `node_10_remove_protection_cover_from_crimper` |  |
| node_22_press_pedal_2 | 21 | 0/21 | 0.0% | （显示 10/21）<br>`sample_000071` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000113` → `node_23_inspect_sample`<br>`sample_000162` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000176` → `node_1_unlock_crimper`<br>`sample_000308` → `node_23_inspect_sample`<br>`sample_000327` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000351` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000365` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000400` → `node_35_lock_crimper`<br>`sample_000414` → `node_10_remove_protection_cover_from_crimper` |  |
| node_23_inspect_sample | 17 | 4/17 | 23.5% | （显示 10/13）<br>`sample_000023` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000086` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000100` → `node_35_lock_crimper`<br>`sample_000114` → `node_1_unlock_crimper`<br>`sample_000163` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000177` → `node_27_put_protection_cover_on_crimper`<br>`sample_000201` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000328` → `node_1_unlock_crimper`<br>`sample_000352` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000366` → `node_10_remove_protection_cover_from_crimper` |  |
| node_24_put_sample_on_table | 24 | 0/24 | 0.0% | （显示 10/24）<br>`sample_000024` → `node_3_turn_on_main_switch`<br>`sample_000038` → `node_2_put_lock_on_table`<br>`sample_000087` → `node_31_move_pedal_to_original_place`<br>`sample_000101` → `node_2_put_lock_on_table`<br>`sample_000226` → `node_3_turn_on_main_switch`<br>`sample_000259` → `node_28_turn_off_extractor_fan`<br>`sample_000296` → `node_25_put_plier_on_table`<br>`sample_000309` → `node_3_turn_on_main_switch`<br>`sample_000315` → `node_2_put_lock_on_table`<br>`sample_000367` → `node_15_press_pedal_1` |  |
| node_26_take_protection_cover_from_ground | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000047` → `node_13_grip_sample_from_table_1`<br>`sample_000124` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000181` → `node_25_put_plier_on_table`<br>`sample_000271` → `node_25_put_plier_on_table`<br>`sample_000376` → `node_13_grip_sample_from_table_1` |  |
| node_27_put_protection_cover_on_crimper | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000048` → `node_9_move_pedal_to_safe_location`<br>`sample_000125` → `node_13_grip_sample_from_table_1`<br>`sample_000182` → `node_13_grip_sample_from_table_1`<br>`sample_000272` → `node_16_put_sample_on_machine_table_1`<br>`sample_000335` → `node_9_move_pedal_to_safe_location` |  |
| node_28_turn_off_extractor_fan | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000042` → `node_25_put_plier_on_table`<br>`sample_000119` → `node_8_turn_on_extractor_fan`<br>`sample_000185` → `node_8_turn_on_extractor_fan`<br>`sample_000268` → `node_3_turn_on_main_switch`<br>`sample_000337` → `node_3_turn_on_main_switch`<br>`sample_000370` → `node_25_put_plier_on_table` |  |
| node_29_turn_off_water_pump | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000043` → `node_25_put_plier_on_table`<br>`sample_000120` → `node_25_put_plier_on_table`<br>`sample_000186` → `node_34_take_lock_from_table`<br>`sample_000267` → `node_25_put_plier_on_table`<br>`sample_000338` → `node_3_turn_on_main_switch`<br>`sample_000372` → `node_25_put_plier_on_table` |  |
| node_30_turn_off_air_compressor | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000041` → `node_3_turn_on_main_switch`<br>`sample_000118` → `node_33_turn_off_main_switch`<br>`sample_000184` → `node_33_turn_off_main_switch`<br>`sample_000269` → `node_25_put_plier_on_table`<br>`sample_000336` → `node_3_turn_on_main_switch`<br>`sample_000371` → `node_3_turn_on_main_switch` |  |
| node_32_turn_off_crimper | 6 | 2/6 | 33.3% | （显示 4/4）<br>`sample_000117` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000183` → `node_31_move_pedal_to_original_place`<br>`sample_000270` → `node_25_put_plier_on_table`<br>`sample_000331` → `node_6_turn_on_air_compressor` |  |
| node_34_take_lock_from_table | 5 | 0/5 | 0.0% | （显示 5/5）<br>`sample_000045` → `node_2_put_lock_on_table`<br>`sample_000122` → `node_18_reverse_sample`<br>`sample_000188` → `node_12_take_plier_from_table`<br>`sample_000332` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000374` → `node_2_put_lock_on_table` |  |
| node_35_lock_crimper | 5 | 0/5 | 0.0% | （显示 5/5）<br>`sample_000046` → `node_9_move_pedal_to_safe_location`<br>`sample_000123` → `node_9_move_pedal_to_safe_location`<br>`sample_000189` → `node_9_move_pedal_to_safe_location`<br>`sample_000333` → `node_9_move_pedal_to_safe_location`<br>`sample_000375` → `node_10_remove_protection_cover_from_crimper` |  |

##### S2 — EMG Dilated Tier3→M2 Node

| 低 Recall Node | 支持 | 正确 | Recall | 随机抽取误分类样本 → 预测 Node | 备注 |
| --- | --- | --- | --- | --- | --- |
| node_1_unlock_crimper | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000005` → `node_9_move_pedal_to_safe_location`<br>`sample_000051` → `node_9_move_pedal_to_safe_location`<br>`sample_000130` → `node_9_move_pedal_to_safe_location`<br>`sample_000204` → `node_9_move_pedal_to_safe_location`<br>`sample_000274` → `node_9_move_pedal_to_safe_location` |  |
| node_2_put_lock_on_table | 6 | 3/6 | 50.0% | （显示 3/3）<br>`sample_000006` → `node_7_turn_on_water_pump`<br>`sample_000275` → `node_34_take_lock_from_table`<br>`sample_000380` → `node_3_turn_on_main_switch` |  |
| node_3_turn_on_main_switch | 6 | 3/6 | 50.0% | （显示 3/3）<br>`sample_000050` → `node_19_put_sample_on_machine_table_2`<br>`sample_000127` → `node_11_put_protection_cover_on_ground`<br>`sample_000276` → `node_2_put_lock_on_table` |  |
| node_4_turn_on_crimper | 6 | 2/6 | 33.3% | （显示 4/4）<br>`sample_000053` → `node_2_put_lock_on_table`<br>`sample_000132` → `node_3_turn_on_main_switch`<br>`sample_000210` → `node_1_unlock_crimper`<br>`sample_000280` → `node_6_turn_on_air_compressor` |  |
| node_5_adjust_parameters | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000011` → `node_23_inspect_sample`<br>`sample_000060` → `node_13_grip_sample_from_table_1`<br>`sample_000137` → `node_1_unlock_crimper`<br>`sample_000214` → `node_1_unlock_crimper`<br>`sample_000284` → `node_23_inspect_sample`<br>`sample_000389` → `node_10_remove_protection_cover_from_crimper` |  |
| node_6_turn_on_air_compressor | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000003` → `node_4_turn_on_crimper`<br>`sample_000056` → `node_3_turn_on_main_switch`<br>`sample_000133` → `node_2_put_lock_on_table`<br>`sample_000209` → `node_2_put_lock_on_table`<br>`sample_000385` → `node_2_put_lock_on_table` |  |
| node_8_turn_on_extractor_fan | 6 | 2/6 | 33.3% | （显示 4/4）<br>`sample_000004` → `node_33_turn_off_main_switch`<br>`sample_000054` → `node_7_turn_on_water_pump`<br>`sample_000278` → `node_28_turn_off_extractor_fan`<br>`sample_000383` → `node_6_turn_on_air_compressor` |  |
| node_10_remove_protection_cover_from_crimper | 6 | 3/6 | 50.0% | （显示 3/3）<br>`sample_000128` → `node_25_put_plier_on_table`<br>`sample_000212` → `node_13_grip_sample_from_table_1`<br>`sample_000386` → `node_13_grip_sample_from_table_1` |  |
| node_12_take_plier_from_table | 24 | 13/24 | 54.2% | （显示 10/11）<br>`sample_000012` → `node_11_put_protection_cover_on_ground`<br>`sample_000026` → `node_34_take_lock_from_table`<br>`sample_000061` → `node_13_grip_sample_from_table_1`<br>`sample_000103` → `node_34_take_lock_from_table`<br>`sample_000138` → `node_2_put_lock_on_table`<br>`sample_000215` → `node_6_turn_on_air_compressor`<br>`sample_000285` → `node_34_take_lock_from_table`<br>`sample_000317` → `node_34_take_lock_from_table`<br>`sample_000341` → `node_11_put_protection_cover_on_ground`<br>`sample_000404` → `node_34_take_lock_from_table` |  |
| node_13_grip_sample_from_table_1 | 24 | 15/24 | 62.5% | （显示 9/9）<br>`sample_000013` → `node_17_grip_sample_from_machine_table_2`<br>`sample_000027` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000153` → `node_1_unlock_crimper`<br>`sample_000216` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000241` → `node_12_take_plier_from_table`<br>`sample_000299` → `node_2_put_lock_on_table`<br>`sample_000312` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000342` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000405` → `node_10_remove_protection_cover_from_crimper` |  |
| node_14_put_sample_under_electrodes_1 | 24 | 9/24 | 37.5% | （显示 10/15）<br>`sample_000014` → `node_15_press_pedal_1`<br>`sample_000063` → `node_13_grip_sample_from_table_1`<br>`sample_000077` → `node_13_grip_sample_from_table_1`<br>`sample_000105` → `node_12_take_plier_from_table`<br>`sample_000168` → `node_13_grip_sample_from_table_1`<br>`sample_000242` → `node_12_take_plier_from_table`<br>`sample_000313` → `node_15_press_pedal_1`<br>`sample_000319` → `node_13_grip_sample_from_table_1`<br>`sample_000343` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000392` → `node_10_remove_protection_cover_from_crimper` |  |
| node_15_press_pedal_1 | 24 | 13/24 | 54.2% | （显示 10/11）<br>`sample_000029` → `node_23_inspect_sample`<br>`sample_000078` → `node_14_put_sample_under_electrodes_1`<br>`sample_000141` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000169` → `node_14_put_sample_under_electrodes_1`<br>`sample_000237` → `node_14_put_sample_under_electrodes_1`<br>`sample_000243` → `node_14_put_sample_under_electrodes_1`<br>`sample_000264` → `node_14_put_sample_under_electrodes_1`<br>`sample_000288` → `node_23_inspect_sample`<br>`sample_000314` → `node_14_put_sample_under_electrodes_1`<br>`sample_000320` → `node_14_put_sample_under_electrodes_1` |  |
| node_16_put_sample_on_machine_table_1 | 21 | 0/21 | 0.0% | （显示 10/21）<br>`sample_000079` → `node_2_put_lock_on_table`<br>`sample_000107` → `node_34_take_lock_from_table`<br>`sample_000142` → `node_2_put_lock_on_table`<br>`sample_000170` → `node_2_put_lock_on_table`<br>`sample_000194` → `node_2_put_lock_on_table`<br>`sample_000219` → `node_34_take_lock_from_table`<br>`sample_000244` → `node_12_take_plier_from_table`<br>`sample_000289` → `node_12_take_plier_from_table`<br>`sample_000321` → `node_2_put_lock_on_table`<br>`sample_000394` → `node_34_take_lock_from_table` |  |
| node_17_grip_sample_from_machine_table_2 | 21 | 11/21 | 52.4% | （显示 10/10）<br>`sample_000017` → `node_33_turn_off_main_switch`<br>`sample_000031` → `node_18_reverse_sample`<br>`sample_000080` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000171` → `node_18_reverse_sample`<br>`sample_000220` → `node_26_take_protection_cover_from_ground`<br>`sample_000245` → `node_26_take_protection_cover_from_ground`<br>`sample_000290` → `node_9_move_pedal_to_safe_location`<br>`sample_000303` → `node_13_grip_sample_from_table_1`<br>`sample_000322` → `node_13_grip_sample_from_table_1`<br>`sample_000360` → `node_33_turn_off_main_switch` |  |
| node_18_reverse_sample | 21 | 14/21 | 66.7% | （显示 7/7）<br>`sample_000018` → `node_19_put_sample_on_machine_table_2`<br>`sample_000081` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000109` → `node_16_put_sample_on_machine_table_1`<br>`sample_000304` → `node_15_press_pedal_1`<br>`sample_000347` → `node_16_put_sample_on_machine_table_1`<br>`sample_000361` → `node_16_put_sample_on_machine_table_1`<br>`sample_000410` → `node_16_put_sample_on_machine_table_1` |  |
| node_19_put_sample_on_machine_table_2 | 21 | 12/21 | 57.1% | （显示 9/9）<br>`sample_000019` → `node_16_put_sample_on_machine_table_1`<br>`sample_000033` → `node_33_turn_off_main_switch`<br>`sample_000110` → `node_6_turn_on_air_compressor`<br>`sample_000159` → `node_16_put_sample_on_machine_table_1`<br>`sample_000173` → `node_16_put_sample_on_machine_table_1`<br>`sample_000247` → `node_16_put_sample_on_machine_table_1`<br>`sample_000292` → `node_24_put_sample_on_table`<br>`sample_000324` → `node_16_put_sample_on_machine_table_1`<br>`sample_000425` → `node_24_put_sample_on_table` |  |
| node_20_grip_sample_from_machine_table_3 | 21 | 15/21 | 71.4% | （显示 6/6）<br>`sample_000020` → `node_19_put_sample_on_machine_table_2`<br>`sample_000111` → `node_26_take_protection_cover_from_ground`<br>`sample_000160` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000223` → `node_34_take_lock_from_table`<br>`sample_000293` → `node_31_move_pedal_to_original_place`<br>`sample_000349` → `node_18_reverse_sample` |  |
| node_21_put_sample_under_electrodes_2 | 21 | 5/21 | 23.8% | （显示 10/16）<br>`sample_000021` → `node_22_press_pedal_2`<br>`sample_000035` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000070` → `node_30_turn_off_air_compressor`<br>`sample_000084` → `node_18_reverse_sample`<br>`sample_000098` → `node_33_turn_off_main_switch`<br>`sample_000161` → `node_12_take_plier_from_table`<br>`sample_000175` → `node_26_take_protection_cover_from_ground`<br>`sample_000249` → `node_13_grip_sample_from_table_1`<br>`sample_000294` → `node_3_turn_on_main_switch`<br>`sample_000326` → `node_13_grip_sample_from_table_1` |  |
| node_22_press_pedal_2 | 21 | 14/21 | 66.7% | （显示 7/7）<br>`sample_000036` → `node_15_press_pedal_1`<br>`sample_000113` → `node_9_move_pedal_to_safe_location`<br>`sample_000148` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000162` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000176` → `node_21_put_sample_under_electrodes_2`<br>`sample_000295` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000308` → `node_23_inspect_sample` |  |
| node_23_inspect_sample | 17 | 10/17 | 58.8% | （显示 7/7）<br>`sample_000072` → `node_9_move_pedal_to_safe_location`<br>`sample_000100` → `node_26_take_protection_cover_from_ground`<br>`sample_000163` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000177` → `node_25_put_plier_on_table`<br>`sample_000201` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000352` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000415` → `node_26_take_protection_cover_from_ground` |  |
| node_24_put_sample_on_table | 24 | 3/24 | 12.5% | （显示 10/21）<br>`sample_000038` → `node_11_put_protection_cover_on_ground`<br>`sample_000150` → `node_19_put_sample_on_machine_table_2`<br>`sample_000178` → `node_19_put_sample_on_machine_table_2`<br>`sample_000226` → `node_3_turn_on_main_switch`<br>`sample_000238` → `node_3_turn_on_main_switch`<br>`sample_000315` → `node_2_put_lock_on_table`<br>`sample_000329` → `node_3_turn_on_main_switch`<br>`sample_000353` → `node_25_put_plier_on_table`<br>`sample_000367` → `node_33_turn_off_main_switch`<br>`sample_000416` → `node_3_turn_on_main_switch` |  |
| node_25_put_plier_on_table | 24 | 17/24 | 70.8% | （显示 7/7）<br>`sample_000151` → `node_34_take_lock_from_table`<br>`sample_000227` → `node_34_take_lock_from_table`<br>`sample_000239` → `node_2_put_lock_on_table`<br>`sample_000297` → `node_34_take_lock_from_table`<br>`sample_000310` → `node_2_put_lock_on_table`<br>`sample_000316` → `node_2_put_lock_on_table`<br>`sample_000368` → `node_34_take_lock_from_table` |  |
| node_26_take_protection_cover_from_ground | 6 | 4/6 | 66.7% | （显示 2/2）<br>`sample_000181` → `node_25_put_plier_on_table`<br>`sample_000271` → `node_2_put_lock_on_table` |  |
| node_28_turn_off_extractor_fan | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000042` → `node_33_turn_off_main_switch`<br>`sample_000119` → `node_29_turn_off_water_pump`<br>`sample_000185` → `node_29_turn_off_water_pump`<br>`sample_000268` → `node_29_turn_off_water_pump`<br>`sample_000337` → `node_25_put_plier_on_table`<br>`sample_000370` → `node_33_turn_off_main_switch` |  |
| node_34_take_lock_from_table | 5 | 2/5 | 40.0% | （显示 3/3）<br>`sample_000045` → `node_25_put_plier_on_table`<br>`sample_000122` → `node_26_take_protection_cover_from_ground`<br>`sample_000332` → `node_25_put_plier_on_table` |  |
| node_35_lock_crimper | 5 | 0/5 | 0.0% | （显示 5/5）<br>`sample_000046` → `node_31_move_pedal_to_original_place`<br>`sample_000123` → `node_31_move_pedal_to_original_place`<br>`sample_000189` → `node_9_move_pedal_to_safe_location`<br>`sample_000333` → `node_9_move_pedal_to_safe_location`<br>`sample_000375` → `node_31_move_pedal_to_original_place` |  |

##### S3 — IMU ResNet10 Tier3→M2 Node

| 低 Recall Node | 支持 | 正确 | Recall | 随机抽取误分类样本 → 预测 Node | 备注 |
| --- | --- | --- | --- | --- | --- |
| node_5_adjust_parameters | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000011` → `node_6_turn_on_air_compressor`<br>`sample_000060` → `node_2_put_lock_on_table`<br>`sample_000137` → `node_2_put_lock_on_table`<br>`sample_000214` → `node_4_turn_on_crimper`<br>`sample_000284` → `node_32_turn_off_crimper`<br>`sample_000389` → `node_6_turn_on_air_compressor` |  |
| node_6_turn_on_air_compressor | 6 | 4/6 | 66.7% | （显示 2/2）<br>`sample_000056` → `node_4_turn_on_crimper`<br>`sample_000209` → `node_12_take_plier_from_table` |  |
| node_7_turn_on_water_pump | 6 | 2/6 | 33.3% | （显示 4/4）<br>`sample_000002` → `node_8_turn_on_extractor_fan`<br>`sample_000055` → `node_12_take_plier_from_table`<br>`sample_000135` → `node_12_take_plier_from_table`<br>`sample_000277` → `node_24_put_sample_on_table` |  |
| node_12_take_plier_from_table | 24 | 17/24 | 70.8% | （显示 7/7）<br>`sample_000012` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000190` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000215` → `node_34_take_lock_from_table`<br>`sample_000298` → `node_34_take_lock_from_table`<br>`sample_000355` → `node_34_take_lock_from_table`<br>`sample_000390` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000404` → `node_34_take_lock_from_table` |  |
| node_13_grip_sample_from_table_1 | 24 | 19/24 | 79.2% | （显示 5/5）<br>`sample_000062` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000167` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000262` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000286` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000312` → `node_10_remove_protection_cover_from_crimper` |  |
| node_24_put_sample_on_table | 24 | 9/24 | 37.5% | （显示 10/15）<br>`sample_000024` → `node_16_put_sample_on_machine_table_1`<br>`sample_000073` → `node_12_take_plier_from_table`<br>`sample_000178` → `node_12_take_plier_from_table`<br>`sample_000226` → `node_12_take_plier_from_table`<br>`sample_000232` → `node_12_take_plier_from_table`<br>`sample_000238` → `node_12_take_plier_from_table`<br>`sample_000259` → `node_12_take_plier_from_table`<br>`sample_000309` → `node_34_take_lock_from_table`<br>`sample_000315` → `node_12_take_plier_from_table`<br>`sample_000367` → `node_12_take_plier_from_table` |  |
| node_25_put_plier_on_table | 24 | 15/24 | 62.5% | （显示 9/9）<br>`sample_000227` → `node_34_take_lock_from_table`<br>`sample_000233` → `node_2_put_lock_on_table`<br>`sample_000239` → `node_2_put_lock_on_table`<br>`sample_000260` → `node_34_take_lock_from_table`<br>`sample_000297` → `node_34_take_lock_from_table`<br>`sample_000310` → `node_34_take_lock_from_table`<br>`sample_000316` → `node_13_grip_sample_from_table_1`<br>`sample_000354` → `node_34_take_lock_from_table`<br>`sample_000403` → `node_34_take_lock_from_table` |  |
| node_29_turn_off_water_pump | 6 | 3/6 | 50.0% | （显示 3/3）<br>`sample_000043` → `node_28_turn_off_extractor_fan`<br>`sample_000186` → `node_12_take_plier_from_table`<br>`sample_000267` → `node_34_take_lock_from_table` |  |
| node_31_move_pedal_to_original_place | 6 | 4/6 | 66.7% | （显示 2/2）<br>`sample_000180` → `node_34_take_lock_from_table`<br>`sample_000273` → `node_26_take_protection_cover_from_ground` |  |
| node_35_lock_crimper | 5 | 1/5 | 20.0% | （显示 4/4）<br>`sample_000123` → `node_32_turn_off_crimper`<br>`sample_000189` → `node_32_turn_off_crimper`<br>`sample_000333` → `node_32_turn_off_crimper`<br>`sample_000375` → `node_32_turn_off_crimper` |  |

##### S4 — IMU Dilated Tier3→M2 Node

| 低 Recall Node | 支持 | 正确 | Recall | 随机抽取误分类样本 → 预测 Node | 备注 |
| --- | --- | --- | --- | --- | --- |
| node_3_turn_on_main_switch | 6 | 4/6 | 66.7% | （显示 2/2）<br>`sample_000001` → `node_32_turn_off_crimper`<br>`sample_000206` → `node_12_take_plier_from_table` |  |
| node_4_turn_on_crimper | 6 | 4/6 | 66.7% | （显示 2/2）<br>`sample_000132` → `node_32_turn_off_crimper`<br>`sample_000280` → `node_32_turn_off_crimper` |  |
| node_5_adjust_parameters | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000011` → `node_4_turn_on_crimper`<br>`sample_000060` → `node_1_unlock_crimper`<br>`sample_000137` → `node_4_turn_on_crimper`<br>`sample_000214` → `node_4_turn_on_crimper`<br>`sample_000284` → `node_4_turn_on_crimper`<br>`sample_000389` → `node_4_turn_on_crimper` |  |
| node_6_turn_on_air_compressor | 6 | 4/6 | 66.7% | （显示 2/2）<br>`sample_000279` → `node_30_turn_off_air_compressor`<br>`sample_000385` → `node_9_move_pedal_to_safe_location` |  |
| node_7_turn_on_water_pump | 6 | 2/6 | 33.3% | （显示 4/4）<br>`sample_000002` → `node_16_put_sample_on_machine_table_1`<br>`sample_000055` → `node_1_unlock_crimper`<br>`sample_000135` → `node_8_turn_on_extractor_fan`<br>`sample_000207` → `node_12_take_plier_from_table` |  |
| node_11_put_protection_cover_on_ground | 6 | 4/6 | 66.7% | （显示 2/2）<br>`sample_000058` → `node_9_move_pedal_to_safe_location`<br>`sample_000283` → `node_9_move_pedal_to_safe_location` |  |
| node_12_take_plier_from_table | 24 | 19/24 | 79.2% | （显示 5/5）<br>`sample_000075` → `node_34_take_lock_from_table`<br>`sample_000138` → `node_34_take_lock_from_table`<br>`sample_000215` → `node_2_put_lock_on_table`<br>`sample_000390` → `node_34_take_lock_from_table`<br>`sample_000404` → `node_34_take_lock_from_table` |  |
| node_13_grip_sample_from_table_1 | 24 | 15/24 | 62.5% | （显示 9/9）<br>`sample_000027` → `node_2_put_lock_on_table`<br>`sample_000062` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000153` → `node_17_grip_sample_from_machine_table_2`<br>`sample_000167` → `node_34_take_lock_from_table`<br>`sample_000216` → `node_34_take_lock_from_table`<br>`sample_000241` → `node_31_move_pedal_to_original_place`<br>`sample_000299` → `node_34_take_lock_from_table`<br>`sample_000312` → `node_12_take_plier_from_table`<br>`sample_000356` → `node_17_grip_sample_from_machine_table_2` |  |
| node_19_put_sample_on_machine_table_2 | 21 | 13/21 | 61.9% | （显示 8/8）<br>`sample_000033` → `node_13_grip_sample_from_table_1`<br>`sample_000173` → `node_24_put_sample_on_table`<br>`sample_000247` → `node_34_take_lock_from_table`<br>`sample_000254` → `node_18_reverse_sample`<br>`sample_000292` → `node_24_put_sample_on_table`<br>`sample_000305` → `node_25_put_plier_on_table`<br>`sample_000411` → `node_2_put_lock_on_table`<br>`sample_000425` → `node_20_grip_sample_from_machine_table_3` |  |
| node_24_put_sample_on_table | 24 | 8/24 | 33.3% | （显示 10/16）<br>`sample_000038` → `node_12_take_plier_from_table`<br>`sample_000101` → `node_12_take_plier_from_table`<br>`sample_000178` → `node_12_take_plier_from_table`<br>`sample_000226` → `node_12_take_plier_from_table`<br>`sample_000259` → `node_12_take_plier_from_table`<br>`sample_000265` → `node_12_take_plier_from_table`<br>`sample_000309` → `node_34_take_lock_from_table`<br>`sample_000315` → `node_12_take_plier_from_table`<br>`sample_000329` → `node_12_take_plier_from_table`<br>`sample_000416` → `node_34_take_lock_from_table` |  |
| node_25_put_plier_on_table | 24 | 10/24 | 41.7% | （显示 10/14）<br>`sample_000074` → `node_18_reverse_sample`<br>`sample_000165` → `node_12_take_plier_from_table`<br>`sample_000203` → `node_18_reverse_sample`<br>`sample_000233` → `node_34_take_lock_from_table`<br>`sample_000239` → `node_34_take_lock_from_table`<br>`sample_000260` → `node_34_take_lock_from_table`<br>`sample_000266` → `node_34_take_lock_from_table`<br>`sample_000310` → `node_34_take_lock_from_table`<br>`sample_000354` → `node_34_take_lock_from_table`<br>`sample_000417` → `node_18_reverse_sample` |  |
| node_26_take_protection_cover_from_ground | 6 | 4/6 | 66.7% | （显示 2/2）<br>`sample_000047` → `node_31_move_pedal_to_original_place`<br>`sample_000376` → `node_31_move_pedal_to_original_place` |  |
| node_28_turn_off_extractor_fan | 6 | 4/6 | 66.7% | （显示 2/2）<br>`sample_000268` → `node_8_turn_on_extractor_fan`<br>`sample_000337` → `node_8_turn_on_extractor_fan` |  |
| node_29_turn_off_water_pump | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000043` → `node_28_turn_off_extractor_fan`<br>`sample_000120` → `node_28_turn_off_extractor_fan`<br>`sample_000186` → `node_26_take_protection_cover_from_ground`<br>`sample_000267` → `node_12_take_plier_from_table`<br>`sample_000338` → `node_28_turn_off_extractor_fan`<br>`sample_000372` → `node_28_turn_off_extractor_fan` |  |
| node_31_move_pedal_to_original_place | 6 | 3/6 | 50.0% | （显示 3/3）<br>`sample_000126` → `node_34_take_lock_from_table`<br>`sample_000180` → `node_30_turn_off_air_compressor`<br>`sample_000273` → `node_9_move_pedal_to_safe_location` |  |

##### S5 — EMG ResNet10 Direct Node

| 低 Recall Node | 支持 | 正确 | Recall | 随机抽取误分类样本 → 预测 Node | 备注 |
| --- | --- | --- | --- | --- | --- |
| node_1_unlock_crimper | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000005` → `node_31_move_pedal_to_original_place`<br>`sample_000051` → `node_31_move_pedal_to_original_place`<br>`sample_000130` → `node_31_move_pedal_to_original_place`<br>`sample_000204` → `node_31_move_pedal_to_original_place`<br>`sample_000274` → `node_32_turn_off_crimper`<br>`sample_000379` → `node_23_inspect_sample` |  |
| node_2_put_lock_on_table | 6 | 3/6 | 50.0% | （显示 3/3）<br>`sample_000131` → `node_34_take_lock_from_table`<br>`sample_000205` → `node_25_put_plier_on_table`<br>`sample_000275` → `node_7_turn_on_water_pump` |  |
| node_3_turn_on_main_switch | 6 | 3/6 | 50.0% | （显示 3/3）<br>`sample_000127` → `node_19_put_sample_on_machine_table_2`<br>`sample_000206` → `node_4_turn_on_crimper`<br>`sample_000276` → `node_34_take_lock_from_table` |  |
| node_4_turn_on_crimper | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000007` → `node_32_turn_off_crimper`<br>`sample_000132` → `node_19_put_sample_on_machine_table_2`<br>`sample_000210` → `node_11_put_protection_cover_on_ground`<br>`sample_000280` → `node_19_put_sample_on_machine_table_2`<br>`sample_000382` → `node_19_put_sample_on_machine_table_2` |  |
| node_5_adjust_parameters | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000011` → `node_22_press_pedal_2`<br>`sample_000060` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000137` → `node_23_inspect_sample`<br>`sample_000214` → `node_27_put_protection_cover_on_crimper`<br>`sample_000284` → `node_23_inspect_sample`<br>`sample_000389` → `node_20_grip_sample_from_machine_table_3` |  |
| node_6_turn_on_air_compressor | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000003` → `node_18_reverse_sample`<br>`sample_000056` → `node_19_put_sample_on_machine_table_2`<br>`sample_000133` → `node_32_turn_off_crimper`<br>`sample_000209` → `node_34_take_lock_from_table`<br>`sample_000279` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000385` → `node_34_take_lock_from_table` |  |
| node_7_turn_on_water_pump | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000002` → `node_24_put_sample_on_table`<br>`sample_000055` → `node_29_turn_off_water_pump`<br>`sample_000135` → `node_29_turn_off_water_pump`<br>`sample_000277` → `node_6_turn_on_air_compressor`<br>`sample_000384` → `node_25_put_plier_on_table` |  |
| node_8_turn_on_extractor_fan | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000004` → `node_33_turn_off_main_switch`<br>`sample_000054` → `node_29_turn_off_water_pump`<br>`sample_000134` → `node_30_turn_off_air_compressor`<br>`sample_000208` → `node_28_turn_off_extractor_fan`<br>`sample_000278` → `node_19_put_sample_on_machine_table_2`<br>`sample_000383` → `node_29_turn_off_water_pump` |  |
| node_9_move_pedal_to_safe_location | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000010` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000059` → `node_31_move_pedal_to_original_place`<br>`sample_000136` → `node_19_put_sample_on_machine_table_2`<br>`sample_000211` → `node_31_move_pedal_to_original_place`<br>`sample_000281` → `node_31_move_pedal_to_original_place`<br>`sample_000388` → `node_31_move_pedal_to_original_place` |  |
| node_10_remove_protection_cover_from_crimper | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000008` → `node_13_grip_sample_from_table_1`<br>`sample_000057` → `node_25_put_plier_on_table`<br>`sample_000128` → `node_27_put_protection_cover_on_crimper`<br>`sample_000212` → `node_25_put_plier_on_table`<br>`sample_000386` → `node_13_grip_sample_from_table_1` |  |
| node_11_put_protection_cover_on_ground | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000009` → `node_34_take_lock_from_table`<br>`sample_000058` → `node_25_put_plier_on_table`<br>`sample_000129` → `node_25_put_plier_on_table`<br>`sample_000213` → `node_25_put_plier_on_table`<br>`sample_000283` → `node_25_put_plier_on_table` |  |
| node_12_take_plier_from_table | 24 | 7/24 | 29.2% | （显示 10/17）<br>`sample_000075` → `node_29_turn_off_water_pump`<br>`sample_000089` → `node_2_put_lock_on_table`<br>`sample_000138` → `node_25_put_plier_on_table`<br>`sample_000152` → `node_2_put_lock_on_table`<br>`sample_000166` → `node_2_put_lock_on_table`<br>`sample_000215` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000228` → `node_2_put_lock_on_table`<br>`sample_000261` → `node_2_put_lock_on_table`<br>`sample_000341` → `node_11_put_protection_cover_on_ground`<br>`sample_000355` → `node_16_put_sample_on_machine_table_1` |  |
| node_13_grip_sample_from_table_1 | 24 | 4/24 | 16.7% | （显示 10/20）<br>`sample_000013` → `node_18_reverse_sample`<br>`sample_000090` → `node_25_put_plier_on_table`<br>`sample_000153` → `node_25_put_plier_on_table`<br>`sample_000167` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000229` → `node_12_take_plier_from_table`<br>`sample_000241` → `node_12_take_plier_from_table`<br>`sample_000299` → `node_12_take_plier_from_table`<br>`sample_000318` → `node_32_turn_off_crimper`<br>`sample_000356` → `node_25_put_plier_on_table`<br>`sample_000419` → `node_18_reverse_sample` |  |
| node_14_put_sample_under_electrodes_1 | 24 | 0/24 | 0.0% | （显示 10/24）<br>`sample_000063` → `node_12_take_plier_from_table`<br>`sample_000091` → `node_12_take_plier_from_table`<br>`sample_000105` → `node_12_take_plier_from_table`<br>`sample_000140` → `node_26_take_protection_cover_from_ground`<br>`sample_000168` → `node_2_put_lock_on_table`<br>`sample_000217` → `node_12_take_plier_from_table`<br>`sample_000230` → `node_26_take_protection_cover_from_ground`<br>`sample_000287` → `node_29_turn_off_water_pump`<br>`sample_000357` → `node_25_put_plier_on_table`<br>`sample_000420` → `node_10_remove_protection_cover_from_crimper` |  |
| node_15_press_pedal_1 | 24 | 2/24 | 8.3% | （显示 10/22）<br>`sample_000078` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000141` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000155` → `node_23_inspect_sample`<br>`sample_000193` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000231` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000237` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000314` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000344` → `node_22_press_pedal_2`<br>`sample_000393` → `node_22_press_pedal_2`<br>`sample_000407` → `node_10_remove_protection_cover_from_crimper` |  |
| node_16_put_sample_on_machine_table_1 | 21 | 0/21 | 0.0% | （显示 10/21）<br>`sample_000107` → `node_2_put_lock_on_table`<br>`sample_000142` → `node_2_put_lock_on_table`<br>`sample_000219` → `node_2_put_lock_on_table`<br>`sample_000244` → `node_2_put_lock_on_table`<br>`sample_000251` → `node_12_take_plier_from_table`<br>`sample_000321` → `node_2_put_lock_on_table`<br>`sample_000345` → `node_25_put_plier_on_table`<br>`sample_000359` → `node_2_put_lock_on_table`<br>`sample_000394` → `node_2_put_lock_on_table`<br>`sample_000422` → `node_2_put_lock_on_table` |  |
| node_17_grip_sample_from_machine_table_2 | 21 | 8/21 | 38.1% | （显示 10/13）<br>`sample_000017` → `node_33_turn_off_main_switch`<br>`sample_000031` → `node_18_reverse_sample`<br>`sample_000108` → `node_26_take_protection_cover_from_ground`<br>`sample_000143` → `node_26_take_protection_cover_from_ground`<br>`sample_000220` → `node_26_take_protection_cover_from_ground`<br>`sample_000245` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000303` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000346` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000360` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000423` → `node_18_reverse_sample` |  |
| node_18_reverse_sample | 21 | 11/21 | 52.4% | （显示 10/10）<br>`sample_000018` → `node_19_put_sample_on_machine_table_2`<br>`sample_000032` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000081` → `node_28_turn_off_extractor_fan`<br>`sample_000109` → `node_19_put_sample_on_machine_table_2`<br>`sample_000304` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000347` → `node_19_put_sample_on_machine_table_2`<br>`sample_000361` → `node_19_put_sample_on_machine_table_2`<br>`sample_000396` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000410` → `node_31_move_pedal_to_original_place`<br>`sample_000424` → `node_10_remove_protection_cover_from_crimper` |  |
| node_20_grip_sample_from_machine_table_3 | 21 | 0/21 | 0.0% | （显示 10/21）<br>`sample_000020` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000034` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000069` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000083` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000111` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000160` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000174` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000349` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000398` → `node_24_put_sample_on_table`<br>`sample_000426` → `node_10_remove_protection_cover_from_crimper` |  |
| node_21_put_sample_under_electrodes_2 | 21 | 0/21 | 0.0% | （显示 10/21）<br>`sample_000021` → `node_15_press_pedal_1`<br>`sample_000035` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000112` → `node_12_take_plier_from_table`<br>`sample_000161` → `node_18_reverse_sample`<br>`sample_000199` → `node_13_grip_sample_from_table_1`<br>`sample_000224` → `node_12_take_plier_from_table`<br>`sample_000249` → `node_28_turn_off_extractor_fan`<br>`sample_000350` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000413` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000427` → `node_10_remove_protection_cover_from_crimper` |  |
| node_22_press_pedal_2 | 21 | 2/21 | 9.5% | （显示 10/19）<br>`sample_000036` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000085` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000113` → `node_23_inspect_sample`<br>`sample_000250` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000295` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000308` → `node_23_inspect_sample`<br>`sample_000327` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000365` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000400` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000414` → `node_10_remove_protection_cover_from_crimper` |  |
| node_23_inspect_sample | 17 | 9/17 | 52.9% | （显示 8/8）<br>`sample_000023` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000086` → `node_1_unlock_crimper`<br>`sample_000177` → `node_25_put_plier_on_table`<br>`sample_000258` → `node_25_put_plier_on_table`<br>`sample_000328` → `node_25_put_plier_on_table`<br>`sample_000352` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000415` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000429` → `node_10_remove_protection_cover_from_crimper` |  |
| node_24_put_sample_on_table | 24 | 0/24 | 0.0% | （显示 10/24）<br>`sample_000024` → `node_2_put_lock_on_table`<br>`sample_000101` → `node_18_reverse_sample`<br>`sample_000164` → `node_2_put_lock_on_table`<br>`sample_000202` → `node_11_put_protection_cover_on_ground`<br>`sample_000226` → `node_2_put_lock_on_table`<br>`sample_000232` → `node_2_put_lock_on_table`<br>`sample_000238` → `node_2_put_lock_on_table`<br>`sample_000259` → `node_28_turn_off_extractor_fan`<br>`sample_000265` → `node_18_reverse_sample`<br>`sample_000353` → `node_12_take_plier_from_table` |  |
| node_25_put_plier_on_table | 24 | 11/24 | 45.8% | （显示 10/13）<br>`sample_000039` → `node_34_take_lock_from_table`<br>`sample_000165` → `node_34_take_lock_from_table`<br>`sample_000233` → `node_2_put_lock_on_table`<br>`sample_000239` → `node_12_take_plier_from_table`<br>`sample_000260` → `node_34_take_lock_from_table`<br>`sample_000297` → `node_34_take_lock_from_table`<br>`sample_000316` → `node_2_put_lock_on_table`<br>`sample_000330` → `node_34_take_lock_from_table`<br>`sample_000368` → `node_34_take_lock_from_table`<br>`sample_000417` → `node_34_take_lock_from_table` |  |
| node_26_take_protection_cover_from_ground | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000047` → `node_13_grip_sample_from_table_1`<br>`sample_000124` → `node_18_reverse_sample`<br>`sample_000181` → `node_34_take_lock_from_table`<br>`sample_000271` → `node_25_put_plier_on_table`<br>`sample_000334` → `node_29_turn_off_water_pump` |  |
| node_27_put_protection_cover_on_crimper | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000048` → `node_23_inspect_sample`<br>`sample_000125` → `node_13_grip_sample_from_table_1`<br>`sample_000182` → `node_13_grip_sample_from_table_1`<br>`sample_000272` → `node_24_put_sample_on_table`<br>`sample_000335` → `node_32_turn_off_crimper` |  |
| node_28_turn_off_extractor_fan | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000042` → `node_29_turn_off_water_pump`<br>`sample_000119` → `node_29_turn_off_water_pump`<br>`sample_000268` → `node_29_turn_off_water_pump`<br>`sample_000337` → `node_29_turn_off_water_pump`<br>`sample_000370` → `node_7_turn_on_water_pump` |  |
| node_29_turn_off_water_pump | 6 | 3/6 | 50.0% | （显示 3/3）<br>`sample_000120` → `node_28_turn_off_extractor_fan`<br>`sample_000186` → `node_2_put_lock_on_table`<br>`sample_000338` → `node_19_put_sample_on_machine_table_2` |  |
| node_30_turn_off_air_compressor | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000041` → `node_3_turn_on_main_switch`<br>`sample_000118` → `node_4_turn_on_crimper`<br>`sample_000184` → `node_3_turn_on_main_switch`<br>`sample_000269` → `node_34_take_lock_from_table`<br>`sample_000336` → `node_3_turn_on_main_switch`<br>`sample_000371` → `node_3_turn_on_main_switch` |  |
| node_32_turn_off_crimper | 6 | 4/6 | 66.7% | （显示 2/2）<br>`sample_000270` → `node_25_put_plier_on_table`<br>`sample_000369` → `node_25_put_plier_on_table` |  |
| node_33_turn_off_main_switch | 5 | 1/5 | 20.0% | （显示 4/4）<br>`sample_000121` → `node_3_turn_on_main_switch`<br>`sample_000187` → `node_3_turn_on_main_switch`<br>`sample_000339` → `node_3_turn_on_main_switch`<br>`sample_000373` → `node_3_turn_on_main_switch` |  |
| node_34_take_lock_from_table | 5 | 0/5 | 0.0% | （显示 5/5）<br>`sample_000045` → `node_25_put_plier_on_table`<br>`sample_000122` → `node_25_put_plier_on_table`<br>`sample_000188` → `node_12_take_plier_from_table`<br>`sample_000332` → `node_25_put_plier_on_table`<br>`sample_000374` → `node_12_take_plier_from_table` |  |
| node_35_lock_crimper | 5 | 0/5 | 0.0% | （显示 5/5）<br>`sample_000046` → `node_31_move_pedal_to_original_place`<br>`sample_000123` → `node_31_move_pedal_to_original_place`<br>`sample_000189` → `node_31_move_pedal_to_original_place`<br>`sample_000333` → `node_31_move_pedal_to_original_place`<br>`sample_000375` → `node_31_move_pedal_to_original_place` |  |

##### S6 — EMG Dilated Direct Node

| 低 Recall Node | 支持 | 正确 | Recall | 随机抽取误分类样本 → 预测 Node | 备注 |
| --- | --- | --- | --- | --- | --- |
| node_1_unlock_crimper | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000005` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000051` → `node_31_move_pedal_to_original_place`<br>`sample_000130` → `node_9_move_pedal_to_safe_location`<br>`sample_000204` → `node_31_move_pedal_to_original_place`<br>`sample_000274` → `node_31_move_pedal_to_original_place`<br>`sample_000379` → `node_9_move_pedal_to_safe_location` |  |
| node_2_put_lock_on_table | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000006` → `node_25_put_plier_on_table`<br>`sample_000052` → `node_34_take_lock_from_table`<br>`sample_000131` → `node_34_take_lock_from_table`<br>`sample_000205` → `node_34_take_lock_from_table`<br>`sample_000275` → `node_34_take_lock_from_table`<br>`sample_000380` → `node_3_turn_on_main_switch` |  |
| node_3_turn_on_main_switch | 6 | 2/6 | 33.3% | （显示 4/4）<br>`sample_000001` → `node_18_reverse_sample`<br>`sample_000050` → `node_19_put_sample_on_machine_table_2`<br>`sample_000127` → `node_33_turn_off_main_switch`<br>`sample_000276` → `node_27_put_protection_cover_on_crimper` |  |
| node_4_turn_on_crimper | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000007` → `node_6_turn_on_air_compressor`<br>`sample_000053` → `node_30_turn_off_air_compressor`<br>`sample_000132` → `node_30_turn_off_air_compressor`<br>`sample_000280` → `node_6_turn_on_air_compressor`<br>`sample_000382` → `node_6_turn_on_air_compressor` |  |
| node_5_adjust_parameters | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000011` → `node_23_inspect_sample`<br>`sample_000060` → `node_21_put_sample_under_electrodes_2`<br>`sample_000137` → `node_23_inspect_sample`<br>`sample_000214` → `node_23_inspect_sample`<br>`sample_000284` → `node_23_inspect_sample`<br>`sample_000389` → `node_2_put_lock_on_table` |  |
| node_6_turn_on_air_compressor | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000003` → `node_19_put_sample_on_machine_table_2`<br>`sample_000133` → `node_3_turn_on_main_switch`<br>`sample_000209` → `node_30_turn_off_air_compressor`<br>`sample_000279` → `node_25_put_plier_on_table`<br>`sample_000385` → `node_24_put_sample_on_table` |  |
| node_7_turn_on_water_pump | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000002` → `node_25_put_plier_on_table`<br>`sample_000055` → `node_29_turn_off_water_pump`<br>`sample_000135` → `node_33_turn_off_main_switch`<br>`sample_000277` → `node_33_turn_off_main_switch`<br>`sample_000384` → `node_25_put_plier_on_table` |  |
| node_8_turn_on_extractor_fan | 6 | 3/6 | 50.0% | （显示 3/3）<br>`sample_000004` → `node_33_turn_off_main_switch`<br>`sample_000278` → `node_33_turn_off_main_switch`<br>`sample_000383` → `node_33_turn_off_main_switch` |  |
| node_9_move_pedal_to_safe_location | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000010` → `node_33_turn_off_main_switch`<br>`sample_000059` → `node_24_put_sample_on_table`<br>`sample_000136` → `node_30_turn_off_air_compressor`<br>`sample_000211` → `node_30_turn_off_air_compressor`<br>`sample_000388` → `node_25_put_plier_on_table` |  |
| node_10_remove_protection_cover_from_crimper | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000008` → `node_18_reverse_sample`<br>`sample_000128` → `node_25_put_plier_on_table`<br>`sample_000212` → `node_13_grip_sample_from_table_1`<br>`sample_000282` → `node_25_put_plier_on_table`<br>`sample_000386` → `node_25_put_plier_on_table` |  |
| node_11_put_protection_cover_on_ground | 6 | 4/6 | 66.7% | （显示 2/2）<br>`sample_000009` → `node_2_put_lock_on_table`<br>`sample_000387` → `node_29_turn_off_water_pump` |  |
| node_12_take_plier_from_table | 24 | 10/24 | 41.7% | （显示 10/14）<br>`sample_000075` → `node_7_turn_on_water_pump`<br>`sample_000190` → `node_25_put_plier_on_table`<br>`sample_000215` → `node_18_reverse_sample`<br>`sample_000234` → `node_34_take_lock_from_table`<br>`sample_000285` → `node_25_put_plier_on_table`<br>`sample_000298` → `node_34_take_lock_from_table`<br>`sample_000317` → `node_34_take_lock_from_table`<br>`sample_000341` → `node_6_turn_on_air_compressor`<br>`sample_000390` → `node_34_take_lock_from_table`<br>`sample_000404` → `node_34_take_lock_from_table` |  |
| node_13_grip_sample_from_table_1 | 24 | 5/24 | 20.8% | （显示 10/19）<br>`sample_000090` → `node_25_put_plier_on_table`<br>`sample_000139` → `node_25_put_plier_on_table`<br>`sample_000153` → `node_25_put_plier_on_table`<br>`sample_000191` → `node_18_reverse_sample`<br>`sample_000235` → `node_25_put_plier_on_table`<br>`sample_000262` → `node_26_take_protection_cover_from_ground`<br>`sample_000312` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000342` → `node_18_reverse_sample`<br>`sample_000405` → `node_18_reverse_sample`<br>`sample_000419` → `node_18_reverse_sample` |  |
| node_14_put_sample_under_electrodes_1 | 24 | 1/24 | 4.2% | （显示 10/23）<br>`sample_000014` → `node_22_press_pedal_2`<br>`sample_000063` → `node_25_put_plier_on_table`<br>`sample_000077` → `node_18_reverse_sample`<br>`sample_000091` → `node_12_take_plier_from_table`<br>`sample_000140` → `node_21_put_sample_under_electrodes_2`<br>`sample_000154` → `node_21_put_sample_under_electrodes_2`<br>`sample_000192` → `node_18_reverse_sample`<br>`sample_000242` → `node_12_take_plier_from_table`<br>`sample_000343` → `node_17_grip_sample_from_machine_table_2`<br>`sample_000406` → `node_18_reverse_sample` |  |
| node_15_press_pedal_1 | 24 | 3/24 | 12.5% | （显示 10/21）<br>`sample_000029` → `node_23_inspect_sample`<br>`sample_000064` → `node_23_inspect_sample`<br>`sample_000141` → `node_18_reverse_sample`<br>`sample_000155` → `node_22_press_pedal_2`<br>`sample_000193` → `node_23_inspect_sample`<br>`sample_000243` → `node_21_put_sample_under_electrodes_2`<br>`sample_000301` → `node_23_inspect_sample`<br>`sample_000320` → `node_17_grip_sample_from_machine_table_2`<br>`sample_000344` → `node_23_inspect_sample`<br>`sample_000358` → `node_23_inspect_sample` |  |
| node_16_put_sample_on_machine_table_1 | 21 | 1/21 | 4.8% | （显示 10/20）<br>`sample_000030` → `node_18_reverse_sample`<br>`sample_000065` → `node_25_put_plier_on_table`<br>`sample_000079` → `node_2_put_lock_on_table`<br>`sample_000170` → `node_34_take_lock_from_table`<br>`sample_000219` → `node_34_take_lock_from_table`<br>`sample_000302` → `node_34_take_lock_from_table`<br>`sample_000321` → `node_34_take_lock_from_table`<br>`sample_000345` → `node_25_put_plier_on_table`<br>`sample_000359` → `node_25_put_plier_on_table`<br>`sample_000408` → `node_34_take_lock_from_table` |  |
| node_17_grip_sample_from_machine_table_2 | 21 | 1/21 | 4.8% | （显示 10/20）<br>`sample_000017` → `node_18_reverse_sample`<br>`sample_000031` → `node_18_reverse_sample`<br>`sample_000094` → `node_18_reverse_sample`<br>`sample_000143` → `node_18_reverse_sample`<br>`sample_000171` → `node_18_reverse_sample`<br>`sample_000195` → `node_18_reverse_sample`<br>`sample_000245` → `node_18_reverse_sample`<br>`sample_000290` → `node_18_reverse_sample`<br>`sample_000322` → `node_18_reverse_sample`<br>`sample_000409` → `node_18_reverse_sample` |  |
| node_20_grip_sample_from_machine_table_3 | 21 | 0/21 | 0.0% | （显示 10/21）<br>`sample_000020` → `node_18_reverse_sample`<br>`sample_000034` → `node_18_reverse_sample`<br>`sample_000083` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000097` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000174` → `node_18_reverse_sample`<br>`sample_000255` → `node_21_put_sample_under_electrodes_2`<br>`sample_000293` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000349` → `node_18_reverse_sample`<br>`sample_000398` → `node_18_reverse_sample`<br>`sample_000412` → `node_18_reverse_sample` |  |
| node_21_put_sample_under_electrodes_2 | 21 | 3/21 | 14.3% | （显示 10/18）<br>`sample_000021` → `node_23_inspect_sample`<br>`sample_000035` → `node_18_reverse_sample`<br>`sample_000070` → `node_30_turn_off_air_compressor`<br>`sample_000084` → `node_18_reverse_sample`<br>`sample_000224` → `node_12_take_plier_from_table`<br>`sample_000249` → `node_25_put_plier_on_table`<br>`sample_000307` → `node_12_take_plier_from_table`<br>`sample_000326` → `node_18_reverse_sample`<br>`sample_000399` → `node_18_reverse_sample`<br>`sample_000413` → `node_18_reverse_sample` |  |
| node_22_press_pedal_2 | 21 | 5/21 | 23.8% | （显示 10/16）<br>`sample_000036` → `node_23_inspect_sample`<br>`sample_000085` → `node_17_grip_sample_from_machine_table_2`<br>`sample_000099` → `node_23_inspect_sample`<br>`sample_000113` → `node_27_put_protection_cover_on_crimper`<br>`sample_000162` → `node_17_grip_sample_from_machine_table_2`<br>`sample_000200` → `node_23_inspect_sample`<br>`sample_000250` → `node_23_inspect_sample`<br>`sample_000308` → `node_23_inspect_sample`<br>`sample_000351` → `node_15_press_pedal_1`<br>`sample_000414` → `node_23_inspect_sample` |  |
| node_23_inspect_sample | 17 | 6/17 | 35.3% | （显示 10/11）<br>`sample_000037` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000072` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000086` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000163` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000201` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000258` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000328` → `node_25_put_plier_on_table`<br>`sample_000352` → `node_18_reverse_sample`<br>`sample_000415` → `node_18_reverse_sample`<br>`sample_000429` → `node_18_reverse_sample` |  |
| node_24_put_sample_on_table | 24 | 1/24 | 4.2% | （显示 10/23）<br>`sample_000038` → `node_19_put_sample_on_machine_table_2`<br>`sample_000073` → `node_33_turn_off_main_switch`<br>`sample_000087` → `node_19_put_sample_on_machine_table_2`<br>`sample_000226` → `node_18_reverse_sample`<br>`sample_000232` → `node_34_take_lock_from_table`<br>`sample_000238` → `node_18_reverse_sample`<br>`sample_000265` → `node_34_take_lock_from_table`<br>`sample_000309` → `node_33_turn_off_main_switch`<br>`sample_000329` → `node_19_put_sample_on_machine_table_2`<br>`sample_000416` → `node_3_turn_on_main_switch` |  |
| node_25_put_plier_on_table | 24 | 12/24 | 50.0% | （显示 10/12）<br>`sample_000151` → `node_34_take_lock_from_table`<br>`sample_000165` → `node_34_take_lock_from_table`<br>`sample_000233` → `node_34_take_lock_from_table`<br>`sample_000239` → `node_34_take_lock_from_table`<br>`sample_000260` → `node_34_take_lock_from_table`<br>`sample_000266` → `node_34_take_lock_from_table`<br>`sample_000297` → `node_34_take_lock_from_table`<br>`sample_000310` → `node_2_put_lock_on_table`<br>`sample_000368` → `node_34_take_lock_from_table`<br>`sample_000417` → `node_3_turn_on_main_switch` |  |
| node_26_take_protection_cover_from_ground | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000047` → `node_25_put_plier_on_table`<br>`sample_000124` → `node_25_put_plier_on_table`<br>`sample_000181` → `node_25_put_plier_on_table`<br>`sample_000271` → `node_25_put_plier_on_table`<br>`sample_000376` → `node_18_reverse_sample` |  |
| node_27_put_protection_cover_on_crimper | 6 | 3/6 | 50.0% | （显示 3/3）<br>`sample_000182` → `node_18_reverse_sample`<br>`sample_000272` → `node_24_put_sample_on_table`<br>`sample_000335` → `node_3_turn_on_main_switch` |  |
| node_28_turn_off_extractor_fan | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000042` → `node_25_put_plier_on_table`<br>`sample_000119` → `node_8_turn_on_extractor_fan`<br>`sample_000185` → `node_8_turn_on_extractor_fan`<br>`sample_000268` → `node_29_turn_off_water_pump`<br>`sample_000337` → `node_8_turn_on_extractor_fan`<br>`sample_000370` → `node_33_turn_off_main_switch` |  |
| node_29_turn_off_water_pump | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000043` → `node_33_turn_off_main_switch`<br>`sample_000120` → `node_25_put_plier_on_table`<br>`sample_000186` → `node_30_turn_off_air_compressor`<br>`sample_000267` → `node_25_put_plier_on_table`<br>`sample_000338` → `node_33_turn_off_main_switch`<br>`sample_000372` → `node_33_turn_off_main_switch` |  |
| node_30_turn_off_air_compressor | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000041` → `node_3_turn_on_main_switch`<br>`sample_000118` → `node_3_turn_on_main_switch`<br>`sample_000269` → `node_25_put_plier_on_table`<br>`sample_000336` → `node_31_move_pedal_to_original_place`<br>`sample_000371` → `node_19_put_sample_on_machine_table_2` |  |
| node_31_move_pedal_to_original_place | 6 | 2/6 | 33.3% | （显示 4/4）<br>`sample_000049` → `node_19_put_sample_on_machine_table_2`<br>`sample_000126` → `node_19_put_sample_on_machine_table_2`<br>`sample_000273` → `node_3_turn_on_main_switch`<br>`sample_000378` → `node_18_reverse_sample` |  |
| node_32_turn_off_crimper | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000117` → `node_25_put_plier_on_table`<br>`sample_000183` → `node_25_put_plier_on_table`<br>`sample_000270` → `node_4_turn_on_crimper`<br>`sample_000331` → `node_6_turn_on_air_compressor`<br>`sample_000369` → `node_25_put_plier_on_table` |  |
| node_34_take_lock_from_table | 5 | 1/5 | 20.0% | （显示 4/4）<br>`sample_000045` → `node_12_take_plier_from_table`<br>`sample_000122` → `node_25_put_plier_on_table`<br>`sample_000332` → `node_25_put_plier_on_table`<br>`sample_000374` → `node_25_put_plier_on_table` |  |
| node_35_lock_crimper | 5 | 0/5 | 0.0% | （显示 5/5）<br>`sample_000046` → `node_31_move_pedal_to_original_place`<br>`sample_000123` → `node_31_move_pedal_to_original_place`<br>`sample_000189` → `node_31_move_pedal_to_original_place`<br>`sample_000333` → `node_31_move_pedal_to_original_place`<br>`sample_000375` → `node_31_move_pedal_to_original_place` |  |

##### S7 — IMU ResNet10 Direct Node

| 低 Recall Node | 支持 | 正确 | Recall | 随机抽取误分类样本 → 预测 Node | 备注 |
| --- | --- | --- | --- | --- | --- |
| node_2_put_lock_on_table | 6 | 3/6 | 50.0% | （显示 3/3）<br>`sample_000131` → `node_34_take_lock_from_table`<br>`sample_000205` → `node_12_take_plier_from_table`<br>`sample_000275` → `node_24_put_sample_on_table` |  |
| node_5_adjust_parameters | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000011` → `node_4_turn_on_crimper`<br>`sample_000060` → `node_2_put_lock_on_table`<br>`sample_000137` → `node_4_turn_on_crimper`<br>`sample_000214` → `node_4_turn_on_crimper`<br>`sample_000284` → `node_4_turn_on_crimper`<br>`sample_000389` → `node_4_turn_on_crimper` |  |
| node_6_turn_on_air_compressor | 6 | 3/6 | 50.0% | （显示 3/3）<br>`sample_000133` → `node_30_turn_off_air_compressor`<br>`sample_000279` → `node_30_turn_off_air_compressor`<br>`sample_000385` → `node_29_turn_off_water_pump` |  |
| node_7_turn_on_water_pump | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000002` → `node_8_turn_on_extractor_fan`<br>`sample_000055` → `node_8_turn_on_extractor_fan`<br>`sample_000135` → `node_8_turn_on_extractor_fan`<br>`sample_000207` → `node_8_turn_on_extractor_fan`<br>`sample_000277` → `node_12_take_plier_from_table`<br>`sample_000384` → `node_29_turn_off_water_pump` |  |
| node_8_turn_on_extractor_fan | 6 | 2/6 | 33.3% | （显示 4/4）<br>`sample_000004` → `node_28_turn_off_extractor_fan`<br>`sample_000208` → `node_28_turn_off_extractor_fan`<br>`sample_000278` → `node_9_move_pedal_to_safe_location`<br>`sample_000383` → `node_28_turn_off_extractor_fan` |  |
| node_12_take_plier_from_table | 24 | 16/24 | 66.7% | （显示 8/8）<br>`sample_000012` → `node_34_take_lock_from_table`<br>`sample_000138` → `node_34_take_lock_from_table`<br>`sample_000190` → `node_26_take_protection_cover_from_ground`<br>`sample_000215` → `node_34_take_lock_from_table`<br>`sample_000298` → `node_34_take_lock_from_table`<br>`sample_000355` → `node_34_take_lock_from_table`<br>`sample_000390` → `node_34_take_lock_from_table`<br>`sample_000404` → `node_34_take_lock_from_table` |  |
| node_13_grip_sample_from_table_1 | 24 | 11/24 | 45.8% | （显示 10/13）<br>`sample_000027` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000062` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000167` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000216` → `node_12_take_plier_from_table`<br>`sample_000229` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000235` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000241` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000312` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000318` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000356` → `node_20_grip_sample_from_machine_table_3` |  |
| node_14_put_sample_under_electrodes_1 | 24 | 6/24 | 25.0% | （显示 10/18）<br>`sample_000063` → `node_21_put_sample_under_electrodes_2`<br>`sample_000105` → `node_21_put_sample_under_electrodes_2`<br>`sample_000154` → `node_21_put_sample_under_electrodes_2`<br>`sample_000168` → `node_23_inspect_sample`<br>`sample_000217` → `node_21_put_sample_under_electrodes_2`<br>`sample_000236` → `node_21_put_sample_under_electrodes_2`<br>`sample_000287` → `node_21_put_sample_under_electrodes_2`<br>`sample_000300` → `node_21_put_sample_under_electrodes_2`<br>`sample_000343` → `node_21_put_sample_under_electrodes_2`<br>`sample_000420` → `node_21_put_sample_under_electrodes_2` |  |
| node_15_press_pedal_1 | 24 | 19/24 | 79.2% | （显示 5/5）<br>`sample_000015` → `node_22_press_pedal_2`<br>`sample_000029` → `node_22_press_pedal_2`<br>`sample_000155` → `node_22_press_pedal_2`<br>`sample_000237` → `node_22_press_pedal_2`<br>`sample_000320` → `node_22_press_pedal_2` |  |
| node_16_put_sample_on_machine_table_1 | 21 | 11/21 | 52.4% | （显示 10/10）<br>`sample_000079` → `node_25_put_plier_on_table`<br>`sample_000093` → `node_25_put_plier_on_table`<br>`sample_000107` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000142` → `node_1_unlock_crimper`<br>`sample_000156` → `node_12_take_plier_from_table`<br>`sample_000219` → `node_12_take_plier_from_table`<br>`sample_000289` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000321` → `node_12_take_plier_from_table`<br>`sample_000345` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000359` → `node_25_put_plier_on_table` |  |
| node_19_put_sample_on_machine_table_2 | 21 | 6/21 | 28.6% | （显示 10/15）<br>`sample_000033` → `node_13_grip_sample_from_table_1`<br>`sample_000096` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000145` → `node_24_put_sample_on_table`<br>`sample_000159` → `node_16_put_sample_on_machine_table_1`<br>`sample_000173` → `node_16_put_sample_on_machine_table_1`<br>`sample_000254` → `node_16_put_sample_on_machine_table_1`<br>`sample_000292` → `node_24_put_sample_on_table`<br>`sample_000305` → `node_16_put_sample_on_machine_table_1`<br>`sample_000324` → `node_16_put_sample_on_machine_table_1`<br>`sample_000397` → `node_16_put_sample_on_machine_table_1` |  |
| node_21_put_sample_under_electrodes_2 | 21 | 10/21 | 47.6% | （显示 10/11）<br>`sample_000021` → `node_22_press_pedal_2`<br>`sample_000070` → `node_14_put_sample_under_electrodes_1`<br>`sample_000112` → `node_16_put_sample_on_machine_table_1`<br>`sample_000147` → `node_14_put_sample_under_electrodes_1`<br>`sample_000175` → `node_14_put_sample_under_electrodes_1`<br>`sample_000249` → `node_14_put_sample_under_electrodes_1`<br>`sample_000294` → `node_14_put_sample_under_electrodes_1`<br>`sample_000307` → `node_14_put_sample_under_electrodes_1`<br>`sample_000326` → `node_14_put_sample_under_electrodes_1`<br>`sample_000399` → `node_22_press_pedal_2` |  |
| node_22_press_pedal_2 | 21 | 9/21 | 42.9% | （显示 10/12）<br>`sample_000036` → `node_15_press_pedal_1`<br>`sample_000148` → `node_15_press_pedal_1`<br>`sample_000225` → `node_15_press_pedal_1`<br>`sample_000250` → `node_15_press_pedal_1`<br>`sample_000257` → `node_15_press_pedal_1`<br>`sample_000295` → `node_15_press_pedal_1`<br>`sample_000327` → `node_15_press_pedal_1`<br>`sample_000365` → `node_15_press_pedal_1`<br>`sample_000400` → `node_15_press_pedal_1`<br>`sample_000414` → `node_15_press_pedal_1` |  |
| node_24_put_sample_on_table | 24 | 6/24 | 25.0% | （显示 10/18）<br>`sample_000087` → `node_12_take_plier_from_table`<br>`sample_000101` → `node_12_take_plier_from_table`<br>`sample_000150` → `node_12_take_plier_from_table`<br>`sample_000202` → `node_12_take_plier_from_table`<br>`sample_000259` → `node_12_take_plier_from_table`<br>`sample_000265` → `node_12_take_plier_from_table`<br>`sample_000309` → `node_34_take_lock_from_table`<br>`sample_000315` → `node_12_take_plier_from_table`<br>`sample_000416` → `node_12_take_plier_from_table`<br>`sample_000430` → `node_12_take_plier_from_table` |  |
| node_25_put_plier_on_table | 24 | 5/24 | 20.8% | （显示 10/19）<br>`sample_000039` → `node_12_take_plier_from_table`<br>`sample_000088` → `node_2_put_lock_on_table`<br>`sample_000266` → `node_12_take_plier_from_table`<br>`sample_000297` → `node_12_take_plier_from_table`<br>`sample_000310` → `node_34_take_lock_from_table`<br>`sample_000330` → `node_34_take_lock_from_table`<br>`sample_000354` → `node_34_take_lock_from_table`<br>`sample_000368` → `node_34_take_lock_from_table`<br>`sample_000403` → `node_34_take_lock_from_table`<br>`sample_000417` → `node_34_take_lock_from_table` |  |
| node_27_put_protection_cover_on_crimper | 6 | 4/6 | 66.7% | （显示 2/2）<br>`sample_000048` → `node_23_inspect_sample`<br>`sample_000272` → `node_23_inspect_sample` |  |
| node_28_turn_off_extractor_fan | 6 | 4/6 | 66.7% | （显示 2/2）<br>`sample_000268` → `node_8_turn_on_extractor_fan`<br>`sample_000337` → `node_8_turn_on_extractor_fan` |  |
| node_29_turn_off_water_pump | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000043` → `node_28_turn_off_extractor_fan`<br>`sample_000120` → `node_8_turn_on_extractor_fan`<br>`sample_000186` → `node_8_turn_on_extractor_fan`<br>`sample_000267` → `node_34_take_lock_from_table`<br>`sample_000338` → `node_28_turn_off_extractor_fan`<br>`sample_000372` → `node_8_turn_on_extractor_fan` |  |
| node_31_move_pedal_to_original_place | 6 | 3/6 | 50.0% | （显示 3/3）<br>`sample_000126` → `node_34_take_lock_from_table`<br>`sample_000273` → `node_26_take_protection_cover_from_ground`<br>`sample_000340` → `node_26_take_protection_cover_from_ground` |  |
| node_35_lock_crimper | 5 | 1/5 | 20.0% | （显示 4/4）<br>`sample_000123` → `node_1_unlock_crimper`<br>`sample_000189` → `node_1_unlock_crimper`<br>`sample_000333` → `node_32_turn_off_crimper`<br>`sample_000375` → `node_32_turn_off_crimper` |  |

##### S8 — IMU Dilated Direct Node

| 低 Recall Node | 支持 | 正确 | Recall | 随机抽取误分类样本 → 预测 Node | 备注 |
| --- | --- | --- | --- | --- | --- |
| node_2_put_lock_on_table | 6 | 2/6 | 33.3% | （显示 4/4）<br>`sample_000006` → `node_25_put_plier_on_table`<br>`sample_000131` → `node_34_take_lock_from_table`<br>`sample_000205` → `node_34_take_lock_from_table`<br>`sample_000275` → `node_34_take_lock_from_table` |  |
| node_3_turn_on_main_switch | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000001` → `node_33_turn_off_main_switch`<br>`sample_000050` → `node_33_turn_off_main_switch`<br>`sample_000206` → `node_12_take_plier_from_table`<br>`sample_000276` → `node_26_take_protection_cover_from_ground`<br>`sample_000381` → `node_12_take_plier_from_table` |  |
| node_5_adjust_parameters | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000011` → `node_18_reverse_sample`<br>`sample_000060` → `node_3_turn_on_main_switch`<br>`sample_000137` → `node_4_turn_on_crimper`<br>`sample_000214` → `node_4_turn_on_crimper`<br>`sample_000284` → `node_4_turn_on_crimper`<br>`sample_000389` → `node_2_put_lock_on_table` |  |
| node_6_turn_on_air_compressor | 6 | 3/6 | 50.0% | （显示 3/3）<br>`sample_000056` → `node_12_take_plier_from_table`<br>`sample_000209` → `node_12_take_plier_from_table`<br>`sample_000385` → `node_29_turn_off_water_pump` |  |
| node_7_turn_on_water_pump | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000002` → `node_16_put_sample_on_machine_table_1`<br>`sample_000055` → `node_35_lock_crimper`<br>`sample_000135` → `node_8_turn_on_extractor_fan`<br>`sample_000207` → `node_12_take_plier_from_table`<br>`sample_000384` → `node_1_unlock_crimper` |  |
| node_8_turn_on_extractor_fan | 6 | 3/6 | 50.0% | （显示 3/3）<br>`sample_000054` → `node_32_turn_off_crimper`<br>`sample_000134` → `node_12_take_plier_from_table`<br>`sample_000278` → `node_32_turn_off_crimper` |  |
| node_9_move_pedal_to_safe_location | 6 | 4/6 | 66.7% | （显示 2/2）<br>`sample_000010` → `node_11_put_protection_cover_on_ground`<br>`sample_000281` → `node_34_take_lock_from_table` |  |
| node_12_take_plier_from_table | 24 | 16/24 | 66.7% | （显示 8/8）<br>`sample_000012` → `node_34_take_lock_from_table`<br>`sample_000138` → `node_34_take_lock_from_table`<br>`sample_000190` → `node_26_take_protection_cover_from_ground`<br>`sample_000215` → `node_34_take_lock_from_table`<br>`sample_000341` → `node_34_take_lock_from_table`<br>`sample_000390` → `node_34_take_lock_from_table`<br>`sample_000404` → `node_34_take_lock_from_table`<br>`sample_000418` → `node_34_take_lock_from_table` |  |
| node_13_grip_sample_from_table_1 | 24 | 9/24 | 37.5% | （显示 10/15）<br>`sample_000062` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000076` → `node_34_take_lock_from_table`<br>`sample_000104` → `node_26_take_protection_cover_from_ground`<br>`sample_000139` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000153` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000216` → `node_34_take_lock_from_table`<br>`sample_000262` → `node_10_remove_protection_cover_from_crimper`<br>`sample_000299` → `node_34_take_lock_from_table`<br>`sample_000318` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000356` → `node_20_grip_sample_from_machine_table_3` |  |
| node_14_put_sample_under_electrodes_1 | 24 | 4/24 | 16.7% | （显示 10/20）<br>`sample_000014` → `node_21_put_sample_under_electrodes_2`<br>`sample_000140` → `node_21_put_sample_under_electrodes_2`<br>`sample_000154` → `node_21_put_sample_under_electrodes_2`<br>`sample_000168` → `node_21_put_sample_under_electrodes_2`<br>`sample_000217` → `node_21_put_sample_under_electrodes_2`<br>`sample_000230` → `node_21_put_sample_under_electrodes_2`<br>`sample_000313` → `node_21_put_sample_under_electrodes_2`<br>`sample_000357` → `node_21_put_sample_under_electrodes_2`<br>`sample_000392` → `node_22_press_pedal_2`<br>`sample_000420` → `node_21_put_sample_under_electrodes_2` |  |
| node_15_press_pedal_1 | 24 | 9/24 | 37.5% | （显示 10/15）<br>`sample_000029` → `node_22_press_pedal_2`<br>`sample_000141` → `node_22_press_pedal_2`<br>`sample_000169` → `node_22_press_pedal_2`<br>`sample_000218` → `node_22_press_pedal_2`<br>`sample_000231` → `node_22_press_pedal_2`<br>`sample_000243` → `node_21_put_sample_under_electrodes_2`<br>`sample_000264` → `node_22_press_pedal_2`<br>`sample_000288` → `node_22_press_pedal_2`<br>`sample_000301` → `node_22_press_pedal_2`<br>`sample_000320` → `node_22_press_pedal_2` |  |
| node_16_put_sample_on_machine_table_1 | 21 | 14/21 | 66.7% | （显示 7/7）<br>`sample_000093` → `node_26_take_protection_cover_from_ground`<br>`sample_000156` → `node_32_turn_off_crimper`<br>`sample_000219` → `node_12_take_plier_from_table`<br>`sample_000289` → `node_26_take_protection_cover_from_ground`<br>`sample_000302` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000321` → `node_12_take_plier_from_table`<br>`sample_000359` → `node_20_grip_sample_from_machine_table_3` |  |
| node_19_put_sample_on_machine_table_2 | 21 | 6/21 | 28.6% | （显示 10/15）<br>`sample_000068` → `node_25_put_plier_on_table`<br>`sample_000110` → `node_25_put_plier_on_table`<br>`sample_000159` → `node_16_put_sample_on_machine_table_1`<br>`sample_000173` → `node_12_take_plier_from_table`<br>`sample_000222` → `node_16_put_sample_on_machine_table_1`<br>`sample_000247` → `node_34_take_lock_from_table`<br>`sample_000254` → `node_16_put_sample_on_machine_table_1`<br>`sample_000324` → `node_16_put_sample_on_machine_table_1`<br>`sample_000397` → `node_20_grip_sample_from_machine_table_3`<br>`sample_000411` → `node_2_put_lock_on_table` |  |
| node_20_grip_sample_from_machine_table_3 | 21 | 15/21 | 71.4% | （显示 6/6）<br>`sample_000069` → `node_26_take_protection_cover_from_ground`<br>`sample_000097` → `node_26_take_protection_cover_from_ground`<br>`sample_000111` → `node_26_take_protection_cover_from_ground`<br>`sample_000293` → `node_26_take_protection_cover_from_ground`<br>`sample_000306` → `node_26_take_protection_cover_from_ground`<br>`sample_000325` → `node_26_take_protection_cover_from_ground` |  |
| node_21_put_sample_under_electrodes_2 | 21 | 13/21 | 61.9% | （显示 8/8）<br>`sample_000070` → `node_14_put_sample_under_electrodes_1`<br>`sample_000098` → `node_16_put_sample_on_machine_table_1`<br>`sample_000112` → `node_16_put_sample_on_machine_table_1`<br>`sample_000175` → `node_14_put_sample_under_electrodes_1`<br>`sample_000307` → `node_14_put_sample_under_electrodes_1`<br>`sample_000326` → `node_25_put_plier_on_table`<br>`sample_000364` → `node_19_put_sample_on_machine_table_2`<br>`sample_000399` → `node_22_press_pedal_2` |  |
| node_24_put_sample_on_table | 24 | 6/24 | 25.0% | （显示 10/18）<br>`sample_000087` → `node_12_take_plier_from_table`<br>`sample_000101` → `node_12_take_plier_from_table`<br>`sample_000150` → `node_12_take_plier_from_table`<br>`sample_000178` → `node_12_take_plier_from_table`<br>`sample_000202` → `node_26_take_protection_cover_from_ground`<br>`sample_000226` → `node_12_take_plier_from_table`<br>`sample_000265` → `node_12_take_plier_from_table`<br>`sample_000315` → `node_12_take_plier_from_table`<br>`sample_000329` → `node_12_take_plier_from_table`<br>`sample_000416` → `node_34_take_lock_from_table` |  |
| node_25_put_plier_on_table | 24 | 18/24 | 75.0% | （显示 6/6）<br>`sample_000039` → `node_34_take_lock_from_table`<br>`sample_000074` → `node_12_take_plier_from_table`<br>`sample_000165` → `node_34_take_lock_from_table`<br>`sample_000227` → `node_34_take_lock_from_table`<br>`sample_000266` → `node_34_take_lock_from_table`<br>`sample_000403` → `node_2_put_lock_on_table` |  |
| node_27_put_protection_cover_on_crimper | 6 | 4/6 | 66.7% | （显示 2/2）<br>`sample_000272` → `node_23_inspect_sample`<br>`sample_000335` → `node_23_inspect_sample` |  |
| node_28_turn_off_extractor_fan | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000042` → `node_8_turn_on_extractor_fan`<br>`sample_000119` → `node_8_turn_on_extractor_fan`<br>`sample_000185` → `node_8_turn_on_extractor_fan`<br>`sample_000268` → `node_8_turn_on_extractor_fan`<br>`sample_000337` → `node_4_turn_on_crimper`<br>`sample_000370` → `node_8_turn_on_extractor_fan` |  |
| node_29_turn_off_water_pump | 6 | 1/6 | 16.7% | （显示 5/5）<br>`sample_000120` → `node_7_turn_on_water_pump`<br>`sample_000186` → `node_12_take_plier_from_table`<br>`sample_000267` → `node_12_take_plier_from_table`<br>`sample_000338` → `node_8_turn_on_extractor_fan`<br>`sample_000372` → `node_7_turn_on_water_pump` |  |
| node_31_move_pedal_to_original_place | 6 | 0/6 | 0.0% | （显示 6/6）<br>`sample_000049` → `node_26_take_protection_cover_from_ground`<br>`sample_000126` → `node_26_take_protection_cover_from_ground`<br>`sample_000180` → `node_26_take_protection_cover_from_ground`<br>`sample_000273` → `node_26_take_protection_cover_from_ground`<br>`sample_000340` → `node_26_take_protection_cover_from_ground`<br>`sample_000378` → `node_26_take_protection_cover_from_ground` |  |
| node_35_lock_crimper | 5 | 3/5 | 60.0% | （显示 2/2）<br>`sample_000189` → `node_25_put_plier_on_table`<br>`sample_000375` → `node_30_turn_off_air_compressor` |  |

S9–S12 为 Direct Tier3，结果文件没有 Node 输出，因此不存在可列出的低 Recall Node 或 Node 误分类样本；其 31 Tier3 类别影响已纳入 5.1.3。

双视角帧与右手 EMG/IMU 的小规模检查版见 [S1–S12 低 Recall 多模态质量检查 Pilot](analysis/a_as_test_seed_1/S1_S12_LOW_RECALL_MULTIMODAL_QUALITY_CHECK_PILOT.md)。

## 6. 类别级影响：31 Tier3

| ID | Tier3 | 支持 | A0 R | A0 F1 | A1 ΔR/ΔF1/Δ正确 | A2 ΔR/ΔF1/Δ正确 | A3 ΔR/ΔF1/Δ正确 | A4 ΔR/ΔF1/Δ正确 | A5 ΔR/ΔF1/Δ正确 | A6 ΔR/ΔF1/Δ正确 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | adjust parameters | 6 | 100.0% | 100.0% | -33.3 / -20.0 / -2 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 1 | grip sample from machine table | 42 | 90.5% | 95.0% | +9.5 / +2.7 / +4 | +7.1 / +3.8 / +3 | +2.4 / +0.1 / +1 | +4.8 / +2.6 / +2 | +2.4 / +0.1 / +1 | +2.4 / +1.3 / +1 |
| 2 | grip sample from table | 24 | 95.8% | 97.9% | +4.2 / +2.1 / +1 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 3 | inspect sample | 17 | 94.1% | 97.0% | +5.9 / +3.0 / +1 | +5.9 / +3.0 / +1 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +5.9 / +3.0 / +1 | +0.0 / +0.0 / +0 |
| 4 | lock crimper | 5 | 100.0% | 83.3% | -20.0 / +5.6 / -1 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | -20.0 / -10.6 / -1 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 5 | move pedal to original location | 6 | 100.0% | 100.0% | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / -7.7 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 6 | move pedal to safe location | 6 | 100.0% | 100.0% | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 7 | place sample under electrodes | 45 | 100.0% | 98.9% | +0.0 / +1.1 / +0 | +0.0 / +1.1 / +0 | +0.0 / -2.1 / +0 | +0.0 / -2.1 / +0 | +0.0 / +1.1 / +0 | +0.0 / +1.1 / +0 |
| 8 | press pedal | 45 | 97.8% | 98.9% | +2.2 / +1.1 / +1 | +2.2 / +1.1 / +1 | -4.4 / -2.3 / -2 | -4.4 / -2.3 / -2 | +2.2 / +1.1 / +1 | +2.2 / +1.1 / +1 |
| 9 | put lock on table | 6 | 100.0% | 100.0% | -16.7 / -28.6 / -1 | +0.0 / -14.3 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 10 | put plier on table | 24 | 100.0% | 100.0% | -20.8 / -11.6 / -5 | -4.2 / -2.1 / -1 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 11 | put protection cover on crimper | 6 | 100.0% | 92.3% | +0.0 / +7.7 / +0 | +0.0 / +7.7 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 12 | put protection cover on ground | 6 | 100.0% | 92.3% | +0.0 / +7.7 / +0 | +0.0 / +7.7 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 13 | put sample on machine table | 42 | 100.0% | 100.0% | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | -2.4 / -1.2 / -1 | +0.0 / +0.0 / +0 | -2.4 / -1.2 / -1 | +0.0 / +0.0 / +0 |
| 14 | put sample on table | 24 | 41.7% | 57.1% | +37.5 / +31.2 / +9 | +25.0 / +20.9 / +6 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 15 | remove protection cover from crimper | 6 | 83.3% | 90.9% | +16.7 / +9.1 / +1 | +16.7 / +9.1 / +1 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 16 | reverse sample | 21 | 100.0% | 89.4% | +0.0 / +10.6 / +0 | +0.0 / +8.3 / +0 | +0.0 / +1.9 / +0 | +0.0 / +4.0 / +0 | +0.0 / +4.0 / +0 | +0.0 / +1.9 / +0 |
| 17 | take lock from table | 5 | 100.0% | 90.9% | +0.0 / -7.6 / +0 | +0.0 / +0.0 / +0 | +0.0 / +9.1 / +0 | +0.0 / +0.0 / +0 | +0.0 / +9.1 / +0 | +0.0 / +0.0 / +0 |
| 18 | take plier from table | 24 | 95.8% | 75.4% | +4.2 / +13.5 / +1 | +0.0 / +9.8 / +0 | +0.0 / -1.2 / +0 | +0.0 / +0.0 / +0 | +0.0 / -1.2 / +0 | +0.0 / +0.0 / +0 |
| 19 | take protection cover from ground | 6 | 83.3% | 90.9% | +16.7 / +9.1 / +1 | +16.7 / +9.1 / +1 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 20 | turn off air compressor | 6 | 33.3% | 44.4% | +50.0 / +46.5 / +3 | +50.0 / +46.5 / +3 | +16.7 / +15.6 / +1 | +16.7 / +15.6 / +1 | +33.3 / +28.3 / +2 | +0.0 / +0.0 / +0 |
| 21 | turn off crimper | 6 | 83.3% | 90.9% | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +16.7 / +9.1 / +1 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 22 | turn off extractor fan | 6 | 100.0% | 92.3% | -16.7 / -1.4 / -1 | -16.7 / -1.4 / -1 | -16.7 / -9.0 / -1 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | -16.7 / -1.4 / -1 |
| 23 | turn off main switch | 5 | 100.0% | 100.0% | -20.0 / -11.1 / -1 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 24 | turn off water pump | 6 | 83.3% | 90.9% | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / -7.6 / +0 | +0.0 / +0.0 / +0 | +0.0 / -7.6 / +0 | +0.0 / +0.0 / +0 |
| 25 | turn on air compressor | 6 | 83.3% | 62.5% | +16.7 / +12.5 / +1 | +16.7 / +17.5 / +1 | +0.0 / +4.2 / +0 | +0.0 / +4.2 / +0 | +0.0 / +8.9 / +0 | +0.0 / +0.0 / +0 |
| 26 | turn on crimper | 6 | 83.3% | 83.3% | +0.0 / -11.9 / +0 | +0.0 / -6.4 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 27 | turn on extractor fan | 6 | 66.7% | 80.0% | +16.7 / +3.3 / +1 | +16.7 / +3.3 / +1 | +0.0 / -7.3 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +16.7 / +3.3 / +1 |
| 28 | turn on main switch | 6 | 100.0% | 100.0% | +0.0 / -7.7 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |
| 29 | turn on water pump | 6 | 100.0% | 92.3% | -16.7 / -9.0 / -1 | -16.7 / -9.0 / -1 | -16.7 / -9.0 / -1 | +0.0 / +0.0 / +0 | -16.7 / -9.0 / -1 | +0.0 / +0.0 / +0 |
| 30 | unlock crimper | 6 | 66.7% | 72.7% | +0.0 / +0.0 / +0 | +0.0 / +7.3 / +0 | +0.0 / +0.0 / +0 | +0.0 / +7.3 / +0 | +0.0 / +0.0 / +0 | +0.0 / +0.0 / +0 |

## 7. 混淆对变化

### 7.1 A0 当前前 12 个 Node 混淆对在各融合中的变化

| 真实 → 预测 | A0 | A1 数量(Δ) | A2 数量(Δ) | A3 数量(Δ) | A4 数量(Δ) | A5 数量(Δ) | A6 数量(Δ) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| node_24_put_sample_on_table → node_12_take_plier_from_table | 13 | 1 (-12) | 5 (-8) | 14 (+1) | 13 (+0) | 14 (+1) | 13 (+0) |
| node_30_turn_off_air_compressor → node_6_turn_on_air_compressor | 4 | 1 (-3) | 1 (-3) | 3 (-1) | 3 (-1) | 2 (-2) | 4 (+0) |
| node_17_grip_sample_from_machine_table_2 → node_18_reverse_sample | 3 | 0 (-3) | 1 (-2) | 2 (-1) | 1 (-2) | 2 (-1) | 2 (-1) |
| node_23_inspect_sample → node_18_reverse_sample | 1 | 0 (-1) | 0 (-1) | 1 (+0) | 1 (+0) | 0 (-1) | 1 (+0) |
| node_1_unlock_crimper → node_4_turn_on_crimper | 1 | 2 (+1) | 1 (+0) | 1 (+0) | 1 (+0) | 1 (+0) | 1 (+0) |
| node_1_unlock_crimper → node_35_lock_crimper | 1 | 0 (-1) | 1 (+0) | 1 (+0) | 1 (+0) | 1 (+0) | 1 (+0) |
| node_4_turn_on_crimper → node_35_lock_crimper | 1 | 0 (-1) | 1 (+0) | 1 (+0) | 1 (+0) | 1 (+0) | 1 (+0) |
| node_6_turn_on_air_compressor → node_30_turn_off_air_compressor | 1 | 0 (-1) | 0 (-1) | 1 (+0) | 1 (+0) | 1 (+0) | 1 (+0) |
| node_8_turn_on_extractor_fan → node_28_turn_off_extractor_fan | 1 | 0 (-1) | 0 (-1) | 1 (+0) | 1 (+0) | 1 (+0) | 0 (-1) |
| node_32_turn_off_crimper → node_1_unlock_crimper | 1 | 0 (-1) | 0 (-1) | 1 (+0) | 0 (-1) | 1 (+0) | 1 (+0) |
| node_29_turn_off_water_pump → node_7_turn_on_water_pump | 1 | 1 (+0) | 1 (+0) | 1 (+0) | 1 (+0) | 1 (+0) | 1 (+0) |
| node_10_remove_protection_cover_from_crimper → node_27_put_protection_cover_on_crimper | 1 | 0 (-1) | 0 (-1) | 1 (+0) | 1 (+0) | 1 (+0) | 1 (+0) |

### 7.2 各方法新引入/放大的主要混淆

- **A1**：`node_25_put_plier_on_table → node_12_take_plier_from_table` 5 次（比 A0 +5）；`node_24_put_sample_on_table → node_2_put_lock_on_table` 3 次（比 A0 +3）；`node_1_unlock_crimper → node_4_turn_on_crimper` 2 次（比 A0 +1）；`node_35_lock_crimper → node_1_unlock_crimper` 1 次（比 A0 +1）；`node_33_turn_off_main_switch → node_3_turn_on_main_switch` 1 次（比 A0 +1）；`node_32_turn_off_crimper → node_4_turn_on_crimper` 1 次（比 A0 +1）
- **A2**：`node_24_put_sample_on_table → node_2_put_lock_on_table` 2 次（比 A0 +2）；`node_32_turn_off_crimper → node_4_turn_on_crimper` 1 次（比 A0 +1）；`node_28_turn_off_extractor_fan → node_8_turn_on_extractor_fan` 1 次（比 A0 +1）；`node_25_put_plier_on_table → node_12_take_plier_from_table` 1 次（比 A0 +1）；`node_7_turn_on_water_pump → node_6_turn_on_air_compressor` 1 次（比 A0 +1）
- **A3**：`node_22_press_pedal_2 → node_21_put_sample_under_electrodes_2` 3 次（比 A0 +2）；`node_24_put_sample_on_table → node_12_take_plier_from_table` 14 次（比 A0 +1）；`node_28_turn_off_extractor_fan → node_8_turn_on_extractor_fan` 1 次（比 A0 +1）；`node_19_put_sample_on_machine_table_2 → node_20_grip_sample_from_machine_table_3` 1 次（比 A0 +1）；`node_7_turn_on_water_pump → node_29_turn_off_water_pump` 1 次（比 A0 +1）
- **A4**：`node_22_press_pedal_2 → node_21_put_sample_under_electrodes_2` 3 次（比 A0 +2）；`node_35_lock_crimper → node_31_move_pedal_to_original_place` 1 次（比 A0 +1）
- **A5**：`node_24_put_sample_on_table → node_12_take_plier_from_table` 14 次（比 A0 +1）；`node_19_put_sample_on_machine_table_2 → node_20_grip_sample_from_machine_table_3` 1 次（比 A0 +1）；`node_7_turn_on_water_pump → node_29_turn_off_water_pump` 1 次（比 A0 +1）
- **A6**：`node_28_turn_off_extractor_fan → node_8_turn_on_extractor_fan` 1 次（比 A0 +1）

## 8. 右手 IMU 与 EMG 的互补性

| 样本关系 | clips | 含义 |
| --- | --- | --- |
| A4 对、A5 错 | 4 | 偏向 IMU 有利样本 |
| A5 对、A4 错 | 6 | 偏向 EMG 有利样本 |
| A4/A5 都错，A6 修正 | 1 | 传感器内部互补的直接证据 |
| A4/A5 至少一个对，A6 变错 | 6 | 联合融合覆盖单传感器优势的代价 |
| A0 错且 A4/A5/A6 至少一个修正 | 8 | 可穿戴信息池的潜在上限线索 |

A6 是否优于 A4/A5，不能只看总分；关键是它能否保留两种单传感器各自修正的类别，同时减少联合后新引入的错误。上表与第 5 节的类别净正确数应结合解读。

## 9. S1–S12 Sensor-only 与摄像头模型联合分析

S1–S8 中最好的 sensor-only Node 模型是 **S3（IMU ResNet10 Tier3→M2 Node）**，Node Macro-F1 为 79.95%，仍比 A0 低 -10.10 pp。S1–S12 中 Tier3 Macro-F1 最高的是 **S3（IMU ResNet10 Tier3→M2 Node）**，为 77.69%；若只看 S9–S12 Direct Tier3，则最高的是 **S11**，为 67.20%。因此这些信号目前不适合替代摄像头，但仍需看其错误是否与摄像头错在不同样本上。

### 9.1 总体、Normal/Fault 与 Stage

| 条件 | 输入/训练 | Node Acc | Node Macro-F1 | ΔA0 pp | 最弱 Node Recall | Tier3 Acc | Tier3 Macro-F1 | ΔA0 pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | EMG ResNet10 Tier3→M2 Node | 25.29% | 20.61% | -69.45 | 0.00% | 25.52% | 20.91% | -68.11 |
| S2 | EMG Dilated Tier3→M2 Node | 50.81% | 47.57% | -42.49 | 0.00% | 52.20% | 47.38% | -41.64 |
| S3 | IMU ResNet10 Tier3→M2 Node | 84.69% | 79.95% | -10.10 | 0.00% | 84.92% | 77.69% | -11.33 |
| S4 | IMU Dilated Tier3→M2 Node | 76.10% | 70.97% | -19.09 | 0.00% | 77.26% | 69.00% | -20.02 |
| S5 | EMG ResNet10 Direct Node | 23.20% | 17.62% | -72.44 | 0.00% | 23.90% | 18.37% | -70.65 |
| S6 | EMG Dilated Direct Node | 25.99% | 21.94% | -68.11 | 0.00% | 28.54% | 23.78% | -65.24 |
| S7 | IMU ResNet10 Direct Node | 59.40% | 58.99% | -31.07 | 0.00% | 71.00% | 63.18% | -25.84 |
| S8 | IMU Dilated Direct Node | 58.47% | 54.62% | -35.44 | 0.00% | 67.52% | 57.36% | -31.66 |
| S9 | EMG ResNet10 Direct Tier3 | NA | NA | NA | NA | 24.13% | 17.17% | -71.85 |
| S10 | EMG Dilated Direct Tier3 | NA | NA | NA | NA | 34.80% | 25.01% | -64.02 |
| S11 | IMU ResNet10 Direct Tier3 | NA | NA | NA | NA | 74.01% | 67.20% | -21.82 |
| S12 | IMU Dilated Direct Tier3 | NA | NA | NA | NA | 66.59% | 57.87% | -31.15 |

| 条件 | 子集 | N | Node Acc | Node Macro-F1 |
| --- | --- | --- | --- | --- |
| S1 | Normal | 294 | 25.51% | 20.78% |
| S1 | Fault | 137 | 24.82% | 20.16% |
| S1 | Stage 1 | 66 | 25.76% | 25.31% |
| S1 | Stage 2 | 308 | 25.32% | 27.60% |
| S1 | Stage 3 | 57 | 24.56% | 27.38% |
| S2 | Normal | 294 | 53.06% | 48.31% |
| S2 | Fault | 137 | 45.99% | 45.09% |
| S2 | Stage 1 | 66 | 46.97% | 47.20% |
| S2 | Stage 2 | 308 | 49.03% | 55.32% |
| S2 | Stage 3 | 57 | 64.91% | 63.03% |
| S3 | Normal | 294 | 88.44% | 82.86% |
| S3 | Fault | 137 | 76.64% | 73.47% |
| S3 | Stage 1 | 66 | 77.27% | 76.38% |
| S3 | Stage 2 | 308 | 87.01% | 90.27% |
| S3 | Stage 3 | 57 | 80.70% | 80.34% |
| S4 | Normal | 294 | 78.91% | 72.50% |
| S4 | Fault | 137 | 70.07% | 67.11% |
| S4 | Stage 1 | 66 | 69.70% | 70.75% |
| S4 | Stage 2 | 308 | 78.25% | 81.60% |
| S4 | Stage 3 | 57 | 71.93% | 72.02% |
| S5 | Normal | 294 | 26.87% | 20.64% |
| S5 | Fault | 137 | 15.33% | 10.85% |
| S5 | Stage 1 | 66 | 15.15% | 20.96% |
| S5 | Stage 2 | 308 | 23.70% | 25.28% |
| S5 | Stage 3 | 57 | 29.82% | 30.16% |
| S6 | Normal | 294 | 26.53% | 20.68% |
| S6 | Fault | 137 | 24.82% | 21.77% |
| S6 | Stage 1 | 66 | 21.21% | 28.28% |
| S6 | Stage 2 | 308 | 27.27% | 26.48% |
| S6 | Stage 3 | 57 | 24.56% | 28.21% |
| S7 | Normal | 294 | 65.31% | 63.96% |
| S7 | Fault | 137 | 46.72% | 45.31% |
| S7 | Stage 1 | 66 | 63.64% | 63.71% |
| S7 | Stage 2 | 308 | 57.14% | 58.91% |
| S7 | Stage 3 | 57 | 66.67% | 68.20% |
| S8 | Normal | 294 | 63.27% | 57.44% |
| S8 | Fault | 137 | 48.18% | 46.04% |
| S8 | Stage 1 | 66 | 54.55% | 59.25% |
| S8 | Stage 2 | 308 | 59.09% | 62.75% |
| S8 | Stage 3 | 57 | 59.65% | 62.25% |

### 9.2 M2 历史、编码器与训练目标的影响

| Tier3 encoder→M2 Node | 独立 Direct Node | ΔNode Macro-F1 pp | ΔNode Acc pp | ΔFault Node F1 pp |
| --- | --- | --- | --- | --- |
| S1 | S5 | +2.99 | +2.09 | +9.30 |
| S2 | S6 | +25.62 | +24.83 | +23.32 |
| S3 | S7 | +20.96 | +25.29 | +28.16 |
| S4 | S8 | +16.35 | +17.63 | +21.07 |

| 比较范围 | Dilated−ResNet10 | ΔMacro-F1 pp | ΔAccuracy pp |
| --- | --- | --- | --- |
| EMG M2 Node | S2−S1 | +26.96 | +25.52 |
| IMU M2 Node | S4−S3 | -8.99 | -8.58 |
| EMG Direct Node | S6−S5 | +4.32 | +2.78 |
| IMU Direct Node | S8−S7 | -4.37 | -0.93 |
| EMG Direct Tier3 | S10−S9 | +7.84 | +10.67 |
| IMU Direct Tier3 | S12−S11 | -9.33 | -7.42 |

结果呈现明确的模态依赖：Dilated 对 EMG 更有利，而 ResNet10 对 IMU 更有利；S1–S4 的历史 M2 在四个配对中均优于各自独立 Direct Node，但提升幅度差异很大。这说明“是否使用历史”与“1D encoder 选择”不能跨模态共用一个结论。不过这些配对同时改变了上游训练目标（Tier3 预训练后冻结 vs Direct Node 端到端），因此增益属于完整训练流程，不能全部归因于 M2 历史。

### 9.3 Sensor-only 类别级影响

S1–S8 的 35 Node 图与 S1–S12 的 31 Tier3 图已统一放在 **5.1.2–5.1.3**，便于与 A0–A6 连续比较。sensor-only 的最弱 Node Recall 均为 0，说明每个模型至少完全漏掉一个测试中存在的 Node；但局部高 Recall 类别仍是后续门控融合可能利用的候选信息。

| 条件 | Recall高于A0类数 | 相同 | 低于A0类数 | 相对A0 Recall变化最高的3类 |
| --- | --- | --- | --- | --- |
| S1 | 0 | 1 | 34 | node_33_turn_off_main_switch (+0.0)；node_25_put_plier_on_table (-12.5)；node_31_move_pedal_to_original_place (-16.7) |
| S2 | 2 | 3 | 30 | node_30_turn_off_air_compressor (+50.0)；node_32_turn_off_crimper (+16.7)；node_7_turn_on_water_pump (+0.0) |
| S3 | 10 | 9 | 16 | node_30_turn_off_air_compressor (+66.7)；node_1_unlock_crimper (+16.7)；node_8_turn_on_extractor_fan (+16.7) |
| S4 | 6 | 5 | 24 | node_30_turn_off_air_compressor (+66.7)；node_1_unlock_crimper (+33.3)；node_8_turn_on_extractor_fan (+16.7) |
| S5 | 0 | 1 | 34 | node_31_move_pedal_to_original_place (+0.0)；node_19_put_sample_on_machine_table_2 (-9.5)；node_32_turn_off_crimper (-16.7) |
| S6 | 0 | 1 | 34 | node_33_turn_off_main_switch (+0.0)；node_19_put_sample_on_machine_table_2 (-9.5)；node_8_turn_on_extractor_fan (-16.7) |
| S7 | 8 | 4 | 23 | node_30_turn_off_air_compressor (+66.7)；node_1_unlock_crimper (+16.7)；node_4_turn_on_crimper (+16.7) |
| S8 | 7 | 2 | 26 | node_30_turn_off_air_compressor (+50.0)；node_1_unlock_crimper (+33.3)；node_10_remove_protection_cover_from_crimper (+16.7) |

S1–S8 所有低 Recall Node 的完整样本名、正确/错误状态和预测类别见 `analysis/a_as_test_seed_1/SENSOR_LOW_RECALL_NODE_SAMPLE_INDEX.md` 与 `sensor_low_recall_node_samples.csv`。

### 9.4 模态与训练方式的互补性

![模态与训练方式互补性](analysis/a_as_test_seed_1/modality_training_complementarity.png)

图 A 比较独立性能；图 B 将每个模型相对 A0 的修正与损害拆开；图 C 计算任意两模型只要一个预测正确就算正确的 oracle 上限。Oracle 增益表示错误集合不重叠，并不等于当前 late fusion/gate 可以达到的真实收益。

| Sensor模型 | 修正A0错误 | 破坏A0正确 | 净正确 | A0+Sensor Oracle Acc | Oracle较好单模增益pp |
| --- | --- | --- | --- | --- | --- |
| S1 | 5 | 291 | -286 | 92.81% | +1.16 |
| S2 | 15 | 191 | -176 | 95.13% | +3.48 |
| S3 | 22 | 52 | -30 | 96.75% | +5.10 |
| S4 | 19 | 86 | -67 | 96.06% | +4.41 |
| S5 | 3 | 298 | -295 | 92.34% | +0.70 |
| S6 | 3 | 286 | -283 | 92.34% | +0.70 |
| S7 | 15 | 154 | -139 | 95.13% | +3.48 |
| S8 | 16 | 159 | -143 | 95.36% | +3.71 |

在当前预测上，A0+S3 的 oracle accuracy 为 96.75%，比 A0 高 +5.10 pp；A0+A1 的 oracle accuracy 更高，达到 97.68%。这说明 IMU 仍含有摄像头未覆盖的信息，但第二视角的互补上限在本次运行中更大。

这里最值得区分的是“独立性能”和“互补潜力”：一个 sensor-only 模型即使总体较弱，仍可能修正少量 A0 错误；但如果同时破坏大量 A0 正确样本，就必须采用以 A0 为锚点、初始严格回退 A0 的稀疏 gate/residual，不能直接平均概率。

### 9.5 训练拟合与跨参与者泛化差距

| 条件 | 训练目标 | Epoch | 首轮Train Acc | 末轮Train Acc | 末轮Loss | Test Acc | Train−Test pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A3 | node | 50 | 100.00% | 100.00% | 0.0000 | 90.95% | 9.05 |
| S1 | node | 50 | 77.25% | 100.00% | 0.0006 | 25.29% | 74.71 |
| S2 | node | 50 | 78.69% | 100.00% | 0.0007 | 50.81% | 49.19 |
| S3 | node | 50 | 81.69% | 100.00% | 0.0001 | 84.69% | 15.31 |
| S4 | node | 50 | 83.54% | 98.98% | 0.0501 | 76.10% | 22.87 |
| S5 | node | 50 | 14.62% | 92.21% | 0.2384 | 23.20% | 69.01 |
| S6 | node | 50 | 6.97% | 88.39% | 0.3129 | 25.99% | 62.40 |
| S7 | node | 50 | 33.33% | 93.72% | 0.1432 | 59.40% | 34.32 |
| S8 | node | 50 | 32.79% | 95.70% | 0.1091 | 58.47% | 37.23 |
| S9 | tier3 | 50 | 23.63% | 94.19% | 0.1629 | 24.13% | 70.06 |
| S10 | tier3 | 50 | 9.97% | 94.95% | 0.1398 | 34.80% | 60.14 |
| S11 | tier3 | 50 | 39.21% | 97.88% | 0.0753 | 74.01% | 23.87 |
| S12 | tier3 | 50 | 38.66% | 98.98% | 0.0321 | 66.59% | 32.39 |

S1–S3 的末轮训练准确率达到 100%，S4 也接近 99%，但测试性能差异很大，尤其 EMG 条件存在显著跨参与者泛化缺口；因此低测试性能主要不是简单的训练集欠拟合。A3 从首轮开始训练准确率即为 100%、loss 已接近 0，说明冻结 A0 anchor 在训练样本上几乎没有可供 cross-view adapter 学习的错误，当前 residual 更容易学习置信度微调而不是稳健修错。该表只作诊断，不使用测试集选择 epoch。

## 10. 探索性 paired clip-level bootstrap

使用 10000 次配对 clip bootstrap；每次在 `Normal/Fault × Stage` 联合层内有放回抽样，保持各层样本量，并在同一次抽样中同时计算候选与 A0。CI 是本次 A_as_test/seed_1 测试集上的采样不确定性，不包含换测试者、换 seed 或重新训练的不确定性。

| 条件 | 指标 | 平均 Δ pp | 95% CI (pp) | P(Δ>0) |
| --- | --- | --- | --- | --- |
| A1 | Node accuracy | +2.80 | [+0.00, +5.57] | 97.2% |
| A1 | Node Macro-F1 | +1.83 | [-2.38, +5.81] | 80.8% |
| A1 | Node Macro-Recall | +1.40 | [-2.59, +5.06] | 77.1% |
| A1 | Tier3 Macro-F1 | +1.88 | [-2.88, +6.35] | 78.7% |
| A1 | Normal Node Macro-F1 | -0.12 | [-5.45, +4.81] | 49.0% |
| A1 | Fault Node Macro-F1 | +6.82 | [-0.00, +13.52] | 97.5% |
| A2 | Node accuracy | +3.48 | [+1.39, +5.80] | 99.9% |
| A2 | Node Macro-F1 | +3.87 | [+0.40, +7.28] | 98.5% |
| A2 | Node Macro-Recall | +3.70 | [+0.56, +6.75] | 98.7% |
| A2 | Tier3 Macro-F1 | +4.15 | [+0.28, +7.96] | 98.1% |
| A2 | Normal Node Macro-F1 | +1.78 | [-2.70, +5.99] | 79.2% |
| A2 | Fault Node Macro-F1 | +8.65 | [+3.15, +14.29] | 99.9% |
| A3 | Node accuracy | -0.69 | [-1.86, +0.46] | 8.2% |
| A3 | Node Macro-F1 | -0.46 | [-2.46, +1.45] | 31.4% |
| A3 | Node Macro-Recall | -0.74 | [-2.50, +0.90] | 18.8% |
| A3 | Tier3 Macro-F1 | -0.31 | [-2.51, +1.75] | 38.5% |
| A3 | Normal Node Macro-F1 | -1.49 | [-4.02, +0.22] | 5.9% |
| A3 | Fault Node Macro-F1 | +0.98 | [-2.41, +4.25] | 66.6% |
| A4 | Node accuracy | +0.24 | [-0.93, +1.39] | 57.6% |
| A4 | Node Macro-F1 | +0.49 | [-1.38, +2.56] | 68.8% |
| A4 | Node Macro-Recall | +0.39 | [-1.46, +2.26] | 66.5% |
| A4 | Tier3 Macro-F1 | +0.64 | [-1.44, +2.91] | 72.2% |
| A4 | Normal Node Macro-F1 | +0.35 | [-1.82, +2.72] | 62.6% |
| A4 | Fault Node Macro-F1 | +0.61 | [-2.55, +3.65] | 59.1% |
| A5 | Node accuracy | +0.69 | [-0.46, +1.86] | 82.3% |
| A5 | Node Macro-F1 | +1.10 | [-0.76, +3.01] | 87.2% |
| A5 | Node Macro-Recall | +0.77 | [-0.87, +2.43] | 82.3% |
| A5 | Tier3 Macro-F1 | +1.19 | [-0.85, +3.29] | 86.9% |
| A5 | Normal Node Macro-F1 | +0.21 | [-2.10, +2.55] | 57.4% |
| A5 | Fault Node Macro-F1 | +2.81 | [-0.06, +5.95] | 92.6% |
| A6 | Node accuracy | +0.46 | [-0.46, +1.39] | 77.1% |
| A6 | Node Macro-F1 | +0.34 | [-1.23, +2.09] | 66.2% |
| A6 | Node Macro-Recall | +0.27 | [-1.10, +1.68] | 64.5% |
| A6 | Tier3 Macro-F1 | +0.25 | [-1.49, +2.21] | 62.8% |
| A6 | Normal Node Macro-F1 | +0.20 | [-2.24, +2.69] | 55.6% |
| A6 | Fault Node Macro-F1 | +0.58 | [+0.00, +2.00] | 63.2% |

若 CI 跨 0，应视为本次测试集不足以区分候选与 A0；即便 CI 不跨 0，也仍需完成其余 11 个 fold×seed，才能判断训练稳定性与跨参与者泛化。

## 11. 当前证据对 Phase A 门槛的回答

### 11.1 只针对当前一次运行的方向性检查

| 条件 | Δ总体 Node Macro-F1 pp | Δ最弱 Node Recall pp | ΔFault Node Macro-F1 pp | Macro-F1+最弱Recall同升 | Fault非劣(−0.5 pp) |
| --- | --- | --- | --- | --- | --- |
| A1 | +1.81 | +33.33 | +6.23 | 是 | 是 |
| A2 | +3.70 | +33.33 | +8.02 | 是 | 是 |
| A3 | -0.43 | +8.33 | +2.01 | 否 | 是 |
| A4 | +0.50 | +8.33 | +1.13 | 是 | 是 |
| A5 | +1.09 | +8.33 | +3.53 | 是 | 是 |
| A6 | +0.32 | +0.00 | +0.48 | 否 | 是 |

A1, A2, A4, A5 在这一运行中满足 Macro-F1 与最弱 Recall 同升；A3, A6 不满足。这不是正式通过：正式门槛要求上述方向在 12 个 fold×seed 中至少 7 个成立，并同时检查 Fault 非劣。

### 11.2 完整 Phase A 状态

| 门槛 | 当前状态 | 说明 |
| --- | --- | --- |
| 12 个 fold×seed 中多数正增益 | 未满足/未评估 | 当前只有 A_as_test × seed_1（1/12） |
| Node Macro-F1 与最弱类别 Recall 同时改善 | 可做单次检查 | 总体表给出本次结果；仍需 12 次一致性 |
| Fault 不退化 | 可做单次检查 | Normal/Fault 表给出本次变化；正式阈值为 -0.5 pp 非劣界 |
| 缺失模态/时间偏差仍回退接近 A0 | 部分满足 | A3-A6 的零新增模态回退数值等价已验证；A2 缺第二相机以及失步压力测试仍需检查；sensor-only S1-S12 本身没有 A0 回退路径 |
| 延迟与吞吐满足硬件预算 | 未评估 | 配置中的目标硬件、P95 延迟和最低吞吐预算仍为空 |

## 12. 建议的下一步

1. 先把 A_as_test 的 seed 2、42 补齐，观察本报告中最显著的类别增益是否换 seed 后仍存在；若类别方向反复翻转，暂不扩大到四折。
2. 对 A2/A3 增加第二相机缺失与时间失步测试，并对 A4–A6 运行缺失模态与 ±5%、±10%、±20% 时间偏移压力测试；重点检查总体、Fault、边界相关类别及 A0 回退差距。
3. A3 本次低于 A0 且明显低于 A2。下一步先检查 gate 激活分布、cross-view residual 范数和训练曲线，再补 seed 2/42；不要在 A_as_test 上搜索 gate 超参数或融合权重。
4. S1–S12 表明 IMU 明显强于 EMG、ResNet10 更适合当前 IMU、Dilated 更适合当前 EMG，且历史 M2 普遍优于 Direct Node。若继续融合，应优先尝试 A0 + S3 的严格 A0-anchor gated residual，并保留缺失模态回退。
5. 在扩展到 12 个 fold×seed 前填写目标硬件预算，并分别记录 RGB、EMG/IMU encoder、历史 M2、融合与后处理的端到端延迟。

## 13. 可复核产物

- `analysis/a_as_test_seed_1/node_classwise_deltas_vs_A0.csv`：35 Node 完整类别指标与差值。
- `analysis/a_as_test_seed_1/tier3_classwise_deltas_vs_A0.csv`：31 Tier3 完整类别指标与差值。
- `analysis/a_as_test_seed_1/node_correction_flow_vs_A0.csv`：修正/损害/净正确数。
- `analysis/a_as_test_seed_1/node_rescue_harm_by_true_class.csv`：按真实 Node 的修正与损害计数。
- `analysis/a_as_test_seed_1/node_class_impact_heatmap.png`：报告内嵌的 35 Node Recall/F1 类别影响总览图。
- `analysis/a_as_test_seed_1/node_class_impact_heatmap.svg`：同一图的可无限放大矢量版本。
- `analysis/a_as_test_seed_1/sensor_node_class_impact_heatmap.png`：S1–S8 的 Node Recall/F1 类别影响。
- `analysis/a_as_test_seed_1/sensor_tier3_class_impact_heatmap.png`：S1–S12 的 Tier3 Recall/F1 类别影响。
- `analysis/a_as_test_seed_1/modality_training_complementarity.png`：总体性能、A0 修正/损害和两模型 oracle 互补矩阵。
- `analysis/a_as_test_seed_1/sensor_node_classwise_deltas_vs_A0.csv` / `sensor_tier3_classwise_deltas_vs_A0.csv`：S1–S12 类别级结果。
- `analysis/a_as_test_seed_1/modality_rescue_harm_vs_A0.csv` / `pairwise_oracle_complementarity.csv`：模态互补性原始计数。
- `analysis/a_as_test_seed_1/training_generalization_gap.csv`：A3 与 S1–S12 的训练拟合和测试差距。
- `analysis/a_as_test_seed_1/LOW_RECALL_NODE_SAMPLE_INDEX.md`：每个方法低 Recall 类别的完整样本名，区分正确/错误。
- `analysis/a_as_test_seed_1/low_recall_node_samples.csv`：低 Recall 类别逐样本明细，可按方法、类别、Normal/Fault、Stage、run 筛选。
- `analysis/a_as_test_seed_1/SENSOR_LOW_RECALL_NODE_SAMPLE_INDEX.md` / `sensor_low_recall_node_samples.csv`：S1–S8 低 Recall 类别与样本索引。
- `analysis/a_as_test_seed_1/manual_low_recall_sample_notes.csv`：按方法、真实 Node、样本名保存人工备注，重新生成报告时会保留。
- [S1–S12 低 Recall 多模态质量检查 Pilot](analysis/a_as_test_seed_1/S1_S12_LOW_RECALL_MULTIMODAL_QUALITY_CHECK_PILOT.md)：每个 S 条件试选 1 个误分类样本，提供原始 RGB 逐帧索引，并在完整 run 的右手 EMG/IMU 图中标出 MindRove 与 A0 RGB 边界。
- `analysis/a_as_test_seed_1/paired_bootstrap_exploratory.json`：探索性 bootstrap 原始汇总。
- 复现命令：`python tools/analyze_small_scope_a_as_test.py`。
