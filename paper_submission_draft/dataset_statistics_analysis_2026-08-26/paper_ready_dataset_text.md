# 可直接用于论文的数据集文字与关键统计

## 核心统计

| 项目 | 数值 |
|---|---:|
| 参与者 | 4（A、D、J、M） |
| 执行运行 | 103（76 正常，27 含故障/偏差） |
| 预分割动作片段 | 1,895（1,441 正常运行，454 故障运行） |
| 流程节点 / Tier-3 动作 | 35 / 31 |
| 裁剪后参考相机帧 | 251,132 |
| 裁剪后连续时长 | 约 139.4 min |
| 动作 / 背景帧 | 94,698 / 156,434 |
| 背景占比 | 62.29% |

## 中文数据集描述

本研究的数据来自一个多传感器热压接流程数据集，共包含 4 名参与者完成的 103 次流程执行，其中 76 次为正常运行，27 次包含记录到的流程偏差。三路同步 RGB 相机以 1280×720 分辨率记录工位，左右两侧 MindRove 设备分别提供 8 通道肌电、三轴加速度和三轴角速度信号。基于人工动作注释，我们构建了两个同源任务视图。对于流程节点识别，103 次执行被划分为 1,895 个预分割动作片段，并标注为三个流程阶段中的 35 个任务图节点；这些节点映射到 31 个 Tier-3 细粒度动作类别。对于连续动作分割，每次执行保留从首个动作前 30 个参考相机帧到末个动作后 30 帧的连续序列，共包含 251,132 个参考相机帧，约 139.4 分钟。逐帧标签中动作占 37.71%，背景占 62.29%。所有实验均应在窗口生成前按参与者执行四折留一评估，以避免同一运行的相邻片段或重叠窗口跨训练集和测试集。

## English dataset paragraph

The dataset comprises 103 thermal-crimping executions performed by four participants (A, D, J, and M), including 76 normal runs and 27 runs containing recorded procedural deviations. The workstation was observed by three RGB cameras at 1280 × 720 resolution and by bilateral MindRove devices; each wearable stream contains eight electromyography channels, a three-axis accelerometer, and a three-axis gyroscope. We organize the same recordings into two task views. For process-node recognition, expert action boundaries define 1,895 clips labelled as 35 task-graph nodes across three process stages and mapped to 31 fine-grained Tier-3 action categories. For temporal action segmentation, each run is retained continuously from 30 reference-camera frames before the first action to 30 frames after the last action. The retained reference view contains 251,132 frames (approximately 139.4 min), of which 94,698 (37.71%) are action frames and 156,434 (62.29%) are background.

## English annotation and evaluation paragraph

We use the action-recognition-boundary v1 segmentation annotations, which preserve the recognition boundaries whenever the actions do not overlap and label every complementary interval as background. Nine overlapping action instances were converted to a single-label sequence by ending the preceding action at one frame before the next action onset, resolving 26 overlapping frames. Segmentation start and end indices refer to the original one-based full-run frame coordinates, whereas the retained frame tables provide both a re-indexed frame_idx and the corresponding original_frame_idx. All recognition and segmentation experiments use four-fold leave-one-subject-out evaluation. Complete participant runs are assigned to a fold before any clip sampling or sliding-window generation.

## Real-time segmentation task distinction

The recognition branch assumes known clip boundaries, whereas the real-time segmentation branch operates causally on continuous sensor streams and jointly predicts the current frame-level action label and temporal boundaries without access to future observations.

## 提交前必须处理

- 修复 7 个 `label.txt` 与 manifest Tier-3 不一致的样本。
- 训练分割模型时区分原始 `start_idx/end_idx`、裁剪后 `frame_idx` 和 `original_frame_idx`。
- 在切窗前按完整参与者/运行建立 LOSO 清单。
- 将现有论文中“不评估 online boundary detection”的表述改为按任务区分。
- 如果实时分割使用多视角或 MindRove，更新“仅使用 camera 001484412812”的全局描述。
