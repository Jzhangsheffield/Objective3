# IMU/EMG 实时动作边界实验协议

## 研究问题

本实验分别回答：仅使用双腕 IMU、仅使用双腕 EMG，能否在连续数据流中因果检测动作开始和结束。第一版不做 IMU+EMG 融合，也不把真实动作边界、未来传感器样本或 held-out participant 统计量输入模型。

## 固定条件

- 数据集：103 个连续 run。
- 标注：`action_recognition_boundaries_with_background_v1`，包含短 background。
- 参与者：A、D、J、M。
- LOSO held-out：A、D、J、M。
- 随机种子：1、2、42。
- 训练范围：`normal_only` 和 `all_runs`。
- 测试集：`test_normal`、`test_fault`、`test_all`。
- 决策频率：20 Hz，即每 50 ms 一次预测。

训练/验证按 run 划分。不能把同一个 run 的窗口随机分到训练与验证。滤波参数由配置固定，归一化均值和标准差只能由当前 LOSO 的训练 run 计算。阈值如需调参，只能使用训练参与者的 validation runs。

## IMU 条件

- 原始通道：左右手各 3 轴 accelerometer 和 3 轴 gyroscope，共 12 个物理通道。
- 同步：按 `board_ts`，使用 previous-sample hold；这不会读取目标时刻之后的样本。
- 处理频率：100 Hz。
- 因果低通：40 Hz，4 阶。
- 局部窗口：过去 0.5 s，共 50 个样本。
- 额外输入：左右手各一个 validity channel，总输入 14 通道。

## EMG 条件

- 原始通道：左右手各 8 路 EMG，共 16 个物理通道。
- 同步：按 `board_ts`，使用 previous-sample hold。
- 处理频率：500 Hz。
- 因果带通：20–200 Hz，4 阶。由于 Nyquist 频率约为 250 Hz，不能使用 20–500 Hz。
- 50 Hz notch 默认关闭；应在训练数据 PSD 显示稳定工频峰后才启用。
- 局部窗口：过去 0.2 s，共 100 个样本。
- 额外输入：左右手各一个 validity channel，总输入 18 通道。

## 模型

每个过去窗口先由局部 1D CNN 编码成 128 维特征。随后使用五层 dilation 为 1、2、4、8、16 的 causal TCN。输出三个任务：

1. 当前状态：background/action；
2. action start；
3. action end。

边界标签在 20 Hz 网格中使用 ±100 ms 膨胀。动作相邻、没有 background 时，仍根据 segmentation row 保留前一动作的 end 与后一动作的 start。

## 在线解码

状态机为：

```text
BACKGROUND → START_CANDIDATE → ACTION → END_CANDIDATE → BACKGROUND
```

默认 start/end debounce 为 2 steps（100 ms），最短动作 6 steps（300 ms），短间隔合并阈值 2 steps（100 ms）。结束片段先放进 pending buffer；只有超过 merge gap 才正式发出，因此合并不是事后修改已经发出的结果。

## 评价

- Boundary Precision/Recall/F1：±50、±100、±200、±500 ms；
- start/end 的 signed median、absolute median 和 P90 error；
- Segmental F1@10、@25、@50；
- Edit Score；
- frame state accuracy、precision、recall、F1；
- predicted segments/minute；
- 小于 200 ms 的 fragment fraction；
- emission delay；
- compute time 和 real-time factor。

IMU 与 EMG 必须使用同一 fold、seed、scope、测试 split 配对比较。完成单模态结果后，才进入 IMU+EMG 融合及与 RGB M3 Atomic-tail 的端到端 node 实验。
