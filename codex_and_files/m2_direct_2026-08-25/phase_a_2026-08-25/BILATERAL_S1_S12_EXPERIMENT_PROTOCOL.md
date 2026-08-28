# 双手 EMG/IMU S1–S12 实验协议

## 1. 目的与隔离范围

本轮完整复现 S1–S12，但把输入从右手信号改为双手信号。旧实验不修改，也不复用其 checkpoint、feature cache 或完成标记。新输出固定写入：

- `outputs/bilateral_signal_cache/`：每折双手信号与两种测试标准化；
- `outputs/supplementary_bilateral/`：双手 S1–S12 模型、逐样本概率、压力、延迟、bootstrap 和汇总。

四折仍为 A/D/J/M LOSO，seed 仍为 1/2/42，使用相同 `all_runs` train/test manifest。S1–S4 的上游 Tier3 encoder、冻结特征、scratch M2 与 node head 都重新训练；S5–S12 的 encoder/head 也全部从头训练。

## 2. `mindrove.pt` 与长度插值

输入直接读取 map-style dataset 中每个样本的 `mindrove.pt`：

- `left_emg [L_left,8]`、`right_emg [L_right,8]`；
- `left_acc/gyro [L_left,3]`、`right_acc/gyro [L_right,3]`；
- 文件中同时保留 board timestamp 和 annotation metadata，但本轮模型输入预处理不再次使用它们重切片。

`mindrove.pt` 的生成脚本已经使用同一对动作注释起止时间切出左右手数据；这对时间也是 RGB 动作片段的定义边界。左右手 manifest 首末 board timestamp 的轻微差异，只表示两个设备在共同边界内实际落到的首末采样点和样本数量不同，不需要在本轮再次对齐或裁剪。

为与原右手 S1–S12 保持严格的单变量对照，本实验直接对 `mindrove.pt` 中已经切好的完整数组做长度插值。处理顺序为：

1. 完整读取左手和右手 EMG，不删除任何采样点；
2. 左手 EMG 按自身完整序列长度线性插值到512点；
3. 右手 EMG 按自身完整序列长度线性插值到512点；
4. 左右手 Acc XYZ 与 Gyro XYZ 先在各自侧拼为6通道；
5. 左右手 IMU 分别按自身完整序列长度线性插值到256点；
6. 最后只沿 channel 维拼接为 EMG `[16,512]` 和 IMU `[12,256]`。

该流程不读取 `left/right_board_ts` 建立第二个公共时间网格，不取左右手时间交集，不补边界，也不改变 `mindrove.pt` 已有的 RGB/动作片段定义。其插值方式与原右手实验相同，唯一新增变量是左手通道。

## 3. 输入通道

| 模态 | 通道顺序 | 形状 |
|---|---|---|
| 双手 EMG | left EMG 1–8，right EMG 1–8 | `[16,512]` |
| 双手 IMU | left Acc XYZ、left Gyro XYZ、right Acc XYZ、right Gyro XYZ | `[12,256]` |

ResNet10-1D 和 Dilated Conv1D 只改变第一层输入通道数；其余网络宽度、512-D特征、训练目标和 M2 结构保持与右手 S1–S12 一致。

## 4. 训练标准化

训练集采用 `participant_train_only_channel_zscore`：

1. 每个 LOSO fold 只读取该折 `train.jsonl`；
2. 对训练参与者分别计算逐通道 mean/std；
3. 某参与者的训练 clip 只使用该参与者自己的统计量；
4. 左右手 EMG 16通道、IMU 12通道均分别标准化；
5. 统计量在时间偏移增强之前计算；标准差下限为 `1e-6`。

同时保存该折所有训练参与者合并后的 pooled mean/std，供第一种测试协议使用。

## 5. 两种测试协议

每个 held-out participant 默认选择按 run 名排序后的第一个 run 作为无标签 calibration run。可在 `config/bilateral_supplementary_experiments.json` 的 `explicit_runs` 中为 A/D/J/M 指定明确 run；一旦指定便不再自动选择。

校准 run 不参与模型训练、不使用动作标签计算统计量，并从最终计分样本中排除。两种测试协议使用完全相同的其余 test clips，因此预测可以按 `sample_name` 配对。

### pooled_train

- 使用该 fold 全部训练参与者合并后的逐通道 mean/std；
- 不读取 held-out participant 的统计量；
- 表示无需新用户校准的严格 fallback。

### participant_calibrated

- 只使用 held-out participant 的无标签 calibration run 计算逐通道 mean/std；
- 使用该统计量标准化其余 test runs；
- 表示部署前允许一次短暂新用户校准。

注意：由于 calibration run 被排除，本轮双手结果的测试样本数少于旧右手全 test 结果。双手模型间以及两种标准化协议间可严格配对；若要与旧右手 S1–S12 做数值比较，应另将旧模型限制到相同非校准样本集合。

## 6. S1–S12 forward

| ID | 双手输入 | Encoder | 目标与 forward |
|---|---|---|---|
| S1 | EMG 16ch | S9 ResNet10-1D encoder | 冻结 Tier3 encoder 的 512-D current/history → scratch M2 → 35 node |
| S2 | EMG 16ch | S10 Dilated-1D encoder | 冻结 Tier3 encoder 的 512-D current/history → scratch M2 → 35 node |
| S3 | IMU 12ch | S11 ResNet10-1D encoder | 冻结 Tier3 encoder的 512-D current/history → scratch M2 → 35 node |
| S4 | IMU 12ch | S12 Dilated-1D encoder | 冻结 Tier3 encoder的 512-D current/history → scratch M2 → 35 node |
| S5 | EMG 16ch | ResNet10-1D | current clip → 512-D →独立 Linear(512,35) |
| S6 | EMG 16ch | Dilated-1D | current clip → 512-D →独立 Linear(512,35) |
| S7 | IMU 12ch | ResNet10-1D | current clip → 512-D →独立 Linear(512,35) |
| S8 | IMU 12ch | Dilated-1D | current clip → 512-D →独立 Linear(512,35) |
| S9 | EMG 16ch | ResNet10-1D | current clip → 512-D →独立 Linear(512,31 Tier3)；也是 S1 上游 |
| S10 | EMG 16ch | Dilated-1D | current clip → 512-D →独立 Linear(512,31 Tier3)；也是 S2 上游 |
| S11 | IMU 12ch | ResNet10-1D | current clip → 512-D →独立 Linear(512,31 Tier3)；也是 S3 上游 |
| S12 | IMU 12ch | Dilated-1D | current clip → 512-D →独立 Linear(512,31 Tier3)；也是 S4 上游 |

S5–S8 不加载 S9–S12；Direct Node 与 Direct Tier3 始终分开独立训练。S1–S4 不加载 A0 或右手 S 模型的任何权重。

## 7. 自动运行

完整入口：`scripts/run_all_bilateral_s1_s12.ps1`。参数、断点续跑、PlanOnly、逐任务 stdout/stderr 日志、ContinueOnError、压力、延迟、完整网格 bootstrap 与汇总行为和 `run_all_phase_a.ps1` 保持一致。

先在 PowerShell 中生成计划：

```powershell
& .\scripts\run_all_bilateral_s1_s12.ps1 `
  -Python 'C:\path\to\env\python.exe' `
  -Participants A `
  -Seeds 1 `
  -Experiments S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12 `
  -PlanOnly
```

小范围正式运行：

```powershell
& .\scripts\run_all_bilateral_s1_s12.ps1 `
  -Python 'C:\path\to\env\python.exe' `
  -Device cuda `
  -Participants A `
  -Seeds 1 `
  -SkipBootstrap
```

脚本自动完成：运行时检查→双手 cache→Tier3 dependencies→冻结特征→S1–S4 M2→S5–S8 Direct Node→压力→延迟→bootstrap→两种测试协议分别汇总。

## 8. 结果目录

每个模型：`outputs/supplementary_bilateral/S#/PARTICIPANT_as_test/seed_#/`。

- `test_results/pooled_train/`：pooled训练统计量测试；
- `test_results/participant_calibrated/`：新参与者校准测试；
- 两个目录均包含总体、Normal/Fault、Stage、逐类别、前12混淆、逐样本预测和完整概率；
- `summary/pooled_train/` 与 `summary/participant_calibrated/`：四折三seed汇总、paired bootstrap及门槛结果。

日志写入独立的 `logs/bilateral_run_时间戳/`。
