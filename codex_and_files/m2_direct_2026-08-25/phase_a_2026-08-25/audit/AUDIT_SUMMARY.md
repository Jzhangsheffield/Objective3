# Phase A 启动前审计结论（2026-08-25）

## 结论

元数据、文件完整性、四折 LOSO 隔离和现有 A0 产物均通过。Phase A 可以进入准备阶段，但正式 GPU 训练前仍有两项必须确认、两项必须补做：第二相机选择、目标硬件预算、全 MindRove 张量数值审计、跨设备时间同步证据。

## 已确认

- 数据共 1,895 个唯一 clip：A=431、D=462、J=555、M=447。
- 三路相机 `001484412812`、`001528512812`、`001431512812` 及 `mindrove.pt` 均为 0 缺失。
- 右手通道元数据一致：EMG=8、accelerometer=3、gyroscope=3；每个 clip 的三种右手信号长度完全一致。
- 右手信号长度 min/median/max 为 154/662/10875；长短差异很大，Phase A 会重采样，实时阶段不能沿用整 clip 时间拉伸。
- 三相机 clip 帧数差的 median=1、max=13，说明 clip 级大体对齐，但它不能替代 frame timestamp 审计。
- 由右手 board start/end timestamp 反推的有效采样率 median=501.70 Hz；9 个 clip 落在 480–520 Hz 之外。其中 `sample_000450` 约为 398.84 Hz，是需要重点复核的时间戳/丢包异常；其余 8 个约为 523.73–550.74 Hz。
- 四个 fold 均为严格 LOSO，train/test sample overlap=0；原数据 fold manifest 与现有 A0 `all_runs` protocol 的 sample 集完全相同。
- A0 的 12 个 fold×seed 均完整：checkpoint、主相机 train/test feature cache、总体/Normal/Fault 指标、逐 clip 预测和 node 概率均存在。
- 当前 A0 12-run 等权均值：总体 Node accuracy=90.57%、Node Macro-F1=87.81%；Fault Node accuracy=89.75%、Fault Node Macro-F1=86.34%。
- 第二相机四折三 seed backbone/feature cache 全部尚缺；实验包已将其列为 upstream 必跑阶段。

## 非错误但需注意

- 全局 `3_camera_mindrove_manifest.jsonl` 不含 `node_idx` 和 `stage_id`。四折 `train/test_manifest.jsonl` 与现有 protocol 已包含这些字段，因此实验包只以 enriched fold protocol 作为标签源。
- 当前默认第二相机 `001528512812`，因为本地已有与该视角有关的历史 J-fold RGB 实验痕迹；这不足以证明它一定优于 `001431512812`。正式开跑前需固定选择。
- 当前 Codex Python 没有 PyTorch，无法读取 `.pt` 的真实数值。`dataset_audit.json` 因此明确标记为 `METADATA_PASS_TENSOR_AUDIT_PENDING`，不能视为全张量审计完成。
- Phase A 的长度重采样会弱化采样率异常对输入 shape 的影响，但不能据此忽略异常；实时流必须按 timestamp 重采样并显式处理丢包/抖动。

## 训练前必须做

1. 在实际 PyTorch 训练环境运行 `python tools/audit_dataset.py --load-tensors`，要求 1,895 个文件的 `right_emg/right_acc/right_gyro` key、shape、有限值全部通过。
2. 确认第二相机 ID；若改为 `001431512812`，只在任何 A1-A7 任务开始前改 `config/phase_a.json`。
3. 提供目标硬件名称、batch=1 的端到端 p95 latency 和最低吞吐预算；否则延迟门槛只能显示 `UNSET`。
4. 实时融合前补充相机 frame timestamp 或同步日志，用于估计 RGB-MindRove 固定 offset、漂移和抖动。

详细机器可读审计见 `dataset_audit.json`。
