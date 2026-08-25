# Phase A：M2-Direct 多视角/右手 EMG/IMU 增量价值实验包

本实验包把现有 `cam 001484412812 + M2-Direct + all_runs` 固定为 A0，不重新挑 seed、不改变四折 LOSO manifest，也不使用 graph-valid shuffle。目标是判断新增信息是否在 12 个 fold×seed 上稳定改善 35-node 识别，而不是追逐单次最好结果。

## 当前状态

- 数据元信息审计已完成：1,895 个 clip，A/D/J/M 分别为 431/462/555/447；三路相机文件和 `mindrove.pt` 均无缺失。
- A0 的 12 个 fold×seed 均有总体/Normal/Fault 指标、逐样本预测和 35-node 概率，可直接复用。
- 第二相机的四折三 seed backbone/512-D 特征尚不存在，必须先运行 upstream 阶段。
- 默认第二相机暂定为 `001528512812`；`001431512812` 数据同样完整，可在正式启动前只改配置一次。
- 本包不包含已跑完的 A1-A7 数值结果；它提供可复现训练、评估、压力测试、paired bootstrap 和汇总工具。GPU 任务完成后自动生成最终结果表。

## 快速入口

如果要在一台 Windows 训练机上按依赖顺序自动运行全部实验，请先阅读 [AUTOMATION_GUIDE.md](AUTOMATION_GUIDE.md)，然后使用 `scripts/run_all_phase_a.ps1`。脚本支持完成标记检测、自动续跑、逐任务日志和最终汇总。

1. 阅读 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)，尤其是 A0-A7 forward、统计单位和停止/通过规则。
2. 在训练环境执行全张量审计：

   `python tools/audit_dataset.py --load-tensors`

3. 对四折各构建一次右手信号 cache：

   `python tools/build_signal_cache.py --participant A`

4. 对每个 fold×seed 训练并提取第二相机特征：

   `python tools/prepare_secondary_camera.py --participant A --seed 1 --execute`

5. 运行 A1、A3-A7；A2 必须在相同 fold×seed 的 A1 完成后运行：

   `python tools/train_condition.py --condition A7 --participant A --seed 1 --device cuda`

   `python tools/evaluate_a2_late_fusion.py --participant A --seed 1`

6. 对 A3-A7 运行压力和缓存特征范围延迟测试，再执行 bootstrap 与汇总：

   `python tools/run_stress_tests.py --condition A7 --participant A --seed 1 --device cuda`

   `python tools/paired_bootstrap.py --condition A7`

   `python tools/summarize_phase_a.py`

## 目录

- `config/phase_a.json`：唯一预注册配置入口。
- `phase_a/`：数据 cache、模型、训练、评估和指标代码。
- `tools/`：可单独重跑的命令。
- `scripts/`：任务矩阵生成与批量运行入口。
- `audit/`：数据审计结果。
- `outputs/`：upstream、A1-A7、压力、延迟、bootstrap 与总表。

## 重要约束

- 不允许用 test 集选择第二相机、融合权重、epoch 或 gate 超参数。
- A2 固定 `0.5/0.5` 融合 35-node 概率；Tier3 概率由 node 概率按固定 node→Tier3 映射求和。
- EMG/IMU 只读取右手：EMG 8 通道；IMU 为 right_acc(3) + right_gyro(3)。
- A3-A7 从对应 A0 精确加载并冻结 M2/history/node head，只训练新增 adapter；全部新增模态缺失时必须数值等价于 A0。
- 信号均值/标准差只由每折 train manifest 计算，并复用于该折三个 seed 与全部 test 子集。
- `test_normal` 和 `test_fault` 只做最终分层报告，不参与训练或选择。
- 当前延迟工具仅覆盖“缓存 RGB 特征以后”的融合/M2/head；最终实时预算必须再加入视频解码、采样和一/两个 RGB backbone。
