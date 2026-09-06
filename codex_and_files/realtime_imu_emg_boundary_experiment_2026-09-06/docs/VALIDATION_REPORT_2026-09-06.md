# 实验包生成与 Smoke 验证报告

日期：2026-09-06

## 已完成核查

- Python 源码通过 `compileall`。
- 6 个单元测试全部通过：UTC timestamp 转换、因果 previous-sample 重采样、past-only 窗口、相邻动作独立边界、pending merge、模型输出 shape。
- IMU 与 EMG 的浅层 setup validation 均通过。
- 数据集识别到 103 个 run：A=24、D=25、J=30、M=24。
- 随包的 32 个 LOSO JSONL 文件通过 train/test、normal/fault/all 一致性检查。
- 使用真实 MindRove CSV 完成两种模态各 8 个 run 的 smoke 缓存提取。
- `run_sample_000001` IMU 缓存 shape 为 `[3151, 50, 14]`：3151 个 20 Hz 决策、50 个过去 IMU 样本、12 个物理通道加 2 个 validity channel。
- 同一 run 的 EMG 缓存 shape 为 `[3151, 100, 18]`：100 个过去 EMG 样本、16 个物理通道加 2 个 validity channel。
- 两种缓存都恢复出 25 个 start 和 25 个 end，与该 run 的动作段数一致。
- IMU 与 EMG 均完成 2 epochs smoke 训练，产生 `best.pth` 和 `last.pth`。
- 两种模态均完成 validation-only online calibration。
- 两种模态均完成 normal/fault/all smoke 评价和单 run 在线回放。
- 汇总工具成功生成 6 行 smoke summary。

## Smoke 产物

```text
cache\imu
cache\emg
outputs_smoke\imu
outputs_smoke\emg
outputs_smoke\smoke_summary.csv
validation\imu\setup_validation.json
validation\emg\setup_validation.json
```

这些产物只证明流程可执行。Smoke 仅使用 heldout A、seed 1、4 个训练 run、每个测试 split 最多 2 个 run和 2 epochs；任何 accuracy、F1 或 modality 差异都不能作为正式实验结论。

## 正式运行前仍需做的检查

1. 在运行机器上执行 `scripts\run_loso_grid.bat both validate`，完成全部 sensor timestamp 列的 deep validation。
2. 检查 EMG 训练数据 PSD 后再决定是否启用 50 Hz notch。
3. 观察完整缓存的 validity fraction 和磁盘规模。
4. 先完成一组正式 heldout/seed/scope 并检查训练曲线，再启动全部 48 个模型。
5. 对传统 IMU change-point 与 EMG onset baseline 建立单独对照；当前版本只实现神经 MVP。
