# Phase A 一键自动运行说明

## 1. 最简单的完整运行

在 PowerShell 中进入实验包根目录，使用安装了 PyTorch、NumPy、TorchVision 且可以访问 GPU 的 Python：

```powershell
Set-Location -LiteralPath 'D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\m2_direct_single_camera_sensor_fusion_analysis_2026-08-25\phase_a_multimodal_incremental_value_2026-08-25'

.\scripts\run_all_phase_a.ps1 `
  -Python 'C:\path\to\your\python.exe' `
  -Device 'cuda' `
  -NumWorkers 0
```

脚本默认自动完成：

1. PyTorch/CUDA 环境检查；
2. 1,895 个 MindRove 文件的全张量审计；
3. 四折 train-only EMG/IMU normalization 与 cache；
4. 第二相机 12 个 fold×seed backbone 和特征提取；
5. A1 训练与 A2 late fusion；
6. A3-A7 的 60 个 adapter 模型训练；
7. A3-A7 缺失模态与时间偏差压力测试；
8. 代表性缓存特征延迟测试；
9. A1-A7 paired clip bootstrap；
10. 总体、Normal/Fault、Stage、31 Tier3、35 node 和混淆对汇总。

A0 不重新训练；脚本直接读取已有 12 次 A0 结果。

## 2. 自动续跑

`Resume` 默认是 `$true`。重新执行同一命令时，脚本检查每项任务的 checkpoint/cache/`completed.json`，已经完成的任务显示 `[SKIP]`，只继续缺失任务。

```powershell
.\scripts\run_all_phase_a.ps1 -Python 'C:\path\to\python.exe' -Device cuda -Resume $true
```

不要给子工具加 `--overwrite`。已有但不完整的任务会由该项工具明确报错，避免把残留结果误认为完整实验；此时应先检查对应日志，再决定是否移动该任务目录后重跑。

如果只想预览将要执行的命令而不启动训练：

```powershell
.\scripts\run_all_phase_a.ps1 -Python 'C:\path\to\python.exe' -PlanOnly
```

## 3. 先做小规模验证

建议先只运行 A fold、seed 1 和 A1/A2/A4/A5/A6，确认显存、MindRove tensor 和输出格式：

```powershell
.\scripts\run_all_phase_a.ps1 `
  -Python 'C:\path\to\python.exe' `
  -Device cuda `
  -Participants A `
  -Seeds 1 `
  -Conditions A1,A2,A4,A5,A6 `
  -SkipStress `
  -SkipLatency `
  -SkipBootstrap
```

注意：这个子集只用于 pipeline 验证，不能作为 Phase A 科学结论。正式结果必须恢复四折三 seed 和 A1-A7 全条件。

## 4. 常用选项

```text
-Python            PyTorch 训练环境的 python.exe；不要使用不含 torch 的系统 Python
-Device            auto / cpu / cuda / cuda:0 等
-NumWorkers         Windows 建议先用 0；稳定后可尝试 2–4
-Participants       默认 A,D,J,M
-Seeds              默认 1,2,42
-Conditions         默认 A1,A2,A3,A4,A5,A6,A7
-Resume             默认 true，跳过已有完整任务
-SkipTensorAudit    跳过全 MindRove tensor 检查，不建议正式运行使用
-SkipStress         暂时不跑压力测试
-SkipLatency        暂时不跑缓存特征延迟测试
-SkipBootstrap      暂时不跑 paired bootstrap；非完整四折三 seed 时也会自动跳过
-PlanOnly           只生成任务计划和状态 CSV，不运行 Python
-ContinueOnError    单项失败后继续其他独立任务；默认失败即停止
```

## 5. 日志与最终结果

每次启动都会创建独立日志目录：

```text
logs/run_YYYYMMDD_HHMMSS/
  run_status.csv
  00_runtime_check.log
  20_upstream_A_s1.log
  40_A7_M_s42.log
  ...
```

`run_status.csv` 记录每项任务的 `COMPLETED`、`SKIPPED_COMPLETE`、`FAILED` 和耗时。

最终结果位于：

```text
outputs/summary/PHASE_A_RESULTS.md
outputs/summary/condition_summary.csv
outputs/summary/per_stage.csv
outputs/summary/per_31_tier3.csv
outputs/summary/per_35_node.csv
outputs/summary/top_12_confusions.csv
outputs/summary/paired_bootstrap_Ax_vs_A0.json
outputs/summary/incremental_value_gates.json
```

## 6. 运行时间与并行提醒

这个脚本是单机单进程顺序调度，优点是最稳妥、可续跑，但完整 Phase A 很重：包含 12 个第二相机 100-epoch backbone、72 个 A1/A3-A7 模型，以及大量压力测试。若使用 HPC 或多 GPU，应依据 `scripts/phase_a_job_matrix.csv` 按 stage 依赖并行提交；不要同时让两个任务写入同一个 condition/fold/seed 目录。

正式启动前仍应先确认配置中的第二相机 ID，以及目标硬件延迟/吞吐预算。
