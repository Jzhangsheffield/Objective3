# Phase A 一键自动运行说明

同一个 `scripts/run_all_phase_a.ps1` 现在同时控制原 A1–A7 和补充 S1–S12。未提供 `-SupplementaryExperiments` 时，原有运行行为不变；不会意外启动信号补充实验。

## 1. 最简单的完整运行

在 PowerShell 中进入实验包根目录，使用安装了 PyTorch、NumPy、TorchVision 且可以访问 GPU 的 Python：

```powershell
Set-Location -LiteralPath 'D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\m2_direct_2026-08-25\phase_a_2026-08-25'

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

如果实验包被复制到另一台电脑，必须先打开 `config/phase_a.json`，把以下路径改成该电脑上的真实绝对路径：

```json
{
  "dataset_root": "C:/.../Stage_2_Mapstyle_Dataset",
  "m2_project_root": "D:/.../graph_history_rgb_cross_person_ADM_2026-07-22"
}
```

主控脚本现在会在启动 Python 前检查这两个目录；路径错误时直接给出明确提示。

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

## 4. 运行 A3 与 S1–S12

先预览完整任务依赖，不启动训练：

```powershell
$S = @('S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12')
& .\scripts\run_all_phase_a.ps1 `
  -Python 'C:\path\to\your\python.exe' `
  -Device cuda `
  -Conditions A3 `
  -SupplementaryExperiments $S `
  -PlanOnly
```

确认计划后删除 `-PlanOnly` 即可正式运行。统一脚本按以下依赖顺序执行：

1. runtime、原模型和补充模型 smoke test；
2. 四折右手 signal cache；
3. A3 需要的第二相机 upstream；
4. A3；
5. S9–S12 Direct Tier3；
6. 提取 S9–S12 的冻结 512-D signal feature；
7. S1–S4 scratch sensor-only M2；
8. S5–S8 独立 Direct Node；
9. 压力、延迟、paired bootstrap 和汇总。

只请求一个 M2 条件时会自动处理依赖。例如下面的命令会自动训练 S9、提取其特征，再训练 S1，不必同时写 `S9`：

```powershell
& .\scripts\run_all_phase_a.ps1 `
  -Python 'C:\path\to\your\python.exe' `
  -Device cuda `
  -Participants A `
  -Seeds 1 `
  -Conditions A3 `
  -SupplementaryExperiments S1 `
  -SkipStress -SkipLatency -SkipBootstrap
```

只跑 Direct Node 与 Direct Tier3 时可使用：

```powershell
$S = @('S5','S6','S7','S8','S9','S10','S11','S12')
& .\scripts\run_all_phase_a.ps1 `
  -Python 'C:\path\to\your\python.exe' `
  -Device cuda `
  -Conditions A3 `
  -SupplementaryExperiments $S
```

`-Conditions A3` 只选择原实验 A3；`-SupplementaryExperiments` 只选择 S 系列，两者互不替代。若不想运行任何原 A 条件，可显式传入空数组：`-Conditions @()`。

## 5. 常用选项

```text
-Python            PyTorch 训练环境的 python.exe；不要使用不含 torch 的系统 Python
-Device            auto / cpu / cuda / cuda:0 等
-NumWorkers         Windows 建议先用 0；稳定后可尝试 2–4
-Participants       默认 A,D,J,M
-Seeds              默认 1,2,42
-Conditions         默认 A1,A2,A3,A4,A5,A6,A7
-SupplementaryExperiments  默认空；可选 S1...S12，S1-S4 会自动补足其 Tier3 上游
-Resume             默认 true，跳过已有完整任务
-SkipTensorAudit    跳过全 MindRove tensor 检查，不建议正式运行使用
-SkipStress         暂时不跑压力测试
-SkipLatency        暂时不跑缓存特征延迟测试
-SkipBootstrap      暂时不跑 paired bootstrap；非完整四折三 seed 时也会自动跳过
-PlanOnly           只生成任务计划和状态 CSV，不运行 Python
-ContinueOnError    单项失败后继续其他独立任务；默认失败即停止
```

## 6. 日志与最终结果

每次启动都会创建独立日志目录：

```text
logs/run_YYYYMMDD_HHMMSS/
  run_status.csv
  00_runtime_check.log
  20_upstream_A_s1.log
  40_A7_M_s42.log
  ...
```

每个任务保留三类日志：

```text
TASK_NAME.log          # 合并后的命令、exit code、stdout、stderr
TASK_NAME.stdout.txt   # Python 原始标准输出
TASK_NAME.stderr.txt   # Python 原始 traceback/标准错误
```

即使 Python 失败，完整 traceback 也会保留在 `.stderr.txt` 和合并后的 `.log`，不会再被 PowerShell 的 `NativeCommandError` 截断。

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
outputs/supplementary/summary/SUPPLEMENTARY_RESULTS.md
outputs/supplementary/summary/condition_summary.csv
outputs/supplementary/summary/per_stage.csv
outputs/supplementary/summary/per_31_tier3.csv
outputs/supplementary/summary/per_35_node.csv
outputs/supplementary/summary/top_12_confusions.csv
outputs/supplementary/summary/low_recall_misclassified_samples.csv
outputs/supplementary/summary/paired_fold_seed_deltas.csv
outputs/supplementary/summary/incremental_value_gates.json
outputs/supplementary/summary/paired_bootstrap_Sx_vs_Sy.json
```

S9–S12 是 Direct Tier3 模型，没有 35-node 输出；汇总中的 node 会显示为不适用。详细 forward、条件对应和计划比较见 `SUPPLEMENTARY_EXPERIMENT_PROTOCOL.md`。

## 7. 运行时间与并行提醒

这个脚本是单机单进程顺序调度，优点是最稳妥、可续跑，但完整 Phase A 很重：包含 12 个第二相机 100-epoch backbone、72 个 A1/A3-A7 模型，以及大量压力测试。若使用 HPC 或多 GPU，应依据 `scripts/phase_a_job_matrix.csv` 按 stage 依赖并行提交；不要同时让两个任务写入同一个 condition/fold/seed 目录。

正式启动前仍应先确认配置中的第二相机 ID，以及目标硬件延迟/吞吐预算。
