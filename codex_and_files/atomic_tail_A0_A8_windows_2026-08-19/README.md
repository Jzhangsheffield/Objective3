# Atomic-Tail DualPos 独立实验包（Windows）

本实验包用于系统评估 atomic-tail shuffle augmentation 的改进项。它包含独立实现的训练代码、本地任务图和 LOSO 协议；体积较大的冻结 RGB 特征缓存及已有 M2-Direct 权重从一个共享目录只读复用。运行时不会导入旧实验包的 Python 文件，也不会向共享目录写入。

默认实验网格为：4 个跨人测试折 `A/D/J/M` × 3 个随机种子 `1/2/42` × `all_runs`。所有模型在测试时统一使用真实时间顺序，不包含实时动作边界检测。

DualPos 对每个历史动作同时编码真实 recency `r` 与 shuffle 位移 `Δ=p−r`。实际输入为 `feature projection + E_true(r) + E_shift(Δ)`；actual/test view 中 `Δ=0`，augmented view 中被移动动作具有非零位移，因此 shuffle 对 single-query attention 真正可见，同时不伪造真实时间。

## 1. 首次使用

只需先修改一个文件：

`config\experiment_config.json`

至少确认：

- `paths.python_executable`：带有 PyTorch 的 Python 可执行文件；
- `paths.input_root`：包内 LOSO 协议目录；
- `paths.shared_artifacts_root`：唯一的共享特征/权重根目录，迁移电脑后通常只需修改这一项；
- `paths.output_root`：所有新结果的唯一输出根目录；
- `training.device`：`auto`、`cuda` 或 `cpu`；
- `training.num_workers`、`batch_size`、各阶段 epoch 数。

在 PowerShell 中先验证路径和任务展开：

```powershell
cd D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\atomic_tail_A0_A8_windows_2026-08-19
& "C:\Users\mes19jz\AppData\Local\miniconda3\envs\pytorch\python.exe" .\tools\test_dualpos_torch.py
& "C:\Users\mes19jz\AppData\Local\miniconda3\envs\pytorch\python.exe" .\tools\test_dualpos_integration.py
.\run_experiments.ps1 -Experiments "A3-DualPos,A4-DualPos" -DryRun
.\run_experiments.ps1 -Experiments "A3-DualPos,A4-DualPos" -ValidateOnly
```

## 2. 选择运行实验

运行单个新实验：

```powershell
.\run_experiments.ps1 -Experiments "A3-DualPos"
```

运行完整的新实验组合：

```powershell
.\run_experiments.ps1 -Experiments "A3-DualPos,A4-DualPos"
```

运行若干实验、指定折和 seed：

```powershell
.\run_experiments.ps1 `
  -Experiments "A3-DualPos,A4-DualPos" `
  -Participants "A,D,J,M" `
  -Seeds "1,2,42" `
  -Scopes "all_runs"
```

也可以从 cmd 使用简化入口：

```bat
run_experiments.bat A3-DualPos,A4-DualPos
```

选择 `A4-DualPos` 时，调度器会自动加入 A0。默认配置下 A0 只加载共享 `m2_direct\last.pth` 重新评估，不复制 checkpoint、也不重新训练；`A4-DualPos` 从相同 participant/seed/scope 的共享 A0 权重热启动。已有 `completed.json` 的任务会跳过。只有明确传入 `-Overwrite` 才会覆盖同一任务的结果文件。

当前默认只安排两个新实验：

1. `A3-DualPos`：从头训练，检验 true recency + shuffle displacement 本身；
2. `A4-DualPos`：A0 热启动，检验 DualPos paired training + calibration 完整方案。

`A3-full-shuffle`、旧 `A4`、A5–A8 均标记为 `deferred`，不会出现在默认运行列表中，但仍保留旧定义和显式选择能力。

## 3. 实验一览

| ID | 主要变化 | 初始化/训练 | 测试顺序 |
|---|---|---|---|
| A0 | M2-Direct 真实历史基线 | 默认复用共享 checkpoint；可切换为从头训练 | actual |
| A1 | 旧版 atomic-tail once | 从头训练 50 epoch | actual |
| A2 | 仅 active tail 样本增强 | 从头训练 50 epoch | actual |
| A3 | A2 + 保留真实 recency position ID | 从头训练 50 epoch | actual |
| A3-full-shuffle | broad shuffle + true recency；对当前 attention 基本排列不变 | **deferred** | actual |
| **A3-DualPos** | A3 + shuffle displacement embedding | 从头训练 50 epoch | actual，shift=0 |
| A4 | 旧 true-recency paired 方案 | **deferred** | actual |
| **A4-DualPos** | true recency + displacement，actual/aug 0.6/0.4 | A0 热启动；2 epoch shift 预热 + 8 epoch 联合微调 + 3 epoch 校准 | actual，shift=0 |
| A5 | A4 + 置信度门控预测一致性 | **暂存，当前不运行** | actual |
| A6 | A5 + 转移概率加权、距离约束采样 | **暂存，当前不运行** | actual |
| A7 | A6 + valid/corrupted tail 顺序辅助任务 | **暂存，当前不运行** | actual |
| A8 | A7 + Tier-3 聚合辅助损失 | **暂存，当前不运行** | actual |

每项的准确定义、公式、对照关系和文献依据见 [EXPERIMENT_CONFIGURATION.md](EXPERIMENT_CONFIGURATION.md)。

## 4. 输出结构

```text
outputs\
  A4-DualPos\
    all_runs\
      A_as_test\
        seed_1\
          resolved_run_config.json
          augmentation_audit.json
          train_log.json
          after_dualpos_shift_warmup.pth
          after_dualpos_mixed_finetune.pth
          after_actual_calibration.pth
          last.pth
          test_results_actual_order\
            test_normal_metrics.json
            test_fault_metrics.json
            test_all_metrics.json
            *_predictions.csv
            *_probabilities.pt
          completed.json
```

`augmentation_audit.json` 除 active-tail 覆盖率和 Kendall 距离外，还会记录 `shifted_history_token_fraction` 与 `mean_absolute_position_shift`，用于确认 DualPos 扰动确实进入模型。`A4-DualPos` 会保存 `after_dualpos_shift_warmup.pth`、`after_dualpos_mixed_finetune.pth`、`after_actual_calibration.pth` 和最终 `last.pth`。

汇总全部完成任务：

```powershell
& "<你的Python路径>" .\tools\summarize_results.py --split test_all
```

输出位于 `outputs\summary`，包含各实验均值/标准差，以及与同折同 seed A0 的 paired delta 和 win/tie/loss。

## 5. 公平实验要求

- A0–A8 必须使用同一 participant、seed、scope 的缓存与协议；
- A3-DualPos 与 A3 都从头训练；二者核心差异是非零 shuffle displacement embedding；
- A4-DualPos 必须从对应 A0 检查点热启动；只有新增 shift embedding 没有共享预训练权重；
- actual/test view 的 displacement 恒为 0，零位移 embedding 固定为零；
- 所有测试集只用 actual chronological history；
- 重排函数不接收当前动作标签，也不读取当前样本之后的历史；
- A6 的转移统计只从该 LOSO 外层训练折 manifest 估计；
- 不要依据最终测试折反复调权重。先固定 A/D/M，最后一次报告 J，或增加严格的训练内验证。

## 6. 包内文件

- `config\experiment_config.json`：唯一集中配置文件；
- `atomic_tail_exp\`：独立数据、图、增强、模型、损失、训练和评估实现；
- `assets\`：复制后的任务图和关系矩阵；
- `inputs\`：包内 LOSO 协议；不再保存 `.pt` 特征副本；
- `tools\run_grid.py`：选择并调度全部实验，包括两个 DualPos 实验；
- `tools\summarize_results.py`：跨折/跨 seed 汇总；
- `tools\audit_without_torch.py`：训练前审核各增强策略，无需 PyTorch；
- `tools\smoke_test.py`：不需要 GPU 的核心增强逻辑测试。
- `tools\test_dualpos_torch.py`：验证零位移兼容、true-only 排列不变性和 displacement 可见性；需要 PyTorch。
- `tools\test_dualpos_integration.py`：读取一组共享特征/A0 权重，验证实际 batch、兼容加载和 shift-only 梯度；需要 PyTorch。

## 7. 依赖与注意事项

依赖见 `requirements.txt`：Python 3.10+、PyTorch 2.1+、NumPy 1.24+。包不重新训练 R3D-18；它通过 `shared_artifacts_root` 读取已有 512-D 冻结特征缓存，因此主要显存消耗来自小型 history 模型。

## 8. 迁移到另一台电脑

特征和旧 M2-Direct 权重只保留一份。迁移时复制：

1. 本 A0–A8 代码包；
2. 一份共享 artifacts 目录，其中保留旧 `outputs\<participant>_as_test\...\features` 和 `history_models\direct_head_fusion` 结构。

然后只修改 `experiment_config.json` 中：

```json
"shared_artifacts_root": "E:\\your_shared_artifacts\\outputs"
```

如果希望重新训练 A0，将 `training.reuse_shared_a0_checkpoint` 改成 `false`。此时 A0 会在新包输出目录生成自己的 `last.pth`，A4-DualPos（以及显式运行的旧 warm-start 实验）会改为读取该本地 A0 权重。

Windows 长路径可能影响非常深的自定义输出目录。默认输出路径已刻意保持较短，不建议再增加多层目录。
