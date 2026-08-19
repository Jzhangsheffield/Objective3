# Atomic-Tail A0–A8 独立实验包（Windows）

本实验包用于系统评估 atomic-tail shuffle augmentation 的改进项。它包含独立实现的训练代码、本地任务图和 LOSO 协议；体积较大的冻结 RGB 特征缓存及已有 M2-Direct 权重从一个共享目录只读复用。运行时不会导入旧实验包的 Python 文件，也不会向共享目录写入。

默认实验网格为：4 个跨人测试折 `A/D/J/M` × 3 个随机种子 `1/2/42` × `all_runs`。所有模型在测试时统一使用真实时间顺序，不包含实时动作边界检测。

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
.\run_experiments.ps1 -Experiments "A0,A4,A5" -DryRun
.\run_experiments.ps1 -Experiments "A0,A4,A5" -ValidateOnly
```

## 2. 选择运行 A0–A8

运行单个实验：

```powershell
.\run_experiments.ps1 -Experiments "A2"
```

运行若干实验、指定折和 seed：

```powershell
.\run_experiments.ps1 `
  -Experiments "A0,A3,A4,A5,A6" `
  -Participants "A,D,J,M" `
  -Seeds "1,2,42" `
  -Scopes "all_runs"
```

也可以从 cmd 使用简化入口：

```bat
run_experiments.bat A0,A2,A3
```

若只选择 A4–A8，调度器默认自动加入 A0。默认配置下 A0 只加载共享 `m2_direct\last.pth` 重新评估，不复制 checkpoint、也不重新训练；A4–A8 直接从同一个共享 checkpoint 热启动。已有 `completed.json` 的任务会跳过。只有明确传入 `-Overwrite` 才会覆盖同一任务的结果文件。

建议分阶段运行，避免一次提交 108 个训练任务：

1. `A0,A1,A2,A3`，先确认 active-tail-only 和 true-recency 的独立收益；
2. `A4,A5,A6`，寻找主要提升；
3. `A7,A8`，确认辅助目标是否继续增益。

## 3. A0–A8 一览

| ID | 主要变化 | 初始化/训练 | 测试顺序 |
|---|---|---|---|
| A0 | M2-Direct 真实历史基线 | 默认复用共享 checkpoint；可切换为从头训练 | actual |
| A1 | 旧版 atomic-tail once | 从头训练 50 epoch | actual |
| A2 | 仅 active tail 样本增强 | 从头训练 50 epoch | actual |
| A3 | A2 + 保留真实 recency position ID | 从头训练 50 epoch | actual |
| A4 | A3 + actual/atomic 成对训练 | A0 热启动，20 epoch + 5 epoch actual 校准 | actual |
| A5 | A4 + 置信度门控预测一致性 | 同 A4 | actual |
| A6 | A5 + 转移概率加权、距离约束采样 | 同 A4 | actual |
| A7 | A6 + valid/corrupted tail 顺序辅助任务 | 同 A4 | actual |
| A8 | A7 + Tier-3 聚合辅助损失 | 同 A4 | actual |

每项的准确定义、公式、对照关系和文献依据见 [EXPERIMENT_CONFIGURATION.md](EXPERIMENT_CONFIGURATION.md)。

## 4. 输出结构

```text
outputs\
  A5\
    all_runs\
      A_as_test\
        seed_1\
          resolved_run_config.json
          augmentation_audit.json
          train_log.json
          last.pth
          test_results_actual_order\
            test_normal_metrics.json
            test_fault_metrics.json
            test_all_metrics.json
            *_predictions.csv
            *_probabilities.pt
          completed.json
```

`augmentation_audit.json` 会记录 active-tail 覆盖率、实际改变顺序的比例、平均 Kendall 距离和 A7 可用样本比例，便于判断增强是否真正生效。

汇总全部完成任务：

```powershell
& "<你的Python路径>" .\tools\summarize_results.py --split test_all
```

输出位于 `outputs\summary`，包含各实验均值/标准差，以及与同折同 seed A0 的 paired delta 和 win/tie/loss。

## 5. 公平实验要求

- A0–A8 必须使用同一 participant、seed、scope 的缓存与协议；
- A0–A3 的差异仅是配置表中声明的增强机制；
- A4–A8 必须从对应 A0 检查点热启动；默认直接读取共享旧 checkpoint；
- 所有测试集只用 actual chronological history；
- 重排函数不接收当前动作标签，也不读取当前样本之后的历史；
- A6 的转移统计只从该 LOSO 外层训练折 manifest 估计；
- 不要依据最终测试折反复调权重。先固定 A/D/M，最后一次报告 J，或增加严格的训练内验证。

## 6. 包内文件

- `config\experiment_config.json`：唯一集中配置文件；
- `atomic_tail_exp\`：独立数据、图、增强、模型、损失、训练和评估实现；
- `assets\`：复制后的任务图和关系矩阵；
- `inputs\`：包内 LOSO 协议；不再保存 `.pt` 特征副本；
- `tools\run_grid.py`：选择并调度 A0–A8；
- `tools\summarize_results.py`：跨折/跨 seed 汇总；
- `tools\audit_without_torch.py`：训练前审核各增强策略，无需 PyTorch；
- `tools\smoke_test.py`：不需要 GPU 的核心增强逻辑测试。

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

如果希望重新训练 A0，将 `training.reuse_shared_a0_checkpoint` 改成 `false`。此时 A0 会在新包输出目录生成自己的 `last.pth`，A4–A8 也会改为读取该本地 A0 权重。

Windows 长路径可能影响非常深的自定义输出目录。默认输出路径已刻意保持较短，不建议再增加多层目录。
