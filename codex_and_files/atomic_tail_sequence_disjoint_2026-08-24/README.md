# Atomic-tail sequence-disjoint 实验包

## 1. 实验目的

本包用于检验以下假设：

> 当原始训练 run 中没有出现测试 run 的完整真实 Node 顺序时，Atomic-tail graph-valid shuffle augmentation 是否能够通过合成新的合法历史顺序，提高跨 participant 动作识别性能，并超过只使用真实历史顺序的 M2-Direct fusion。

这里的“顺序隔离”只约束原始观测数据：与测试 run 具有完全相同 Node 顺序的训练 run 会被整体删除。A1-Legacy/A3-DualPos 的 graph-valid augmenter 可以自然生成与测试顺序相同或相近的历史顺序；程序不会拒绝这种情况，因为扩展顺序覆盖正是 augmentation 的目的。

严格禁止的是测试引导采样：训练代码不会读取测试顺序、不会根据测试顺序选择 candidate，也不会为了命中测试顺序而重采样。测试顺序覆盖只在训练协议固定后进行事后审计。

## 2. 与旧实验代码的关系

本包经过对以下三个旧实验包的代码与配置进行对照后建立：

- `graph_history_rgb_experiments_2026-07-20`：最初的 M0–M6、history dataset 和 graph-valid ordering 实现；
- `graph_history_rgb_cross_person_ADM_2026-07-22`：严格 A/D/J/M LOSO、M2-Direct、Dynamic shuffle、Atomic-tail 训练与评估；
- `atomic_tail_A0_A8_windows_2026-08-19`：A0、A1-Legacy、A3-DualPos、DualPos model、deterministic refresh 和 actual-order 测试实现。

本包不复制旧核心模型代码。视觉阶段只读调用 `graph_history_rgb_cross_person_ADM_2026-07-22` 中的 backbone 训练和特征提取入口；history 阶段只读复用 `atomic_tail_A0_A8_windows_2026-08-19/atomic_tail_exp`。新增部分包括：

1. run-level sequence-disjoint protocol；
2. 每个 fold×seed 在过滤后训练集上从头重训 R3D-18 backbone；
3. 使用新 backbone 重新提取过滤后 train 和完整 test_all 特征；
4. M2、A1-Legacy 与 A3-DualPos 五配置矩阵；
5. once/every-10-replace 调度；
6. 输入验证、augmentation 覆盖审计和 normal/fault/all 汇总。

## 3. 一个重要的实现事实

M2、A1-Legacy 与普通 A3 使用相同的 single-query Direct fusion 网络：当前 clip 是 query，历史 token 提供 context。真正的差异来自 history view 与 position semantics。

A1-Legacy 先重排 history，再按照重排后的 presented order 分配 position ID，因此动作特征与位置编码的配对发生变化。A3 则让动作携带真实 recency；在 single-query attention 下，单纯改变 token 排列可能近似不可见，所以本轮不再把普通 A3 放入主矩阵。A3-DualPos 保留真实 recency，同时通过 shift embedding 显式编码重排位移。

旧 augmenter 针对“每个当前 clip 之前的 history prefix”独立重排，并不是一次生成一条完整、内部一致的新 run。`once` 表示每个训练 sample 的增强 history 在全程固定；`every-10-replace` 表示这些 sample-level histories 每10个 epoch 重新确定性采样并替换旧 view，但模型、position embedding 和 optimizer state 均继续保留。

## 4. 主实验矩阵

| 实验 ID | 旧配置来源 | 训练历史 | Position | Refresh |
|---|---|---|---|---|
| `M2-Direct-RealOrder` | A0 | 真实时间顺序 | actual/presented 等价 | 无 shuffle |
| `A1-Legacy-Once` | A1 | Legacy atomic-tail；无 active tail 时 broad graph-valid | 重排后的 presented position | 全程固定一次 |
| `A1-Legacy-Every10-Replace` | A1 | 同上 | 重排后的 presented position | 每10 epoch替换 view |
| `A3-DualPos-Once` | A3-DualPos | active-tail graph-valid | true recency + displacement | 全程固定一次 |
| `A3-DualPos-Every10` | A3-DualPos | active-tail graph-valid | true recency + displacement | 每10 epoch |

M2 始终使用真实训练顺序，不存在 M2-once 或 M2-every10。

所有配置：

- 外层 folds：A、D、J、M；
- seeds：1、2、42；
- scope：`all_runs`；
- camera：`001484412812`；
- backbone：每个 fold×seed 使用过滤后的 all-runs train manifest 从头训练 R3D-18，100 epochs、batch size 16、AdamW、LR 1e-4；
- features：从新 backbone 的 last checkpoint 重新提取过滤后 train 与完整 test_all 的512-D特征；
- history：五个模型共用同一 fold×seed 新特征，50 epochs，batch size 64，AdamW，LR 1e-3，weight decay 1e-4；
- 测试：`test_normal`、`test_fault`、`test_all`；
- 测试 history：始终为 actual chronological order；
- checkpoint：last epoch；
- 不复用旧 backbone checkpoint、旧 feature cache 或旧 M2 checkpoint。

完整配置见 [EXPERIMENT_CONFIGURATION.md](EXPERIMENT_CONFIGURATION.md) 和 `config/experiment_config.json`。

## 5. Sequence-disjoint protocol

对每个 outer fold：

1. 按 `(participant, run)` 分组；
2. 按 `annotation_row_index` 排序；
3. 使用完整 `node_idx` 序列作为 run signature；
4. 默认保留重复 Node，不压缩连续重复；
5. 如果训练 run signature 与任一 `test_all` run signature 完全相同，删除整个训练 run；
6. `test_normal`、`test_fault`、`test_all` 保持原样；
7. 重新验证过滤后的训练 run 与测试 run 精确 Node 顺序交集为零。

Tier-3 sequence 只用于审计，不作为当前删除条件。每个 fold 会输出：

- `sequence_disjoint_report.json`；
- `run_sequence_index.csv`；
- 过滤后的 `all_runs/train.jsonl`；
- 未改变的三个测试 manifest。

本包创建时已经生成并验证四个 fold，实际过滤规模如下：

| 测试 fold | 原训练 runs | 删除 normal/fault | 保留 normal/fault | 保留 runs | 保留 clips |
|---|---:|---:|---:|---:|---:|
| A | 79 | 41 / 11 | 20 / 7 | 27 | 623 |
| D | 78 | 32 / 8 | 23 / 15 | 38 | 869 |
| J | 73 | 35 / 5 | 20 / 13 | 33 | 698 |
| M | 79 | 37 / 10 | 20 / 12 | 32 | 681 |

四个 fold 过滤后的完整 Node 顺序重叠均为0，并且仍保留全部35个 Node和31个 Tier-3 类别。删除比例较高，说明标准流程顺序在 participants 之间重复很多；因此结果必须与本包重新训练的 M2 比较，不能使用旧 M2 数值作为直接基线。

## 6. 推荐运行顺序

以下命令均在本包根目录执行。可直接使用 `run_experiments.ps1`，也可以逐个调用 `tools` 中的脚本。

### 6.1 生成 sequence-disjoint manifests

```powershell
.\run_experiments.ps1 -Action Prepare -Python python
```

若需要重建已有 manifests：

```powershell
.\run_experiments.ps1 -Action Prepare -Python python -Overwrite
```

### 6.2 验证输入、路径和顺序隔离

```powershell
.\run_experiments.ps1 -Action Validate -Python python
```

验证包含：

- 五个实验定义是否一致；
- M2 是否保持 actual/no-shuffle；
- 旧 backbone/feature/history 代码、task graph、relation matrix 是否存在；
- 过滤后训练与测试完整 Node 顺序是否仍有交集；
- 过滤后是否缺失 Node 或 Tier-3 类别；
- 新 backbone checkpoint 和 feature cache 是否已经生成。

第一次验证时，新 upstream artifacts 尚不存在会显示 warning，这是预期状态；顺序隔离或代码路径错误才会作为 error。

### 6.3 重训 backbone 并重新提取特征

先查看12个 fold×seed upstream jobs：

```powershell
.\run_experiments.ps1 -Action UpstreamDryRun -Python C:\path\to\pytorch\python.exe -DatasetRoot C:\path\to\Stage_2_Mapstyle_Dataset
```

正式运行：

```powershell
.\run_experiments.ps1 -Action UpstreamRun -Python C:\path\to\pytorch\python.exe -DatasetRoot C:\path\to\Stage_2_Mapstyle_Dataset
```

每个 fold×seed 会依次执行：

1. 使用过滤后 `all_runs/train.jsonl` 从头训练100 epochs R3D-18；
2. 在 normal/fault/all 上评估 Tier-3 backbone；
3. 从该新 checkpoint 提取过滤后 train 特征；
4. 从同一 checkpoint 提取完整 test_all 特征。

因此共训练12个独立 backbone，并生成12对 train/test feature caches。产物只写入 `outputs/upstream/`。

### 6.4 查看并运行五个 history 模型

在 upstream 完成后，先执行一个真实 forward smoke test：

```powershell
.\run_experiments.ps1 -Action Smoke -Python C:\path\to\pytorch\python.exe
```

它会加载 A fold、seed 1 的新 feature cache，分别执行 actual/augmented forward，但不会训练或写 checkpoint。

查看60个 history jobs：

```powershell
.\run_experiments.ps1 -Action HistoryDryRun -Python C:\path\to\pytorch\python.exe
```

默认是5 models × 4 folds × 3 seeds = 60 jobs。即使 upstream 尚未完成，dry-run 也会列出计划并明确标记缺失的新 feature caches。

也可以只查看部分任务：

```powershell
python .\tools\run_grid.py --dry-run --experiments A1-Legacy-Once,A1-Legacy-Every10-Replace --participants A,J --seeds 1
```

正式运行 history grid：

```powershell
.\run_experiments.ps1 -Action HistoryRun -Python C:\path\to\pytorch\python.exe
```

建议先运行单个 smoke job：

```powershell
python .\tools\run_grid.py --experiments M2-Direct-RealOrder --participants A --seeds 1
```

确认完成后再运行完整 grid。已有 `completed.json` 的 job 默认跳过。

也可以连续运行 upstream 和 history 两个阶段：

```powershell
.\run_experiments.ps1 -Action Full -Python C:\path\to\pytorch\python.exe -DatasetRoot C:\path\to\Stage_2_Mapstyle_Dataset
```

### 6.5 事后审计 augmentation 是否扩展测试 history-prefix 覆盖

```powershell
.\run_experiments.ps1 -Action Coverage -Python python
```

该步骤分别按 A1-Legacy 和 A3-DualPos 的真实 `active_tail_only` 配置、seed 和 refresh-round 规则重建 shuffled histories，再与测试 history prefixes 比较。它不会影响训练，也不会把测试顺序反馈给 sampler。这里比较的是 sample-level history prefix，不是完整合成 run。

本包创建时的 deterministic audit 得到以下“augmentation 新增覆盖测试 prefix 数量”：

| Fold | A1 Once | A1 Every10 | DualPos Once | DualPos Every10 |
|---|---:|---:|---:|---:|
| A | 3 | 7 | 1 | 1 |
| D | 5 | 9 | 0 | 1 |
| J | 3 | 7 | 1 | 2 |
| M | 5 | 11 | 0 | 2 |

A1 的 changed fraction 约52%–60%，DualPos/A3 active-tail-only 约28%–35%。这说明 Legacy broad fallback 确实产生更多新顺序；但 coverage audit 只是机制证据，最终性能仍必须由五模型配对实验决定。

### 6.6 汇总结果

```powershell
.\run_experiments.ps1 -Action Summarize -Python python
```

汇总文件位于 `outputs/summary/`：

- `fold_seed_metrics.csv`：每个 fold×seed×split 的完整指标；
- `aggregate_metrics.csv`：normal/fault/all 的均值与 SD；
- `paired_deltas_vs_M2.csv`：每个增强模型相对 M2 的配对差值；
- `summary.json`：机器可读完整汇总。

## 7. 输出目录

```text
outputs/
  upstream/
    A_as_test/cam_001484412812/seed_1/
      backbone/all_runs/last.pth
      features/retrained_all_runs/train_all.pt
      features/retrained_all_runs/test_all.pt
  history_models/
    M2-Direct-RealOrder/
    A1-Legacy-Once/
    A1-Legacy-Every10-Replace/
    A3-DualPos-Once/
    A3-DualPos-Every10/
      all_runs/A_as_test/seed_1/
        resolved_run_config.json
        augmentation_audit.json
        train_log.json
        last.pth
        test_results_actual_order/
        completed.json
  summary/
```

## 8. 主要比较

必须预先关注以下配对比较：

1. `A1-Legacy-Once − M2-Direct-RealOrder`；
2. `A1-Legacy-Every10-Replace − M2-Direct-RealOrder`；
3. `A1-Legacy-Every10-Replace − A1-Legacy-Once`；
4. `A3-DualPos-Once − M2-Direct-RealOrder`；
5. `A3-DualPos-Every10 − M2-Direct-RealOrder`；
6. `A3-DualPos-Every10 − A3-DualPos-Once`。

主要 endpoint 建议固定为 `test_all Node accuracy`。同时必须报告 normal、fault、all 的 Node/Tier-3 accuracy、macro-F1、balanced accuracy 和 support。

如果 A1-Legacy 高于 M2，可以支持“先 graph-valid 重排、再赋 presented position 的 Legacy augmentation 在原始顺序未见条件下更有效”。DualPos 是另一种显式表示重排的方案，应分别与 M2及自身 once/every-10 配对比较。

## 9. 解释限制

- 本实验保证原始完整 run 顺序不重叠，不保证所有局部 prefix 或 transition 都不重叠；这些局部覆盖会在 audit 中报告。
- 删除重叠 run 会改变训练数据量，因此新结果只能在同一新 protocol 内公平比较，不能把新 M2 数值直接与旧 M2 90.57% 当作同一条件比较。
- A1-Legacy 设置 `active_tail_only=false`：有 active tail 时固定 tail 并重排其余历史；无 active tail 时仍执行 broad graph-valid shuffle。它与 active-tail-only 的 A2/A3 不同。
- A1-Legacy 将 presented position 重新分配给重排后的动作；A3-DualPos 则同时保留 true recency 并编码 displacement，两者检验的机制不同。
- `once` 和 `every-10` 的 augmentation audit 必须结合实际 changed fraction 和唯一 history 数解释，而不能只按名称推断增强强度。

## 10. 安全与可复现性

- 新包不会修改三个旧实验包；
- 旧 task graph 与经过验证的训练入口只读复用，但 backbone checkpoint 和 feature caches 在本包中重新生成；
- 新训练结果只写入本包 `outputs/`；
- manifest 生成默认拒绝覆盖；
- 所有训练配置会复制到每个输出的 `resolved_run_config.json`；
- deterministic seed 与旧 A0–A8 实现保持一致；
- 测试始终使用真实时间历史，不对测试集做 augmentation。
