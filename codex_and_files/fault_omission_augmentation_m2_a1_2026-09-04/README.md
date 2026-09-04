# Normal-run omission augmentation：M2 / A1 Once / A1 Every20 六组对照

本实验包使用 normal run 模拟动作遗漏，对照测试这种增强能否改善 Normal、Fault、All 的节点识别性能。

**状态：实验包已构建并验证；正式 72 次、每次 100 epochs 的训练尚未启动。** `outputs_smoke_v2_every20` 只有程序验证结果，不能作为方法性能结果。

## 1. 六组实验

| 实验 ID | 历史策略 | normal-run 遗漏增强 | Epochs |
|---|---|---|---:|
| `M2-Direct-RealOrder-Control` | 真实顺序 | 无 | 100 |
| `M2-Direct-RealOrder-FaultAug` | 删除节点后保持剩余真实顺序 | 有 | 100 |
| `A1-Legacy-Control` | Legacy atomic-tail graph-valid shuffle，Once | 无 | 100 |
| `A1-Legacy-FaultAug` | 先删除节点，再做 legacy shuffle，Once | 有 | 100 |
| `A1-Legacy-Every20-Control` | Legacy shuffle，每20 epochs更新 | 无 | 100 |
| `A1-Legacy-Every20-FaultAug` | 先删除节点，再做 legacy shuffle，每20 epochs更新 | 有 | 100 |

这里 Control 的“无增强”专指**无错误增强**；A1-Control 仍有其原本的 shuffle。六组都是随机初始化后重新训练，不读取旧 M2/A1 的分类器 checkpoint。

默认采用 **ADM 原实验包的 all-runs 跨人划分**，不是 sequence-disjoint 划分：A、D、J、M 四个 fold，seed 1、2、42，共 `6 × 4 × 3 = 72` 次训练。三种历史策略共用同一 Direct Fusion 架构；差别在训练历史构造。

## 2. 错误概率

### 不包含 Stage 3 的 normal run

包括只有 Stage 2、以及 Stage 1＋Stage 2 的 run。E1 和 E3 分别以 0.3 概率独立发生：

- 无错误：49%；
- 仅 E1：21%；
- 仅 E3：21%；
- E1＋E3：9%。

E1 只删除 node 16–22，**不包含检查 node 23**；E3 单独删除 node 23。

### 包含完整 Stage 2＋Stage 3 的 normal run

先选择错误数量，再从 E1、E3–E10 中等概率、不放回选择：

| 错误数 | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| 概率 | 50% | 20% | 20% | 10% |

E2 不生成。真实数据中的 E2 run 和重复节点仍保留。错误方案在每个 epoch 按整条 run 重新抽样；同一 run 内所有样本共享同一方案。删除动作后重新构造历史，被删除节点也不再作为该轮的训练目标。

## 3. 重要约定

- 复用原 ADM `features/retrained_all_runs` 的 512-D RGB 特征，camera=`001484412812`。本包不重训 backbone、不重提特征，也不训练 MindRove 模型。
- 真实训练 fault run 不施加新的遗漏。A1 仍对其执行原有的历史重排逻辑；重复历史触发 legacy fallback。
- A1 Once 固定独立 shuffle 种子；Every20 在第1、21、41、61、81轮使用新的周期种子，替换上一周期重排，不累积重排副本。两者都先重排再添加位置编码，不重置模型权重或位置embedding。
- FaultAug 的错误方案仍每epoch刷新，因此即使在同一个20轮周期内，可用历史变化也可能导致重排结果变化。详见配置说明第5节。
- 三个 FaultAug 组在相同 fold/seed/epoch 使用完全相同的错误方案与保留目标集合。
- 六组在同一 seed 下初始化相同；学习率与优化器配置相同。
- 六组测试均不施加遗漏、不做 shuffle，按真实历史推理。
- 正式测试采用更正后的 23 条 fault 清单；A28、J31、J32、J34 作为 normal。
- 错误组合可能违反正常完整任务图，这是预期的合成 fault，不进行“缺少前置动作即丢弃样本”的过滤。

## 4. 当前电脑如何运行

先进入此实验包目录：

```powershell
Set-Location 'D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\fault_omission_augmentation_m2_a1_2026-09-04'
```

依次执行：

```powershell
# 准备更正后的 train/test 清单；不修改旧实验包
.\run_experiments.ps1 -Action prepare

# 检查24个训练/测试特征缓存
.\run_experiments.ps1 -Action check

# 可选：六组各2 epochs，每epoch最多3 batches，结果只进 outputs_smoke_v2_every20
.\run_experiments.ps1 -Action smoke

# 正式训练全部72个实验，顺序运行
.\run_experiments.ps1 -Action train

# 汇总正式完成的实验，缺失项会在 coverage.json 中明确标记
.\run_experiments.ps1 -Action summarize
```

只运行某个 fold/seed：

```powershell
.\run_experiments.ps1 -Action train -Folds A -Seeds 1
```

只运行某个实验组：

```powershell
.\run_experiments.ps1 -Action train -Groups 'M2-Direct-RealOrder-FaultAug'
```

只运行新增的Every20配对组：

```powershell
.\run_experiments.ps1 -Action train -Groups 'A1-Legacy-Every20-Control','A1-Legacy-Every20-FaultAug'
```

中断后续训：

```powershell
.\run_experiments.ps1 -Action train -Resume
```

已完成且配置一致的实验会跳过；未完成实验必须显式 `-Resume`。默认每10 epochs及最后一轮保存 `last.pth`，续训从最近保存的 checkpoint 继续。配置、模型源代码或输入指纹改变时拒绝混合续训，请使用新的 `output_root`。

脚本默认 Python 为 `C:\Users\digit\anaconda3\envs\Pytorch\python.exe`。可以通过 `-Python` 指定另一环境。若 PowerShell 执行策略阻止 `.ps1`，可以直接使用 Python 入口，不必修改系统执行策略：

```powershell
& 'C:\Users\digit\anaconda3\envs\Pytorch\python.exe' -B .\tools\run_experiments.py train
```

## 5. 换电脑需要什么

1. 拷贝整个实验包，保留 `vendor`、`assets`、`config`、`inputs`。
2. 保留原 ADM 输出中的四 fold `protocols` 和每个 seed 的 `features/retrained_all_runs/{train_all,test_all}.pt`。
3. 修改 `config/experiment_config.json` 的 `paths.adm_root`，指向新电脑的 ADM 包根目录。
4. 通过 `-Python` 指定具备 PyTorch、NumPy 的环境。当前验证环境为 PyTorch `2.6.0+cu126`，CUDA RTX 4090；没有 CUDA 时 `device=auto` 使用 CPU。
5. 先 `check`，再 `smoke`，最后 `train`。

正式训练不依赖旧分类器 checkpoint，也不要求小型 repaired dataset 或原始视频存在。`tools/verify_package.py` 是额外的开发回归检查，需要旧 M2 checkpoint/预测文件及原 atomic-tail 包作为对照；**不是正式训练的前置依赖**。

## 6. 主要文件

| 文件/目录 | 含义 |
|---|---|
| `EXPERIMENT_CONFIGURATION.md` | 完整方法、节点定义、概率、scheduler、统计与注意事项 |
| `config/experiment_config.json` | 可编辑实验配置 |
| `inputs/*_as_test/{train,test}.jsonl` | 保持原样本集合的更正标签清单 |
| `inputs/preparation_report.json` | 每个fold的节点/run数量及重复节点提示 |
| `fault_aug/core.py` | 错误采样、run删除、因果历史、legacy shuffle、LR日程 |
| `fault_aug/runtime.py` | 缓存检查、训练、续训、测试、汇总 |
| `vendor/` | 原 Direct Fusion 和 atomic-tail 的原样代码快照 |
| `verification/` | 单元测试、缓存校验、100epoch抽样审计、旧checkpoint回归检查 |
| `outputs_smoke_v2_every20/` | 最新六组短训练验证，不能用于报告方法性能 |
| `outputs_smoke/` | 保留的旧四组初步验证，不属于正式结果 |
| `outputs/<group>/<fold>/seed_<s>/` | 正式训练后生成的输出 |

每次正式训练输出包括 `resolved_config.json`、`train_log.json`、`last.pth`、逐 epoch `run_masks`、首轮样本历史、Normal/Fault/All 预测与指标。两种汇总口径分别保存：12 fold×seed 的 mean±SD，以及四 fold 合并后3个seed的 mean±SD。

## 7. 已完成的验证

- 19项单元测试通过：概率分布、E1/E3独立性、节点删除、同组run一致性、真实fault保留、位置编码、legacy fallback、Every20边界和scheduler等。
- 全部24个缓存通过样本集合、标签、相机、特征形状及有限值检查。
- 六组都通过小规模训练和实际测试输出检查。
- 三个增强组错误方案一致；六组初始化权重一致。
- 另外对真实A-fold训练历史检查第1–20、21–40、41–60、61–80、81–100轮的重排周期，避免仅靠2轮smoke遗漏Every20边界。
- 12个旧 ADM M2 checkpoint 的 **5,685 条 Node 和 Tier-3 预测全部复现，差异为0**。
- 正式100epoch训练未启动；没有使用smoke精度作效果结论。

详细记录见 `verification/validation_summary.json`。
