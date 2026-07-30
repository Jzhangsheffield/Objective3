# Direct Head Fusion补充实验

日期：2026-07-28  
状态：新增独立实验阶段；不修改、不覆盖原M0–M6结果

## 1. 研究问题

原M1–M3使用冻结M0的35-node logits作为基线，只学习history delta：

```text
logits = frozen_m0_logits + history_scale × history_delta
```

本阶段检验另一种机制：

> 使用Tier3预训练backbone产生的冻结RGB特征，将当前clip与同run历史先融合，再由可训练的
> 35-node分类头直接产生最终logits。

新模型不加载M0 checkpoint，不预测logit delta，也不让35-node head额外经历“M0 50 epoch +
history 50 epoch”两阶段训练。fusion和随机初始化的35-node head在一个50-epoch阶段内联合训练。

## 2. 严格保持不变的实验逻辑

- 相机：`001484412812`；
- held-out participant：A、D、J、M；
- seed：1、2、42；
- train scope：`normal_only`和`all_runs`；
- 严格LOSO，不使用held-out participant训练或选择模型；
- normal-only/all-runs分别复用各自对应的Tier3 backbone feature cache；
- 无validation、无early stopping、无best checkpoint；
- 使用最后一个epoch的`last.pth`；
- 测试`test_normal`、`test_fault`、`test_all`；
- 输出35-node指标，并将35-node概率聚合为31类Tier3指标；
- M3继续使用相同seed下的确定性graph-valid历史重排；
- 旧checkpoint、prediction、metrics、summary和完成标记不作任何修改。

## 3. 新模型

| 模型 | History | Position | 输出 |
|---|---|---|---|
| `m1_direct` | actual order | 无 | fused feature → 35-node head |
| `m2_direct` | actual order | 有 | fused feature → 35-node head |
| `m3_direct` | graph-valid order | 有 | fused feature → 35-node head |

设当前Tier3视觉特征为`x ∈ R^512`，历史attention context为`c ∈ R^256`：

```text
c = Attention(Project(x), Project(history))
x_fused = Linear([x; c]) ∈ R^512
logits = NodeClassifier(x_fused) ∈ R^35
```

融合层初始化为：

```text
W_fusion = [I_512, 0]
b_fusion = 0
```

因此初始`x_fused == x`，训练开始时模型等价于在冻结Tier3特征上训练35-node head，随后才学习
history对特征空间的修正。最终logits完全由可训练node head生成。

## 4. 代码入口

```text
graph_history/models.py
tools/train_direct_history_model.py
tools/smoke_test_direct_models.py
tools/summarize_direct_head_fusion.py
```

原`tools/train_history_model.py`和原M1–M3模型构造没有改动。

## 5. 输入依赖

每个participant、seed和scope必须先具有原严格实验产生的：

```text
normal_only:
  features/retrained_normal_only/completed.json
  features/retrained_normal_only/train_all.pt
  features/retrained_normal_only/test_all.pt

all_runs:
  features/retrained_all_runs/completed.json
  features/retrained_all_runs/train_all.pt
  features/retrained_all_runs/test_all.pt
```

feature cache中的metadata保存实际Tier3 checkpoint路径；该metadata还会被写入新模型checkpoint和
`experiment_config.json`，用于确认视觉表示来源。新阶段不需要M0 checkpoint。

A/D/M的seed 1 normal-only缓存来自较早的完整实验，包含`train_all.pt`、`test_all.pt`及各自
metadata，但没有后来新增的`completed.json`。新入口会对这三个组合显式验证两个`.pt`缓存后复用，
不会因此重新训练backbone或重新提取特征；其余组合仍优先要求完成标记。

## 6. Windows使用

```bat
cd /d D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\graph_history_rgb_cross_person_ADM_2026-07-22
set PYTHON_BIN=C:\Users\digit\anaconda3\envs\Pytorch\python.exe
```

### 单折单seed，两个scope

```bat
set TEST_PARTICIPANT=A
set SEED=1
call bat\run_direct_head_fusion_one_fold.bat
```

### 单折单seed，只运行一个scope

```bat
call bat\31_train_direct_head_fusion_normal_only.bat
call bat\32_train_direct_head_fusion_all_runs.bat
```

### 完整四折三seed

```bat
call bat\run_direct_head_fusion_ADJM.bat
```

完整入口按A、D、J、M和seed 1、2、42顺序运行两个scope，最后执行严格配对汇总。

## 7. HPC/Slurm使用

### 单折单seed

```bash
bash slurm/submit_direct_head_fusion_one_fold.sh A 1 both
```

第三个参数也可使用`normal_only`或`all_runs`。

### 完整四折三seed

```bash
bash slurm/submit_direct_head_fusion_ADJM.sh
```

该入口提交24个array jobs：

```text
4 participants × 3 seeds × 2 scopes
```

每个array job含3个task，对应M1/M2/M3 Direct。全部训练成功后，脚本自动提交
`35_summarize_direct_head_fusion_ADJM_3seeds.slurm`。

## 8. 手工运行单个模型

下面以A、seed 1、all-runs、M3 Direct为例：

```bat
"%PYTHON_BIN%" tools\train_direct_history_model.py ^
  --model m3_direct ^
  --train-scope all_runs ^
  --protocol-root "outputs\A_as_test\cam_001484412812\protocols" ^
  --train-cache "outputs\A_as_test\cam_001484412812\seed_1\features\retrained_all_runs\train_all.pt" ^
  --test-cache "outputs\A_as_test\cam_001484412812\seed_1\features\retrained_all_runs\test_all.pt" ^
  --task-graph "assets\integrated_task_graph_latest.json" ^
  --relation-matrix "assets\integrated_feature_history_matrix.json" ^
  --output-root "outputs\A_as_test\cam_001484412812\seed_1\history_models\direct_head_fusion" ^
  --epochs 50 ^
  --batch-size 64 ^
  --num-workers 8 ^
  --seed 1
```

## 9. 输出和防覆盖

```text
outputs\<P>_as_test\cam_001484412812\seed_<S>\
history_models\direct_head_fusion\<scope>\<model>\
├── last.pth
├── train_log.json
├── experiment_config.json
├── learned_parameters.json
├── test_results\
│   ├── test_normal_metrics.json
│   ├── test_normal_predictions.csv
│   ├── test_fault_metrics.json
│   ├── test_fault_predictions.csv
│   ├── test_all_metrics.json
│   └── test_all_predictions.csv
└── completed.json
```

所有标准入口：

- 完成标记存在时安全跳过；
- 不向旧`retrained_normal_only`或`retrained_all_runs`写入；
- 不传`--overwrite`；
- 非空但不完整的新实验目录会触发停止，要求人工检查。

## 10. 汇总

```bat
call bat\33_summarize_direct_head_fusion_ADJM_3seeds.bat
```

或：

```bash
sbatch slurm/35_summarize_direct_head_fusion_ADJM_3seeds.slurm
```

完整网格预期：

```text
4 participants × 3 seeds × 2 scopes × 3 models × 3 splits = 216 direct rows
```

汇总输出：

```text
outputs/direct_head_fusion_summary_ADJM_3seeds/
├── direct_head_metrics.csv
├── direct_head_paired_deltas.csv
├── direct_head_aggregate.csv
└── completed.json
```

配对差值包括：

```text
m1_direct − m0
m1_direct − m1
m2_direct − m0
m2_direct − m2
m3_direct − m0
m3_direct − m3
```

每个差值严格匹配participant、seed、train scope和test split。总体结果先在participant内部平均
三个seed，再对四位participant等权汇总。
