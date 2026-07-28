# RGB Task-Graph History 严格四折三Seed实验结果分析

首次报告日期：2026-07-24
完整更新日期：2026-07-27
相机：`001484412812`
任务：35-node分类，并将35-node概率聚合为31类Tier3结果
主要实验设计：A/D/J/M严格四折LOSO，`seed_1`、`seed_2`、`seed_42`

## 1. 执行摘要

本次更新纳入了新完成的A/D/J/M四折、三seed、normal-only与all-runs完整结果，解决了旧报告中
normal-only seed数量不足和J折backbone来源不一致两个关键缺口。

当前最重要的结果如下：

1. **严格结果网格完整。**
   共包含`4 participants × 3 seeds × 2 train scopes × 10 models × 3 splits = 720`
   条模型级结果；normal-only与all-runs严格配对差值为360条，没有缺折、缺seed、缺模型或缺split。

2. **历史信息稳定提高35-node分类。**
   在完整all-runs的`test_all`上，M3的Node Accuracy为`84.74%`，M0为`69.81%`，
   平均提高`14.94`个百分点；12/12个participant-seed配对全部提高。

3. **M3是all-runs下最强的综合模型。**
   M3在`test_all`同时获得最高Node Accuracy、Node Macro-F1、Tier3 Accuracy和Tier3 Macro-F1：
   `84.74 / 83.89 / 85.63 / 84.58`。

4. **收益主要来自流程node消歧，而不是简单Tier3外观识别。**
   all-runs下M3相对M0的Stage 2 Node Accuracy提高`18.88`个百分点，而Stage 2 Tier3 Accuracy
   只提高`1.76`个百分点。四组重复动作node之间的双向误判下降`84.7%–98.4%`。

5. **all-runs训练明显改善完整pipeline。**
   在相同participant、seed、model和split的严格配对中，M3的all-runs相对normal-only在
   `test_all`提高`5.44`个百分点Node Accuracy和`4.65`个百分点Tier3 Accuracy；
   四位participant的三seed均值全部为正。

6. **精确实际顺序不是获得历史收益的必要条件。**
   在all-runs下，graph-valid重排的M3相对真实顺序M2提高`0.59`个百分点Node Accuracy和
   `0.64`个百分点Tier3 Accuracy；优势不大，但分别在9/12配对中为正。

7. **relation bias有效果，但仍不是主要性能来源。**
   M5 oracle relation和M6 soft relation相对M4均有小幅平均提升，但幅度通常小于1个百分点，
   且没有稳定超过M3。当前证据更支持将M3作为主模型，将M4/M5/M6作为relation消融。

8. **run级结果支持同一结论。**
   将每个participant-run先在三个seed上平均后，all-runs M3相对M0的Node Accuracy在
   103/103个测试run上全部提高；run等权平均提升`15.78`个百分点。

这些结果支持以下核心解释：

> 历史和task graph信息的主要价值，是把视觉上相同或相似的动作定位到正确的流程node。
> 它对35-node流程状态识别的帮助远大于对31类Tier3外观分类的帮助。

## 2. 分析范围与结果完整性

### 2.1 本次使用的结果

本报告的正式数值均来自以下实验包：

```text
D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\
graph_history_rgb_cross_person_ADM_2026-07-22\outputs
```

纳入：

- held-out participant：A、D、J、M；
- seed：1、2、42；
- train scope：`normal_only`和`all_runs`；
- 模型：M0–M6及3个E2E对照；
- split：`test_normal`、`test_fault`、`test_all`；
- overall、per-stage、prediction及严格training-scope delta结果。

旧J先导实验包不再进入四折均值。它仅用于历史对照，以判断早期定性结论是否在严格J折中复现。

### 2.2 完整性检查

| 结果文件 | 行数 | 预期内容 |
|---|---:|---|
| `all_model_metrics.csv` | 720 | 4人 × 3seed × 2scope × 10模型 × 3split |
| `all_model_training_scope_deltas.csv` | 360 | 4人 × 3seed × 10模型 × 3split |
| `all_model_cross_person_aggregate.csv` | 60 | 2scope × 10模型 × 3split |
| `all_model_training_scope_delta_aggregate.csv` | 30 | 10模型 × 3split |
| `all_model_per_stage_metrics.csv` | 2160 | 720个结果 × 3stage |

本次检查确认：

- participant完整：A、D、J、M；
- seed完整：1、2、42；
- representation scope与train scope严格匹配；
- 每个node模型都具有Node和Tier3指标；
- 直接Tier3模型没有虚构Node指标；
- 每个实验都有normal、fault和all三个测试split。

### 2.3 严格LOSO设置

每一折只使用另外三位参与者训练，held-out participant不进入训练或validation。所有backbone均：

- 从scratch训练；
- 不使用validation或early stopping；
- 不根据held-out participant选择checkpoint；
- 使用最后一个epoch的`last.pth`；
- normal-only与all-runs分别训练独立backbone和下游模型。

因此，本报告中的A/D/J/M可以合并为严格一致的四折结果。

## 3. 模型与评估定义

| 模型 | 输入与图信息 | 定位 |
|---|---|---|
| M0 | 仅当前clip的冻结RGB特征 | 无历史baseline |
| M1 | 当前clip + 同run历史，无位置编码 | basic history attention |
| M2 | 实际发生顺序历史 + 位置编码 | actual-order history |
| M3 | task graph允许顺序的确定性重排历史 + 位置编码 | graph-valid order |
| M4 | candidate history attention，无relation bias | no-relation消融 |
| M5 | 真实历史node标签产生relation bias | oracle上限，不可部署 |
| M6 | 冻结M0历史node概率产生soft relation bias | 可部署soft graph |
| E2E-Tier3-Scratch | 直接从RGB预测31类Tier3 | Tier3视频baseline |
| E2E-Node-Scratch | 从scratch直接预测35-node | Node视频baseline |
| E2E-Node-From-Tier3 | Tier3 backbone初始化后预测35-node | transfer baseline |

主要指标：

- Node Accuracy / Macro-F1 / Balanced Accuracy；
- 聚合后的Tier3 Accuracy / Macro-F1 / Balanced Accuracy；
- per-stage结果；
- 同participant-seed内的配对差值；
- run级等权描述性结果。

## 4. 测试数据规模与统计口径

### 4.1 测试规模

| Held-out participant | test normal | test fault | test all | fault node覆盖 | fault Tier3覆盖 |
|---|---:|---:|---:|---:|---:|
| A | 294 | 137 | 431 | 35/35 | 31/31 |
| D | 400 | 62 | 462 | 30/35 | 26/31 |
| J | 387 | 168 | 555 | 33/35 | 29/31 |
| M | 360 | 87 | 447 | 34/35 | 30/31 |
| 合计 | 1441 | 454 | 1895 | — | — |

四位participant共有103个测试run，其中76个normal run、27个fault run。

D的fault split只有62个clip，并缺失5个node和5个Tier3类别。不同participant的fault Macro-F1
不是在完全相同的类别集合上计算，因此fault结论必须同时参考accuracy、macro-F1、样本数和类别覆盖。

### 4.2 报告中的聚合方式

正式总体表采用：

1. 先在每位participant内部平均三个seed；
2. 再对四位participant等权求均值；
3. “±”为四位participant均值之间的样本标准差。

这样不会因为J有555个clip而给J更大权重，也不会把12个participant-seed组合错误地当作12位独立受试者。

配对计数如“12/12”为描述性稳定性指标，不等于12个独立统计样本。当前真正的外层独立单位只有
4位participant，因此本报告不做普通clip级t-test，也不宣称统计显著性。

## 5. Strict normal-only结果

### 5.1 test_all四折三seed结果

| 模型 | Node Acc | Node Macro-F1 | Tier3 Acc | Tier3 Macro-F1 |
|---|---:|---:|---:|---:|
| M0 | 66.76 ± 3.86 | 68.38 ± 2.90 | 79.33 ± 5.65 | 76.21 ± 3.05 |
| **M1** | 78.59 ± 5.53 | **78.71 ± 3.13** | 81.01 ± 4.99 | **80.84 ± 2.60** |
| M2 | 79.30 ± 7.70 | 78.47 ± 5.65 | 81.08 ± 6.70 | 80.03 ± 4.34 |
| M3 | 79.30 ± 6.96 | 78.48 ± 5.26 | 80.99 ± 6.00 | 79.92 ± 4.23 |
| M4 | 79.32 ± 5.74 | 78.01 ± 4.22 | 81.25 ± 5.21 | 79.31 ± 3.66 |
| M5 | 79.63 ± 6.65 | 78.24 ± 5.06 | 81.24 ± 6.42 | 79.39 ± 4.40 |
| **M6** | **79.76 ± 6.38** | 78.34 ± 4.63 | 81.39 ± 5.84 | 79.49 ± 3.71 |
| E2E-Node-Scratch | 71.77 ± 4.73 | 72.44 ± 3.30 | 78.84 ± 5.99 | 77.07 ± 2.75 |
| E2E-Node-From-Tier3 | 74.88 ± 4.88 | 74.52 ± 3.73 | **81.82 ± 5.51** | 78.67 ± 3.62 |
| E2E-Tier3-Scratch | — | — | 81.22 ± 4.39 | 78.07 ± 2.40 |

normal-only的主要现象：

1. 所有历史模型都比M0高约11.8–13.0个百分点Node Accuracy；
2. M1–M6的总体Node Accuracy非常接近，最大差距只有1.17个百分点；
3. M6获得最高Node Accuracy，M1获得最高Node Macro-F1和Tier3 Macro-F1；
4. E2E-Node-From-Tier3获得最高Tier3 Accuracy，但Node Accuracy明显低于所有M1–M6；
5. normal-only下没有足够证据把M3定义为绝对最优，因为M2–M6在不同指标上互有胜负。

Balanced Accuracy提供相同方向：M1的Node/Tier3 Balanced Accuracy为`80.25/82.47`，是
normal-only `test_all`的最高值。

### 5.2 三个split中的最佳模型

| Split | 最佳Node Acc | 最佳Node Macro-F1 | 最佳Tier3 Acc | 最佳Tier3 Macro-F1 |
|---|---|---|---|---|
| normal | M1, 81.63 | M1, 82.75 | M1, 83.67 | M1, 84.89 |
| fault | M4, 77.86 | M4, 74.02 | E2E-Node-From-Tier3, 81.88 | M4, 75.68 |
| all | M6, 79.76 | M1, 78.71 | E2E-Node-From-Tier3, 81.82 | M1, 80.84 |

M1在normal-only的正常流程上表现很强，但没有保持fault上的同等优势。这说明不带位置和graph
关系的history attention容易学习正常流程中的序列模式，却不一定能稳定迁移到异常顺序。

## 6. 完整all-runs结果

### 6.1 test_all四折三seed结果

| 模型 | Node Acc | Node Macro-F1 | Tier3 Acc | Tier3 Macro-F1 |
|---|---:|---:|---:|---:|
| M0 | 69.81 ± 3.94 | 72.91 ± 3.51 | 83.32 ± 5.66 | 81.41 ± 3.98 |
| M1 | 79.67 ± 7.72 | 80.93 ± 5.85 | 84.85 ± 5.44 | 84.37 ± 3.86 |
| M2 | 84.15 ± 5.95 | 83.06 ± 4.61 | 84.99 ± 5.67 | 83.67 ± 3.87 |
| **M3** | **84.74 ± 5.81** | **83.89 ± 4.16** | **85.63 ± 5.08** | **84.58 ± 3.25** |
| M4 | 83.01 ± 5.51 | 81.74 ± 4.25 | 84.58 ± 5.24 | 82.76 ± 3.54 |
| M5 | 83.78 ± 6.12 | 83.19 ± 3.75 | 84.98 ± 5.36 | 84.14 ± 2.71 |
| M6 | 83.71 ± 6.96 | 82.65 ± 5.59 | 85.05 ± 5.76 | 83.53 ± 4.50 |
| E2E-Node-Scratch | 74.72 ± 6.68 | 75.09 ± 6.63 | 81.92 ± 7.62 | 79.65 ± 6.50 |
| E2E-Node-From-Tier3 | 77.74 ± 5.02 | 78.56 ± 4.45 | 85.46 ± 5.01 | 82.85 ± 4.10 |
| E2E-Tier3-Scratch | — | — | 84.92 ± 4.95 | 82.52 ± 4.03 |

M3同时获得：

- 最高Node Accuracy：84.74%；
- 最高Node Macro-F1：83.89%；
- 最高Tier3 Accuracy：85.63%；
- 最高Tier3 Macro-F1：84.58%；
- 最高Node Balanced Accuracy：84.82%；
- 最高Tier3 Balanced Accuracy：85.65%。

因此，在完整all-runs条件下，将M3作为当前主模型是有数据依据的。

### 6.2 三个split中的最佳模型

| Split | 最佳Node Acc | 最佳Node Macro-F1 | 最佳Tier3 Acc | 最佳Tier3 Macro-F1 |
|---|---|---|---|---|
| normal | M3, 85.62 | M3, 84.74 | M3, 86.52 | M3, 85.44 |
| fault | M3, 84.31 | M3, 81.37 | E2E-Node-From-Tier3, 86.59 | E2E-Node-From-Tier3, 81.55 |
| all | M3, 84.74 | M3, 83.89 | M3, 85.63 | M3, 84.58 |

fault split仍有例外：E2E-Node-From-Tier3的Tier3 Accuracy比M3高1.70个百分点，Tier3
Macro-F1高0.02个百分点。但M3的Node Accuracy仍高5.06个百分点，说明两者解决的问题不同：

- E2E transfer更擅长fault clip的动作外观类别；
- M3更擅长将动作放到正确的35-node流程位置。

## 7. 历史模型相对M0的严格配对收益

### 7.1 test_all结果

下表对相同participant、seed和scope直接计算“模型 − M0”。正向次数的分母为12。

| Scope | 模型 | Δ Node Acc | Node正向 | Δ Node Macro-F1 | Δ Tier3 Acc | Tier3正向 | Δ Tier3 Macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| normal-only | M1 | +11.83 | 12/12 | +10.32 | +1.68 | 10/12 | +4.62 |
| normal-only | M2 | +12.55 | 12/12 | +10.09 | +1.75 | 8/12 | +3.81 |
| normal-only | M3 | +12.54 | 12/12 | +10.10 | +1.65 | 9/12 | +3.71 |
| normal-only | M4 | +12.56 | 12/12 | +9.63 | +1.92 | 9/12 | +3.09 |
| normal-only | M5 | +12.87 | 12/12 | +9.86 | +1.90 | 10/12 | +3.18 |
| normal-only | M6 | **+13.00** | 12/12 | +9.96 | **+2.06** | 10/12 | +3.28 |
| all-runs | M1 | +9.87 | 11/12 | +8.02 | +1.54 | 10/12 | +2.96 |
| all-runs | M2 | +14.35 | 12/12 | +10.15 | +1.68 | 10/12 | +2.26 |
| all-runs | **M3** | **+14.94** | **12/12** | **+10.98** | **+2.32** | **12/12** | **+3.17** |
| all-runs | M4 | +13.20 | 12/12 | +8.83 | +1.27 | 11/12 | +1.35 |
| all-runs | M5 | +13.97 | 12/12 | +10.29 | +1.66 | 12/12 | +2.73 |
| all-runs | M6 | +13.91 | 12/12 | +9.74 | +1.73 | 11/12 | +2.12 |

最稳健的结论是：

- 历史信息对35-node分类的帮助极其稳定；
- normal-only和all-runs中，M1–M6的Node Accuracy均大幅超过M0；
- all-runs M3是唯一同时达到12/12 Node和12/12 Tier3正向的配置；
- Tier3提升明显小于Node提升，再次说明历史主要解决流程位置消歧。

### 7.2 M3相对M0的participant一致性

| Scope | Participant | M0 Node | M3 Node | Δ Node | M0 Tier3 | M3 Tier3 | Δ Tier3 |
|---|---|---:|---:|---:|---:|---:|---:|
| normal-only | A | 63.96 | 76.10 | +12.14 | 76.10 | 78.42 | +2.32 |
| normal-only | D | 64.07 | 75.47 | +11.40 | 77.85 | 77.99 | +0.14 |
| normal-only | J | 72.19 | 89.73 | +17.54 | 87.69 | 89.97 | +2.28 |
| normal-only | M | 66.82 | 75.91 | +9.10 | 75.69 | 77.55 | +1.86 |
| all-runs | A | 65.27 | 79.27 | +14.00 | 78.19 | 80.74 | +2.55 |
| all-runs | D | 69.55 | 81.53 | +11.98 | 81.17 | 83.26 | +2.09 |
| all-runs | J | 74.89 | 92.49 | +17.60 | 91.35 | 92.55 | +1.20 |
| all-runs | M | 69.50 | 85.68 | +16.18 | 82.55 | 85.98 | +3.43 |

四位participant在两种scope下的M3 Node增益全部为正，范围为`+9.10`至`+17.60`个百分点。

## 8. all-runs与normal-only严格训练范围比较

### 8.1 test_all配对差值

下表严格使用相同participant、seed、model和split，计算：

```text
完整all-runs pipeline − 完整normal-only pipeline
```

“±”为四位participant各自三seed平均差值之间的标准差。

| 模型 | Δ Node Acc | Δ Tier3 Acc | Node正向seed-fold | Tier3正向seed-fold |
|---|---:|---:|---:|---:|
| M0 | +3.05 ± 1.75 | +3.98 ± 2.03 | 9/12 | 11/12 |
| M1 | +1.08 ± 3.82 | +3.84 ± 3.08 | 8/12 | 11/12 |
| M2 | +4.85 ± 2.39 | +3.91 ± 2.22 | 9/12 | 10/12 |
| **M3** | **+5.44 ± 3.24** | **+4.65 ± 2.85** | **10/12** | **10/12** |
| M4 | +3.69 ± 3.30 | +3.34 ± 2.66 | 10/12 | 10/12 |
| M5 | +4.15 ± 1.58 | +3.74 ± 2.41 | 11/12 | **12/12** |
| M6 | +3.95 ± 3.34 | +3.66 ± 2.99 | 10/12 | 10/12 |
| E2E-Node-Scratch | +2.96 ± 2.01 | +3.08 ± 1.97 | **12/12** | **12/12** |
| E2E-Node-From-Tier3 | +2.86 ± 1.25 | +3.65 ± 1.03 | 11/12 | 11/12 |
| E2E-Tier3-Scratch | — | +3.71 ± 2.49 | — | 11/12 |

all-runs训练不仅改善history模型，也改善M0和三个E2E对照，说明收益的一部分来自更充分的视觉
训练分布，而不是仅来自history attention。

M3的平均提升最大：

- Node Accuracy：+5.44；
- Node Macro-F1：+5.41；
- Tier3 Accuracy：+4.65；
- Tier3 Macro-F1：+4.66。

### 8.2 normal与fault上的差异

| 模型 | Normal Δ Node | Normal Δ Tier3 | Fault Δ Node | Fault Δ Tier3 |
|---|---:|---:|---:|---:|
| M0 | +3.23 | +3.94 | +1.80 | +3.38 |
| M1 | **-1.18** | +2.12 | **+8.43** | **+10.15** |
| M2 | +4.04 | +3.27 | +6.62 | +4.91 |
| **M3** | **+4.84** | **+4.15** | **+7.33** | **+6.20** |
| M4 | +3.31 | +3.00 | +3.91 | +3.70 |
| M5 | +3.27 | +3.18 | +6.18 | +4.76 |
| M6 | +3.50 | +3.43 | +5.35 | +4.01 |

主要解释：

1. all-runs对fault的改善通常大于对normal的改善；
2. M2、M3、M4、M5和M6在normal与fault上均为正；
3. M1是唯一出现明确权衡的history模型：normal Node下降1.18，但fault Node提高8.43；
4. M3在normal与fault上都获得较大改善，是更平衡的配置；
5. all-runs不是简单“只拟合fault测试”，因为M3的normal Node和Tier3也分别提高4.84和4.15。

### 8.3 M3训练范围提升的participant一致性

| Participant | Δ Node Acc | Δ Node Macro-F1 | Δ Tier3 Acc | Δ Tier3 Macro-F1 |
|---|---:|---:|---:|---:|
| A | +3.17 | +2.33 | +2.32 | +1.34 |
| D | +6.06 | +6.54 | +5.27 | +6.40 |
| J | +2.76 | +3.65 | +2.58 | +3.62 |
| M | +9.77 | +9.11 | +8.43 | +7.26 |

四位participant的平均差值全部为正。M的受益最大，A和J较小，但仍保持正向。

### 8.4 run级训练范围比较

将每个participant-run的三个seed先平均，再对run等权：

| Split | Run数 | M3 Δ Node均值 | 中位数 | Node正向run | M3 Δ Tier3均值 | Tier3正向run |
|---|---:|---:|---:|---:|---:|---:|
| normal | 76 | +5.11 | +4.08 | 54/76 | +4.46 | 53/76 |
| fault | 27 | +6.46 | +7.14 | 21/27 | +5.68 | 20/27 |
| all | 103 | +5.47 | +4.76 | 75/103 | +4.78 | 73/103 |

run级结果与participant级结果方向一致，但run并非跨participant完全独立，仍应视为描述性证据。

## 9. 顺序与graph relation消融

### 9.1 M2与M3：实际顺序和graph-valid重排

| Scope | 比较 | Δ Node Acc | Node正向 | Δ Node Macro-F1 | Δ Tier3 Acc | Tier3正向 | Δ Tier3 Macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| normal-only | M3 − M2 | -0.00 | 6/12 | +0.01 | -0.10 | 5/12 | -0.11 |
| all-runs | M3 − M2 | **+0.59** | **9/12** | +0.83 | **+0.64** | **9/12** | +0.91 |

normal-only下M2和M3几乎完全相同；all-runs下M3有小幅、较一致的优势。因此可以得出：

- 模型不需要严格复现历史动作的实际精确顺序才能获得收益；
- task graph允许的相对顺序足以保留有用历史结构；
- 当前数据支持“graph-valid order具有轻微正则化作用”，但不支持宣称其优势非常大；
- 该结果符合多人协作场景：历史动作可能由不同人完成，准确个人执行顺序不一定是关键。

### 9.2 M1与M2：位置编码的作用

| Scope | M2 − M1 Node Acc | Node正向 | Tier3 Acc | Tier3 Macro-F1 |
|---|---:|---:|---:|---:|
| normal-only | +0.72 | 6/12 | +0.07 | -0.81 |
| all-runs | **+4.48** | **12/12** | +0.14 | -0.70 |

all-runs下位置编码在12/12配对中提高Node Accuracy，却几乎不改变Tier3 Accuracy。
这进一步证明位置/顺序主要用于回答：

```text
“当前视觉动作位于task graph的哪个node？”
```

而不是回答：

```text
“当前clip看起来是哪一种Tier3动作？”
```

normal-only下这一优势很弱，说明当训练历史只覆盖标准正常流程时，M1也能从高度规律的历史中获得
较好结果；加入fault run后，显式位置结构变得更重要。

### 9.3 M4、M5与M6：relation bias

| Scope | 比较 | Δ Node Acc | Node正向 | Δ Tier3 Acc | Tier3正向 | Δ Tier3 Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| normal-only | M5 − M4 | +0.31 | 6/12 | -0.01 | 5/12 | +0.09 |
| normal-only | M6 − M4 | +0.44 | 10/12 | +0.14 | 6/12 | +0.19 |
| all-runs | M5 − M4 | +0.77 | 6/12 | +0.39 | 6/12 | +1.38 |
| all-runs | M6 − M4 | +0.70 | 8/12 | +0.46 | 8/12 | +0.77 |
| all-runs | M3 − M6 | +1.03 | 8/12 | +0.59 | 8/12 | +1.05 |

修正后的结论是：

1. oracle和soft relation在平均值上都优于M4，说明relation信息有价值；
2. 提升幅度小于1个百分点，远小于“加入历史”本身相对M0的约13–15个百分点提升；
3. M5 oracle并未在多数Node/Tier3配对中稳定击败M4，不能把它解释成强oracle上限；
4. M6相对M4的方向更一致，但仍低于M3；
5. M6在normal-only平均Node Accuracy最高，说明soft relation并非无效，只是尚未成为all-runs最佳模型。

因此，论文中更稳妥的层级是：

```text
主要贡献：历史信息 + graph-valid位置结构
次要增益：relation bias
待改进模块：从M0概率构造的soft relation
```

## 10. Stage分析

### 10.1 all-runs test_all分阶段结果

| 模型 | Stage 1 Node | Stage 2 Node | Stage 3 Node | Stage 2 Tier3 |
|---|---:|---:|---:|---:|
| M0 | 81.57 | 65.73 | 80.28 | 84.07 |
| M1 | 83.66 | 77.56 | **86.43** | 84.67 |
| M2 | 83.91 | 84.18 | 83.64 | 85.32 |
| **M3** | **84.72** | **84.61** | 84.86 | **85.83** |
| M4 | 82.17 | 83.18 | 82.19 | 85.33 |
| M5 | 83.96 | 83.36 | 84.91 | 85.01 |
| M6 | 84.07 | 83.71 | 82.60 | 85.57 |

M3相对M0：

| Split | Stage 1 Δ Node | Stage 2 Δ Node | Stage 3 Δ Node | Stage 2 Δ Tier3 |
|---|---:|---:|---:|---:|
| normal | +3.66 | **+18.92** | +4.74 | +1.81 |
| fault | +1.14 | **+19.85** | +4.12 | +2.14 |
| all | +3.16 | **+18.88** | +4.58 | +1.76 |

Stage 2在normal和fault中都获得约19个百分点Node提升，而Tier3提升仅约2个百分点。
这不是由某一个测试划分独占，而是任务结构本身导致的稳定现象。

### 10.2 重复动作node混淆

聚合四位participant、三个all-runs seed的`test_all` prediction，比较M0与M3：

| 相同Tier3动作对应node | M0双向误判 | M3双向误判 | 降幅 |
|---|---:|---:|---:|
| node 14 ↔ 21：`put sample under electrodes` | 166 | 6 | **96.4%** |
| node 15 ↔ 22：`press pedal` | 247 | 4 | **98.4%** |
| node 16 ↔ 19：`put sample on machine table` | 206 | 13 | **93.7%** |
| node 17 ↔ 20：`grip sample from machine table` | 150 | 23 | **84.7%** |

最大的单node recall提升：

| Node | 动作 | M0 Recall | M3 Recall | 提升 |
|---|---|---:|---:|---:|
| 15 | press pedal（第一次） | 36.25 | 88.35 | +52.10 |
| 22 | press pedal（第二次） | 46.80 | 87.88 | +41.08 |
| 16 | put sample on machine table（第一次） | 36.70 | 72.73 | +36.03 |
| 14 | put sample under electrodes（第一次） | 49.84 | 84.47 | +34.63 |
| 19 | put sample on machine table（第二次） | 56.57 | 90.24 | +33.67 |
| 20 | grip sample from machine table（后续位置） | 38.72 | 65.66 | +26.94 |
| 17 | grip sample from machine table（前序位置） | 61.95 | 80.13 | +18.18 |
| 21 | put sample under electrodes（第二次） | 67.68 | 83.16 | +15.49 |

这组结果是task-history方法最直接的机制证据：模型大幅减少了“同一种Tier3动作、不同流程node”
之间的混淆。

M3仍较困难的node包括：

| Node | 动作 | M3 Recall |
|---|---|---:|
| 34 | take lock from table | 60.00 |
| 20 | grip sample from machine table | 65.66 |
| 1 | unlock crimper | 72.73 |
| 16 | put sample on machine table | 72.73 |
| 7 | turn on water pump | 72.73 |
| 35 | lock crimper | 74.67 |

后续应优先检查这些node的clip边界、样本数量、视觉相似性以及历史缺失情况。

### 10.3 run级M3相对M0

每个participant-run先对三个seed平均：

| Scope / Split | Run数 | Δ Node均值 | 中位数 | Node正向run | Δ Tier3均值 | Tier3正向run |
|---|---:|---:|---:|---:|---:|---:|
| normal-only / normal | 76 | +14.54 | +12.50 | 76/76 | +2.11 | 48/76 |
| normal-only / fault | 27 | +9.73 | +8.33 | 23/27 | -1.01 | 7/27 |
| normal-only / all | 103 | +13.28 | +12.00 | 99/103 | +1.29 | 55/103 |
| all-runs / normal | 76 | +16.07 | +14.78 | **76/76** | +2.19 | 53/76 |
| all-runs / fault | 27 | +14.96 | +13.89 | **27/27** | +1.60 | 16/27 |
| all-runs / all | 103 | **+15.78** | +14.29 | **103/103** | +2.04 | 69/103 |

all-runs M3的Node优势覆盖所有测试run，是非常强的方向一致性证据。Tier3并非每个run都提升，
再次说明方法的主要目标应表述为node级流程定位。

## 11. Participant差异

### 11.1 all-runs M3结果

| Participant | test normal Node | test fault Node | test all Node | test all Tier3 |
|---|---:|---:|---:|---:|
| A | 83.11 ± 2.51 | 71.05 ± 2.76 | 79.27 ± 2.59 | 80.74 ± 1.45 |
| D | 80.58 ± 0.76 | 87.63 ± 2.46 | 81.53 ± 0.70 | 83.26 ± 0.98 |
| J | 94.06 ± 1.03 | 88.89 ± 2.94 | **92.49 ± 1.53** | **92.55 ± 1.63** |
| M | 84.72 ± 2.50 | **89.66 ± 2.30** | 85.68 ± 2.46 | 85.98 ± 2.03 |

这里的“±”为该participant三个seed之间的样本标准差。

主要观察：

- J是最容易的held-out participant；
- A的fault Node Accuracy只有71.05%，明显低于其他三人；
- D的fault样本只有62个，虽然accuracy高，但不应解释为更稳定；
- 四折总体标准差主要来自participant差异，而不是seed波动。

### 11.2 M3不是每一位participant上都绝对第一

all-runs三seed均值中：

- A和M的最高Node Accuracy为M3；
- D的M2/M6略高于M3约0.43个百分点；
- J的M5/M6略高于M3约0.18个百分点；
- 但M3在四人等权总体上获得最高综合结果。

因此，应表述为“M3是四折总体最优且最平衡”，而不是“每一折每个指标都第一”。

## 12. Seed稳定性

下表先对四位participant平均，再计算三个seed均值的样本标准差。

| Scope | 模型 | Test-all Node | Seed SD | Test-all Tier3 | Seed SD |
|---|---|---:|---:|---:|---:|
| normal-only | M0 | 66.76 | 0.63 | 79.33 | 1.96 |
| normal-only | M1 | 78.59 | 0.88 | 81.01 | 1.21 |
| normal-only | M2 | 79.30 | 0.73 | 81.08 | 0.46 |
| normal-only | M3 | 79.30 | 0.99 | 80.99 | 0.53 |
| normal-only | M4 | 79.32 | 0.76 | 81.25 | 0.42 |
| normal-only | M5 | 79.63 | 1.26 | 81.24 | 0.55 |
| normal-only | M6 | 79.76 | 1.54 | 81.39 | 0.93 |
| all-runs | M0 | 69.81 | 1.99 | 83.32 | 0.68 |
| all-runs | M1 | 79.67 | 1.44 | 84.85 | 0.94 |
| all-runs | M2 | 84.15 | 1.31 | 84.99 | 0.70 |
| all-runs | **M3** | **84.74** | **0.98** | **85.63** | **0.48** |
| all-runs | M4 | 83.01 | 1.00 | 84.58 | **0.15** |
| all-runs | M5 | 83.78 | 1.04 | 84.98 | 0.58 |
| all-runs | M6 | 83.71 | **0.48** | 85.05 | 0.49 |

M3的all-runs seed均值：

| Seed | Node Accuracy | Tier3 Accuracy |
|---|---:|---:|
| 1 | 85.15 | 85.82 |
| 2 | 83.63 | 85.09 |
| 42 | 85.45 | 85.99 |

M3的seed波动小于1个百分点Node和0.5个百分点Tier3；M6的Node最稳定，M4的Tier3最稳定，
但二者平均性能低于M3。

## 13. E2E对照

### 13.1 Tier3预训练迁移

all-runs `test_all`：

| 模型 | Node Acc | Node Macro-F1 | Tier3 Acc | Tier3 Macro-F1 |
|---|---:|---:|---:|---:|
| E2E-Node-Scratch | 74.72 | 75.09 | 81.92 | 79.65 |
| E2E-Node-From-Tier3 | 77.74 | 78.56 | 85.46 | 82.85 |
| 差值 | **+3.02** | **+3.47** | **+3.54** | **+3.20** |

Tier3预训练明确改善35-node端到端模型，说明Tier3视觉表征是有效的初始化来源。

### 13.2 M3与E2E-Node-From-Tier3

| Split | Δ Node Acc | Δ Node Macro-F1 | Δ Tier3 Acc | Δ Tier3 Macro-F1 |
|---|---:|---:|---:|---:|
| normal | +7.88 | +6.11 | +0.92 | +2.31 |
| fault | +5.06 | +3.90 | -1.70 | -0.02 |
| all | **+7.00** | **+5.33** | +0.17 | +1.73 |

M3的优势主要集中在Node空间；Tier3 Accuracy与E2E transfer非常接近。这是符合预期的：
history/task graph增加的是流程位置证据，而不是新的当前clip视觉信息。

### 13.3 M3与直接Tier3分类

all-runs `test_all`：

```text
M3聚合Tier3 Accuracy：85.63
E2E-Tier3-Scratch：   84.92
差值：                +0.71

M3聚合Tier3 Macro-F1：84.58
E2E-Tier3-Scratch：    82.52
差值：                 +2.05
```

如果应用只需要31类Tier3动作标签，直接Tier3模型已经非常有竞争力。如果应用需要：

- 区分同一动作在流程中的不同位置；
- 预测35个task graph node；
- 判断是否满足前置关系；
- 为后续漏做、多做或非法跳转检测提供状态；

则M3的Node优势更重要。

## 14. 严格J折与旧先导结论

新J折完全使用与A/D/M一致的scratch LOSO策略。

| Scope | 模型 | Node Acc | Node Macro-F1 | Tier3 Acc | Tier3 Macro-F1 |
|---|---|---:|---:|---:|---:|
| normal-only | M0 | 72.19 ± 3.04 | 72.62 ± 3.15 | 87.69 ± 3.35 | 80.78 ± 3.63 |
| normal-only | M3 | **89.73 ± 4.08** | **85.99 ± 4.71** | **89.97 ± 3.70** | **85.60 ± 4.52** |
| all-runs | M0 | 74.89 ± 3.35 | 77.20 ± 1.74 | 91.35 ± 0.54 | 86.92 ± 1.78 |
| all-runs | M3 | **92.49 ± 1.53** | **89.64 ± 1.09** | **92.55 ± 1.63** | **89.22 ± 1.30** |

严格J normal-only中，M3相对M0：

```text
Node Accuracy：+17.54
Tier3 Accuracy：+2.28
```

严格J all-runs中：

```text
Node Accuracy：+17.60
Tier3 Accuracy：+1.20
```

因此，旧J先导实验关于“历史大幅提高Node分类”的定性结论在不使用J validation、重新从scratch
训练的严格J折中得到复现。旧先导数值不再用于正式四折均值。

## 15. 修正后的总体结论

### 15.1 得到强支持的结论

1. **历史信息稳定提高35-node分类。**
   M3在all-runs提高14.94个百分点Node Accuracy，12/12 participant-seed配对为正，
   103/103测试run为正。

2. **收益机制是流程位置消歧。**
   Stage 2 Node提高18.88个百分点，而Tier3只提高1.76；重复动作node误判下降84.7%–98.4%。

3. **M3是all-runs四折总体最优模型。**
   它在test_all的Accuracy、Macro-F1和Balanced Accuracy的Node/Tier3指标上均为最高。

4. **all-runs训练优于normal-only。**
   M3严格配对提高5.44个百分点Node和4.65个百分点Tier3，四位participant均值全部为正。

5. **精确实际顺序不是必要条件。**
   M3至少不弱于M2，并在all-runs中有小幅、较一致的优势。

6. **严格J折复现了旧先导结论。**
   去除旧checkpoint和validation可比性问题后，J折仍显示约17.5个百分点Node提升。

### 15.2 得到部分支持的结论

1. **graph relation bias有小幅附加价值。**
   M5/M6平均优于M4，但幅度通常小于1个百分点，且不是每个配对都提高。

2. **M6具有可部署潜力，但不是当前最佳。**
   M6在normal-only获得最高平均Node Accuracy，在all-runs具有较好seed稳定性，但总体仍低于M3。

3. **M1能够利用历史，但缺少位置结构时不够平衡。**
   M1在normal-only normal split很强；all-runs相对normal-only时出现normal Node下降、fault大幅提高的权衡。

4. **Tier3改善有限。**
   history模型的Tier3提升通常只有1–3个百分点；若只做动作外观分类，E2E Tier3仍是强baseline。

### 15.3 仍不能得出的结论

当前实验不能证明：

- 已经完成fault detection；
- 能直接识别漏做、多做、重复或非法顺序；
- clip可以被视为相互独立样本进行普通显著性检验；
- M5 oracle relation是强上限；
- M6 soft relation已经优于所有其他graph使用方式；
- 单相机结果能够直接推广到其他相机或新场景。

## 16. 下一步实验建议

旧报告中“补齐A/D/M normal-only多seed”和“重做严格J折”两项已完成。新的优先级建议如下。

### 16.1 做正式run-level置信区间

当前run级方向统计已经完成，但还应增加：

- participant内run bootstrap；
- 外层participant bootstrap或hierarchical bootstrap；
- normal与fault分别报告95% confidence interval；
- M3−M0、M3−M2、all-runs−normal-only分别计算；
- 不将三个seed当作独立participant。

### 16.2 改进M6 soft relation

建议：

1. 用out-of-fold M0预测构造训练历史概率；
2. 对M0概率做temperature calibration；
3. 使用top-k或置信度门控；
4. 低置信度时退回M4；
5. 加入relation dropout；
6. 检查各attention head学习到的I/M/O/X/S bias；
7. 针对node 20、34、35分析历史概率与relation分布。

### 16.3 将graph信息扩展为显式embedding

当前M3主要利用graph-valid顺序，M5/M6主要利用relation bias。下一步可比较：

- node embedding；
- stage embedding；
- graph distance embedding；
- must/optional/immediate relation embedding；
- GNN编码的node representation；
- graph-constrained classifier或decoder。

### 16.4 从clip分类扩展到sequence anomaly detection

增加独立输出：

- 当前node是否违反task graph；
- 缺少哪些must previous nodes；
- 是否发生非法重复或跳转；
- run-level fault score；
- graph-constrained decoding；
- 漏做、多做和错误顺序的独立评估。

### 16.5 扩展外部有效性

- 增加其他相机；
- 多相机融合；
- 新participant；
- 不同光照或设备设置；
- camera/domain shift；
- 跨数据采集批次验证。

## 17. 推荐用于论文或阶段汇报的核心结果

建议优先报告以下五项：

1. **严格四折三seed all-runs总体：**
   M3 Node Accuracy `84.74 ± 5.81`，Tier3 Accuracy `85.63 ± 5.08`。

2. **M3相对M0：**
   Node Accuracy `+14.94`，Tier3 Accuracy `+2.32`；Node 12/12配对为正。

3. **训练范围：**
   M3 all-runs相对normal-only，Node `+5.44`，Tier3 `+4.65`。

4. **Stage 2机制：**
   M3相对M0，Node `+18.88`，Tier3仅`+1.76`。

5. **重复node与run级证据：**
   重复动作node双向误判下降`84.7%–98.4%`；all-runs Node在103/103测试run上提高。

推荐总结语：

> 在严格四折三seed LOSO中，历史和task graph结构大幅提升35-node流程状态识别，尤其减少相同动作
> 在不同流程位置之间的混淆。graph-valid历史不依赖实际精确执行顺序，并在完整all-runs训练中取得
> 最佳综合性能；relation bias提供小幅附加收益，但soft relation仍有进一步改进空间。

## 18. 结果来源

normal-only四折三seed汇总：

```text
D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\
graph_history_rgb_cross_person_ADM_2026-07-22\outputs\
cross_person_summary_normal_only_ADJM_3seeds
```

all-runs四折三seed汇总：

```text
D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\
graph_history_rgb_cross_person_ADM_2026-07-22\outputs\
cross_person_summary_all_runs_ADJM_3seeds
```

严格训练范围比较：

```text
D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\
graph_history_rgb_cross_person_ADM_2026-07-22\outputs\
training_scope_comparison_ADJM_3seeds
```

最重要的源文件：

```text
all_model_metrics.csv
all_model_cross_person_aggregate.csv
all_model_training_scope_deltas.csv
all_model_training_scope_delta_aggregate.csv
all_model_per_stage_metrics.csv
all_model_per_stage_cross_person_aggregate.csv
```

混淆与run级分析来自各fold、seed、scope和model目录下的：

```text
test_results\test_normal_predictions.csv
test_results\test_fault_predictions.csv
test_results\test_all_predictions.csv
```

所有表格均由实际CSV结果重新计算。报告未修改checkpoint、prediction、probability、metrics或原始summary文件。
