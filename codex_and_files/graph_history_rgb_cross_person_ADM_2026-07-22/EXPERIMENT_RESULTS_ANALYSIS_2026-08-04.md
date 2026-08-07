# RGB Task-Graph History严格四折三Seed及完整Atomic-tail Direct实验结果分析

首次报告日期：2026-07-24
完整更新日期：2026-08-05
相机：`001484412812`
任务：35-node分类，并将35-node概率聚合为31类Tier3结果
主要实验设计：A/D/J/M严格四折LOSO，`seed_1`、`seed_2`、`seed_42`
Atomic-tail实验：A/D/J/M、normal-only与all-runs、Direct Fusion、三seed、三种刷新策略；另补充真实历史顺序测试

## 1. 执行摘要

本次更新在原有A/D/J/M四折、三seed、normal-only与all-runs严格结果，以及完整Direct Head
Fusion、Dynamic Epoch Graph-Valid Shuffle和Atomic-tail Direct实验之外，又对全部72个Atomic-tail
checkpoint统一使用真实历史时间顺序重新测试。新评估不重新训练、不读取当前target，仅改变测试阶段的
history排列，因此可以将Atomic-tail与同样使用真实测试顺序的M2 Direct进行干净比较。

当前最重要的结果如下：

1. **四折主结果网格和Atomic-tail Direct网格均完整。**
   原实验包含`720`条模型级结果；Direct和Dynamic实验各包含
   `4 participants × 3 seeds × 2 train scopes × 3 models × 3 splits = 216`条结果。
   四折主网格合计`1152`条。Atomic-tail另有
   `4 participants × 3 seeds × 2 train scopes × 3 refresh policies × 1 model × 3 splits = 216`条，
   原训练协议总计`1368`条模型-split结果。新增真实顺序测试另有216条重评估记录，因此当前共有
   `1584`条“模型 × 测试协议 × split”评估记录；Atomic-tail仍只有72个训练任务，没有重复训练。

2. **统一为真实测试顺序后，Atomic once仍取得最高四折均值，但优势不稳定。**
   all-runs的`test_all`上，Atomic `refresh_once`达到`91.03 ± 3.13%` Node Accuracy和
   `91.18 ± 3.06%` Tier3 Accuracy；相对M2 Direct提高`0.46/0.54`个百分点。Node和Tier3都只有
   6/12个participant-seed配对提高、6/12下降，Node配对95%区间为`[-1.45, +2.37]`，仍跨0。
   在all-runs的`test_all`上，M2 Direct的Node Accuracy为`90.57 ± 2.66%`，
   Tier3 Accuracy为`90.64 ± 2.64%`；相对M0分别提高`20.76`和`7.32`个百分点，
   两项均在12/12个participant-seed配对中提高。

3. **直接微调分类头明显优于delta修正。**
   all-runs下，M2 Direct相对对应的M2 delta模型提高`6.41`个百分点Node Accuracy和
   `5.65`个百分点Tier3 Accuracy；M3 Direct相对M3提高`5.31`和`4.64`个百分点。
   M2 Direct的Node Accuracy在12/12个配对中高于M2。

4. **位置编码是Direct方案的关键组件。**
   all-runs下，M2 Direct相对无位置编码的M1 Direct提高`10.57`个百分点Node Accuracy和
   `5.67`个百分点Tier3 Accuracy，12/12个配对均提高。

5. **graph-valid重排没有进一步提高Direct准确率。**
   all-runs下，M3 Direct相对M2 Direct的Node Accuracy低`0.52`个百分点、Tier3 Accuracy低
   `0.37`个百分点；normal-only下两者基本持平。因此Direct主结果应优先采用实际顺序的M2。

6. **收益仍主要来自流程node消歧。**
   all-runs下M2 Direct相对M0的Stage 2 Node Accuracy提高`26.72`个百分点，而Stage 2
   Tier3 Accuracy提高`8.48`个百分点。四组重复动作node的双向误判由M0的
   `166/247/206/150`次降至`0/0/3/1`次。

7. **Direct M2的改进具有较好的跨参与者一致性。**
   all-runs下，A/D/J/M四折相对M2的Node Accuracy分别提高
   `10.90 / 6.57 / 1.92 / 6.26`个百分点；相对M0则在103/103个测试run上提高。

8. **all-runs对Direct模型有中等幅度收益。**
   M2 Direct的all-runs相对normal-only提高`1.93`个百分点Node Accuracy和`1.46`个百分点
   Tier3 Accuracy，但两项都只在7/12个seed配对中为正，稳定性弱于原M3的训练范围收益。

9. **原delta实验的结论仍作为机制与消融证据保留。**
   在原实验中，M3是最强delta模型：all-runs Node Accuracy为`84.74%`，相对M0提高
   `14.94`个百分点；其Node指标在12/12配对和103/103个测试run上提高。

10. **relation bias有效果，但仍不是主要性能来源。**
   M5 oracle relation和M6 soft relation相对M4均有小幅平均提升，但幅度通常小于1个百分点，
   且没有稳定超过M3。当前证据更支持将M3作为主模型，将M4/M5/M6作为relation消融。

11. **每个epoch重新进行graph-valid重排没有带来总体提升。**
    all-runs下，Dynamic Frozen-M0 Delta相对固定重排M3的Node/Tier3 Accuracy分别低
    `0.51/0.50`个百分点；Dynamic Direct相对固定M3 Direct低`0.26/0.26`个百分点。
    逐样本比较也显示两组实验的新增纠正数都少于新增退化数，因此不能把动态重排解释为有效增益。

12. **Dynamic实验再次支持“联合训练分类头”而不是冻结M0。**
    all-runs下，Dynamic Joint-Head Delta相对Dynamic Frozen-M0 Delta提高
    `1.14/0.93`个百分点Node/Tier3 Accuracy；Dynamic Direct Fusion进一步提高
    `4.41/3.96`个百分点，达到`89.79/90.02%`，但仍未超过固定顺序的M2 Direct。

13. **Dynamic Direct的主要失效集中在少数参与者、run和跨动作视觉混淆。**
    D的平均Node Accuracy最低（`87.37%`），A则有最多三seed一致错误（25个样本）；
    最难node为1、34、4、8和24。剩余高频错误多为放置/抓取、开/关设备及工具取放之间的
    不同Tier3混淆，而不再主要是同一动作在重复流程node间的混淆。

14. **Atomic-tail的最佳刷新频率随训练范围变化，但每epoch刷新始终不是最佳。**
    真实顺序测试下，all-runs的`refresh_once`为`91.03/91.18%`，高于
    `refresh_every_10`的`89.73/90.29%`和`refresh_every_1`的`89.18/89.45%`；
    相对每epoch刷新提高`1.85/1.74`个百分点。normal-only下仍是`refresh_every_10`最佳
    （`89.42/89.85%`），说明最优刷新频率不是与train scope无关的常数。

15. **真实顺序测试消除了协议混杂，但Atomic-tail仍未稳定超过M2 Direct。**
    all-runs `refresh_once`相对M2 Direct提高`0.46/0.54`个百分点，Node/Tier3均为6胜6负；
    normal-only最佳的`refresh_every_10`提高`0.77/0.67`个百分点，两项均为6胜、1平、5负。
    因此现在可以确认小幅均值优势并非来自Atomic-tail测试重排，但仍应报告为最高均值的结构消融，
    不能表述为跨participant和seed稳定取代M2 Direct。

16. **跨模型失效机制发生了阶段性迁移。**
    M0的Node错误中`45.8%`仍属于同一Tier3的重复流程node混淆；M2 Direct中该比例已降至
    `0.8%`。此后剩余错误约99%是跨Tier3错误，主要具有五类特征：放置/抓取混淆、样品/工具
    混淆、设备开/关状态混淆、lock/unlock状态混淆，以及reverse/inspect等短时动作语义混淆。

这些结果支持以下核心解释：

> 历史和task graph信息的主要价值，是把视觉上相同或相似的动作定位到正确的流程node。
> 它对35-node流程状态识别的帮助远大于对31类Tier3外观分类的帮助；在冻结视觉表征的前提下，
> 让history fusion与新的node分类头共同学习，比只训练delta去修正旧分类头更有效。

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
- 原模型：M0–M6及3个E2E对照；
- Direct模型：M1 Direct、M2 Direct、M3 Direct；
- Dynamic模型：Dynamic Frozen-M0 Delta、Dynamic Joint-Head Delta、Dynamic Direct Fusion；
- Atomic-tail模型：A/D/J/M的Atomic Direct Fusion，`refresh_every_1`、`refresh_every_10`、
  `refresh_once`，同时覆盖`normal_only`与`all_runs`；
- Atomic-tail真实顺序重评估：上述72个checkpoint均额外使用`actual_chronological`测试历史；
- split：`test_normal`、`test_fault`、`test_all`；
- overall、per-stage、prediction、严格training-scope delta、Direct/Dynamic严格配对，以及
  Atomic-tail四折两个scope的刷新策略、逐样本与participant-seed严格配对结果。

旧J先导实验包不再进入四折均值。它仅用于历史对照，以判断早期定性结论是否在严格J折中复现。

### 2.2 完整性检查

| 结果文件 | 行数 | 预期内容 |
|---|---:|---|
| `all_model_metrics.csv` | 720 | 4人 × 3seed × 2scope × 10模型 × 3split |
| `all_model_training_scope_deltas.csv` | 360 | 4人 × 3seed × 10模型 × 3split |
| `all_model_cross_person_aggregate.csv` | 60 | 2scope × 10模型 × 3split |
| `all_model_training_scope_delta_aggregate.csv` | 30 | 10模型 × 3split |
| `all_model_per_stage_metrics.csv` | 2160 | 720个结果 × 3stage |
| `direct_head_metrics.csv` | 216 | 4人 × 3seed × 2scope × 3 Direct模型 × 3split |
| `direct_head_paired_deltas.csv` | 432 | 每个Direct模型分别与M0及对应delta模型比较 |
| `direct_head_aggregate.csv` | 36 | 2scope × 3 Direct模型 × 3split × 2 comparison |
| `dynamic_epoch_shuffle_metrics.csv` | 216 | 4人 × 3seed × 2scope × 3 Dynamic模型 × 3split |
| `dynamic_epoch_shuffle_paired_deltas.csv` | 648 | Dynamic模型与对应静态/动态基线的严格配对 |
| `dynamic_epoch_shuffle_aggregate.csv` | 54 | 2scope × 3 Dynamic模型 × 3split及其预注册比较 |
| `outputs/at_ad/**/completed.json` | 72 | 4人 × 3seed × 2scope × 3刷新策略 × 1 Direct模型 |
| `outputs/at_ad/**/test_results/*_metrics.json` | 216 | 原固定Atomic-tail测试：72个训练任务 × 3split |
| `outputs/at_ad/**/test_results/*_predictions.csv` | 216 | 原固定Atomic-tail测试的逐样本预测 |
| 四折主结果网格 | **1152** | 原720条 + Direct 216条 + Dynamic 216条，共16个模型设置 |
| 原训练协议的模型-split结果 | **1368** | 四折主网格1152条 + Atomic-tail Direct完整网格216条 |
| `outputs/at_ad/**/test_results_actual_order/*_metrics.json` | **216** | 72个Atomic checkpoint × 3split的真实顺序重评估 |
| `outputs/at_ad/**/test_results_actual_order/*_predictions.csv` | **216** | 真实顺序重评估的逐样本预测 |
| `outputs/at_actual/atomic_actual_order_metrics.csv` | **216** | 真实顺序Atomic与对应M2 Direct的严格配对明细 |
| 包含两个Atomic测试协议的全部评估记录 | **1584** | 原1368条 + 真实顺序重评估216条；训练任务数不变 |

本次检查确认：

- participant完整：A、D、J、M；
- seed完整：1、2、42；
- representation scope与train scope严格匹配；
- 每个node模型都具有Node和Tier3指标；
- 直接Tier3模型没有虚构Node指标；
- 每个实验都有normal、fault和all三个测试split。
- 72个Direct训练单元（4人 × 3seed × 2scope × 3模型）均有完成标记、checkpoint与测试结果；
- Direct汇总CSV与216份原始metrics JSON逐项一致，最大浮点差为`1.11×10^-16`。
- 72个Dynamic训练单元也均有完成标记、checkpoint、三split指标与prediction文件；
- Dynamic汇总包含216条metrics、648条严格paired delta和54条participant-first aggregate，
  并与216份原始metrics JSON的实验键逐项核对，无缺失或重复。
- Atomic-tail 72/72个训练任务、72个checkpoint、216份metrics和216份prediction均存在；
- Atomic-tail的72份`test_all_metrics.json`与逐样本prediction重算结果完全一致，最大浮点差为
  `0`；72份shuffle audit的`atomic_tail_violations`总数为0，配置边界核验无异常。
- 72/72个Atomic checkpoint均完成真实顺序重评估，产生216份metrics和216份prediction；
  `outputs/at_actual/completed.json`确认participant、seed、scope、refresh policy和split网格均完整。

### 2.3 严格LOSO设置

每一折只使用另外三位参与者训练，held-out participant不进入训练或validation。所有backbone均：

- 从scratch训练；
- 不使用validation或early stopping；
- 不根据held-out participant选择checkpoint；
- 使用最后一个epoch的`last.pth`；
- normal-only与all-runs分别训练独立backbone和下游模型。

因此，原模型、Direct、Dynamic和Atomic-tail均可按A/D/J/M四折、participant-first方式汇总。
Atomic-tail仍只完成Direct Fusion分支；原计划中的Atomic Frozen-M0 Delta与Atomic Joint-Head Delta
没有运行，不能把本节结论推广到这两种结构。

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
| M1 Direct | 当前clip + 历史，无位置编码；fusion后直接35-node分类 | direct-head基础对照 |
| M2 Direct | 实际发生顺序历史 + 位置编码；fusion后直接35-node分类 | direct-head actual-order |
| M3 Direct | graph-valid重排历史 + 位置编码；fusion后直接35-node分类 | direct-head graph-valid |
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

## 5. 原十三模型Strict normal-only统一结果（Dynamic另见第20节）

### 5.1 test_all四折三seed结果

| 模型 | Node Acc | Node Macro-F1 | Tier3 Acc | Tier3 Macro-F1 |
|---|---:|---:|---:|---:|
| M0 | 66.76 ± 3.86 | 68.38 ± 2.90 | 79.33 ± 5.65 | 76.21 ± 3.05 |
| M1 | 78.59 ± 5.53 | 78.71 ± 3.13 | 81.01 ± 4.99 | 80.84 ± 2.60 |
| M2 | 79.30 ± 7.70 | 78.47 ± 5.65 | 81.08 ± 6.70 | 80.03 ± 4.34 |
| M3 | 79.30 ± 6.96 | 78.48 ± 5.26 | 80.99 ± 6.00 | 79.92 ± 4.23 |
| M4 | 79.32 ± 5.74 | 78.01 ± 4.22 | 81.25 ± 5.21 | 79.31 ± 3.66 |
| M5 | 79.63 ± 6.65 | 78.24 ± 5.06 | 81.24 ± 6.42 | 79.39 ± 4.40 |
| M6 | 79.76 ± 6.38 | 78.34 ± 4.63 | 81.39 ± 5.84 | 79.49 ± 3.71 |
| E2E-Node-Scratch | 71.77 ± 4.73 | 72.44 ± 3.30 | 78.84 ± 5.99 | 77.07 ± 2.75 |
| E2E-Node-From-Tier3 | 74.88 ± 4.88 | 74.52 ± 3.73 | 81.82 ± 5.51 | 78.67 ± 3.62 |
| E2E-Tier3-Scratch | — | — | 81.22 ± 4.39 | 78.07 ± 2.40 |
| M1 Direct | 80.52 ± 4.93 | 80.86 ± 4.28 | 83.16 ± 4.80 | 82.71 ± 3.69 |
| M2 Direct | 88.64 ± 3.50 | **85.64 ± 3.88** | 89.18 ± 3.56 | **85.11 ± 3.92** |
| **M3 Direct** | **88.72 ± 3.97** | 85.51 ± 4.22 | **89.29 ± 3.96** | 84.94 ± 4.22 |

normal-only的主要现象：

1. 原M1–M6相对M0提高约11.8–13.0个百分点Node Accuracy；
2. Direct M2/M3进一步将Node Accuracy提高到约88.7%，相对M0提高约22个百分点；
3. M3 Direct获得最高Node和Tier3 Accuracy，M2 Direct获得最高Node/Tier3 Macro-F1；
4. M2 Direct与M3 Direct非常接近，Node/Tier3 Accuracy只差`0.08/0.12`个百分点；
5. 原十模型继续作为delta与E2E消融保留，但不再代表全部实验的最高结果。

Balanced Accuracy同样由Direct模型领先：M2 Direct的Node/Tier3 Balanced Accuracy为
`85.94/85.39`，均高于原十模型。

### 5.2 十三模型在三个split中的最佳模型

| Split | 最佳Node Acc | 最佳Node Macro-F1 | 最佳Tier3 Acc | 最佳Tier3 Macro-F1 |
|---|---|---|---|---|
| normal | M3 Direct, 91.05 | M3 Direct, 87.29 | M3 Direct, 91.21 | M2 Direct, 86.51 |
| fault | M2 Direct, 83.24 | M2 Direct, 78.58 | M2 Direct, 85.05 | M2 Direct, 78.32 |
| all | M3 Direct, 88.72 | M2 Direct, 85.64 | M3 Direct, 89.29 | M2 Direct, 85.11 |

Direct M2/M3覆盖normal-only三个split的全部最佳项。M1 Direct仍明显落后，表明提升不是单纯来自
替换分类头，而是来自位置编码与history fusion共同作用。

## 6. 原十三模型完整all-runs统一结果（Dynamic另见第20节）

### 6.1 test_all四折三seed结果

| 模型 | Node Acc | Node Macro-F1 | Tier3 Acc | Tier3 Macro-F1 |
|---|---:|---:|---:|---:|
| M0 | 69.81 ± 3.94 | 72.91 ± 3.51 | 83.32 ± 5.66 | 81.41 ± 3.98 |
| M1 | 79.67 ± 7.72 | 80.93 ± 5.85 | 84.85 ± 5.44 | 84.37 ± 3.86 |
| M2 | 84.15 ± 5.95 | 83.06 ± 4.61 | 84.99 ± 5.67 | 83.67 ± 3.87 |
| M3 | 84.74 ± 5.81 | 83.89 ± 4.16 | 85.63 ± 5.08 | 84.58 ± 3.25 |
| M4 | 83.01 ± 5.51 | 81.74 ± 4.25 | 84.58 ± 5.24 | 82.76 ± 3.54 |
| M5 | 83.78 ± 6.12 | 83.19 ± 3.75 | 84.98 ± 5.36 | 84.14 ± 2.71 |
| M6 | 83.71 ± 6.96 | 82.65 ± 5.59 | 85.05 ± 5.76 | 83.53 ± 4.50 |
| E2E-Node-Scratch | 74.72 ± 6.68 | 75.09 ± 6.63 | 81.92 ± 7.62 | 79.65 ± 6.50 |
| E2E-Node-From-Tier3 | 77.74 ± 5.02 | 78.56 ± 4.45 | 85.46 ± 5.01 | 82.85 ± 4.10 |
| E2E-Tier3-Scratch | — | — | 84.92 ± 4.95 | 82.52 ± 4.03 |
| M1 Direct | 79.99 ± 6.52 | 80.69 ± 4.10 | 84.97 ± 4.96 | 84.04 ± 2.24 |
| **M2 Direct** | **90.57 ± 2.66** | **87.81 ± 2.79** | **90.64 ± 2.64** | 87.06 ± 3.00 |
| M3 Direct | 90.05 ± 3.31 | 87.60 ± 3.32 | 90.27 ± 3.10 | **87.23 ± 3.05** |

十三模型统一比较中：

- M2 Direct获得最高Node Accuracy：`90.57%`；
- M2 Direct获得最高Node Macro-F1：`87.81%`；
- M2 Direct获得最高Tier3 Accuracy：`90.64%`；
- M3 Direct获得最高Tier3 Macro-F1：`87.23%`；
- M2 Direct获得最高Node Balanced Accuracy：`88.28%`；
- M3 Direct获得最高Tier3 Balanced Accuracy：`88.04%`。

因此，在完整all-runs条件下，应将M2 Direct作为Accuracy主模型；原M3保留为最强delta模型，
M3 Direct保留为graph-valid顺序消融。

### 6.2 十三模型在三个split中的最佳模型

| Split | 最佳Node Acc | 最佳Node Macro-F1 | 最佳Tier3 Acc | 最佳Tier3 Macro-F1 |
|---|---|---|---|---|
| normal | M2 Direct, 91.26 | M2 Direct, 88.19 | M2 Direct, 91.30 | M3 Direct, 87.57 |
| fault | M3 Direct, 89.79 | M3 Direct, 86.84 | M3 Direct, 89.97 | M3 Direct, 86.17 |
| all | M2 Direct, 90.57 | M2 Direct, 87.81 | M2 Direct, 90.64 | M3 Direct, 87.23 |

Direct M2/M3覆盖all-runs三个split的全部最佳项。M2 Direct在normal和test_all的Accuracy上
最好，M3 Direct在fault及部分macro/balanced指标上略优，反映实际顺序与graph-valid重排之间的
小幅权衡。

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
| normal-only | M6 | +13.00 | 12/12 | +9.96 | +2.06 | 10/12 | +3.28 |
| normal-only | M1 Direct | +13.76 | 12/12 | +12.47 | +3.83 | 10/12 | +6.49 |
| normal-only | M2 Direct | +21.88 | 12/12 | **+17.26** | +9.85 | 12/12 | **+8.89** |
| normal-only | **M3 Direct** | **+21.96** | 12/12 | +17.13 | **+9.96** | 12/12 | +8.73 |
| all-runs | M1 | +9.87 | 11/12 | +8.02 | +1.54 | 10/12 | +2.96 |
| all-runs | M2 | +14.35 | 12/12 | +10.15 | +1.68 | 10/12 | +2.26 |
| all-runs | M3 | +14.94 | 12/12 | +10.98 | +2.32 | 12/12 | +3.17 |
| all-runs | M4 | +13.20 | 12/12 | +8.83 | +1.27 | 11/12 | +1.35 |
| all-runs | M5 | +13.97 | 12/12 | +10.29 | +1.66 | 12/12 | +2.73 |
| all-runs | M6 | +13.91 | 12/12 | +9.74 | +1.73 | 11/12 | +2.12 |
| all-runs | M1 Direct | +10.19 | 12/12 | +7.78 | +1.65 | 11/12 | +2.63 |
| all-runs | **M2 Direct** | **+20.76** | **12/12** | **+14.90** | **+7.32** | **12/12** | +5.65 |
| all-runs | M3 Direct | +20.25 | 12/12 | +14.69 | +6.96 | 12/12 | **+5.82** |

最稳健的结论是：

- 历史信息对35-node分类的帮助极其稳定；
- normal-only和all-runs中，原M1–M6以及Direct模型的Node Accuracy均大幅超过M0；
- Direct M2/M3在两个scope均达到12/12 Node和12/12 Tier3 Accuracy正向；
- M2/M3 Direct相对M0的提升明显高于对应delta模型，证明联合学习fusion与新head更有效；
- Tier3提升仍小于Node提升，历史的首要作用仍是流程位置消歧。

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
| M1 Direct | -0.53 ± 5.47 | +1.81 ± 5.04 | 6/12 | 7/12 |
| M2 Direct | +1.93 ± 2.53 | +1.46 ± 2.50 | 7/12 | 7/12 |
| M3 Direct | +1.33 ± 3.14 | +0.98 ± 2.90 | 7/12 | 6/12 |

all-runs训练不仅改善history模型，也改善M0和三个E2E对照，说明收益的一部分来自更充分的视觉
训练分布，而不是仅来自history attention。

在原十模型中，M3的平均提升最大：

- Node Accuracy：+5.44；
- Node Macro-F1：+5.41；
- Tier3 Accuracy：+4.65；
- Tier3 Macro-F1：+4.66。

Direct M2/M3的all-runs平均值也高于normal-only，但正向配对仅约一半，说明Direct模型的
训练范围收益主要集中在部分participant/seed，不能表述为逐seed稳定提高。

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
| M1 Direct | -1.75 | +0.79 | +2.70 | +5.03 |
| M2 Direct | +0.71 | +0.60 | +6.51 | +4.81 |
| M3 Direct | -0.40 | -0.37 | **+8.13** | **+6.19** |

主要解释：

1. all-runs对fault的改善通常大于对normal的改善；
2. 原M2–M6在normal与fault上均为正；
3. M1是唯一出现明确权衡的history模型：normal Node下降1.18，但fault Node提高8.43；
4. M3在normal与fault上都获得较大改善，是更平衡的配置；
5. Direct M2/M3的收益更集中在fault；M3 Direct的fault Node提高8.13，但normal Node略降0.40；
6. 因此，all-runs对原M3是全面收益，对Direct则更像是fault-domain适配收益。

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
| normal-only | M3 Direct − M2 Direct | +0.08 | 4/12 | -0.13 | +0.12 | 3/12 | -0.16 |
| all-runs | M3 Direct − M2 Direct | -0.52 | 4/12 | -0.21 | -0.37 | 4/12 | +0.17 |

normal-only下原M2/M3及Direct M2/M3都非常接近。all-runs下，原delta模型使用graph-valid重排
有小幅优势，但Direct模型使用实际顺序的M2更好。因此可以得出：

- 模型不需要严格复现历史动作的实际精确顺序才能获得收益；
- task graph允许的相对顺序足以保留有用历史结构；
- graph-valid order对delta模型可能具有轻微正则化作用，但在Direct模型中没有额外Accuracy收益；
- 该结果符合多人协作场景：历史动作可能由不同人完成，准确个人执行顺序不一定是关键。

### 9.2 M1与M2：位置编码的作用

| Scope | 比较 | Δ Node Acc | Node正向 | Δ Tier3 Acc | Δ Tier3 Macro-F1 |
|---|---|---:|---:|---:|---:|
| normal-only | M2 − M1 | +0.72 | 6/12 | +0.07 | -0.81 |
| all-runs | M2 − M1 | +4.48 | 12/12 | +0.14 | -0.70 |
| normal-only | M2 Direct − M1 Direct | **+8.12** | **12/12** | **+6.02** | +2.40 |
| all-runs | M2 Direct − M1 Direct | **+10.57** | **12/12** | **+5.67** | +3.02 |

原delta模型中，位置编码主要提高Node定位而几乎不改变Tier3 Accuracy；Direct模型中，
位置编码同时大幅改善Node和Tier3，并在两个scope均为12/12 Node配对提高。这进一步证明：

```text
“当前视觉动作位于task graph的哪个node？”
```

而不是回答：

```text
“当前clip看起来是哪一种Tier3动作？”
```

原delta模型在normal-only下的位置优势很弱，但Direct模型在normal-only下仍提高8.12个百分点
Node Accuracy。因此，对Direct Head Fusion而言，显式位置结构是必要组件，而不只是all-runs
条件下的辅助正则化。

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
| M1 | 83.66 | 77.56 | 86.43 | 84.67 |
| M2 | 83.91 | 84.18 | 83.64 | 85.32 |
| M3 | 84.72 | 84.61 | 84.86 | 85.83 |
| M4 | 82.17 | 83.18 | 82.19 | 85.33 |
| M5 | 83.96 | 83.36 | 84.91 | 85.01 |
| M6 | 84.07 | 83.71 | 82.60 | 85.57 |
| **M1 Direct** | **85.40** | 78.29 | 83.06 | 85.09 |
| **M2 Direct** | 83.48 | **92.45** | 87.34 | **92.55** |
| **M3 Direct** | 84.32 | 91.49 | **87.77** | 91.79 |

M3相对M0：

| Split | Stage 1 Δ Node | Stage 2 Δ Node | Stage 3 Δ Node | Stage 2 Δ Tier3 |
|---|---:|---:|---:|---:|
| normal | +3.66 | **+18.92** | +4.74 | +1.81 |
| fault | +1.14 | **+19.85** | +4.12 | +2.14 |
| all | +3.16 | **+18.88** | +4.58 | +1.76 |

Stage 2在normal和fault中都获得约19个百分点Node提升，而Tier3提升仅约2个百分点。
这不是由某一个测试划分独占，而是任务结构本身导致的稳定现象。

M2 Direct相对M0在all-runs `test_all`的Stage 1/2/3 Node差值为
`+1.92 / +26.72 / +7.06`个百分点，Stage 2 Tier3提高`8.48`个百分点。统一13模型表进一步说明，
Direct新增收益也主要集中在Stage 2。

### 10.2 重复动作node混淆

聚合四位participant、三个all-runs seed的`test_all` prediction，同时比较M0、原M3和M2 Direct：

| 相同Tier3动作对应node | M0双向误判 | 原M3双向误判 | M2 Direct双向误判 | M2 Direct相对M0降幅 |
|---|---:|---:|---:|---:|
| node 14 ↔ 21：`put sample under electrodes` | 166 | 6 | **0** | **100.0%** |
| node 15 ↔ 22：`press pedal` | 247 | 4 | **0** | **100.0%** |
| node 16 ↔ 19：`put sample on machine table` | 206 | 13 | **3** | **98.5%** |
| node 17 ↔ 20：`grip sample from machine table` | 150 | 23 | **1** | **99.3%** |

原M3相对M0最大的单node recall提升：

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

这组结果是task-history方法最直接的机制证据：原M3已经大幅减少“同一种Tier3动作、不同流程node”
之间的混淆，M2 Direct又将四组误判进一步压缩到接近零。

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

## 11. Participant差异：原M3与Direct主模型

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

### 11.2 原十模型中M3不是每一位participant上都绝对第一

all-runs三seed均值中：

- A和M的最高Node Accuracy为M3；
- D的M2/M6略高于M3约0.43个百分点；
- J的M5/M6略高于M3约0.18个百分点；
- 但M3在四人等权总体上获得最高综合结果。

因此，在原十模型范围内应表述为“M3是四折总体最优且最平衡”，而不是“每一折每个指标都第一”。

### 11.3 all-runs M2 Direct结果

| Participant | test normal Node | test fault Node | test all Node | test all Tier3 |
|---|---:|---:|---:|---:|
| A | 92.29 ± 0.86 | 83.70 ± 4.02 | 89.56 ± 1.81 | 89.64 ± 1.74 |
| D | 88.25 ± 4.09 | 90.32 ± 1.61 | 88.53 ± 3.69 | 88.67 ± 3.60 |
| J | **95.61 ± 1.18** | 91.87 ± 0.69 | **94.47 ± 1.02** | **94.53 ± 1.10** |
| M | 88.89 ± 1.92 | **93.10 ± 1.15** | 89.71 ± 1.36 | 89.71 ± 1.36 |

M2 Direct在四位participant上的`test_all` Node Accuracy均高于对应M2 delta模型；A/D/J/M分别
提高`10.90 / 6.57 / 1.92 / 6.26`个百分点。Direct的总体优势不是由单一participant驱动。

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
| normal-only | M1 Direct | 80.52 | 2.22 | 83.16 | 0.97 |
| normal-only | M2 Direct | 88.64 | 0.88 | 89.18 | 0.77 |
| normal-only | M3 Direct | **88.72** | 0.69 | **89.29** | 0.32 |
| all-runs | M0 | 69.81 | 1.99 | 83.32 | 0.68 |
| all-runs | M1 | 79.67 | 1.44 | 84.85 | 0.94 |
| all-runs | M2 | 84.15 | 1.31 | 84.99 | 0.70 |
| all-runs | M3 | 84.74 | 0.98 | 85.63 | 0.48 |
| all-runs | M4 | 83.01 | 1.00 | 84.58 | 0.15 |
| all-runs | M5 | 83.78 | 1.04 | 84.98 | 0.58 |
| all-runs | M6 | 83.71 | 0.48 | 85.05 | 0.49 |
| all-runs | M1 Direct | 79.99 | 1.43 | 84.97 | 0.65 |
| all-runs | **M2 Direct** | **90.57** | 1.87 | **90.64** | 1.84 |
| all-runs | M3 Direct | 90.05 | **0.43** | 90.27 | 0.27 |

M3的all-runs seed均值：

| Seed | Node Accuracy | Tier3 Accuracy |
|---|---:|---:|
| 1 | 85.15 | 85.82 |
| 2 | 83.63 | 85.09 |
| 42 | 85.45 | 85.99 |

M3的seed波动小于1个百分点Node和0.5个百分点Tier3；M6的Node最稳定，M4的Tier3最稳定，
但二者平均性能低于M3。

Direct模型的all-runs seed均值：

| 模型 | Seed | Node Accuracy | Tier3 Accuracy |
|---|---:|---:|---:|
| M2 Direct | 1 | 92.28 | 92.33 |
| M2 Direct | 2 | 90.85 | 90.90 |
| M2 Direct | 42 | 88.57 | 88.68 |
| M3 Direct | 1 | 90.37 | 90.43 |
| M3 Direct | 2 | 90.22 | 90.43 |
| M3 Direct | 42 | 89.57 | 89.96 |

M2 Direct平均性能最高，但seed间差异较大；M3 Direct的Node/Tier3 seed SD仅`0.43/0.27`，
稳定性更好。这是“最高均值”和“最低seed波动”之间的取舍。

## 13. E2E对照

### 13.1 Tier3预训练迁移

all-runs `test_all`：

| 模型 | Node Acc | Node Macro-F1 | Tier3 Acc | Tier3 Macro-F1 |
|---|---:|---:|---:|---:|
| E2E-Node-Scratch | 74.72 | 75.09 | 81.92 | 79.65 |
| E2E-Node-From-Tier3 | 77.74 | 78.56 | 85.46 | 82.85 |
| 差值 | **+3.02** | **+3.47** | **+3.54** | **+3.20** |

Tier3预训练明确改善35-node端到端模型，说明Tier3视觉表征是有效的初始化来源。

### 13.2 M2 Direct与E2E-Node-From-Tier3

| Split | Δ Node Acc | Δ Node Macro-F1 | Δ Tier3 Acc | Δ Tier3 Macro-F1 |
|---|---:|---:|---:|---:|
| normal | +13.52 | +9.56 | +5.70 | +4.30 |
| fault | +10.50 | +8.87 | +3.26 | +3.83 |
| all | **+12.82** | **+9.25** | **+5.18** | **+4.22** |

M2 Direct在normal、fault和all三个split的Node与Tier3指标上均超过端到端Tier3迁移模型。
这说明冻结视觉backbone并不会阻止强node识别；当history fusion与node head联合优化时，
流程信息还能反过来改善聚合后的Tier3判断。

### 13.3 M2 Direct与直接Tier3分类

all-runs `test_all`：

```text
M2 Direct聚合Tier3 Accuracy：90.64
E2E-Tier3-Scratch：          84.92
差值：                       +5.72

M2 Direct聚合Tier3 Macro-F1：87.06
E2E-Tier3-Scratch：           82.52
差值：                        +4.54
```

如果应用只需要31类Tier3动作标签，直接Tier3模型已经非常有竞争力。如果应用需要：

- 区分同一动作在流程中的不同位置；
- 预测35个task graph node；
- 判断是否满足前置关系；
- 为后续漏做、多做或非法跳转检测提供状态；

则M2 Direct的Node优势更重要。原M3相对E2E的比较仍用于说明delta路线，但当前统一13模型结论
应以Direct主模型为准。

## 14. 严格J折与旧先导结论

新J折完全使用与A/D/M一致的scratch LOSO策略。

| Scope | 模型 | Node Acc | Node Macro-F1 | Tier3 Acc | Tier3 Macro-F1 |
|---|---|---:|---:|---:|---:|
| normal-only | M0 | 72.19 ± 3.04 | 72.62 ± 3.15 | 87.69 ± 3.35 | 80.78 ± 3.63 |
| normal-only | M3 | 89.73 ± 4.08 | 85.99 ± 4.71 | 89.97 ± 3.70 | 85.60 ± 4.52 |
| normal-only | M2 Direct | 92.55 ± 1.66 | 87.97 ± 2.45 | 93.21 ± 2.24 | 87.12 ± 2.93 |
| normal-only | **M3 Direct** | **93.21 ± 0.28** | **88.50 ± 0.48** | **93.87 ± 0.62** | **87.77 ± 0.52** |
| all-runs | M0 | 74.89 ± 3.35 | 77.20 ± 1.74 | 91.35 ± 0.54 | 86.92 ± 1.78 |
| all-runs | M3 | 92.49 ± 1.53 | 89.64 ± 1.09 | 92.55 ± 1.63 | 89.22 ± 1.30 |
| all-runs | M2 Direct | 94.47 ± 1.02 | 91.66 ± 1.29 | 94.53 ± 1.10 | 91.07 ± 1.76 |
| all-runs | **M3 Direct** | **94.59 ± 0.48** | **92.19 ± 0.69** | **94.59 ± 0.48** | **91.58 ± 0.92** |

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
训练的严格J折中得到复现。Direct M2/M3进一步提高了严格J折结果；旧先导数值不再用于正式四折均值。

## 15. 修正后的总体结论

### 15.1 得到强支持的结论

1. **Direct Head Fusion是当前最有效的冻结视觉表征方案。**
   all-runs M2 Direct取得`90.57%` Node Accuracy和`90.64%` Tier3 Accuracy；
   相对M0分别提高`20.76`和`7.32`个百分点。

2. **收益机制是流程位置消歧。**
   M2 Direct相对M0的Stage 2 Node提高`26.72`个百分点；四组重复动作node双向误判下降
   `98.5%–100%`。

3. **位置编码对Direct模型非常重要。**
   M2 Direct相对M1 Direct提高`10.57`个百分点Node Accuracy和`5.67`个百分点Tier3 Accuracy，
   两项均为12/12配对提高。

4. **原delta实验仍稳定证明history有效。**
   在原十模型中，M3是最强delta模型；相对M0提高`14.94`个百分点Node Accuracy，
   12/12 participant-seed和103/103测试run均为正。

5. **Direct模型不需要graph-valid重排才能取得最佳准确率。**
   all-runs M3 Direct相对M2 Direct的Node/Tier3 Accuracy低`0.52/0.37`个百分点；
   实际顺序的M2 Direct应作为主配置。

6. **严格J折复现了旧先导结论。**
   去除旧checkpoint和validation可比性问题后，J折仍显示约17.5个百分点Node提升。

### 15.2 得到部分支持的结论

1. **graph relation bias有小幅附加价值。**
   M5/M6平均优于M4，但幅度通常小于1个百分点，且不是每个配对都提高。

2. **M6具有可部署潜力，但不是当前最佳。**
   M6在原normal-only十模型中获得最高平均Node Accuracy，在all-runs具有较好seed稳定性，
   但总体低于原M3和Direct M2/M3。

3. **M1能够利用历史，但缺少位置结构时不够平衡。**
   原M1在normal-only normal split很强；M1 Direct相对M1 delta在all-runs下也基本持平，
   进一步说明没有位置结构时直接更换head并不足以稳定提高性能。

4. **all-runs对Direct的平均收益存在，但跨seed一致性有限。**
   M2 Direct相对normal-only平均提高`1.93`个百分点Node和`1.46`个百分点Tier3 Accuracy，
   但两项均只有7/12配对为正。

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
- M2 Direct−M0、M2 Direct−M2、M3−M0及all-runs−normal-only分别计算；
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

### 16.6 Direct模型的下一步消融

- 比较只训练node head、训练head+768→512 fusion、训练完整history fusion三种冻结层级；
- 对M2 Direct测试部分解冻ResNet layer4，但保持严格LOSO和固定训练预算；
- 比较随机node head与M0 node head初始化，区分“直接目标”与“初始化来源”的影响；
- 对all-runs M2 Direct增加seed，以确认当前seed 1/2/42之间的差异；
- 在不读取历史真值的前提下，将M6 soft relation加入Direct head，检验relation是否能在更强head上获益。

## 17. 推荐用于论文或阶段汇报的核心结果

建议优先报告以下六项：

1. **严格四折三seed all-runs总体：**
   M2 Direct Node Accuracy `90.57 ± 2.66`，Tier3 Accuracy `90.64 ± 2.64`。

2. **M2 Direct相对M0：**
   Node Accuracy `+20.76`，Tier3 Accuracy `+7.32`；六项Node/Tier3指标均为12/12配对提高。

3. **Direct相对delta：**
   M2 Direct相对M2，Node Accuracy `+6.41`，Tier3 Accuracy `+5.65`；
   Node Accuracy为12/12配对提高。

4. **位置编码与历史顺序：**
   M2 Direct相对M1 Direct，Node `+10.57`，Tier3 `+5.67`；
   M3 Direct并未进一步超过M2 Direct。

5. **Stage 2与重复node机制：**
   M2 Direct相对M0的Stage 2 Node Accuracy提高`26.72`个百分点；四组重复动作node误判
   从`166/247/206/150`次降至`0/0/3/1`次。

6. **跨参与者与run级证据：**
   A/D/J/M四折的M2 Direct均高于对应M2；相对M0的Node Accuracy在103/103测试run上提高。

推荐总结语：

> 在严格四折三seed LOSO中，冻结Tier3预训练视觉表征并联合训练history fusion与新的node分类头，
> 比通过delta修正旧分类头更有效。实际顺序加位置编码的M2 Direct在标准Direct网格中取得最高
> Accuracy，显著减少相同动作在不同流程位置之间的混淆；统一使用真实测试顺序后，Atomic once
> 取得`91.03/91.18%`的更高四折均值，但participant-seed仍为6胜6负，不能宣称稳定优于M2 Direct。

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

Direct Head Fusion四折三seed汇总：

```text
D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\
graph_history_rgb_cross_person_ADM_2026-07-22\outputs\
direct_head_fusion_summary_ADJM_3seeds
```

Dynamic Epoch Graph-Valid Shuffle四折三seed汇总：

```text
D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\
graph_history_rgb_cross_person_ADM_2026-07-22\outputs\
dynamic_epoch_shuffle_summary_ADJM_3seeds
```

Atomic-tail Direct Fusion四折、两个scope原始结果：

```text
D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\
graph_history_rgb_cross_person_ADM_2026-07-22\outputs\at_ad
```

最重要的源文件：

```text
all_model_metrics.csv
all_model_cross_person_aggregate.csv
all_model_training_scope_deltas.csv
all_model_training_scope_delta_aggregate.csv
all_model_per_stage_metrics.csv
all_model_per_stage_cross_person_aggregate.csv
direct_head_metrics.csv
direct_head_paired_deltas.csv
direct_head_aggregate.csv
dynamic_epoch_shuffle_metrics.csv
dynamic_epoch_shuffle_paired_deltas.csv
dynamic_epoch_shuffle_aggregate.csv
outputs\at_ad\{A,D,J,M}_s{1,2,42}\{normal_only,all_runs}\
  {refresh_every_1,refresh_every_10,refresh_once}\
  m3_atomic_tail_direct_fusion\{completed.json,shuffle_audit.json,test_results\*}
```

混淆与run级分析来自各fold、seed、scope和model目录下的：

```text
test_results\test_normal_predictions.csv
test_results\test_fault_predictions.csv
test_results\test_all_predictions.csv
```

统一逐样本失效分析使用：

```text
tools\analyze_dynamic_epoch_shuffle_failures.py
tools\analyze_all_experiment_failures.py
tools\analyze_atomic_tail_full_grid.py
```

所有表格均由实际CSV结果重新计算。报告未修改checkpoint、prediction、probability、metrics或
原始summary文件。

## 19. Direct Head Fusion新增实验

### 19.1 实验问题与设计

原M1–M3保留冻结M0的35-node分类头，并学习history-conditioned delta来修正原logit。
Direct Head Fusion检验另一种更直接的方案：

1. 加载对应participant、seed和train scope的Tier3预训练backbone特征；
2. 冻结RGB backbone，不重新训练视频表征；
3. 将分类头替换为35-node head；
4. 联合训练history fusion与新的node分类头；
5. 保持与原M1–M3相同的历史构造、位置编码和graph-valid重排定义。

因此，Direct与原delta模型的主要差异不是使用了更多测试信息，而是允许分类头与历史融合表示共同适配
35-node目标。Direct每个模型训练50 epochs，未重新训练100-epoch RGB backbone。

### 19.2 完整性与统计口径

Direct结果覆盖：

- A、D、J、M四个held-out participant；
- seed 1、2、42；
- `normal_only`与`all_runs`两个train scope；
- M1 Direct、M2 Direct、M3 Direct；
- `test_normal`、`test_fault`、`test_all`。

共计`72`个训练单元和`216`条模型-split指标。所有完成标记、checkpoint、metrics和prediction文件均存在。
本节数值由原始metrics/prediction重新计算，并与Direct汇总CSV核对。总体均值先在每个participant内
平均三个seed，再对A/D/J/M等权平均；“±”为四个participant均值之间的样本标准差。

### 19.3 Direct总体结果

#### normal-only训练，test_all

| 模型 | Node Acc | Node Macro-F1 | Node Bal Acc | Tier3 Acc | Tier3 Macro-F1 | Tier3 Bal Acc |
|---|---:|---:|---:|---:|---:|---:|
| M1 Direct | 80.52 ± 4.93 | 80.86 ± 4.28 | 81.94 ± 3.98 | 83.16 ± 4.80 | 82.71 ± 3.69 | 83.93 ± 3.27 |
| M2 Direct | **88.64 ± 3.50** | **85.64 ± 3.88** | **85.94 ± 3.78** | 89.18 ± 3.56 | **85.11 ± 3.92** | **85.39 ± 3.73** |
| M3 Direct | 88.72 ± 3.97 | 85.51 ± 4.22 | 85.71 ± 3.91 | **89.29 ± 3.96** | 84.94 ± 4.22 | 85.02 ± 3.74 |

normal-only下M2 Direct与M3 Direct基本持平。两者Node Accuracy只差`0.08`个百分点，
Tier3 Accuracy只差`0.12`个百分点，远小于跨participant标准差。

#### all-runs训练，test_all

| 模型 | Node Acc | Node Macro-F1 | Node Bal Acc | Tier3 Acc | Tier3 Macro-F1 | Tier3 Bal Acc |
|---|---:|---:|---:|---:|---:|---:|
| M1 Direct | 79.99 ± 6.52 | 80.69 ± 4.10 | 81.97 ± 3.75 | 84.97 ± 4.96 | 84.04 ± 2.24 | 85.13 ± 1.93 |
| M2 Direct | **90.57 ± 2.66** | **87.81 ± 2.79** | **88.28 ± 2.54** | **90.64 ± 2.64** | 87.06 ± 3.00 | 87.59 ± 2.74 |
| M3 Direct | 90.05 ± 3.31 | 87.60 ± 3.32 | 88.27 ± 2.89 | 90.27 ± 3.10 | **87.23 ± 3.05** | **88.04 ± 2.61** |

以主要Accuracy指标衡量，M2 Direct是本节Direct三模型中的最佳模型。M3 Direct在Tier3 Macro-F1和
Balanced Accuracy上略高，但没有超过M2 Direct的Node或Tier3 Accuracy。

### 19.4 相对M0与原delta模型

下表为`test_all`配对差值，单位为百分点。

#### normal-only训练

| Direct模型 | 比较对象 | ΔNode Acc | ΔNode Macro-F1 | ΔTier3 Acc | ΔTier3 Macro-F1 |
|---|---|---:|---:|---:|---:|
| M1 Direct | M0 | +13.76 ± 4.92 | +12.47 ± 5.18 | +3.83 ± 3.40 | +6.49 ± 4.51 |
| M1 Direct | M1 | +1.93 ± 4.44 | +2.15 ± 3.00 | +2.14 ± 2.64 | +1.87 ± 1.57 |
| M2 Direct | M0 | **+21.88 ± 3.75** | **+17.26 ± 4.39** | +9.85 ± 3.61 | +8.89 ± 3.84 |
| M2 Direct | M2 | +9.34 ± 6.02 | +7.17 ± 5.30 | +8.10 ± 4.83 | +5.08 ± 4.43 |
| M3 Direct | M0 | **+21.96 ± 3.89** | **+17.13 ± 4.37** | **+9.96 ± 3.52** | **+8.73 ± 3.68** |
| M3 Direct | M3 | +9.42 ± 4.71 | +7.03 ± 3.93 | +8.31 ± 3.65 | +5.03 ± 3.16 |

#### all-runs训练

| Direct模型 | 比较对象 | ΔNode Acc | ΔNode Macro-F1 | ΔTier3 Acc | ΔTier3 Macro-F1 |
|---|---|---:|---:|---:|---:|
| M1 Direct | M0 | +10.19 ± 3.58 | +7.78 ± 2.67 | +1.65 ± 0.87 | +2.63 ± 2.00 |
| M1 Direct | M1 | +0.32 ± 1.89 | -0.24 ± 2.11 | +0.12 ± 0.91 | -0.33 ± 1.72 |
| M2 Direct | M0 | **+20.76 ± 2.40** | **+14.90 ± 3.08** | **+7.32 ± 3.38** | +5.65 ± 2.99 |
| M2 Direct | M2 | +6.41 ± 3.67 | +4.75 ± 2.95 | +5.65 ± 3.50 | +3.40 ± 2.55 |
| M3 Direct | M0 | +20.25 ± 2.83 | +14.69 ± 3.19 | +6.96 ± 3.37 | **+5.82 ± 2.51** |
| M3 Direct | M3 | +5.31 ± 3.35 | +3.71 ± 2.29 | +4.64 ± 2.91 | +2.66 ± 1.75 |

核心判断：

- M1 Direct与M1 delta在all-runs下基本持平，说明没有位置编码时，直接head本身不足以稳定改善模型；
- M2/M3 Direct相对M0及对应delta模型均有明显提升；
- all-runs M2 Direct相对M0的六项Node/Tier3指标均为12/12配对提高；
- M2 Direct相对M2的Node Accuracy、Node Macro-F1、Node Balanced Accuracy和Tier3 Accuracy
  均为12/12配对提高，Tier3 Macro-F1与Balanced Accuracy为10/12提高；
- M3 Direct相对M3的Node Accuracy为12/12提高，其余主要指标为10/12或11/12提高。

这表明性能提升来自“history fusion与node head共同优化”，而不只是随机训练波动。

### 19.5 位置编码与graph-valid重排

#### Direct模型内部差值，test_all

| Train scope | 比较 | ΔNode Acc | ΔNode Macro-F1 | ΔTier3 Acc | ΔTier3 Macro-F1 |
|---|---|---:|---:|---:|---:|
| normal-only | M2 Direct − M1 Direct | +8.12 ± 1.87 | +4.78 | +6.02 | +2.40 |
| normal-only | M3 Direct − M2 Direct | +0.08 ± 0.53 | -0.13 | +0.12 | -0.16 |
| all-runs | M2 Direct − M1 Direct | **+10.57 ± 3.91** | +7.12 | **+5.67** | +3.02 |
| all-runs | M3 Direct − M2 Direct | -0.52 ± 0.92 | -0.21 | -0.37 | +0.17 |

M2 Direct相对M1 Direct的Node和Tier3 Accuracy在两个scope均为12/12配对提高，说明位置编码
是Direct Head Fusion的关键部分。相反，all-runs下M3 Direct只在4/12个Node Accuracy配对和
4/12个Tier3 Accuracy配对中超过M2 Direct。因此：

- **历史的位置信息非常重要；**
- **graph-valid重排并不比实际顺序更适合Direct head；**
- 当前主模型应选M2 Direct，M3 Direct保留为顺序鲁棒性/graph-order消融。

### 19.6 normal、fault与all split

| Train scope | 模型 | test_normal Node/Tier3 Acc | test_fault Node/Tier3 Acc | test_all Node/Tier3 Acc |
|---|---|---:|---:|---:|
| normal-only | M1 Direct | 83.17 / 85.38 | 74.78 / 78.08 | 80.52 / 83.16 |
| normal-only | M2 Direct | 90.55 / 90.70 | 83.24 / 85.05 | 88.64 / 89.18 |
| normal-only | M3 Direct | 91.05 / 91.21 | 81.66 / 83.78 | 88.72 / 89.29 |
| all-runs | M1 Direct | 81.42 / 86.18 | 77.47 / 83.11 | 79.99 / 84.97 |
| all-runs | M2 Direct | **91.26 / 91.30** | **89.75 / 89.86** | **90.57 / 90.64** |
| all-runs | M3 Direct | 90.65 / 90.84 | 89.79 / 89.97 | 90.05 / 90.27 |

all-runs训练对M2/M3 Direct的fault结果帮助最明显，使normal与fault之间的差距大幅缩小。
M2 Direct的fault Node Accuracy由`83.24%`提高到`89.75%`。

### 19.7 跨参与者、seed与run稳定性

#### all-runs test_all按held-out participant

| Participant | M1 Direct Node/Tier3 Acc | M2 Direct Node/Tier3 Acc | M3 Direct Node/Tier3 Acc |
|---|---:|---:|---:|
| A | 76.26 / 80.12 | 89.56 / 89.64 | 89.25 / 89.56 |
| D | 75.97 / 82.97 | 88.53 / 88.67 | 86.65 / 87.23 |
| J | 89.67 / 91.77 | **94.47 / 94.53** | 94.59 / 94.59 |
| M | 78.08 / 85.01 | 89.71 / 89.71 | 89.71 / 89.71 |

M2 Direct相对M2的Node Accuracy在A/D/J/M分别提高
`10.90 / 6.57 / 1.92 / 6.26`个百分点，四折方向一致。

按seed对四位participant平均：

| Train scope | 模型 | seed 1 Node/Tier3 | seed 2 Node/Tier3 | seed 42 Node/Tier3 |
|---|---|---:|---:|---:|
| normal-only | M2 Direct | 87.68 / 88.29 | 89.42 / 89.70 | 88.82 / 89.55 |
| normal-only | M3 Direct | 88.85 / 89.31 | 89.34 / 89.61 | 87.97 / 88.97 |
| all-runs | M2 Direct | 92.28 / 92.33 | 90.85 / 90.90 | 88.57 / 88.68 |
| all-runs | M3 Direct | 90.37 / 90.43 | 90.22 / 90.43 | 89.57 / 89.96 |

all-runs M2 Direct的三个seed均值存在一定差异，但每个seed仍明显高于基线。M3 Direct的seed均值
更集中，但平均Accuracy略低。

将每个participant-run先在三个seed上平均后，all-runs共有103个测试run：

| 比较 | run等权平均ΔNode Acc | 中位数 | 正向run |
|---|---:|---:|---:|
| M1 Direct − M0 | +10.80 | +10.67 | 97/103 |
| M2 Direct − M0 | **+21.66** | +20.83 | **103/103** |
| M3 Direct − M0 | +21.29 | +20.83 | **103/103** |
| M1 Direct − M1 | -0.02 | 0.00 | 48/103 |
| M2 Direct − M2 | **+6.50** | +5.13 | 83/103 |
| M3 Direct − M3 | +5.51 | +4.00 | 82/103 |

### 19.8 Stage机制与重复node混淆

#### all-runs test_all按stage

| 模型 | Stage 1 Node/Tier3 Acc | Stage 2 Node/Tier3 Acc | Stage 3 Node/Tier3 Acc |
|---|---:|---:|---:|
| M1 Direct | 85.40 / 85.40 | 78.29 / 85.09 | 83.06 / 83.06 |
| M2 Direct | 83.48 / 83.48 | **92.45 / 92.55** | 87.34 / 87.34 |
| M3 Direct | 84.32 / 84.32 | 91.49 / 91.79 | **87.77 / 87.77** |

M2 Direct相对M0的Stage 1/2/3 Node Accuracy差值分别为
`+1.92 / +26.72 / +7.06`个百分点；相对M2 delta则为
`-0.43 / +8.27 / +3.70`个百分点。Direct方案的主要新增收益集中在Stage 2，正是重复动作node
最多、最需要流程位置消歧的阶段。

对四组已知重复动作node，汇总12个participant-seed的all-runs `test_all`双向误判次数：

| 重复node对 | M0 | 原M3 | M2 Direct | M3 Direct | M2 Direct相对M0下降 |
|---|---:|---:|---:|---:|---:|
| 14 ↔ 21 | 166 | 6 | **0** | 5 | 100.0% |
| 15 ↔ 22 | 247 | 4 | **0** | 2 | 100.0% |
| 16 ↔ 19 | 206 | 13 | **3** | **3** | 98.5% |
| 17 ↔ 20 | 150 | 23 | **1** | 2 | 99.3% |

这提供了最直接的机制证据：M2 Direct几乎消除了同一Tier3动作在不同流程node之间的混淆。
剩余较困难类别仍包括node 1和node 34，说明非重复node上的视觉或跨参与者差异尚未完全解决。

### 19.9 train scope影响

all-runs减去normal-only的`test_all`差值：

| 模型 | ΔNode Acc | ΔNode Macro-F1 | ΔTier3 Acc | ΔTier3 Macro-F1 | Node正向配对 | Tier3正向配对 |
|---|---:|---:|---:|---:|---:|---:|
| M1 Direct | -0.53 ± 5.47 | -0.17 | +1.81 ± 5.04 | +1.33 | 6/12 | 7/12 |
| M2 Direct | **+1.93 ± 2.53** | +2.17 | **+1.46 ± 2.50** | +1.96 | 7/12 | 7/12 |
| M3 Direct | +1.33 ± 3.14 | +2.09 | +0.98 ± 2.90 | +2.29 | 7/12 | 6/12 |

all-runs对M2/M3 Direct的平均结果有帮助，尤其改善fault split，但跨seed正向比例仅约一半。
因此可以报告其平均收益，但不应描述为每个seed都稳定提高。相较之下，原M3 delta的训练范围收益更一致。

### 19.10 结论与使用建议

1. **可行性得到验证：**冻结RGB backbone，仅联合训练history fusion和新node head，不需要重新训练
   100 epochs backbone，即可取得高准确率。
2. **推荐主配置：**M2 Direct + all-runs；它在主要Node/Tier3 Accuracy、跨参与者一致性和run级结果上最强。
3. **推荐消融：**M1 Direct验证位置编码，M3 Direct验证graph-valid重排；二者不应替代M2 Direct主结果。
4. **保留原实验：**M0–M6仍用于说明delta路线、history收益、relation bias与graph-order消融，
   不应被Direct结果覆盖或删除。
5. **限制：**backbone仍是冻结表征，本实验没有证明端到端联合微调一定更优；此外当前只覆盖单相机和
   A/D/J/M四位participant，仍需外部数据验证。

## 20. Dynamic Epoch Graph-Valid Shuffle实验结果（2026-07-30）

### 20.1 实验问题、配置与完整性

本实验检验：固定随机种子下，如果不再让每个样本整个训练过程只使用一个固定graph-valid历史顺序，
而是在每个epoch为其重新生成一个可复现的合法顺序，是否能通过顺序数据增强提高泛化。三个模型与
原实验完全隔离：

| 模型 | 初始化与训练边界 | 主要严格比较 |
|---|---|---|
| Dynamic Frozen-M0 Delta | 加载并冻结M0，只训练attention与delta | 固定重排M3 |
| Dynamic Joint-Head Delta | **不加载M0**；随机node head与attention、delta联合训练 | Dynamic Frozen、固定M3 |
| Dynamic Direct Fusion | 不加载M0、不使用delta；直接训练fusion与node head | 固定M3 Direct、Dynamic Joint |

动态重排只发生在训练阶段；测试仍使用固定seeded graph-valid顺序，因而模型间测试输入保持一致。
结果覆盖A/D/J/M、seed 1/2/42、`normal_only`/`all_runs`及三个测试split，共72个训练单元、
216条metrics、216份prediction和648条严格配对差值，均完整。以下总体统计继续采用
participant-first口径：先对每位participant的三个seed求均值，再对四位participant等权平均；
“±”为四个participant均值的样本标准差。

### 20.2 Dynamic总体结果

`test_all`结果如下：

| Train scope | Dynamic模型 | Node Acc | Node Macro-F1 | Tier3 Acc | Tier3 Macro-F1 |
|---|---|---:|---:|---:|---:|
| normal-only | Frozen-M0 Delta | 79.20 ± 6.83 | 78.33 | 81.16 ± 6.14 | 79.91 |
| normal-only | Joint-Head Delta | 83.11 ± 6.54 | 82.69 | 84.60 ± 5.09 | 83.73 |
| normal-only | Direct Fusion | **86.92 ± 4.84** | **84.58** | **87.38 ± 5.04** | **84.29** |
| all-runs | Frozen-M0 Delta | 84.24 ± 6.11 | 83.48 | 85.13 ± 5.69 | 84.24 |
| all-runs | Joint-Head Delta | 85.38 ± 5.23 | 84.36 | 86.06 ± 4.93 | 84.76 |
| all-runs | Direct Fusion | **89.79 ± 3.35** | **86.97** | **90.02 ± 3.19** | **86.33** |

三种Dynamic结构的排序在两个train scope中一致：

> **Dynamic Direct Fusion > Dynamic Joint-Head Delta > Dynamic Frozen-M0 Delta。**

这再次表明允许node head随history representation共同学习，比冻结M0分类头只学习delta更有效；
而在联合训练方案内，直接进行feature fusion又明显优于保留“base logits + delta”的结构。
在Dynamic实验及此前标准网格内，最佳仍是all-runs M2 Direct的`90.57/90.64%` Node/Tier3
Accuracy，Dynamic Direct没有刷新最佳结果；相对M2 Direct分别低`0.78/0.62`个百分点。后续补齐的
Atomic once均值略高，原测试协议见第21节，真实顺序统一比较见第23节。

### 20.3 与固定重排及初始化方案的严格配对比较

下表为all-runs `test_all`的participant-first配对差值；正数表示前者更高。

| 比较 | ΔNode Acc | Node正/平/负 | ΔTier3 Acc | Tier3正/平/负 |
|---|---:|---:|---:|---:|
| Dynamic Frozen − 固定M3 | **-0.51** | 5/1/6 | **-0.50** | 5/0/7 |
| Dynamic Joint − 固定M3 | +0.64 | 7/0/5 | +0.43 | 7/1/4 |
| Dynamic Direct − 固定M3 Direct | **-0.26** | 5/1/6 | **-0.26** | 4/1/7 |
| Dynamic Joint − Dynamic Frozen | +1.14 | 7/0/5 | +0.93 | 9/0/3 |
| Dynamic Direct − Dynamic Joint | **+4.41** | 11/0/1 | **+3.96** | 11/0/1 |
| Dynamic Direct − M0 | +19.98 | 12/0/0 | +6.70 | 12/0/0 |

其中第一行和第三行只改变“固定一次还是每epoch重排”，是判断动态重排本身最干净的比较：
两项平均差值均为负，且正向配对没有超过半数。因此当前数据不支持“增加合法排列多样性会自动提高
准确率”。Dynamic Joint虽略高于固定M3，但它同时改变了M0初始化与head训练边界，不能把这
`0.64/0.43`个百分点归因于动态重排。

normal-only下结论更明显：Dynamic Direct相对固定M3 Direct的Node/Tier3 Accuracy分别低
`1.80/1.91`个百分点，仅有4/12和5/12个配对提高。由此看，每epoch强制更换顺序在训练数据较少时
更可能增加优化噪声，而不是形成有效正则化。

逐样本比较也得到相同结论。all-runs下共比较`1895 × 3 seeds = 5685`个样本-seed预测：

| 比较 | Node纠正 | Node退化 | Tier3纠正 | Tier3退化 |
|---|---:|---:|---:|---:|
| Dynamic Frozen vs 固定M3 | 94 | **119** | 74 | **99** |
| Dynamic Direct vs 固定M3 Direct | 151 | **164** | 140 | **153** |
| Dynamic Joint vs Dynamic Frozen | **194** | 133 | **156** | 106 |

也就是说，动态重排确实改变了模型学到的决策边界，而不是完全无效；但相对对应静态模型，它带来的
新增退化略多于新增纠正。Joint相对Frozen则有正净收益，支持联合训练head本身，而不是支持
每epoch重排。

### 20.4 train scope、split与seed稳定性

all-runs模型在三个测试split上的结果：

| Dynamic模型 | test_normal Node/Tier3 | test_fault Node/Tier3 | test_all Node/Tier3 |
|---|---:|---:|---:|
| Frozen-M0 Delta | 85.35 / 86.15 | 82.65 / 83.66 | 84.24 / 85.13 |
| Joint-Head Delta | 86.32 / 86.95 | 84.09 / 84.87 | 85.38 / 86.06 |
| Direct Fusion | **90.77 / 90.96** | **88.11 / 88.48** | **89.79 / 90.02** |

all-runs减去normal-only的`test_all`严格配对差值为：

| Dynamic模型 | ΔNode Acc | Node正向配对 | ΔTier3 Acc | Tier3正向配对 |
|---|---:|---:|---:|---:|
| Frozen-M0 Delta | **+5.04** | 10/12 | **+3.97** | 11/12 |
| Joint-Head Delta | +2.27 | 9/12 | +1.46 | 9/12 |
| Direct Fusion | +2.87 | 11/12 | +2.64 | 10/12 |

Dynamic模型从fault训练样本中获益明显；这组train-scope收益比固定Direct实验更一致，尤其是
Dynamic Direct的Node Accuracy在11/12配对中提高。

all-runs `test_all`按seed对四位participant平均：

| Dynamic模型 | seed 1 Node/Tier3 | seed 2 Node/Tier3 | seed 42 Node/Tier3 |
|---|---:|---:|---:|
| Frozen-M0 Delta | 85.55 / 85.88 | 82.30 / 84.21 | 84.86 / 85.31 |
| Joint-Head Delta | 86.50 / 86.72 | 86.13 / 86.29 | 83.52 / 85.18 |
| Direct Fusion | **90.50 / 90.80** | **89.42 / 89.70** | **89.44 / 89.56** |

Dynamic Direct的三个seed只相差约1.1个百分点，是三者中最稳定的；Joint的seed 42 Node结果偏低，
说明随机初始化head与delta联合优化仍有一定训练方差。

### 20.5 Participant、Stage与局部顺序机制

all-runs `test_all`按held-out participant：

| Participant | Frozen-M0 Delta Node/Tier3 | Joint-Head Delta Node/Tier3 | Direct Fusion Node/Tier3 |
|---|---:|---:|---:|
| A | 79.12 / 80.20 | 81.44 / 81.75 | 87.63 / 88.32 |
| D | 81.96 / 82.47 | 83.55 / 84.56 | 87.37 / 87.45 |
| J | **93.09 / 93.21** | **93.09 / 93.15** | **94.59 / 94.59** |
| M | 82.77 / 84.64 | 83.45 / 84.79 | 89.56 / 89.71 |

J在三种模型上都最容易；Frozen与Joint最困难的是A。Dynamic Direct中D的平均Node Accuracy最低，
但只比A低`0.25`个百分点，二者应共同视为主要跨参与者困难折。Dynamic Direct相对固定M3 Direct
在D上反而提高`0.72`个百分点Node Accuracy，却在A上下降`1.62`个百分点，说明动态重排的作用
具有明显participant依赖性。

分阶段结果：

| Dynamic模型 | Stage 1 Node/Tier3 | Stage 2 Node/Tier3 | Stage 3 Node/Tier3 |
|---|---:|---:|---:|
| Frozen-M0 Delta | **85.00 / 85.00** | 83.97 / 85.20 | 84.27 / 84.27 |
| Joint-Head Delta | 83.99 / 83.99 | 85.45 / 86.38 | 85.94 / 85.94 |
| Direct Fusion | 83.43 / 83.43 | **91.44 / 91.76** | **86.92 / 86.92** |

Dynamic Direct的优势仍集中在包含大量相似和重复动作的Stage 2。与固定M3 Direct相比，其Stage
1/2/3 Node Accuracy分别约低`0.89/0.05/0.85`个百分点：动态重排没有改善Stage 2，也没有出现
某一stage灾难性崩溃。

对“最新历史node应当靠近当前动作”的局部机制再做两项检查：

| 目标组 | M0 Node/Tier3 | 固定M3 | 固定M3 Direct | Dynamic Frozen | Dynamic Joint | Dynamic Direct |
|---|---:|---:|---:|---:|---:|---:|
| immediate-target nodes | 65.26 / 83.72 | 84.36 / 85.59 | **91.28 / 91.59** | 83.76 / 84.99 | 85.39 / 86.32 | 91.24 / 91.56 |
| Stage2 nodes 13–25 | 64.07 / 83.87 | 84.27 / 85.59 | 91.51 / 91.84 | 83.58 / 84.91 | 85.31 / 86.31 | **91.60 / 91.95** |

Dynamic Direct没有在immediate-target组上崩溃，但也没有超过固定M3 Direct；这与总体结论一致：
全局graph-valid随机性能够保留大部分历史收益，却没有充分保护“最新atomic前缀”的局部邻近信号。
完整四折Atomic-tail结果进一步表明：保护局部atomic前缀后，all-runs的`refresh_once`优于高频
刷新；normal-only则以`refresh_every_10`最佳，详见第21节。

### 20.6 详细失效分析

#### 20.6.1 最难node与participant特异错误

Dynamic Direct的最低召回node如下。Recall为四位participant等权平均，support为四折中的唯一
测试样本数；低support类别的百分比应谨慎解释。

| Node | 动作 | Recall | Unique support |
|---:|---|---:|---:|
| 1 | unlock crimper | **62.22%** | 22 |
| 34 | take lock from table | **70.92%** | 25 |
| 4 | turn on crimper | 74.17% | 22 |
| 8 | turn on extractor fan | 76.39% | 22 |
| 24 | put sample on table | 77.28% | 103 |
| 35 | lock crimper | 78.01% | 25 |
| 6 | turn on air compressor | 78.06% | 22 |
| 7 | turn on water pump | 78.61% | 22 |
| 30 | turn off air compressor | 79.91% | 26 |
| 20 | grip sample from machine table | 83.10% | 99 |

其中node 24不仅support较大，而且在A折Recall只有`36.11%`，是比低support node 1更可靠的
系统性弱点。participant特异的最低Recall还包括：D的node 18（reverse sample，`41.67%`）、
J的node 34（`20.83%`）以及M的node 1（`33.33%`）。因此不能用单一“最难动作”概括所有人；
跨参与者外观、操作习惯和场景差异会改变主要错误类别。

#### 20.6.2 高频混淆

下表汇总all-runs Dynamic Direct在四折三seed上的有向错误次数；同一真实样本会随三个seed计三次。

| 真实node → 预测node | 动作混淆 | 错误次数 |
|---|---|---:|
| 20 → 19 | grip sample from machine table → put sample on machine table | 24 |
| 24 → 12 | put sample on table → take plier from table | 23 |
| 24 → 25 | put sample on table → put plier on table | 22 |
| 24 → 34 | put sample on table → take lock from table | 18 |
| 16 → 17 | put sample on machine table → grip sample from machine table | 17 |
| 18 → 23 | reverse sample → inspect sample | 17 |
| 34 → 24 | take lock from table → put sample on table | 17 |
| 1 → 4 | unlock crimper → turn on crimper | 13 |
| 8 → 28 | turn on extractor fan → turn off extractor fan | 13 |
| 30 → 6 | turn off air compressor → turn on air compressor | 13 |
| 7 → 29 | turn on water pump → turn off water pump | 12 |
| 6 → 30 | turn on air compressor → turn off air compressor | 11 |

这些高频错误全部跨Tier3，而不是同一Tier3动作在不同node位置之间的互换。剩余瓶颈已经从原M0的
“流程位置消歧”部分转向更困难的视觉语义问题：放置与抓取、样品与工具、设备开与关，以及相邻
操作间过渡帧外观相似。改进这类错误更可能需要时序片段、手-物交互或设备状态特征，而不是单纯增加
更多历史排列。

四组已知重复动作node的双向误判次数进一步支持这一判断：

| 模型 | 14↔21 | 15↔22 | 16↔19 | 17↔20 | 合计 |
|---|---:|---:|---:|---:|---:|
| M0 | 166 | 247 | 206 | 150 | 769 |
| 固定M3 | 6 | 4 | 13 | 23 | 46 |
| 固定M3 Direct | 5 | 2 | 3 | 2 | **12** |
| Dynamic Frozen | 2 | 3 | 18 | 22 | 45 |
| Dynamic Joint | 9 | 14 | 7 | 7 | 37 |
| Dynamic Direct | 1 | 2 | 4 | 6 | 13 |

Dynamic Direct仍几乎消除了重复node混淆，但合计13次并未优于固定M3 Direct的12次。Dynamic
Joint在14↔21和15↔22上明显多于Frozen，说明“联合训练head”带来的平均提升并不保证所有
重复node对都同步改善。

#### 20.6.3 三seed一致错误、困难run与置信度

Dynamic Direct共有`79/1895`个唯一测试样本在三个seed中全部预测错误。按participant为
A 25、D 22、M 19、J 13；按stage为Stage 1/2/3的`23/39/17`个。Stage 2错误数最高主要受其
样本量更大影响，不等于Stage 2错误率最高。三seed一致错误最多的node是24（17个）、18（9个）
和34（6个），说明这些错误不是简单依靠更换seed即可消除。

seed平均后的最困难run为：

| Participant / run | 样本数 | Node Acc | Tier3 Acc |
|---|---:|---:|---:|
| A / run_16 | 25 | **66.67%** | 68.00% |
| M / run_19 | 25 | 69.33% | 69.33% |
| A / run_28 | 24 | 72.22% | 73.61% |
| A / run_27 | 6 | 72.22% | 72.22% |
| D / run_10 | 25 | 73.33% | 73.33% |
| D / run_19 | 25 | 74.67% | 74.67% |
| J / run_35 | 14 | 76.19% | 76.19% |

在103个run的等权比较中，Dynamic Frozen相对固定M3有30个提高、32个持平、41个下降，
平均Node差`-0.49`个百分点；Dynamic Direct相对固定M3 Direct为33提高、28持平、42下降，
平均差`-0.40`个百分点。这说明总体负差不是由单个异常run造成，而是小幅、分散地出现在更多run上。

最后，三种Dynamic模型均存在高置信错误：

| Dynamic模型 | 正确样本平均置信度 | 错误样本平均置信度 | ≥0.9错误/全部错误 | 10-bin ECE |
|---|---:|---:|---:|---:|
| Frozen-M0 Delta | 97.47% | 83.85% | 485/866（56.0%） | 10.63% |
| Joint-Head Delta | 97.79% | 84.77% | 469/805（58.3%） | 10.11% |
| Direct Fusion | **98.01%** | **79.76%** | 262/565（46.4%） | **6.13%** |

Dynamic Direct准确率和校准都最好，但仍有近一半错误的置信度不低于0.9。部署或在线异常检测时，
不能只用最大softmax阈值识别失败；更适合结合graph不一致、动作持续时间、预测跳变和模型集成分歧。

### 20.7 Dynamic实验结论

1. **每epoch graph-valid重排可正常训练，但不是当前的性能改进项。**两个最干净的静态—动态比较
   均为小幅负差，逐样本纠正少于退化。
2. **Dynamic Joint优于Dynamic Frozen，说明重新学习node head是有价值的；**但其优势混合了
   初始化、训练边界与结构差异，不能归因于重排频率。
3. **Dynamic Direct是三种Dynamic模型中明确最佳，**但仍低于固定M2 Direct，也未超过固定M3
   Direct；正式主模型与原有结论无需更换。
4. **历史信息已经基本解决重复流程node消歧，**剩余错误更多是跨Tier3视觉/动作语义混淆和
   participant/run域偏移。
5. **完整Atomic-tail结果验证了局部顺序假设的一部分：**保护最新atomic前缀后，每epoch强增强在
   两个scope都不是最佳；all-runs偏好固定一次，normal-only偏好每10 epochs刷新。

## 21. Atomic-tail Direct Fusion完整实验结果：原固定Atomic-tail测试协议（2026-08-04）

### 21.1 实际运行范围与实验边界

原设计包含Frozen-M0 Delta、Joint-Head Delta和Direct Fusion三个Atomic-tail模型。受计算时间限制，
实际仍只运行最有希望的Direct Fusion，但现已补齐A、D、J、M及两个train scope：

```text
4 participants × 3 seeds × 2 train scopes × 3 refresh policies × 1 model = 72 training units
72 training units × 3 test splits = 216 model-split results
```

模型为`m3_atomic_tail_direct_fusion`：随机初始化并训练node head与feature fusion，不加载M0，不使用
delta，RGB backbone仍通过Tier3 feature cache冻结。Atomic-tail构造不读取当前真实target，只把
“真实最新历史node所属的未完成atomic前缀”固定在重排末尾，其余历史保持graph-valid随机化。

三个刷新策略只改变训练期重排频率：

| 策略 | 50 epochs内刷新轮数 | 含义 |
|---|---:|---|
| `refresh_once` | 1 | 每个样本整个训练过程使用一个固定atomic-tail合法顺序 |
| `refresh_every_10` | 5 | 每10 epochs重新生成一次 |
| `refresh_every_1` | 50 | 每个epoch重新生成一次 |

测试阶段三者都使用固定seeded atomic-tail顺序，因此三种Atomic策略之间的比较是干净的训练刷新频率
比较。本节保留首次评估结果以保证实验记录可追溯；统一使用真实测试顺序后的干净比较见第23节。

### 21.2 完整性与重排审计

72/72个训练任务均存在`last.pth`、`completed.json`、`shuffle_audit.json`、三个split的metrics、
prediction和probability。72份`test_all`逐样本预测重算结果与metrics JSON完全一致，最大差为`0`；
配置核验也确认全部任务均为50 epochs、随机可训练node head、不加载M0、不使用delta。

12个participant-seed训练单元上的平均审计结果：

| Train scope | 策略 | Atomic-tail适用比例 | 多顺序样本比例 | 至少一次不同于实际顺序 | 刷新轮数 | Tail违规 |
|---|---|---:|---:|---:|---:|---:|
| all-runs | `refresh_once` | 69.39% | 0.00% | 32.81% | 1 | **0** |
| all-runs | `refresh_every_10` | 69.39% | 34.68% | 34.85% | 5 | **0** |
| all-runs | `refresh_every_1` | 69.39% | 34.82% | 34.92% | 50 | **0** |
| normal-only | `refresh_once` | 73.41% | 0.00% | 32.86% | 1 | **0** |
| normal-only | `refresh_every_10` | 73.41% | 34.52% | 34.73% | 5 | **0** |
| normal-only | `refresh_every_1` | 73.41% | 34.66% | 34.80% | 50 | **0** |

两个scope分别约69.4%和73.4%的训练样本存在可固定的active atomic tail；72份audit全部无违规。
动态策略确实让约三分之一的样本经历多个合法顺序，因此性能差异不是因为“动态刷新没有实际改变数据”。

### 21.3 四折总体结果

以下结果先在每位participant内平均三个seed，再对A/D/J/M等权平均；“±”为四位participant均值的
样本标准差。

| Train scope | 模型/策略 | Node Acc | Tier3 Acc |
|---|---|---:|---:|
| all-runs | Atomic `refresh_once` | **90.99 ± 3.03** | **91.11 ± 2.98** |
| all-runs | M2 Direct / 实际顺序 | 90.57 ± 2.66 | 90.64 ± 2.64 |
| all-runs | M3 Direct / 固定graph-valid | 90.05 ± 3.31 | 90.27 ± 3.10 |
| all-runs | Atomic `refresh_every_10` | 89.78 ± 3.31 | 90.27 ± 3.09 |
| all-runs | Dynamic Direct | 89.79 ± 3.35 | 90.02 ± 3.19 |
| all-runs | Atomic `refresh_every_1` | 89.12 ± 3.61 | 89.34 ± 3.56 |
| normal-only | Atomic `refresh_every_10` | **89.41 ± 2.98** | **89.84 ± 2.95** |
| normal-only | Atomic `refresh_once` | 88.90 ± 2.32 | 89.53 ± 2.36 |
| normal-only | M3 Direct / 固定graph-valid | 88.72 ± 3.97 | 89.29 ± 3.96 |
| normal-only | M2 Direct / 实际顺序 | 88.64 ± 3.50 | 89.18 ± 3.56 |
| normal-only | Atomic `refresh_every_1` | 87.61 ± 5.16 | 88.30 ± 5.39 |
| normal-only | Dynamic Direct | 86.92 ± 4.84 | 87.38 ± 5.04 |

三个测试split的Node/Tier3 Accuracy如下：

| Train scope / 策略 | test_normal | test_fault | test_all |
|---|---:|---:|---:|
| all-runs / every 1 | 89.67 / 89.82 | 88.60 / 89.04 | 89.12 / 89.34 |
| all-runs / every 10 | 90.54 / 91.06 | 88.84 / 89.06 | 89.78 / 90.27 |
| all-runs / once | **91.67 / 91.76** | **89.99 / 90.11** | **90.99 / 91.11** |
| normal-only / every 1 | 89.38 / 89.64 | **84.14 / 86.11** | 87.61 / 88.30 |
| normal-only / every 10 | **91.29 / 91.45** | 84.00 / 85.56 | **89.41 / 89.84** |
| normal-only / once | 90.89 / 91.16 | 84.06 / 85.73 | 88.90 / 89.53 |

all-runs保持A/D先导实验的`once > every 10 > every 1`排序；加入J/M后差距缩小为
`1.21`和`1.87`个百分点。normal-only则由`every 10`领先once `0.50`个百分点，说明“固定一次必然
最佳”不能跨scope推广；但每epoch刷新在两个scope都不是`test_all`最佳。all-runs once相对同scope
M2 Direct在test_normal/test_fault的Node增益为`+0.41/+0.24`个百分点；normal-only every 10则为
`+0.74/+0.76`，说明小幅正差同时出现在normal和fault，而不是单一split造成。

### 21.4 严格participant-seed配对

单位为百分点，每行均为12个四折participant-seed配对：

| Scope | Atomic策略 | 比较对象 | ΔNode | Node正/平/负 | ΔTier3 | Tier3正/平/负 |
|---|---|---|---:|---:|---:|---:|
| all-runs | every 1 | M2 Direct | -1.45 | 2/0/10 | -1.30 | 2/0/10 |
| all-runs | every 10 | M2 Direct | -0.78 | 6/0/6 | -0.37 | 6/0/6 |
| all-runs | once | M2 Direct | **+0.42** | 6/0/6 | **+0.47** | 5/1/6 |
| all-runs | once | M3 Direct | +0.94 | 6/1/5 | +0.83 | 6/0/6 |
| all-runs | once | Dynamic Direct | +1.20 | 8/0/4 | +1.09 | 8/0/4 |
| all-runs | once | every 10 | +1.21 | 7/0/5 | +0.83 | 7/0/5 |
| all-runs | once | every 1 | **+1.87** | **9/0/3** | **+1.76** | **9/0/3** |
| normal-only | every 1 | M2 Direct | -1.03 | 6/0/6 | -0.87 | 6/0/6 |
| normal-only | every 10 | M2 Direct | **+0.76** | 6/2/4 | **+0.66** | 6/0/6 |
| normal-only | once | M2 Direct | +0.26 | 4/1/7 | +0.35 | 5/1/6 |
| normal-only | every 10 | Dynamic Direct | +2.49 | 10/0/2 | +2.46 | 10/0/2 |
| normal-only | every 10 | every 1 | +1.79 | 7/0/5 | +1.54 | 5/1/6 |
| normal-only | every 10 | once | +0.50 | 8/0/4 | +0.31 | 8/0/4 |

最稳健的刷新结论是all-runs的`once > every 1`，两项均在9/12配对中提高。相对M2 Direct，
两个scope的最佳Atomic均值为正，但胜负并不一边倒：all-runs Node为6胜6负，normal-only为
6胜2平4负。以12个配对计算的描述性95%区间，all-runs Node为`[-1.41, +2.26]`个百分点，
normal-only Node为`[-0.65, +2.17]`，均跨0，故不能宣称稳定显著优于M2 Direct。

逐样本上，四折共有`1895 × 3 seeds = 5685`个样本-seed预测。最佳Atomic与参照模型比较如下：

| Scope / Atomic | 比较对象 | Node纠正 | Node退化 | 净纠正 | Tier3纠正 | Tier3退化 |
|---|---|---:|---:|---:|---:|---:|
| all-runs / once | M2 Direct | 155 | 130 | +25 | 153 | 126 |
| all-runs / once | M3 Direct | 195 | 142 | +53 | 187 | 140 |
| all-runs / once | Dynamic Direct | 191 | 125 | **+66** | 181 | 121 |
| normal-only / every 10 | M2 Direct | 240 | 195 | +45 | 224 | 185 |
| normal-only / every 10 | M3 Direct | 248 | 209 | +39 | 228 | 197 |
| normal-only / every 10 | Dynamic Direct | 319 | 186 | **+133** | 303 | 172 |

Atomic会同时纠正和引入大量样本错误；小幅均值提升来自净平衡，而不是对原预测的单调修正。

### 21.5 Participant与seed差异

| Scope | Participant | every 1 Node/Tier3 | every 10 Node/Tier3 | once Node/Tier3 | M2 Direct Node/Tier3 |
|---|---|---:|---:|---:|---:|
| all-runs | A | 88.63 / 89.25 | 88.63 / 88.86 | **90.87 / 91.11** | 89.56 / 89.64 |
| all-runs | D | 86.00 / 86.22 | 88.17 / 88.89 | **88.82 / 88.96** | 88.53 / 88.67 |
| all-runs | J | 94.29 / 94.35 | 94.71 / 94.89 | **95.32 / 95.32** | 94.47 / 94.53 |
| all-runs | M | 87.55 / 87.55 | 87.62 / 88.44 | 88.96 / 89.04 | **89.71 / 89.71** |
| normal-only | A | 89.17 / 90.02 | 89.10 / 89.33 | 88.48 / 89.10 | **89.95 / 90.33** |
| normal-only | D | 86.15 / 86.80 | 87.81 / 88.46 | **89.11 / 89.39** | 87.81 / 88.46 |
| normal-only | J | 93.69 / 94.59 | **93.69 / 94.11** | 91.83 / 92.67 | 92.55 / 93.21 |
| normal-only | M | 81.43 / 81.80 | **87.02 / 87.47** | 86.20 / 86.95 | 84.27 / 84.71 |

all-runs once相对M2 Direct的participant均值在A/D/J分别为`+1.31/+0.29/+0.84`个百分点，
在M为`-0.75`；normal-only every 10则在A/D/J/M分别为`-0.85/0.00/+1.14/+2.76`。
因此四折均值的正差不是所有人共同提高，而是由不同scope下的受益者抵消退化者形成。

四人等权的逐seed结果也显示刷新策略与seed有交互：

| Scope / 策略 | seed 1 Node/Tier3 | seed 2 Node/Tier3 | seed 42 Node/Tier3 |
|---|---:|---:|---:|
| all-runs / every 1 | 90.63 / 90.80 | 88.67 / 88.83 | 88.06 / 88.40 |
| all-runs / every 10 | 90.96 / 91.03 | 90.19 / 90.80 | 88.20 / 88.98 |
| all-runs / once | 90.28 / 90.40 | 91.02 / 91.13 | **91.68 / 91.79** |
| normal-only / every 1 | 87.80 / 88.42 | **84.41 / 84.99** | 90.62 / 91.50 |
| normal-only / every 10 | 88.44 / 88.71 | 91.07 / 91.37 | 88.71 / 89.44 |
| normal-only / once | 88.03 / 88.67 | 89.01 / 89.73 | 89.67 / 90.19 |

all-runs once在三个seed间最平稳且没有seed 42下降；normal-only every 10仍有约2.6个百分点的
seed范围。每epoch策略在normal-only seed 2降至84.41%，是其participant SD和seed波动较大的来源。

### 21.6 Stage、immediate node与重复动作

四折最佳策略的Node Accuracy如下：

| Scope / 策略 | Stage 1 | Stage 2 | Stage 3 | Immediate targets | Non-immediate | 重复node组 |
|---|---:|---:|---:|---:|---:|---:|
| all-runs / once | 84.17 | **92.81** | 87.67 | **92.58** | 86.47 | **93.75** |
| normal-only / every 10 | 80.53 | 91.65 | 85.88 | 90.86 | 85.23 | 93.07 |

all-runs once在Stage 2和immediate-target上最强，方向上符合atomic-tail保护局部邻近顺序的动机；
Stage 1和non-immediate仍明显更难。四组已知重复动作node在all-runs once下仅有5次互相误判
（14↔21为2次、15↔22为0次、16↔19为2次、17↔20为1次），说明流程位置消歧基本保留，
但相对M2 Direct的总体净收益主要来自其他错误，而非只修复重复node对。

### 21.7 Atomic-tail失效分析

all-runs `refresh_once`四折总体最难的node及主要误判方向：

| Node | 动作 | Recall | 主要预测为 | 三seed错误次数 |
|---:|---|---:|---|---:|
| 34 | take lock from table | **60.00%** | 24 put sample on table | 20 |
| 1 | unlock crimper | 65.15% | 4 turn on crimper | 12 |
| 8 | turn on extractor fan | 75.76% | 28 turn off extractor fan | 13 |
| 6 | turn on air compressor | 77.27% | 30 turn off air compressor | 11 |
| 4 | turn on crimper | 78.79% | 1 unlock crimper | 7 |
| 35 | lock crimper | 80.00% | 32 turn off crimper | 7 |
| 24 | put sample on table | 82.20% | 12 take plier from table | **30** |

participant特异性非常清楚：A最难是node 24（Recall `47.22%`，主要24→12）；D是node 18
（`48.61%`，主要18→23）；J是node 34（`16.67%`，主要34→24）；M则集中在setup/shutdown，
node 1和4均为`53.33%`、node 7为`60.00%`。J的node 34只有8个唯一测试样本，百分比应结合
support解释；A-node24和D-node18的support较大，更接近稳定失效模式。

Atomic once共有62个唯一测试样本在三个seed中全部错误，其中A/D/J/M为`14/16/10/22`；
一致错误最多的node为24（10个）、34（7个）、12（6个）和18（5个）。499个样本-seed错误中，
`99.0%`为跨Tier3错误，平均错误置信度`80.8%`，`47.7%`错误置信度不低于0.9。normal-only最佳的
every 10共有74个三seed一致错误，且错误数589，高于all-runs once，进一步支持使用all-runs。

### 21.8 Atomic-tail结论

1. **实现与结果网格完整。**72个训练任务配置一致，平均71.4%训练样本触发tail，72个audit零违规。
2. **all-runs最佳策略是`refresh_once`，normal-only最佳是`refresh_every_10`。**每epoch刷新在两个
   scope都不是最佳，说明过强顺序随机化无益；但固定一次并非跨scope普适最优。
3. **Atomic取得当前最高四折均值。**all-runs once为`90.99/91.11%`，相对M2 Direct为
   `+0.42/+0.47`个百分点；normal-only every 10相对M2 Direct为`+0.76/+0.66`。
4. **均值领先不等于稳定领先。**all-runs Node配对仅6胜6负，且M折平均退化0.75个百分点；
   normal-only也存在A折退化。两个Node配对95%区间均跨0。
5. **本节结论已由真实顺序重评估复核。**测试history统一后的Atomic once仍保持最高均值，说明本节
   的均值优势不是测试重排造成；但跨配对仍不稳定，完整结论见第23节。

## 22. 全模型统一失效分析

### 22.1 分析口径

为了回答“每个人、所有人及不同模型分别错在哪”，本节统一使用`all_runs + test_all`：

- 四折模型：M0–M6、M1–M3 Direct、三个Dynamic模型、E2E Node Scratch和
  E2E Node From Tier3，共15个35-node模型、180份prediction；
- Atomic-tail：A/D/J/M、三seed、三刷新策略，共36份真实顺序`test_all` prediction；
- 合计216份prediction、`102,330`个样本-seed预测行；
- E2E Tier3 Scratch没有35-node输出，因此不进入node失效表；
- 所有模型均先在participant内平均seed，再对A/D/J/M等权；
- “错误次数”汇总三个seed，因此一个真实样本最多计3次；“Unique support”则不重复计seed。

### 22.2 从流程位置错误到视觉语义错误

四折模型的总体错误结构：

| 模型 | Node Acc | Node错误数 | 同Tier3错误占比 | 四组重复node双向错误 |
|---|---:|---:|---:|---:|
| M0 | 69.81 | 1698 | **45.8%** | **769** |
| E2E Node Scratch | 74.72 | 1405 | 30.0% | 414 |
| E2E Node From Tier3 | 77.74 | 1245 | 35.7% | 439 |
| M1 | 79.67 | 1120 | 25.4% | 280 |
| M1 Direct | 79.99 | 1106 | 25.1% | 273 |
| M4 | 83.01 | 940 | 9.5% | 87 |
| M6 | 83.71 | 894 | 8.1% | 69 |
| M5 | 83.78 | 892 | 7.2% | 62 |
| M2 | 84.15 | 872 | 5.3% | 45 |
| Dynamic Frozen-M0 Delta | 84.24 | 866 | 5.5% | 45 |
| M3 | 84.74 | 841 | 5.7% | 46 |
| Dynamic Joint-Head Delta | 85.38 | 805 | 4.6% | 37 |
| Dynamic Direct | 89.79 | 565 | 2.3% | 13 |
| M3 Direct | 90.05 | 552 | 2.2% | 12 |
| M2 Direct | **90.57** | **524** | **0.8%** | **4** |
| Atomic Direct / once / actual-order | **91.03** | **496** | 1.6% | 7 |

M0、E2E和M1主要仍在解决“同一动作属于哪个流程位置”；加入位置编码和graph history后，同Tier3
错误快速下降。M2 Direct和Atomic once之后，几乎所有剩余错误已经是跨Tier3视觉语义错误。因此继续只强化流程
位置约束的边际收益会变小，下一步更需要改进当前clip的时序与手-物状态表征。

### 22.3 每个模型总体最容易错的node及去向

下表列出每个四折模型Recall最低的三个node。“真→预测”为该node最常见错误去向：

| 模型 | 最难node 1 | 最难node 2 | 最难node 3 |
|---|---|---|---|
| M0 | 16→19（37.1%） | 15→22（37.7%） | 20→17（38.8%） |
| E2E Node Scratch | 21→14（44.1%） | 15→22（45.9%） | 22→15（49.4%） |
| E2E Node From Tier3 | 22→15（44.0%） | 15→22（51.9%） | 21→14（57.3%） |
| M1 | 20→17（47.1%） | 16→19（57.7%） | 19→16（61.8%） |
| M1 Direct | 20→17（41.9%） | 19→16（47.0%） | 34→24（58.2%） |
| M2 | 34→24（66.4%） | 20→19（68.6%） | 16→17（68.9%） |
| M3 | 34→24（64.3%） | 20→19（65.9%） | 16→20（70.6%） |
| M4 | 34→24（63.5%） | 1→35（63.9%） | 20→16（67.9%） |
| M5 | 20→19（63.4%） | 34→24（63.5%） | 16→20（69.8%） |
| M6 | 34→24（65.6%） | 20→16（66.4%） | 35→1（68.1%） |
| Dynamic Frozen | 34→24（63.5%） | 20→19（65.3%） | 16→17（69.1%） |
| Dynamic Joint | 1→35（65.6%） | 34→24（70.3%） | 20→19（70.8%） |
| Dynamic Direct | 1→4（62.2%） | 34→24（70.9%） | 4→1（74.2%） |
| M3 Direct | 1→35（64.4%） | 34→24（73.0%） | 7→29（75.3%） |
| M2 Direct | 1→35（62.5%） | 34→24（64.6%） | 4→1（75.8%） |
| Atomic Direct / once / actual-order | 34→24（61.3%） | 1→4（65.2%） | 8→28（77.3%） |

主要node含义：

- 14/21都是`place sample under electrodes`；
- 15/22都是`press pedal`；
- 16/19都是`put sample on machine table`；
- 17/20都是`grip sample from machine table`；
- 1/35分别是`unlock/lock crimper`；
- 34/24分别是`take lock/put sample on table`；
- 6/30、7/29、8/28分别是三类设备的`turn on/turn off`。

这张表显示模型演化非常清楚：M0/E2E/M1最难的是重复流程node；M2–M6转向机器台上的放置/抓取和
table上的物体混淆；Direct模型进一步把难点压缩到低support的开关/锁状态及node 34。

### 22.4 所有人合并后的主要有向混淆

| 模型 | 高频真实→预测 | 三seed错误次数 | 特性 |
|---|---|---:|---|
| M0 | 15→22 | 136 | 同Tier3、重复press pedal位置 |
| M0 | 14→21 | 122 | 同Tier3、重复place sample位置 |
| M0 | 16→19 | 111 | 同Tier3、重复put sample位置 |
| M3 | 20→19 | 47 | grip与put、动作方向不同 |
| M3 | 24→12 | 36 | sample与plier、物体和动作均不同 |
| M2 Direct | 20→19 | 38 | grip sample→put sample |
| M2 Direct | 24→12 | 33 | put sample→take plier |
| Dynamic Direct | 20→19 | 24 | grip sample→put sample |
| Dynamic Direct | 24→12 | 23 | put sample→take plier |
| Dynamic Direct | 24→25 | 22 | put sample→put plier |
| Atomic once actual-order（四折） | 24→12 | 30 | A主导的table区域物体混淆 |
| Atomic once actual-order（四折） | 19→20 | 21 | put sample→grip sample |
| Atomic once actual-order（四折） | 34→24 | 19 | J主导的lock→sample物体混淆 |
| Atomic once actual-order（四折） | 24→25 | 16 | sample→plier物体混淆 |
| Atomic once actual-order（四折） | 18→23 | 13 | reverse sample→inspect sample |

### 22.5 Participant A

| 模型 | Node Acc | Recall最低的三个node及主要去向 |
|---|---:|---|
| M0 | 65.27 | 22→23（17.5%）；16→19（17.5%）；20→17（23.8%） |
| M3 | 79.27 | 24→12（33.3%）；22→21（55.6%）；20→18（58.7%） |
| M2 Direct | 89.56 | 24→12（40.3%）；30→6（55.6%）；1→4（66.7%） |
| M3 Direct | 89.25 | 24→12（40.3%）；30→6（61.1%）；1→4（66.7%） |
| Dynamic Direct | 87.63 | 24→12（36.1%）；30→6（50.0%）；1→4（66.7%） |
| Atomic once / actual-order | **90.72** | 24→12（47.2%）；1→4（66.7%）；8→6（77.8%） |

A是最明显的node 24困难折：该node有24个唯一测试样本，所有history/direct模型都主要把
`put sample on table`错成node 12 `take plier from table`。这不是小样本偶然；它反映A在table区域
操作时，sample、plier和hand trajectory的视觉特征高度相似。Atomic once改善但没有消除该问题。
A的次要错误是设备关闭/开启及unlock/turn-on边界，说明setup/shutdown阶段的状态变化仍不充分。

### 22.6 Participant D

| 模型 | Node Acc | Recall最低的三个node及主要去向 |
|---|---:|---|
| M0 | 69.55 | 19→16（27.8%）；16→20（37.5%）；18→23（38.9%） |
| M3 | 81.53 | 16→20（41.7%）；18→23（48.6%）；20→19（50.0%） |
| M2 Direct | 88.53 | 18→23（50.0%）；6→30（55.6%）；35→32（60.0%） |
| M3 Direct | 86.65 | 35→32（46.7%）；2→34（50.0%）；20→19（58.3%） |
| Dynamic Direct | 87.37 | 18→23（41.7%）；6→30（55.6%）；35→32（60.0%） |
| Atomic once / actual-order | **89.03** | 18→23（48.6%）；6→30（55.6%）；35→32（60.0%） |

D的系统性弱点是node 18 `reverse sample`被预测为node 23 `inspect sample`，该node有24个唯一样本，
比设备开关类低support错误更可信。两者都围绕sample、手部移动幅度小且可能处于相邻流程区间；
单帧/短clip很容易只捕获“拿着并观察样品”的共同外观。node 6→30和35→32则属于设备on/off或
lock/turn-off状态方向混淆。

### 22.7 Participant J

| 模型 | Node Acc | Recall最低的三个node及主要去向 |
|---|---:|---|
| M0 | 74.89 | 14→21（8.9%）；15→22（11.1%）；34→24（16.7%） |
| M3 | 92.49 | 34→24（20.8%）；1→35（60.0%）；20→19（63.4%） |
| M2 Direct | 94.47 | 34→24（16.7%）；1→35（60.0%）；31→9（70.8%） |
| M3 Direct | **94.59** | 34→24（29.2%）；1→35（66.7%）；4→1（73.3%） |
| Dynamic Direct | **94.59** | 34→24（20.8%）；1→4（60.0%）；31→35（66.7%） |
| Atomic once / actual-order | **95.56** | 34→24（16.7%）；1→4（53.3%）；31→27（62.5%） |

J总体最容易，Atomic once actual-order相对M2 Direct平均提高1.08个百分点；但node 34 `take lock from table`
始终非常困难，主要错成node 24 `put sample on table`。
该node在J只有8个唯一样本，Recall因此很容易被一两个样本显著改变；不能据此说J整体很差。
J的M0错误几乎完全由重复流程node驱动，history一加入就从74.89%跃升至92%以上，是task graph
流程消歧最典型的受益者。

### 22.8 Participant M

| 模型 | Node Acc | Recall最低的三个node及主要去向 |
|---|---:|---|
| M0 | 69.50 | 19→16（13.0%）；22→15（31.9%）；17→16（34.8%） |
| M3 | 85.68 | 4→32（53.3%）；17→16（56.5%）；7→29（60.0%） |
| M2 Direct | **89.71** | 1→35（40.0%）；4→1（46.7%）；7→29（60.0%） |
| M3 Direct | **89.71** | 4→1（46.7%）；1→35（46.7%）；7→29（53.3%） |
| Dynamic Direct | 89.56 | 1→4（33.3%）；4→1（46.7%）；7→29（60.0%） |
| Atomic once / actual-order | 88.81 | 1→35（53.3%）；4→1（53.3%）；7→29（60.0%） |

M在history解决重复node后，剩余困难集中于setup/shutdown设备状态：unlock、turn on crimper、
turn on/off water pump。这些node各只有约5个唯一样本，低support和短暂状态切换共同造成较大seed
波动。M的Stage 2主体动作已较稳定，Atomic once actual-order相对M2 Direct反而低0.89个百分点，新增历史排列
没有给M带来平均收益。

### 22.9 错误node的共同特性

1. **同动作、不同流程位置。**
   这是M0/E2E/M1的主错误，典型为14↔21、15↔22、16↔19、17↔20。位置编码和history几乎解决。
2. **相反设备状态。**
   6↔30、7↔29、8↔28，以及1/35/4/32间的混淆，通常发生在开关动作短、设备外观变化不明显时。
3. **相同空间区域、不同物体。**
   node 24常被预测为12、25或34：都发生在table附近，但涉及sample、plier、lock不同物体。
4. **相同物体、相反操作方向。**
   16/19的put sample与17/20的grip sample共享machine table和sample，只依赖手运动方向区分。
5. **短时语义边界。**
   node 18 reverse与23 inspect、15 press pedal与14 place sample相邻且动作短，clip可能覆盖过渡帧。
6. **低support类别。**
   node 1、4、6–8、30、31、34、35在部分participant只有5–8个唯一样本，Recall和seed差异更大。
7. **跨参与者域偏移。**
   A的核心问题是node 24，D是node 18，J是node 34，M是setup/shutdown设备状态；统一模型无法用
   单一错误模式概括所有人。
8. **高置信错误。**
   M2 Direct、M3 Direct、Dynamic Direct和Atomic once actual-order仍分别有约48.1%、51.3%、46.4%和46.8%的
   错误置信度≥0.9。最大softmax不能可靠识别失败。

### 22.10 针对失效模式的下一步建议

1. **同时报告M2 Direct与Atomic once actual-order。**Atomic once平均高0.46个百分点，但只有
   6/12个Node配对提高；真实顺序测试已排除协议差异，剩余不确定性来自participant与seed稳定性。
2. **Atomic Direct网格已经补齐，无需再补J/M。**如果继续投入，应转向冻结/联合head的Atomic结构
   消融；M2与Atomic的真实测试history已经统一，无需重复该比较，也不建议继续投入every-1策略。
3. **为node 24、12、25、34加入物体感知。**优先考虑hand-object crop、object token或桌面区域检测。
4. **为on/off和grip/put加入短时运动信息。**使用更长clip、双向帧差、optical flow或动作前后状态差。
5. **针对node 18/23增加动作边界监督。**区分reverse与inspect需要覆盖完整手部轨迹，而非单个稳定姿态。
6. **报告低support类别时同时给support。**避免把J-node34或M-node1等小样本Recall当成稳定总体结论。
7. **部署时加入结构化不确定性。**结合graph合法性、持续时间、预测跳变与多seed分歧，而非只设
   softmax阈值。

## 23. Atomic-tail真实历史顺序重评估（2026-08-05）

### 23.1 评估目的、协议与完整性

本次不重新训练模型，而是读取第21节已经训练完成的72个Atomic-tail Direct Fusion checkpoint，
将测试阶段history从固定seeded atomic-tail重排改为样本的真实时间顺序
`actual_chronological`。训练阶段仍保持各自的Atomic-tail策略，因此本节回答的是：

> 在训练增强不变的条件下，使用与M2 Direct相同的真实测试顺序，Atomic-tail是否仍有收益？

评估覆盖：

```text
4 participants × 3 seeds × 2 train scopes × 3 refresh policies × 3 splits = 216 results
```

72/72个checkpoint均完成评估；每个任务均产生`test_normal`、`test_fault`和`test_all`的metrics与
prediction，并独立保存在原模型目录的`test_results_actual_order`中，没有覆盖第21节的原测试结果。
汇总文件位于`outputs/at_actual`：

- `atomic_actual_order_metrics.csv`：216条逐participant-seed严格配对明细；
- `atomic_actual_order_seed_aggregate.csv`：54条按seed汇总；
- `atomic_actual_order_participant_aggregate.csv`：72条按participant汇总；
- `atomic_actual_order_aggregate.csv`：18条四折总体汇总；
- `completed.json`：网格完整性与`actual_chronological`协议标记。

### 23.2 仅改变测试顺序带来的影响

下表比较同一checkpoint在原固定Atomic-tail测试顺序与真实测试顺序下的`test_all`结果，单位为
百分点，均采用participant-first聚合：

| Train scope | 训练刷新策略 | 原Node | 真实顺序Node | ΔNode | 原Tier3 | 真实顺序Tier3 | ΔTier3 |
|---|---|---:|---:|---:|---:|---:|---:|
| all-runs | every 1 | 89.119 | 89.184 | +0.065 | 89.343 | 89.445 | +0.103 |
| all-runs | every 10 | 89.784 | 89.728 | -0.056 | 90.272 | 90.290 | +0.018 |
| all-runs | once | 90.992 | 91.031 | +0.038 | 91.105 | 91.181 | +0.076 |
| normal-only | every 1 | 87.611 | 87.430 | -0.181 | 88.305 | 88.089 | -0.216 |
| normal-only | every 10 | 89.405 | 89.415 | +0.010 | 89.842 | 89.852 | +0.010 |
| normal-only | once | 88.904 | 88.904 | 0.000 | 89.528 | 89.532 | +0.004 |

全部变化都小于0.22个百分点，两个scope的最佳策略也没有改变。对主配置`all-runs + once`，
5685个样本-seed预测中只有39个Node预测发生改变：17个由错变对、14个由对变错、8个错到另一错误，
净增加3个正确预测。说明Atomic模型对这两种合法测试history较鲁棒；其平均优势不是测试阶段采用
Atomic-tail重排人为造成的。

### 23.3 与M2 Direct统一测试协议后的总体比较

#### 23.3.1 六种训练配置的test_all结果

| Train scope | Atomic训练刷新 | Atomic Node/Tier3 | M2 Direct Node/Tier3 | ΔNode/ΔTier3 |
|---|---|---:|---:|---:|
| all-runs | every 1 | 89.18 / 89.45 | 90.57 / 90.64 | -1.38 / -1.19 |
| all-runs | every 10 | 89.73 / 90.29 | 90.57 / 90.64 | -0.84 / -0.35 |
| all-runs | once | **91.03 / 91.18** | 90.57 / 90.64 | **+0.46 / +0.54** |
| normal-only | every 1 | 87.43 / 88.09 | 88.64 / 89.18 | -1.21 / -1.09 |
| normal-only | every 10 | **89.42 / 89.85** | 88.64 / 89.18 | **+0.77 / +0.67** |
| normal-only | once | 88.90 / 89.53 | 88.64 / 89.18 | +0.26 / +0.35 |

真实顺序测试保持了原刷新频率结论：all-runs最佳为`refresh_once`，normal-only最佳为
`refresh_every_10`，每epoch刷新在两个scope都最差或接近最差。

#### 23.3.2 两个scope最佳配置的三个split

| Scope / Atomic策略 | Split | Atomic Node/Tier3 | M2 Direct Node/Tier3 | ΔNode/ΔTier3 |
|---|---|---:|---:|---:|
| all-runs / once | normal | 91.69 / 91.80 | 91.26 / 91.30 | +0.43 / +0.50 |
| all-runs / once | fault | 90.19 / 90.37 | 89.75 / 89.86 | +0.44 / +0.51 |
| all-runs / once | all | **91.03 / 91.18** | 90.57 / 90.64 | **+0.46 / +0.54** |
| normal-only / every 10 | normal | 91.20 / 91.39 | 90.55 / 90.70 | +0.65 / +0.68 |
| normal-only / every 10 | fault | 84.28 / 85.75 | 83.24 / 85.05 | +1.05 / +0.70 |
| normal-only / every 10 | all | **89.42 / 89.85** | 88.64 / 89.18 | **+0.77 / +0.67** |

均值上，两个最佳Atomic配置在normal、fault和all三个split均超过M2 Direct；但这只是四折等权均值，
不能替代下面的participant-seed配对稳定性检查。

### 23.4 Participant差异

`test_all`逐participant三seed均值如下：

| Scope / Atomic策略 | Participant | Atomic Node/Tier3 | M2 Direct Node/Tier3 | ΔNode/ΔTier3 |
|---|---|---:|---:|---:|
| all-runs / once | A | 90.72 / 91.03 | 89.56 / 89.64 | +1.16 / +1.39 |
| all-runs / once | D | 89.03 / 89.18 | 88.53 / 88.67 | +0.51 / +0.51 |
| all-runs / once | J | 95.56 / 95.56 | 94.47 / 94.53 | +1.08 / +1.02 |
| all-runs / once | M | 88.81 / 88.96 | 89.71 / 89.71 | -0.89 / -0.75 |
| normal-only / every 10 | A | 89.02 / 89.25 | 89.95 / 90.33 | -0.93 / -1.08 |
| normal-only / every 10 | D | 87.88 / 88.53 | 87.81 / 88.46 | +0.07 / +0.07 |
| normal-only / every 10 | J | 93.81 / 94.23 | 92.55 / 93.21 | +1.26 / +1.02 |
| normal-only / every 10 | M | 86.95 / 87.40 | 84.27 / 84.71 | +2.68 / +2.68 |

all-runs once在A、D、J提高，但M下降；normal-only every 10则主要由J、M拉高，A下降。两个scope的
四折正均值都不是所有participant共同改善造成的。

### 23.5 Seed与严格配对稳定性

`test_all`四人等权的逐seed结果：

| Scope / Atomic策略 | Seed | Atomic Node/Tier3 | M2 Direct Node/Tier3 | ΔNode/ΔTier3 |
|---|---:|---:|---:|---:|
| all-runs / once | 1 | 90.15 / 90.27 | 92.28 / 92.33 | -2.13 / -2.06 |
| all-runs / once | 2 | 91.28 / 91.45 | 90.85 / 90.90 | +0.43 / +0.54 |
| all-runs / once | 42 | 91.66 / 91.83 | 88.57 / 88.68 | +3.09 / +3.15 |
| normal-only / every 10 | 1 | 88.83 / 89.10 | 87.68 / 88.29 | +1.15 / +0.81 |
| normal-only / every 10 | 2 | 90.92 / 91.23 | 89.42 / 89.70 | +1.50 / +1.53 |
| normal-only / every 10 | 42 | 88.49 / 89.23 | 88.82 / 89.55 | -0.33 / -0.32 |

12个participant-seed严格配对中：

- all-runs once的Node和Tier3均为`6胜/0平/6负`；Node均值差`+0.46`，描述性95%区间
  `[-1.45, +2.37]`；
- normal-only every 10的Node和Tier3均为`6胜/1平/5负`；Node均值差`+0.77`，描述性95%区间
  `[-0.80, +2.34]`；
- all-runs once在seed 1明显低于M2、seed 42明显高于M2，证明模型与seed存在较强交互。

因此，统一测试协议强化了“Atomic平均值略高”的可信度，却没有强化“Atomic稳定更高”的证据。

### 23.6 真实顺序下的失效模式

all-runs once真实顺序测试共有5685个样本-seed预测、496次Node错误；participant-first Node Accuracy
为91.03%。其中8次（1.6%）属于同Tier3错误，四组重复流程node双向错误共7次，说明流程位置消歧
仍基本保留。错误平均置信度为80.5%，232/496（46.8%）错误置信度不低于0.9。

最难node和主要去向为：node 34→24（Recall 61.3%）、1→4（65.2%）、8→28（77.3%）、
4→1（78.8%）、6→30（78.8%）及24→12（82.2%）。最高频错误仍是24→12（30次）、
19→20（21次）、34→24（19次）、24→25（16次）和18→23（13次），与第22节总结的物体、
动作方向、开关状态和短时语义边界失效一致。

三个seed全部错误的唯一样本共有63个，A/D/J/M分别为17/17/8/21个；最集中的真实node为24
（10个）、34（7个）和18（5个）。因此真实测试顺序并未改变主要失效机制，只在少量边界样本上改变预测。

### 23.7 更新后的Atomic-tail结论

1. **真实顺序重评估完整且没有覆盖原结果。**72个checkpoint、216个split全部完成，原固定
   Atomic-tail测试协议仍可复现。
2. **测试顺序影响很小。**六种配置的`test_all` Node变化均不超过0.19个百分点，最佳策略不变。
3. **Atomic once保持最高四折均值。**all-runs once达到`91.03/91.18%` Node/Tier3 Accuracy，
   相对M2 Direct为`+0.46/+0.54`；这是一项统一测试协议后的真实平均收益。
4. **均值优势仍不稳定。**all-runs为6胜6负且M下降0.89个百分点；normal-only也存在A下降，两个
   Node配对区间均跨0。
5. **最终推荐表述：**M2 Direct仍是更简单且跨seed行为更容易解释的主基线；Atomic all-runs once
   是当前最高四折均值配置，应作为有希望的训练增强共同报告，但不能宣称稳定显著优于M2 Direct。
