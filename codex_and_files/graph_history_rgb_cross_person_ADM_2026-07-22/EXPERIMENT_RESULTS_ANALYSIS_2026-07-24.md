# RGB Task-Graph History 多轮实验结果分析

日期：2026-07-24  
相机：`001484412812`  
任务：35-node分类，同时将35-node概率聚合为31类Tier3结果

## 1. 分析范围

本报告读取并比较以下两个实验包中的实际结果文件，未重新训练模型，也未修改已有结果：

```text
D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\graph_history_rgb_experiments_2026-07-20
D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\graph_history_rgb_cross_person_ADM_2026-07-22
```

纳入分析的实验包括：

1. J-as-test先导实验：normal-only history models，M0–M6，3个seed；
2. A/D/M跨人normal-only实验：M0–M6和3个E2E对照，目前只有`seed_1`；
3. A/D/M跨人完整all-runs实验：M0–M6和3个E2E对照，`seed_1`、`seed_2`、`seed_42`；
4. normal-only与all-runs在相同`seed_1`下的配对比较；
5. normal、fault和all三个测试划分；
6. Stage 1、Stage 2和Stage 3分阶段结果；
7. 35-node混淆矩阵和重复Tier3动作对应的node混淆。

### 1.1 模型配置

| 模型 | 历史信息与图信息 |
|---|---|
| M0 | 仅当前clip的冻结RGB特征，不使用历史 |
| M1 | 使用同run历史，但没有位置编码 |
| M2 | 使用真实发生顺序的历史和位置编码 |
| M3 | 使用task graph允许的顺序重排历史 |
| M4 | candidate history attention，不使用graph relation bias |
| M5 | 使用真实历史node标签生成oracle graph relation bias |
| M6 | 使用冻结M0预测的历史node概率生成soft graph relation bias |
| E2E-Tier3-Scratch | 直接从RGB端到端预测31类Tier3 |
| E2E-Node-Scratch | 从scratch端到端预测35-node |
| E2E-Node-From-Tier3 | 由Tier3 backbone初始化并端到端预测35-node |

### 1.2 测试规模和类别覆盖

| Held-out参与者 | test normal | test fault | test all | fault node覆盖 | fault Tier3覆盖 |
|---|---:|---:|---:|---:|---:|
| J | 387 | 168 | 555 | 33/35 | 29/31 |
| A | 294 | 137 | 431 | 35/35 | 31/31 |
| D | 400 | 62 | 462 | 30/35 | 26/31 |
| M | 360 | 87 | 447 | 34/35 | 30/31 |

因此，fault macro-F1在不同参与者之间不是完全同一组类别上的平均。D的fault集合尤其小，
只有62个clip，而且缺失5个node和5个Tier3类别。分析fault结果时应同时查看accuracy、
macro-F1和类别覆盖。

## 2. 可比性和解释边界

### 2.1 J与A/D/M的绝对结果不能直接视为严格四折LOSO

J先导实验使用已有Tier3 `last.pth`抽取特征，而A/D/M normal-only实验为每个fold从scratch
重新训练backbone。J实验包还记录了两个重要限制：

1. 现有J checkpoint训练时曾将J test manifest用作validation；
2. 该backbone见过A/D/M的fault runs。

因此，J的绝对准确率主要用于验证方法是否具有潜力，不应与A/D/M scratch backbone结果直接
合并为严格四人LOSO均值。更可靠的是观察每个实验内部相对于M0的增益方向是否一致。

### 2.2 normal-only与all-runs的多seed数量不对称

- A/D/M normal-only：只有`seed_1`；
- A/D/M完整all-runs：有`seed_1`、`seed_2`、`seed_42`。

因此，训练范围的严格公平比较只能使用同一`seed_1`。all-runs三个seed的均值和标准差可用于
评估all-runs自身的稳定性，但不能把它与单个normal-only seed当作完全对称的多seed比较。

### 2.3 指标聚合方式

- J表格：直接在3个seed之间计算均值和样本标准差；
- A/D/M all-runs跨人表格：先在每位参与者内部平均3个seed，再在A/D/M之间计算均值和样本标准差；
- paired training-scope表格：对A、D、M各自的相同`seed_1`计算
  `all-runs - normal-only`，再跨3位参与者取均值和样本标准差；
- 多个seed不是新的独立参与者，因此不将9个fold-seed结果当作9个独立受试者做显著性检验。

报告中的数值均转换为百分比；“+1.00”表示提高1个百分点。

## 3. J-as-test先导实验

### 3.1 test_all三seed结果

| 模型 | Node Acc | Node Macro-F1 | Tier3 Acc | Tier3 Macro-F1 |
|---|---:|---:|---:|---:|
| M0 | 75.20 ± 1.02 | 75.95 ± 0.36 | 86.79 ± 0.21 | 82.34 ± 0.46 |
| M1 | 74.89 ± 2.71 | 73.80 ± 1.45 | 82.76 ± 1.98 | 76.72 ± 1.09 |
| M2 | 88.77 ± 0.55 | 86.29 ± 1.11 | 89.43 ± 0.58 | 86.29 ± 0.99 |
| **M3** | **89.97 ± 0.28** | **87.69 ± 0.98** | **90.27 ± 0.18** | 87.50 ± 1.14 |
| M4 | 87.57 ± 1.00 | 84.56 ± 2.38 | 88.11 ± 1.25 | 84.48 ± 2.65 |
| **M5** | 89.49 ± 1.10 | 87.64 ± 1.56 | **90.27 ± 1.00** | **87.71 ± 1.88** |
| M6 | 87.09 ± 0.73 | 83.72 ± 2.26 | 87.93 ± 0.79 | 83.70 ± 2.60 |

主要观察：

1. M2–M6都明显提高35-node accuracy，证明历史信息的主要价值不是简单复制当前clip分类；
2. M3获得最高node accuracy、node macro-F1和并列最高Tier3 accuracy；
3. M5的Tier3 macro-F1最高，且node结果与M3非常接近；
4. M1没有位置编码，在J上整体低于M0，说明“把历史特征直接放入attention”不够；
5. M6明显低于M3和M5，说明soft relation bias尚未达到oracle关系或graph-valid history的效果。

### 3.2 相对于M0的test_all提升

| 模型 | Δ Node Acc | Δ Node Macro-F1 | Δ Tier3 Acc | Δ Tier3 Macro-F1 |
|---|---:|---:|---:|---:|
| M1 | -0.30 | -2.15 | -4.02 | -5.62 |
| M2 | +13.57 | +10.35 | +2.64 | +3.95 |
| **M3** | **+14.77** | **+11.74** | **+3.48** | +5.15 |
| M4 | +12.37 | +8.61 | +1.32 | +2.13 |
| M5 | +14.29 | +11.70 | +3.48 | **+5.37** |
| M6 | +11.89 | +7.77 | +1.14 | +1.36 |

M3与M5在三个seed中都表现出大幅node提升，并且seed标准差较小。这构成了后续A/D/M
跨人实验的主要依据。

### 3.3 J的分阶段现象

| 模型 | Stage 1 Node Acc | Stage 2 Node Acc | Stage 3 Node Acc |
|---|---:|---:|---:|
| M0 | 80.61 | 74.06 | 77.63 |
| M1 | 86.06 | 76.02 | **60.53** |
| M2 | 86.06 | 90.64 | 80.26 |
| M3 | 84.85 | **91.75** | **83.77** |
| M4 | 81.21 | 89.70 | 80.26 |
| M5 | **87.27** | 90.80 | **83.77** |
| M6 | 81.21 | 89.07 | 80.26 |

M1在Stage 3降到60.53%，解释了其整体失败。加入位置或task-graph结构后，M2–M6在
Stage 2均接近或超过89%。这说明历史特征必须带有“历史中处于什么位置/关系”的信息，
否则attention容易学习不稳定的历史捷径。

## 4. A/D/M normal-only结果

本节只有`seed_1`。表中“±”表示A/D/M之间的标准差，不是seed标准差。

### 4.1 test_all结果

| 模型 | Node Acc | Node Macro-F1 | Tier3 Acc | Tier3 Macro-F1 |
|---|---:|---:|---:|---:|
| M0 | 63.87 ± 2.44 | 65.42 ± 2.48 | 73.29 ± 2.44 | 72.47 ± 0.71 |
| M1 | 74.96 ± 1.78 | **76.71 ± 1.99** | 76.70 ± 1.38 | **79.54 ± 2.67** |
| M2 | 73.96 ± 3.27 | 73.99 ± 1.99 | 76.22 ± 2.33 | 76.54 ± 0.54 |
| M3 | 74.84 ± 1.03 | 76.13 ± 0.29 | 76.17 ± 1.05 | 78.42 ± 0.41 |
| **M4** | **75.95 ± 1.20** | 76.03 ± 3.13 | 78.07 ± 1.31 | 78.05 ± 4.72 |
| M5 | 74.82 ± 1.37 | 73.84 ± 1.37 | 76.56 ± 2.03 | 75.58 ± 2.46 |
| M6 | 73.17 ± 3.23 | 72.85 ± 1.97 | 76.01 ± 1.75 | 75.22 ± 1.72 |
| E2E-Node-Scratch | 69.12 ± 0.64 | 71.04 ± 3.02 | 75.01 ± 2.69 | 75.86 ± 2.91 |
| E2E-Node-From-Tier3 | 73.03 ± 1.78 | 72.48 ± 0.07 | **79.82 ± 1.55** | 76.71 ± 1.75 |
| E2E-Tier3-Scratch | — | — | 76.47 ± 3.59 | 75.14 ± 0.57 |

normal-only下，历史模型相对M0仍然有明显node增益：

- M1：+11.09 node accuracy；
- M2：+10.08；
- M3：+10.96；
- M4：+12.08；
- M5：+10.95；
- M6：+9.30。

但模型排序与J不同：M4的node accuracy最高，M1的macro-F1最高。这提示单个seed下的模型排序
可能受到初始化影响，也可能说明normal-only训练对fault run泛化不稳定。

尤其需要注意：

- M1在test_normal上node增益很大，但test_fault增益只有约+1.43；
- M1在test_fault的Tier3 accuracy反而比M0低约3.29；
- M4在normal-only下是最好的node accuracy配置，但在完整all-runs中被M3超过。

## 5. A/D/M完整all-runs结果

每位参与者使用3个seed平均，然后在A/D/M之间汇总。“±”表示参与者间标准差。

### 5.1 test_all总体结果

| 模型 | Node Acc | Node Macro-F1 | Tier3 Acc | Tier3 Macro-F1 |
|---|---:|---:|---:|---:|
| M0 | 68.11 ± 2.46 | 71.48 ± 2.50 | 80.64 ± 2.23 | 79.57 ± 1.87 |
| M1 | 76.04 ± 3.20 | 78.16 ± 2.26 | 82.35 ± 2.58 | 82.54 ± 1.53 |
| M2 | 81.35 ± 2.45 | 80.83 ± 1.44 | 82.37 ± 2.65 | 81.80 ± 1.26 |
| **M3** | **82.16 ± 3.25** | **81.97 ± 1.96** | **83.33 ± 2.62** | **83.03 ± 1.23** |
| M4 | 80.53 ± 2.95 | 79.89 ± 2.57 | 82.15 ± 2.36 | 81.22 ± 2.12 |
| M5 | 80.81 ± 1.85 | 81.33 ± 0.49 | 82.41 ± 1.89 | 82.84 ± 0.96 |
| M6 | 80.72 ± 4.37 | 80.19 ± 3.21 | 82.49 ± 3.23 | 81.50 ± 2.41 |
| E2E-Node-Scratch | 71.40 ± 0.86 | 71.81 ± 1.31 | 78.18 ± 1.72 | 76.40 ± 0.41 |
| E2E-Node-From-Tier3 | 75.45 ± 2.50 | 76.39 ± 1.26 | 82.98 ± 0.78 | 80.84 ± 1.00 |
| E2E-Tier3-Scratch | — | — | 82.54 ± 1.62 | 80.63 ± 1.67 |

M3是完整all-runs条件下最平衡的配置：

- test_all四个主要指标全部第一；
- node accuracy比M0高14.05个百分点；
- Tier3 accuracy比M0高2.69个百分点；
- node accuracy比E2E-Node-From-Tier3高6.71个百分点；
- Tier3 accuracy比E2E-Node-From-Tier3高0.35个百分点；
- Tier3 macro-F1比E2E-Node-From-Tier3高2.19个百分点。

### 5.2 三个测试划分中的最佳配置

| Split | 最佳Node Acc | 最佳Node Macro-F1 | 最佳Tier3 Acc | 最佳Tier3 Macro-F1 |
|---|---|---|---|---|
| normal | M3, 82.80 | M3, 82.21 | M3, 83.98 | M3, 83.23 |
| fault | M3, 82.78 | M3, 81.10 | E2E-Node-From-Tier3, 84.77 | M5, 81.97 |
| all | M3, 82.16 | M3, 81.97 | M3, 83.33 | M3, 83.03 |

fault上有一个重要例外：E2E-Node-From-Tier3的Tier3 accuracy为84.77%，比M3的83.56%
高1.21个百分点。但M3的node accuracy仍高5.09个百分点，Tier3 macro-F1也高约0.60个百分点。
这说明E2E transfer在fault数据上更擅长预测聚合Tier3类别，但不能同样准确地区分35个graph node。

### 5.3 相对于M0的all-runs提升

下表在9个participant-seed配对中直接计算模型减M0。`正向次数`表示9个配对中严格大于0的次数；
这些配对不是9个独立受试者，因此只用于稳定性描述。

| 模型 | Δ Node Acc | Node正向次数 | Δ Tier3 Acc | Tier3正向次数 |
|---|---:|---:|---:|---:|
| M1 | +7.93 | 8/9 | +1.71 | 7/9 |
| M2 | +13.24 | **9/9** | +1.73 | 7/9 |
| **M3** | **+14.05** | **9/9** | **+2.69** | **9/9** |
| M4 | +12.42 | **9/9** | +1.51 | 8/9 |
| M5 | +12.70 | **9/9** | +1.78 | **9/9** |
| M6 | +12.61 | **9/9** | +1.85 | 8/9 |

M3是唯一同时满足以下条件的配置：

1. 9/9配对node accuracy全部提高；
2. 9/9配对Tier3 accuracy全部提高；
3. 平均node增益最大；
4. 平均Tier3增益最大。

这比只比较最终平均准确率更有说服力。

## 6. 真实顺序、graph-valid重排和relation bias消融

### 6.1 M2与M3：真实精确顺序不是必要条件

在完整all-runs的test_all上：

| 比较 | Δ Node Acc | Δ Tier3 Acc | Δ Tier3 Macro-F1 |
|---|---:|---:|---:|
| M3 − M2 | +0.81 | +0.96 | +1.23 |

M3在9个participant-seed配对中：

- node accuracy有7/9高于M2；
- Tier3 accuracy有7/9高于M2；
- Tier3 macro-F1有9/9高于M2。

这与研究假设一致：模型不一定需要复现同一操作者的精确历史顺序；只要历史集合满足task graph
允许的相对关系，仍可获得有效上下文，甚至比实际观测顺序更稳定。该结果也适合解释多人协作场景：
之前的机器准备动作可能由其他人完成，但它们仍然是当前动作的合法历史。

### 6.2 M4、M5与M6：graph relation bias有小幅收益，但soft版本尚未成为最佳

test_all平均结果：

| 比较 | Δ Node Acc | Δ Tier3 Acc | Δ Tier3 Macro-F1 |
|---|---:|---:|---:|
| M5 oracle relation − M4 no relation | +0.29 | +0.26 | **+1.63** |
| M6 soft relation − M4 no relation | +0.20 | +0.34 | +0.29 |
| M6 soft relation − M5 oracle relation | -0.09 | +0.08 | **-1.34** |
| M3 graph-valid history − M6 soft relation | +1.44 | +0.84 | +1.53 |

结论是：

- oracle relation bias确实提供额外信息，尤其提高Tier3 macro-F1；
- M6的soft relation在accuracy上略高于M4，但提升很小；
- M6没有超过M3，也没有稳定达到M5的macro-F1；
- 当前结果还不支持“soft graph relation bias是最优模型”的结论；
- 更稳妥的论文表述是：graph约束历史重排M3目前表现最好，oracle relation M5证明关系类型有潜力，
  但从M0概率构造的soft relation仍需要改进。

### 6.3 M1与M2：位置主要改善node消歧，而不是Tier3外观分类

test_all上，M2相对M1：

```text
Node accuracy：+5.32个百分点
Tier3 accuracy：+0.02个百分点
Tier3 macro-F1：-0.74个百分点
```

位置编码几乎不改变Tier3 accuracy，却大幅提高35-node accuracy。这是一个非常关键的结果：
位置/顺序主要帮助识别“当前视觉动作对应graph中的哪一个node”，而不是识别动作外观本身。

## 7. Stage分析：历史收益主要来自Stage 2 node消歧

### 7.1 all-runs test_all分阶段结果

| 模型 | Stage 1 Node | Stage 2 Node | Stage 3 Node | Stage 2 Tier3 |
|---|---:|---:|---:|---:|
| M0 | 79.46 | 63.45 | 80.87 | 80.72 |
| M1 | 81.45 | 72.78 | **87.47** | 81.47 |
| M2 | 81.58 | 80.63 | 84.47 | 82.03 |
| **M3** | **82.86** | **81.23** | 85.95 | **82.84** |
| M4 | 80.67 | 79.96 | 82.98 | 82.20 |
| M5 | 82.46 | 79.28 | 86.75 | 81.48 |
| M6 | 81.99 | 80.07 | 82.36 | 82.51 |

M3相对M0：

```text
Stage 1 node accuracy：+3.40
Stage 2 node accuracy：+17.78
Stage 3 node accuracy：+5.09
Stage 2 Tier3 accuracy：+2.12
```

Stage 2的node提升达到17.78个百分点，而Tier3只提高2.12个百分点。这说明M0本来已经能识别
大部分Tier3动作，但无法区分Stage 2线性流程中重复出现的相同动作node。历史与task graph提供的
主要信息正是“这是该动作在流程中的第几次出现”。

### 7.2 重复动作node的混淆大幅下降

将A/D/M三个参与者、三个all-runs seed的test_all混淆矩阵相加，比较M0与M3在Stage 2重复动作
node之间的双向误判次数：

| 相同Tier3动作对应的两个node | M0互相误判 | M3互相误判 | 降幅 |
|---|---:|---:|---:|
| `place sample under electrodes`：node 14 ↔ 21 | 84 | 5 | 94.0% |
| `press pedal`：node 15 ↔ 22 | 152 | 4 | 97.4% |
| `put sample on machine table`：node 16 ↔ 19 | 134 | 13 | 90.3% |
| `grip sample from machine table`：node 17 ↔ 20 | 125 | 23 | 81.6% |

最大的单node recall提升包括：

| Node | 动作 | M0 Recall | M3 Recall | 提升 |
|---|---|---:|---:|---:|
| 22 | press pedal（第二次） | 30.39 | 82.35 | +51.96 |
| 19 | put sample on machine table（第二次） | 38.73 | 85.78 | +47.06 |
| 15 | press pedal（第一次） | 46.58 | 84.47 | +37.90 |
| 20 | grip sample from machine table（第二次） | 41.18 | 66.67 | +25.49 |
| 17 | grip sample from machine table（第一次） | 48.53 | 73.53 | +25.00 |
| 21 | place sample under electrodes（第二次） | 52.94 | 75.49 | +22.55 |

这是目前最直接支持task-history方法有效性的证据：模型不是仅利用历史提高整体分类，而是在
task graph最需要消歧的位置显著减少了同动作、不同node之间的混淆。

M3仍然较困难的node包括：

- node 16 `put sample on machine table`：recall 60.78%；
- node 20 `grip sample from machine table`：66.67%；
- node 7 `turn on water pump`：68.63%；
- node 35 `lock crimper`：70.59%。

后续可重点检查这些node的视觉相似性、clip边界和历史缺失情况。

## 8. normal-only与all-runs训练范围比较

下表只使用相同`seed_1`，数值为A/D/M上的
`完整all-runs pipeline − 完整normal-only pipeline`平均差值。

### 8.1 test_all配对差值

| 模型 | Δ Node Acc | Δ Node Macro-F1 | Δ Tier3 Acc | Δ Tier3 Macro-F1 |
|---|---:|---:|---:|---:|
| M0 | +5.30 | +6.71 | +7.29 | +7.25 |
| M1 | +0.00 | +1.25 | +5.65 | +3.50 |
| M2 | **+8.93** | **+8.51** | +7.12 | +6.67 |
| M3 | +8.06 | +6.89 | **+7.62** | +5.60 |
| M4 | +5.37 | +5.59 | +4.35 | +4.81 |
| M5 | +4.94 | +6.52 | +4.85 | +6.54 |
| M6 | +7.32 | +6.92 | +5.90 | +5.71 |
| E2E-Node-Scratch | +1.58 | -0.59 | +1.97 | -0.86 |
| E2E-Node-From-Tier3 | +3.44 | +4.71 | +3.38 | +4.39 |
| E2E-Tier3-Scratch | — | — | +5.29 | +5.21 |

加入训练人员的fault runs后，M0和几乎所有history模型都有明显提升。对M2、M3、M6而言，
提升不只是fault测试集上的适配，而是整体视觉表征、node分类和Tier3分类共同改善。

### 8.2 normal与fault上的差异

| 模型 | Fault Δ Node | Fault Δ Tier3 | Normal Δ Node | Normal Δ Tier3 |
|---|---:|---:|---:|---:|
| M0 | +3.47 | +7.80 | +5.43 | +6.57 |
| M1 | +9.41 | **+14.28** | **-2.75** | +3.49 |
| M2 | +9.58 | +7.29 | +8.16 | +6.48 |
| M3 | +10.21 | +9.67 | +7.48 | **+7.02** |
| M4 | +7.23 | +5.46 | +4.59 | +3.75 |
| M5 | +7.42 | +6.10 | +3.84 | +3.93 |
| M6 | **+11.76** | +8.24 | +6.01 | +5.00 |
| E2E-Node-Scratch | +2.97 | +2.43 | +1.28 | +1.93 |
| E2E-Node-From-Tier3 | +5.70 | +4.26 | +3.04 | +3.31 |
| E2E-Tier3-Scratch | — | +6.59 | — | +4.63 |

关键结论：

1. all-runs训练对fault测试的帮助总体更大；
2. M2、M3、M6在normal和fault上都提高，未显示明显的正常流程性能代价；
3. M1是主要例外：fault性能大幅改善，但normal node accuracy下降2.75个百分点；
4. M1的这种权衡再次说明没有位置/graph结构的历史attention容易依赖训练分布中的序列捷径；
5. M3在normal和fault上都保持较大、较均衡的提升，是更可靠的训练配置。

### 8.3 participant一致性

在相同`seed_1`的test_all上：

- M3的node提升：A +7.42、D +6.93、M +9.84；
- M3的Tier3提升：A +7.19、D +6.06、M +9.62；
- M6的node提升：A +7.89、D +7.36、M +6.71；
- M6的Tier3提升：A +5.34、D +5.19、M +7.16；
- M2、M3、M5、M6的node提升在A/D/M三人上均为正。

因此all-runs的主要提升不是由单个参与者独占。

## 9. all-runs随机种子稳定性

下表先在A/D/M之间平均，再计算`seed_1`、`seed_2`、`seed_42`的标准差。

| 模型 | Test-all Node Acc | Seed SD | Test-all Tier3 Acc | Seed SD |
|---|---:|---:|---:|---:|
| M0 | 68.11 | 2.39 | 80.64 | 1.09 |
| M1 | 76.04 | 1.62 | 82.35 | 1.52 |
| M2 | 81.35 | 2.15 | 82.37 | 1.30 |
| M3 | **82.16** | 1.81 | **83.33** | 1.18 |
| M4 | 80.53 | **0.73** | 82.15 | **0.33** |
| M5 | 80.81 | 1.86 | 82.41 | 1.28 |
| M6 | 80.72 | 0.97 | 82.49 | 0.84 |
| E2E-Node-Scratch | 71.40 | 0.74 | 78.18 | 1.33 |
| E2E-Node-From-Tier3 | 75.45 | 1.61 | 82.98 | 1.02 |
| E2E-Tier3-Scratch | — | — | 82.54 | 1.29 |

M4和M6最稳定，但平均性能低于M3。M3的seed波动约1.8个百分点node、1.2个百分点Tier3，
属于可见但可接受的训练方差。`seed_2`对M2/M3整体较弱，`seed_42`整体较强，因此后续论文
结果应继续报告多seed均值，而不是只使用最好的seed。

## 10. 参与者差异

all-runs、3-seed平均、test_all：

| 参与者 | M0 Node | M3 Node | M3−M0 | M0 Tier3 | M3 Tier3 | M3−M0 |
|---|---:|---:|---:|---:|---:|---:|
| A | 65.27 | 79.27 | +14.00 | 78.19 | 80.74 | +2.55 |
| D | 69.55 | 81.53 | +11.98 | 81.17 | 83.26 | +2.09 |
| M | 69.50 | 85.68 | +16.18 | 82.55 | 85.98 | +3.43 |

A是最困难的held-out参与者，M最容易；但M3相对M0的node增益在三人上都超过11个百分点。
这说明历史方法的收益具有跨人一致性，同时绝对性能仍受到参与者外观、操作风格和数据质量影响。

fault结果的参与者间标准差明显更大。例如M3的fault node accuracy跨人标准差约10.21个百分点。
这与fault样本数和类别覆盖不均有关，不应仅依据fault均值宣称稳定的异常泛化能力。

## 11. E2E对照的含义

### 11.1 从Tier3迁移明显优于35-node从scratch

all-runs test_all：

```text
E2E-Node-Scratch：
  Node Acc 71.40，Tier3 Acc 78.18

E2E-Node-From-Tier3：
  Node Acc 75.45，Tier3 Acc 82.98
```

Tier3预训练带来约+4.05 node accuracy和+4.80 Tier3 accuracy，说明现有Tier3视觉任务确实提供
有价值的动作外观表征。

### 11.2 历史M3对35-node的优势远大于对Tier3的优势

M3相对E2E-Node-From-Tier3：

```text
test_all：
  Node Acc +6.71
  Tier3 Acc +0.35
  Tier3 Macro-F1 +2.19

test_normal：
  Node Acc +7.46
  Tier3 Acc +0.94
  Tier3 Macro-F1 +2.65

test_fault：
  Node Acc +5.09
  Tier3 Acc -1.21
  Tier3 Macro-F1 +0.60
```

这进一步证明该方法的核心价值是node级流程定位，而不是单纯替代Tier3视频分类器。

### 11.3 直接Tier3分类仍是必要对照

all-runs下，E2E-Tier3-Scratch的test_all accuracy为82.54%，M3聚合Tier3为83.33%，差距只有
0.79个百分点。若研究目标只需要31类Tier3动作，直接Tier3模型已经很有竞争力；如果需要区分
35个task graph node、判断流程位置或为后续错误检测提供状态，M3的优势才更加明确。

## 12. 总体结论

### 12.1 得到较强支持的结论

1. **历史信息能稳定提高35-node分类。**  
   J上M3相对M0提高14.77个百分点；A/D/M all-runs上提高14.05个百分点，并在9/9
   participant-seed配对中为正。

2. **收益主要来自重复动作node消歧。**  
   Stage 2 node accuracy提高17.78个百分点，多个相同Tier3动作对应node之间的互相误判下降
   81.6%–97.4%。

3. **精确实际顺序不是必要条件。**  
   graph-valid shuffled history的M3平均优于使用真实顺序和位置编码的M2。

4. **完整all-runs训练明显优于normal-only。**  
   在严格配对的`seed_1`比较中，M3 test_all node/Tier3分别提高8.06和7.62个百分点；
   normal和fault划分都获得提升。

5. **M3是当前最可靠的综合配置。**  
   它在all-runs的normal、all中四个主要指标均第一，在fault中node指标第一，并且相对M0
   的方向最一致。

### 12.2 得到部分支持但仍需改进的结论

1. **relation bias具有潜力，但soft graph版本尚未成熟。**  
   M5相对M4的Tier3 macro-F1提高1.63个百分点，证明oracle关系有信息；M6提升较小且没有超过M3。

2. **M1不够稳定。**  
   它在部分normal结果上很好，但J、fault和all-runs范围比较中出现明显退化或权衡。没有位置或
   graph结构的历史attention不应作为最终方法。

3. **fault性能仍受测试规模和类别缺失影响。**  
   当前结果能说明“在故障run中的clip分类”改善，但不能等同于已经完成fault detection或
   sequence anomaly detection。

## 13. 下一步实验建议

按优先级建议如下。

### 13.1 补齐严格公平的normal-only多seed

为A/D/M normal-only补跑`seed_2`和`seed_42`，使其与all-runs完全对称。然后重新计算：

```text
同participant + 同seed + 同model + 同split
all-runs − normal-only
```

这是当前最重要的缺口。补齐后才能把训练范围提升与随机初始化方差彻底分离。

### 13.2 对J运行严格scratch LOSO

使用跨人实验包对J重新训练：

- normal-only scratch backbone；
- all-runs scratch backbone；
- 相同3个seed；
- 不使用J作为validation；
- 与A/D/M相同的最后epoch策略。

这样才能形成严格一致的A/D/J/M四折结果。

### 13.3 将M3作为当前主模型，将M2/M4/M5/M6作为消融

建议当前论文结构：

```text
Baseline：M0
History basic：M1
Actual ordered history：M2
Graph-valid order invariant history：M3（当前主结果）
No relation bias：M4
Oracle relation bias：M5
Predicted soft relation bias：M6
```

不要预先将M6定义为最终最佳模型；当前实验更支持M3。

### 13.4 改进M6

建议依次验证：

1. 用out-of-fold M0预测构造训练历史概率，避免同训练集预测过度自信；
2. 对M0历史概率做temperature calibration；
3. 对relation expectation使用top-k或置信度门控，低置信度时退回M4；
4. 单独报告每个attention head学到的relation bias；
5. 加入relation dropout，防止模型过度依赖错误soft relation；
6. 比较冻结M0与联合微调M0；
7. 检查M6在node 16、20、35上的历史概率和relation分布。

### 13.5 做run-level统计

同一run内clip不是独立样本。建议基于每个run计算模型差值，然后进行：

- paired bootstrap confidence interval；
- participant内run bootstrap；
- normal和fault分别统计；
- 最终在participant层面汇总。

不要直接把所有clip当作独立样本做普通t-test。

### 13.6 从分类扩展到流程异常识别

当前模型仍是“给定当前clip及其历史，预测当前动作/node”。如果后续目标是识别漏做、多做、
重复或顺序错误，需要增加独立的sequence-level输出，例如：

- 当前node是否违反task graph；
- 缺失的must previous node；
- 重复动作或非法跳转；
- run-level fault score；
- graph-constrained decoding。

这应作为后续任务，不应与当前clip分类结果混为一谈。

## 14. 推荐用于论文或阶段汇报的核心结果

如果只选最有解释力的结果，建议报告以下四项：

1. J三seed：M3相对M0，node accuracy `+14.77`，Tier3 accuracy `+3.48`；
2. A/D/M all-runs三seed：M3相对M0，node accuracy `+14.05`，Tier3 accuracy `+2.69`；
3. Stage 2：M3相对M0，node accuracy `+17.78`，Tier3 accuracy仅`+2.12`；
4. 重复动作node双向混淆下降`81.6%–97.4%`。

这四项共同说明：

> 历史与task graph信息的核心作用不是替代视觉动作识别，而是将视觉上相同或相似的动作定位到
> 正确的流程node；graph-valid历史即使不保持实际精确顺序，仍能提供稳定且跨人的判别信息。

## 15. 结果来源

主要汇总文件：

```text
J三seed：
D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\graph_history_rgb_experiments_2026-07-20\outputs\J_as_test\cam_001484412812\history_models\existing_last*\experiment_summary.csv

A/D/M normal-only：
D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\graph_history_rgb_cross_person_ADM_2026-07-22\outputs\cross_person_summary_with_e2e

A/D/M all-runs：
D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\graph_history_rgb_cross_person_ADM_2026-07-22\outputs\cross_person_summary_all_runs

normal-only与all-runs配对比较：
D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\graph_history_rgb_cross_person_ADM_2026-07-22\outputs\training_scope_comparison
```

所有表格均由上述实际JSON/CSV结果计算。报告没有修改checkpoint、prediction、probability或原始
summary文件。
