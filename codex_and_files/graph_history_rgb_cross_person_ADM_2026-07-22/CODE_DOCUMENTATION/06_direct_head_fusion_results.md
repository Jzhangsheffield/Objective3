# 06 Direct Head Fusion实现与新增结果

## 1. 名称

代码、配置、输出目录和完成标记统一使用`Direct Head Fusion`或`direct_head_fusion`。本项目中没有
名为`direction fusion`的模型；若口头使用“direction fusion head”，本文按Direct Head Fusion理解。

## 2. 研究问题

原M1–M3先训练M0的35-node classifier，再冻结M0并学习logit residual：

```text
final_logits = frozen_M0_logits + sigmoid(scale) * history_delta
```

这意味着35-node决策经历两个50-epoch阶段：

```text
先训练M0 node head
→ 冻结M0
→ 再训练history delta
```

Direct版本检验：

```text
冻结Tier3视觉feature
→ 当前feature与history context融合
→ 随机初始化35-node head直接输出最终logits
```

它不加载M0，不受M0 logit空间约束，fusion与node head在同一个50-epoch阶段联合训练。

## 3. 三个模型

| 模型 | 历史顺序 | 位置embedding | 分类机制 |
|---|---|---|---|
| `m1_direct` | actual | 无 | 融合后直接35-node head |
| `m2_direct` | actual | 有 | 融合后直接35-node head |
| `m3_direct` | graph-valid | 有 | 融合后直接35-node head |

## 4. 初始化

fusion为`Linear(512+256,512)`。初始化：

```text
W[:, 0:512]   = I512
W[:, 512:768] = 0
b              = 0
```

所以初始时：

```text
fusion([current_feature; context]) = current_feature
```

紧随其后的LayerNorm和随机node head仍参与训练。这个初始化的意义是让history context从零贡献起步，
避免随机context在训练初期立即破坏Tier3视觉feature。

## 5. 结果完整性

实际读取：

```text
outputs/direct_head_fusion_summary_ADJM_3seeds/completed.json
outputs/direct_head_fusion_summary_ADJM_3seeds/direct_head_metrics.csv
outputs/direct_head_fusion_summary_ADJM_3seeds/direct_head_paired_deltas.csv
outputs/direct_head_fusion_summary_ADJM_3seeds/direct_head_aggregate.csv
```

完成标记记录：

```text
4 participants × 3 seeds × 2 scopes × 3 models × 3 splits = 216 metric rows
216 rows × 2 references = 432 paired-delta rows
36 aggregate rows
```

聚合方式：

1. 每位participant内部先平均seed 1/2/42；
2. 再对A/D/J/M四人等权平均；
3. `±`是四个participant均值之间的样本标准差。

## 6. test_all主要结果

### 6.1 normal-only

| 模型 | Node Acc | Node Macro-F1 | Tier3 Acc | 相对M0 Node | 相对原同名模型 Node |
|---|---:|---:|---:|---:|---:|
| `m1_direct` | 80.52 ± 4.93 | 80.86 | 83.16 ± 4.80 | +13.76 | +1.93 |
| `m2_direct` | 88.64 ± 3.50 | 85.64 | 89.18 ± 3.56 | +21.88 | +9.34 |
| `m3_direct` | **88.72 ± 3.97** | 85.51 | **89.29 ± 3.96** | **+21.96** | **+9.42** |

相对原模型的Tier3 Accuracy：

```text
m1_direct - M1 = +2.14个百分点
m2_direct - M2 = +8.10个百分点
m3_direct - M3 = +8.31个百分点
```

### 6.2 all-runs

| 模型 | Node Acc | Node Macro-F1 | Tier3 Acc | 相对M0 Node | 相对原同名模型 Node |
|---|---:|---:|---:|---:|---:|
| `m1_direct` | 79.99 ± 6.52 | 80.69 | 84.97 ± 4.96 | +10.19 | +0.32 |
| `m2_direct` | **90.57 ± 2.66** | **87.81** | **90.64 ± 2.64** | **+20.76** | **+6.41** |
| `m3_direct` | 90.05 ± 3.31 | 87.60 | 90.27 ± 3.10 | +20.25 | +5.31 |

相对原模型的Tier3 Accuracy：

```text
m1_direct - M1 = +0.12个百分点
m2_direct - M2 = +5.65个百分点
m3_direct - M3 = +4.64个百分点
```

## 7. 新结果说明什么

### 7.1 Direct head对带位置模型非常有效

`m2_direct`和`m3_direct`的提升远大于`m1_direct`。这表明性能来源不能简单描述成“重新训练一个
分类头”，因为没有位置embedding的M1 Direct只略高于原M1，而M2/M3 Direct提高5到9个百分点Node。

更合理的解释是：

- 位置结构让attention context携带强流程定位信息；
- 直接在融合后的512维空间训练node head，比把历史限制为冻结M0 logits上的residual更容易使用
  这类信息；
- 原delta设计可能受冻结M0决策边界和history scale约束。

### 7.2 all-runs下M2 Direct略高于M3 Direct

在`test_all`：

```text
M2 Direct Node Accuracy - M3 Direct = +0.52个百分点
M2 Direct Tier3 Accuracy - M3 Direct = +0.37个百分点
```

差距很小，且当前表是四participant平均，不应据此宣称actual order必然优于graph-valid order。
应进一步查看participant/seed配对、run级结果和置信区间。

### 7.3 normal-only下两者几乎相同

```text
M3 Direct - M2 Direct:
Node Accuracy  +0.08个百分点
Tier3 Accuracy +0.11个百分点
```

这与原M2/M3在normal-only中接近的现象一致：正常流程顺序高度规律，actual与graph-valid历史提供的
结构非常相似。

### 7.4 scope效应不是所有模型都一致

直接从216行metric重新按participant内seed平均后计算`all-runs - normal-only`：

| 模型 | test_all Δ Node | 四位participant各自Δ |
|---|---:|---|
| `m1_direct` | -0.53 | -5.03, -5.48, +4.08, +4.33 |
| `m2_direct` | +1.93 | -0.39, +0.72, +1.92, +5.44 |
| `m3_direct` | +1.33 | -1.08, -0.72, +1.38, +5.74 |

因此Direct模型的all-runs优势小于原M3的`+5.44`，且M2/M3 Direct并非四位participant全部为正。
这部分属于描述性重新计算，当前专用summary没有正式输出scope-delta CSV；若用于论文，应由独立、
可复现的统计脚本生成并加入置信区间。

## 8. 当前可支持的结论

可以支持：

- Direct Head Fusion完整网格已完成；
- M2/M3 Direct明显优于M0和原M2/M3；
- 直接feature fusion比冻结M0 logit delta更适合当前带位置的history表示；
- all-runs总体最佳Direct配置是M2 Direct，normal-only的M2/M3 Direct几乎并列。

暂不能支持：

- Direct模型在新participant总体上具有统计显著优势；
- actual order确定优于graph-valid order；
- all-runs对每位participant都改善Direct模型；
- Direct Head Fusion已经能够检测流程异常；
- 216个模型结果或12个participant-seed可当作独立受试者。

## 9. 建议后续分析

1. 对`m2_direct - m2`、`m3_direct - m3`做participant-run层级paired bootstrap。
2. 分stage检查提升是否仍集中在Stage 2重复动作node。
3. 聚合prediction，重新计算四组重复node双向混淆。
4. 对比Direct与delta版本的attention entropy、history scale和fusion权重范数。
5. 检查fusion矩阵的history部分`W[:,512:768]`是否学到非零、稳定结构。
6. 分析M2 Direct与M3 Direct差异是否由fault run中的重复/非法顺序驱动。

