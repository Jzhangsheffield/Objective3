# 01 张量、字段与文件数据字典

## 1. 全局符号

| 符号 | 含义 | 默认/范围 |
|---|---|---|
| `B` | batch size | backbone默认16；history默认64 |
| `T` | 每个clip选取的帧数 | 16 |
| `L` | 当前batch padding后的最大历史长度 | 0到35 |
| `F` | 冻结RGB feature维度 | 512 |
| `D` | history attention内部维度 | 256 |
| `N` | Task Graph node数 | 35 |
| `C` | Tier3类别数 | 31 |
| `A` | attention head数 | 4 |
| `Dh` | 每个head维度 | `D/A=64` |
| `S` | 某个split的样本数 | 随participant/split变化 |

## 2. 索引约定

| 概念 | 文件/报告 | 模型内部 |
|---|---:|---:|
| graph node | `node_idx=1..35` | `node_target=0..34` |
| Tier3 | `tier3_id=0..30` | 不变，`0..30` |
| stage | `1..3` | 不变，`1..3` |

写prediction CSV时，node预测会加1恢复为`1..35`；Tier3不加1。

## 3. Manifest行

每一行是一个JSON对象。代码直接依赖的字段：

| 字段 | 类型 | 内容 |
|---|---|---|
| `sample_name` | string | clip唯一标识，也是cache查找键 |
| `participant` | string | `A/D/J/M` |
| `run` | string | participant内部run标识 |
| `annotation_row_index` | int | run内时序位置；更小表示更早 |
| `node_idx` | int | 1到35 |
| `tier3_id` | int | 0到30 |
| `stage_id` | int | 1、2或3 |
| `<camera_id>_rgb` | string | 相对dataset root的RGB tensor路径 |

`run_key(row)`返回`(participant, run)`，用于严格限制历史来源。

## 4. RGB tensor

磁盘对象可以是tensor本身，也可以是包含`frames`的字典：

```text
原始 video: [T_raw, 3, H_raw, W_raw]
```

约束：

- 必须是PyTorch tensor；
- 必须4维；
- 第二维必须为3；
- `T_raw > 0`。

处理后单样本：

```text
均匀采样      [16, 3, H_raw, W_raw]
transform后   [16, 3, 224, 224]
permute后     [3, 16, 224, 224]
```

DataLoader默认collate后：

```text
video [B, 3, 16, 224, 224], float32
```

训练transform包含随机裁剪、ColorJitter、随机灰度；水平和垂直翻转概率均显式设为0。
验证/测试只resize、转float并归一化。

## 5. RGBClipDataset返回值

| key | 单样本类型/shape | batch后 |
|---|---|---|
| `video` | float tensor `[3,T,224,224]` | `[B,3,T,224,224]` |
| `tier3_target` | Python int | long tensor `[B]` |
| `node_target` | Python int，0-based | long tensor `[B]` |
| `stage_id` | Python int | long tensor `[B]` |
| `sample_name` | string | `list[str]`, 长度B |
| `participant` | string | `list[str]` |
| `run` | string | `list[str]` |
| `annotation_row_index` | int | 默认collate为tensor `[B]` |

## 6. Feature cache

顶层必须是字典，并至少包含：

| key | 内容 |
|---|---|
| `features` | `[S,512]`，分类头之前的冻结RGB特征 |
| `tier3_logits` | `[S,31]` |
| `records` | 长度S的manifest行/样本metadata |
| `metadata` | checkpoint、相机、提取配置等来源信息 |

`load_feature_cache`检查`len(records) == features.shape[0]`，但不会自动修复错位。`sample_name`
是selection manifest与cache对齐的主键。

## 7. HistoryExample

这是冻结dataclass，不包含复制后的feature，只保存索引和行对象：

| 字段 | 类型 | 内容 |
|---|---|---|
| `current_cache_index` | int | 当前clip在cache第一维的行号 |
| `history_cache_indices` | `tuple[int,...]` | 历史clips在cache中的行号 |
| `current_row` | dict | 当前manifest行 |
| `history_rows` | `tuple[dict,...]` | 以实际或graph-valid顺序排列的历史行 |

这样多个Dataset样本可以共享同一个`features [S,512]`，避免为每个样本复制历史特征。

## 8. FeatureHistoryDataset单样本

若当前历史长度为`l`：

| key | shape/类型 | 内容 |
|---|---|---|
| `current_feature` | `[512]`, float32 | 当前clip冻结特征 |
| `history_features` | `[l,512]`, float32 | 有序历史特征 |
| `history_position_ids` | `[l]`, long | `l,l-1,...,1`，1代表最近 |
| `history_node_classes` | `[l]`, long | 历史真实node，0-based |
| `node_target` | int | 当前真实node，0-based |
| `tier3_target` | int | 当前真实Tier3 |
| `stage_id` | int | 当前stage |
| `history_sample_names` | `list[str]` | 与历史特征顺序完全一致 |

空历史：

```text
history_features      [0,512]
history_position_ids  [0]
history_node_classes  [0]
```

## 9. collate_history_batch

设batch内最大历史长度为`L`：

| key | shape/dtype | padding规则 |
|---|---|---|
| `current_feature` | `[B,512]` float32 | stack |
| `history_features` | `[B,L,512]` float32 | 0 |
| `history_position_ids` | `[B,L]` long | 0 |
| `history_node_classes` | `[B,L]` long | -1 |
| `history_padding_mask` | `[B,L]` bool | padding=True，有效=False |
| `node_target` | `[B]` long | 无 |
| `tier3_target` | `[B]` long | 无 |
| `stage_id` | `[B]` long | 无 |

如果整个batch都是run的第一个clip，`L=0`，张量shape合法为`[B,0,...]`。模型额外添加null token，
因此attention仍能运行。

## 10. TaskGraphSpec

| 字段 | shape/类型 | 内容 |
|---|---|---|
| `relation_ids` | `[35,35]` long | `[候选当前node, 历史node]`关系ID |
| `node_to_tier3` | `[35]` long | 0到30 |
| `node_to_stage` | `[35]` long | 1到3 |
| `all_must_previous` | `dict[int,tuple[int,...]]` | 使用1-based node；M3排序实际使用 |
| `immediate_previous` | `dict[int,int|None]` | 使用1-based node；当前M3排序未直接使用 |
| `atomic_sequences` | tuple of tuples | 不可拆分的局部顺序 |

## 11. 模型张量

### M0

```text
input features   [B,512]
normalized       [B,512]
logits           [B,35]
probabilities    [B,35]
```

### SingleQueryHistoryModel

```text
current projection       [B,256]
history projection       [B,L,256]
null token               [B,1,256]
history with null        [B,L+1,256]
query                    [B,1,256]
attention output         [B,1,256]
context after squeeze    [B,256]
concat current/context   [B,512]
history delta            [B,35]
baseline logits          [B,35]
final logits             [B,35]
```

### DirectSingleQueryHistoryModel

```text
current feature          [B,512]
attention context        [B,256]
concat                   [B,768]
fusion linear output     [B,512]，诊断字典中的fused_feature
node classifier LN       [B,512]
node classifier Linear   [B,35]
```

### CandidateHistoryModel

```text
candidate embedding      [35,256]
expanded candidates      [B,35,256]
queries                  [B,4,35,64]
keys/values              [B,4,L+1,64]
attention scores         [B,4,35,L+1]
history node probs       [B,L,35]
graph bias               [B,4,35,L]
context                  [B,35,256]
delta                    [B,35]
```

M5的history node probabilities来自one-hot真实标签；padding位置归零。M6由冻结M0输出softmax；
M4即使计算路径需要兼容shape，其graph bias返回全零。

## 12. 概率聚合

输入允许任意前导维度：

```text
node_probabilities [...,35]
node_to_tier3      [35]
result             [...,31]
```

例如普通batch为`[B,35] → [B,31]`。`scatter_add_`确保映射到同一Tier3的多个node概率相加。

## 13. 指标JSON

顶层：

```text
split
samples
node
tier3
per_stage
```

`node`和`tier3`均包含：

```text
accuracy
macro_f1
balanced_accuracy
present_class_count
total_class_count
per_class_precision
per_class_recall
per_class_f1
support
confusion_matrix
```

Node confusion matrix为`[35,35]`，Tier3为`[31,31]`。行是真实类别，列是预测类别。

## 14. Prediction CSV

历史模型评估至少写：

```text
sample_name
participant
run
annotation_row_index
stage_id
true_node_idx
pred_node_idx
true_tier3_id
pred_tier3_id
node_confidence
tier3_confidence
```

`true_node_idx/pred_node_idx`恢复为1-based；概率文件中的35维列仍按内部0-based位置排列。
