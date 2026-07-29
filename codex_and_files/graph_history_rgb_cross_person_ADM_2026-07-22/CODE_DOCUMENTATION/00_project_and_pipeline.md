# 00 项目、版本与端到端流程

## 1. 研究任务

输入是单相机`001484412812`采集的RGB视频clip。基础视觉任务有两种标签空间：

- 35个Task Graph node：表达动作位于工艺流程中的具体位置；
- 31个Tier3动作：若不同node具有相同动作语义，会聚合到同一个Tier3类别。

因此，35-node任务比31类动作识别多解决一层“流程位置消歧”。例如Stage 2中两次`press pedal`
视觉外观相同，但属于不同node；仅看当前clip容易互相混淆，而同run历史可提示当前处于第一次还是
第二次压踏板。

## 2. 严格实验单位

正式网格：

```text
participant = A, D, J, M
seed        = 1, 2, 42
scope       = normal_only, all_runs
models      = M0-M6 + 3 E2E + 3 Direct扩展
test split  = test_normal, test_fault, test_all
```

每个fold只用另外三位participant训练。held-out participant不进入训练、validation或checkpoint
选择。没有early stopping，统一使用最后epoch的`last.pth`。

normal-only和all-runs是两个完整、独立的representation/training scope：

- `normal_only` backbone仅用训练participants的normal runs；
- `all_runs` backbone使用训练participants的normal与fault runs；
- 两种scope各自提取feature cache并训练下游模型；
- 严格scope差值使用相同participant、seed、model和split配对。

## 3. Task Graph

Task Graph资产：

```text
assets/integrated_task_graph_latest.json
assets/integrated_feature_history_matrix.json
```

每个可预测node至少给出：

- `node_idx`：JSON使用1到35；模型内部转换为0到34；
- `stage_id`：1、2或3；
- `action_id_tier3`：0到30；
- 必须前序node；
- 可选前序node；
- 必须立即前序node。

relation matrix方向固定：

```text
row    = 当前候选node v
column = 历史node u
```

代码将`I/M/O/X/S`映射为整数`0/1/2/3/4`。JSON中的`.`被规范化为`X`。固定relation matrix
本身不训练；M5/M6学习的是每个attention head对五类relation的标量bias。

## 4. 完整数据流

### 4.1 协议生成

```text
原始manifest
  → 按held-out participant划分train/test
  → 用fault manifests识别完整fault run
  → normal_only/train.jsonl
  → all_runs/train.jsonl
  → test_normal.jsonl
  → test_fault.jsonl
  → test_all.jsonl
```

历史始终只来自相同`participant + run`且`annotation_row_index`更小的clip。没有未来信息，
没有跨run memory。

### 4.2 RGB backbone

```text
RGB tensor [原始帧数, 3, H0, W0]
  → 均匀选择16帧
  → resize/crop、颜色增强和归一化
  → 转为[3, 16, 224, 224]
  → batch后[B, 3, 16, 224, 224]
  → ResNet3D-18
  → 31类Tier3 logits [B, 31]
```

训练后保存`last.pth`。特征提取阶段读取分类头之前的512维向量：

```text
feature [B, 512]
tier3_logits [B, 31]
```

并连同逐样本record与metadata保存到`.pt` feature cache。

### 4.3 历史样本

`FeatureHistoryDataset`先按`participant + run`分组，再按`annotation_row_index`排序。对于run中
位置`t`的当前clip，历史是`[0:t]`。

- M1/M2/M4/M5/M6与Direct M1/M2使用实际发生顺序；
- M3与Direct M3使用确定性的graph-valid随机拓扑序；
- graph-valid排序只依据已经观察到的历史node之间的关系；
- 当前目标node不传给排序函数，防止标签泄漏；
- 历史中node重复时回退实际顺序。

batch collate把变长历史padding到当前batch最大长度`L`：

```text
current_feature          [B, 512]
history_features         [B, L, 512]
history_position_ids     [B, L]
history_node_classes     [B, L]
history_padding_mask     [B, L]
node_target              [B]
tier3_target             [B]
```

padding mask中`True`表示padding、不可被attention读取；有效历史为`False`。

## 5. 模型族

### 5.1 M0

```text
LayerNorm(512)
→ Dropout
→ Linear(512, 35)
```

只读取当前clip特征，训练35-node分类头。

### 5.2 原M1–M3：frozen M0 + history delta

```text
baseline_logits = frozen_M0(current_feature)          [B,35]
context = single-query attention(current, history)    [B,256]
delta = MLP([current_projected, context])              [B,35]
logits = baseline_logits + sigmoid(scale_logit)*delta  [B,35]
```

- M1：无位置embedding；
- M2：实际历史 + 位置embedding；
- M3：graph-valid历史 + 位置embedding。

覆盖训练模式时，baseline始终保持`eval()`，且其参数`requires_grad=False`。

### 5.3 M4–M6：candidate history attention

35个候选node分别形成query，因此attention张量保留candidate维：

```text
queries      [B, heads, 35, head_dim]
keys/values  [B, heads, L+1, head_dim]
scores       [B, heads, 35, L+1]
context      [B, 35, 256]
delta        [B, 35]
```

额外的`+1`是可学习null-history token，保证空历史样本仍有合法attention key。

- M4：graph source=`none`，bias全零；
- M5：历史真实node one-hot概率，oracle；
- M6：冻结M0对历史feature的35-node softmax概率。

### 5.4 E2E

- E2E-Tier3-Scratch：RGB直接预测31类，只报告Tier3。
- E2E-Node-Scratch：ResNet3D从scratch预测35-node。
- E2E-Node-From-Tier3：加载Tier3 backbone兼容参数，跳过形状不同的最终fc，再训练35-node。

### 5.5 Direct Head Fusion

```text
current_feature [B,512]
history context [B,256]
concat          [B,768]
Linear(768,512) + LayerNorm
node head       Linear(512,35)
```

它不读取M0 checkpoint，不做logit residual。fusion层初始权重为`[I512, 0]`，bias为0，因此训练
开始时`fused_feature == current_feature`；随后attention、fusion和node head联合学习。

## 6. 35-node到31类Tier3

设`p_node[..., n]`为node概率，`map[n]`为该node对应的Tier3类别，则：

```text
p_tier3[..., c] = Σ p_node[..., n], 对所有map[n] == c的node求和
```

代码使用`scatter_add_`实现，因此保留完整概率质量。它不是先取最大概率node再做标签映射。

## 7. 训练与评估

history/direct模型默认主loss：

```text
L_node = CrossEntropy(node_logits, node_target)
```

可选Tier3辅助loss从聚合概率中取真实Tier3概率的负对数：

```text
L = L_node + action_loss_weight * L_tier3
```

当前默认`action_loss_weight=0.0`，即只优化node交叉熵。

评估同时生成：

- 35-node accuracy、macro-F1、balanced accuracy；
- 31类Tier3相同指标；
- Stage 1/2/3分阶段指标；
- 逐样本预测与置信度；
- node probability tensor。

macro-F1与balanced accuracy只在当前split中真实出现的类别上平均，因此比较不同participant的
fault split时必须同时检查`present_class_count`。

