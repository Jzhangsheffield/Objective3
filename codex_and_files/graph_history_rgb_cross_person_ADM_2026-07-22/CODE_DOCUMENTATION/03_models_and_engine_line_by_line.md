# 03 模型与训练Engine逐段解析

## 1. `models.py`

### 1.1 `FeatureNodeClassifier`（12–25）

#### 初始化（15–21）

```text
self.feature_dim = feature_dim
self.num_nodes   = num_nodes
self.norm        = LayerNorm(feature_dim)
self.dropout     = Dropout(dropout)
self.fc          = Linear(feature_dim,num_nodes)
```

默认输入`[B,512]`，输出`[B,35]`。LayerNorm按每个样本最后一维归一化；Dropout只在train模式生效。

#### forward（23–25）

逻辑为：

```text
features [B,512]
→ norm [B,512]
→ dropout [B,512]
→ fc [B,35]
```

没有softmax，因为`cross_entropy`需要raw logits；softmax只在评估或概率构造时调用。

### 1.2 `freeze_module`（27–31）

先`eval()`，再遍历所有parameters设置`requires_grad=False`。冻结同时包含：

- 不计算参数梯度；
- dropout关闭；
- 若模块包含BatchNorm，不更新running statistics。

### 1.3 `SingleQueryHistoryModel`（33–113）

#### `__init__`（36–71）

输入参数：

| 参数 | 含义 |
|---|---|
| `baseline` | 已训练M0 `FeatureNodeClassifier` |
| `feature_dim` | 512 |
| `d_model` | 256 |
| `num_heads` | 4 |
| `max_history` | 35 |
| `dropout` | 0.1 |
| `use_position` | M1 False；M2/M3 True |

构造顺序：

1. 保存并冻结baseline；
2. 当前与历史各用`Linear(512,256)+LayerNorm(256)`投影；
3. 建立`Embedding(36,256)`，ID 0给padding，1..35表示历史距离；
4. 建立可学习null token`[1,1,256]`；
5. 建立batch-first MultiheadAttention；
6. delta head读取`[current;context] [B,512]`并输出`[B,35]`；
7. `history_scale_logit`初始化为-2，所以初始`sigmoid≈0.119`，历史残差从较小幅度开始。

#### `train`（73–76）

调用父类切换整体模式后，立即把baseline重新设为eval。这样即使外层`model.train()`，冻结M0仍不启用
dropout。

#### `forward`（78–113）

输入shape：

```text
current_feature       [B,512]
history_features      [B,L,512]
history_position_ids  [B,L]
history_padding_mask  [B,L]
```

逐段：

1. 当前投影`current [B,256]`；
2. 历史投影`history [B,L,256]`；
3. 若使用位置且`L>0`，clamp ID到0..35并加embedding；
4. null token扩展为`[B,1,256]`并拼到历史最前，得到`[B,L+1,256]`；
5. 拼一个全False null mask，得到`[B,L+1]`；
6. query为`current.unsqueeze(1) [B,1,256]`，key/value为历史；
7. attention返回`context [B,1,256]`及weights；
8. `squeeze(1)`得到`[B,256]`；
9. 拼接current/context为`[B,512]`；
10. delta head输出`[B,35]`；
11. `scale=sigmoid(history_scale_logit)`，标量；
12. `no_grad`计算冻结baseline logits`[B,35]`；
13. 最终`baseline + scale*delta`；
14. 返回logits和诊断字典。

null token使空历史batch的key长度仍为1；这时模型可学习“无历史”的默认context。

### 1.4 `DirectSingleQueryHistoryModel`（116–197）

#### 与SingleQuery相同的部分

- 当前/历史投影；
- 可选位置embedding；
- null token；
- MultiheadAttention；
- 空历史与padding处理。

#### Direct特有层（151–160）

```text
fusion = Linear(512+256,512)
node_classifier = LayerNorm(512) → Dropout(0) → Linear(512,35)
```

初始化先把fusion权重和bias清零，然后把左侧`512×512`块复制为单位矩阵。classifier保持PyTorch默认
随机初始化。

#### forward（162–197）

1. 保留原始`current_feature [B,512]`；
2. 投影当前query与历史；
3. 可选位置；
4. 添加null并做attention；
5. context从`[B,1,256]`变`[B,256]`；
6. 拼接`[current_feature;context] [B,768]`；
7. fusion得到`[B,512]`；
8. `fused_feature`送入`FeatureNodeClassifier`；
9. classifier内部LayerNorm、Dropout(0)和Linear得到`[B,35]`；
10. 返回logits和`attention/context/fused_feature`诊断值。

这里没有baseline、history scale或delta。

### 1.5 `CandidateHistoryModel`（200–371）

#### 初始化（203–254）

关键检查：

- `d_model % num_heads == 0`；
- `graph_source`只能是`none/oracle/predicted`。

关键成员：

```text
baseline                    frozen M0
relation_ids                buffer [35,35]
current_projection          512→256
history_projection          512→256
position_embedding          36×256
candidate_embedding         35×256
query/key/value projections 256→256
relation_bias               [heads,5]
immediate_not_last_bias     [heads]
null_history                [1,1,256]
delta_head                  对每个candidate输出标量
```

`register_buffer`让relation IDs随模型移动设备并写入state dict，但不参与梯度。

#### `train`（256–259）

与SingleQuery相同：整体切train后强制baseline为eval。

#### `_history_node_probabilities`（261–280）

按graph source分支：

- `none`：返回全零`[B,L,35]`；
- `oracle`：对`history_node_classes`做one-hot；先clamp避免padding=-1报错，再用mask清零padding；
- `predicted`：展平历史为`[B*L,512]`，冻结baseline输出`[B*L,35]`，softmax后reshape
  `[B,L,35]`，padding清零；
- `L=0`时维持合法空shape。

M6训练时读取的是M0对历史clip的当前预测，不是历史真实node。

#### `_graph_bias`（282–310）

输入：

```text
history_probabilities [B,L,35]
position_ids          [B,L]
padding_mask          [B,L]
```

1. `none`或空历史返回`[B,A,35,L]`零tensor；
2. `self.relation_bias[:, self.relation_ids]`得到`pair_bias [A,35,35]`；
3. einsum将历史node概率对最后一个历史node维加权求和：

```text
graph_bias[b,h,v,l] = Σ_u p_history[b,l,u] * bias[h, relation(v,u)]
```

4. 单独计算历史token成为候选node立即前序的概率；
5. 若该token位置ID不等于1，则加`immediate_not_last_bias`惩罚/奖励；
6. 返回每head、每候选node、每历史位置的bias。

#### `forward`（312–371）

1. 取`B/L`；
2. 当前、历史投影并加位置；
3. candidate embedding扩展为`[B,35,256]`；
4. 当前投影与candidate相加，再投影为query；
5. null与历史拼接，投影key/value；
6. reshape和transpose拆成4个head：

```text
Q [B,4,35,64]
K,V [B,4,L+1,64]
```

7. 点积/根号64得到scores`[B,4,35,L+1]`；
8. 构造history node probabilities和graph bias；
9. 为null位置补零bias并与历史bias拼接；
10. 拼padding mask并把不可读位置设为dtype最小有限值；
11. softmax + dropout得到attention；
12. attention乘values得到context；
13. 合并heads为`[B,35,256]`；
14. 拼`current expanded/context/candidate`，逐candidate输出delta`[B,35]`；
15. 冻结baseline输出；
16. baseline + scale×delta；
17. 返回logits和完整诊断字典。

### 1.6 `build_context_model`（374–404）

工厂映射：

```text
m1 → SingleQuery(use_position=False)
m2/m3 → SingleQuery(use_position=True)
m4 → Candidate(graph_source=none)
m5 → Candidate(graph_source=oracle)
m6 → Candidate(graph_source=predicted)
```

M2与M3模型结构相同，区别来自Dataset的history order。

### 1.7 `build_direct_context_model`（407–429）

```text
m1_direct → DirectSingleQuery(use_position=False)
m2_direct/m3_direct → DirectSingleQuery(use_position=True)
```

M2/M3 Direct结构相同，仍由Dataset决定actual或graph-valid。

## 2. `engine.py`

### 2.1 `move_batch_to_device`（17–21）

字典推导逐key处理：

- Tensor调用`.to(device, non_blocking=True)`；
- 字符串列表等metadata原样保留。

返回新dict，不原地修改raw batch。

### 2.2 `forward_node_model`（24–34）

- 若模型是M0 `FeatureNodeClassifier`，只传`current_feature`并返回空诊断字典；
- 其他history/direct模型按命名参数传五个batch tensor；
- 统一返回`(logits, diagnostics)`，让训练循环不需要为模型族写多个分支。

### 2.3 `compute_loss`（37–58）

输入：

```text
logits        [B,35]
node_target   [B]
tier3_target  [B]
mapping       [35]
```

逐行：

1. node交叉熵；
2. 默认总loss=node loss；
3. 创建同device/dtype的0标量action loss；
4. 权重大于0时softmax node；
5. 聚合到`[B,31]`；
6. `gather`每个样本真实Tier3概率，结果`[B,1]→[B]`；
7. `clamp_min(1e-12)`避免log(0)；
8. 负对数均值；
9. 加权加入总loss；
10. 返回可反传loss和已detach的日志标量。

### 2.4 `train_feature_model`（61–116）

初始化AMP GradScaler、history list并把mapping移到device。每个epoch：

1. `model.train()`；history模型内部会保持baseline eval；
2. 初始化计时、loss、correct、total；
3. 遍历loader并搬device；
4. `zero_grad(set_to_none=True)`减少内存写；
5. AMP上下文内forward和loss；
6. scaler缩放后backward；
7. unscale；
8. 对所有可训练参数做global norm裁剪，阈值1.0；
9. optimizer step与scaler update；
10. loss按batch size加权累积；
11. argmax node计数；
12. epoch结束记录平均loss、accuracy和秒数；
13. 打印并append；
14. 所有epoch后返回训练日志list。

没有validation、scheduler、early stopping或best checkpoint逻辑。

### 2.5 `evaluate_feature_model`（120–216）

装饰器`@torch.no_grad()`关闭梯度。流程：

1. eval模式；
2. mapping移到device；
3. 初始化真实/预测/stage/row/probability容器；
4. 对每个batch统一forward；
5. node softmax`[B,35]`；
6. 聚合Tier3`[B,31]`；
7. 取得两个argmax；
8. node probabilities移CPU收集；
9. 逐样本构造可追溯prediction row；
10. 计算overall node/Tier3；
11. 对stage 1/2/3筛选索引并分别计算；
12. 写`*_metrics.json`；
13. 写`*_predictions.csv`；
14. 写`*_probabilities.pt`，含`[S,35]`和rows；
15. 返回metrics。

置信度是预测类别的最大概率，不是校准后的概率保证。
