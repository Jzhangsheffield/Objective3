### class CandidateHistoryModel(nn.Module):
```python
class CandidateHistoryModel(nn.Module):
    """M4-M6: 35 candidate queries with optional task-graph relation bias."""

    def __init__(
        self,
        baseline: FeatureNodeClassifier,
        relation_ids: torch.Tensor,
        graph_source: str,
        feature_dim: int = 512,
        d_model: int = 256,
        num_heads: int = 4,
        max_history: int = 35,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if graph_source not in {"none", "oracle", "predicted"}:
            raise ValueError(f"Unsupported graph_source: {graph_source}")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.baseline = baseline
        freeze_module(self.baseline)
        self.graph_source = graph_source
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.max_history = int(max_history)
        self.current_projection = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))
        self.history_projection = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))
        self.position_embedding = nn.Embedding(max_history + 1, d_model)
        self.candidate_embedding = nn.Embedding(NUM_GRAPH_NODES, d_model)
        self.null_history = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.null_history, std=0.02)
        nn.init.normal_(self.candidate_embedding.weight, std=0.02)

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.register_buffer("relation_ids", relation_ids.long().clone(), persistent=True)

        initial = torch.tensor([0.2, 0.1, 0.0, -0.2, -0.1], dtype=torch.float32)
        self.relation_bias = nn.Parameter(initial.repeat(num_heads, 1))
        self.immediate_not_last_bias = nn.Parameter(torch.full((num_heads,), -0.2))

        self.delta_head = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        self.history_scale_logit = nn.Parameter(torch.tensor(-2.0))

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline.eval()
        return self

    def _history_node_probabilities(
        self,
        history_features: torch.Tensor,
        history_node_classes: torch.Tensor,
        history_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, history_length, _ = history_features.shape
        if history_length == 0:
            return history_features.new_zeros((batch_size, 0, NUM_GRAPH_NODES))
        if self.graph_source == "oracle":
            safe_classes = history_node_classes.clamp(min=0)
            probabilities = F.one_hot(safe_classes, num_classes=NUM_GRAPH_NODES).float()
        elif self.graph_source == "predicted":
            with torch.no_grad():
                probabilities = F.softmax(self.baseline(history_features), dim=-1)
        else:
            probabilities = history_features.new_zeros(
                (batch_size, history_length, NUM_GRAPH_NODES)
            )
        return probabilities.masked_fill(history_padding_mask.unsqueeze(-1), 0.0)

    def _graph_bias(
        self,
        history_probabilities: torch.Tensor,
        history_position_ids: torch.Tensor,
        history_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, history_length, _ = history_probabilities.shape
        if self.graph_source == "none" or history_length == 0:
            return history_probabilities.new_zeros(
                (batch_size, self.num_heads, NUM_GRAPH_NODES, history_length)
            )
        # [H,V,U]: learned scalar for every head and fixed candidate/history relation.
        pair_bias = self.relation_bias[:, self.relation_ids]
        graph_bias = torch.einsum("blu,hvu->bhvl", history_probabilities, pair_bias)

        # I is fully meaningful only when the observed history token is last (distance=1).
        immediate_matrix = (self.relation_ids == RELATION_TO_ID["I"]).to(history_probabilities.dtype)
        immediate_probability = torch.einsum(
            "blu,vu->bvl", history_probabilities, immediate_matrix
        )
        not_last = (
            (history_position_ids != 1) & (~history_padding_mask)
        ).to(history_probabilities.dtype)
        graph_bias = graph_bias + (
            self.immediate_not_last_bias.view(1, self.num_heads, 1, 1)
            * immediate_probability.unsqueeze(1)
            * not_last.unsqueeze(1).unsqueeze(1)
        )
        return graph_bias

    def forward(
        self,
        current_feature: torch.Tensor,
        history_features: torch.Tensor,
        history_position_ids: torch.Tensor,
        history_node_classes: torch.Tensor,
        history_padding_mask: torch.Tensor,
        **_: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = current_feature.shape[0]
        history_length = history_features.shape[1]
        current = self.current_projection(current_feature)
        history = self.history_projection(history_features)
        if history_length:
            positions = history_position_ids.clamp(min=0, max=self.max_history)
            history = history + self.position_embedding(positions)

        candidates = self.candidate_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        queries = self.query_projection(current.unsqueeze(1) + candidates)
        null = self.null_history.expand(batch_size, -1, -1)
        history_with_null = torch.cat([null, history], dim=1)
        keys = self.key_projection(history_with_null)
        values = self.value_projection(history_with_null)

        queries = queries.view(batch_size, NUM_GRAPH_NODES, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, history_length + 1, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, history_length + 1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.einsum("bhnc,bhlc->bhnl", queries, keys) / math.sqrt(self.head_dim)

        history_probabilities = self._history_node_probabilities(
            history_features, history_node_classes, history_padding_mask
        )
        graph_bias = self._graph_bias(
            history_probabilities, history_position_ids, history_padding_mask
        )
        null_bias = graph_bias.new_zeros((batch_size, self.num_heads, NUM_GRAPH_NODES, 1))
        scores = scores + torch.cat([null_bias, graph_bias], dim=-1)
        null_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=current_feature.device)
        key_padding_mask = torch.cat([null_mask, history_padding_mask], dim=1)
        scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attention = self.attention_dropout(F.softmax(scores, dim=-1))
        context = torch.einsum("bhnl,bhlc->bhnc", attention, values)
        context = context.transpose(1, 2).contiguous().view(batch_size, NUM_GRAPH_NODES, self.d_model)
        context = self.output_projection(context)

        current_expanded = current.unsqueeze(1).expand(-1, NUM_GRAPH_NODES, -1)
        delta_input = torch.cat([current_expanded, context, candidates], dim=-1)
        delta = self.delta_head(delta_input).squeeze(-1)
        scale = torch.sigmoid(self.history_scale_logit)
        with torch.no_grad():
            baseline_logits = self.baseline(current_feature)
        logits = baseline_logits + scale * delta
        return logits, {
            "baseline_logits": baseline_logits,
            "history_delta": delta,
            "history_scale": scale.detach(),
            "attention": attention,
            "graph_bias": graph_bias,
            "history_node_probabilities": history_probabilities,
        }
```
```python
initial = torch.tensor([0.2, 0.1, 0.0, -0.2, -0.1], dtype=torch.float32)
self.relation_bias = nn.Parameter(initial.repeat(num_heads, 1))
self.immediate_not_last_bias = nn.Parameter(torch.full((num_heads,), -0.2))
```
self.relation_bias 将initial 在行的方向上进行复制，得到维度为[4, 5]的可训练张量
self.immediate_not_last_bias 得到维度为[4]，填充值为-0.2的可训练张量。

```python
self.delta_head = nn.Sequential(
    nn.LayerNorm(3 * d_model),
    nn.Linear(3 * d_model, d_model),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_model, 1),
)
nn.init.zeros_(self.delta_head[-1].weight)
nn.init.zeros_(self.delta_head[-1].bias)
self.history_scale_logit = nn.Parameter(torch.tensor(-2.0))
```
构建修正头，这里和初始的M1-M3模型是一样的，是冻结M0模型，使用历史信息模块训练一个修正头来修正M0模型的预测。

```python
def train(self, mode: bool = True):
    super().train(mode)
    self.baseline.eval()
    return self
```
同样只训练历史信息模块，M0模型保持冻结。

```python
def _history_node_probabilities(
    self,
    history_features: torch.Tensor, # 维度[batch_size, length, feature_dim]
    history_node_classes: torch.Tensor, # 维度[batch_size, length]
    history_padding_mask: torch.Tensor,  # 维度[batch_size, length]
) -> torch.Tensor:
    batch_size, history_length, _ = history_features.shape
    if history_length == 0:
        return history_features.new_zeros((batch_size, 0, NUM_GRAPH_NODES))
    # 如果没有历史长度，就新建一个和history_features有相同device, dtype等属性，维度为[batch_size, 0, 35]的全0张量。并返回。
    if self.graph_source == "oracle":
        safe_classes = history_node_classes.clamp(min=0)
        # "oracle"甲骨文表示使用真实的历史节点的类别，就将history_node_classes中的最小值设置为0，因为真实的最小值只能是0（不是开始节点)
        probabilities = F.one_hot(safe_classes, num_classes=NUM_GRAPH_NODES).float()
        # 构建一个维度为[batch_size, length, 35]的duress向量。
    elif self.graph_source == "predicted":
        with torch.no_grad():
            probabilities = F.softmax(self.baseline(history_features), dim=-1)
        # 如果使用预测历史节点，则probabilities维度为[batch_size, length, 35]
    else:
        probabilities = history_features.new_zeros(
            (batch_size, history_length, NUM_GRAPH_NODES)
        )
        # 如果是其他设置，则probabilities是维度为[ batch_size, length, 34]的全0张量
    return probabilities.masked_fill(history_padding_mask.unsqueeze(-1), 0.0)
    # 先将history_padding_mask的维度变为 [batch_size, length, 1]
    # .masked_fill()表示将history_padding_mask.unsqueeze(-1)中True的位置的值变成0，其余地方的值保持不变。
```

对于torch.nn.functional.one_hot()的进一步解释，直接结果上，其会生成一个维度为[*safe_classes.shape, num_classes]的张量。具体可以理解为，其会先构建一下的关系，以num_classes=3为例：0：[1, 0, 0], 1:[0, 1, 0], 2:[0, 0, 1]
然后再将safe_classes这个张量中对应的值替换成对应的ont-hot向量。

```text
3 个类别：0, 1, 2

因此：
0 → [1, 0, 0]
1 → [0, 1, 0]
2 → [0, 0, 1]

于是原来的：
tensor([
    [0, 1],
    [2, 0],
    [1, 2]
])

逐元素转换以后就是：
tensor([
    [
        [1, 0, 0],   # 0
        [0, 1, 0]    # 1
    ],

    [
        [0, 0, 1],   # 2
        [1, 0, 0]    # 0
    ],

    [
        [0, 1, 0],   # 1
        [0, 0, 1]    # 2
    ]
])
```

对于masked_fill(), 首先probabilities维度为[batch_size, length, 35], history_padding_mask维度为[batch_size, length], 补充最后一个维度后变为[batch_size, length, 1], 此时再进行填充是，会有broadcast机制，在最后一个维度，也就是列方向进行重复。一个例子如下：

```text
假设：probabilities.shape = [2, 3, 3]
内容是:
probabilities = tensor([
    [
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.3, 0.5],
    ],

    [
        [0.6, 0.3, 0.1],
        [0.2, 0.2, 0.6],
        [0.1, 0.4, 0.5],
    ]
])

history_padding_mask = tensor([
    [False, False, True],
    [False, True,  True]
])

那history_padding_mask.unsqueeze(-1)后变为
history_padding_mask = tensor(
    [
      [
        [False], 
        [False], 
        [True],
      ],
      [
        [False], 
        [True],  
        [True],
      ]
])

在进行.masked_fill()时会自动将进行广播变成:
history_padding_mask = tensor(
    [
      [
        [False], [False], [False], 
        [False], [False], [False], 
        [True],  [True],  [True],
      ],
      [
        [False], [False], [False], 
        [True],  [True],  [True],
        [True],  [True],  [True],
      ]
])

True的地方填写0，其余地方不变则，probabilities变为：
probabilities = tensor([
    [
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.,   0.,  0.],
    ],

    [
        [0.6, 0.3, 0.1],
        [0.,   0.,  0.],
        [0.,   0.,  0.],
    ]
])
```

```python
def _graph_bias(
    self,
    history_probabilities: torch.Tensor,
    history_position_ids: torch.Tensor,
    history_padding_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, history_length, _ = history_probabilities.shape
    if self.graph_source == "none" or history_length == 0:
        return history_probabilities.new_zeros(
            (batch_size, self.num_heads, NUM_GRAPH_NODES, history_length)
        )
    # [H,V,U]: learned scalar for every head and fixed candidate/history relation.
    pair_bias = self.relation_bias[:, self.relation_ids]
    # relation_bias维度为[4, 5], relation_ids维度为[35, 35]
    # pair_bias 维度为[4, 35, 35]
    graph_bias = torch.einsum("blu,hvu->bhvl", history_probabilities, pair_bias)

    # I is fully meaningful only when the observed history token is last (distance=1).
    immediate_matrix = (self.relation_ids == RELATION_TO_ID["I"]).to(history_probabilities.dtype)
    immediate_probability = torch.einsum(
        "blu,vu->bvl", history_probabilities, immediate_matrix
    )
    not_last = (
        (history_position_ids != 1) & (~history_padding_mask)
    ).to(history_probabilities.dtype)
    graph_bias = graph_bias + (
        self.immediate_not_last_bias.view(1, self.num_heads, 1, 1)
        * immediate_probability.unsqueeze(1)
        * not_last.unsqueeze(1).unsqueeze(1)
    )
    return graph_bias
```
有一些难理解先放一下。

```python
def forward(
    self,
    current_feature: torch.Tensor,
    history_features: torch.Tensor,
    history_position_ids: torch.Tensor,
    history_node_classes: torch.Tensor,
    history_padding_mask: torch.Tensor,
    **_: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = current_feature.shape[0]
    history_length = history_features.shape[1]
    current = self.current_projection(current_feature)
    history = self.history_projection(history_features)
    if history_length:
        positions = history_position_ids.clamp(min=0, max=self.max_history)
        history = history + self.position_embedding(positions)

    candidates = self.candidate_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
    queries = self.query_projection(current.unsqueeze(1) + candidates)
    null = self.null_history.expand(batch_size, -1, -1)
    history_with_null = torch.cat([null, history], dim=1)
    keys = self.key_projection(history_with_null)
    values = self.value_projection(history_with_null)

    queries = queries.view(batch_size, NUM_GRAPH_NODES, self.num_heads, self.head_dim).transpose(1, 2)
    keys = keys.view(batch_size, history_length + 1, self.num_heads, self.head_dim).transpose(1, 2)
    values = values.view(batch_size, history_length + 1, self.num_heads, self.head_dim).transpose(1, 2)
    scores = torch.einsum("bhnc,bhlc->bhnl", queries, keys) / math.sqrt(self.head_dim)

    history_probabilities = self._history_node_probabilities(
        history_features, history_node_classes, history_padding_mask
    )
    graph_bias = self._graph_bias(
        history_probabilities, history_position_ids, history_padding_mask
    )
    null_bias = graph_bias.new_zeros((batch_size, self.num_heads, NUM_GRAPH_NODES, 1))
    scores = scores + torch.cat([null_bias, graph_bias], dim=-1)
    null_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=current_feature.device)
    key_padding_mask = torch.cat([null_mask, history_padding_mask], dim=1)
    scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
    attention = self.attention_dropout(F.softmax(scores, dim=-1))
    context = torch.einsum("bhnl,bhlc->bhnc", attention, values)
    context = context.transpose(1, 2).contiguous().view(batch_size, NUM_GRAPH_NODES, self.d_model)
    context = self.output_projection(context)

    current_expanded = current.unsqueeze(1).expand(-1, NUM_GRAPH_NODES, -1)
    delta_input = torch.cat([current_expanded, context, candidates], dim=-1)
    delta = self.delta_head(delta_input).squeeze(-1)
    scale = torch.sigmoid(self.history_scale_logit)
    with torch.no_grad():
        baseline_logits = self.baseline(current_feature)
    logits = baseline_logits + scale * delta
    return logits, {
        "baseline_logits": baseline_logits,
        "history_delta": delta,
        "history_scale": scale.detach(),
        "attention": attention,
        "graph_bias": graph_bias,
        "history_node_probabilities": history_probabilities,
    }
```
```python
batch_size = current_feature.shape[0]
history_length = history_features.shape[1]
current = self.current_projection(current_feature)
history = self.history_projection(history_features)
if history_length:
    positions = history_position_ids.clamp(min=0, max=self.max_history)
    history = history + self.position_embedding(positions)
```
获得batchsize，历史长度，当前节点的特征映射，以及历史节点的特征映射，并且为历史节点添加上位置编码。

```python
candidates = self.candidate_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
queries = self.query_projection(current.unsqueeze(1) + candidates)
null = self.null_history.expand(batch_size, -1, -1)
history_with_null = torch.cat([null, history], dim=1)
keys = self.key_projection(history_with_null)
values = self.value_projection(history_with_null)
```
candidates： self.candidate_embedding 的维度为[35, 256],先增添一维变成[1, 35, 256], 再进行扩展变成[batch_size, 35, 256]
queries: current的维度应该为：[batch_size, 256], 其先添加一维变成[batch_size, 1, 256], 再与candidates相加，得到[batch_size, 35, 256]
self.null_history维度为:[1, 1, 256], 进行扩展后变为[batch_size, 1, 256]
将history 和 null沿着第一维度进行拼接，得到history_with_null维度为：[batch_size, length+1, 256]
随后对history_with_null进行映射得到keys 和 values

```python
queries = queries.view(batch_size, NUM_GRAPH_NODES, self.num_heads, self.head_dim).transpose(1, 2)
keys = keys.view(batch_size, history_length + 1, self.num_heads, self.head_dim).transpose(1, 2)
values = values.view(batch_size, history_length + 1, self.num_heads, self.head_dim).transpose(1, 2)
scores = torch.einsum("bhnc,bhlc->bhnl", queries, keys) / math.sqrt(self.head_dim)
```
queries原本维度为: [batch_size, 35, 256], 将其变成[batch_size, 35, 4, 64]并调换一下维度得到新的维度:
[batch_size, 4, 35, 64]
keys和values 也进行类似操作后得到新的维度为: [batch_size, 4, length+1, 64]
计算特征间相似度得到如下维度张量:
[batch_size, 4, 34, length+1]

```python
history_probabilities = self._history_node_probabilities(
    history_features, history_node_classes, history_padding_mask
)
graph_bias = self._graph_bias(
    history_probabilities, history_position_ids, history_padding_mask
)
```
分别计算历史节点的概率和图偏置。

```python
null_bias = graph_bias.new_zeros((batch_size, self.num_heads, NUM_GRAPH_NODES, 1))
```
null没有图偏置，所以为其添加一个，维度为[batch_size, 4, 35, 1]的全零张量

```python
scores = scores + torch.cat([null_bias, graph_bias], dim=-1)
null_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=current_feature.device)
key_padding_mask = torch.cat([null_mask, history_padding_mask], dim=1)
scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
```
先将图偏置添加到特征相似度中，得到scores维度为:[batch_size, 4, 35, length+1]
再将null位置对应的mask也添加上。
得到key_padding_mask维度为: [batch_size, length+1]
使用masked_fill()将不合法，不存在的历史特征填充为float32的最小值，最后scores维度还是[batch_size, 4, 35, length+1]。
这里key_padding_mask[:, None, None, :] 中间的None, None, 表示在这两个位置扩展一个维度。
torch.finfo(scores.dtype).min：finfo 是 floating-point information，专门用来查询浮点类型的数值范围。
所以表示是取scores这个数据类型能表达的最小值。

```python
attention = self.attention_dropout(F.softmax(scores, dim=-1))
context = torch.einsum("bhnl,bhlc->bhnc", attention, values)
context = context.transpose(1, 2).contiguous().view(batch_size, NUM_GRAPH_NODES, self.d_model)
context = self.output_projection(context)
```
使用softmax和dropout得到最终的注意力分数，维度还是[batch_size, 4, 35, length+1]
然后"bhnl,bhlc->bhnc"得到有上下文信息的新的特征，
维度变化为"bhnl": [batch_size, 4, 35, length+1] "bhlc": [batch_size, 4, length+1, 64], "bhnc": [batch_size, 4, 35, 64]
最后进行维度处理，先转置成 [batch_size, 35, 4, 64],再变换维度到[batch_size, 35, 256], 最后再经过一个线性映射得到最终输出。

```python
current_expanded = current.unsqueeze(1).expand(-1, NUM_GRAPH_NODES, -1)
delta_input = torch.cat([current_expanded, context, candidates], dim=-1)
delta = self.delta_head(delta_input).squeeze(-1)
scale = torch.sigmoid(self.history_scale_logit)
with torch.no_grad():
    baseline_logits = self.baseline(current_feature)
logits = baseline_logits + scale * delta
return logits, {
    "baseline_logits": baseline_logits,
    "history_delta": delta,
    "history_scale": scale.detach(),
    "attention": attention,
    "graph_bias": graph_bias,
    "history_node_probabilities": history_probabilities,
}
```
current_expanded：current 特征进行扩充由[batch_size, 256], 变成[batch_size, 35, 256]
delta_input： 将扩充后的current，上下文信息，和candidates编码进行拼接，得到[batch_size, 35, 768]
送入self.delat_head,得到delat输出为[batch_size, 35, 1], 再squeze变成[batch_size, 35]
最后将baseline的输出和修正delta进行结合得到最终的输出。

```python
def build_context_model(
    model_name: str,
    baseline: FeatureNodeClassifier,
    relation_ids: torch.Tensor,
    feature_dim: int,
    d_model: int,
    num_heads: int,
    max_history: int,
    dropout: float,
) -> nn.Module:
    if model_name == "m1":
        return SingleQueryHistoryModel(
            baseline, feature_dim, d_model, num_heads, max_history, dropout, use_position=False
        )
    if model_name in {"m2", "m3"}:
        return SingleQueryHistoryModel(
            baseline, feature_dim, d_model, num_heads, max_history, dropout, use_position=True
        )
    graph_sources = {"m4": "none", "m5": "oracle", "m6": "predicted"}
    if model_name in graph_sources:
        return CandidateHistoryModel(
            baseline=baseline,
            relation_ids=relation_ids,
            graph_source=graph_sources[model_name],
            feature_dim=feature_dim,
            d_model=d_model,
            num_heads=num_heads,
            max_history=max_history,
            dropout=dropout,
        )
    raise ValueError(f"Not a context model: {model_name}")


def build_direct_context_model(
    model_name: str,
    feature_dim: int,
    d_model: int,
    num_heads: int,
    max_history: int,
    dropout: float,
) -> DirectSingleQueryHistoryModel:
    """Build the isolated direct-head variants without loading or freezing M0."""
    if model_name == "m1_direct":
        use_position = False
    elif model_name in {"m2_direct", "m3_direct"}:
        use_position = True
    else:
        raise ValueError(f"Not a direct-head context model: {model_name}")
    return DirectSingleQueryHistoryModel(
        feature_dim=feature_dim,
        d_model=d_model,
        num_heads=num_heads,
        max_history=max_history,
        dropout=dropout,
        use_position=use_position,
    )
```
用于创建不同的模型。

## metrics.py
### def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
```python
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    # y_true和y_pred的维度应该是两个一维张量。
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    # 初始化一个全零的矩阵
    for truth, pred in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= truth < num_classes and 0 <= pred < num_classes:
            matrix[truth, pred] += 1
            # 将对应位置的值增加1，每一行是一个ground truth类别，列是预测类别
    return matrix
```

```python
def metrics_from_confusion(matrix: np.ndarray) -> dict[str, Any]:
    support = matrix.sum(axis=1)
    # 计算每个类别实际上有多少个样本
    predicted = matrix.sum(axis=0)
    # 计算有多少个样本被预测为每一个类别
    tp = np.diag(matrix).astype(np.float64)
    # 预测正确的样本数量
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    # recall = tp/(tp+fn)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    # precision = tp/(tp+fp)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    # f1 = 2*recall*precision / (recall + precision)
    present = support > 0
    total = matrix.sum()
    return {
        "accuracy": float(tp.sum() / total) if total else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "present_class_count": int(present.sum()),
        "total_class_count": int(matrix.shape[0]),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": matrix.tolist(),
    }
```

```python
def classification_metrics(y_true: list[int], y_pred: list[int], num_classes: int) -> dict[str, Any]:
    true_array = np.asarray(y_true, dtype=np.int64)
    pred_array = np.asarray(y_pred, dtype=np.int64)
    return metrics_from_confusion(confusion_matrix(true_array, pred_array, num_classes))
```
用于调用前两个函数进行参数计算。

```python
def aggregate_node_probabilities(
    node_probabilities: torch.Tensor, # [batch_size, length, 35]
    node_to_tier3: torch.Tensor, # [35]
) -> torch.Tensor:
    output_shape = (*node_probabilities.shape[:-1], num_tier3)
    # [batch_size, length, 31]
    result = torch.zeros(output_shape, device=node_probabilities.device, dtype=node_probabilities.dtype)
    index = node_to_tier3.view(*([1] * (node_probabilities.ndim - 1)), -1).expand_as(node_probabilities)
    # node_to_tier3.view(*([1] * (node_probabilities.ndim - 1)), -1) 变成[[[31]]] 即维度为[1, 1, 35]
    # .expand_as() 变成维度[batch_size, length, 35]
    result.scatter_add_(-1, index, node_probabilities)
    # index维度: [batch_size, length, 35] node_probabilities维度: [batch_size, length, 35]
    # result 维度: [batch_size, length, 31]
    return result
```
这里需要注意的是.scatter_add_(dim, index, values)的用法，一般来说result, index, values的ndim都是相同的，并且index和values维度完全相同。当指定哪一个dim后，其他dim在result, index, values中正常对应的索引，但是指定维度中的值，变成指定维度取值时的index 了。
```text
result = torch.zeros(2, 4)

index = torch.tensor([
    [0, 2, 0],
    [1, 3, 1]
])

src = torch.tensor([
    [10., 20., 30.],
    [40., 50., 60.]
])

result.scatter_add_(dim=1, index=index, src=src)

因为：
dim=1

所以：
index 控制列号，行号保持原来的位置。

第一行：
src[0,0]=10
index[0,0]=0
→ result[0,0] += 10

src[0,1]=20
index[0,1]=2
→ result[0,2] += 20

src[0,2]=30
index[0,2]=0
→ result[0,0] += 30

于是第一行：
[40, 0, 20, 0]

第二行：
40 → 第1列
50 → 第3列
60 → 第1列

得到：
[0,100,0,50]

最终：
tensor([
    [40.,   0., 20.,  0.],
    [ 0., 100.,  0., 50.]
])
```

## engine.py
### def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
```python
def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
```
这里如果字典的值是tensor, 就将其送到指定设备，如果不是则保持原样。
留意这里的用法key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value，可以写成
key: （value.to(device, non_blocking=True) if torch.is_tensor(value) else value）
也可以对key进行调整比如 ("tensor_"+key if torch.is_tensor(value) else key): value

### def forward_node_model(model: torch.nn.Module, batch: dict[str, Any]):
```python
def forward_node_model(model: torch.nn.Module, batch: dict[str, Any]):
    if isinstance(model, FeatureNodeClassifier):
        logits = model(batch["current_feature"])
        return logits, {}
    return model(
        current_feature=batch["current_feature"],
        history_features=batch["history_features"],
        history_position_ids=batch["history_position_ids"],
        history_node_classes=batch["history_node_classes"],
        history_padding_mask=batch["history_padding_mask"],
    )
``` 
将节点分类头单独拿出来

### def compute_loss(）
```python
def compute_loss(
    logits: torch.Tensor,
    node_target: torch.Tensor,
    tier3_target: torch.Tensor,
    node_to_tier3: torch.Tensor,
    action_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    node_loss = F.cross_entropy(logits, node_target)
    loss = node_loss
    action_loss = logits.new_tensor(0.0)
    # 参考logits的device 和 dtype新建一个值为0.0的张量
    if action_loss_weight > 0:
        node_probabilities = F.softmax(logits, dim=-1)
        action_probabilities = aggregate_node_probabilities(
            node_probabilities, node_to_tier3, NUM_TIER3_CLASSES
        )
        selected = action_probabilities.gather(1, tier3_target[:, None]).squeeze(1).clamp_min(1e-12)
        # action_probabilities 维度为: [batch_size, length, 31]
        # tier3_target[:, None]维度变为: [batch_size, length, 1]
        ## 筛选完维度变为: [batch_size, length, 1] 再squeeze去，最终得到 [batch_size, length]
        action_loss = -selected.log().mean()
        # 也就是实现cross_entropy loss
        loss = loss + float(action_loss_weight) * action_loss
    return loss, {
        "node_loss": float(node_loss.detach()),
        "action_loss": float(action_loss.detach()),
    }
```
这里要注意gather的用法，其和scatter一个原理，index, 和 src 要保持完全一致的维度，指定维度具体的src中的值，由index决定，其他维度正常对应索引。

### def train_feature_model()
```python
def train_feature_model(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    node_to_tier3: torch.Tensor,
    epochs: int,
    action_loss_weight: float = 0.0,
    amp: bool = False,
) -> list[dict[str, float]]:
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    history: list[dict[str, float]] = []
    node_to_tier3 = node_to_tier3.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.time()
        loss_sum = 0.0
        correct = 0
        total = 0
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits, _ = forward_node_model(model, batch)
                loss, _ = compute_loss(
                    logits,
                    batch["node_target"],
                    batch["tier3_target"],
                    node_to_tier3,
                    action_loss_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(batch["node_target"].shape[0])
            loss_sum += float(loss.detach()) * batch_size
            correct += int((logits.argmax(dim=-1) == batch["node_target"]).sum())
            total += batch_size
        row = {
            "epoch": float(epoch),
            "train_loss": loss_sum / max(1, total),
            "train_node_accuracy": correct / max(1, total),
            "seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d}/{epochs:03d} "
            f"loss={row['train_loss']:.6f} node_acc={row['train_node_accuracy']:.4f} "
            f"seconds={row['seconds']:.1f}",
            flush=True,
        )
    return history
```
```python
scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
history: list[dict[str, float]] = []
node_to_tier3 = node_to_tier3.to(device)
```
创建混合精度训练用的梯度缩放器。

```python
for epoch in range(1, epochs + 1):
    model.train()
    started = time.time()
    loss_sum = 0.0
    correct = 0
    total = 0
```
每一个epoch将模型设置为训练模型，初始化损失，预测正确样本数量，以及总样本数量。并且记录开始时间。

```python
for raw_batch in loader:
    batch = move_batch_to_device(raw_batch, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
        logits, _ = forward_node_model(model, batch)
        loss, _ = compute_loss(
            logits,
            batch["node_target"],
            batch["tier3_target"],
            node_to_tier3,
            action_loss_weight,
        )
```
对于每一个batch, 先将数据送到GPU, 然后启动自动混合精度训练上下文，计算logits和损失。

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(
    [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
)
scaler.step(optimizer)
scaler.update()
```
这里因为启动了自动混合精度训练，有些位置可能使用floatpoint 16进行运算，这导致其能表示的最小值远远不如正常的fp32,所以
scaler.scale(loss).backward() 将损失放大后，再进行反向传播，可以获得能够用fp16表示的梯度。
但在真正优化时，我们要优化的是原始的梯度，而不是放大后的梯度，所以要记得将scaler.unscale_(optimizer) 还原一下。
torch.nn.utils.clip_grad_norm_(
    [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
)
这里表示进行梯度裁剪，将所有由梯度的参数的整体的范数的最大值限制为1.
scaler.step(optimizer)
scaler.update()
优化参数，并更新缩放器。

整体理解上，scaler中保存着损失要缩放的倍数，在前向传播得到损失后，将损失先按照倍数方法，然后反向传播获得梯度，现在有梯度的parameter.grad中保存的是放大后的梯度，所以在更新参数时，要让optimier知道真实的梯度，所以要unscale，以使用正常未缩放的梯度进行参数更新。（optimizer 本身就保存着它所管理的所有 parameter 的引用。unscale_() 并不是修改 optimizer，而是借助 optimizer 找到所有 parameter，再修改它们的 .grad）
更新完成后，要对scaler进行更新，得到新的缩放倍数。

```python
batch_size = int(batch["node_target"].shape[0])
loss_sum += float(loss.detach()) * batch_size
correct += int((logits.argmax(dim=-1) == batch["node_target"]).sum())
total += batch_size
```

获得batch_size大小，统计损失，已经训练过的样本数量以及预测正确的样本数量。

```python
row = {
        "epoch": float(epoch),
        "train_loss": loss_sum / max(1, total),
        "train_node_accuracy": correct / max(1, total),
        "seconds": time.time() - started,
    }
    history.append(row)
    print(
        f"epoch={epoch:03d}/{epochs:03d} "
        f"loss={row['train_loss']:.6f} node_acc={row['train_node_accuracy']:.4f} "
        f"seconds={row['seconds']:.1f}",
        flush=True,
    )
return history
```
保存并打印一轮训练的结果。

### evaluate_feature_model()
```python
@torch.no_grad()
def evaluate_feature_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    node_to_tier3: torch.Tensor,
    output_dir: str | Path,
    split_name: str,
) -> dict[str, Any]:
    model.eval()
    node_to_tier3 = node_to_tier3.to(device)
    node_true: list[int] = []
    node_pred: list[int] = []
    tier3_true: list[int] = []
    tier3_pred: list[int] = []
    stages: list[int] = []
    rows: list[dict[str, Any]] = []
    all_node_probabilities: list[torch.Tensor] = []

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        logits, _ = forward_node_model(model, batch)
        node_probabilities = F.softmax(logits, dim=-1)
        tier3_probabilities = aggregate_node_probabilities(
            node_probabilities, node_to_tier3, NUM_TIER3_CLASSES
        )
        predicted_nodes = node_probabilities.argmax(dim=-1)
        predicted_actions = tier3_probabilities.argmax(dim=-1)
        all_node_probabilities.append(node_probabilities.cpu())

        for index in range(logits.shape[0]):
            truth_node = int(batch["node_target"][index])
            pred_node = int(predicted_nodes[index])
            truth_action = int(batch["tier3_target"][index])
            pred_action = int(predicted_actions[index])
            stage = int(batch["stage_id"][index])
            node_true.append(truth_node)
            node_pred.append(pred_node)
            tier3_true.append(truth_action)
            tier3_pred.append(pred_action)
            stages.append(stage)
            rows.append(
                {
                    "sample_name": raw_batch["sample_name"][index],
                    "participant": raw_batch["participant"][index],
                    "run": raw_batch["run"][index],
                    "annotation_row_index": raw_batch["annotation_row_index"][index],
                    "stage_id": stage,
                    "true_node_idx": truth_node + 1,
                    "pred_node_idx": pred_node + 1,
                    "true_tier3_id": truth_action,
                    "pred_tier3_id": pred_action,
                    "node_confidence": float(node_probabilities[index, pred_node]),
                    "tier3_confidence": float(tier3_probabilities[index, pred_action]),
                }
            )

    metrics: dict[str, Any] = {
        "split": split_name,
        "samples": len(rows),
        "node": classification_metrics(node_true, node_pred, NUM_GRAPH_NODES),
        "tier3": classification_metrics(tier3_true, tier3_pred, NUM_TIER3_CLASSES),
        "per_stage": {},
    }
    for stage in (1, 2, 3):
        indices = [idx for idx, value in enumerate(stages) if value == stage]
        metrics["per_stage"][str(stage)] = {
            "samples": len(indices),
            "node": classification_metrics(
                [node_true[idx] for idx in indices],
                [node_pred[idx] for idx in indices],
                NUM_GRAPH_NODES,
            ),
            "tier3": classification_metrics(
                [tier3_true[idx] for idx in indices],
                [tier3_pred[idx] for idx in indices],
                NUM_TIER3_CLASSES,
            ),
        }

    output_dir = ensure_dir(output_dir)
    write_json(output_dir / f"{split_name}_metrics.json", metrics)
    with (output_dir / f"{split_name}_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    torch.save(
        {
            "node_probabilities": torch.cat(all_node_probabilities, dim=0)
            if all_node_probabilities else torch.empty((0, NUM_GRAPH_NODES)),
            "rows": rows,
        },
        output_dir / f"{split_name}_probabilities.pt",
    )
    return metrics
```
先跳过，主要是测试模型并将结果和一些输出文件保存并记录下来。


