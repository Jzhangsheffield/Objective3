# 完整实验配置说明：M0–M6、E2E、Direct与Dynamic

首次建立：2026-07-28

完整恢复与更新：2026-07-29

状态：记录当前严格四折三seed的全部实验配置；Direct和Dynamic均为独立新增阶段，不修改原实验

## 1. 文档范围

本文件统一说明以下四组实验：

1. 冻结RGB特征实验：M0–M6；
2. RGB端到端对照：E2E-Tier3-Scratch、E2E-Node-Scratch、E2E-Node-From-Tier3；
3. Direct Head Fusion：M1 Direct、M2 Direct、M3 Direct；
4. Dynamic Epoch Shuffle：Frozen-M0 Delta、Joint-Head Delta、Direct Fusion。

正式结果使用A/D/J/M严格四折、seed 1/2/42、normal-only/all-runs两个完整pipeline。
旧实验的checkpoint、prediction、metrics、summary和完成标记均保留；Direct和Dynamic分别只写入
各自的新目录。

## 2. 所有实验共同遵守的协议

### 2.1 数据与LOSO

| 配置 | 值 | 作用与解释 |
|---|---|---|
| 模态 | RGB | 不使用深度、热成像或其他相机信号 |
| 相机 | `001484412812` | 全部模型使用同一固定视角，避免相机差异成为混杂因素 |
| participant | A、D、J、M | 四位参与者分别作为held-out测试对象 |
| fold | 严格LOSO | 每折只用另外三位参与者训练；held-out participant不进入训练或模型选择 |
| seed | 1、2、42 | 控制模型初始化、DataLoader shuffle及M3确定性graph-valid重排 |
| validation | 无 | 不划分validation，也不使用held-out数据调参 |
| checkpoint | 最后epoch `last.pth` | 不做early stopping，不选择best checkpoint |

四折为：

| Fold | 训练参与者 | 测试参与者 |
|---|---|---|
| A-as-test | D、J、M | A |
| D-as-test | A、J、M | D |
| J-as-test | A、D、M | J |
| M-as-test | A、D、J | M |

### 2.2 两种train scope

| Scope | 训练数据 | 测试数据 | 解释 |
|---|---|---|---|
| `normal_only` | 训练参与者的normal runs | held-out的normal、fault、all | 检验只从标准流程学习后对故障流程的泛化 |
| `all_runs` | 训练参与者的normal + fault runs | held-out的normal、fault、all | 检验训练时纳入真实故障/偏离流程数据的效果 |

当前严格正式结果中，两个scope各自拥有从scratch训练的Tier3 backbone、独立feature cache、
M0–M6和E2E模型。`representation_scope`必须与`train_scope`一致：

```text
normal_only backbone → normal_only features → normal_only models
all_runs backbone    → all_runs features    → all_runs models
```

早期曾有“normal-only backbone + all-runs history model”的辅助条件，它只用于历史探索，不进入
当前四折三seed严格scope比较，不能与完整`all_runs → all_runs`结果混合。

### 2.3 测试split

| Split | 内容 |
|---|---|
| `test_normal` | held-out participant的normal runs |
| `test_fault` | held-out participant的fault runs |
| `test_all` | 两者合并 |

固定测试规模：

| Held-out | normal clips | fault clips | all clips |
|---|---:|---:|---:|
| A | 294 | 137 | 431 |
| D | 400 | 62 | 462 |
| J | 387 | 168 | 555 |
| M | 360 | 87 | 447 |

测试数据只在最终checkpoint写出后加载。训练过程中不会使用测试指标选择epoch。

## 3. RGB输入、预处理与Tier3 backbone

### 3.1 Clip构造与图像变换

| 配置 | 值 | 详细解释 |
|---|---|---|
| 每clip帧数 | 16 | 使用`linspace(0, last_frame, 16)`在整个clip内均匀取帧；帧数不足时允许重复索引 |
| 输入大小 | 224 × 224 | 所有视频帧使用相同空间大小 |
| RGB mean | `[0.5369, 0.5295, 0.5208]` | 逐通道标准化均值 |
| RGB std | `[0.2311, 0.2360, 0.2363]` | 逐通道标准化标准差 |

训练增强：

- `RandomResizedCrop(224)`，scale为`0.6–1.0`，aspect ratio为`0.75–1.3333`；
- ColorJitter的brightness/contrast/saturation为`0.24`，hue为`0.16`，整体启用概率`0.5`；
- RandomGrayscale概率`0.5`；
- horizontal/vertical flip概率均为`0.0`，即代码中保留算子但实际不翻转，避免破坏操作方向。

测试变换只做`Resize(224,224)`、float缩放和相同mean/std标准化，不使用随机增强。

### 3.2 Backbone结构

| 配置 | 值 | 详细解释 |
|---|---|---|
| 网络 | ResNet3D-18 | 3D卷积同时编码时间和空间信息 |
| block数 | `[2,2,2,2]` | 对应四个ResNet stage |
| 最终特征 | 512维 | global average pooling后、fc分类头前的clip表示 |
| Tier3输出 | 31类 | backbone预训练任务是31类动作识别 |
| 初始化 | scratch | 每个participant、seed、scope独立从随机初始化训练 |

### 3.3 Backbone训练配置

| 配置 | 值 | 详细解释 |
|---|---:|---|
| epochs | 100 | 完整训练RGB视频网络 |
| batch size | 16 | 受3D视频张量显存占用限制 |
| optimizer | AdamW | 对全部backbone参数优化 |
| learning rate | `1e-4` | 初始学习率 |
| weight decay | `1e-4` | AdamW解耦权重衰减 |
| scheduler | MultiStepLR | epoch 50和75后将学习率乘`0.1` |
| loss | 31类Cross Entropy | 直接监督Tier3标签 |
| gradient clip | `1.0` | 限制梯度范数 |
| AMP | 默认开启 | CUDA环境下使用混合精度 |
| checkpoint | epoch 100 `last.pth` | 无validation和best checkpoint |

### 3.4 冻结feature cache

训练完成后，从对应scope的`last.pth`提取每个clip的512维fc前特征：

```text
features/retrained_normal_only/train_all.pt
features/retrained_normal_only/test_all.pt

features/retrained_all_runs/train_all.pt
features/retrained_all_runs/test_all.pt
```

M0–M6和Direct模型只读取这些缓存，因此它们训练时RGB backbone保持冻结。cache metadata保存实际
Tier3 checkpoint路径，用于追踪表征来源。每折、seed、scope内的模型共享同一份feature cache，
保证模型间比较只改变history/head结构。

## 4. Task Graph、标签与history构造

### 4.1 输出标签

- node目标：35个Task Graph node，代码内部使用0–34，原始`node_idx`为1–35；
- Tier3目标：31类动作；
- 35-node模型先得到35维softmax，再按照固定node-to-Tier3映射对概率求和得到31维概率；
- Tier3结果不是把node argmax简单映射成动作，而是聚合完整概率分布。

### 4.2 因果history

每个当前clip只能访问：

```text
相同 participant
+ 相同 run
+ annotation_row_index 更小的clips
```

不会访问未来clip，不会跨run读取历史，也没有全局memory bank。最大history长度为35。
batch内不同长度用padding mask屏蔽；另加入可学习`null_history` token，使run首个clip也有合法attention
输入。

位置ID表示当前展示序列中的距离：

```text
1 = 最近一个history token
2 = 倒数第二个
...
```

### 4.3 actual与graph-valid顺序

| History order | 使用模型 | 解释 |
|---|---|---|
| actual | M1、M2、M4–M6、M1/M2 Direct | 保留数据中真实发生顺序 |
| graph-valid | M3、M3 Direct | 只根据已观察history node及Task Graph约束做确定性随机拓扑重排 |

graph-valid重排不读取当前clip真实标签，避免label leakage。随机性由
`SHA256(seed + sample_name)`稳定确定，同一seed和sample每次运行得到相同顺序。它保留atomic
sequence，并满足已观察node之间的`all_must_previous`依赖；如果history中出现重复graph node，
则回退到actual order。

### 4.4 Relation matrix

关系矩阵方向固定为：

```text
row    = 当前候选node
column = 历史node
```

关系类型编码为`I/M/O/X/S`，其矩阵内容固定，不在训练中修改。M5/M6为每个attention head学习
五种关系各自的scalar bias。初始值依次为：

```text
I: +0.2
M: +0.1
O:  0.0
X: -0.2
S: -0.1
```

对`I`关系，如果相应history token不是最近位置（position ID不等于1），每个head另有初始值
`-0.2`的可学习惩罚。所有关系都只是attention logit bias，包含X在内均不是硬mask。

## 5. M0：冻结特征node baseline

### 5.1 结构

```text
512-D frozen RGB feature
→ LayerNorm(512)
→ Dropout(0.0)
→ Linear(512, 35)
→ node logits
```

M0只使用当前clip，不使用history或Task Graph。它回答“同一冻结Tier3视觉表征本身能做到多少
35-node识别”，并作为M1–M6的冻结logit baseline。

### 5.2 训练

M0使用第10节的feature-level公共配置：50 epochs、batch 64、AdamW、lr `1e-3`、
weight decay `1e-4`、node Cross Entropy、无scheduler、AMP默认关闭、gradient clip `1.0`。

## 6. 原M1–M3：single-query history delta

### 6.1 公共结构

原M1–M3先加载同participant/seed/scope已经训练完成的M0并完全冻结：

```text
baseline_logits = frozen_M0(current_feature)
history_delta   = DeltaHead(current, Attention(current, history))
final_logits    = baseline_logits + sigmoid(scale_logit) × history_delta
```

详细配置：

| 部件 | 配置 | 解释 |
|---|---|---|
| current projection | Linear 512→256 + LayerNorm | 将当前特征投影到history attention空间 |
| history projection | Linear 512→256 + LayerNorm | 对每个历史clip独立投影 |
| position embedding | 36 × 256 | ID 0用于padding，1–35表示历史距离；M1禁用，M2/M3启用 |
| attention | 256维、4 heads、dropout 0.1 | 一个当前query读取全部同run causal history和null token |
| delta head | LN(512) → Linear 512→256 → GELU → Dropout 0.1 → Linear 256→35 | 根据当前表示与history context预测35维修正量 |
| delta末层初始化 | weight/bias全0 | 训练开始时delta为0，模型等价于冻结M0 |
| history scale | `sigmoid(-2)≈0.119`起始 | 单个可学习标量控制delta强度 |

只有history模块、delta head和scale参与训练，M0参数保持冻结。

### 6.2 M1：无位置编码

| 配置 | 值 |
|---|---|
| history order | actual |
| position embedding | 不加入history |
| relation bias | 无 |

M1只知道“哪些历史视觉特征出现过”，不知道它们在序列中离当前clip有多远。它是history内容本身的
基础对照。

### 6.3 M2：实际顺序 + 位置编码

| 配置 | 值 |
|---|---|
| history order | actual |
| position embedding | 启用 |
| relation bias | 无 |

M2在M1基础上增加相对当前clip的历史距离。M2−M1用于衡量显式序列位置的贡献。

### 6.4 M3：graph-valid重排 + 位置编码

| 配置 | 值 |
|---|---|
| history order | graph-valid deterministic shuffle |
| position embedding | 启用 |
| relation bias | 无 |

M3结构与M2相同，只改变history展示顺序。M3用于检验模型是否必须依赖数据中的精确执行顺序，
或只要顺序满足Task Graph约束仍能获得历史收益。

## 7. 原M4–M6：candidate history与relation消融

### 7.1 公共结构

M4–M6同样加载并冻结M0，最终仍预测logit delta。与M1–M3不同，它们为35个候选node分别构造query：

```text
candidate_query_v = Project(current_256 + candidate_embedding_v)
candidate_context_v = Attention(candidate_query_v, history)
delta_v = Head([current_256; candidate_context_v; candidate_embedding_v])
final_logits_v = frozen_M0_logits_v + sigmoid(scale_logit) × delta_v
```

| 部件 | 配置 | 解释 |
|---|---|---|
| feature projection | current/history各512→256 + LayerNorm | 形成统一attention空间 |
| position embedding | 36 × 256，启用 | 所有M4–M6使用实际顺序和距离 |
| candidate embedding | 35 × 256 | 每个候选node拥有独立可学习表示 |
| attention | 4 heads，每head 64维 | 每个候选node分别从history取信息 |
| Q/K/V/O projection | 256→256 | 自定义candidate attention |
| delta head | LN(768) → Linear 768→256 → GELU → Dropout 0.1 → Linear 256→1 | 为每个候选node输出一个delta |
| delta与scale初始化 | 末层全0，scale logit为-2 | 初始输出等价于M0 |

### 7.2 M4：无relation

| 配置 | 值 |
|---|---|
| graph source | `none` |
| history node identity | 不提供 |
| relation bias | 返回全0 |

M4保留candidate-specific attention和位置编码，但Task Graph关系不参与attention。它是M5/M6的
结构匹配对照。

### 7.3 M5：Oracle relation

| 配置 | 值 |
|---|---|
| graph source | `oracle` |
| history node identity | 真实历史node one-hot |
| relation bias | 对真实history node查I/M/O/X/S并加到attention score |

M5直接读取历史真实node标签，只用于估计relation信息理想可用时的上限。因为部署时通常不知道历史
真实node，所以M5不能作为可部署主结果。

### 7.4 M6：Soft predicted relation

| 配置 | 值 |
|---|---|
| graph source | `predicted` |
| history node identity | 冻结M0对每个历史clip产生35-node softmax |
| relation bias | 对所有可能history node的relation bias按M0概率加权 |

M6不使用历史真值，推理时可部署。它保留M0预测不确定性，而不是先做hard argmax。
M6−M4衡量soft relation bias的额外贡献；M6−M5反映可部署估计与Oracle之间的差距。

## 8. 三个E2E对照

### 8.1 E2E-Tier3-Scratch

这就是第3节从scratch训练的31类ResNet3D-18。新增E2E汇总阶段只重新加载其`last.pth`并评估，
不重新训练、不复制或修改权重。

- 输出：31类Tier3；
- 训练：100 epochs，配置见第3.3节；
- Node指标：不存在，汇总中保持为空，不能伪造35-node结果。

### 8.2 E2E-Node-Scratch

```text
RGB clips → randomly initialized ResNet3D-18 → Linear(512,35)
```

- 整个网络从scratch训练35-node；
- 100 epochs、batch 16、AdamW、lr/weight decay均为`1e-4`；
- MultiStepLR milestones `[50,75]`、gamma `0.1`；
- node Cross Entropy、gradient clip `1.0`、AMP默认开启；
- 最终35-node概率再聚合为Tier3概率。

它用于比较“冻结Tier3特征 + history”与“直接从RGB端到端学习node”的差异。

### 8.3 E2E-Node-From-Tier3

```text
加载同fold/seed/scope Tier3 last.pth
→ 加载除fc.weight/fc.bias外的全部backbone参数
→ 替换为随机35-node fc
→ 全网络联合微调100 epochs
```

这里不是只训练fc，也不是冻结backbone；Tier3预训练只提供初始化，之后全部ResNet3D-18参数和
35-node fc共同更新。其他优化配置与E2E-Node-Scratch相同。

## 9. 新增Direct Head Fusion：M1–M3 Direct

### 9.1 研究问题与隔离原则

原M1–M3学习delta修正冻结M0：

```text
final logits = frozen M0 logits + learned history delta
```

Direct实验改为：

```text
frozen Tier3 RGB feature
→ current/history fusion
→ trainable 35-node classifier
→ final logits
```

Direct不加载M0 checkpoint、不使用M0 logits、不预测delta。它复用对应scope的Tier3 feature
cache，因此冻结的是RGB backbone；feature-level history fusion和新node head在同一个50-epoch
阶段联合训练。它不需要重新进行100-epoch视频backbone训练。

### 9.2 Direct公共结构

设当前冻结视觉特征`x ∈ R^512`，attention context为`c ∈ R^256`：

```text
q = LayerNorm(Linear_512→256(x))
h = LayerNorm(Linear_512→256(history)) + optional_position
c = MultiHeadAttention(q, [null; h], [null; h])
x_fused = Linear_768→512([x; c])
logits = Linear_512→35(LayerNorm(x_fused))
```

| 部件 | 配置 | 详细解释 |
|---|---|---|
| current/history projection | 分别512→256 + LayerNorm | 与原M1–M3使用相同history attention维度 |
| position embedding | 36 × 256 | M1 Direct禁用；M2/M3 Direct启用 |
| attention | 256维、4 heads、dropout 0.1 | 一个当前query读取同run causal history |
| fusion | Linear(512+256, 512) | 把history context写回512维视觉特征空间 |
| node head | LayerNorm(512) + Linear(512,35) | 直接生成最终node logits |
| M0 checkpoint | 不使用 | 避免继承旧node head及其logit约束 |
| delta/scale | 不存在 | 最终预测完全来自Direct head |

Fusion使用identity-safe初始化：

```text
W_fusion[:, current_512] = I_512
W_fusion[:, history_256] = 0
b_fusion = 0
```

因此初始`x_fused = x`，训练开始时先等价于在冻结Tier3特征上学习node head，之后再逐步学习history
修正。35-node head随机初始化。

每个Direct模型共有`949,027`个参数，全部可训练，其中fusion层`393,728`个参数，node classifier
`18,979`个参数。这里的“fusion模块”包含current/history projections、position embedding、
null token、multi-head attention和768→512 fusion层；视觉ResNet3D-18本身不更新。

### 9.3 M1 Direct

| 配置 | 值 |
|---|---|
| history order | actual |
| position embedding | 禁用 |
| 输出 | fused feature → 新35-node head |

它检验“直接head + 无位置信息history”，对应原M1。

### 9.4 M2 Direct

| 配置 | 值 |
|---|---|
| history order | actual |
| position embedding | 启用 |
| 输出 | fused feature → 新35-node head |

它检验实际历史顺序与距离位置编码，是当前Direct主配置。

### 9.5 M3 Direct

| 配置 | 值 |
|---|---|
| history order | graph-valid deterministic shuffle |
| position embedding | 启用 |
| 输出 | fused feature → 新35-node head |

它与M2 Direct结构完全相同，只把history改为Task Graph允许的确定性重排，用于顺序消融。

## 10. Feature-level模型公共训练配置

以下配置适用于M0–M6和M1–M3 Direct：

| 配置 | 值 | 详细解释 |
|---|---:|---|
| epochs | 50 | 只训练轻量feature-level模型，不重训100-epoch RGB backbone |
| batch size | 64 | 输入是缓存512维特征，显存需求远小于视频训练 |
| optimizer | AdamW | 只接收`requires_grad=True`的参数 |
| learning rate | `1e-3` | 无scheduler，50 epochs保持固定学习率 |
| weight decay | `1e-4` | AdamW权重衰减 |
| primary loss | 35-node Cross Entropy | 直接监督流程node |
| action loss weight | `0.0` | Tier3聚合辅助loss代码存在，但正式实验关闭 |
| gradient clip | `1.0` | 对所有可训练参数限制梯度范数 |
| AMP | 默认关闭 | feature模型规模较小，正式默认使用FP32 |
| train shuffle | 开启 | DataLoader每epoch打乱当前clip样本 |
| checkpoint | epoch 50 `last.pth` | 无validation、无early stopping |

M1–M6只更新history/delta相关参数，冻结M0；Direct更新全部feature-level fusion和node head参数，
但冻结产生cache的RGB backbone。

## 11. 评估指标与汇总规则

每个35-node模型在三个split报告：

- Node Accuracy、Macro-F1、Balanced Accuracy；
- 35维node概率聚合后的Tier3 Accuracy、Macro-F1、Balanced Accuracy；
- Stage 1、2、3分阶段结果；
- 每个样本的true/pred node、Tier3、概率和history信息；
- run级与重复node混淆分析所需prediction。

跨seed/participant正式聚合：

1. 在每个participant内部平均seed 1/2/42；
2. 对A/D/J/M四个participant等权平均；
3. 标准差是四个participant seed均值之间的样本标准差。

严格配对必须匹配：

```text
participant + seed + train scope + model/comparison + split
```

Direct配对包括：

```text
m1_direct − m0
m1_direct − m1
m2_direct − m0
m2_direct − m2
m3_direct − m0
m3_direct − m3
```

## 12. 输出目录与防覆盖

### 12.1 原严格实验

```text
outputs\<P>_as_test\cam_001484412812\seed_<S>\
├── backbone\normal_only\
├── backbone\all_runs\
├── features\retrained_normal_only\
├── features\retrained_all_runs\
├── history_models\retrained_normal_only\normal_only\m0 ... m6\
├── history_models\retrained_all_runs\all_runs\m0 ... m6\
├── e2e_baselines\normal_only\
└── e2e_baselines\all_runs\
```

### 12.2 Direct独立目录

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

标准BAT和Slurm入口均不传`--overwrite`：

- 发现`completed.json`时安全跳过；
- 目标目录非空但无完成标记时停止，要求人工检查；
- Direct不向原`retrained_normal_only`、`retrained_all_runs`、E2E或summary目录写入；
- 如实验中断，应先人工备份该实验自己的不完整目录，不应删除其他阶段结果。

## 13. Windows运行方法

### 13.1 环境

```bat
cd /d D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\graph_history_rgb_cross_person_ADM_2026-07-22
set PYTHON_BIN=C:\Users\digit\anaconda3\envs\Pytorch\python.exe
```

实际环境路径也可统一在`bat\config_windows.bat`中修改。

### 13.2 原严格四折三seed实验

```bat
call bat\run_recommended_strict_experiments.bat
```

该入口用于补齐/续跑normal-only与all-runs完整pipeline，并生成四折三seed汇总。已完成步骤根据
`completed.json`自动跳过，不会因为运行汇总而重训已有模型。

单独续跑一个fold/scope时使用README中对应的`run_normal_only_complete_one_fold.bat`、
`run_all_runs_one_fold.bat`或`run_strict_J_one_seed.bat`。

### 13.3 Direct单折单seed、两个scope

```bat
set TEST_PARTICIPANT=A
set SEED=1
call bat\run_direct_head_fusion_one_fold.bat
```

只运行单个scope：

```bat
call bat\31_train_direct_head_fusion_normal_only.bat
call bat\32_train_direct_head_fusion_all_runs.bat
```

### 13.4 Direct完整四折三seed

```bat
call bat\run_direct_head_fusion_ADJM.bat
```

入口按A/D/J/M和seed 1/2/42运行两个scope，每个scope训练M1/M2/M3 Direct，最后执行严格汇总。

### 13.5 手工运行一个Direct模型

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

## 14. HPC/Slurm运行方法

原严格实验：

```bash
bash slurm/submit_recommended_strict_experiments.sh
```

Direct单折单seed：

```bash
bash slurm/submit_direct_head_fusion_one_fold.sh A 1 both
```

第三个参数可改为`normal_only`或`all_runs`。Direct完整四折三seed：

```bash
bash slurm/submit_direct_head_fusion_ADJM.sh
```

该入口提交：

```text
4 participants × 3 seeds × 2 scopes = 24个array jobs
```

每个array job含3个task，对应M1/M2/M3 Direct。全部成功后自动提交Direct汇总任务。

## 15. Direct汇总

Windows：

```bat
call bat\33_summarize_direct_head_fusion_ADJM_3seeds.bat
```

HPC：

```bash
sbatch slurm/35_summarize_direct_head_fusion_ADJM_3seeds.slurm
```

完整网格：

```text
4 participants × 3 seeds × 2 scopes × 3 models × 3 splits = 216 rows
```

输出：

```text
outputs\direct_head_fusion_summary_ADJM_3seeds\
├── direct_head_metrics.csv
├── direct_head_paired_deltas.csv
├── direct_head_aggregate.csv
└── completed.json
```

汇总启用完整网格检查。任何participant、seed、scope、model或split缺失都会停止，不会把不完整结果
误当作正式四折三seed结果。

## 16. 各配置回答的研究问题

| 比较 | 回答的问题 |
|---|---|
| M0 vs E2E-Node-Scratch | 冻结Tier3表征与端到端node训练哪个更有效 |
| E2E-Node-From-Tier3 vs E2E-Node-Scratch | Tier3预训练是否提供node任务迁移收益 |
| M1 − M0 | 无位置信息的同run历史是否有用 |
| M2 − M1 | 历史距离位置编码是否有用 |
| M3 − M2 | graph-valid重排能否替代精确实际顺序 |
| M5 − M4 | Oracle relation bias的上限收益 |
| M6 − M4 | 可部署soft relation bias的额外收益 |
| M6 − M5 | 历史node预测误差造成的relation差距 |
| M1 Direct − M1 | 无位置条件下direct head是否优于delta |
| M2 Direct − M2 | actual-order条件下联合训练fusion与新head是否优于delta |
| M3 Direct − M3 | graph-valid条件下联合训练fusion与新head是否优于delta |
| M2 Direct − M1 Direct | Direct结构中位置编码的贡献 |
| M3 Direct − M2 Direct | Direct结构中graph-valid重排的贡献 |
| Dynamic Frozen-M0 Delta − M3 | 每epoch重新采样合法顺序是否优于每样本固定重排 |
| Dynamic Joint-Head Delta − Dynamic Frozen-M0 Delta | 不加载M0、联合训练新head与delta的作用 |
| Dynamic Direct Fusion − Dynamic Joint-Head Delta | 相同动态历史下feature fusion与logit delta的差异 |
| Dynamic Direct Fusion − M3 Direct | 动态重排是否改善Direct结构 |
| all-runs − normal-only | 训练时纳入fault runs的影响 |

这四组实验应共同保留：原M0–M6提供delta、顺序和relation消融，E2E提供视频级对照，
Direct Head Fusion比较直接feature fusion，Dynamic实验检验每epoch合法顺序增强。

## 17. Dynamic Epoch Graph-Valid Shuffle新增实验

### 17.1 研究问题

原M3与M3 Direct在Dataset初始化时为每个样本生成一次graph-valid重排；同一seed下，该样本在全部
50 epochs中始终使用相同顺序。Dynamic实验保持模型结构、训练样本和优化参数不变，只把训练history
改为：

```text
每个epoch、每个样本重新采样一个graph-valid顺序
```

epoch顺序由以下seed确定：

```text
SHA256(base_seed : epoch : sample_name)
```

因此同一base seed的完整训练可以复现，但同一样本在不同epoch通常会看到不同合法顺序。随机采样不
强制相邻epoch必须不同；只有一个合法拓扑序时自然保持不变。

### 17.2 三个模型的严格定义

| 模型ID | 当前clip分支 | History分支 | M0使用 | 最终输出 |
|---|---|---|---|---|
| `m3_dynamic_frozen_m0_delta` | 冻结M0 head | position + attention + delta，全部训练 | 加载并冻结 | `M0 logits + scale × delta` |
| `m3_dynamic_joint_head_delta` | 随机35-node head，参与训练 | position + attention + delta，全部训练 | **不加载** | `trainable head logits + scale × delta` |
| `m3_dynamic_direct_fusion` | 随机35-node head，参与训练 | position + attention + feature fusion，全部训练 | 不加载 | `head(Fusion(current, context))` |

三个模型都使用对应scope的冻结Tier3 512维feature cache，不重新训练100-epoch RGB backbone。

#### Frozen-M0 Delta

结构与原M3完全相同，唯一变化是训练history每epoch重新重排。M0的LayerNorm和35-node Linear全部
冻结；训练history projections、position embedding、attention、delta head和history scale。
该模型与原M3的差值能够单独归因于动态重排。

#### Joint-Head Delta

该模型明确不加载M0：

```text
current_logits = TrainableNodeHead(current_feature)
delta = DeltaHead(current, Attention(current, epoch_history))
logits = current_logits + sigmoid(scale) × delta
```

35-node head随机初始化，与attention和delta在同一个50-epoch阶段联合训练。它与Direct模型使用
相同的Tier3 feature来源、随机node head和训练预算；差别只在history作用于logit还是feature。

Tier3原fc为31类，node head为35类，因此不加载Tier3 fc参数。“Tier3初始化”指使用Tier3 backbone
产生的512维冻结视觉特征。

#### Direct Fusion

结构与M3 Direct完全相同，没有delta：

```text
context = Attention(current, epoch_history)
fused_feature = Linear([current_feature; context])
logits = TrainableNodeHead(fused_feature)
```

fusion继续采用`[I,0]`初始化，训练开始时`fused_feature == current_feature`。

### 17.3 训练与测试重排策略

| 阶段 | History order | 原因 |
|---|---|---|
| 训练 | `graph_valid_epoch_shuffle` | 每epoch增加合法顺序多样性 |
| 主测试 | `graph_valid_static_seeded` | 与原M3/M3 Direct使用完全相同的固定测试顺序，保证严格配对 |

测试阶段不进行随机重排，也不对多次测试取最好结果。当前target node不参与拓扑重排，继续避免
current-label leakage。history长度≤1时无法重排；history中包含重复node时沿用原逻辑回退actual
order。

### 17.4 实际重排覆盖率

对A/D/J/M、两个scope的真实训练manifest模拟50 epochs：

- 约`33.5%–37.6%`的训练样本出现多个合法顺序；
- 约`10.4%–10.9%`的样本history长度≤1；
- repeated-node回退在normal-only约`0%–0.7%`，all-runs约`2.5%–3.6%`。

每个训练目录保存`shuffle_audit.json`，记录多顺序样本数、回退数、每epoch相对actual变化数和
相邻epoch顺序变化数。

### 17.5 公共训练配置

| 配置 | 值 |
|---|---:|
| epochs | 50 |
| batch size | 64 |
| optimizer | AdamW |
| learning rate | `1e-3` |
| weight decay | `1e-4` |
| loss | 35-node Cross Entropy |
| action loss weight | `0.0` |
| gradient clip | `1.0` |
| AMP | 默认关闭 |
| validation | 无 |
| checkpoint | 最后epoch `last.pth` |

Dynamic训练DataLoader关闭`persistent_workers`，使更新后的epoch在Windows spawn和Linux fork
worker中都可靠生效。该设置只用于新Dynamic入口，不改变原实验DataLoader行为。

### 17.6 输出隔离与防覆盖

```text
outputs\<P>_as_test\cam_001484412812\seed_<S>\
history_models\dynamic_epoch_shuffle\<scope>\<model>\
├── last.pth
├── train_log.json
├── experiment_config.json
├── learned_parameters.json
├── shuffle_audit.json
├── test_results\
└── completed.json
```

该路径不位于以下任何原目录：

```text
history_models\retrained_normal_only
history_models\retrained_all_runs
history_models\direct_head_fusion
e2e_baselines
```

标准入口不传`--overwrite`。存在`completed.json`时安全跳过；目录非空但无完成标记时停止。

### 17.7 Windows运行

单折单seed、两个scope：

```bat
set TEST_PARTICIPANT=A
set SEED=1
call bat\run_dynamic_epoch_shuffle_one_fold.bat
```

分scope：

```bat
call bat\34_train_dynamic_epoch_shuffle_normal_only.bat
call bat\35_train_dynamic_epoch_shuffle_all_runs.bat
```

完整A/D/J/M四折三seed：

```bat
call bat\run_dynamic_epoch_shuffle_ADJM.bat
```

### 17.8 HPC/Slurm运行

单折单seed：

```bash
bash slurm/submit_dynamic_epoch_shuffle_one_fold.sh A 1 both
```

完整四折三seed：

```bash
bash slurm/submit_dynamic_epoch_shuffle_ADJM.sh
```

每个participant-seed-scope提交一个3-task array，对应三个Dynamic模型。

### 17.9 汇总

Windows：

```bat
call bat\36_summarize_dynamic_epoch_shuffle_ADJM_3seeds.bat
```

HPC：

```bash
sbatch slurm/38_summarize_dynamic_epoch_shuffle_ADJM_3seeds.slurm
```

完整网格：

```text
4 participants × 3 seeds × 2 scopes × 3 models × 3 splits = 216 rows
```

输出：

```text
outputs\dynamic_epoch_shuffle_summary_ADJM_3seeds\
├── dynamic_epoch_shuffle_metrics.csv
├── dynamic_epoch_shuffle_paired_deltas.csv
├── dynamic_epoch_shuffle_aggregate.csv
└── completed.json
```

严格配对包括Dynamic模型相对M0、原M3、原M3 Direct以及Dynamic模型彼此之间的差值。总体统计继续
先在participant内平均三个seed，再对A/D/J/M等权汇总。
