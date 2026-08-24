# Sequence-disjoint Atomic-tail 实验配置

## 1. 研究变量

### 固定变量

- participant LOSO folds：A、D、J、M；
- train scope：all-runs；
- seeds：1、2、42；
- camera：001484412812；
- visual backbone：每个 fold×seed 在过滤后的 all-runs 训练集上从头训练 R3D-18；
- backbone epochs：100；batch size：16；AdamW LR：0.0001；weight decay：0.0001；
- backbone checkpoint：last epoch；不使用 validation 或旧 checkpoint；
- feature extraction：用该 fold×seed 新 checkpoint 重新提取512-D train/test特征；
- output classes：35 Node、31 Tier-3；
- Direct fusion：当前 clip query + history attention context；
- d_model：256；
- attention heads：4；
- max history：35；
- dropout：0.1；
- history optimizer：AdamW；
- history epochs：50；
- history batch size：64；
- history LR：0.001；
- history weight decay：0.0001；
- gradient clipping：1.0；
- checkpoint：last epoch；
- evaluation history：actual chronological；
- test splits：normal、fault、all。

### 自变量

1. history training policy：actual vs A1-Legacy graph-valid vs A3-DualPos active-tail；
2. refresh interval：once vs 10 epochs；
3. position semantics：重排后的 presented position vs true recency + displacement。

### 主要因变量

- primary：test_all Node accuracy；
- secondary：normal/fault/all Node/Tier-3 accuracy、macro-F1、balanced accuracy；
- mechanism audit：augmentation changed fraction、测试 history-prefix coverage、participant×seed paired delta。

## 2. 原始顺序隔离

Run signature 定义为：

```text
signature(run) = tuple(node_idx ordered by annotation_row_index)
```

默认不压缩重复 Node，因为 fault run 中的重复、回退或重新操作本身属于真实流程顺序的一部分。

对 held-out participant 的所有 `test_all` runs 建立 signature 集合 `S_test`。训练 run `r` 的处理为：

```text
retain(r) iff signature(r) not in S_test
```

删除单位是完整 run，而不是单个 clip。过滤后重新计算训练与测试 signature 交集，必须为零，否则 protocol 生成失败。

Tier-3 signature 记录在 `run_sequence_index.csv` 中，但本轮不因为 Tier-3 完全相同而额外删除训练 run。

## 3. Augmentation 与测试顺序

Graph-valid augmentation 被允许自然产生与测试顺序相同的 history。该行为不构成 protocol 失败。

边界如下：

- 允许：sampler 根据固定 task graph 和训练 seed 随机生成顺序，事后发现与测试 history 相同；
- 禁止：sampler 读取测试 signature；
- 禁止：为命中测试顺序而选择 candidate；
- 禁止：根据测试性能调整 refresh round；
- 禁止：使用测试 labels 训练 transition model。

当前 A1-Legacy 与 A3-DualPos 均使用 uniform graph-valid sampling，不使用训练 transition model。Task graph 与 relation matrix 来自实验开始前已经固定的旧资产。

## 4. 新 backbone 与新特征流水线

每个 `(test_participant, seed)` 独立执行以下步骤，共12套 upstream artifacts：

1. 读取该 fold 已过滤的 `all_runs/train.jsonl`；
2. 随机初始化 R3D-18 Tier-3 classifier；
3. 在过滤后的训练 clips 上训练100 epochs；
4. 保存 last-epoch backbone checkpoint；
5. 在未修改的 test_normal、test_fault、test_all 上评估 backbone；
6. 使用该 checkpoint 提取过滤后 train 的512-D features；
7. 使用同一 checkpoint 提取完整 test_all 的512-D features；
8. 五个 history 配置共享这对 fold×seed feature caches。

严禁从旧实验复制 `last.pth`、`train_all.pt` 或 `test_all.pt`。旧包只提供已验证的代码入口、RGB normalization 和网络定义。

本阶段的路径为：

```text
outputs/upstream/{participant}_as_test/cam_001484412812/seed_{seed}/
  backbone/all_runs/last.pth
  features/retrained_all_runs/train_all.pt
  features/retrained_all_runs/test_all.pt
```

## 5. 五个实验

### M2-Direct-RealOrder

- 旧配置基底：A0；
- train view：actual；
- shuffle：无；
- position mode：presented；在 actual history 中与真实 recency 等价；
- active-tail gating：关闭；
- paired views：关闭；
- warm start：无；
- 必须在过滤后 manifest 上从头训练50 epochs；
- 禁止复用旧 `m2_direct/last.pth`。

### A1-Legacy-Once

- 旧配置基底：A1；
- train view：augmented；
- active-tail-only：false；
- sampling：uniform；
- position mode：presented；先 graph-valid 重排，再按照重排后的 presented order 分配位置；
- refresh interval：once；
- 有 active incomplete tail 时固定 tail、重排其余历史；
- 无 active tail 时仍执行普通 broad graph-valid shuffle；
- 从头训练50 epochs。

### A1-Legacy-Every10-Replace

除 refresh interval 为10以外，与 A1-Legacy-Once 完全相同。每次 refresh 用新 augmented view 替换旧 view；不会重置模型、position embedding 或 optimizer state，也不会显式回放旧 view。

旧 dataset 的 refresh round 为：

```text
refresh_round = floor((epoch - 1) / 10)
```

因此50 epochs 对应 rounds 0、1、2、3、4。每个 sample、seed、round 通过 stable hash 得到确定性 shuffle seed。

### A3-DualPos-Once

- 旧配置基底：A3-DualPos；
- active-tail-only：true；sampling：uniform；
- position mode：true_plus_shift；
- token 使用真实 recency embedding 和 displacement embedding；
- displacement 定义为 `presented_position - true_recency_position`；
- actual/test view 中 displacement 为0；
- refresh interval：once；
- 从头训练50 epochs。

### A3-DualPos-Every10

除 refresh interval 为10以外，与 A3-DualPos-Once 完全相同。

## 6. 公平性控制

五个配置必须共享：

- 同一个 fold 的过滤后 train manifest；
- 同一个 test manifest；
- 同一个 fold×seed 新训练 backbone；
- 从该新 backbone 重新提取的同一对 train/test feature caches；
- 相同 seed；
- 相同 epoch、batch、optimizer、LR 和 checkpoint policy；
- 相同测试顺序和评估代码。

只有增强模型使用 graph-valid history。M2 不参与 once/every-10 调度。

## 7. 结果判定

每个配置产生12个 fold×seed 配对记录。比较时优先使用同 participant、同 seed 的差值，不只比较总体平均值。

建议的判断顺序：

1. A1-Legacy-Once 与 A1-Legacy-Every10-Replace 是否分别高于 M2；
2. 提升是否在四个 participant 中方向一致；
3. fault 是否明显下降；
4. every-10 是否高于 once；
5. A1-Legacy 与 DualPos 各自的收益和稳定性是否不同；
6. 提升是否集中在 augmentation 新覆盖的测试 history prefixes。

不能因为五个配置中的最高均值略高于 M2 就直接宣称成功。至少需要报告 fold×seed 配对差值、胜负数量、SD/CI 以及 normal/fault/all 的一致性。

## 8. 路径与依赖

主配置文件：`config/experiment_config.json`。

它引用：

- 原始 RGB dataset；
- 旧 strict cross-person 包中的 backbone 训练与 feature extraction 代码；
- 旧 A0–A8 Python package；
- 旧 task graph 和 relation matrix；
- 本包生成的 sequence-disjoint manifests；
- 本包独立 outputs。

如果目录移动，只需更新配置中：

- `dataset_root`；
- `legacy_graph_history_package_root`；
- `legacy_atomic_package_root`；
- 必要时 `python_executable`。

不要把 `reuse_shared_a0_checkpoint` 或 `reuse_old_feature_cache` 改为 true，因为旧 backbone、feature cache 和 M2 checkpoint 都是在未过滤训练集上产生的，不符合本轮 sequence-disjoint protocol。
