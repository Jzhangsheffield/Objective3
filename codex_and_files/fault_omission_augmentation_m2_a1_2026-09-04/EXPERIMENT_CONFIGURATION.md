# 实验配置与方法说明

版本日期：2026-09-04。目的：检验在训练 normal run 中合成动作遗漏，能否提升 M2-Direct-RealOrder 与 A1-Legacy 在真实 Normal/Fault run 上的识别表现。

## 1. 实验范围与假设

此前 graph-valid shuffle 只重排已存在的历史节点，不改变动作集合。Stage 2 是线性 atomic sequence，纯 Stage 2 样本经常没有重排自由度；重复节点又会触发回退。本次直接改变历史包含的动作，模拟实际发现的遗漏类型。

本包只实现**特征级、run级的节点遗漏增强**。不修改源视频、MindRove文件、原标注或原feature cache；不生成连续合成视频，也不声称能模拟遗漏后全部物理状态变化。

例如，删除第二端压接流程后保留的检查/放回样本特征，仍来自原先正常执行的视频。这是对“当前视觉证据不变、历史动作集合变化”的训练扰动，不等于真实重新录制的一次压接过程。

## 2. 数据划分与特征来源

### 2.1 采用 ADM all-runs，不采用 sequence-disjoint

源目录由 `paths.adm_root` 指定。使用源包：

```text
outputs/<P>_as_test/cam_001484412812/protocols/all_runs/train.jsonl
outputs/<P>_as_test/cam_001484412812/protocols/all_runs/test_all.jsonl
outputs/<P>_as_test/cam_001484412812/seed_<s>/features/retrained_all_runs/train_all.pt
outputs/<P>_as_test/cam_001484412812/seed_<s>/features/retrained_all_runs/test_all.pt
```

测试参与者 P 的所有 run 保持在测试集；其余三位参与者全部 run 构成训练集。准备工具验证训练集合确实等于其余三个测试参与者的样本并集，避免误用 sequence-disjoint 子集。

每个 fold/seed 使用其原来独立训练的 RGB backbone 所生成的对应缓存。六组共用同一个 fold/seed 的缓存，不重新训练 backbone，也不使用已修复fault小数据集中的donor节点。

缓存是 512-D RGB 特征，camera=`001484412812`。已有缓存metadata记录的提取设置为16帧、224 RGB大小；本包不重新执行这些预处理。原缓存metadata可能含另一电脑的旧绝对路径，仅作为来源记录；实际读取路径始终按当前配置重建。

### 2.2 更正后的 Normal/Fault 清单

| 参与者 | Fault run |
|---|---|
| A | 20、21、22、23、24、25、26、27 |
| D | 28、29、31、32 |
| J | 26、27、28、29、33、35 |
| M | 24、25、26、27、29 |

A28、J31、J32、J34 采用用户更正后的 normal 身份。整个数据集103条run，其中normal80、fault23。Normal中59条不含Stage3，21条含完整Stage3。

本次 **Fault测试包括真实E2重复循环节点**，共377个节点，不是之前history-repair配对实验排除42个节点后的335个。新包不做修复前后配对测试，所有六组都使用同一未修复测试集。

| 测试fold | 原始训练节点数 | 训练normal/fault节点 | 测试Normal | 测试Fault | 测试All |
|---|---:|---|---:|---:|---:|
| A | 1464 | 1200 / 264 | 318 | 113 | 431 |
| D | 1433 | 1118 / 315 | 400 | 62 | 462 |
| J | 1340 | 1078 / 262 | 440 | 115 | 555 |
| M | 1448 | 1158 / 290 | 360 | 87 | 447 |
| 四fold测试并集 | — | — | 1518 | 377 | 1895 |

新包另外保留 `legacy_is_fault`，测试输出中附带旧27条fault划分的参考指标；主结果只采用更正后的 `is_fault`。两套标签都不作为模型输入。

### 2.3 已知重复节点

预检查还发现原来标为normal的 D/run_14 含两次 node28（sample_000525、sample_000527）。本包不据此自动重新判fault，不改动原数据：仍按当前确认的run身份处理，并在 `preparation_report.json` 中警告。

删除规则按node身份操作：如果D14抽中E4，将删除其所有node28出现，而不是任意保留一次；这样增强后的run才真正缺失该动作。未删除重复节点时，A1仍遵循原实现的 repeated-node fallback。

## 3. 六组对照

| Group ID | Model | FaultAug |
|---|---|---|
| M2-Direct-RealOrder-Control | M2真实顺序 | false |
| M2-Direct-RealOrder-FaultAug | M2真实顺序 | true |
| A1-Legacy-Control | legacy atomic-tail once | false |
| A1-Legacy-FaultAug | legacy atomic-tail once | true |
| A1-Legacy-Every20-Control | legacy atomic-tail，每20轮替换重排 | false |
| A1-Legacy-Every20-FaultAug | legacy atomic-tail，每20轮替换重排 | true |

网格：6 groups × 4 folds × 3 seeds（1、2、42），共72次训练。每次100epochs。Control与FaultAug均从随机初始化开始，不沿用已训练50epochs的checkpoint，不做warm start。

每个seed六组模型初始化相同，并保存权重SHA256。三个FaultAug组共享按 `(seed, fold, epoch, participant, run)` 确定的错误方案，保证三种历史策略面对同一批合成错误。错误随机数流与模型dropout、DataLoader、shuffle随机数流分离。

## 4. 错误类型与概率

### 4.1 删除映射

| 错误 | 含义 | 删除node | 正常情况下删除片段数 |
|---|---|---|---:|
| E1 | 忘记第二端压接流程 | 16、17、18、19、20、21、22 | 7 |
| E3 | 忘记检查样本 | 23 | 1 |
| E4 | 忘记关闭抽风机 | 28 | 1 |
| E5 | 忘记关闭水泵 | 29 | 1 |
| E6 | 忘记关闭空压机 | 30 | 1 |
| E7 | 忘记关闭主开关 | 33 | 1 |
| E8 | 忘记取回并装回保护罩 | 26、27 | 2 |
| E9 | 忘记将踏板复位 | 31 | 1 |
| E10 | 忘记取锁并重新上锁 | 34、35 | 2 |

E2完全不在合成操作列表中。不复制特征，不生成多余压接循环。真实E2 run仍保留。

E1与E3必须可分离：E1删除16–22但保留23；E1+E3才删除16–23。真实一次压接run中同时缺失检查，并不意味着所有合成E1也必须自动包含E3。

node16–22依次为：放样本到机器台面、夹取、翻转、再放到机器台面、再次夹取、放到电极下、第二次踩踏板。它们作为完整流程一起删除，不按重复的Tier-3标签删除所有同名动作。第一端node14、15保留。

node24（放回样本）、25（放回钳子）以及node32（关闭压接机）不在删除映射中。

### 4.2 不含Stage3的normal run

对E1、E3分别抽独立Bernoulli(0.3)。同时覆盖纯Stage2和Stage1+Stage2；Stage1节点不受删除影响。

| 结果 | 概率 | Stage2保留序列 |
|---|---:|---|
| 无错误 | 0.49 | 12–25完整 |
| 仅E1 | 0.21 | 12→13→14→15→23→24→25 |
| 仅E3 | 0.21 | 12→…→22→24→25 |
| E1+E3 | 0.09 | 12→13→14→15→24→25 |

注意：发生任意错误概率为51%，不是60%。在这种run中，node16–22及node23各自有70%的期望保留率，其余节点不因错误增强删除。

### 4.3 含Stage2和Stage3的normal run

首先抽错误个数K：P(K=0)=0.5，P(K=1)=0.2，P(K=2)=0.2，P(K=3)=0.1。之后从九种错误中等概率、不放回抽K种，将其删除节点集合取并集。

因此E[K]=0.9，每个具体错误在完整run中的边际被选概率为0.9/9=0.1。E1、E3等各为10%，**不是再次独立使用0.3**。这些错误之间因为先限制个数而不独立。

K=0时保持完整run，但如果模型为A1，仍然进行其正常的legacy shuffle。错误数量不是删除节点数量：K=1若抽到E1，会删除7个节点。

所有组合都是基于已观察到的错误类型合成，不宣称每种组合在真实数据中都出现过。只有完整包含26–35的normal run才进入此分支；若以后换入部分Stage3的normal数据，程序报错，要求先确定采集范围，不静默当作缺失错误。

### 4.4 run级替换，不追加副本

每个epoch从原始normal run重新开始，在内存中应用当轮删除方案。不会把上一轮已经删除的run继续删下去。增强版替换该轮的run，不在原始run之外额外追加一份副本。

被删节点：不作当轮训练目标，也不进入任何后续历史。保留节点：仍用其原样本特征和原node标签。原标注行号保留用于审计，不用它直接充当位置编码。

缺失节点可能在下一轮未被选中删除时重新作为训练目标出现。三个增强组同fold/seed/epoch保留的目标集合完全相同。

## 5. 删除、shuffle和位置编码的先后顺序

### 5.1 M2-FaultAug

```text
原始run → 抽错误方案 → 删除节点 → 为每个保留节点取前缀历史
        → 保持真实剩余顺序 → 根据剩余历史长度生成位置ID → 模型
```

例如预测node25，错误后的历史为 `[12,13,14,15,24]`，其位置ID为 `[5,4,3,2,1]`。最近历史为1。不为删除的16–23留空位。

### 5.2 A1-FaultAug

```text
原始run → 抽错误方案 → 删除节点 → 为每个保留节点取前缀历史
        → legacy active-tail判定及graph-valid重排 → 添加呈现顺序位置ID → 模型
```

A1不使用DualPos或true-recency位置；先shuffle再编码位置。

采用旧A1的 `active_tail_only=false, sampling=uniform`：有active tail就固定该前缀在末尾并重排其余历史；没有active tail也尝试普通graph-valid shuffle；含重复node的历史直接保持真实顺序。

沿用原实现时，`candidate_count`、Kendall上限等参数只影响plausibility-weighted分支；本包uniform分支不施加这些额外过滤，不能把它描述为扰动强度受0.35约束的shuffle。

### 5.3 once的确切含义

A1-Control每个sample使用固定shuffle种子：`stable_seed(seed, 0, sample_name)`，原始历史又不变，因此100epochs内同一样本的重排保持相同。

A1-FaultAug也固定这个shuffle种子，不额外每10epoch刷新。但错误方案每epoch刷新，导致某个sample的可用历史集合或active-tail判定可能改变；在新历史上使用相同种子，其输出顺序可能不同。**once固定的是独立shuffle随机数，不是把首次错误历史永久保存。**

不能先对完整正常run shuffle再删节点；那会改变active-tail判断依据，偏离本次确认的“先产生错误、再按原A1处理”协议。

### 5.4 Every20的确切含义

新增的Every20采用**周期替换**，不将新重排追加到训练集中，也不是在上次重排结果上继续重排。始终从本epoch删节点后得到的实际前缀重新构建历史。

| Epoch区间 | shuffle round | 种子 |
|---|---:|---|
| 1–20 | 0 | stable_seed(seed,0,sample_name) |
| 21–40 | 1 | stable_seed(seed,1,sample_name) |
| 41–60 | 2 | stable_seed(seed,2,sample_name) |
| 61–80 | 3 | stable_seed(seed,3,sample_name) |
| 81–100 | 4 | stable_seed(seed,4,sample_name) |

代码为 `round=(epoch-1)//20`。第1轮构建第一次shuffle，第21、41、61、81轮刷新，共5个周期；不是在第20轮提前刷新。Once始终使用round=0，因此Every20与相应Once组在前20轮的输入规则相同。

Every20-Control的同一样本在周期内保持相同顺序；进入下一周期时重新抽合法顺序。**换种子不保证每条历史一定变序**：某些历史只有唯一合法排列，重复节点也会回退。

Every20-FaultAug有两个独立时间尺度：错误方案每epoch刷新，shuffle种子每20epochs刷新。因此它不是“20轮保持同一个错误run”，也不保证周期内同一样本的最终历史完全相同。当前可用历史不变时，同周期的shuffle结果才相同。

刷新时不重建模型、optimizer或scheduler，不清空Adam动量，不重置位置embedding。位置ID依据这次呈现的顺序重新生成，而同一套可学习embedding参数贯穿100轮训练。测试仍一律真实顺序，Every20只改变训练输入。

配置中 `groups[].shuffle_refresh` 对Once设为 `"once"`，对Every20设为 `20`；顶层 `a1.shuffle_refresh="once"` 是默认协议，Every20由组内字段覆盖。汇总按variant区分Once和Every20，避免把两个A1的配对结果混在一起。

### 5.5 因果性和图约束

当前节点不参与tail判定，且任何历史样本的原annotation_row_index都小于当前节点。代码对这些条件逐样本断言。没有未来节点、测试donor或跨run历史。

图检查只约束已观察历史节点之间的顺序，不要求缺失的正常前置动作重新出现。否则合成遗漏会被过滤掉。纯Stage2缺失样本可能仍没有重排自由度，因此必须记录实际shuffle比例，而不能把调用了augment函数等同于发生了有效shuffle。

## 6. 模型与优化器

| 项目 | 配置 |
|---|---|
| 架构 | 原ADM `DirectSingleQueryHistoryModel` 原样代码 |
| 当前/历史特征 | 512维冻结RGB特征 |
| Query/history projection | Linear(512,256)+LayerNorm |
| Attention | 单current query、4 heads、d_model=256、null-history token |
| Position embedding | 学习型recency embedding，max_history=35 |
| Fusion | 当前512维与context256维拼接后Linear到512维 |
| 分类头 | LayerNorm+Linear，35个node类别 |
| Dropout | 0.1（attention） |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Loss | node cross entropy；无额外Tier-3辅助loss |
| Gradient clipping | global norm 1.0 |
| Batch size | 64；不drop最后一个batch |
| Epochs | 100 |
| AMP | 关闭，FP32 |
| Workers | 默认0，可改；每epoch新建loader，persistent_workers=false |

原Direct Fusion初始化将fusion设为当前特征的恒等映射，history列初始为0；node head随机初始化。所有组同seed用同一架构和初始化顺序。

训练入口在每个epoch单独设置dropout与DataLoader种子，支持从已保存epoch确定性续训。CUDA确定性GEMM所需的 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 在导入torch前设置；默认启用确定性算法。不同GPU/PyTorch版本不保证浮点逐位相同。

## 7. LR scheduler

使用按epoch更新的 `LambdaLR`：前5epochs线性warm-up，随后95epochs余弦衰减。

| Epoch | 该轮实际学习率 |
|---|---:|
| 1 | 0.0001 |
| 2 | 0.000325 |
| 3 | 0.00055 |
| 4 | 0.000775 |
| 5 | 0.001 |
| 6–99 | 从峰值开始按余弦下降 |
| 100 | 0.00001 |

令e为从1开始的epoch，w=5，T=100，eta_max=1e-3，eta_min=1e-5：

- e≤w：`eta(e)=eta_max × [0.1 + 0.9×(e−1)/(w−1)]`；
- e>w：`eta(e)=eta_min + (eta_max−eta_min)×[1+cos(pi×(e−w)/(T−w))]/2`。

先使用该轮LR完成训练，再调用scheduler.step，保存optimizer和scheduler完整状态。续训不重新warm-up、不清空Adam动量。

六组使用相同epoch日程。增强删除目标后每epoch训练样本数与步数可能下降，本包不补采样到固定步数；因此此版本是**等epoch比较，不是严格等更新次数比较**。每轮记录 `trained_targets`、`optimizer_steps`、`deleted_targets`，用于后续解释。

## 8. 测试、指标和统计

测试只在最后一个epoch的checkpoint写出后进行，不逐epoch查看测试精度，不选择best-test checkpoint。六组都使用实际观察历史，测试不shuffle、不删除、不修复。

主输出：Normal、Fault、All 的 Node/Tier-3 accuracy、macro-F1、balanced accuracy，以及分Stage指标、逐类支持数与混淆矩阵。Tier-3预测通过35个node softmax概率按映射求和得到，再取argmax；不是简单将预测node的标签当作Tier-3预测。

Macro-F1和balanced accuracy按当前统计子集真实标签中有support的类别计算，和旧包的 `graph_history.metrics` 定义一致。所有35/31类支持数仍完整保存。

正式结果汇总：

| 文件 | 统计单位 |
|---|---|
| `per_fold_seed.csv` | group × fold × seed × split |
| `mean_sd_12_fold_seed.csv` | 每组每split的12个fold×seed等权均值和样本SD |
| `pooled_4fold_by_seed.csv` | 同seed四个测试fold合并，按样本数加权的accuracy |
| `pooled_mean_sd_3seed.csv` | 上述合并accuracy在3个seed上的均值和样本SD |
| `paired_deltas.csv` | 同variant/fold/seed，FaultAug减Control；单位pp。M2、A1 Once、A1 Every20分别配对 |
| `paired_delta_mean_sd.csv` | 配对差值的均值和SD，不能用两组SD直接相减 |
| `coverage.json` | 预期72次训练、已完成数量、是否partial |

同一数据在不同seed下重复评估，不是新增独立受试者。仅凭均值提升或seed一致方向不宣称统计显著。首先比较每个模型内部FaultAug−Control，再判断错误增强和shuffle是否互补。

## 9. 输出审计、续训与防覆盖

每个运行目录：

```text
outputs/<group>/<P>_as_test/seed_<s>/
  resolved_config.json
  train_log.json
  last.pth
  completed.json
  run_masks/epoch_001.jsonl ... epoch_100.jsonl
  epoch_001_histories.jsonl
  test_results/metrics.json
  test_results/Normal_predictions.csv
  test_results/Fault_predictions.csv
  test_results/All_predictions.csv
  test_results/probabilities.pt
```

run_masks记录错误列表、删除node集合、删除sample名、原始/保留node序列及目标数量。首轮history文件记录actual与presented样本列表、位置ID和tail原因。train_log记录每epoch的错误数量分布、每类错误频次、各node保留支持数、shuffle改变比例和tail原因计数。

同配置完成任务自动跳过；不完整目录要求 `--resume`。每10epochs及最终epoch保存last，checkpoint使用临时文件后原子替换；若在epoch13中断，默认从保存的epoch10恢复。可能已存在的epoch11–13审计文件会在重跑对应epoch时重写；原数据不受影响。

运行指纹覆盖配置、group、fold/seed、训练/测试清单、缓存、任务图以及核心/依赖代码SHA256。指纹改变时拒绝续训或覆盖完成结果。改概率、改epoch、改学习率后应指定新的 `paths.output_root`，而不是接着旧实验训练。prepared manifest不一致时也要求使用新的 `input_root`。

相对路径均相对于实验包根目录，不取决于调用时的当前工作目录。正式训练不会自动从原metadata的旧电脑路径寻找文件。

## 10. 本次构建验证与边界

已通过19项单元测试、24个特征缓存校验、六组小规模训练、旧12个M2 checkpoint的5685条Node/Tier-3预测回归复现。验证报告见 `verification/validation_summary.json`。

在4fold×3seed×100epochs的**仅错误方案抽样审计**中：

| normal类型 | run-epoch抽样次数 | 无错误 | 1个错误 | 2个错误 | 3个错误 |
|---|---:|---:|---:|---:|---:|
| 不含Stage3 | 53100 | 48.85% | 41.97% | 9.18% | 不适用 |
| 含Stage3 | 18900 | 49.54% | 20.26% | 19.75% | 10.45% |

第一行的1个错误包含“仅E1”与“仅E3”，理论合计42%。这是有限次随机抽样，不强制每轮精确达到概率比例。另有20700条真实fault run-epoch记录均未施加新增遗漏。

这些计数是同一run在不同fold/seed/epoch的抽样次数，不是新增独立run，也不是模型训练结果。`outputs_smoke_v2_every20`只训练A fold seed1六组各2epochs、每轮最多3batches；它仅证明程序通路可运行，不能推断增强效果。原 `outputs_smoke` 的四组初步检查保留，不混入新验证或正式结果。短训练将scheduler压缩到2轮（无5轮warmup），正式100轮scheduler另由单元测试验证。

2轮smoke不会跨越第21轮，故额外用真实A-fold的1464个训练目标，比较5个周期的起止轮历史：周期内完全相同，跨周期确实有历史变化；结果保存于 `verification/every20_shuffle_audit.json`。此检查只生成历史，不训练模型，不应被解释成Every20有效性结果。

正式100epochs训练尚未启动。下一步运行README中的 `train` 命令，结束后查看上述三种测试split的配对差值及每轮删除强度。
