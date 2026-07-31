# 08 配置、资产、依赖与完整文件清单

## 1. `configs/experiment_defaults.json`

这是实验设计快照和默认参数说明，不是所有BAT/Slurm唯一读取源；实际运行值还可能由环境变量和命令行
覆盖。因此复现时必须同时保存最终`experiment_config.json`。

### 顶层任务配置

| key | 当前值 | 含义 |
|---|---|---|
| `test_participants` | A,D,M | 早期默认三折；严格扩展另含J |
| `optional_strict_fourth_fold` | J | 后加入严格第四折 |
| `camera_id` | 001484412812 | 固定RGB相机 |
| `feature_dim` | 512 | backbone fc前feature |
| `num_nodes` | 35 | graph node |
| `num_tier3_classes` | 31 | 动作类别 |
| `n_frames` | 16 | 每clip输入帧 |
| `rgb_size` | 224 | 方形输入 |
| `rgb_mean/std` | 三个float | 通道归一化 |

### `history`

```text
d_model=256
num_heads=4
max_history=35
dropout=0.1
epochs=50
batch_size=64
learning_rate=0.001
weight_decay=0.0001
action_loss_weight=0.0
```

`action_loss_weight=0`表示当前正式history模型只优化35-node交叉熵。

### `backbone`

```text
architecture=resnet3d_18
epochs=100
batch_size=16
learning_rate=0.0001
weight_decay=0.0001
checkpoint_policy=last_epoch_only
```

### `additional_e2e_baselines`

记录normal-only增量E2E的100 epoch、batch 16、学习率/衰减和复用Tier3 last checkpoint策略。

### `complete_all_runs_pipeline`

明确all-runs需要独立scratch backbone、独立features、M0–M6、三个E2E、三个测试split和matched
normal-only比较。

### `recommended_strict_multiseed_extension`

记录seed 1/2/42、A/D/J/M、J严格重训、无validation、last-only以及scope配对键。

### `direct_head_fusion_extension`

```text
models=m1_direct,m2_direct,m3_direct
scopes=normal_only,all_runs
reuse_existing_tier3_feature_caches=true
reuse_m0_checkpoint=false
uses_logit_delta=false
fusion_initialization=identity_current_zero_history
epochs=50
batch_size=64
learning_rate=0.001
weight_decay=0.0001
```

这些字段明确Direct实验与原delta模型隔离。

## 2. `assets/integrated_task_graph_latest.json`

这是35个可预测node的Task Graph快照。文档阅读时重点检查：

```text
nodes
atomic_sequences
feature_history_constraints
execution_constraints
action_id_tier3
stage_id
```

代码要求1..35全部存在；缺少任意node会在`TaskGraphSpec.load`停止。

## 3. `assets/integrated_feature_history_matrix.json`

包含：

```text
column_node_idx
rows[current_node_idx].values
```

行列顺序不假定天然为1..35，代码显式建立`column_lookup`再填`[35,35]`。`.`会转成`X`。

## 4. `assets/README.md`

说明Task Graph与history matrix来源、方向、relation含义和快照使用约定。若未来替换资产，必须同时：

1. 更新两个JSON；
2. 保证node mapping一致；
3. 运行setup/smoke tests；
4. 使用新输出目录；
5. 不把不同graph版本结果直接合并。

## 5. `requirements.txt`

声明项目Python依赖。核心运行依赖包括：

- PyTorch与torchvision；
- NumPy；
- 视频/图像和数据处理相关库。

GPU版PyTorch必须与CUDA/驱动匹配，不能只依赖通用`pip install`命令。正式环境路径由Windows/HPC
配置指定。

## 6. 原包级说明文件

### `README.md`

按时间记录：

- 初始A/D/M跨人实验；
- 增量E2E；
- 完整all-runs；
- 严格四折三seed；
- Direct Head Fusion；
- Dynamic Epoch Graph-Valid Shuffle；
- A/D Atomic-tail Direct Fusion缩减实验。

因为它保留历史演化，早期章节中的默认三折或辅助all-runs不能覆盖后期严格定义。

### `EXPERIMENT_RESULTS_ANALYSIS_2026-07-30.md`

正式结果解释报告。当前版本已扩展Direct、Dynamic和A/D Atomic-tail结果，并包含统一逐人/逐模型
失效分析。统计结论应追溯到`outputs`中的CSV/JSON和predictions。

### `COMPLETE_EXPERIMENT_CONFIGURATION.md`

记录M0–M6、E2E、Direct与Dynamic实验的完整配置、运行方式、输出隔离和比较关系。它是当前统一的
配置/实验设计说明；本技术手册进一步解释实际类、函数、shape和调用链。

## 7. Python源码清单

### `graph_history/`

| 文件 | 行数 | 职责 |
|---|---:|---|
| `__init__.py` | 4 | 包初始化 |
| `constants.py` | 21 | 全局常量和关系ID |
| `utils.py` | 175 | I/O、seed、设备、checkpoint、安全目录 |
| `backbone.py` | 185 | ResNet3D |
| `data.py` | 281 | RGB和history datasets/collate |
| `graph.py` | 167 | graph加载与graph-valid排序 |
| `protocols.py` | 144 | LOSO协议 |
| `metrics.py` | 60 | confusion与概率聚合 |
| `models.py` | 430 | M0–M6和Direct |
| `engine.py` | 217 | feature模型训练评估 |
| `video_evaluation.py` | 203 | E2E视频评估 |
| `dynamic_data.py` | 166 | 逐epoch、逐样本确定性graph-valid重排 |
| `dynamic_models.py` | 112 | Dynamic Joint-Head Delta |
| `dynamic_engine.py` | 84 | epoch状态注入与dynamic训练循环 |
| `atomic_tail_data.py` | 295 | atomic tail判定、合法重排、刷新周期Dataset与审计 |

### `tools/`

| 文件 | 行数 | 职责 |
|---|---:|---|
| `validate_setup.py` | 52 | 环境检查 |
| `prepare_protocols.py` | 53 | 协议入口 |
| `guard_output_dir.py` | 24 | 防覆盖 |
| `train_backbone.py` | 163 | Tier3训练 |
| `extract_features.py` | 108 | feature cache |
| `train_history_model.py` | 190 | M0–M6 |
| `train_direct_history_model.py` | 206 | Direct |
| `evaluate_e2e_tier3.py` | 105 | Tier3复测 |
| `train_e2e_node.py` | 205 | E2E node |
| `smoke_test_models.py` | 69 | M1–M6 smoke |
| `smoke_test_direct_models.py` | 78 | Direct smoke |
| `summarize_results.py` | 54 | 旧fold汇总 |
| `summarize_cross_person.py` | 182 | 早期跨人 |
| `summarize_all_models.py` | 583 | 正式十模型 |
| `summarize_direct_head_fusion.py` | 287 | Direct汇总 |
| `train_dynamic_epoch_shuffle.py` | 308 | 三种Dynamic模型训练 |
| `summarize_dynamic_epoch_shuffle.py` | 379 | Dynamic严格配对汇总 |
| `smoke_test_dynamic_epoch_shuffle.py` | 259 | Dynamic数据、训练循环和模型回归测试 |
| `train_atomic_tail_graph_valid.py` | 342 | 三模型Atomic-tail训练和固定测试 |
| `summarize_atomic_tail_graph_valid.py` | 489 | Atomic-tail完整网格与配对汇总 |
| `preview_atomic_tail_reorders.py` | 86 | canonical history重排预览 |
| `smoke_test_atomic_tail_graph_valid.py` | 191 | tail规则和刷新频率回归测试 |

当前共15个包模块与22个Python工具入口；Dynamic和Atomic-tail均独立实现，原训练入口未改写。

## 8. 编排脚本清单

完整逐文件用途见第5卷。数量：

```text
BAT                  59
Slurm job scripts    41
HPC config/submit sh 19
```

Atomic-tail新增BAT 37–39、Slurm 39–41及独立组合/提交入口；原编号与调用链保持不变。

## 9. 结果文件不属于源码

`outputs/`包含大量：

```text
.pth
.pt
.json
.csv
logs
completed markers
```

它们是实验证据，不应复制进源码文档或批量格式化。文档只读这些结果，新增文档不修改结果目录。

## 10. 当前文档覆盖表

| 内容 | 位置 |
|---|---|
| 研究问题与全流程 | 第0卷 |
| 所有关键数据字段/shape | 第1卷 |
| 15个底层模块及函数 | 第2、3、9、10卷 |
| 22个Python入口及函数 | 第4、9、10卷 |
| 59个BAT | 第5、9、10卷 |
| 41个Slurm作业 | 第5、9、10卷 |
| 19个shell配置/提交入口 | 第5、9、10卷 |
| Direct新增结果 | 第6卷 |
| Dynamic逐epoch重排新增阶段 | 第9卷 |
| Atomic-tail及刷新频率新增阶段 | 第10卷 |
| 输出与防覆盖 | 第7卷 |
| assets/config/requirements/原报告 | 本卷 |
