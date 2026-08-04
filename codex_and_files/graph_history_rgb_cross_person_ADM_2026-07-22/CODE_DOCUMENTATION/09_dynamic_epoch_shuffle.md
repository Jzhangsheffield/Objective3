# Dynamic Epoch Graph-Valid Shuffle实验记录

文档版本：2026-07-30
状态：实现、smoke test及正式A/D/J/M × seeds 1/2/42结果均已完成；结果分析见
`EXPERIMENT_RESULTS_ANALYSIS_2026-08-04.md`

## 1. 实验目的

原M3与M3 Direct在一个seed实验中为每个样本生成一次固定的graph-valid history顺序。新增阶段保持原
协议、特征、损失、测试split和静态实验不变，只把训练history改为“每个epoch重新生成一次合法顺序”，
用于区分模型是否真正学习顺序扰动下稳定的历史信息。

## 2. 三个新增模型

| 模型 | Node head初始化与训练 | History分支 | 是否加载M0 |
|---|---|---|---|
| `m3_dynamic_frozen_m0_delta` | 从M0加载并冻结 | 只训练attention与delta head | 是 |
| `m3_dynamic_joint_head_delta` | 随机初始化并训练 | node head、attention与delta联合训练 | **否** |
| `m3_dynamic_direct_fusion` | 随机初始化并训练 | 特征融合后直接分类，不使用delta | 否 |

特别注意：Joint-Head Delta不是“从M0解冻”。它从随机node head开始，因此可以单独检验联合训练相对
冻结M0基线的影响。

## 3. 重排与可复现性

训练样本的局部随机种子为：

```text
SHA256(base_seed:epoch:sample_name)
```

因此同一base seed、epoch和样本会得到相同顺序，不同epoch通常产生不同合法顺序。排序仍调用原
graph-valid约束；如果历史长度不超过1，或重复node使合法候选只有一种，顺序保持不变。每次训练写出
`shuffle_audit.json`，记录多epoch出现多种顺序、相对actual变化和退化样本数量。

训练DataLoader强制`persistent_workers=False`，确保Windows多进程worker在每个epoch读取到更新后的
epoch状态。主测试仍用原M3的固定seeded graph-valid顺序，保证与静态M3和M3 Direct严格配对。

## 4. 实现边界

新增文件：

```text
graph_history/dynamic_data.py
graph_history/dynamic_models.py
graph_history/dynamic_engine.py
tools/train_dynamic_epoch_shuffle.py
tools/summarize_dynamic_epoch_shuffle.py
tools/smoke_test_dynamic_epoch_shuffle.py
```

原`data.py`、`models.py`、`engine.py`及原训练入口不修改，旧实验仍走原逻辑。Dynamic只复用原有
graph-valid排序函数、M3 frozen-delta模型和M3 Direct模型。

## 5. 输出隔离与防覆盖

训练输出只写到：

```text
outputs/<participant>_as_test/cam_001484412812/seed_<seed>/
history_models/dynamic_epoch_shuffle/<scope>/<model>/
```

汇总输出只写到：

```text
outputs/dynamic_epoch_shuffle_summary_ADJM_3seeds/
```

训练目标目录非空即拒绝启动；已存在`completed.json`时，BAT/Slurm入口跳过已完成任务。不会写入原
`retrained_normal_only`、`retrained_all_runs`、`direct_head_fusion`或E2E目录。

## 6. 运行方式

Windows单折单seed：

```bat
set TEST_PARTICIPANT=A
set SEED=1
call bat\run_dynamic_epoch_shuffle_one_fold.bat
```

Windows完整实验：

```bat
call bat\run_dynamic_epoch_shuffle_ADJM.bat
```

HPC单折单seed：

```bash
bash slurm/submit_dynamic_epoch_shuffle_one_fold.sh A 1 both
```

HPC完整实验：

```bash
bash slurm/submit_dynamic_epoch_shuffle_ADJM.sh
```

完整配置、超参数、配对基线和输出schema见包根目录
`COMPLETE_EXPERIMENT_CONFIGURATION.md`第17节。
