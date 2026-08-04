# Atomic-tail Graph-Valid实验技术记录

文档版本：2026-08-04
状态：实现、规则测试与独立运行入口已完成；A/D/J/M、normal-only与all-runs、Direct Fusion、
三seed、三刷新策略的完整网格已完成，结果位于`outputs/at_ad`。原计划Atomic Frozen-M0 Delta与
Atomic Joint-Head Delta仍未运行。

## 1. 与旧实验的边界

本阶段不修改`graph.py`中的原`randomized_graph_valid_history`，也不修改原Dataset、M0–M6、Direct或
Dynamic训练入口。新增实现位于`graph_history/atomic_tail_data.py`，输出位于独立
`history_models/atomic_tail_graph_valid`目录。

## 2. Atomic tail判定

`select_atomic_tail`只读取实际历史：

1. 历史node必须唯一，否则沿用原重复node回退；
2. 真实最新历史node必须属于一个atomic sequence；
3. 该sequence在历史中的node必须恰好形成以最新node结束的合法前缀；
4. 前缀必须未完成；
5. 前缀不能是其他剩余历史node的必需前序。

满足条件时，其他历史调用原graph-valid随机排序，随后把atomic前缀按sequence顺序追加到末尾。当前
target从未传入判定函数。

## 3. 刷新频率

`AtomicTailGraphValidHistoryDataset`把epoch映射为refresh round：

```text
interval=1     round=(epoch-1)
interval=10    round=(epoch-1)//10
interval=once  round=0
```

顺序seed为：

```text
SHA256(atomic_tail:base_seed:refresh_round:sample_name)
```

测试Dataset固定使用`once`，所以训练刷新策略不改变测试随机性。训练worker保持
`persistent_workers=False`，确保每个刷新点获得新round。

## 4. 回退与审计

`shuffle_audit.json`记录：

- `decision_reason_counts`；
- `atomic_tail_applied`及比例；
- tail长度分布；
- epoch到refresh round的映射；
- refresh之间实际改变顺序的样本数；
- `atomic_tail_violations`。

重复node回退actual；完整atomic sequence、非前缀或不满足tail条件时回退原graph-valid。

## 5. 模型和输出

三个模型分别复用Frozen-M0 Delta、随机Joint-Head Delta和Direct Fusion结构。Joint-Head与Direct
不加载M0。目录为：

```text
history_models/atomic_tail_graph_valid/
<scope>/<refresh_policy>/<model>/
```

标准入口不传`--overwrite`，不会写入任何旧实验目录。

## 6. 入口

Windows：

```bat
call bat\run_atomic_tail_one_fold.bat
call bat\run_atomic_tail_ADJM.bat
```

HPC：

```bash
bash slurm/submit_atomic_tail_one_fold.sh A 1 both
bash slurm/submit_atomic_tail_ADJM.sh
```

排序预览：

```bat
python tools\preview_atomic_tail_reorders.py ^
  --task-graph assets\integrated_task_graph_latest.json ^
  --relation-matrix assets\integrated_feature_history_matrix.json ^
  --last-current-node 20
```

完整实验定义和单模型参数见包根目录`COMPLETE_EXPERIMENT_CONFIGURATION.md`第18节。

## 7. 真实顺序测试

`m3_atomic_tail_direct_fusion`现在支持训练时Atomic-tail重排、测试时真实时间顺序：

```text
--evaluation-history-order actual
```

默认仍为`atomic_tail`，不改变任何旧命令。现有checkpoint可通过以下入口直接重测，无需训练：

```bat
call bat\run_at_actual_eval_ADJM.bat
```

逐checkpoint结果保存到`test_results_actual_order`，旧`test_results`不变；四折与同seed M2 Direct
的配对汇总保存到`outputs\at_actual`。
