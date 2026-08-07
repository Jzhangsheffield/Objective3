# 实时动作边界检测实验包（2026-08-07）

本目录是一个与现有 M0–M6、Direct、Dynamic 和 Atomic-tail 输出完全隔离的新实验包。它使用新生成的 `action_recognition_boundaries_with_background_v1` 标注：动作起止与动作识别标注一致，动作之间原先被融合的短 background 被恢复。正式第一版不会在标签或后处理阶段重新合并这些短 background。

## 1. 第一版实验是什么

第一版采用“冻结已有 RGB ResNet3D-18 backbone + 独立因果 boundary TCN + 在线状态机 + 已训练 M3 Atomic-tail Direct Fusion”的模块化方案。

数据流如下：

1. 对连续视频的每个当前帧，读取当前及之前共 16 帧；开头不足 16 帧时只复制第一帧进行左侧补齐。
2. 用相应 LOSO fold、seed 和训练范围的冻结 Tier3 backbone 提取 512 维特征。
3. 因果 TCN 同时输出 `background/action` 状态、`start` 概率和 `end` 概率。全部卷积只做左侧填充，归一化只发生在同一时间点的通道之间。
4. 在线状态机依靠阈值、防抖和最短持续时间生成闭合片段。正式配置 `merge_gap_steps=0`，不合并短 background。
5. 片段结束后，才能从该片段内均匀采样 16 帧并调用 M3；M3 history 只加入先前已经检测并预测的片段特征，按真实预测时间顺序排列。
6. Task Graph 在线状态记录此前预测 node，并报告当前预测是否满足直接前驱约束；第一版不使用 graph hard mask，避免把图约束变成隐藏 oracle。

该方案的 boundary detector 与 node classifier 分开训练，适合先确认“边界本身能否可靠学习”。后续多任务版本应作为独立第二阶段，与本结果配对比较，不能覆盖本目录的 v1 输出。

## 2. 数据和已有模型依据

- 连续数据清单：`Action_Segmentation_Dataset/manifest.jsonl`，共 103 个 `run_sample`，包含 participant、source run、相机目录和标注路径。
- 新边界标注：`annotations/action_recognition_boundaries_with_background_v1`，包含每个 run 的 segmentation CSV 和逐帧 CSV。
- RGB 相机：`001484412812`，与已有动作识别实验一致。
- RGB backbone：原项目 `outputs/{heldout}_as_test/cam_001484412812/seed_{seed}/backbone/{scope}/last.pth`。
- M3：原项目 `outputs/at_ad/{heldout}_s{seed}/{scope}/refresh_once/m3_atomic_tail_direct_fusion/last.pth`。
- Task Graph：原项目 `assets/integrated_task_graph_latest.json`。
- LOSO/fault 逻辑：直接读取原项目每个 fold 的 `protocols/{normal_only|all_runs}/{train|test_normal|test_fault|test_all}.jsonl`，再按 `(participant, run)` 映射为连续 run；本包不重新定义 fault。

所有绝对路径已写入 [base.json](configs/base.json)。如果移动数据，只修改配置，不修改代码。

## 3. 安装与快速核查

在本目录运行：

```powershell
$PYTHON = "C:\Users\digit\anaconda3\envs\Pytorch\python.exe"
& $PYTHON tools/validate_setup.py --config configs/base.json --deep
& $PYTHON -m unittest discover -s tests -v
& $PYTHON tools/prepare_protocols.py --config configs/base.json
```

`--deep` 会读取 103 个新逐帧标注并确认每一张标注帧真实存在。核查报告写入 `validation/setup_validation.json`。协议生成只写本实验包的 `protocols/`，不会修改原项目协议。

本次已完成的核查结果见 [VALIDATION_REPORT_2026-08-07.md](docs/VALIDATION_REPORT_2026-08-07.md)。

## 4. 运行单个 LOSO 条件

以下示例是 A held out、seed 1、all-runs：

```powershell
& $PYTHON tools/extract_boundary_features.py --config configs/base.json --heldout A --seed 1 --scope all_runs --splits train test_all
& $PYTHON tools/train_boundary.py --config configs/base.json --heldout A --seed 1 --scope all_runs
& $PYTHON tools/evaluate_boundary.py --config configs/base.json --heldout A --seed 1 --scope all_runs
& $PYTHON tools/run_online_pipeline.py --config configs/base.json --heldout A --seed 1 --scope all_runs --run run_sample_000001
& $PYTHON tools/evaluate_end_to_end.py --config configs/base.json --heldout A --seed 1 --scope all_runs
```

正式配置 `stride_frames=1`，用于保留不足 8/10 帧的短 background，特征提取计算量较大。可先用 `configs/smoke_stride4.json` 验证流程，但 stride 4 的结果不能作为短 background 正式结论，因为它有最多 3 帧的采样量化误差。

批量运行入口：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_loso_grid.ps1 -Stage prepare
powershell -ExecutionPolicy Bypass -File scripts/run_loso_grid.ps1 -Stage extract -Scope both
powershell -ExecutionPolicy Bypass -File scripts/run_loso_grid.ps1 -Stage train -Scope both
powershell -ExecutionPolicy Bypass -File scripts/run_loso_grid.ps1 -Stage evaluate -Scope both
powershell -ExecutionPolicy Bypass -File scripts/run_loso_grid.ps1 -Stage end_to_end -Scope both
```

建议先完成一个 fold/seed 的 smoke，再开启完整 4 folds × 3 seeds × 2 scopes 网格。

## 5. 严格 LOSO 协议

每个 held-out participant（A/D/J/M）独立训练，训练集中绝不包含该 participant。每个条件运行 seeds 1/2/42：

- `normal_only`：训练仅使用非故障 runs；测试仍分别报告 held-out 的 normal、fault 和 all。
- `all_runs`：训练使用其他三位 participant 的全部 runs；测试同样报告 normal、fault 和 all。
- 训练集内部按 run 做确定性 15% validation 切分；不按帧随机拆分，避免同一 run 泄漏。
- backbone、boundary cache、boundary checkpoint 和 M3 checkpoint 的 heldout/seed/scope 必须完全一致。

完整规则见 [EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)。

## 6. 标签定义

逐帧 `action != background` 为 action 状态；每个动作 segment 的第一帧为 start、最后一帧为 end。相邻动作之间即使只有 1–9 帧 background，也保留为 background。对于动作直接相邻而没有 background 的情况，前一动作的 end 和后一动作的 start 分别保留。

训练时 start/end 标签可在精确位置周围扩张 `boundary_label_radius_frames=2`，只是处理类别极度不平衡；评估始终按未扩张边界，并使用 ±3/±5/±10 帧的一对一事件匹配。stride 大于 1 时，事件只量化到事件发生后第一个可用 anchor，绝不提前放到未来事件之前。

标注来源与 54 条差异的处理见 [ANNOTATIONS.md](docs/ANNOTATIONS.md)。原有 `annotation_boundary_timestamp_audit_2026-08-07.csv` 保留不变。

## 7. 输出目录和防覆盖

```text
protocols/{A|D|J|M}_as_test/{normal_only|all_runs}/
cache/features/{fold}/{scope}/seed_{seed}/stride_{stride}/
outputs/{fold}/{scope}/seed_{seed}/causal_boundary_tcn_v1/
  resolved_config.json
  training_log.jsonl
  best.pth
  last.pth
  evaluation/{test_normal|test_fault|test_all}/
    metrics.json
    predicted_segments.jsonl
  online_pipeline/{run_sample}.jsonl
```

训练目录已有内容时默认拒绝继续，必须显式传 `--overwrite`。该选项只允许本包自己的目标目录，不会写入旧项目。

## 8. 指标

当前评估实现：

- Boundary start/end Precision、Recall、F1，容差 ±3/±5/±10 帧，一对一匹配；
- 有符号边界误差与绝对边界误差；
- 在线状态机从判定片段结束到实际输出片段的 emission delay；
- Segmental F1@10/@25/@50；
- Edit Score；
- action/background 帧级 Precision、Recall、F1、Accuracy；
- 预测片段数量与真实片段数量。

端到端 Node Accuracy 由 `evaluate_end_to_end.py` 计算：预测片段与真实动作按 IoU≥0.5 一对一匹配；`conditional_node_accuracy` 只统计成功匹配片段，`end_to_end_node_accuracy=正确 node 数/真实动作总数`，因此漏检会被计为错误。误检片段也会进入预测 history，符合真实在线运行。该结果写入独立的 `end_to_end/{split}`，不与 boundary-only 指标混写。

## 9. 因果性与延迟

- 视觉上下文：16 帧，只有过去/当前；不会产生未来帧泄漏，但系统至少需要积累实际帧流。
- feature stride：正式版 1 帧；模型每帧更新一次。
- boundary TCN：感受野为 125 个 feature steps，但全部来自过去，不引入 look-ahead。
- end debounce：默认 2 steps，因此理论最小输出延迟约 1 帧加运算时间。
- node 输出：只在动作片段闭合之后产生；延迟还包括一次片段 3D backbone 和 M3 forward。

应在目标硬件记录视频解码、backbone、boundary head、状态机和 M3 的分项 wall-clock latency；训练机上的吞吐不能代替部署延迟。

## 10. 已知风险

- stride 1 的滑窗 3D backbone 重复计算较多；第一版先保证定义正确，后续可缓存卷积特征或改为流式 backbone。
- 少于 16 帧的动作在 M3 阶段需要重复采样帧，分布与原动作 clip 可能略有差异。
- 极短 background 在视觉上可能不可分，且 2 帧防抖本身会带来小延迟；不得仅靠增大合并阈值掩盖错误。
- M3 训练使用真实动作片段，而线上使用预测片段，存在 segment distribution shift。
- Task Graph 目前只做在线一致性审计；若启用 hard mask，必须单独报告，并证明不会利用真实流程阶段。

## 11. 文件入口

- `boundary_experiment/features.py`：因果窗口与闭合片段特征；
- `boundary_experiment/models.py`：因果 TCN 与多头损失；
- `boundary_experiment/online.py`：实时状态机；
- `boundary_experiment/m3_adapter.py`：M3 history 和 Task Graph 在线状态；
- `boundary_experiment/metrics.py`：边界、segmental、edit、帧级指标；
- `tools/`：协议、核查、缓存、训练、评估和端到端入口。
