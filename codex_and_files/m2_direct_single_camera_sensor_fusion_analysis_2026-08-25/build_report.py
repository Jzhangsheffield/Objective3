from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def pct(value: float) -> str:
    return f"{100 * float(value):.2f}%"


def overall_table(frame: pd.DataFrame) -> str:
    labels = {"normal": "Normal", "fault": "Fault", "all": "All"}
    rows = [
        "| Split | Node Accuracy | Node Macro-F1 | Node Balanced Acc. | Tier3 Accuracy | Tier3 Macro-F1 | Tier3 Balanced Acc. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("normal", "fault", "all"):
        r = frame[frame["split"] == split].iloc[0]
        rows.append(
            f"| {labels[split]} | {pct(r.node_accuracy)} ± {pct(r.node_accuracy_participant_sd)} | "
            f"{pct(r.node_macro_f1)} | {pct(r.node_balanced_accuracy)} | "
            f"{pct(r.tier3_accuracy)} ± {pct(r.tier3_accuracy_participant_sd)} | "
            f"{pct(r.tier3_macro_f1)} | {pct(r.tier3_balanced_accuracy)} |"
        )
    return "\n".join(rows)


def participant_table(frame: pd.DataFrame) -> str:
    rows = [
        "| Held-out participant | Clips | Node Accuracy | Node Macro-F1 | Tier3 Accuracy | Tier3 Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in frame[frame["split"] == "all"].sort_values("participant").iterrows():
        rows.append(
            f"| {r.participant} | {int(r.samples)} | {pct(r.node_accuracy)} | "
            f"{pct(r.node_macro_f1)} | {pct(r.tier3_accuracy)} | {pct(r.tier3_macro_f1)} |"
        )
    return "\n".join(rows)


def stage_table(frame: pd.DataFrame) -> str:
    rows = [
        "| Stage | Clips/seed | Node Accuracy | Tier3 Accuracy |",
        "|---:|---:|---:|---:|",
    ]
    for _, r in frame.sort_values("stage").iterrows():
        rows.append(
            f"| {int(r.stage)} | {int(r.samples_per_seed)} | {pct(r.node_accuracy)} ± {pct(r.node_accuracy_participant_sd)} | "
            f"{pct(r.tier3_accuracy)} ± {pct(r.tier3_accuracy_participant_sd)} |"
        )
    return "\n".join(rows)


def class_table(frame: pd.DataFrame, kind: str) -> str:
    id_col = f"{kind}_id"
    rows = [
        f"| {kind.title()} ID | Label | Stage | N | Precision | Recall | F1 | Normal Recall (N) | Fault Recall (N) | Worst fold recall |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in frame.sort_values(id_col).iterrows():
        rows.append(
            f"| {int(r[id_col])} | {r.label} | {r.stage} | {int(r.support_all)} | "
            f"{pct(r.precision_all_mean)} | {pct(r.recall_all_mean)} | {pct(r.f1_all_mean)} | "
            f"{pct(r.recall_normal_mean)} ({int(r.support_normal)}) | "
            f"{pct(r.recall_fault_mean)} ({int(r.support_fault)}) | "
            f"{r.worst_participant}: {pct(r.worst_participant_recall)} |"
        )
    return "\n".join(rows)


def confusion_table(frame: pd.DataFrame, kind: str, limit: int = 12) -> str:
    true_col, pred_col = f"true_{kind}_id", f"pred_{kind}_id"
    rows = [
        "| True | Predicted | Mean errors/seed | Share of true class |",
        "|---|---|---:|---:|",
    ]
    for _, r in frame.head(limit).iterrows():
        rows.append(
            f"| {int(r[true_col])} {r.true_label} | {int(r[pred_col])} {r.pred_label} | "
            f"{float(r.mean_errors_per_seed):.2f} | {pct(r.fraction_of_true_class)} |"
        )
    return "\n".join(rows)


def repeated_table(frame: pd.DataFrame) -> str:
    rows = [
        "| True node → predicted node | Tier3 label | Errors (3 seeds) | Error rate |",
        "|---|---|---:|---:|",
    ]
    for _, r in frame.iterrows():
        rows.append(
            f"| {int(r.true_node)} → {int(r.pred_node)} | {r.true_label} | "
            f"{int(r.errors_across_3_seeds)} | {pct(r.error_rate)} |"
        )
    return "\n".join(rows)


def main() -> None:
    overall = pd.read_csv(HERE / "m2_overall_metrics.csv")
    participant = pd.read_csv(HERE / "m2_participant_metrics.csv")
    stages = pd.read_csv(HERE / "m2_stage_metrics.csv")
    tier3 = pd.read_csv(HERE / "m2_tier3_per_class_metrics.csv")
    node = pd.read_csv(HERE / "m2_node_per_class_metrics.csv")
    tier3_confusions = pd.read_csv(HERE / "m2_tier3_top_confusions.csv")
    node_confusions = pd.read_csv(HERE / "m2_node_top_confusions.csv")
    repeated = pd.read_csv(HERE / "m2_repeated_node_pair_errors.csv")
    summary = json.loads((HERE / "analysis_summary.json").read_text(encoding="utf-8"))
    ambiguity = summary["ambiguity"]

    report = f"""# 单摄像头 M2-Direct fusion 的 Tier3 / Node 性能与多传感器扩展方案

日期：2026-08-25  
分析对象：`cam_001484412812 + all_runs + m2_direct`  
用途：确定第二摄像头、EMG/IMU 加入 M2 的方式，并为后续实时动作分割设计统一的多模态路径。

## 1. 结论先行

1. **当前最佳单摄像头 M2-Direct 的总体性能是可靠的，但仍有明确的长尾。** 四个 LOSO participant 先在各自内部平均 seed 1/2/42，再对 A/D/J/M 等权平均，All split 的 Node Accuracy 为 **90.57%**，Tier3 Accuracy 为 **90.64%**；Fault 分别为 **89.75% / 89.86%**。
2. **Node 与 Tier3 的差距只有 0.07 个百分点。** 三个 seed 合计 {ambiguity['node_errors_across_3_seeds']} 个 node 错误中，只有 {ambiguity['same_tier3_node_errors_across_3_seeds']} 个（{pct(ambiguity['same_tier3_fraction_of_node_errors'])}）是“Tier3 正确、重复 node 位置错误”。M2 的真实历史与 position 已经基本解决 Node 14/21、15/22、16/19、17/20 的重复步骤消歧。
3. **新增传感器的主要目标不应被定义为‘修复重复 node 编号’，而应是修复动作语义和相位。** 当前主要混淆是 `grip ↔ put`、`take ↔ put`、`turn on ↔ turn off`、`unlock ↔ lock`，以及 fault 中 `put sample on table` 的明显退化。
4. **推荐的 M2 扩展是分层、带质量门控的 residual fusion，而不是一次性 concat。** 先在同一时间窗内融合两个 RGB view，再融合 EMG/IMU；融合层从现有单摄像头 RGB identity 初始化，最后才进入 M2 的 current-query/history attention。这样能够保留已经验证的 M2 表示，并允许新模态只学习“纠正量”。
5. **实时动作分割必须有两条多模态通路。** 高频 EMG/IMU 和因果 RGB token 进入 boundary/progress 分支；同一批模态在片段闭合或持续更新时形成 observation token，进入 M2 node 分支。只在最终 node head 前加传感器，无法解决当前在线原型最严重的过分割和 history 污染。

## 2. 本次分析范围与统计口径

- 单摄像头：`001484412812`；视频输入为 16 帧、224×224，Tier3 RGB backbone 输出 512-D frozen feature。
- M2-Direct：当前 clip 投影为 query；同一 run 中较早 clip 的 frozen RGB features 是 causal history；history 使用真实 chronological order 与 recency position embedding；attention context 与当前 512-D feature 融合后，直接由可训练 35-node head 分类。
- 训练：`all_runs`；A/D/J/M 四折严格 LOSO；seed 1、2、42；使用最后 epoch checkpoint，没有 validation/early stopping。
- 测试：Normal、Fault、All；每个 seed 的 All split 合计 {summary['experiment']['test_all_unique_clips_per_seed']} 个 out-of-fold clips。
- 总体指标沿用项目既有的 **participant-first** 口径：participant 内平均 3 seeds，再对 4 participants 等权平均；表中的 `±` 是四个 participant 均值之间的样本标准差。
- 逐类别/逐 node 表：每个 seed 将四个 held-out folds 拼成完整 out-of-fold prediction，再由 confusion matrix 计算 P/R/F1，最后对 3 seeds 求均值。`N` 是一个 seed 中四折合计的唯一 clip 数，未把三个 seed 当成三倍独立样本。
- 本报告分析的是原始完整协议的当前最佳 M2，不把最近 sequence-disjoint 重训 M2 的较低数值混入本表。

## 3. 总体、跨人和跨 Stage 性能

### 3.1 总体

{overall_table(overall)}

Accuracy 高于 Macro-F1 约 3–4 个百分点，说明主要剩余问题集中在低支持类别，而不是所有动作均匀变差。

### 3.2 Held-out participant

{participant_table(participant)}

J 最好；D 的 Accuracy 最低；M 的 Tier3 Macro-F1 最低。多模态实验必须继续使用四折等权汇总，否则较大的 J fold 会掩盖 A/D/M 的弱点。

### 3.3 Stage

{stage_table(stages)}

Stage 1 最弱，Node Accuracy 仅 83.48%；Stage 3 为 87.34%；动作密集且历史结构稳定的 Stage 2 反而达到 92.45%。这进一步支持：第二视角应优先补设备状态和小物体交互，EMG/IMU 应补启动/结束与动作方向，而不是继续加强已很强的重复步骤 recency 编码。

## 4. 31 个 Tier3 类别的性能

{class_table(tier3, 'tier3')}

### 4.1 最需要新增模态帮助的 Tier3

- `take lock from table`：Recall 58.67%，Precision 54.05%，是最弱类别；容易与 `put sample on table` 互相混淆。第二视角的物体可见性最可能直接有帮助。
- `unlock crimper`：Recall 63.64%，常被判为 `lock crimper`。IMU 的旋转方向、EMG 的用力起止和历史/进度联合使用，理论上比单纯增加 RGB feature 维数更对症。
- 三个设备 `turn on` 类别 Recall 约 75.76%–78.79%，且常被各自的 `turn off` 类别吸收。这里需要“方向/前后状态”，第二视角和 IMU 都可能有贡献。
- `put sample on table`：All Recall 78.64%，但 Fault Recall 只有 58.02%；这是最明确的 fault 定向改进目标之一。
- `press pedal`、`place sample under electrodes` 等高支持 Tier3 已达到约 96%–98%，不应让它们主导传感器选择。

### 4.2 Tier3 主要混淆

{confusion_table(tier3_confusions, 'tier3')}

## 5. 35 个 Node 的性能

{class_table(node, 'node')}

### 5.1 重复 Tier3 对应 node 的消歧

{repeated_table(repeated)}

四组重复动作共八个有向互换，在三个 seed、四个 participant 的全部预测中只发生 4 次；`place sample under electrodes` 与 `press pedal` 的两次出现甚至没有互换。因而“Node 比 Tier3 难很多”并不符合当前 M2 的实际结果。

### 5.2 Node 主要混淆

{confusion_table(node_confusions, 'node')}

Node 20 → 19、Node 16 → 17 等高频错误是相邻动作语义混淆，不是同 Tier3 重复位置互换。多模态模型应学习动作相位（接近、接触、抓取、移动、释放）和对象状态，而不能只用 task graph 进一步压缩合法顺序。

## 6. 在 M2 上加入第二摄像头与 EMG/IMU 的推荐结构

### 6.1 推荐：分层 gated residual multimodal M2

```text
camera 001484412812 ─ RGB encoder ─┐
                                  ├─ cross-view fusion ─ visual token v_t ─┐
second camera ────── RGB encoder ─┘                                        │
                                                                           ├─ quality-gated
left/right EMG ───── causal 1D encoder ─ EMG token e_t ─────────────────────┤  residual fusion
left/right IMU ───── causal 1D encoder ─ IMU token i_t ─────────────────────┘
                                                                                 │
z_t = LN(x_cam1 + g_view Δv + g_emg Δe + g_imu Δi)                              │
                                                                                 ▼
                         M2 current query attends to completed-history z tokens
                                             │
                                  35-node head + Tier3 aggregation
```

关键设计：

- **保留现有 RGB 主干为 anchor。** 初始化时令 `g_view/g_emg/g_imu ≈ 0`，使 `z_t ≈ x_cam1`；这与当前 M2 fusion 的 `[I, 0]` identity 初始化完全一致，可先复现 90.57% 再逐步学习新模态修正。
- **三层融合顺序固定为：view → sensor → history。** 两个相机是相同物理时刻的不同观察，应先做 cross-view；EMG/IMU 是不同采样率、不同噪声机制的动作信号，再通过时间/模态 embedding 与视觉 token 结合；M2 history 最后建模跨动作的程序上下文。
- **current 与 history 使用同一种 fused representation。** 不能只给当前 clip 加传感器、历史仍保留 RGB-only feature，否则 attention 同时面对两个不同的 feature domain。
- **每个 gate 读取 availability、signal quality、时间偏差和当前特征。** 摄像头丢帧、EMG 饱和/电极漂移、IMU 缺包时自动退回单摄像头 M2。EVI-MAE 的实验也显示 IMU 能补视觉退化，但简单拼接在视觉退化下表现较差；MMG-Ego4D 则支持 modality dropout 与跨模态对齐用于缺失模态鲁棒性。[EVI-MAE](https://arxiv.org/abs/2407.06628)，[MMG-Ego4D](https://arxiv.org/abs/2305.07214)

### 6.2 第二摄像头：如何加、如何选

建议把 `001431512812` 与 `001528512812` 都作为候选，不根据单相机总体 Accuracy 直接拍板。第二视角的选择标准应是：

1. 在当前 M2 错误 clip 上能否看清主相机被遮挡的手、锁、保护盖、样本与工作台；
2. 对 `take/put`、`on/off`、`lock/unlock` 的错误是否与主相机互补；
3. 同步丢帧和时间漂移是否可控；
4. 增加的推理延迟是否满足实时预算。

模型上按以下顺序做基线：

- V0：两个单相机 M2；
- V1：两个模型的 calibrated probability average（最便宜的 late-fusion sanity check）；
- V2：共享 RGB backbone + camera-specific LayerNorm/adapter + gated feature average；
- V3：双向 cross-view attention + quality gate（推荐主模型）。

多视角研究通常用独立 view encoder 与跨 view 交互来保留互补信息；针对多摄像头动作定位的工作也采用 frame-level prediction 后的跨视角结合。[DVANet](https://ojs.aaai.org/index.php/AAAI/article/view/28290)，[Akdag et al., 2023](https://openaccess.thecvf.com/content/CVPR2023W/AICity/html/Akdag_Transformer-Based_Fusion_of_2D-Pose_and_Spatio-Temporal_Embeddings_for_Distracted_Driver_CVPRW_2023_paper.html)

### 6.3 EMG 与 IMU：分开编码，不先混成一条信号

- **EMG token**：针对肌肉激活的快速起止、抓握/释放、踩压/放松；使用 per-session/per-channel robust normalization、notch/band-pass 后的 raw/envelope 双分支或多尺度 causal 1D CNN/TCN。EMG participant/electrode domain shift 很大，必须保留 channel mask、side embedding 与 signal-quality feature。
- **IMU token**：针对移动方向、旋转、反转、速度和静止状态；加速度与角速度应分组归一化，用多尺度 causal 1D encoder。传感器放置决定它能解决什么：腕部 IMU 适合 `grip/put/reverse/lock`，脚/踝 IMU 才能直接帮助 `press pedal`。
- **左右 MindRove 分支**：先用共享权重编码，再加 left/right device embedding；允许单侧缺失。若左右设备动作角色不同，可在共享底层后使用轻量 side adapter。
- **同步**：全部保留原始硬件 timestamp；以视觉 anchor time 为中心取过去窗口，不使用未来 sensor sample；训练时加入小幅 time-jitter，并使用 relative-time embedding。WACV 2026 的 MAD-DG 明确针对异步模态提出 temporal binding 与对齐，支持这里把同步误差建模为训练变量而不是假设完美同步。[MAD-DG](https://openaccess.thecvf.com/content/WACV2026/html/Ji_Alignment_and_Distillation_A_Robust_Framework_for_Multimodal_Domain_Generalizable_WACV_2026_paper.html)

### 6.4 不推荐的做法

- 把两个 512-D RGB、EMG、IMU 全部直接 concat 后喂给 35-node MLP；小数据下容易由维数最大/最稳定的 RGB 独占，且缺失模态时行为不可控。
- 对不同采样率信号只按数组长度插值，不以 timestamp 对齐。
- current clip 用多模态、history 用主相机 RGB-only。
- 一开始端到端解冻两个视频 backbone；参数量翻倍会让传感器融合收益无法归因。
- 只报告 All Accuracy；新增模态可能只改善高频类，同时使 `take lock` 等长尾更差。

## 7. 面向实时动作分割的最佳结合方式

### 7.1 现有在线原型给出的本地证据

仓库中的 `realtime_action_boundary_experiment_2026-08-07` 已实现严格因果流程：16 帧过去窗口 → frozen RGB feature → 5-layer causal Boundary TCN → 在线状态机 → 闭合片段 → 历史模型 node 分类。第一轮 A/seed-1 的核心问题是：431 个真实动作被切成 2525 个预测片段，80.79% 的片段恰好只有 3 帧；错误短片段写入 history 后，conditional Node Accuracy 只有 68.46%，端到端 Node Accuracy 为 41.30%。validation-only decoder 校准把预测片段降到 411，Segment F1@0.5 从约 17.6% 提高到 62.95%。

因此，多模态实时系统的首要原则是：**先稳定 segment，再让 segment 进入 M2 memory。**

### 7.2 推荐的双路径在线系统

```text
同步后的 causal sensor/video tokens
        ├─ Fast path:  EMG + IMU + lightweight RGB/view token
        │              → causal boundary/state/progress head
        │              → pending/hysteresis decoder
        │              → start / ongoing / end
        │
        └─ Semantic path: dual-view RGB + EMG/IMU gated fused z_t
                       → ongoing segment accumulator
                       → segment closes (or periodic provisional update)
                       → M2 current-query + confirmed-history memory
                       → node posterior

node posterior + boundary/progress + task graph
                       → small causal beam / soft graph decoder
                       → final online segment and node
```

- **Fast boundary path**：EMG/IMU 以原生高采样率下采样为短步长 token，RGB 用较轻的 causal window；输出 action/background、start、end、progress。ASRF 证明把 action classification 与 boundary regression 分支结合可降低过分割；ProTAS 则在在线程序任务中结合 causal segmentation、progress 与 task graph。[ASRF](https://openaccess.thecvf.com/content/WACV2021/html/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.html)，[ProTAS](https://openaccess.thecvf.com/content/CVPR2024/html/Shen_Progress-Aware_Online_Action_Segmentation_for_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.html)
- **Semantic M2 path**：不要等每个 3 帧波动就调用 M2。使用 corrected pending merge、minimum duration、hysteresis；片段确认结束后，将其 multimodal segment token 写入 history。
- **软 history 而非硬错误复制**：保存 node posterior/entropy 与 segment quality，不只保存 argmax node；低质量片段进入 quarantine，不立即污染 confirmed history。
- **小 beam 而非 graph hard mask**：维护 K=3–5 个 `(history, node, score)` 假设，用 graph transition 作为 soft bias。这样既利用任务图，又允许 fault、漏检和图中未建模路径。
- **长短期缓存**：最近 1–2 秒保留高分辨率 token，较老的 completed segments 压成一个 token；E2E-LOAD 的 stream buffer、长/短期分支与 token reuse 为这种部署结构提供了直接参考。[E2E-LOAD](https://openaccess.thecvf.com/content/ICCV2023/html/Cao_E2E-LOAD_End-to-End_Long-form_Online_Action_Detection_ICCV_2023_paper.html)

### 7.3 训练损失建议

```text
L = L_node
  + λ_tier3 L_aggregated_tier3
  + λ_state L_action/background
  + λ_boundary (L_start + L_end)
  + λ_progress L_progress
  + λ_smooth L_causal_temporal_smoothness
  + λ_align L_cross_modal_alignment
  + λ_consistency L_view/modal_consistency
```

- `L_node` 仍是主目标；`L_tier3` 只作辅助，避免为提高 Tier3 而牺牲 node。
- Boundary loss 必须独立报告事件级 P/R/F1、Segment F1、Edit、Pred/GT 和真实 emission latency。
- 用 modality dropout、camera dropout、EMG channel dropout、IMU device dropout 和 time-jitter 训练退化路径。
- 训练期可用 teacher 的完整片段，部署评估必须使用 predicted boundary 与 predicted history；两者结果分开报告。

## 8. 推荐实验路线与停止条件

### Phase A：先确认哪一种新信息有增量价值

固定当前四折三 seed、`all_runs`、同一 train/test manifest：

| ID | 当前观察 | 目的 |
|---|---|---|
| A0 | cam `001484412812` M2 | 已有基线 |
| A1 | 第二相机单独 M2 | 测该 view 自身质量 |
| A2 | 双相机 late probability fusion | 零/少参数互补性检查 |
| A3 | 双相机 gated residual/cross-view | 主多视角候选 |
| A4 | cam1 + IMU | 运动方向/相位增量 |
| A5 | cam1 + EMG | 激活/接触边界增量 |
| A6 | cam1 + EMG + IMU | 可穿戴内部互补性 |
| A7 | 双相机 + EMG + IMU | 完整模型 |

每个条件必须输出：总体、Normal/Fault、Stage、31 Tier3、35 node、当前前 12 个混淆对，以及 paired clip-level bootstrap CI。传感器有价值的最低证据不是一次最好 seed，而是：

- 12 个 fold×seed 中多数为正增益；
- Node Macro-F1 和最弱类别 Recall 同时改善；
- Fault 不退化；
- 对缺失模态/时间偏差的压力测试仍能回退到接近 A0；
- 延迟与吞吐满足目标硬件预算。

### Phase B：再把最佳 fusion 接入实时 boundary

1. 先完成现有 decoder 的 pending merge、阈值校准、overlap loss mask 和正确 segment GT；
2. 使用 M2-Direct 替换当前在线原型中的 M3 Atomic-tail classifier，保持 actual predicted chronological history；
3. 先加 EMG/IMU boundary token，不改 M2；
4. 再加多模态 M2 semantic token；
5. 最后尝试 boundary/node/progress 联合微调，并与模块化版本配对比较。

### Phase C：实时验收指标

- Boundary F1 @ ±3/±5/±10 frames；
- Segmental F1@10/25/50、Edit、frame accuracy；
- Pred/GT、false detections/min、过短片段比例；
- conditional Node Accuracy 与 end-to-end Node Accuracy；
- oracle-boundary 与 predicted-boundary 差距；
- start latency、end latency、decoder holdback、node-ready 总延迟；
- 每模态编码、fusion、boundary、M2 的分项 wall-clock latency；
- 单相机/单侧 MindRove/丢包/±时间偏移压力测试。

## 9. 最终建议

最合适的下一代模型不是“把更多输入直接塞进 M2”，而是建立一个 **multimodal observation encoder + 原 M2 history fusion**：

1. 先用两个视角和 EMG/IMU 生成质量感知的 `z_t`；
2. 保持 M2 的 single-query、actual-order、position-aware history 机制；
3. 让新模态以 zero-initialized residual 方式逐步修正当前 RGB M2；
4. 实时阶段让 EMG/IMU 先参与 boundary/progress，只有经过因果稳定器确认的 segment 才进入 M2 memory；
5. 用 soft graph bias/beam 管理 predicted history，避免一次边界或 node 错误锁死后续流程。

从当前错误分布看，**优先级建议是：第二视角互补性审计 → IMU → EMG → 完整融合**。如果目标首先是改善实时分割边界，则顺序改为：**EMG/IMU boundary fusion → decoder 修复 → 第二视角 semantic fusion → 联合模型**。

## 10. 产出文件

- `M2单摄像头性能与多传感器融合方案.md`：本报告。
- `m2_tier3_per_class_metrics.csv`：31 类完整统计。
- `m2_node_per_class_metrics.csv`：35 node 完整统计。
- `m2_tier3_top_confusions.csv`、`m2_node_top_confusions.csv`：主要混淆。
- `m2_repeated_node_pair_errors.csv`：重复 Tier3 node 的有向互换。
- `m2_overall_metrics.csv`、`m2_participant_metrics.csv`、`m2_stage_metrics.csv`：总体分解。
- `m2_fold_seed_metrics.csv`：12 个 fold×seed 的可追溯指标。
- `analysis_summary.json`：输入清单与关键审计结果。
- `analyze_m2.py`、`build_report.py`：可复算脚本。
"""
    (HERE / "M2单摄像头性能与多传感器融合方案.md").write_text(report, encoding="utf-8")
    print("report_written")


if __name__ == "__main__":
    main()
