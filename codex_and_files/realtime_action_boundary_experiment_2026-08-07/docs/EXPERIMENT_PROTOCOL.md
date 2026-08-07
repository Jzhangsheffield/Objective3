# 实验协议与配置说明

## 主问题

V1 只回答：在不知道真实动作边界的连续视频中，冻结既有视觉 backbone 后，独立因果 boundary detector 能否在严格 LOSO 下可靠检出动作片段，并为 M3 提供足够好的预测片段？

不在 V1 中联合微调 backbone 或 M3。这样可以把边界误差与 node 分类误差分开归因，也能最大限度复用已完成的动作识别实验。

## 实验矩阵

| 维度 | 取值 |
|---|---|
| Held-out participant | A、D、J、M |
| Seed | 1、2、42 |
| Train scope | normal-only、all-runs |
| Test split | normal、fault、all |
| 正式 feature stride | 1 frame |
| annotation | recognition boundaries + restored short background v1 |

共 4 × 3 × 2 = 24 个训练条件，每个条件产生 3 个测试报告。normal-only 和 all-runs 的测试集合相同，差别只在训练 runs 和对应 backbone/M3 checkpoint。

## 训练集内 validation

从当前 LOSO train runs 按 `sha256(seed:run_sample)` 排序，抽取 15% run 做 validation。切分单位是整 run，不允许把同一连续 run 的帧分到 train 和 validation 两边。held-out participant 从不参与阈值、epoch 或超参数选择。

如果需要统一选择在线阈值，应只在四个 fold 各自的 validation 上选择，再固定阈值测试；不能根据 test_normal/test_fault/test_all 反调。

## 模型配置

输入是 512-D frozen backbone 特征。TCN 隐藏维 256、5 个 residual blocks、kernel 3、dilation 1/2/4/8/16，每块两层因果卷积；总感受野 125 steps。三个目标头共享 TCN：

- 2-class state cross entropy；
- start sigmoid BCE；
- end sigmoid BCE。

总损失为 `1.0*state + 0.5*start + 0.5*end`。start/end 正样本权重初始设为 25，训练边界半径为 ±2 帧。这两个值只能根据 train/validation 调整。

## 在线状态机

默认阈值均为 0.55，start/end debounce 均为 2 steps，动作最短 3 steps，重复片段合并 gap 为 0。状态机允许：

`BACKGROUND → start candidate → ACTION → end candidate → emitted segment`

若证据未持续达到 debounce，候选被撤销。片段输出时记录 `emitted_at_index`，用于测量系统实际输出延迟。

## M3 与 history

M3 只在片段输出后运行。当前片段 feature 来自预测 start/end 内均匀采样的 16 帧。history 仅含此前预测片段的 feature，不含真实 node、真实边界、未来片段或重新排序后的 oracle 顺序。最大历史长度 35。

Atomic-tail 的 graph-valid/atomic-tail 规则是训练时 history augmentation；在线推理保持真实预测时间顺序。Task Graph 状态根据预测 node 更新，只审计直接前驱合法性。第一版不 hard-mask node logits。

## 指标与统计

边界事件按时间距离最小的一对一贪心匹配，分别报告 start/end 的 P/R/F1。容差为 ±3/±5/±10 帧。必须同时报告：

- mean signed error：正值代表预测偏晚；
- mean absolute error；
- emission delay：状态机确认 end 后才输出造成的额外延迟；
- Segmental F1@10/@25/@50；
- Edit Score；
- 帧级 action/background accuracy、precision、recall、F1；
- detected segment count 与 GT count；
- 端到端 node accuracy，以及仅在成功匹配片段上的 conditional node accuracy。

最终汇总先对每个 run 计算，再做 run-macro；同时建议增补全帧 micro 结果。跨 seed 报 mean±SD。Atomic-tail 对 M2 Direct 的比较应继续使用 participant-seed 配对，不用单次最好 seed 代替总体结论。

## 消融顺序

1. V1：state + start/end，多头 boundary detector，merge gap 0。
2. 去掉 start/end head，仅 state transition。
3. 去掉 state head，仅 boundary events。
4. debounce 1/2/3，min duration 1/3/5；只在 validation 选定。
5. stride 1 vs 2 vs 4，量化精度与算力权衡。
6. 独立模型稳定后，再建立联合 boundary + node 多任务版本；不得用真实历史 node。

## 失败判据

出现以下任一情况，结果不得宣称实时有效：使用中心窗口或未来帧；按测试结果选择阈值；M3 history 使用真实 node；直接使用真实 segment 抽取 node feature；训练/validation 按帧随机切分；短 background 在 GT 或后处理中被默认合并；fold/seed/scope checkpoint 混用。
