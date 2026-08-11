# 实时动作边界实验第一轮结果分析

日期：2026-08-10  
实验包：`realtime_action_boundary_experiment_2026-08-07`  
结果目录：`outputs/A_as_test/all_runs/seed_1/causal_boundary_tcn_v1`

## 1. 结论摘要

第一轮正式结果证明了端到端因果流程能够完整运行，但当前版本还没有得到可接受的在线分段结果。主要问题不是模型完全找不到动作，而是在线状态机把动作区间切成了大量极短碎片。

核心结果如下：

- 当前只有一个正式条件：A held-out、seed 1、all-runs、stride 1。尚不能得出跨被试、跨seed稳定性结论。
- 最优checkpoint出现在第4个epoch；此后训练loss继续下降，但validation loss持续恶化，存在明显过拟合。
- `test_all`共431个真实动作片段，模型输出2525个片段，是GT的5.86倍。
- 80.79%的预测片段长度恰好等于配置的最短长度3帧；78.41%的相邻预测片段只隔1帧。
- 当前IoU≥0.5检测precision为10.30%，recall为60.32%；主要矛盾是过分割造成的极低precision，而不是recall完全不足。
- 成功匹配片段上的M3 Node Accuracy为68.46%，完整端到端Node Accuracy为41.30%。与Atomic-tail使用真实片段时约91%的结果相比，自动边界误差及错误history污染造成了明显下降。
- 2525个预测中2265个没有匹配任何GT；但未匹配预测的M3置信度中位数仍为96.26%，说明M3对边界模型产生的短碎片明显过度自信。
- 只使用12个validation runs选择解码参数，并正确合并间隔不超过3帧的碎片后，held-out A测试集的预测片段可由2525降至411，IoU@0.5微平均Segment F1可由约17.60%提高到62.95%。这项只读诊断说明：当前首要工作应是修正并校准在线解码器，不应立即重训RGB backbone。

## 2. 结果范围与完整性

### 2.1 已完成条件

本次`outputs`中只有以下条件：

| 项目 | 值 |
|---|---:|
| Held-out participant | A |
| Seed | 1 |
| Training scope | all-runs |
| Feature stride | 1 frame |
| RGB causal window | 16 frames |
| Boundary model | 5-layer causal TCN |
| Training epochs | 40 |
| 测试runs | 24 |
| Normal测试runs | 15 |
| Fault测试runs | 9 |

结果目录中包含：

- `best.pth`和`last.pth`；
- 40行`training_log.jsonl`；
- normal、fault、all三套boundary evaluation；
- normal、fault、all三套end-to-end M3 evaluation；
- 每个测试run的预测片段和预测node记录。

三套end-to-end计数满足严格加和关系：

```text
normal + fault = all
GT:      294 + 137 = 431
Pred:   1855 + 670 = 2525
Matched: 178 + 82  = 260
Correct: 122 + 56  = 178
```

因此当前单条件结果文件内部是完整且自洽的。

### 2.2 尚未完成的实验范围

当前结果不能代表完整LOSO结论，因为还缺少：

- D、J、M三个held-out folds；
- seeds 2和42；
- normal-only训练scope；
- 跨fold、跨seed均值、标准差和配对比较。

本报告中的normal/fault差异仅描述participant A、seed 1，不应解释为总体fault泛化结论。

## 3. 数据划分和模型配置核查

训练协议包含79个非A runs，其中：

- 67个用于训练；
- 12个用于validation；
- validation由D/J/M各4个run组成；
- held-out A的24个run没有进入训练或validation。

因此当前划分保持了A-as-test的跨人原则。

训练集共有163,572个stride-1时间步，动作帧占39.41%，包含1236个start和1236个end事件。Validation共有31,330步，动作帧占37.43%，包含228对边界。Test-all共有56,230步，动作帧占32.92%。训练/validation与held-out A之间存在一定动作占比差异，可能影响固定阈值的校准。

边界标签半径为2帧，每个孤立边界通常扩展为约5个正样本。按训练集统计，边界正样本比例大约为3.8%，对应负正比约25:1，因此当前start/end的`pos_weight=25`在数量级上合理，不是过分割的首要嫌疑。

TCN的5个dilation层为1、2、4、8、16，每层包含两次因果卷积，理论感受野为125个feature steps。再考虑每个feature本身由过去16帧构成，单个输出最多利用约140帧的历史RGB信息。模型具备足够的短期上下文，当前失败更像是训练/解码方式问题，而不是完全缺少时序感受野。

## 4. 训练过程分析

### 4.1 最优epoch

| Epoch | Train loss | Validation loss | Validation state | Validation start | Validation end |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.3564 | 1.0895 | 0.3083 | 0.8138 | 0.7486 |
| 4（best） | 0.8540 | **0.8984** | 0.2497 | 0.6911 | 0.6063 |
| 10 | 0.6289 | 0.9202 | 0.2322 | 0.7505 | 0.6254 |
| 20 | 0.4378 | 1.0818 | 0.2516 | 1.0081 | 0.6523 |
| 30 | 0.3538 | 1.4392 | 0.2454 | 1.5152 | 0.8723 |
| 40（last） | 0.2728 | 1.6340 | 0.2998 | 1.7480 | 0.9205 |

从epoch 1到40，train loss下降79.89%；但从最佳epoch 4到epoch 40，validation loss上升81.88%。其中start loss恶化最明显。

评估脚本默认读取`best.pth`，checkpoint元数据也确认`best.pth`对应epoch 4，因此已报告测试指标没有误用epoch 40的`last.pth`。不过训练40 epochs明显浪费计算，并显示模型很早开始过拟合。

建议后续加入：

- patience约5–8的early stopping；
- 每epoch计算validation segment/boundary指标；
- 除validation loss外，保存validation Segment F1或Boundary F1最佳checkpoint；
- 保留top-k checkpoint，避免loss与最终解码指标不一致。

## 5. 原始Boundary结果

### 5.1 帧级、Edit和Segmental指标（run宏平均）

| Test split | Frame P | Frame R | Frame F1 | Frame Acc | Edit | Segment F1@10 | F1@25 | F1@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal | 67.55% | 67.75% | 67.27% | 74.55% | 16.86 | 24.45% | 23.06% | 16.84% |
| Fault | 67.48% | 61.65% | 63.98% | 75.64% | 22.56 | 30.68% | 29.17% | 20.12% |
| All | 67.52% | 65.46% | 66.04% | 74.96% | 19.00 | 26.79% | 25.36% | 18.07% |

按全部帧汇总的test-all微平均结果为：precision 67.72%、recall 68.95%、F1 68.33%、accuracy 78.96%。GT动作帧为18,509，预测动作帧为18,846，两者总量非常接近。

这说明模型大体知道“哪些时间区域像动作”，但没有形成正确的连续片段结构。较合理的帧级结果与很低的Edit/Segment F1同时出现，正是严重碎片化的典型表现。

### 5.2 Boundary容差指标

以下为test-all原始结果的run宏平均：

| 容差 | Start P | Start R | Start F1 | End P | End R | End F1 |
|---:|---:|---:|---:|---:|---:|---:|
| ±3 frames | 12.66% | 66.96% | 21.11% | 11.35% | 59.57% | 18.89% |
| ±5 frames | 13.43% | 71.75% | 22.43% | 12.90% | 68.16% | 21.48% |
| ±10 frames | 15.21% | 81.74% | 25.40% | 15.18% | 80.76% | 25.31% |

匹配成功的边界本身并不非常偏：±3帧匹配下，start MAE约1.29帧，end MAE约1.32帧。但precision非常低，说明大量额外边界才是主要问题。

### 5.3 过分割的定量证据

| 指标 | Test-all结果 |
|---|---:|
| 真实动作数 | 431 |
| 预测片段数 | 2525 |
| Pred/GT | 5.86× |
| 预测长度中位数 | 3 frames |
| 恰好3帧的预测 | 80.79% |
| 不超过5帧的预测 | 81.35% |
| 相邻预测间隔中位数 | 1 frame |
| 间隔≤1帧的相邻预测 | 78.41% |

配置中的`min_action_steps=3`直接决定了大量片段停在3帧。很多输出表现为：

```text
3帧action → 1帧background → 3帧action → 1帧background → ...
```

这不是数据中真实动作的结构，而是状态机在动作概率抖动时不断重启。

### 5.4 Normal与Fault

| Split | GT | Pred | Pred/GT | IoU@0.5 Detection P | Detection R | End-to-end Node Acc |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 294 | 1855 | 6.31× | 9.60% | 60.54% | 41.50% |
| Fault | 137 | 670 | 4.89× | 12.24% | 59.85% | 40.88% |
| All | 431 | 2525 | 5.86× | 10.30% | 60.32% | 41.30% |

Fault的检测recall和conditional Node Accuracy与normal非常接近；当前单条件下没有看到fault明显崩溃。Fault甚至具有略好的precision和Segment F1，主要因为过分割倍率较低。但只有A/seed-1，不能据此判断fault总体更容易。

## 6. 端到端Node结果

### 6.1 总体结果

| 指标 | Normal | Fault | All |
|---|---:|---:|---:|
| GT segments | 294 | 137 | 431 |
| Predicted segments | 1855 | 670 | 2525 |
| IoU≥0.5 matched | 178 | 82 | 260 |
| Node correct | 122 | 56 | 178 |
| Detection precision | 9.60% | 12.24% | 10.30% |
| Detection recall | 60.54% | 59.85% | 60.32% |
| Conditional Node Accuracy | 68.54% | 68.29% | 68.46% |
| End-to-end Node Accuracy | 41.50% | 40.88% | 41.30% |

指标关系为：

```text
end_to_end_node_accuracy = correct / GT
                         = detection_recall × conditional_node_accuracy
                         ≈ 0.6032 × 0.6846
                         = 0.4130
```

需要注意：`correct/GT`不会直接把2265个未匹配false positives加入分母。因此End-to-end Node Accuracy必须与Detection precision、Edit Score和Segment F1一起报告；单独报告41.30%会弱化过分割问题。

### 6.2 错误边界对M3 history的影响

Test-all中：

- 2265/2525，即89.70%的预测片段没有匹配GT；
- 只有19.92%的预测node满足当前Task Graph审计；
- 只有25.19%的预测满足immediate predecessor检查；
- 63.21%的相邻预测给出相同node；
- 全部预测的confidence中位数为96.88%；
- 未匹配预测的confidence中位数仍为96.26%。

因此大量3帧碎片被M3当成完整动作，写入history并产生高置信node。M3只使用最后35个feature history，但每个run很快就会被false segments填满；Task Graph状态又记录所有预测node。原Atomic-tail模型是在真实动作片段分布上训练的，当前3帧碎片属于明显的输入分布外样本。

匹配片段中最常见的错误包括：

| GT node | Predicted node | 次数 |
|---|---|---:|
| 20 `grip sample from machine table 3` | 17 `grip sample from machine table 2` | 10 |
| 24 `put sample on table` | 12 `take plier from table` | 10 |
| 16 `put sample on machine table 1` | 19 `put sample on machine table 2` | 9 |

第一和第三类仍有重复阶段动作混淆特征；第二类更像是不完整边界、物体信息不足和错误history共同造成。不能把这些错误直接等同于之前oracle-boundary M3的混淆分布。

### 6.3 Run级差异

原始Segment F1@0.5最差的runs：

| Run | GT | Pred | Pred/GT | Segment F1@0.5 |
|---|---:|---:|---:|---:|
| run_sample_000007 | 14 | 120 | 8.57× | 7.46% |
| run_sample_000018 | 6 | 31 | 5.17× | 10.81% |
| run_sample_000002 | 14 | 123 | 8.79× | 11.68% |
| run_sample_000004 | 24 | 194 | 8.08× | 11.93% |
| run_sample_000006 | 14 | 98 | 7.00× | 12.50% |

端到端Node Accuracy最差的是：

| Run | GT | Pred | Matched | Correct | End-to-end Node Acc |
|---|---:|---:|---:|---:|---:|
| run_sample_000018 | 6 | 31 | 2 | 1 | 16.67% |
| run_sample_000007 | 14 | 120 | 5 | 3 | 21.43% |
| run_sample_000013 | 25 | 121 | 14 | 6 | 24.00% |
| run_sample_000010 | 14 | 73 | 7 | 4 | 28.57% |
| run_sample_000011 | 14 | 62 | 7 | 4 | 28.57% |

即使表现最好的run仍普遍存在大量额外片段，说明过分割不是少数异常run造成，而是系统性问题。

## 7. 代码级原因分析

### 7.1 Start和End head当前不是必要条件

在线状态机使用：

```python
start_evidence = start_probability >= start_threshold \
                 or action_probability >= action_threshold

end_evidence = end_probability >= end_threshold \
               or action_probability < (1.0 - action_threshold)
```

所以start head和end head都可以被state head绕过。Test-all中：

- 30.50%的片段在`start_score < 0.55`时仍然启动，说明它们由action state触发；
- 52.71%的片段在`end_score < 0.55`时仍然结束，当前结束步必然由低action state触发。

这解释了为什么模型虽然单独训练了boundary heads，最终解码却主要受state概率抖动影响。

### 7.2 当前merge实现不能真正合并已输出片段

`CausalBoundaryStateMachine`在片段结束时立即把片段返回给调用方。后续如果发现间隔小于`merge_gap_steps`，代码只是把新片段的start改成上一个片段的start，并再次输出；它无法撤回已经输出的前一个片段。因此直接把`merge_gap_steps`从0改成正数会产生重叠/重复记录，不能作为正式修复。

真正的因果合并需要：

1. 片段结束后先放入pending状态；
2. 等待最多`merge_gap_steps`；
3. 若短间隔内重新进入动作，则延长pending segment；
4. 超过允许间隔后才正式emit；
5. 在输出中记录由holdback带来的额外延迟。

### 7.3 Chunk overlap当前同时参与两次loss

训练chunk长度256、overlap 124，124正好接近TCN感受野减1。这一设计原意应是给后一个chunk提供左侧因果上下文。

但当前`BoundaryChunkDataset`和`collate_chunks`把整个chunk的mask都设为True，因此重叠帧会：

- 在前一个chunk尾部以完整历史参与一次loss；
- 在后一个chunk头部以零填充的截断历史再次参与loss。

同一帧被重复训练，而且两次看到的因果上下文不同。正确做法应把后一个chunk前124步作为context-only区域，不参与loss，仅对新core区域计算loss。该问题可能降低边界head的稳定性，建议在第二版训练前修正。

### 7.4 Segmental GT构造遗漏相邻动作内部边界

Boundary evaluation通过二值`state`构造GT segments。若两个不同动作之间没有background帧，二值state会保持1，从而把两个动作合成一个GT segment。

本轮A测试集：

- boundary exact events和end-to-end node annotation共有431个动作；
- 当前Segmental F1的二值state GT只有425个；
- normal少3个，fault少3个，共遗漏6个相邻动作内部边界。

后续Segmental F1应使用`segment_no`或exact start/end重建GT片段，而不是只使用background/action二值状态。

### 7.5 当前“emission delay=1帧”不是完整检测延迟

当前延迟定义为：

```text
predicted emitted_at - predicted end
```

在`end_debounce=2`时，它机械地等于1帧，所以所有split都报告1.0。它没有与真实动作end比较，不能反映模型是提前还是延后结束。

建议同时报告：

- decoder holdback/debounce latency；
- `emitted_at - matched_GT_end`真实可用延迟；
- start detection latency；
- M3完成node预测后的总系统延迟。

## 8. Validation-only解码校准诊断

### 8.1 方法

为避免使用held-out A测试标签调参，本报告进行了只读诊断：

1. 只在训练时预留的12个validation runs上搜索；
2. 搜索boundary threshold、action threshold、debounce、最短长度和正确短间隔合并；
3. 目标为微平均Segment F1@IoU 0.5；
4. 选定一次参数后，再一次性应用于A测试集；
5. 没有修改现有checkpoint、cache或正式结果文件。

Validation选出的设置是：

| 参数 | 当前 | Validation选择 |
|---|---:|---:|
| start threshold | 0.55 | 0.85 |
| end threshold | 0.55 | 0.85 |
| action threshold | 0.55 | 0.65 |
| start debounce | 2 | 2 |
| end debounce | 2 | 2 |
| min action steps | 3 | 5 |
| 正确merge gap | 0 | 3 |

注意：这里的merge是分析中实现的正确合并，不是当前代码中有缺陷的`merge_gap_steps`行为。

### 8.2 Validation变化

| 指标 | 当前解码 | Validation选择解码 |
|---|---:|---:|
| GT segments | 228 | 228 |
| Pred segments | 1633 | 246 |
| Pred/GT | 7.16× | 1.08× |
| IoU@0.5 Precision | 9.43% | 77.64% |
| IoU@0.5 Recall | 67.54% | 83.77% |
| IoU@0.5 F1 | 16.55% | 80.59% |
| Pred length median | 3 frames | 45 frames |

### 8.3 Held-out A一次性诊断结果

| 指标 | 当前正式结果 | Validation选择解码 |
|---|---:|---:|
| GT segments | 431 | 431 |
| Pred segments | 2525 | 411 |
| Pred/GT | 5.86× | 0.95× |
| IoU@0.5 Precision | 10.30% | 64.48% |
| IoU@0.5 Recall | 60.32% | 61.48% |
| IoU@0.5 F1 | 约17.60% | 62.95% |
| Frame macro F1 | 66.04% | 68.52% |
| Edit macro | 19.00 | 87.74 |

Normal与fault的校准后IoU@0.5微平均结果：

| Split | GT | Pred | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Normal | 294 | 289 | 60.90% | 59.86% | 60.38% |
| Fault | 137 | 122 | 72.95% | 64.96% | 68.73% |
| All | 431 | 411 | 64.48% | 61.48% | 62.95% |

Test-all边界微平均F1也明显改善：

| 容差 | 当前 Start F1 | 校准 Start F1 | 当前 End F1 | 校准 End F1 |
|---:|---:|---:|---:|---:|
| ±3 | 19.96% | 28.50% | 18.00% | 35.15% |
| ±5 | 21.45% | 41.09% | 20.64% | 48.22% |
| ±10 | 24.15% | 61.05% | 23.95% | 66.51% |

这说明当前TCN概率中已经包含相当多可用分段信息，主要被过低阈值、3帧最短长度和缺失的正确合并破坏。

校准后预测片段减少83.72%，M3调用次数理论上由2525降至411，约减少到原来的1/6.14。这不仅能减少false positives和history污染，也会显著降低端到端M3计算量。

### 8.4 不能直接当作正式新结果的原因

上述校准结果是诊断性派生结果，尚不能替代正式输出，因为：

- 当前在线代码还没有正确的pending merge实现；
- 合并会增加约数帧的因果holdback延迟，需要正式记录；
- 尚未重新运行M3，因此不知道较干净的history能把68.46%的conditional Node Accuracy提高到多少；
- 参数搜索是在看到第一轮结果后设计的，应在下一阶段固定搜索空间和validation目标，再应用于其他fold/seed；
- 仍需生成新的独立输出目录，不能覆盖本轮基线。

## 9. 推荐下一步

### 阶段R1.1：不重训，先修正和校准decoder

优先级最高，且不需要重新提取特征或重新训练TCN：

1. 实现真正的pending segment和因果短间隔合并；
2. 将start/end/state组合逻辑做成显式配置；
3. 增加validation-only阈值搜索脚本；
4. 同时优化Segment F1@0.5、Boundary F1和预测片段倍率，避免单指标过拟合；
5. 固定参数后重新运行`evaluate`；
6. 使用新片段重新运行`end_to_end`，测量M3 conditional和最终Node Accuracy；
7. 单独记录decoder、M3和总系统延迟。

建议新输出目录示例：

```text
outputs_decoder_calibration_v1/
  A_as_test/all_runs/seed_1/
    validation_search/
    evaluation/
    end_to_end/
```

不要覆盖当前`outputs`，因为它是重要的原始decoder基线。

### 阶段R1.2：修正评估定义

在扩大实验前修正：

- Segmental GT按`segment_no`构造；
- 同时输出macro和micro指标；
- 增加Pred/GT、长度分布、短片段比例和gap分布；
- 报告真实GT相对检测延迟；
- 明确End-to-end Node Accuracy不直接惩罚false positives；
- 增加false detections per minute或每1000帧误报数。

### 阶段R2：必要时重训Boundary TCN

只有在decoder校准后仍不够好时再重训：

1. 把chunk overlap设为context-only，不对重叠前缀重复计算loss；
2. 增加early stopping和top-k checkpoint；
3. 在validation上联合选择checkpoint与decoder；
4. 检查boundary概率校准，并考虑focal loss或temporal smoothness loss；
5. 比较较小TCN、较强dropout和weight decay，缓解epoch 4后的快速过拟合。

### 阶段R3：再扩展完整LOSO网格

先在A/seed-1/all-runs确认以下条件：

- 预测片段倍率接近1；
- Segment F1@0.5明显稳定；
- 正确merge和真实延迟指标通过测试；
- M3 history不再被大量false segments污染；
- end-to-end输出数量、图合法率和置信度合理。

之后再运行：

```text
4 folds × 3 seeds × 2 scopes
```

并保持normal、fault、all三套测试逻辑。

## 10. 最终判断

第一轮不应被解释为“实时边界模型只能达到约18%的Segment F1”。更准确的判断是：

1. Frozen RGB特征和Causal TCN已经学到可用的动作区间信号；
2. 当前固定decoder把这些信号转换成了大量3帧碎片；
3. 过分割进一步污染M3 history，使Node Accuracy从oracle-boundary水平显著下降；
4. Validation-only校准显示无需重训即可大幅改善分段结构；
5. 下一轮最有价值的工作是修正在线状态机、评估定义和chunk loss mask，然后重新进行独立decoder-calibration实验。

因此，推荐的最小下一步不是重新提取特征，也不是重新训练M3，而是建立一个不覆盖当前结果的`decoder_calibration_v1`阶段，正式验证校准后boundary与end-to-end Node结果。

## 11. 主要代码和数据依据

- 实际运行配置：`outputs/A_as_test/all_runs/seed_1/causal_boundary_tcn_v1/resolved_config.json`
- 训练曲线：`outputs/A_as_test/all_runs/seed_1/causal_boundary_tcn_v1/training_log.jsonl`
- Boundary指标：`outputs/A_as_test/all_runs/seed_1/causal_boundary_tcn_v1/evaluation/*/metrics.json`
- Boundary预测：`outputs/A_as_test/all_runs/seed_1/causal_boundary_tcn_v1/evaluation/*/predicted_segments.jsonl`
- End-to-end指标：`outputs/A_as_test/all_runs/seed_1/causal_boundary_tcn_v1/end_to_end/*/metrics.json`
- Node预测与Task Graph审计：`outputs/A_as_test/all_runs/seed_1/causal_boundary_tcn_v1/end_to_end/*/predicted_nodes.jsonl`
- 在线状态机：`boundary_experiment/online.py`
- TCN结构与感受野：`boundary_experiment/models.py`
- Chunk和mask构造：`boundary_experiment/data.py`
- 训练与best checkpoint选择：`tools/train_boundary.py`
- Boundary指标定义：`boundary_experiment/metrics.py`、`boundary_experiment/engine.py`
- End-to-end匹配和Node Accuracy定义：`tools/evaluate_end_to_end.py`
- M3在线history更新：`boundary_experiment/m3_adapter.py`
