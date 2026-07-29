# 07 输出、复现与安全边界

## 1. 单participant/seed根目录

```text
outputs/<P>_as_test/cam_001484412812/seed_<S>/
```

`P ∈ {A,D,J,M}`，`S ∈ {1,2,42}`。

## 2. representation与模型目录

```text
backbone/
├── normal_only/
└── all_runs/

features/
├── retrained_normal_only/
└── retrained_all_runs/

history_models/
├── retrained_normal_only/normal_only/m0..m6/
├── retrained_all_runs/all_runs/m0..m6/
└── direct_head_fusion/
    ├── normal_only/m1_direct..m3_direct/
    └── all_runs/m1_direct..m3_direct/

e2e_baselines/
├── normal_only/
└── all_runs/
```

目录名中的两个scope含义不同：

- `retrained_*`描述feature/backbone representation来源；
- 内层`normal_only/all_runs`描述history模型训练manifest。

正式matched-scope比较要求二者相同。

## 3. 模型目录文件

典型history/direct目录：

```text
last.pth
train_log.json
experiment_config.json
learned_parameters.json
test_results/
  test_normal_metrics.json
  test_normal_predictions.csv
  test_normal_probabilities.pt
  test_fault_metrics.json
  test_fault_predictions.csv
  test_fault_probabilities.pt
  test_all_metrics.json
  test_all_predictions.csv
  test_all_probabilities.pt
completed.json
```

`completed.json`只在训练、三个split评估和必要文件均成功后写入，因此它是整个实验单元的提交标记。

## 4. Checkpoint内容

`last.pth`通常包含：

```text
model_state
optimizer_state
epoch
config
extra metadata
```

不要仅根据文件名判断来源，应检查config/metadata中的participant、seed、scope、feature cache与
backbone checkpoint。

## 5. Feature cache

每个scope至少有训练和held-out测试cache。Direct Head Fusion复用这些cache，不重新提取RGB特征。
metadata中的checkpoint路径可用于追溯representation。

## 6. 严格汇总目录

```text
outputs/cross_person_summary_normal_only_ADJM_3seeds/
outputs/cross_person_summary_all_runs_ADJM_3seeds/
outputs/training_scope_comparison_ADJM_3seeds/
outputs/direct_head_fusion_summary_ADJM_3seeds/
```

关键规模：

```text
原十模型完整网格：
4×3×2×10×3 = 720

原scope配对：
4×3×10×3 = 360

Direct：
4×3×2×3×3 = 216

Direct对两个reference：
216×2 = 432
```

## 7. 防覆盖层级

### Python层

`ensure_new_output_dir`拒绝非空目录，除非显式`--overwrite`。

### 脚本层

标准BAT/Slurm：

- completed存在则跳过；
- 目标非空但completed缺失则停止；
- 不传overwrite；
- 新实验写新目录。

### 汇总层

`--require-complete-grid`在写CSV之前检查网格。缺少任何model/split时停止，而不是用残缺数据生成看似
完整的均值。

## 8. 中断恢复

若任务中断：

1. 先检查日志、目标目录和是否有completed；
2. 若completed存在，视为完成，不重跑；
3. 若目录非空但无completed，人工判断最后成功步骤；
4. 需要重跑时先把“该实验自己的不完整目录”改名备份；
5. 不删除或移动其他scope、其他seed或原M0–M6结果；
6. 重新运行标准入口。

## 9. 复现检查清单

- Python环境是否为预期PyTorch环境；
- dataset root与camera字段；
- Task Graph和relation matrix快照；
- held-out participant未进入训练；
- seed写入独立`seed_N`目录；
- normal/all backbone没有混用；
- feature metadata指向正确last checkpoint；
- M1–M6 baseline来自相同scope/seed M0；
- M6历史概率来自冻结M0，不是真实标签；
- Direct入口没有baseline checkpoint；
- 固定epochs、无validation、使用last；
- 三个split全部完成；
- summary complete-grid通过。

## 10. 统计解释

正式总体：

```text
先participant内平均3 seeds
→ 再对4 participants等权平均
```

不能把以下对象当作独立受试者：

- 12个participant-seed；
- 1895个clips；
- 同run内多个clips；
- 103个run跨participant的简单混合。

seed-fold正向计数与run正向计数是描述性稳定性证据。正式置信区间应使用participant/run层级或paired
bootstrap，并保留participant作为外层独立单位。

## 11. 任务边界

当前模型输出当前clip的node/Tier3类别。它没有直接输出：

- 是否漏做；
- 是否多做；
- 是否非法重复；
- 是否错误跳转；
- run级fault score。

fault split只是包含fault runs的clip分类测试集。因此“fault上的分类性能提高”不能写成“已实现fault
detection”。

## 12. 部署边界

| 模型 | 部署性 |
|---|---|
| M0 | 可部署，无历史 |
| M1–M4 | 历史特征可获得时可部署 |
| M5 | 不可部署，读取真实历史node |
| M6 | 可部署，使用M0 soft probabilities |
| Direct M1–M3 | 可部署，依赖同run历史feature |
| E2E | 可部署，不依赖历史 |

真实在线部署还需要额外解决clip切分、历史缓存更新、run边界识别和延迟；当前离线实验默认这些信息已知。

