# 实验包核查报告（2026-08-07）

在生成实验包后，已使用本机 `C:\Users\digit\anaconda3\envs\Pytorch\python.exe`（PyTorch 2.6.0+cu126、torchvision 0.21.0+cu126）完成以下核查。

## 数据与标注

- 新标注 run 数：103；
- 逐帧记录总数：251,132；
- action 帧：94,698；
- background 帧：156,434；
- 每条逐帧记录对应的 `001484412812` RGB 文件均存在；
- 每个 run 的 `frame_idx` 连续，`original_frame_idx` 严格递增；
- 新 segmentation 中的动作片段与原 graph-node action manifest 按 participant、run、动作顺序映射：103/103 runs 通过，1895/1895 actions 数量一致。

机器可读报告位于 `validation/setup_validation.json`。

## LOSO 协议

协议已从原 Atomic-tail 项目转换到连续 run，并额外验证：train 不含 held-out participant、test 只含 held-out participant、train/test 无 run 重叠、normal/fault 不重叠且并集等于 all。

| Held out | all-runs train | normal-only train | test normal | test fault | test all |
|---|---:|---:|---:|---:|---:|
| A | 79 | 61 | 15 | 9 | 24 |
| D | 78 | 55 | 21 | 4 | 25 |
| J | 73 | 55 | 21 | 9 | 30 |
| M | 79 | 57 | 19 | 5 | 24 |

协议来源和目标路径完整记录在 `protocols/protocol_report.json`。

## Checkpoint

- Tier3 RGB backbone：24/24 个 fold × seed × scope checkpoint 存在；
- M3 Atomic-tail Direct Fusion refresh-once：24/24 个 checkpoint 存在；
- A/all-runs/seed-1 实际加载验证：backbone 122/122 keys，M3 20/20 keys，无 missing/unexpected keys；
- 使用 16 张真实 RGB 帧完成一次 backbone forward，输出 shape `(512,)` 且全部为有限值。

## 代码测试

Python compileall 通过。5 个核心单元测试全部通过：

1. causal prefix invariance：在输入尾部追加未来特征不会改变此前输出；
2. boundary event matching 为一对一；
3. binary segment 重建正确；
4. `merge_gap_steps=0` 时短 background 不被状态机合并。
5. 两个DataLoader workers并行读取时，逐帧顺序、因果滚动窗口和anchor数量保持正确。

尚未执行正式特征缓存、40 epochs 训练或 24 条件完整网格；这些属于下一步耗时实验，而不是实验包生成核查。
