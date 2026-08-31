# 数据与运行环境审计（2026-08-31）

## 结论

生成 B0–B5 实验包所需的实验定义、manifest 字段和固定 A0 产物均已具备，没有需要用户补充的科学设计信息。正式训练前需要处理两个环境项：把 `dataset_root` 指向训练机上的原始数据，并使用安装了 PyTorch/torchvision 的 GPU Python。

## 已确认可用

- 外层参与者：A、D、J、M。
- seed：1、2、42。
- 每个外层 fold 的 `train/test_all/test_normal/test_fault` manifests 均存在。
- 12/12 个 cam0 A0 M2 final checkpoint 存在。
- 12/12 个 A0 `test_all` 概率文件存在；Normal/Fault 评价目录也已建立。
- 12/12 个 cam0 外层 train/test 512-D cache 存在。
- manifest 含 B0–B5 所需的 `sample_name`、participant、run、annotation order、stage、node、Tier3、RGB 相对路径和 MindRove 相对路径。
- 主相机固定为 `001484412812`，第二相机固定为 `001528512812`。

## 已进行实验的完整性

现有 Phase A 汇总显示：

- A1–A6 及 A0 对照的可比较实测集中在 `A_as_test/seed_1`；Phase A 汇总仅发现 54 个条件×split 指标，尚缺 234 个，因此不是完整 4×3 网格。
- 右手 S1–S12 仅发现 36 个指标，尚缺 396 个。
- 双手 S1–S12 的 `pooled_train` 与 `participant_calibrated` 同样只完成小范围 A fold/seed 1。
- B0 不直接复用这些小范围结果到新输出目录；它会按相同 Phase A 实现补齐并保留独立 provenance。

## 当前机器阻塞项

1. `C:/MyFolder/mes19jz/Stage_2_Mapstyle_Dataset` 当前不存在。没有原始 RGB/MindRove 文件就不能训练第二相机、内层 cross-fit backbone 或构建 IMU cache。
2. `python` 未注册为可直接使用的命令；可发现的 bundled Python 不含 `torch`。它只能用于 JSON/语法层检查，不能运行模型 smoke test 或训练。

这两个问题不影响包的生成，也不代表实验数据结构缺失。训练机上只需修改 [config/phase_b.json](config/phase_b.json) 中的 `dataset_root`，并把 `-Python` 指向原实验使用的 PyTorch 环境。

## 正式启动前检查

```powershell
python tools/audit_prerequisites.py --load-tensors
python tools/smoke_test.py
```

审计报告会保存到 `outputs/audit/prerequisite_audit.json`。只有其中 `formal_run_ready=true` 且 smoke test 为 PASS 时才建议加入 `-Execute`。

## 不需要补充的信息

- 不需要重新选择摄像头 ID。
- 不需要重新定义 A0 或 M2 history 顺序。
- 不需要为 B1/B2提供 validation participant；inner-LOSO 已从外层训练三人中定义。
- 不需要把 EMG 或双手数据加入本矩阵；它们是被先前证据主动排除的控制变量，而非缺失输入。

