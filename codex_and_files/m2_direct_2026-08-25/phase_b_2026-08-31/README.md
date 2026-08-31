# Phase B：双摄像头 + 右手 IMU 融合实验包

本实验包实现 B0–B5，目标是验证：第二视角与 IMU 是否应作为与主摄像头同级的专家/输入流参与融合，而不是继续作为 A0 的 residual/delta 修正。正式统计单位固定为 4 个外层 LOSO fold（A/D/J/M）× 3 个 seed（1/2/42），不选择 best seed。

## 1. 先前实验给出的设计依据

当前完整可比较的实测范围只有 `A_as_test/seed_1`，共 431 clips，不能外推为四折结论：

- A0 Node accuracy / Macro-F1 为 91.65% / 90.06%。
- 独立第二相机 A1 为 94.43% / 91.87%；说明第二视角本身很强。
- 0.5/0.5 双相机概率融合 A2 为 95.13% / 93.75%，相对 A0 Macro-F1 +3.70 pp，Fault +8.02 pp。
- gated residual/cross-view A3 相对 A0 Macro-F1 -0.43 pp，相对 A2 -4.12 pp。
- A4/A5/A6 的 Macro-F1 增益仅 +0.50/+1.09/+0.32 pp；简单 EMG+IMU 叠加没有超过 IMU 条件 A5。
- 右手 sensor-only 最好的 S3（IMU）Macro-F1 为 79.95%，但能修正 22 个 A0 错误；说明它不是更强的主模型，却有互补信息。
- 双手 IMU 没有稳定优于右手；participant calibration 对不同信号条件有正有负。因此 B0–B5 固定使用右手 IMU，避免再引入未稳定的左手与测试者校准因素。

这些观察支持两条路线：B1/B2 做无泄漏的三专家决策融合；B3–B5 直接把两个 RGB encoder 和一个 IMU encoder 的表示送入对称融合器。

## 2. B0–B5 矩阵

| 条件 | 输入 | 融合位置 | 历史 | 训练要点 |
|---|---|---|---|---|
| B0 | cam0 + cam1 | 35-node 概率平均 | 各相机保留 M2 | 补齐 Phase A A1/A2 的 12 个 fold×seed；B0 指标取 A2 |
| B1 | cam0 M2 + cam1 M2 + IMU Direct Node | 温度校准后的静态 simplex logit 融合 | 相机专家保留 M2 | 每个外层 fold/seed 用严格 inner-LOSO OOF 预测拟合 3 个温度和 3 个非负权重 |
| B2 | 同 B1 | 样本级质量门控 | 同 B1 | 门控输入为 entropy、top-1、margin、专家间 JS divergence 和可用性；只在 OOF 上训练 |
| B3 | cam0 layer2 tokens + cam1 layer2 tokens + IMU tokens | 对称 bottleneck token fusion | 无 | 三路同级输入、辅助单模态损失、modality dropout；不再以 A0 为主干 |
| B4 | 同 B3 | 同 B3 | actual M2 history | 先融合每个历史 clip 的三路表示，再用 M2 actual-history head 分类 |
| B5 | 同 B4 | 双向 soft temporal alignment + bottleneck | actual M2 history | 摄像头 token↔IMU token 双向 attention，并加 camera–IMU 对比损失 |

详细冻结项、超参数、泄漏边界和评价规则见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)。

## 3. 当前运行前状态

代码与固定 A0 上游适合生成实验包；当前机器不适合直接正式训练：

- 配置中的数据根目录 `C:/MyFolder/mes19jz/Stage_2_Mapstyle_Dataset` 当前不存在。
- 当前可发现的 bundled Python 没有 PyTorch；正式训练需使用原 GPU/PyTorch 环境。
- A0 的 12 个 checkpoint、概率文件、主相机全局特征和四折 protocol 均存在。
- Phase A 新增实验目前只完成 `A_as_test/seed_1`；B0 会补齐另外 11 组 A1/A2 及第二相机上游。

完整审计见 [DATA_AND_RUNTIME_AUDIT.md](DATA_AND_RUNTIME_AUDIT.md)。这不是数据字段缺失；只需在训练机上修改 `config/phase_b.json` 的 `dataset_root` 并选择正确 Python。

## 4. 快速使用

先修改 [config/phase_b.json](config/phase_b.json) 中的 `dataset_root`。然后在本目录运行：

```powershell
.\scripts\run_all_phase_b.ps1 -Python "C:\path\to\pytorch-python.exe" -Device cuda
```

上面只做审计并生成 CSV 计划，不启动训练。确认后显式加入 `-Execute`：

```powershell
.\scripts\run_all_phase_b.ps1 -Python "C:\path\to\pytorch-python.exe" -Device cuda -Execute
```

调度器按 `expected_output` 自动跳过已完成任务；失败时停止并写 `outputs/job_runner_status.json`。重新执行相同命令即可续跑。不要同时启动两个写入同一 fold/seed 目录的任务。

首次迁移到训练机建议先运行：

```powershell
python tools/audit_prerequisites.py --load-tensors
python tools/smoke_test.py
python tools/generate_job_matrix.py --device cuda
```

完整矩阵共 510 个顺序 job（含最终汇总），其中严格 cross-fit 包含 72 个内层相机 backbone。单机顺序运行会非常久；多 GPU/HPC 可读取 `scripts/phase_b_job_matrix.csv`，只在不同输出目录之间并行。

## 5. 输出与汇总

- `outputs/B0_phase_a/`：B0 的第二相机、A1 和 A2。
- `outputs/crossfit_protocols/`：严格 inner-LOSO manifests。
- `outputs/crossfit/`：B1/B2 的 OOF 三专家训练与预测。
- `outputs/outer_experts/`：正式外层 IMU 专家。
- `outputs/temporal_caches/`：B3–B5 的 cam0/cam1/IMU token cache。
- `outputs/B1/` … `outputs/B5/`：各条件 checkpoint、参数与三种 split 结果。

全部完成后运行：

```powershell
python tools/summarize_phase_b.py
```

汇总会生成 `outputs/summary/fold_seed_metrics.csv`、`condition_summary.csv` 和 `completeness.json`，不会选择最优 seed。

## 6. 主要文件

- [config/phase_b.json](config/phase_b.json)：唯一正式配置源。
- [tools/generate_job_matrix.py](tools/generate_job_matrix.py)：生成全部 B0–B5 任务。
- [tools/fit_decision_fusion.py](tools/fit_decision_fusion.py)：B1/B2 严格 OOF 拟合。
- [tools/train_joint_fusion.py](tools/train_joint_fusion.py)：B3/B4/B5 训练与评估。
- [phase_b/models.py](phase_b/models.py)：IMU encoder、对称 token fusion、soft alignment 与 M2 history。
- [phase_b/calibration.py](phase_b/calibration.py)：温度、simplex 和质量门控。
