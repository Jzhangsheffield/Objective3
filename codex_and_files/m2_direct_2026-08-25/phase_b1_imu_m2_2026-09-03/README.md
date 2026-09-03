# B1_IMU_M2 独立追加实验包

本实验回答一个单独问题：在 Phase B 的 B1 三专家静态决策融合中，如果 IMU 也像两路 camera 一样使用实际 M2 历史，能否进一步提高跨参与者识别性能？

实验包只读取既有 `phase_b_2026-08-31` 资产，所有新增 cache、checkpoint、概率和汇总均写入本目录的 `outputs/`。不会覆盖或修改 B0–B5 的代码、配置和结果。

## 结论：想法合理，但需要控制变量

Phase A 中 IMU M2 的结果明显强于对应 Direct Node 条件，说明历史上下文对 IMU 有潜力。但旧比较同时改变了上游训练任务和分类结构，不能把全部提升都归因于历史。

因此这里采用更严格的设计：

- 复用 Phase B 已训练的 `IMUResNet10 Direct Node` encoder，并通过 cache 冻结；
- 在同一 512 维 IMU 表示上，从头训练一个 M2 历史头；
- cam0 M2、cam1 M2、外层划分、seed、B1 的温度与静态权重融合方式均保持不变；
- 用严格 inner-LOSO OOF 预测重新拟合融合参数。

关键比较为 `B1_IMU_M2 − B1`。它衡量的是“固定 IMU encoder 表示后，Direct 头改成 M2 历史头”带来的整体融合变化。

## 新条件的数据流

```text
当前及同一 run 的既往 IMU clip
        │
        ▼
Phase B 已训练的 IMUResNet10 encoder（冻结）
        │ 每个 clip 512-D global feature
        ▼
IMU M2 actual-history head（新增）
        │ 35 类概率
        ├──────────────┐
cam0 M2 概率 ──────────┤
cam1 M2 概率 ──────────┤  每专家温度 + 三个静态 simplex 权重
                       ▼
                 B1_IMU_M2 概率
```

详细模型、历史构造、数据隔离和统计协议见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)。全部可调参数集中在 [config/experiment.json](config/experiment.json)。

## 前置资产

需要完整的 Phase B 输出和原 M2 camera 工程输出。审计脚本逐一检查：

- 36 个 inner 单元：`4 outer × 3 inner × 3 seeds`；
- 12 个 outer 单元：`4 participants × 3 seeds`；
- inner IMU signal cache、Direct encoder checkpoint、两路 camera M2 OOF 概率；
- outer IMU frozen feature cache、B1 参照指标和所有 protocol manifest。

运行前可单独检查：

```powershell
python .\tools\audit_prerequisites.py
python .\tools\smoke_test.py
```

## 完整运行

在本目录打开 PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1 `
  -Python C:\Users\<用户名>\AppData\Local\miniconda3\envs\pytorch\python.exe `
  -Device cuda `
  -NumWorkers 0
```

Windows 上先用 `NumWorkers 0` 最稳妥；确认运行正常后可以改为适合机器的 worker 数量。脚本会先生成 98 个按依赖排序的 job，再顺序执行。

| 阶段 | 数量 | 内容 |
|---|---:|---|
| 00 | 1 | 前置资产审计 |
| 01 | 36 | 用既有 inner IMU encoder 提取冻结 512-D 特征 |
| 02 | 36 | 训练 inner IMU M2 并产生严格 OOF 概率 |
| 03 | 12 | 训练 outer IMU M2 并评估 held-out participant |
| 04 | 12 | 拟合并评估 B1_IMU_M2 |
| 05 | 1 | 汇总及配对比较 |
| 合计 | 98 | 只包含新增实验，不重复训练既有 camera/encoder |

### 中断后续跑

每个训练/提取任务都有 `completed.json` 完成标记。重新运行时已完成任务会自动跳过，也可从指定 job 开始：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1 -StartJob 38
```

不要对已有部分输出直接使用 `-overwrite`，除非确实希望重算该任务；脚本默认拒绝覆盖不完整目录，以避免混合两次运行结果。

## 输出

```text
outputs/
├─ audit/prerequisite_audit.json
├─ imu_features/inner/...       # 新提取的 inner 冻结 IMU feature
├─ imu_m2/inner/...             # inner-LOSO IMU M2 OOF 模型与预测
├─ imu_m2/outer/...             # 12 个 outer IMU M2 模型与预测
├─ B1_IMU_M2/...                # 12 个融合条件结果
└─ summary/
   ├─ run_metrics.csv
   ├─ condition_summary.csv
   ├─ paired_comparisons.csv
   ├─ fusion_parameters.csv
   ├─ completeness.json
   └─ SUMMARY.md
```

`SUMMARY.md` 会同时列出 B0、原 B1、新 B1_IMU_M2、原 IMU Direct 和新 IMU M2，并给出相同 outer×seed 上的配对差值、95% 置信区间和胜/平/负计数。

## 依赖与复现

依赖见 [requirements.txt](requirements.txt)。实验继续使用参与者 `A/D/J/M`、seed `1/2/42`、35 个 Node 和原 Phase B 的 normal/fault 定义。训练固定保存最后一轮，不使用 validation、early stopping 或 best-seed 选择。
