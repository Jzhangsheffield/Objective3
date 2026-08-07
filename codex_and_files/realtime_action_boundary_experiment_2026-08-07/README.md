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

机器相关路径集中在 [config_windows.bat](config_windows.bat)。`base.json` 和 `smoke_stride4.json` 只引用其中的环境变量；迁移电脑时通常不需要修改 JSON 或 Python 代码。

### 2.1 `config_windows.bat`：迁移电脑时首先修改

它沿用 M0–M6 项目 `bat\config_windows.bat` 的模式：所有变量都使用 `if not defined` 设置默认值，因此既可以编辑文件，也可以在 CMD 中预先设置变量进行临时覆盖。

新电脑上首先编辑以下3个输入路径：

```bat
set "DATASET_ROOT=E:\data\Action_Segmentation_Dataset"
set "PYTHON_BIN=C:\Miniconda3\envs\boundary\python.exe"
set "ATOMIC_PROJECT_ROOT=E:\projects\graph_history_rgb_cross_person_ADM_2026-07-22"
```

`PACKAGE_ROOT` 自动取 `config_windows.bat` 所在实验包目录，所以整个实验包移动后不必手工修改。默认结果保存在实验包内；如果希望 cache 和结果写到另一块硬盘，再修改：

```bat
set "PROTOCOL_ROOT=F:\boundary_runs\protocols"
set "FEATURE_CACHE_ROOT=F:\boundary_runs\cache\features"
set "OUTPUTS_ROOT=F:\boundary_runs\outputs"
set "SMOKE_OUTPUTS_ROOT=F:\boundary_runs\outputs_smoke_stride4"
set "VALIDATION_ROOT=F:\boundary_runs\validation"
```

主要变量含义：

| 变量 | 用途 |
|---|---|
| `PACKAGE_ROOT` | 当前边界实验代码目录，自动推导 |
| `DATASET_ROOT` | 连续视频数据集根目录，包含 `manifest.jsonl/raw/annotations` |
| `ANNOTATION_ROOT` | 新边界标注目录，默认从 `DATASET_ROOT` 自动拼接 |
| `PYTHON_BIN` | 新电脑实际使用的 `python.exe` |
| `ATOMIC_PROJECT_ROOT` | 原 M0–M6/Atomic-tail 项目根目录 |
| `CAMERA_ID` | RGB 相机，默认 `001484412812` |
| `PROTOCOL_ROOT` | 新连续 run LOSO 协议输出 |
| `FEATURE_CACHE_ROOT` | 因果512维特征 cache |
| `OUTPUTS_ROOT` | 正式 stride-1 训练和评估结果 |
| `SMOKE_OUTPUTS_ROOT` | stride-4 smoke 结果，与正式输出隔离 |
| `VALIDATION_ROOT` | setup validation 报告 |
| `NUM_WORKERS` | 正式训练 DataLoader workers |
| `RECOMMENDED_PARTICIPANTS/SEEDS/SCOPES` | 完整网格 BAT 的循环范围 |
| `SMOKE_HELDOUT/SEED/SCOPE` | smoke 使用的单个 LOSO 条件，默认 A/1/all-runs |
| `SMOKE_ONLINE_RUN` | `online` 阶段用于功能检查的连续 run |

修改后在 CMD 中检查解析结果：

```bat
call config_windows.bat show
```

两个运行脚本都会自动 `call config_windows.bat`，所以正常使用时不需要手动 call：

```bat
scripts\run_smoke_stride4.bat validate
scripts\run_loso_grid.bat prepare
```

也可以只在当前 CMD 临时覆盖，而不改文件。由于配置使用 `if not defined`，预先设置的值会被保留：

```bat
set "PYTHON_BIN=C:\Users\new_user\miniconda3\envs\boundary\python.exe"
set "FEATURE_CACHE_ROOT=G:\boundary_cache"
scripts\run_smoke_stride4.bat extract
```

如果绕过 BAT、直接运行 `python tools\*.py`，必须先执行 `call config_windows.bat`，否则 JSON 中的 `%RAB_*%` 变量无法解析。

由于它与旧 M0–M6 配置一样采用 `if not defined`，从旧实验切换到本实验时建议打开一个新的 CMD 窗口，避免旧窗口中已经存在的 `DATASET_ROOT/PYTHON_BIN/OUTPUTS_ROOT` 被当成有意覆盖值。两个本实验启动脚本会强制把 `PACKAGE_ROOT` 校正为当前边界实验包，但其他机器变量仍尊重预先设置的值。

## 3. 安装与快速核查

以下脚本均为 Windows CMD/BAT，不需要 PowerShell 执行策略。打开“命令提示符（cmd.exe）”，进入本目录后运行：

```bat
scripts\run_smoke_stride4.bat validate
scripts\run_smoke_stride4.bat prepare
```

脚本默认使用 `C:\Users\digit\anaconda3\envs\Pytorch\python.exe`。如果新电脑的 Python 路径不同，将配置和 Python 路径作为第二、第三参数传入：

```bat
scripts\run_smoke_stride4.bat validate "D:\boundary_experiment\configs\smoke_stride4.json" "C:\Miniconda3\envs\boundary\python.exe"
```

`--deep` 会读取 103 个新逐帧标注并确认每一张标注帧真实存在。核查报告写入 `validation/setup_validation.json`。协议生成只写本实验包的 `protocols/`，不会修改原项目协议。

本次已完成的核查结果见 [VALIDATION_REPORT_2026-08-07.md](docs/VALIDATION_REPORT_2026-08-07.md)。

## 4. 运行单个 LOSO 条件

推荐先运行固定的 A held-out、seed 1、all-runs、stride-4 smoke。一次执行完整基础流程：

```bat
scripts\run_smoke_stride4.bat all
```

`all` 依次执行：单元测试与数据核查、协议生成、特征提取、2 epochs 训练、normal/fault/all 边界评估，以及 `run_sample_000001` 的单 run 在线 M3 推理。任一步返回非零状态时，BAT 会立即停止，不会继续运行后续阶段。

也可以分阶段运行，适合检查速度或中断后续跑：

```bat
scripts\run_smoke_stride4.bat validate
scripts\run_smoke_stride4.bat prepare
scripts\run_smoke_stride4.bat extract
scripts\run_smoke_stride4.bat train
scripts\run_smoke_stride4.bat evaluate
scripts\run_smoke_stride4.bat online
scripts\run_smoke_stride4.bat end_to_end
```

`end_to_end` 不包含在 `all` 中，因为它需要对全部预测片段再次运行 M3，耗时明显更长。特征提取支持跳过已存在的 `.pt` cache；训练目录已有文件时会拒绝覆盖，因此重新训练前应改用新的输出目录，而不是删除已有正式结果。

建议从已经打开的 `cmd.exe` 运行，而不要双击 BAT；这样脚本结束或报错后窗口不会自动关闭。每个 smoke stage 都打印开始时间、结束时间和 `exit_code`。`exit_code=0` 表示成功，其他值表示该阶段失败。

保存完整控制台日志：

```bat
mkdir logs 2>nul
scripts\run_smoke_stride4.bat extract > logs\smoke_extract.log 2>&1
scripts\run_smoke_stride4.bat train > logs\smoke_train.log 2>&1
```

实时查看进度时不要重定向；或者在另一个 CMD 窗口运行 `type logs\smoke_extract.log` 查看已经写入的内容。

如果配置或 Python 不在默认位置：

```bat
scripts\run_smoke_stride4.bat all "E:\experiments\realtime_action_boundary_experiment_2026-08-07\configs\smoke_stride4.json" "C:\Miniconda3\envs\boundary\python.exe"
```

正式配置 `stride_frames=1`，用于保留不足 8/10 帧的短 background，特征提取计算量较大。可先用 `configs/smoke_stride4.json` 验证流程，但 stride 4 的结果不能作为短 background 正式结论，因为它有最多 3 帧的采样量化误差。

完整 LOSO 网格使用 `run_loso_grid.bat`。参数顺序为：

```text
run_loso_grid.bat STAGE [SCOPE] [CONFIG] [PYTHON_EXE]
```

- `STAGE`：`prepare`、`extract`、`train`、`evaluate` 或 `end_to_end`；
- `SCOPE`：`both`、`normal_only` 或 `all_runs`，默认 `both`；
- `CONFIG`：默认 `configs\base.json`；
- `PYTHON_EXE`：默认当前电脑的 Pytorch conda 环境。

完整运行顺序：

```bat
scripts\run_loso_grid.bat prepare
scripts\run_loso_grid.bat extract both
scripts\run_loso_grid.bat train both
scripts\run_loso_grid.bat evaluate both
scripts\run_loso_grid.bat end_to_end both
```

指定自定义配置和 Python：

```bat
scripts\run_loso_grid.bat extract all_runs "E:\experiments\boundary\configs\base.json" "C:\Miniconda3\envs\boundary\python.exe"
```

该脚本的循环顺序为 held-out participant A/D/J/M → seed 1/2/42 → scope。某一条件失败后脚本立即退出；修复问题后重新执行同一 stage 即可。`extract` 会跳过已有 cache，其他 stage 不会自动越过失败条件。

### 4.1 每个运行阶段具体做什么

整个流程的依赖关系是：

```text
validate
   ↓
prepare
   ↓
extract
   ↓
train
   ↓
evaluate
   ↓
online（单 run 功能检查）

train + extract + prepare
   ↓
end_to_end（全部测试 runs 的边界+Node综合评估）
```

| Stage | 是否训练参数 | 是否运行 RGB backbone | 是否运行 M3 | 是否使用 GT 计算指标 |
|---|---|---|---|---|
| `validate` | 否 | 否 | 否 | 不计算指标 |
| `prepare` | 否 | 否 | 否 | 只读取既有 split/fault 定义 |
| `extract` | 否，backbone 冻结 | 是 | 否 | GT 只用于写入训练标签 |
| `train` | 只训练 boundary TCN | 否，读取 cache | 否 | 使用训练/validation 标签 |
| `evaluate` | 否 | 否，读取 cache | 否 | 是，只在预测完成后评分 |
| `online` | 否 | 是，闭合片段重新提特征 | 是 | 不使用 GT 边界/history，不计算总体准确率 |
| `end_to_end` | 否 | 是，闭合片段重新提特征 | 是 | 是，只在预测完成后匹配和评分 |

#### Stage 1：`validate`

命令：

```bat
scripts\run_smoke_stride4.bat validate
```

这个阶段不提取特征、不训练模型，作用是确认新电脑上的环境和迁移文件完整：

1. 运行4个单元测试，检查因果 prefix invariance、一对一边界匹配、segment 重建以及短 background 不合并；
2. 读取 `manifest.jsonl` 和103个逐帧标注；
3. `--deep` 模式逐一确认251,132张标注 RGB 帧真实存在；
4. 检查 `frame_idx` 连续、`original_frame_idx` 严格递增；
5. 检查当前 config 要求的 backbone 和 M3 checkpoint 路径。

输出为 `validation\setup_validation.json`。`status=ok` 且 `problems=[]` 才能继续。这个阶段可反复运行，不会修改数据、checkpoint 或训练结果。

#### Stage 2：`prepare`

命令：

```bat
scripts\run_smoke_stride4.bat prepare
```

这个阶段把原 Atomic-tail 的动作级 LOSO 协议转换成连续视频 run 级协议：

1. 从结构化数据 `manifest.jsonl` 建立 `run_sample ↔ participant/source_run` 映射；
2. 读取旧项目的 `normal_only/all_runs` 与 `train/test_normal/test_fault/test_all`；
3. 按 `(participant, source_run)` 将动作样本合并为连续 run；
4. 验证训练集不含 held-out participant、train/test 无重叠、normal/fault 无重叠且并集等于 test-all；
5. 写入本实验包自己的 `protocols`，不修改旧 Atomic-tail 协议。

Smoke 配置只生成 A held-out 所需协议；正式 `base.json` 生成 A/D/J/M 四折。该阶段不需要 GPU，可确定性重复运行。

#### Stage 3：`extract`

命令：

```bat
scripts\run_smoke_stride4.bat extract
```

这是第一个主要耗时阶段，负责把连续 RGB 视频转换为 boundary 模型使用的因果特征序列：

1. 根据 held-out、seed、scope 加载对应的已训练 Tier3 ResNet3D-18 backbone；
2. 完全冻结 backbone，并切换到 inference/eval 模式；
3. 对每个 anchor 只读取“当前帧及过去帧”，组成16帧因果窗口；
4. 正式版 stride 1 每帧产生一个512维特征；smoke stride 4 每4帧产生一个特征；
5. 将 action/background、start、end 标签对齐到相同 anchor；
6. 保存特征、原始帧号、timestamp、标签、checkpoint 哈希和 annotation 哈希。

它不会训练 boundary TCN，也不会调用 M3。不同 held-out/seed/scope 使用不同 backbone，因此必须分别建立 cache，不能混用。

已有 `.pt` 时脚本默认跳过，适合中断续跑；但它不会自动判断旧 cache 是否来自另一份配置。如果修改了窗口、stride、RGB normalization 或 backbone，应使用新的 cache 路径，不能直接复用旧 `.pt`。

#### Stage 4：`train`

命令：

```bat
scripts\run_smoke_stride4.bat train
```

这个阶段只训练新的因果 Boundary TCN，不训练 RGB backbone，也不训练 M3：

1. 从当前 LOSO `train.jsonl` 找到训练 runs；
2. 按整 run、基于 seed 的确定性哈希划出15% validation，避免同一视频帧泄漏；
3. 从 `.pt` cache 读取 `[时间步,512]` 特征；
4. 切成带历史重叠的 causal chunks；
5. 联合优化 action/background state loss、start loss 和 end loss；
6. 每个 epoch 在 validation runs 上计算 loss；
7. 按最小 validation loss 保存 `best.pth`，最后一个 epoch 保存 `last.pth`。

Smoke 训练2 epochs，只用于验证流程；正式配置训练40 epochs。训练阶段看不到 held-out participant 的测试标签，也不根据 test-normal/test-fault 调参。

BAT 不传 `--overwrite`。如果目标训练目录已有内容，程序会停止，防止意外混合两次训练日志。要重新训练，应修改 `output_template` 或实验名称建立新目录。

#### Stage 5：`evaluate`

命令：

```bat
scripts\run_smoke_stride4.bat evaluate
```

这个阶段评估 boundary detector 本身，不调用 M3、不计算 Node Accuracy：

1. 加载 boundary `best.pth`；
2. 对 `test_normal`、`test_fault`、`test_all` 每个 run 做完整因果 forward；
3. 得到每个 anchor 的 action probability、start probability、end probability；
4. 通过防抖、最短持续时间和 `merge_gap_steps=0` 的在线状态机生成预测片段；
5. 预测结束后才读取 GT 进行指标计算，不把 GT 反馈给状态机。

输出包括 Boundary P/R/F1@±3/±5/±10帧、signed/absolute boundary error、emission delay、Segmental F1@10/25/50、Edit Score 和帧级 action/background 指标。

每个 split 写入 `metrics.json` 和 `predicted_segments.jsonl`。该阶段不更新模型参数，可以在保持 checkpoint 与配置不变时重复运行；改变阈值后应写入新的评估目录或保存清楚对应配置。

#### Stage 6：`online`

命令：

```bat
scripts\run_smoke_stride4.bat online
```

这是单个 run 的端到端功能检查，Smoke 固定使用 `run_sample_000001`：

1. boundary 模型按时间顺序输出概率；
2. 在线状态机在没有真实边界的情况下生成闭合片段；
3. 只有片段确认结束后，才在该预测片段内均匀采样16帧并重新运行 RGB backbone；
4. M3 使用当前预测片段特征和此前预测片段的 feature history 预测 node；
5. history 按实际预测时间顺序更新，不使用真实历史 node；
6. Task Graph 只审计预测 node 的前驱合法性，不 hard-mask logits。

这个阶段不使用真实动作边界、真实历史 node 或未来帧。逐帧 annotation 在这里仅用于定位连续 RGB 文件和原始帧号，不参与边界决策。输出 JSONL 用于人工检查片段、node、confidence、history length、graph validity 和预测可用时间；它不汇总全测试集准确率。

#### 独立阶段：`end_to_end`

命令：

```bat
scripts\run_smoke_stride4.bat end_to_end
```

这个阶段用于获得最终“自动边界检测 + M3 Node识别”的总体指标：

1. 对 normal、fault、all 三个 split 的每个 run 从空 history 开始；
2. 完整执行与 `online` 相同的因果边界检测、片段闭合、片段特征提取和 M3 node 预测；
3. 误检片段同样会进入后续预测 history，模拟真实系统中的误差传播；
4. 所有预测完成后，才使用 GT 片段按 IoU≥0.5 做一对一匹配；
5. 计算 detection precision/recall、成功匹配片段上的 `conditional_node_accuracy`，以及将漏检视为错误的 `end_to_end_node_accuracy`。

GT 只用于最后评分，不用于产生边界、修改 history 或纠正 node。`test_all` 会再次覆盖 normal+fault runs，因此完整 end-to-end 计算量约为只跑一次全部 held-out runs 的两倍；这也是它没有包含在 smoke `all` 中的原因。

输出位于 `end_to_end\test_normal`、`test_fault`、`test_all`，每个目录包含 `metrics.json` 和 `predicted_nodes.jsonl`。它不训练或修改任何 checkpoint。

#### `all` 到底包含什么

```bat
scripts\run_smoke_stride4.bat all
```

等价于依次运行：

```text
validate → prepare → extract → train → evaluate → online
```

它不包含 `end_to_end`。如果 `all` 在某一步失败，修复后可以从失败的 stage 单独继续；但如果 `train` 已经成功产生输出，再次从 `all` 开始会在训练阶段因防覆盖保护而停止。

### 4.2 Smoke 各阶段的预期输出

`validate` 成功时应看到4个单元测试均为 `ok`，并显示：

```text
status=ok
runs=103
action_frames=94698
background_frames=156434
backbone_checkpoints=1
m3_checkpoints=1
problems=[]
```

`prepare` 生成 `protocols\A_as_test\{normal_only|all_runs}`。`extract` 为 A/seed-1/all-runs 的103个连续 runs 生成 stride-4 cache：

```text
cache\features\A_as_test\all_runs\seed_1\stride_4\
  run_sample_000001.pt
  ...
  run_sample_000103.pt
  extraction_manifest.json
```

可在 CMD 中核对 cache 数量：

```bat
dir /b "cache\features\A_as_test\all_runs\seed_1\stride_4\*.pt" | find /c /v ""
```

结果应为 `103`。`train` 仅运行2 epochs，并写入与正式实验隔离的目录：

```text
outputs_smoke_stride4\A_as_test\all_runs\seed_1\causal_boundary_smoke_stride4\
  resolved_config.json
  training_log.jsonl
  best.pth
  last.pth
```

检查训练日志行数：

```bat
find /c /v "" "outputs_smoke_stride4\A_as_test\all_runs\seed_1\causal_boundary_smoke_stride4\training_log.jsonl"
```

应为2行。`evaluate` 在其下生成 `evaluation\test_normal`、`test_fault` 和 `test_all`；每个目录包含 `metrics.json` 和 `predicted_segments.jsonl`。`online` 生成：

```text
outputs_smoke_stride4\A_as_test\all_runs\seed_1\causal_boundary_smoke_stride4\online_pipeline\run_sample_000001.jsonl
```

两轮 smoke 的准确率不用于论文结论；这里只确认特征提取速度、loss 为有限值、checkpoint 能保存、三套指标非空以及 M3 在线输出能够生成。

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
