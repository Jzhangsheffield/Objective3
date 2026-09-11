# 双腕 IMU / EMG 实时动作边界实验包

版本：2026-09-06

这个实验包在同一套严格 LOSO 协议下分别训练：

- `IMU-only`：左右手 accelerometer + gyroscope；
- `EMG-only`：左右手 16 路表面肌电。

两种实验共享标签生成、causal TCN、在线状态机和评价指标，但配置、缓存和输出位于不同子目录。第一版的目标是比较传感器模态的实时动作分割能力，不进行 IMU+EMG 融合，也不修改已有 RGB、M0–M6、Direct、Dynamic 或 Atomic-tail 实验。

## 1. 目录结构

```text
realtime_imu_emg_boundary_experiment_2026-09-06/
├─ config_windows.bat          新电脑上集中修改路径
├─ configs/
│  ├─ common.json              两种模态共享条件
│  ├─ imu/base.json            IMU 正式配置
│  ├─ imu/smoke.json           IMU 小规模测试
│  ├─ emg/base.json            EMG 正式配置
│  └─ emg/smoke.json           EMG 小规模测试
├─ sensor_boundary/            共享 Python 实现
├─ tools/                      各实验阶段入口
├─ scripts/                    Windows .bat 启动器
├─ protocols/                  固定 LOSO run 列表
├─ cache/
│  ├─ imu/                     自动生成
│  └─ emg/                     自动生成
├─ outputs/
│  ├─ imu/                     正式 IMU 结果
│  └─ emg/                     正式 EMG 结果
├─ outputs_smoke/              smoke 结果
├─ validation/                 数据核查报告
├─ docs/                       方法、协议和文献
└─ tests/                      单元测试
```

`cache`、`outputs`、`outputs_smoke` 和 `validation` 不会写入已有 RGB 项目。

## 2. 输入数据

默认数据集：

```text
D:\Junxi_data\MULTISENSOR_DATA_COLLECTION_Stage2_structured_data\Action_Segmentation_Dataset
```

每个 run 使用：

```text
raw\run_sample_xxxxxx\mindrove_left.csv
raw\run_sample_xxxxxx\mindrove_right.csv
annotations\action_recognition_timestamps_canonical_v2\
```

传感器同步以 `board_ts` 为唯一时间依据。动作标签来自包含短 background 的 segmentation annotation。模型看不到 RGB 图像、未来传感器样本、真实边界或真实历史 node。

## 3. 新电脑配置

打开 `config_windows.bat`，至少检查：

```bat
set "DATASET_ROOT=D:\your_path\Action_Segmentation_Dataset"
set "PYTHON_BIN=C:\your_conda\envs\sensor_boundary\python.exe"
```

可选调整：

```bat
set "NUM_WORKERS=4"
set "CACHE_ROOT=E:\sensor_boundary_cache"
set "OUTPUTS_ROOT=E:\sensor_boundary_outputs"
```

查看最终路径：

```bat
call config_windows.bat show
```

安装依赖：

```bat
call config_windows.bat
"%PYTHON_BIN%" -m pip install -r requirements.txt
```

建议使用包含 CUDA 的 PyTorch 环境。特征/窗口提取主要受 CSV 读取、滤波和磁盘写入影响；训练阶段 GPU 收益更明显。

迁移时需要复制：

1. 整个本实验包，包括其中的 `protocols`；
2. 完整 `Action_Segmentation_Dataset`，至少包含 `manifest.jsonl`、`raw`、指定 annotation 子目录；
3. Python/PyTorch 环境，或者在新电脑重新安装 `requirements.txt`。

边界实验本身不需要复制 RGB backbone 或 M3 checkpoint。只有后续把检测片段交给 M3 做 node 分类时才需要 Atomic-tail 项目及 checkpoints。

## 4. 八个阶段分别做什么

### validate

只读检查 manifest、103 个 run、左右手 CSV、所需通道、annotation 和协议目录。正式命令带 `--deep`，会读取每个传感器时间戳列，检查单调性和与 annotation 的时间重叠。

```bat
scripts\run_smoke.bat both validate
scripts\run_loso_grid.bat both validate
```

输出：`validation\imu\setup_validation.json` 和 `validation\emg\setup_validation.json`。

### prepare

验证并固定 A/D/J/M 的 LOSO run 列表，确认：

- held-out participant 不出现在 train；
- train 与 test 没有 run 重叠；
- `test_normal + test_fault = test_all`。

```bat
scripts\run_smoke.bat imu prepare
```

IMU 和 EMG 使用相同协议，因此执行一次即可。协议已随包提供；该阶段仍建议在新电脑运行一次进行完整性检查。

### extract

读取左右手 MindRove CSV，并执行：

1. 根据 `board_ts` 排序并去重；
2. 截取左右手和 annotation 的共同时间范围；
3. previous-sample hold 因果同步；
4. IMU：100 Hz、40 Hz causal low-pass、过去 0.5 s 窗口；
5. EMG：500 Hz、20–200 Hz causal band-pass、过去 0.2 s 窗口；
6. 每 50 ms 建立一个决策点；
7. 按绝对时间生成 state/start/end/segment ID；
8. 保存 float16 窗口缓存和 JSON metadata。

```bat
scripts\run_smoke.bat imu extract
scripts\run_smoke.bat emg extract
```

缓存位置：`cache\imu`、`cache\emg`。缓存与 seed 无关，也不含训练折归一化，因此可在所有 LOSO 条件间复用。程序发现已有 `.pt + .json` 时会跳过；只有显式传入 `--overwrite` 才重建。

### train

对当前 LOSO train runs 先划分 run-level validation，再只用实际 training runs 计算通道均值和标准差。随后联合训练：

```text
past-only local window → local 1D CNN → causal TCN → state/start/end heads
```

```bat
scripts\run_smoke.bat imu train
scripts\run_smoke.bat emg train
```

保存 `best.pth`、`last.pth`、`training_log.jsonl` 和包含训练/验证 run 清单及归一化统计的 `resolved_config.json`。

### calibrate

只在训练参与者的 validation runs 上搜索概率阈值、start/end debounce、最短动作长度和短间隔合并长度。目标函数是 start F1@200 ms、end F1@200 ms 与 Segmental F1@50 的均值。选定参数保存为 `calibrated_online.json`；evaluate 和 online 会自动读取它。

```bat
scripts\run_smoke.bat both calibrate
```

这个阶段不能使用 held-out participant，也不能查看 test 结果后手动选择阈值。

### evaluate

加载 `best.pth`，在 `test_normal`、`test_fault`、`test_all` 上分别运行 causal inference 和在线状态机。输出边界 F1、时延、Segmental F1、Edit Score、帧级指标、碎片比例和 real-time factor。

```bat
scripts\run_smoke.bat both evaluate
```

结果位于相应 condition 下的 `evaluation\test_*`。

### online

按时间顺序回放一个 run，保存每 50 ms 的三个概率以及状态机最终发出的片段：

```bat
scripts\run_smoke.bat imu online
scripts\run_smoke.bat emg online
```

输出：

```text
online_pipeline\run_sample_000001_stream.jsonl
online_pipeline\run_sample_000001_segments.json
```

当前 replay 使用分块加速，但每个输出只依赖当前及过去窗口；在数学上等价于 causal stream。它不测量传感器蓝牙/串口接入延迟，报告的是缓存回放的模型计算开销和算法发出延迟。

### summarize

扫描全部正式 condition 的 `metrics.json`，生成便于统计分析的单一 CSV：

```bat
scripts\run_loso_grid.bat both summarize
```

输出为 `outputs\loso_summary.csv`。Smoke 的汇总可直接调用 `tools\summarize_results.py`，但不应混入正式表格。

## 5. Smoke 从头运行

先进入实验包根目录：

```bat
cd /d D:\path\realtime_imu_emg_boundary_experiment_2026-09-06
call config_windows.bat show
scripts\run_smoke.bat both all
```

也可以逐步运行：

```bat
scripts\run_smoke.bat both validate
scripts\run_smoke.bat both prepare
scripts\run_smoke.bat imu extract
scripts\run_smoke.bat emg extract
scripts\run_smoke.bat imu train
scripts\run_smoke.bat emg train
scripts\run_smoke.bat both calibrate
scripts\run_smoke.bat both evaluate
scripts\run_smoke.bat both online
```

Smoke 固定使用 heldout A、seed 1、all-runs；每种模态只取 4 个训练 run，并对每个测试 split 最多取 2 个 run，训练 2 epochs。因此它用于确认程序、显存、速度和输出结构，不是可报告的实验结果。

自定义调用（自定义 config 时只指定单个 modality）：

```bat
scripts\run_smoke.bat imu train "D:\package\configs\imu\smoke.json" "C:\Miniconda3\envs\sensor\python.exe"
```

## 6. 正式 LOSO 运行

建议逐阶段运行，不要直接连续启动 48 次训练：

```bat
scripts\run_loso_grid.bat both validate
scripts\run_loso_grid.bat imu prepare
scripts\run_loso_grid.bat both extract both
scripts\run_loso_grid.bat imu train both
scripts\run_loso_grid.bat emg train both
scripts\run_loso_grid.bat imu calibrate both
scripts\run_loso_grid.bat emg calibrate both
scripts\run_loso_grid.bat imu evaluate both
scripts\run_loso_grid.bat emg evaluate both
scripts\run_loso_grid.bat both summarize
```

每种模态共：4 held-out participants × 3 seeds × 2 scopes = 24 个训练条件。每个条件输出 normal、fault、all 三套测试结果。

汇总表：`outputs\loso_summary.csv`。

如果正式训练中断：

- `extract` 可以直接重跑，已有缓存会跳过；
- `train` 默认拒绝覆盖非空输出，便于保护已有结果；
- 只有确认需要从头重训时，才直接调用 Python 并添加 `--overwrite`；
- 当前版本不实现 checkpoint resume。

## 7. 关键配置

IMU：`configs\imu\base.json`

```json
"sample_rate_hz": 100,
"decision_rate_hz": 20,
"local_window_seconds": 0.5,
"lowpass_hz": 40.0
```

EMG：`configs\emg\base.json`

```json
"sample_rate_hz": 500,
"decision_rate_hz": 20,
"local_window_seconds": 0.2,
"bandpass_hz": [20.0, 200.0],
"notch_hz": null
```

在线参数位于 `configs\common.json`。在首轮正式结果前不要针对 test split 修改它们；如果校准，应增加独立脚本在 validation runs 上搜索，再冻结后评估测试集。

## 8. 因果性和已知限制

- 重采样采用 previous-sample hold，不用目标时间之后的样本。
- 滤波使用 `sosfilt/lfilter` 正向状态递推，不用 `filtfilt`。
- 每个局部窗口只包含当前及过去采样点。
- TCN 只做左侧 padding。
- 训练标签可以来自完整 GT；推理输入不包含 GT。
- 数据有效性通道会告诉模型左右手最近样本是否超过允许 gap。
- 第一版的 IMU 100 Hz 重采样是工程基线；若频谱审计发现 40–250 Hz 的显著混叠能量，应增加“先在更高统一频率因果低通、再降采样”的消融实验。
- 第一版采用训练折全局均值/标准差。EMG 跨参与者漂移明显时，再比较 median/MAD、因果 running normalization 或无监督域泛化；不能用 held-out 标签调归一化。
- 当前包只完成动作边界/片段分割。把预测片段送入 RGB M3 Atomic-tail 的 node 识别属于下一阶段端到端扩展。

详细协议见 `docs\EXPERIMENT_PROTOCOL.md`，文献见 `docs\METHOD_REFERENCES.md`。

## 9. 第一版范围与后续对照

本包已经实现推荐的神经 MVP：past-only local CNN + causal boundary TCN。分析阶段提出的 IMU energy/change-point 和 EMG RMS/MAV/TKEO threshold 属于应补充的传统基线，目前没有伪装成已实现功能。建议在正式神经结果之前或同时增加独立 baseline 工具；其阈值也必须只由 validation runs 选择。之后再进行 IMU+EMG 融合，以及把传感器检测片段送入 RGB M3 Atomic-tail 的端到端 node 评价。
