# Canonical v3：四人 LOSO 动作边界与识别初步实验

日期：2026-09-11。此目录是独立实验包；不覆盖此前 RGB、IMU/EMG、M0–M6 或 Atomic-tail 的代码、标注、缓存和结果。**本包已做数据/代码检查，尚未运行正式训练，不包含新的精度结果。** `README_previous_rgb.md` 和传感器子目录的旧 README 仅供历史参考；路径、版本和运行方式以本文为准。

## 1. 本轮实验范围

主实验：连续 RGB → 冻结的原 Tier3 RGB backbone → 新训练的因果 boundary TCN → 修正后的在线状态机 → 原 M3 Atomic-tail Direct Fusion 预测 node。

辅助实验：修复时间戳后的 IMU-only、EMG-only → 因果局部编码器 + boundary TCN → 同类状态机，分别报告动作分割。**辅助实验没有 node 分类头，也没有与 RGB 融合；主实验不会读取传感器作为 RGB 模型输入。** 若要比较“传感器检测的片段 → RGB M3 识别”，还需下一阶段新增传感器片段到 RGB 帧的推理适配，本包不冒充已实现该功能。

| 项目 | 初步实验默认值 |
|---|---|
| LOSO | A、D、J、M 分别作为测试者，其他三人训练 |
| seed / scope | seed=1，all_runs；这是四折初步实验，不是三 seed 正式统计 |
| 测试 | 每折分别 test_normal、test_fault、test_all；保持旧协议定义 |
| RGB 正式输入 | 16 帧因果窗口，每帧一个 anchor，stride=1，512 维特征 |
| RGB 训练 | 40 epochs，batch=8，TCN hidden=256、5层 |
| chunk | 长256、重叠124；后续 chunk 的重叠仅作上下文，不再计 loss |
| 选择模型 | 非测试者训练 run 中划出15%验证，按验证 loss 保存 best.pth |
| RGB boundary 标签 | start/end 半径2帧，pos_weight=25 |
| 状态机 | start/end/action阈值0.55，start/end防抖2步，最短3步，merge_gap_steps=0 |
| 识别 | 已训练 M3，refresh_once 权重；只用之前预测片段特征作为 history |
| IMU | 100Hz重采样、0.5秒历史窗口、20Hz决策；40epochs |
| EMG | 500Hz重采样、0.2秒历史窗口、20Hz决策；40epochs |

保留原权重意味着本轮是“新标注训练边界 + 既有识别器迁移评估”，**不是使用 v3 从头重训动作识别**。旧 backbone 的训练曾覆盖非测试参与者的 run，因此边界验证集对边界头是留出的，但不是对整个预训练表示完全独立的验证集。测试参与者仍被 LOSO 排除。

## 2. 数据版本及修正

1. `data/annotations/`：完整复制外部 `action_recognition_timestamps_canonical_v3`。继承 v2 排除1970无效时间帧后的统一索引，叠加人工 Excel 修正；保留短 background。共有103runs、251132frames、1894actions。原v1/v2/v3未被改写。
2. `data/raw/run_sample_XXXXXX/mindrove_left.csv`、`mindrove_right.csv`：复制 `multimodal_training_repaired_2026-09-10` 的206个修复版传感器文件；继承修复的 board_ts，未重新修复或改写信号。
3. `data/sensor_time_quality.csv`、`fully_repaired_runs.txt`、`runs_with_unrepaired_rows.txt`：保留来源质量记录。90个run完全修复，13个run含回退记录。第一轮保留全部103runs以维持协议；之后应增加质量分层敏感性分析，不能称为全部时间已可靠修复。
4. `assets/v3_node_lineage.json`：v3 segment 到原动作识别行的显式对应。尤其 sample83 删除一个动作后，不能再直接把 v3 动作列表与旧 node 列表 zip，否则后续标签错位。该映射仅用于 GT 评估，不进入模型预测。
5. `PACKAGE_PROVENANCE.json` 和 `data/annotations/` 中的审计/生成记录保留来源。`data/manifest.jsonl` 中部分原始绝对路径是追溯信息；运行相机路径由 RGB_DATASET_ROOT + 相对 camera_dirs 确定。

### 本包代码变更依据

| 修正 | 实现位置 |
|---|---|
| 继承 RGB overlap loss mask | `boundary_experiment/data.py`、`tests/test_core.py` |
| 继承 RGB pending片段一次性输出修正 | `boundary_experiment/online.py` |
| 相邻动作不因state连续为1而合并GT | `boundary_experiment/metrics.py`、`engine.py`：使用segment_no；新增test_v3_segments |
| v3动作到原35node对应 | `tools/evaluate_end_to_end.py`、`assets/v3_node_lineage.json` |
| 图像根目录与修复传感器目录分开 | `boundary_experiment/annotations.py`、`config_windows.bat` |
| 传感器chunk上下文排除loss | `sensor_baselines/sensor_boundary/data.py`、`tests/test_context_loss.py` |
| 传感器状态机同步一次性输出逻辑 | `sensor_baselines/sensor_boundary/online.py`，保留其配置键接口 |

全部旧缓存不复用：即使RGB像素没变，缓存还包含标签；新包按fold/seed/scope/stride独立保存。重新运行 extract 默认跳过已有pt，若后来再改标注/权重，请换缓存根目录或明确重新提取，不能只改配置继续使用旧pt。

## 3. 迁移到另一台 Windows 电脑

必须复制两部分：

1. **完整本实验包文件夹**。已包含修复传感器、v3标注、清单、node对应、Task Graph、协议、原模型所需代码，以及四折seed1/all_runs的4个backbone + 4个M3权重。无需再复制完整旧RGB项目、时间修复工程或XDF文件才能运行这轮实验。
2. 原始结构化图像：源 `D:\Junxi_data\MULTISENSOR_DATA_COLLECTION_Stage2_structured_data\Action_Segmentation_Dataset\raw` 下103个run各自的 **001484412812相机目录及全部图像**。目录结构保持不变：`<RGB_DATASET_ROOT>\raw\run_sample_XXXXXX\001484412812\...jpg`。不需要其他相机；不需要将原始未修复mindrove覆盖到本包data/raw。也可以复制整个原数据集，但仍让DATASET_ROOT指向本包data，RGB_DATASET_ROOT指向复制后的原图像数据集根。

迁移不必复制本包的 `cache/`、`outputs*`、`validation*`、`__pycache__`；迁移前若已有新包生成的有效缓存，可一起复制以节省时间，但不要混入旧版本缓存。不要依赖Git自动携带大型CSV、JPG或PTH，复制后务必validate。

推荐新电脑布局：

```text
E:\experiments\realtime_boundary_v3_pilot_2026-09-11\
    data\annotations\                 v3，已打包
    data\raw\run_sample_XXXXXX\       修复后的左右传感器，已打包
    atomic_dependency\                 四折权重与代码，已打包
E:\datasets\Action_Segmentation_Dataset\raw\run_sample_XXXXXX\001484412812\
```

在Python环境安装 `requirements.txt`；运行传感器再安装 `sensor_baselines/requirements.txt`。GPU版torch/torchvision必须与新电脑驱动及彼此版本匹配。不要把旧电脑的整个conda环境文件夹直接当成可移植环境。

修改根目录 `config_windows.bat` 中：

```bat
if not defined PYTHON_BIN set "PYTHON_BIN=C:\Miniconda3\envs\boundary\python.exe"
if not defined RGB_DATASET_ROOT set "RGB_DATASET_ROOT=E:\datasets\Action_Segmentation_Dataset"
```

其他DATASET_ROOT、ANNOTATION_ROOT、ATOMIC_PROJECT_ROOT默认相对本包，不必改。传感器实验还需在 `sensor_baselines/config_windows.bat` 修改PYTHON_BIN，或者在新CMD中先 `set "PYTHON_BIN=..."` 让两套脚本共用。避免先call根config再在同一个CMD直接call传感器config：它们同名环境变量会继承；请使用下面的启动脚本，并在未手动加载config的新CMD中运行。

默认RGB提取worker=4、batch=16、解码batch=64。先保持默认跑通，之后再增加FORMAL_EXTRACT_BATCH_SIZE；NUM_WORKERS是训练加载器，不是EXTRACT_NUM_WORKERS。显存占用低不等于GPU算力没用满，要看每run耗时和吞吐。

## 4. 从零开始运行（CMD，不是 PowerShell）

进入新包根目录，所有命令均在已打开CMD执行，不建议双击bat，以免错误窗口关闭。

```bat
cd /d E:\experiments\realtime_boundary_v3_pilot_2026-09-11
cmd /c config_windows.bat show
scripts\run_pilot.bat validate
```

validate检查四人图像、v3标签、node对应、左右传感器文件、权重路径。报告见 `validation_canonical_v3/pilot_preflight.json`，必须status=ok。预检检查了全体测试者，四折合计覆盖所有103runs，不训练模型。

### 先做 smoke

```bat
scripts\run_smoke_stride4.bat validate
scripts\run_smoke_stride4.bat prepare
scripts\run_smoke_stride4.bat extract
scripts\run_smoke_stride4.bat train
scripts\run_smoke_stride4.bat evaluate
scripts\run_smoke_stride4.bat online
scripts\run_smoke_stride4.bat end_to_end
```

smoke默认仅A/seed1/all_runs，但extract仍覆盖该条件训练+测试的全部run；stride4、2epochs、小TCN。它不是只抽两个run。`all`自动执行到online，不含end_to_end，后者另跑。检查extract进度、每run完成耗时、缓存数量、train loss是否有限、evaluation metrics及online预测文件。stride4和stride1分别缓存，不要拿smoke精度当正式结果。若只测速，可看到数个run完成后用Ctrl+C停止，重启extract跳过已完成pt。

### 正式四折初步实验

```bat
scripts\run_pilot.bat prepare
scripts\run_pilot.bat extract
scripts\run_pilot.bat train
scripts\run_pilot.bat evaluate
scripts\run_pilot.bat end_to_end
scripts\run_pilot.bat summary
```

或 `scripts\run_pilot.bat all`（包括validate至summary）。一折完成后顺序处理其他折，不是四GPU并行。不要重复执行train期待自动断点续训：训练脚本会重新开始并更新本包同条件输出。

| 阶段 | 做什么 / 不做什么 |
|---|---|
| validate | 文件、标注、协议映射检查；不提特征、不训练 |
| prepare | 原动作识别LOSO协议转成连续run列表，保留normal/fault归属 |
| extract | 因果16帧RGB窗口生成512维特征和v3训练标签；冻结backbone，各折独立缓存 |
| train | 仅训练边界TCN，保存last/best、训练日志和配置；不重训M3 |
| evaluate | 对三种测试split逐run因果预测，状态机输出片段，与v3 GT计算分割指标 |
| online | smoke指定单run的流式逻辑回放，已检测片段闭合后M3识别；不是摄像头实时采集工具 |
| end_to_end | 全测试run：预测片段→RGB片段特征→M3及预测history→node，与GT一对一IoU匹配评估 |
| summary | 汇总四折×三个split的12行结果；缺失条件明确missing，不伪造分数 |

RGB正式输出：`outputs_canonical_v3/{F}_as_test/all_runs/seed_1/causal_boundary_tcn_canonical_v3/`，其中`evaluation/{split}/metrics.json`、`end_to_end/{split}/metrics.json`和`predicted_nodes.jsonl`。总表`outputs_canonical_v3/pilot_summary.csv`。保留每折结果，不仅看平均。

### 修复传感器辅助实验

在新的CMD窗口进入本包，不手动call根config。依次运行：

```bat
scripts\run_sensor_pilot.bat both validate
scripts\run_sensor_pilot.bat both prepare
scripts\run_sensor_pilot.bat both extract
scripts\run_sensor_pilot.bat both train
scripts\run_sensor_pilot.bat both evaluate
scripts\run_sensor_pilot.bat both summarize
```

可用 `both all`，也可把both改成imu或emg。输出在 `sensor_baselines/outputs_canonical_v3/{imu|emg}/...`，IMU和EMG互不覆盖。内部模型目录沿用`causal_sensor_boundary_v1`仅表示架构名称，不表示使用v1标签。第一轮默认固定阈值，不运行calibrate；若后来做验证集校准，单独保存固定阈值结果再执行calibrate/evaluate，evaluate会读取已有校准文件，不能混淆两套结果。

## 5. 指标与结论边界

- Boundary P/R/F1：RGB容差3/5/10帧；传感器按其毫秒配置。数值不能不经时间单位换算直接比较。
- Segmental F1@10/25/50：与GT片段一对一IoU匹配。RGB已使用segment_no，避免二值state丢失相邻动作边界。
- Frame Accuracy/F1、当前Edit：仍是background/action二值指标；**不是35node帧级准确率或多类别Edit**。
- conditional_node_accuracy = node正确且检测匹配数 / 检测匹配数；end_to_end_node_accuracy = node正确且检测匹配数 / 全部GT动作数。后者才体现漏检损失，同时须报告检测precision以反映多检。
- 推理没有GT边界或GT history；GT只用于训练标签/评估。M3在检测片段关闭后才识别，并非每帧都有最终node；Task Graph仅审计预测合法性，不hard-mask。
- 片段时间定位误差与片段发出延迟不同；流末flush也不同于正常在线闭合。当前计时是离线回放/模块计算，不能直接声称包含摄像头、磁盘、GPU同步及完整M3的真实端到端延迟。
- 传感器board_ts修复是离线估计；继承信号已被上游预处理。本包新增滤波/窗口按因果方式执行，但不能据此证明整个采集到推理链路严格在线无未来信息。需另做原始信号/实时同步验证。

本轮用来判断更新后四人表现、过分割是否改善和识别漏检损失。若扩展三seed或normal_only，需同时修改JSON与bat的实验网格，并补齐对应backbone/M3权重（本包只携带seed1/all_runs）；summary脚本当前也固定此pilot网格。单seed不能做稳定显著优越性的结论。

## 6. 读代码顺序与检查

先看`data/annotations/README*`、一个run逐帧/segment标注、`assets/v3_node_lineage.json`；再看`config_windows.bat`和`configs/base.json`；沿`tools/prepare_protocols.py → extract_boundary_features.py → train_boundary.py → evaluate_boundary.py → evaluate_end_to_end.py`阅读。核心实现分别在`boundary_experiment/annotations.py`、`features.py`、`data.py`、`models.py`、`engine.py`、`online.py`、`m3_adapter.py`、`metrics.py`。

可重复检查（先激活配置的PyTorch环境，或用该python绝对路径替代python）：

```bat
python -m unittest discover -s tests -v
cd sensor_baselines
python -m unittest discover -s tests -v
cd ..
```

传感器测试也可在sensor_baselines目录运行`python -m unittest discover -s tests -v`。模型兼容检查在加载根config后运行`"%PYTHON_BIN%" tools/check_pilot_weights.py`，只加载八个已有权重并测试M3的空/非空预测历史，不训练。
