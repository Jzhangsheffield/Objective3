# RGB Task-Graph History：A/D/M跨人验证实验包

本代码包用于验证在J-as-test先导实验中观察到的提升，是否能够在另外三位参与者
A、D、M上重复出现。代码包完全独立，不修改原J实验包，也不依赖J实验包的输出。

> **增量扩展（2026-07-22）**：在不覆盖已完成的backbone、features和M0–M6结果的
> 前提下，本包新增`E2E-Tier3-Scratch`复测、`E2E-Node-Scratch`和
> `E2E-Node-From-Tier3`。请使用第12节的`run_additional_e2e_*`入口；不要为了新增
> 对照重新执行已经完成的`run_one_fold.bat`。

包位置：

```text
D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\graph_history_rgb_cross_person_ADM_2026-07-22
```

## 1. 与原J实验包的区别和联系

| 项目 | 原J先导实验包 | 本A/D/M跨人实验包 |
|---|---|---|
| 目录 | `graph_history_rgb_experiments_2026-07-20` | `graph_history_rgb_cross_person_ADM_2026-07-22` |
| 主要目的 | 在J-as-test上建立并验证M0–M6 | 检查提升能否跨人重复 |
| 测试对象 | J | A、D、M |
| 默认backbone | 已有J Tier-3 `last.pth` | 每一折从scratch重新训练 |
| backbone训练数据 | 原checkpoint的既有协议 | 该折其余三人的normal-only runs |
| validation | 不使用 | 不使用 |
| checkpoint选择 | 最后epoch | 最后epoch |
| Task Graph | 35-node固定图 | 使用相同固定图 |
| 评估 | 35-node和31类Tier-3 | 相同，并增加跨人汇总和相对M0提升 |

两套实验共享同一研究设计、M0–M6定义、单相机、Task Graph快照和评价方法。本包的
Python代码由原包独立复制后修改，因此后续修改本包不会影响原J结果。

当前J和A/D/M的绝对准确率不应直接横向比较，因为J默认使用已有backbone，而A/D/M
使用严格normal-only scratch backbone。最可靠的跨人指标是每一折中各模型相对本折
M0的提升：

```text
delta(model, participant) = metric(model) - metric(M0)
```

如需形成严格一致的四折LOSO结果，本包也支持设置`TEST_PARTICIPANT=J`重新训练J折，
但默认批量流程只运行A、D、M。

## 2. 固定实验协议

- 模态：RGB。
- 相机：`001484412812`。
- 外层测试对象：A、D、M分别留一人测试。
- 每折backbone：ResNet3D-18，从scratch训练31类Tier-3。
- 每个clip均匀采样16帧，输入为224×224。
- backbone分类头前抽取512维特征。
- M0–M6训练目标：35个Task Graph node。
- 评估：35-node以及聚合后的31类Tier-3。
- 不使用validation、early stopping或best checkpoint。
- backbone和history model均使用最后epoch的`last.pth`。
- 主实验只使用normal runs训练。
- all-run training作为辅助实验，使用同一个normal-only backbone以避免更换视觉表征造成混杂。
- 历史只来自同一`participant+run`中`annotation_row_index`更小的clips。
- 不使用未来clip，也不使用跨run memory bank。

三折定义：

| Fold | 训练参与者 | 测试参与者 |
|---|---|---|
| A-as-test | D、J、M | A |
| D-as-test | A、J、M | D |
| M-as-test | A、D、J | M |

normal-only协议会利用四个`fault_run_test_manifest.jsonl`识别故障run，并从训练折中完整
移除这些run。测试集继续划分为`test_normal`、`test_fault`和`test_all`。

## 3. M0–M6定义

| 模型 | 输入与作用 |
|---|---|
| M0 | 仅当前clip的512维RGB特征，35-node基线 |
| M1 | 当前特征和实际历史；无位置编码的single-query attention |
| M2 | M1加实际历史距离位置编码 |
| M3 | 历史按Task Graph允许的拓扑顺序确定性重排，然后使用位置编码 |
| M4 | 35个candidate query读取历史，不使用graph relation bias |
| M5 | M4加真实历史node产生的relation bias；Oracle上限 |
| M6 | M4加冻结M0预测历史node概率产生的soft relation bias；可部署版本 |

M4与M6的差值用于判断graph relation bias是否提供额外贡献。M5读取历史真实node标签，
只作为Oracle参考，不能当作实际部署结果。

关系矩阵方向固定为：

```text
row    = 当前候选node
column = 历史node
```

I/M/O/X/S矩阵本身固定不训练；M5和M6为每个attention head学习五种relation type对应的
bias。X使用有限可学习bias，不是硬mask。

## 4. 安装和路径配置

建议使用Python 3.10或更新版本，并安装匹配显卡驱动的PyTorch。其余依赖为：

```bat
pip install -r requirements.txt
```

### Windows换电脑

编辑`bat/config_windows.bat`：

```bat
set "DATASET_ROOT=C:\你的路径\Stage_2_Mapstyle_Dataset"
set "PYTHON_BIN=C:\你的conda环境\python.exe"
```

不需要设置已有backbone路径，因为A、D、M每折都会自己生成：

```text
seed_N/backbone/normal_only/last.pth
```

常用可覆盖变量：

```bat
set TEST_PARTICIPANT=A
set SEED=1
set NUM_WORKERS=4
set BACKBONE_EPOCHS=100
set HISTORY_EPOCHS=50
set RUN_AUXILIARY=0
```

本包的Python入口会自动把包根目录加入模块搜索路径。因此在包根目录下直接运行
`python tools\*.py`时，不再需要手动设置`PYTHONPATH`；BAT仍会设置它作为额外保险。

### HPC路径和环境

编辑`slurm/config_hpc.sh`中的默认值，重点检查：

```bash
DATASET_ROOT=/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Stage_2_Mapstyle_Dataset
ANACONDA_MODULE=Anaconda3/2022.05
CUDNN_MODULE=cuDNN/8.9.2.26-CUDA-12.1.1
CONDA_ENV_NAME=pytorch
```

默认继续使用原实验资源配置：GPU一张、8 CPU、80 GB backbone/feature内存，Slurm
partition为`gpu,gpu-h100,gpu-h100-nvl`，qos为`gpu`。

## 5. Windows运行方法

### 5.1 先做快速检查

```bat
set TEST_PARTICIPANT=A
call bat\00_validate_setup.bat
call bat\01_prepare_protocols.bat
```

`protocol_report.json`会记录每个split的samples、runs、35-node支持度、31类Tier-3支持度
以及缺失类别。正式训练前应检查normal-only train是否存在缺失node。

### 5.2 单折完整主实验

例如A-as-test：

```bat
set TEST_PARTICIPANT=A
set SEED=1
call bat\run_one_fold.bat
```

该脚本依次执行：

```text
环境检查
→ 生成协议
→ 训练normal-only backbone
→ 使用backbone last.pth抽取所有需要的特征
→ 训练normal-only M0
→ 训练normal-only M1–M6
→ 汇总该折结果
```

分别运行D和M：

```bat
set TEST_PARTICIPANT=D
call bat\run_one_fold.bat

set TEST_PARTICIPANT=M
call bat\run_one_fold.bat
```

### 5.3 连续运行A/D/M

```bat
call bat\run_all_ADM.bat
```

三个fold的backbone训练量较大，单机上会按A、D、M顺序执行。HPC版本会并行提交三条
相互独立的依赖链。

### 5.4 all-run辅助实验

完成某一折主流程和feature extraction后：

```bat
set TEST_PARTICIPANT=A
call bat\05_train_aux_all_runs_m0_m6.bat
call bat\06_summarize_results.bat
```

也可以在完整单折运行前设置：

```bat
set RUN_AUXILIARY=1
call bat\run_one_fold.bat
```

### 5.5 跨人汇总

完成三折后：

```bat
call bat\07_summarize_cross_person.bat
```

生成：

```text
outputs/cross_person_summary/
├── cross_person_metrics.csv
├── cross_person_deltas_vs_m0.csv
└── cross_person_aggregate.csv
```

`cross_person_aggregate.csv`先在每位参与者内部平均重复seed，再在A/D/M之间计算均值和
样本标准差，避免把多个seed错误地当成更多独立参与者。

## 6. HPC/Slurm运行方法

### 6.1 提交单折

```bash
cd /path/to/graph_history_rgb_cross_person_ADM_2026-07-22
bash slurm/submit_one_fold.sh A
bash slurm/submit_one_fold.sh D
bash slurm/submit_one_fold.sh M
```

每次提交自动建立依赖：

```text
prepare
→ backbone
→ features
→ M0
→ M1–M6 Slurm array
→ fold summary
```

### 6.2 同时提交三折

```bash
bash slurm/submit_ADM.sh
```

三折独立运行；只有跨人汇总任务会等待三折全部成功。

### 6.3 提交all-run辅助实验

主折feature cache完成后：

```bash
bash slurm/submit_aux_one_fold.sh A
bash slurm/submit_aux_one_fold.sh D
bash slurm/submit_aux_one_fold.sh M
```

### 6.4 覆盖seed或epoch

```bash
SEED=2 BACKBONE_EPOCHS=100 HISTORY_EPOCHS=50 bash slurm/submit_one_fold.sh A
```

不同seed写入不同`seed_N`目录，不会覆盖已有结果。

## 7. 各脚本用途

### Python入口

- `tools/validate_setup.py`：检查指定fold、相机字段、Task Graph和可选checkpoint。
- `tools/prepare_protocols.py`：生成normal-only/all-run协议和类别支持报告。
- `tools/train_backbone.py`：从scratch训练31类Tier-3 ResNet3D-18；无validation。
- `tools/extract_features.py`：使用该折backbone `last.pth`抽取512维特征。
- `tools/train_history_model.py`：统一训练M0–M6并测试normal/fault/all。
- `tools/summarize_results.py`：汇总一个fold、一个seed下的结果。
- `tools/summarize_cross_person.py`：汇总A/D/M并计算相对M0提升。
- `tools/smoke_test_models.py`：用合成tensor检查M1–M6前向、反向和空历史。
- `tools/evaluate_e2e_tier3.py`：只加载已有Tier-3 `last.pth`重新评估，不训练权重。
- `tools/train_e2e_node.py`：训练scratch或Tier-3初始化的端到端35-node模型。
- `tools/summarize_all_models.py`：统一汇总三个E2E基线和M0–M6。

### Windows入口

- `00_validate_setup.bat`：快速环境检查。
- `01_prepare_protocols.bat`：生成当前fold协议。
- `02_train_backbone_normal_only.bat`：训练当前fold backbone。
- `03_extract_features_retrained_last.bat`：从当前fold `last.pth`抽取特征。
- `04_train_main_m0_m6.bat`：normal-only主实验。
- `05_train_aux_all_runs_m0_m6.bat`：all-run辅助实验。
- `06_summarize_results.bat`：当前fold结果汇总。
- `07_summarize_cross_person.bat`：A/D/M跨人汇总。
- `run_one_fold.bat`：完整运行一个fold。
- `run_all_ADM.bat`：顺序运行三折。
- `08_evaluate_e2e_tier3_existing.bat`：复测已有Tier-3 checkpoint。
- `09_train_e2e_node_scratch.bat`：新增scratch 35-node端到端基线。
- `10_train_e2e_node_from_tier3.bat`：新增Tier-3迁移35-node端到端基线。
- `11_summarize_all_models_fold.bat`：单折十模型新汇总。
- `12_summarize_all_models_cross_person.bat`：A/D/M十模型跨人新汇总。
- `run_additional_e2e_one_fold.bat`：只执行单折增量E2E实验。
- `run_additional_e2e_ADM.bat`：只执行A/D/M增量E2E实验。

## 8. 输出结构

```text
outputs/
├── A_as_test/cam_001484412812/
│   ├── protocols/
│   └── seed_1/
│       ├── backbone/normal_only/last.pth
│       ├── features/retrained_normal_only/
│       └── history_models/retrained_normal_only/
│           ├── normal_only/m0 ... m6/
│           ├── all_runs/m0 ... m6/       可选
│           └── experiment_summary.csv
├── D_as_test/...
├── M_as_test/...
└── cross_person_summary/
```

每个M0–M6模型目录包含：

```text
last.pth
train_log.json
learned_parameters.json
test_results/
  test_normal_metrics.json
  test_fault_metrics.json
  test_all_metrics.json
  *_predictions.csv
  *_probabilities.pt
```

## 9. 推荐分析顺序

1. 比较每一折M1/M2与M0，判断同run历史是否稳定有益。
2. 比较M3与M0及M2，判断不依赖真实精确顺序的graph-valid历史是否仍有效。
3. 比较M6与M4，判断relation bias是否提供额外贡献。
4. 比较M6与M5，判断可部署soft graph版本距离Oracle上限还有多远。
5. 分别检查normal、fault、all和Stage 1/2/3。
6. 最后查看`cross_person_deltas_vs_m0.csv`，不要只比较不同人的绝对准确率。

同一个run内的clips并非统计独立。正式显著性分析建议基于run做paired bootstrap或按run
计算模型差值，不建议直接对所有clip做普通独立样本t-test。A/D/M只有三位测试对象，
因此跨人均值和标准差应主要作为一致性证据，而不是夸大为大样本人群结论。

## 10. 公平性与实验边界

- A、D、M的模型结构和超参数必须沿用J先导实验已经确定的配置。
- 不应根据A/D/M测试结果逐人调整epoch、学习率或graph bias结构。
- 每一折的M0–M6共享同一backbone、同一feature cache和同一seed。
- M6历史node概率来自该折自己的冻结M0，不能使用其他fold或真实历史标签。
- 当前任务仍是预切分clip classification，不是连续视频检测或sequence decoding。

## 11. 已验证的数据规模（2026-07-22）

| Fold | normal-only train | test normal | test fault | test all |
|---|---:|---:|---:|---:|
| A-as-test | 1,147 clips / 61 runs | 294 / 15 | 137 / 9 | 431 / 24 |
| D-as-test | 1,041 clips / 55 runs | 400 / 21 | 62 / 4 | 462 / 25 |
| M-as-test | 1,081 clips / 57 runs | 360 / 19 | 87 / 5 | 447 / 24 |

三折的normal-only训练集均覆盖35/35 nodes和31/31 Tier-3类别，因此可以训练完整35-node
分类头。A的fault测试集覆盖全部类别；D的fault测试集只有30/35 nodes和26/31 Tier-3；
M的fault测试集有34/35 nodes和30/31 Tier-3。这不是graph错误，而是对应fault runs中没有
发生这些动作。指标JSON会记录`present_class_count`；macro-F1和balanced accuracy只对该
split中真实出现的类别取平均，因此跨人比较fault macro指标时必须同时查看类别覆盖度和
accuracy。

## 12. 已完成M0–M6后的增量E2E对照实验

这一部分专门用于已经完成原实验的情况。增量入口不会调用以下脚本：

```text
02_train_backbone_normal_only.bat
03_extract_features_retrained_last.bat
04_train_main_m0_m6.bat
05_train_aux_all_runs_m0_m6.bat
```

因此不会重新训练或覆盖现有backbone、feature cache和M0–M6。

### 12.1 新增的三个对照

| 统计名称 | 实际操作 | 是否重新训练 |
|---|---|---|
| `e2e_tier3_scratch` | 加载本折已有`backbone/normal_only/last.pth`重新测试31类Tier-3 | 否 |
| `e2e_node_scratch` | RGB ResNet3D-18从scratch端到端训练35-node | 是 |
| `e2e_node_from_tier3` | 加载已有Tier-3 backbone，替换为35-node fc并全网络微调 | 是 |

`e2e_tier3_scratch`只产生新的evaluation文件和checkpoint引用，不复制、不修改已有权重。
两个node模型直接输出35-node softmax，并通过固定node-to-Tier-3映射求和得到31类Tier-3
概率，因此同时报告node与Tier-3性能。直接Tier-3模型没有node输出，其node统计为空。

### 12.2 新输出目录与防覆盖机制

新增内容只写入：

```text
seed_N/
├── e2e_baselines/
│   └── normal_only/
│       ├── e2e_tier3_scratch/       只复测已有last.pth
│       ├── e2e_node_scratch/         新模型
│       └── e2e_node_from_tier3/      新模型
└── unified_summary_with_e2e/         新的十模型汇总
```

跨人新汇总写入：

```text
outputs/cross_person_summary_with_e2e/
```

原来的这些路径保持不变：

```text
backbone/
features/
history_models/
cross_person_summary/
```

每个新增实验成功完成normal、fault、all测试后写入`completed.json`。再次运行增量入口时，
如果发现该标记就安全跳过；如果目录非空但没有完成标记，脚本会停止并提示人工检查，
不会静默覆盖部分结果。底层Python工具只有显式传入`--overwrite`才允许重写专属新增目录，
BAT和Slurm入口默认永远不传这个选项。

### 12.3 Windows运行

确认`TEST_PARTICIPANT`和`SEED`与已经完成的原实验一致，例如：

```bat
set TEST_PARTICIPANT=A
set SEED=1
call bat\run_additional_e2e_one_fold.bat
```

该脚本只执行：

```text
复测已有Tier-3 last.pth
→ 训练E2E-Node-Scratch
→ 训练E2E-Node-From-Tier3
→ 读取已有M0–M6并生成新的统一汇总
```

依次处理A/D/M并生成跨人新汇总：

```bat
call bat\run_additional_e2e_ADM.bat
```

如需逐步运行：

```bat
call bat\08_evaluate_e2e_tier3_existing.bat
call bat\09_train_e2e_node_scratch.bat
call bat\10_train_e2e_node_from_tier3.bat
call bat\11_summarize_all_models_fold.bat
call bat\12_summarize_all_models_cross_person.bat
```

可在`bat/config_windows.bat`覆盖：

```bat
set E2E_NODE_EPOCHS=100
set E2E_NODE_LR=0.0001
```

### 12.4 HPC/Slurm运行

单折增量提交：

```bash
bash slurm/submit_additional_e2e_one_fold.sh A
```

A/D/M三折：

```bash
bash slurm/submit_additional_e2e_ADM.sh
```

提交脚本会先在登录节点确认以下已有文件：

```text
protocols/normal_only/*.jsonl
backbone/normal_only/last.pth
history_models/retrained_normal_only/normal_only/m0 ... m6/last.pth
```

Tier-3复测、Node-Scratch和Node-From-Tier3会作为三个独立GPU任务并行运行，统一单折
汇总等待三者全部完成；跨人汇总再等待A/D/M的单折汇总完成。

### 12.5 新统一统计

每折和跨人目录都生成：

```text
all_model_metrics.csv
all_model_pairwise_deltas.csv
all_model_cross_person_aggregate.csv
all_model_delta_aggregate.csv
all_model_per_stage_metrics.csv
all_model_per_stage_cross_person_aggregate.csv
```

`all_model_metrics.csv`同时包含：

```text
e2e_tier3_scratch
e2e_node_scratch
e2e_node_from_tier3
m0, m1, m2, m3, m4, m5, m6
```

即共10个模型设置。统计包括：

- 35-node accuracy、macro-F1、balanced accuracy；
- 31类Tier-3 accuracy、macro-F1、balanced accuracy；
- normal、fault、all三个split；
- 每个指标JSON中的Stage 1、2、3结果和present class count；
- 相对于`m0`、`e2e_node_scratch`、`e2e_node_from_tier3`和
  `e2e_tier3_scratch`的可比较指标差值。

其中最重要的比较是：

```text
M6 - M0
M6 - E2E-Node-Scratch
M6 - E2E-Node-From-Tier3
Tier3(M6) - E2E-Tier3-Scratch
```

统一汇总只读取原M0–M6指标文件，不会修改已有checkpoint、predictions或probabilities。

### 12.6 增量扩展交付前验证

- 使用真实RGB clip和已有Tier-3 `last.pth`完成evaluation-only测试；checkpoint完整加载。
- `E2E-Node-Scratch`完成真实clip前向、反向、保存、重载以及node/Tier-3聚合测试。
- `E2E-Node-From-Tier3`成功加载120个backbone参数键，只跳过形状不同的
  `fc.weight`和`fc.bias`，随后完成端到端训练与评估路径测试。
- 非覆盖保护、`completed.json`安全跳过、Windows路径展开和Slurm语法均已验证。
- 统一汇总在测试fixture中识别到全部10个模型，并生成overall、pairwise delta和per-stage
  CSV。完整100-epoch新增训练尚未在本机启动。

## 13. 完整 all-runs 训练实验（新增，且不覆盖既有实验）

### 13.1 实验定义

这一实验把训练人员的正常run和故障run全部用于训练，并继续对held-out参与者分别测试：

```text
all_runs/train.jsonl
→ test_normal.jsonl
→ test_fault.jsonl
→ test_all.jsonl
```

以A-as-test为例，训练集由D/J/M的全部run组成，A的任何clip或run都不会进入训练。
D-as-test和M-as-test采用相同的leave-one-subject-out规则。

当前数据集中的all-runs训练规模为：

| Fold | all-runs train | held-out test all |
|---|---:|---:|
| A-as-test | 1,464 clips / 79 runs | 431 clips / 24 runs |
| D-as-test | 1,433 clips / 78 runs | 462 clips / 25 runs |
| M-as-test | 1,448 clips / 79 runs | 447 clips / 24 runs |

本节的“完整all-runs pipeline”不同于第5.4节已有的辅助实验：

| 条件 | Tier-3 backbone训练集 | M0–M6训练集 | 用途 |
|---|---|---|---|
| 原normal-only主实验 | normal-only | normal-only | 原始主对照 |
| 原all-run辅助实验 | normal-only | all-runs | 只考察历史模型训练集变化 |
| 新完整all-runs实验 | all-runs | all-runs | backbone、特征、历史模型和E2E全部使用all-runs |

汇总文件新增`representation_scope`字段，用于明确特征/backbone来自哪个训练协议，防止把
`normal_only → all_runs`辅助条件与`all_runs → all_runs`完整条件合并为同一实验。

### 13.2 新增的完整实验链

每个fold依次执行：

```text
安全检查或生成protocol
→ 从scratch训练31类all-runs Tier-3 backbone
→ 使用该backbone重新提取512维训练/测试特征
→ 训练all-runs M0–M6
→ 复测all-runs E2E-Tier3-Scratch
→ 训练all-runs E2E-Node-Scratch
→ 使用all-runs Tier-3初始化并训练E2E-Node-From-Tier3
→ 汇总10个模型
→ 计算all-runs minus normal-only差值
```

三类测试集均报告35-node与31类Tier-3指标；直接Tier-3模型没有35-node输出，因此对应node
字段为空。35-node模型的Tier-3结果仍通过35个node概率按映射求和得到，不是只映射argmax node。

### 13.3 独立输出目录

新实验只写入以下新路径：

```text
outputs/<P>_as_test/cam_001484412812/seed_N/
├── backbone/all_runs/
├── features/retrained_all_runs/
├── history_models/retrained_all_runs/all_runs/
│   └── m0 ... m6/
├── e2e_baselines/all_runs/
│   ├── e2e_tier3_scratch/
│   ├── e2e_node_scratch/
│   └── e2e_node_from_tier3/
├── unified_summary_all_runs/
└── training_scope_comparison/

outputs/
├── cross_person_summary_all_runs/
└── training_scope_comparison/
```

原来的以下目录不会被新入口调用或覆盖：

```text
backbone/normal_only/
features/retrained_normal_only/
history_models/retrained_normal_only/
e2e_baselines/normal_only/
unified_summary_with_e2e/
cross_person_summary_with_e2e/
```

### 13.4 防覆盖和中断保护

- backbone、feature cache、M0–M6和三个E2E实验完成后都会写`completed.json`；
- 有完成标记时再次运行会安全跳过；
- 目标目录非空但没有完成标记时会停止，要求先人工检查；
- 新BAT和Slurm入口均不传`--overwrite`；
- protocol完整存在时直接复用；只存在部分文件时停止，不会混合新旧protocol；
- `train_backbone.py`、`train_history_model.py`和`extract_features.py`现在默认拒绝覆盖已有输出。

如果某个新实验确实因中断需要重跑，先人工确认并把该实验自己的不完整目录改名备份；不要删除或
改名normal-only目录。

### 13.5 Windows运行方法

单个fold，例如A-as-test：

```bat
cd /d D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\graph_history_rgb_cross_person_ADM_2026-07-22
set TEST_PARTICIPANT=A
set SEED=1
call bat\run_all_runs_one_fold.bat
```

连续运行A/D/M：

```bat
call bat\run_all_runs_ADM.bat
```

逐步入口为：

```text
13_prepare_protocols_all_runs_safe.bat
14_train_backbone_all_runs.bat
15_extract_features_all_runs.bat
16_train_all_runs_m0_m6.bat
17_evaluate_e2e_tier3_all_runs.bat
18_train_e2e_node_scratch_all_runs.bat
19_train_e2e_node_from_tier3_all_runs.bat
20_summarize_all_runs_fold.bat
21_summarize_training_scope_comparison_fold.bat
22_summarize_all_runs_cross_person.bat
23_summarize_training_scope_comparison_cross_person.bat
```

### 13.6 HPC/Slurm运行方法

单折提交：

```bash
bash slurm/submit_all_runs_one_fold.sh A
```

A/D/M三折提交：

```bash
bash slurm/submit_all_runs_ADM.sh
```

Slurm依赖关系为：

```text
protocol
├── all-runs backbone
│   ├── feature extraction → M0 → M1–M6 array
│   ├── E2E-Tier3 evaluation
│   └── E2E-Node-From-Tier3
└── E2E-Node-Scratch

四个模型分支完成
→ all-runs fold summary
→ normal-only vs all-runs fold comparison
```

三折完成后再生成all-runs跨人汇总和训练协议对比汇总。

### 13.7 换电脑或HPC时需要修改的路径

Windows默认只需检查`bat/config_windows.bat`中的：

```bat
set "DATASET_ROOT=C:\Junxi_data_for_training_speedup\Stage_2_Mapstyle_Dataset"
set "PYTHON_BIN=C:\Users\digit\anaconda3\envs\Pytorch\python.exe"
```

当前电脑的默认配置已经指向上述Pytorch环境。换电脑后如果用户名、Anaconda安装位置或环境名不同，
必须修改`PYTHON_BIN`，也可以先激活正确环境后把它设为`python`。

`PACKAGE_ROOT`默认由脚本所在位置自动推导，移动整个实验包后通常不用修改。若希望把新结果写到
另一块磁盘，可在运行前设置：

```bat
set OUTPUTS_ROOT=E:\my_experiment_outputs
```

HPC主要检查`slurm/config_hpc.sh`中的`DATASET_ROOT`、环境module、conda环境，以及各Slurm文件
顶部的日志路径。`PACKAGE_ROOT`也会由提交脚本自动推导。

### 13.8 新增统计文件

统一汇总工具继续生成原来的6类统计，同时增加：

```text
all_model_training_scope_deltas.csv
all_model_training_scope_delta_aggregate.csv
```

这两个文件在`training_scope_comparison/`目录中包含实际对比结果；只筛选all-runs的汇总目录中
由于没有normal-only配对行，相应文件可能为空。

其中差值固定定义为：

```text
完整all-runs pipeline指标 - 完整normal-only pipeline指标
```

正值代表加入训练故障run后该指标提高，负值代表下降。重点应分别查看：

- `test_normal`：加入故障训练数据是否损害正常流程；
- `test_fault`：对异常流程是否真正改善；
- `test_all`：总体部署分布上的变化；
- A/D/M逐人结果和跨人均值，而不只看合并accuracy；
- fault split的present class count，因为不同人的故障run并不覆盖相同类别。

故障run中的漏做、多做、重复或不符合标准graph顺序均保留为真实训练信息，不会为了匹配标准
task graph而清洗或重排。graph在这里作为结构关系输入，而不是故障数据过滤规则。

## 14. 严格四折、三 seed 补充实验（2026-07-24）

### 14.1 本次补充的目的

本扩展用于补齐报告中两项最重要的公平性缺口：

1. A/D/M 的完整 all-runs 已有 seed 1、2、42，但严格 normal-only 原来只有 seed 1。本次只补跑
   A/D/M 的 normal-only seed 2 和 seed 42。
2. J 原先来自先导实验包，backbone 来源与 A/D/M 不完全一致。本次在本跨人实验包中对 J 重新执行
   normal-only scratch 与 all-runs scratch，并使用 seed 1、2、42。

补齐后的严格网格为：

```text
participant: A, D, J, M
seed:        1, 2, 42
scope:       normal_only, all_runs
model:       M0-M6, E2E-Tier3-Scratch,
             E2E-Node-Scratch, E2E-Node-From-Tier3
split:       test_normal, test_fault, test_all
```

所有 backbone 都从 scratch 训练；不设置 validation，不使用 held-out participant 选择 checkpoint，
并统一使用最后一个 epoch 的 `last.pth`。因此可以严格按以下键配对：

```text
participant + seed + model + split
all-runs metric - normal-only metric
```

本扩展不会读取或覆盖旧 J 先导实验包中的权重。严格 J 结果写入当前包的 `outputs/J_as_test/`，
从而与 A/D/M 使用相同代码、目录结构和训练策略。

### 14.2 Windows：推荐的一键运行方式

先进入实验包：

```bat
cd /d D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\graph_history_rgb_cross_person_ADM_2026-07-22
```

一次完成全部建议实验：

```bat
call bat\run_recommended_strict_experiments.bat
```

该入口会先确认 A/D/M 的现有 all-runs seed 1、2、42 已经完成，然后依次：

```text
补跑 A/D/M normal-only seed 2、42
→ 训练 J normal-only 与 all-runs seed 1、2、42
→ 生成 A/D/J/M 四折、三 seed 的 normal-only 汇总
→ 生成 A/D/J/M 四折、三 seed 的 all-runs 汇总
→ 生成严格配对的 all-runs - normal-only 汇总
```

如果只运行其中一部分：

```bat
REM 只补 A/D/M normal-only seed 2 和 42
call bat\run_normal_only_multiseed_ADM.bat

REM 只运行 J 的一个 seed
set SEED=2
call bat\run_strict_J_one_seed.bat

REM 运行 J 的 seed 1、2、42
call bat\run_strict_J_three_seeds.bat

REM 全部训练完成后重新生成四折汇总
call bat\run_fourfold_ADJM_summaries.bat
```

`run_normal_only_complete_one_fold.bat` 和 `run_strict_J_one_seed.bat` 也可以作为断点续跑入口。
存在 `completed.json` 的步骤会安全跳过；非空但没有完成标记的目录会停止，避免覆盖中断结果。

### 14.3 HPC/Slurm：推荐提交方式

在 HPC 上先检查 `slurm/config_hpc.sh` 中的数据集、环境和输出配置，然后运行：

```bash
cd /mnt/parscratch/users/mes19jz/objective3/codex_and_files/graph_history_rgb_cross_person_ADM_2026-07-22
bash slurm/submit_recommended_strict_experiments.sh
```

分步提交入口如下：

```bash
# 单个 normal-only fold 和 seed
bash slurm/submit_normal_only_one_fold.sh A 2

# A/D/M 的 normal-only seed 2、42
bash slurm/submit_normal_only_multiseed_ADM.sh

# J 的单个严格 seed
bash slurm/submit_strict_J_one_seed.sh 2

# J 的严格 seed 1、2、42
bash slurm/submit_strict_J_three_seeds.sh
```

Slurm 会自动建立依赖关系。normal-only 的 backbone、feature、M0-M6 和三个 E2E 分支完成后才汇总；
J 的 all-runs 链还会等待相同 seed 的 normal-only 链完成。推荐总入口最后再提交三个四折汇总任务。
脚本输出的最后一个数字是最终 comparison job ID，可用于继续添加 `afterok` 依赖。

### 14.4 新实验的完整输出路径

根目录固定为：

```text
D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\
graph_history_rgb_cross_person_ADM_2026-07-22\outputs
```

单个 participant、seed 的 normal-only 结果：

```text
outputs\<P>_as_test\cam_001484412812\seed_<S>\
├── backbone\normal_only\last.pth
├── features\retrained_normal_only\
├── history_models\retrained_normal_only\normal_only\m0 ... m6\
├── e2e_baselines\normal_only\
│   ├── e2e_tier3_scratch\
│   ├── e2e_node_scratch\
│   └── e2e_node_from_tier3\
└── unified_summary_normal_only\
```

其中 A/D/M 本次新增的 `<S>` 为 2、42；J 新增的 `<S>` 为 1、2、42。

J 的 all-runs scratch 结果：

```text
outputs\J_as_test\cam_001484412812\seed_<S>\
├── backbone\all_runs\last.pth
├── features\retrained_all_runs\
├── history_models\retrained_all_runs\all_runs\m0 ... m6\
├── e2e_baselines\all_runs\
├── unified_summary_all_runs\
└── training_scope_comparison\
```

最终四折、三 seed 汇总路径：

```text
outputs\cross_person_summary_normal_only_ADJM_3seeds\
outputs\cross_person_summary_all_runs_ADJM_3seeds\
outputs\training_scope_comparison_ADJM_3seeds\
```

最重要的严格配对结果文件为：

```text
outputs\training_scope_comparison_ADJM_3seeds\
├── all_model_training_scope_deltas.csv
└── all_model_training_scope_delta_aggregate.csv
```

第一个文件保留 participant、seed、model、split 层面的逐项差值；第二个文件先在每个 participant
内部对三 seed 求均值，再报告四人的均值与标准差。

### 14.5 完整性保护

新增四折汇总启用 `--require-complete-grid`。在写 CSV 之前，它会检查每个请求的
participant × seed × scope 是否都具有 10 个模型和 3 个测试 split。只要缺一项，汇总会明确列出
缺失的 participant、seed、scope、model 和 split 并停止，因此不会把不完整实验误当成严格四折结果。

## 15. Direct Head Fusion补充实验（独立新增阶段，2026-07-28）

完整设计和运行说明另见：

```text
COMPLETE_EXPERIMENT_CONFIGURATION.md
```

### 15.1 实验目的与模型

原M1–M3冻结已经训练完成的M0，并预测35维logit delta：

```text
final logits = frozen M0 logits + learned history delta
```

本补充阶段不修改原M1–M3，而是新增：

| 新模型 | 历史顺序 | 位置编码 | 分类方式 |
|---|---|---|---|
| `m1_direct` | actual | 否 | 融合特征直接输入可训练35-node head |
| `m2_direct` | actual | 是 | 融合特征直接输入可训练35-node head |
| `m3_direct` | graph-valid | 是 | 融合特征直接输入可训练35-node head |

计算过程为：

```text
Tier3 backbone产生的当前/历史512维冻结特征
        -> history attention
        -> 当前特征与history context融合回512维
        -> 可训练LayerNorm + Linear(512, 35)
        -> 35-node logits
```

该阶段：

- 直接复用每个participant、seed、scope已经完成的Tier3 feature cache；
- 不重新训练RGB backbone；
- 不加载M0 checkpoint；
- 不使用logit delta；
- fusion和35-node classifier从同一个50-epoch阶段联合训练；
- 无validation、无early stopping，仍使用最后一个epoch的`last.pth`；
- normal-only和all-runs继续使用各自独立的backbone特征缓存；
- 测试仍为`test_normal`、`test_fault`和`test_all`；
- 35-node概率继续聚合为31类Tier3指标。

融合层初始化为`[I, 0]`，所以训练开始时融合特征严格等于当前clip的512维Tier3特征，之后再逐渐
学习history context。35-node head为随机初始化，与现有M0在冻结Tier3特征上训练35-node head的
设置一致。

### 15.2 与原实验的隔离

本阶段只写入：

```text
outputs\<P>_as_test\cam_001484412812\seed_<S>\
history_models\direct_head_fusion\
├── normal_only\
│   ├── m1_direct\
│   ├── m2_direct\
│   └── m3_direct\
└── all_runs\
    ├── m1_direct\
    ├── m2_direct\
    └── m3_direct\
```

不会写入或覆盖：

```text
history_models\retrained_normal_only\
history_models\retrained_all_runs\
```

每个新模型仍使用`ensure_new_output_dir`和`completed.json`保护。完成标记存在时BAT/Slurm入口会
安全跳过；目标目录非空但没有完成标记时训练脚本会停止，除非人工明确传入`--overwrite`。
标准入口均不传`--overwrite`。

### 15.3 Windows运行

先进入实验包并覆盖本机Python路径：

```bat
cd /d D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\graph_history_rgb_cross_person_ADM_2026-07-22
set PYTHON_BIN=C:\Users\digit\anaconda3\envs\Pytorch\python.exe
```

运行单个participant、seed的两个scope：

```bat
set TEST_PARTICIPANT=A
set SEED=1
call bat\run_direct_head_fusion_one_fold.bat
```

只运行normal-only或all-runs：

```bat
call bat\31_train_direct_head_fusion_normal_only.bat
call bat\32_train_direct_head_fusion_all_runs.bat
```

运行A/D/J/M四折、seed 1/2/42、两个scope，并在全部完成后汇总：

```bat
call bat\run_direct_head_fusion_ADJM.bat
```

只重新执行严格汇总：

```bat
call bat\33_summarize_direct_head_fusion_ADJM_3seeds.bat
```

### 15.4 HPC/Slurm运行

单个participant和seed：

```bash
# 同时提交normal-only和all-runs；每个scope是包含3个模型的array job
bash slurm/submit_direct_head_fusion_one_fold.sh A 1 both

# 也可以只提交一个scope
bash slurm/submit_direct_head_fusion_one_fold.sh A 1 normal_only
bash slurm/submit_direct_head_fusion_one_fold.sh A 1 all_runs
```

提交完整A/D/J/M × 3 seeds × 2 scopes实验，并自动以`afterok`依赖提交汇总：

```bash
bash slurm/submit_direct_head_fusion_ADJM.sh
```

底层训练任务为：

```text
slurm/33_train_direct_head_fusion_normal_only.slurm
slurm/34_train_direct_head_fusion_all_runs.slurm
```

每个任务使用array `1-3`，分别对应`m1_direct`、`m2_direct`和`m3_direct`。

### 15.5 严格汇总与比较

完整网格包含：

```text
4 participants × 3 seeds × 2 scopes × 3 direct models × 3 splits
= 216条direct模型结果
```

专用汇总：

```text
outputs\direct_head_fusion_summary_ADJM_3seeds\
├── direct_head_metrics.csv
├── direct_head_paired_deltas.csv
├── direct_head_aggregate.csv
└── completed.json
```

`direct_head_paired_deltas.csv`对相同participant、seed、scope和split计算：

```text
mK_direct − M0
mK_direct − 原mK delta模型
```

其中`K`为1、2或3。`direct_head_aggregate.csv`先在每个participant内部平均三个seed，再对
A/D/J/M等权汇总均值与样本标准差。汇总默认启用完整网格检查，缺少任何direct、M0或对应delta
结果都会停止并报告缺失路径。

## 16. Dynamic Epoch Graph-Valid Shuffle实验（独立新增阶段，2026-07-29）

完整配置、模型公式和输出说明见：

```text
COMPLETE_EXPERIMENT_CONFIGURATION.md
```

### 16.1 三个模型

| 模型 | Node head | History作用 | M0 |
|---|---|---|---|
| `m3_dynamic_frozen_m0_delta` | M0初始化并冻结 | 每epoch graph-valid history产生logit delta | 加载并冻结 |
| `m3_dynamic_joint_head_delta` | 随机初始化并训练 | 每epoch graph-valid history产生logit delta | **不加载** |
| `m3_dynamic_direct_fusion` | 随机初始化并训练 | 每epoch graph-valid history进行feature fusion | 不加载 |

训练history seed为`SHA256(base_seed:epoch:sample_name)`，所以每个epoch重新采样但整个seed实验仍可
复现。主测试继续使用原M3的固定seeded graph-valid顺序，确保能与静态M3/M3 Direct严格配对。

### 16.2 输出隔离

新实验只写入：

```text
outputs\<P>_as_test\cam_001484412812\seed_<S>\
history_models\dynamic_epoch_shuffle\<scope>\<model>\
```

不会写入原`retrained_*`、`direct_head_fusion`或E2E目录。每个模型保存
`shuffle_audit.json`证明每epoch重排确实生效。

### 16.3 Windows运行

```bat
REM 单折单seed，两个scope
set TEST_PARTICIPANT=A
set SEED=1
call bat\run_dynamic_epoch_shuffle_one_fold.bat

REM 完整四折三seed并汇总
call bat\run_dynamic_epoch_shuffle_ADJM.bat
```

分步入口：

```bat
call bat\34_train_dynamic_epoch_shuffle_normal_only.bat
call bat\35_train_dynamic_epoch_shuffle_all_runs.bat
call bat\36_summarize_dynamic_epoch_shuffle_ADJM_3seeds.bat
```

### 16.4 HPC/Slurm运行

```bash
# 单折单seed
bash slurm/submit_dynamic_epoch_shuffle_one_fold.sh A 1 both

# 完整A/D/J/M × seeds 1/2/42 × 两个scope
bash slurm/submit_dynamic_epoch_shuffle_ADJM.sh
```

训练array为：

```text
slurm/36_train_dynamic_epoch_shuffle_normal_only.slurm
slurm/37_train_dynamic_epoch_shuffle_all_runs.slurm
```

完整汇总任务为：

```text
slurm/38_summarize_dynamic_epoch_shuffle_ADJM_3seeds.slurm
```

### 16.5 严格汇总

完整网格包含216条指标：

```text
4 participants × 3 seeds × 2 scopes × 3 models × 3 splits
```

输出：

```text
outputs\dynamic_epoch_shuffle_summary_ADJM_3seeds\
├── dynamic_epoch_shuffle_metrics.csv
├── dynamic_epoch_shuffle_paired_deltas.csv
├── dynamic_epoch_shuffle_aggregate.csv
└── completed.json
```

## 17. Atomic-tail Graph-Valid实验（独立新增阶段，2026-07-30）

该实验不修改原`graph_valid`或Dynamic实验。若真实最新历史node构成某个未完成atomic sequence的
合法前缀，则将整个前缀连续固定在历史末尾，其余历史继续graph-valid随机重排。当前真实target不参与
tail判断。

三个模型：

```text
m3_atomic_tail_frozen_m0_delta
m3_atomic_tail_joint_head_delta
m3_atomic_tail_direct_fusion
```

刷新频率：

```text
1     每epoch重排
10    每10 epochs重排
once  整个训练只生成一次
```

Windows单折：

```bat
set TEST_PARTICIPANT=A
set SEED=1
set ATOMIC_TAIL_REFRESH_POLICIES=1 10 once
call bat\run_atomic_tail_one_fold.bat
```

Windows完整实验：

```bat
call bat\run_atomic_tail_ADJM.bat
```

HPC：

```bash
bash slurm/submit_atomic_tail_one_fold.sh A 1 both
bash slurm/submit_atomic_tail_ADJM.sh
```

输出仅写入：

```text
history_models\atomic_tail_graph_valid\<scope>\<refresh_policy>\<model>\
```

完整配置、回退规则、单模型命令和汇总说明见
`COMPLETE_EXPERIMENT_CONFIGURATION.md`第18节。
