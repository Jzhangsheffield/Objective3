# RGB Task-Graph History：A/D/M跨人验证实验包

本代码包用于验证在J-as-test先导实验中观察到的提升，是否能够在另外三位参与者
A、D、M上重复出现。代码包完全独立，不修改原J实验包，也不依赖J实验包的输出。

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
