# RGB Task-Graph History Experiments

这是一个独立实验代码包，用于 thermal-crimp 数据集上的 J-as-test、单相机
`001484412812` 的 M0–M6 顺序动作分类实验。它不会导入或修改旧的
`codes_of_initial_exp_with_obj2_models`，也不会修改 `obj2_codes`。

> **实验包定位（2026-07-22更新）**：本包是J-as-test先导实验包，默认复用已有J
> Tier-3 `last.pth`，用于建立M0–M6并验证方法可行性。A、D、M三位参与者的独立
> scratch-backbone跨人确认实验位于
> `D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\graph_history_rgb_cross_person_ADM_2026-07-22`。
> 两个包使用相同35-node Task Graph、历史矩阵、模型定义和评价方式，但输出互不依赖、
> 不会相互覆盖。由于backbone来源不同，当前J与A/D/M的绝对分数不宜直接比较；应优先
> 比较每一折中M1–M6相对本折M0的提升。严格四折LOSO可在新包中额外重训J折。

## 1. 固定实验协议

- 模态：RGB；相机：`001484412812`；外层测试：J-as-test。
- 不使用 validation，不执行 early stopping，不选择 best epoch。
- 训练过程中不读取测试集；只保存并测试最后 epoch 的 `last.pth`。
- 主实验只使用正常 run 训练；all-run training 是辅助实验。
- Backbone 是兼容 `obj2_codes` checkpoint 的 ResNet3D-18。
- 每个 clip 均匀采样16帧，输入224×224，提取分类头前512维feature。
- 训练目标是35个task-graph node；评估时聚合为31类Tier-3概率。
- 历史仅来自同一 `participant+run` 中 `annotation_row_index` 更小的clips。
- 不使用未来clip，不使用跨run memory bank。

## 2. 模型定义

| 模型 | 定义 |
|---|---|
| M0 | 当前512维RGB feature，训练35-node分类头 |
| M1 | 当前feature + 实际历史，无位置编码，single-query attention |
| M2 | M1 + 实际时间距离位置编码 |
| M3 | 与M2相同，但历史按task graph合法拓扑顺序确定性重排 |
| M4 | 35个candidate query并行查询实际历史，不使用graph bias |
| M5 | M4 + 真实历史node relation bias，Oracle上限实验 |
| M6 | M4 + 冻结M0预测的历史node概率产生soft relation bias |

M4是M5/M6的结构控制实验。M5在测试时使用历史真实node，不是部署结果；M6不读取
历史真实标签，是实际可用版本。

## 3. Graph bias

关系矩阵固定为 `row=current candidate`、`column=history node`，类型为I/M/O/X/S。
矩阵不训练；每个attention head学习五个标量bias。M6使用：

```text
graph_bias(v, history_clip)
= sum_u P(history_node=u) * beta[relation(v,u)]
```

X使用有限可学习bias，不做硬mask。I同时结合实际距离；I历史不是最近token时会加入
一个额外可学习修正。

## 4. 生成的数据协议

`tools/prepare_protocols.py` 会生成：

```text
protocols/
├── normal_only/
│   ├── train.jsonl       排除A/D/M的全部fault runs
│   ├── test_normal.jsonl J的正常runs
│   ├── test_fault.jsonl  J的fault runs
│   └── test_all.jsonl    J的全部runs
└── all_runs/
    ├── train.jsonl       A/D/M的全部runs
    ├── test_normal.jsonl
    ├── test_fault.jsonl
    └── test_all.jsonl
```

这些都是新副本，不修改原manifest内容。M目录原来的
`falut_run_test_manifest.jsonl` 和 `falut_run_test_stats.json` 已按要求分别重命名为
`fault_run_test_manifest.jsonl` 和 `fault_run_test_stats.json`。四个
`Only_falut_run_as_test_*` 是既有目录名，本包不依赖这些目录，也没有擅自重命名目录。

## 5. 安装与检查

建议Python 3.10或更新版本。PyTorch CUDA版本应根据机器驱动安装。

```bash
pip install -r requirements.txt
```

路径和数据检查：

```bash
python tools/validate_setup.py \
  --dataset-root /path/to/Stage_2_Mapstyle_Dataset \
  --task-graph assets/integrated_task_graph_latest.json \
  --relation-matrix assets/integrated_feature_history_matrix.json \
  --checkpoint /path/to/last.pth
```

如果在Windows直接运行`python tools\validate_setup.py`时提示找不到`graph_history`，
请使用`call bat\00_validate_setup.bat`，或先在包根目录执行`set PYTHONPATH=%CD%`。
这一点已在新的A/D/M跨人实验包中通过入口自动初始化修复。

模型合成输入检查：

```bash
python tools/smoke_test_models.py \
  --task-graph assets/integrated_task_graph_latest.json \
  --relation-matrix assets/integrated_feature_history_matrix.json
```

## 6. Windows运行

### 6.1 换电脑时修改路径

只需编辑 `bat/config_windows.bat`：

```bat
set "DATASET_ROOT=C:\your_path\Stage_2_Mapstyle_Dataset"
set "OBJ2_ROOT=D:\your_path\obj2_codes"
set "EXISTING_BACKBONE=D:\your_path\last.pth"
set "PYTHON_BIN=C:\your_conda_env\python.exe"
```

也可以在命令行覆盖变量，不修改文件：

```bat
set PYTHON_BIN=C:\Miniconda3\envs\pytorch\python.exe
set DATASET_ROOT=E:\datasets\Stage_2_Mapstyle_Dataset
call bat\run_main_pipeline.bat
```

### 6.2 逐步运行

```bat
call bat\00_validate_setup.bat
call bat\01_prepare_protocols.bat
call bat\03_extract_features_existing_last.bat
call bat\04_train_main_m0_m6.bat
call bat\06_summarize_results.bat
```

一键运行上述主流程：

```bat
call bat\run_main_pipeline.bat
```

all-run辅助实验：

```bat
call bat\05_train_aux_all_runs_m0_m6.bat
```

### 6.3 重新训练backbone

```bat
call bat\02_train_backbone_normal_only.bat
```

输出位于：

```text
outputs/J_as_test/cam_001484412812/backbone/normal_only/last.pth
```

使用新backbone重新提取feature时，覆盖三个变量，避免覆盖现有checkpoint实验：

```bat
set EXISTING_BACKBONE=完整包路径\outputs\J_as_test\cam_001484412812\backbone\normal_only\last.pth
set FEATURE_ROOT=完整包路径\outputs\J_as_test\cam_001484412812\features\retrained_normal_only
set MODEL_ROOT=完整包路径\outputs\J_as_test\cam_001484412812\history_models\retrained_normal_only
call bat\03_extract_features_existing_last.bat
call bat\04_train_main_m0_m6.bat
```

## 7. HPC/Slurm运行

路径和环境集中在 `slurm/config_hpc.sh`。默认沿用：

```text
DATASET_ROOT=/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Stage_2_Mapstyle_Dataset
CONDA_ENV_NAME=pytorch
ANACONDA_MODULE=Anaconda3/2022.05
CUDNN_MODULE=cuDNN/8.9.2.26-CUDA-12.1.1
partition=gpu,gpu-h100,gpu-h100-nvl
qos=gpu
```

请先检查 `EXISTING_BACKBONE` 是否与HPC实际路径一致。

自动提交主流程：

```bash
cd /path/to/graph_history_rgb_experiments_2026-07-20
bash slurm/submit_main_pipeline.sh
```

它建立以下依赖：

```text
prepare protocols
→ extract features
→ train M0
→ array train M1–M6
→ summarize
```

单独提交：

```bash
sbatch slurm/01_prepare_protocols.slurm
sbatch slurm/03_extract_features.slurm
sbatch slurm/04_train_m0.slurm
sbatch slurm/05_train_context_models.slurm
sbatch slurm/06_summarize_results.slurm
```

M0必须先完成，M1–M6需要对应协议下M0的 `last.pth`。

重新训练normal-only backbone：

```bash
sbatch slurm/02_train_backbone_normal_only.slurm
```

之后用新checkpoint抽取feature：

```bash
BACKBONE_CKPT=/path/to/backbone/normal_only/last.pth
FEATURE_ROOT=/path/to/features/retrained_normal_only
sbatch --export=ALL,BACKBONE_CKPT,FEATURE_ROOT slurm/03_extract_features.slurm
```

all-run辅助实验分别提交：

```bash
sbatch --export=ALL,TRAIN_SCOPE=all_runs slurm/04_train_m0.slurm
# 等M0完成
sbatch --export=ALL,TRAIN_SCOPE=all_runs slurm/05_train_context_models.slurm
```

## 8. Python入口说明

- `tools/prepare_protocols.py`：生成normal-only/all-run协议和统计。
- `tools/train_backbone.py`：从scratch训练31类Tier-3 backbone；无validation，最终epoch后测试。
- `tools/extract_features.py`：确定性抽取 `[N,512]` feature和 `[N,31]` logits。
- `tools/train_history_model.py`：统一训练M0–M6；最终checkpoint写出后才加载test。
- `tools/summarize_results.py`：汇总node/Tier-3 accuracy、macro-F1和balanced accuracy。
- `tools/validate_setup.py`：检查环境、graph、manifest和路径。
- `tools/smoke_test_models.py`：用合成tensor检查所有模型forward。

## 9. 输出结构

```text
outputs/J_as_test/cam_001484412812/
├── protocols/
├── backbone/
├── features/
└── history_models/
    └── existing_last/
        ├── normal_only/m0 ... m6/
        ├── all_runs/m0 ... m6/
        └── experiment_summary.csv
```

每个模型目录包含：

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

M5/M6的 `learned_parameters.json` 还保存每个head的relation bias。

## 10. 多随机种子

第一轮默认 `SEED=1`。正式结果建议1–5五个seed，并给每个seed独立MODEL_ROOT：

```bat
set SEED=2
set MODEL_ROOT=...\history_models\existing_last_seed2
call bat\04_train_main_m0_m6.bat
```

## 11. 重要解释边界

1. 现有本地 `last.pth` 的原始训练曾把J test manifest作为validation。虽然本包使用
   最终epoch而不重新选择best epoch，正式论文结果仍建议用本包重新训练的normal-only
   backbone。
2. 使用现有backbone时，历史模型只用正常run训练，但backbone原来见过A/D/M的fault
   runs。严格的完全normal-only训练需要重新训练backbone。
3. M6训练历史概率由同一训练集上训练的冻结M0产生，可能比测试概率更自信。后续可把
   out-of-fold历史概率作为扩展实验。
4. M3主要解释正常run的顺序影响。含重复node的fault run会保留实际顺序；其他fault
   顺序的M3结果只作为探索性分析，不能解释为异常修复。
5. 当前任务是预切分clip classification，不是连续视频检测或sequence decoding。

## 12. 本地交付前验证（2026-07-20）

- J-as-test划分生成成功：normal-only训练为1,054个clips/55个runs；all-run训练为
  1,340个clips/73个runs。
- J测试为555个clips/30个runs，其中正常387个clips/21个runs，fault为
  168个clips/9个runs。
- 现有Tier-3 `last.pth` 与本包ResNet3D-18完整兼容：122/122个state keys加载，
  无missing或unexpected keys。
- 使用真实相机 `001484412812` 的一个 `.pt` clip完成端到端抽取，输出shape为
  `[1, 512]`。
- M1–M6均通过forward、backward和空历史（run首动作）检查；Windows检查脚本已实际
  运行通过。
- Slurm文件使用Linux LF换行；本机没有安装WSL发行版，因此最终仍应在HPC上先运行
  `01_prepare_protocols.slurm` 或通过 `SKIP_ENV_SETUP=1` 做一次小规模环境检查。
