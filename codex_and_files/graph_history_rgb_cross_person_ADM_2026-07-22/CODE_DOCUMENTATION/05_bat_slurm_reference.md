# 05 Windows BAT、Shell与Slurm逐脚本参考

## 1. 如何阅读这些脚本

脚本层不实现模型数学；它负责：

- 设置路径、participant、seed、scope和超参数；
- 检查依赖是否完成；
- 把长参数传给`tools/*.py`；
- 建立顺序或Slurm `afterok`依赖；
- 遇到非零退出码立即停止；
- 把不同实验写进互相隔离的目录。

## 2. BAT常见语句

| 语句 | 含义 |
|---|---|
| `@echo off` | 不回显每条命令 |
| `setlocal` | 环境变量修改限制在当前脚本 |
| `call config_windows.bat` | 导入公共路径/默认值 |
| `if errorlevel 1 exit /b 1` | 上一步失败则停止 |
| `if exist completed.json ...` | 完成后安全跳过 |
| `call other.bat` | 在同一cmd进程调用子流程并返回 |
| `"%PYTHON_BIN%" ...` | 使用带空格安全引用的指定Python |
| `^` | Windows命令续行 |

## 3. Windows配置与基础1–12

| BAT | 作用 |
|---|---|
| `config_windows.bat` | 定义package root、dataset、Python、assets、outputs和默认超参数 |
| `00_validate_setup.bat` | 调用`validate_setup.py`检查当前fold |
| `01_prepare_protocols.bat` | 为当前`TEST_PARTICIPANT`生成协议 |
| `02_train_backbone_normal_only.bat` | 从scratch训练normal-only Tier3 backbone |
| `03_extract_features_retrained_last.bat` | 用normal-only `last.pth`提取train/test cache |
| `04_train_main_m0_m6.bat` | 先M0，再顺序训练M1–M6 |
| `05_train_aux_all_runs_m0_m6.bat` | 旧辅助条件：normal-only representation + all-runs history训练 |
| `06_summarize_results.bat` | 单fold/seed早期M0–M6汇总 |
| `07_summarize_cross_person.bat` | 早期A/D/M跨人汇总 |
| `08_evaluate_e2e_tier3_existing.bat` | 复测normal-only Tier3 checkpoint |
| `09_train_e2e_node_scratch.bat` | normal-only 35-node E2E scratch |
| `10_train_e2e_node_from_tier3.bat` | normal-only Tier3迁移35-node |
| `11_summarize_all_models_fold.bat` | 单fold/seed十模型统一汇总 |
| `12_summarize_all_models_cross_person.bat` | 早期多fold十模型汇总 |

## 4. 完整all-runs 13–23

| BAT | 作用 |
|---|---|
| `13_prepare_protocols_all_runs_safe.bat` | 协议完整则复用、部分存在则停止 |
| `14_train_backbone_all_runs.bat` | 独立all-runs Tier3 backbone |
| `15_extract_features_all_runs.bat` | 用all-runs backbone提取独立cache |
| `16_train_all_runs_m0_m6.bat` | 用all-runs cache训练M0–M6 |
| `17_evaluate_e2e_tier3_all_runs.bat` | 复测all-runs Tier3 |
| `18_train_e2e_node_scratch_all_runs.bat` | all-runs E2E node scratch |
| `19_train_e2e_node_from_tier3_all_runs.bat` | all-runs Tier3迁移node |
| `20_summarize_all_runs_fold.bat` | 单fold all-runs十模型 |
| `21_summarize_training_scope_comparison_fold.bat` | 单fold all-runs−normal-only |
| `22_summarize_all_runs_cross_person.bat` | all-runs跨人 |
| `23_summarize_training_scope_comparison_cross_person.bat` | 跨人scope差值 |

## 5. 严格四折多seed 24–30

| BAT | 作用 |
|---|---|
| `24_train_backbone_normal_only_safe.bat` | 带completed保护的normal-only backbone |
| `25_extract_features_normal_only_safe.bat` | 带cache/marker检查的feature extraction |
| `26_train_normal_only_m0_m6_safe.bat` | 安全训练normal-only M0–M6 |
| `27_summarize_normal_only_fold.bat` | 单fold/seed normal-only十模型 |
| `28_summarize_normal_only_ADJM_3seeds.bat` | 严格4人×3seed normal-only |
| `29_summarize_all_runs_ADJM_3seeds.bat` | 严格4人×3seed all-runs |
| `30_summarize_training_scope_comparison_ADJM_3seeds.bat` | 严格360条scope配对 |

## 6. Direct Head Fusion 31–33

| BAT | 作用 |
|---|---|
| `31_train_direct_head_fusion_normal_only.bat` | 当前participant/seed运行3个normal Direct模型 |
| `32_train_direct_head_fusion_all_runs.bat` | 当前participant/seed运行3个all-runs Direct模型 |
| `33_summarize_direct_head_fusion_ADJM_3seeds.bat` | 检查216行网格并生成Direct汇总 |

Direct训练脚本复用已有feature cache，不调用backbone/M0训练。A/D/M seed1旧normal cache缺少后来新增的
completed marker时，脚本通过显式验证两个`.pt`文件允许安全复用。

## 7. Windows组合入口

| BAT | 完整调用链 |
|---|---|
| `run_one_fold.bat` | validate→protocol→normal backbone→features→M0–M6→可选aux→summary |
| `run_all_ADM.bat` | 顺序运行A/D/M `run_one_fold` |
| `run_additional_e2e_one_fold.bat` | 只运行3个E2E和新汇总，不重跑M0–M6 |
| `run_additional_e2e_ADM.bat` | A/D/M增量E2E |
| `run_all_runs_one_fold.bat` | 单fold完整all-runs pipeline |
| `run_all_runs_ADM.bat` | A/D/M完整all-runs |
| `run_normal_only_complete_one_fold.bat` | 单fold/seed严格normal-only完整链 |
| `run_normal_only_multiseed_ADM.bat` | A/D/M补seed 2/42 |
| `run_strict_J_one_seed.bat` | J一个seed的两个scope严格scratch |
| `run_strict_J_three_seeds.bat` | J的1/2/42 |
| `run_fourfold_ADJM_summaries.bat` | 只重建三个严格四折汇总 |
| `run_recommended_strict_experiments.bat` | 补齐严格网格并汇总 |
| `run_direct_head_fusion_one_fold.bat` | 当前participant/seed两个scope的6个Direct |
| `run_direct_head_fusion_ADJM.bat` | 4人×3seed×2scope×3模型并最终汇总 |

## 8. Slurm作业脚本 01–12

| Slurm | GPU/CPU性质 | 对应任务 |
|---|---|---|
| `01_prepare_protocols.slurm` | CPU | 协议 |
| `02_train_backbone_normal_only.slurm` | GPU | normal Tier3 backbone |
| `03_extract_features.slurm` | GPU | normal features |
| `04_train_m0.slurm` | GPU/轻量 | normal M0 |
| `05_train_context_models.slurm` | array | M1–M6 |
| `06_summarize_results.slurm` | CPU | fold汇总 |
| `07_summarize_cross_person.slurm` | CPU | 跨人汇总 |
| `08_evaluate_e2e_tier3_existing.slurm` | GPU | Tier3复测 |
| `09_train_e2e_node_scratch.slurm` | GPU | node scratch |
| `10_train_e2e_node_from_tier3.slurm` | GPU | node迁移 |
| `11_summarize_all_models_fold.slurm` | CPU | 十模型fold |
| `12_summarize_all_models_cross_person.slurm` | CPU | 十模型跨人 |

## 9. Slurm完整all-runs 13–24

| Slurm | 任务 |
|---|---|
| `13_prepare_protocols_all_runs_safe.slurm` | 安全协议 |
| `14_train_backbone_all_runs.slurm` | all-runs backbone |
| `15_extract_features_all_runs.slurm` | all-runs features |
| `16_train_all_runs_m0.slurm` | all-runs M0 |
| `17_train_all_runs_context_models.slurm` | all-runs M1–M6 array |
| `18_evaluate_e2e_tier3_all_runs.slurm` | all Tier3复测 |
| `19_train_e2e_node_scratch_all_runs.slurm` | all node scratch |
| `20_train_e2e_node_from_tier3_all_runs.slurm` | all node迁移 |
| `21_summarize_all_runs_fold.slurm` | fold all summary |
| `22_summarize_training_scope_comparison_fold.slurm` | fold scope delta |
| `23_summarize_all_runs_cross_person.slurm` | 跨人all |
| `24_summarize_training_scope_comparison_cross_person.slurm` | 跨人scope delta |

## 10. Slurm严格补齐与Direct 25–35

| Slurm | 任务 |
|---|---|
| `25_train_backbone_normal_only_safe.slurm` | strict normal backbone |
| `26_extract_features_normal_only_safe.slurm` | strict normal features |
| `27_train_normal_only_m0.slurm` | strict normal M0 |
| `28_train_normal_only_context_models.slurm` | strict normal M1–M6 |
| `29_summarize_normal_only_fold.slurm` | strict normal fold |
| `30_summarize_normal_only_ADJM_3seeds.slurm` | 四折normal |
| `31_summarize_all_runs_ADJM_3seeds.slurm` | 四折all |
| `32_summarize_training_scope_comparison_ADJM_3seeds.slurm` | 四折scope delta |
| `33_train_direct_head_fusion_normal_only.slurm` | array 1–3 normal Direct |
| `34_train_direct_head_fusion_all_runs.slurm` | array 1–3 all Direct |
| `35_summarize_direct_head_fusion_ADJM_3seeds.slurm` | Direct完整汇总 |

Direct array映射：

```text
SLURM_ARRAY_TASK_ID=1 → m1_direct
SLURM_ARRAY_TASK_ID=2 → m2_direct
SLURM_ARRAY_TASK_ID=3 → m3_direct
```

## 11. HPC配置和提交脚本

### `config_hpc.sh`

集中定义dataset、package/output root、module/conda环境、camera、epoch、batch size和资源参数。被作业与
submit脚本source。换HPC时优先修改此文件。

### 单fold提交

| Shell | 依赖图 |
|---|---|
| `submit_one_fold.sh` | prepare→backbone→features→M0→M1–M6→summary |
| `submit_aux_one_fold.sh` | 已有features→aux M0/M1–M6→summary |
| `submit_additional_e2e_one_fold.sh` | 3个E2E并行→fold summary |
| `submit_all_runs_one_fold.sh` | protocol后backbone/history/E2E分支→summary→scope compare |
| `submit_normal_only_one_fold.sh` | strict normal完整链 |
| `submit_strict_J_one_seed.sh` | J同seed normal与all完整链 |
| `submit_direct_head_fusion_one_fold.sh` | normal/all各一个3-task array，可选both |

### 多fold提交

| Shell | 作用 |
|---|---|
| `submit_ADM.sh` | A/D/M主链并最终跨人汇总 |
| `submit_additional_e2e_ADM.sh` | A/D/M增量E2E |
| `submit_all_runs_ADM.sh` | A/D/M all-runs |
| `submit_normal_only_multiseed_ADM.sh` | A/D/M normal seed2/42 |
| `submit_strict_J_three_seeds.sh` | J三个seed |
| `submit_recommended_strict_experiments.sh` | 推荐严格补齐总入口 |
| `submit_direct_head_fusion_ADJM.sh` | 提交24个arrays并以afterok等待全部完成后汇总 |

## 12. Slurm依赖语义

提交脚本保存`sbatch --parsable`返回的job ID，并用：

```text
--dependency=afterok:<jobid>
```

只有上游成功退出才启动下游。多个独立分支用冒号连接job IDs。Direct总入口的汇总依赖24个array jobs
全部成功；任何一个participant/seed/scope的三个模型任务失败，最终summary不会运行。

## 13. 不应混用的入口

- 已完成严格结果后，不要为新增E2E调用旧`run_one_fold.bat`；
- Direct实验不要调用M0或backbone脚本；
- 完整all-runs不要误用旧辅助`05_train_aux_all_runs_m0_m6.bat`；
- 正式四折汇总不要用只默认A/D/M的早期cross-person入口；
- 只想重新汇总时直接调用28–30或33，不要重跑训练链。

