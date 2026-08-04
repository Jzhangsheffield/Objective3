# RGB Task-Graph History 源码技术手册

文档版本：2026-07-29  
对应代码包：`graph_history_rgb_cross_person_ADM_2026-07-22`  
正式实验：A/D/J/M 四折 LOSO，seed 1/2/42，normal-only 与 all-runs  
新增实验：Direct、Dynamic，以及Atomic-tail Graph-Valid三模型

## 1. 这套文档解决什么问题

本目录不是运行说明的简单重复，而是面向源码阅读、实验复现和后续开发的技术手册。文档把研究设计、
数据字段、张量维度、Python实现、Windows/HPC编排和结果文件连接成一条完整链路。

阅读完后应能够回答：

- 一个RGB clip如何变成`[512]`维冻结特征；
- 同run历史如何从manifest构造、排序、padding并送入attention；
- M0–M6分别使用哪些输入，哪些权重被冻结，最终logits如何产生；
- 35-node概率如何聚合成31类Tier3概率；
- Direct Head Fusion为何不加载M0，以及它与原delta head的本质区别；
- 一个fold、一个seed、一个scope由哪些脚本按什么顺序完成；
- 每一种checkpoint、cache、metrics、prediction和summary写到哪里；
- 哪些统计是participant级聚合，哪些只是seed-fold或run级描述。

## 2. 推荐阅读顺序

1. [00_project_and_pipeline.md](00_project_and_pipeline.md)  
   先理解研究目标、版本范围、模型族和端到端数据流。
2. [01_tensor_and_data_dictionary.md](01_tensor_and_data_dictionary.md)  
   建立统一符号，理解manifest、feature cache、batch和输出文件的内容。
3. [02_core_package_reference.md](02_core_package_reference.md)  
   从`constants.py`、`utils.py`、`backbone.py`、`data.py`、`graph.py`开始阅读底层实现。
4. [03_models_and_engine_line_by_line.md](03_models_and_engine_line_by_line.md)  
   逐段理解M0–M6、Direct Head Fusion、loss、训练和评估。
5. [04_tools_reference.md](04_tools_reference.md)  
   阅读所有Python命令行入口及其输入输出。
6. [05_bat_slurm_reference.md](05_bat_slurm_reference.md)  
   理解Windows BAT、HPC Slurm作业和提交脚本如何编排实验。
7. [06_direct_head_fusion_results.md](06_direct_head_fusion_results.md)  
   查看新增Direct Head Fusion完整结果及与M0/原M1–M3的严格配对比较。
8. [07_outputs_reproducibility_and_safety.md](07_outputs_reproducibility_and_safety.md)  
   查看输出树、防覆盖、`completed.json`和结果解释边界。
9. [08_configuration_assets_and_inventory.md](08_configuration_assets_and_inventory.md)  
   查看配置、依赖、资产、原有说明文件和完整源码清单。
10. [09_dynamic_epoch_shuffle.md](09_dynamic_epoch_shuffle.md)
    查看三种dynamic模型、逐epoch合法重排、初始化边界、输出隔离和运行入口。
11. [10_atomic_tail_graph_valid.md](10_atomic_tail_graph_valid.md)
    查看atomic tail判定、刷新频率、回退逻辑、独立输出和运行入口。

## 3. 文档范围

正文以当前正式实验包为唯一实现依据：

```text
D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\
graph_history_rgb_cross_person_ADM_2026-07-22
```

旧`graph_history_rgb_experiments_2026-07-20`只属于先导实验历史，不作为当前四折代码的实现依据。

覆盖内容：

- `graph_history/`：15个Python模块；
- `tools/`：22个Python命令行入口；
- `bat/`：59个Windows入口与组合脚本；
- `slurm/`：41个作业脚本、19个配置/提交shell脚本；
- `assets/`与`configs/`；
- M0–M6、三个E2E对照、三个Direct、三个Dynamic和三个Atomic-tail模型；
- normal-only、all-runs、A/D/J/M、seed 1/2/42和三个测试split；
- 结果、预测、概率、汇总和防覆盖机制。

不把`outputs/`中的二进制checkpoint逐参数展开；文档解释其schema、来源和用途。统计数值只取自实际
CSV/JSON，不从旧报告手工复制。

## 4. 行号与“逐行解析”约定

源码行号以2026-07-29检查到的当前文件为基准。后续修改代码后行号可能移动，因此文档同时使用
“函数名 + 语义块 + 当前行号”定位。

逐行解析遵循以下规则：

- import、赋值、条件、循环、张量运算、I/O、返回值均解释；
- 多行函数调用按一个语义操作解释，同时说明每个实参；
- 空行、纯括号行和只用于排版的换行不单独重复解释；
- 动态列表和字典写“内容schema”，张量写`shape/dtype/device`；
- shape中的`B`、`L`等符号在张量字典中统一定义。

## 5. 重要结论先览

- 当前最终任务是预切分RGB clip分类，不是连续视频检测或fault detection。
- M0–M6预测35个graph node；Tier3指标来自35-node概率求和，不是argmax node再映射。
- M5读取真实历史node，是oracle实验；M6使用冻结M0概率，是可部署soft-relation版本。
- M3使用graph-valid重排，但重排不读取当前目标node，因此避免current-label leakage。
- Direct Head Fusion不加载M0；attention、fusion和随机初始化35-node head联合训练50 epoch。
- Dynamic Frozen-M0 Delta加载并冻结M0；Dynamic Joint-Head Delta明确不加载M0，node head与delta联合训练。
- Dynamic Direct Fusion不使用delta；三个dynamic模型只在训练期逐epoch重排，主测试使用固定合法顺序。
- Atomic-tail只锚定真实最新历史形成的未完成atomic前缀，不读取当前target；支持每1/10 epochs或全程一次。
- 新Direct结果中，`m2_direct`与`m3_direct`大幅高于原delta版本；完整结果见第6卷。

## 6. 与原有说明/结果报告的关系

原包中的三份说明仍保留并应共同阅读：

```text
README.md
EXPERIMENT_RESULTS_ANALYSIS_2026-08-04.md
COMPLETE_EXPERIMENT_CONFIGURATION.md
```

本技术手册侧重“代码如何工作”；实验报告侧重“结果说明什么”；原README侧重“如何运行”。当三者
出现版本差异时，先检查当前源码、实际CSV/JSON与文件更新时间，不直接复制旧结论。
