# 04 `tools`命令行入口逐文件参考

所有入口都会把包根加入`sys.path`，因此可从包根直接运行`python tools/<name>.py`。BAT/Slurm负责
提供长路径和统一参数；直接手工调用时必须自行保证scope、cache和checkpoint匹配。

## 1. 总表

| 文件 | 主要职责 | 是否训练 | 主要写入 |
|---|---|---:|---|
| `validate_setup.py` | 环境、manifest、graph、checkpoint检查 | 否 | 控制台 |
| `prepare_protocols.py` | 生成LOSO协议 | 否 | protocols JSONL/JSON |
| `guard_output_dir.py` | shell前置防覆盖检查 | 否 | 可创建空目录 |
| `train_backbone.py` | 训练31类Tier3 ResNet3D | 是 | backbone scope目录 |
| `extract_features.py` | 提取512维feature cache | 否 | `.pt` cache |
| `train_history_model.py` | 训练M0–M6 | 是 | history model目录 |
| `train_direct_history_model.py` | 训练M1–M3 Direct | 是 | direct_head_fusion目录 |
| `evaluate_e2e_tier3.py` | 复测已有Tier3 checkpoint | 否 | E2E Tier3评估 |
| `train_e2e_node.py` | 训练两种35-node E2E | 是 | E2E node目录 |
| `smoke_test_models.py` | 合成tensor测试M1–M6 | 短反传 | 临时内存 |
| `smoke_test_direct_models.py` | 测试Direct模型与初始化 | 短反传 | 临时内存 |
| `summarize_results.py` | 单fold旧M0–M6汇总 | 否 | CSV |
| `summarize_cross_person.py` | 早期跨人汇总 | 否 | CSV |
| `summarize_all_models.py` | 十模型严格统一汇总 | 否 | 多个CSV |
| `summarize_direct_head_fusion.py` | Direct专用严格汇总 | 否 | 3 CSV + completed |

## 2. `validate_setup.py`（52行）

只有`main()`。

流程：

1. 解析dataset、task graph、relation matrix、test participant、camera和可选checkpoint；
2. 检查路径存在；
3. 加载TaskGraphSpec，验证35 nodes和relation；
4. 定位输入manifest并检查相机字段；
5. 若给checkpoint，尝试兼容加载；
6. 输出检查摘要。

它不生成协议、不训练，也不修改已有输出。适合换电脑/HPC后第一步执行。

## 3. `prepare_protocols.py`（53行）

只有`main()`。

参数：

```text
--dataset-root
--output-root
--test-participant {A,D,J,M}
--camera-id
```

流程：

1. parse；
2. 调用`prepare_protocols(...)`；
3. 写normal-only/all-runs训练manifest与三个test manifests；
4. 写`protocol_report.json`；
5. 打印每个split规模。

协议生成是确定性的，同一dataset与participant应得到相同内容。

## 4. `guard_output_dir.py`（24行）

只有`main()`。接收`--output-dir`，调用非覆盖安全逻辑。它用于BAT/Slurm在启动昂贵任务前提前失败，
避免训练很久后才发现目标目录冲突。

## 5. `train_backbone.py`（163行）

### `train_epoch`（30–53）

- `model.train()`；
- 遍历RGB batch；
- video/target搬device；
- AMP前向；
- Tier3 cross entropy；
- GradScaler backward/step/update；
- 累积按样本加权loss和accuracy；
- 返回epoch指标。

输入video`[B,3,16,224,224]`，logits`[B,31]`。

### `evaluate`（55–89）

在eval/no-grad下测试Tier3模型，收集概率和metadata并写metrics/predictions/probabilities。该内部函数
用于训练结束后的测试，不进行checkpoint选择。

### `main`（91–163）

关键参数：

```text
--dataset-root --protocol-root --train-scope
--output-dir --camera-id
--epochs --batch-size --num-workers
--learning-rate --weight-decay --seed --device
--amp/--no-amp --overwrite
```

执行：

1. seed和device；
2. `ensure_new_output_dir`；
3. 根据scope选择train manifest；
4. 构造RGB datasets/loaders；
5. `generate_model(18,n_classes=31)`；
6. AdamW、GradScaler和`MultiStepLR(milestones=[50,75], gamma=0.1)`；
7. 训练固定epochs，每个epoch结束推进scheduler；
8. 保存`last.pth`和日志；
9. 测三个split；
10. 写`completed.json`。

## 6. `extract_features.py`（108行）

只有`main()`。

参数包括dataset、manifest、Tier3 checkpoint、output、camera、batch size、workers、AMP、seed、
debug sample上限、completion marker和overwrite。

执行：

1. 保护输出文件/目录；
2. 构造非训练RGB dataset；
3. 生成31类ResNet3D并兼容加载checkpoint；
4. eval/no-grad；
5. 对每个batch调用`forward_features`得到`[B,512]`；
6. `forward_head`得到`[B,31]`；
7. 收集records；
8. CPU拼接后保存：

```text
features [S,512]
tier3_logits [S,31]
records length S
metadata
```

9. 可选写completion marker。

它不会更新backbone权重。

## 7. `train_history_model.py`（190行）

### `build_loader`（30–40）

构造DataLoader：

- dataset为FeatureHistoryDataset；
- `collate_fn=collate_history_batch`；
- shuffle由训练/测试决定；
- pin memory只在CUDA时启用；
- persistent workers依赖`num_workers>0`。

### `main`（42–190）

主要参数：

```text
--model {m0..m6}
--train-scope
--protocol-root
--train-cache --test-cache
--task-graph --relation-matrix
--baseline-checkpoint
--output-root
--epochs --batch-size --num-workers
--learning-rate --weight-decay
--d-model --num-heads --max-history --dropout
--action-loss-weight --seed --device --amp --overwrite
```

分支：

- M0：建立FeatureNodeClassifier，不需要baseline checkpoint；
- M1–M6：先建立同结构M0并严格加载指定baseline，然后由工厂构造history模型；
- M3使用`history_order=graph_valid`；
- 其余使用actual；
- M5/M6结构由model factory选择oracle/predicted。

训练集使用scope对应manifest和train cache。测试cache可覆盖held-out participant全部样本，但三个测试
Dataset分别用三个selection manifest过滤。

输出：

```text
last.pth
train_log.json
learned_parameters.json
experiment_config.json
test_results/*
completed.json
```

`learned_parameters.json`记录history scale、relation bias等可解释参数；M0相应字段较少。

## 8. `train_direct_history_model.py`（206行）

### `build_loader`（35–45）

与history loader相同。

### `main`（47–206）

模型限制为`m1_direct/m2_direct/m3_direct`。没有`--baseline-checkpoint`，这是与原history入口的关键
接口差异。

执行：

1. 解析scope和cache；
2. seed/device/output防覆盖；
3. 加载TaskGraphSpec；
4. M3 Direct选择graph-valid，其余actual；
5. 构造train和三个test FeatureHistoryDataset；
6. 从train dataset取得feature_dim；
7. `build_direct_context_model`；
8. AdamW训练固定epochs；
9. 测normal/fault/all；
10. 从feature cache metadata记录Tier3 representation来源；
11. 写checkpoint、配置、learned parameters和completed。

checkpoint不包含M0参数。`learned_parameters.json`应重点查看fusion和attention相关摘要。

## 9. `evaluate_e2e_tier3.py`（105行）

只有`main()`。加载已有31类Tier3 `last.pth`并重新对三个split评估。它：

- 不训练；
- 不复制或修改原checkpoint；
- 使用RGBClipDataset；
- 调用`evaluate_tier3_video_model`；
- node指标为空/不存在；
- 专属新目录受防覆盖和completed保护。

## 10. `train_e2e_node.py`（205行）

### `make_loader`（31–40）

普通RGB DataLoader，无history collate。

### `train_epoch`（42–64）

与backbone训练类似，但target改为35-node，logits`[B,35]`。

### `main`（66–205）

模式由参数决定：

- scratch：随机初始化35-node ResNet3D；
- from-tier3：生成35-node模型，再兼容加载31类checkpoint，最终fc因shape不同跳过。

固定epoch训练全网络；优化器是AdamW，scheduler为
`MultiStepLR(milestones=[50,75], gamma=0.1)`，每batch做梯度范数裁剪1.0。最后用
`evaluate_node_video_model`同时报告node和Tier3。

## 11. Smoke tests

### `smoke_test_models.py`

构造合成：

```text
current [B,512]
history [B,L,512]
positions/nodes/mask
```

逐个M1–M6检查：

- forward shape `[B,35]`；
- loss有限；
- backward可运行；
- baseline无梯度；
- 空历史可运行；
- graph bias诊断shape正确。

### `smoke_test_direct_models.py`

逐个Direct模型检查：

- forward/backward；
- 空历史；
- fusion初始时history权重块为0；
- current权重块为单位矩阵；
- 输出35 logits；
- 没有冻结M0依赖。

Smoke test只验证代码路径和shape，不代表模型性能。

## 12. `summarize_results.py`（54行）

只有`main()`。扫描一个fold/seed下M0–M6的三个metrics JSON，展平核心指标后写CSV。属于早期单fold
汇总，严格四折最终报告优先使用`summarize_all_models.py`。

## 13. `summarize_cross_person.py`（182行）

### 小函数

- `mean`：算术平均；
- `sample_std`：分母`n-1`，少于2项返回0；
- `write_csv`：完整写出，不追加。

### `collect_rows`

扫描指定participants的早期cross-person目录，读取M0–M6 metrics并附上participant/model/split。

### `build_deltas`

按participant与split配对，计算各模型减M0。

### `build_aggregate`

先在participant内聚合重复seed，再跨participant均值/样本标准差。

### `main`

解析root/output/participants，依次写metrics、deltas和aggregate。

## 14. `summarize_all_models.py`（583行）

这是正式十模型和严格scope比较的主汇总器。

### `write_csv`（46–53）

确保父目录，按rows第一行字段完整写CSV；空rows创建空文件/按实现处理。

### `metric_value`（55–60）

从metrics嵌套字典安全取`node/tier3`指标。直接Tier3模型的node返回None。

### `parse_location`（62–99）

根据metrics文件相对`outputs_root`的路径解析：

```text
participant
camera
seed
model
train_scope
representation_scope
split
```

这是防止normal-only表示与all-runs训练条件被误合并的关键。

### `collect_rows`（101–151）

遍历metrics JSON，解析位置，筛选participants/seeds/scopes，生成overall长表。

### `collect_stage_rows`（153–209）

把每个metrics的Stage 1/2/3展开为三行，并保留present class count。

### `require_complete_grid`（211–251）

在写汇总前检查请求网格。缺少任一participant、seed、scope、10 models或3 splits就列出缺项并抛错。

### `build_pairwise_deltas`（253–303）

在相同participant/seed/scope/split内按可比较reference配对。不会跨seed或跨representation scope。

### `mean_or_none` / `std_or_none`（305–311）

过滤None后聚合；直接Tier3的node指标保持空值。

### `aggregate_across_people`（313–359）

第一层按participant平均seeds，第二层对participants等权平均并给样本标准差。

### `aggregate_stages_across_people`（361–372）

相同规则用于per-stage。

### `aggregate_deltas`（374–428）

对严格模型差值先participant内平均seed，再跨participant。

### `build_training_scope_deltas`（430–463）

配对键：

```text
participant + seed + model + split
```

计算：

```text
all_runs - normal_only
```

并要求representation scope与train scope匹配。

### `aggregate_training_scope_deltas`（465–502）

同样采用participant内seed平均、再participant等权。

### `main`（504–583）

控制筛选、完整网格、所有CSV写入与completed marker。严格四折报告的主要源文件由这里产生。

## 15. `summarize_direct_head_fusion.py`（287行）

### `mean` / `sample_std` / `write_csv`

与cross-person工具相同的基础统计与CSV输出。

### `read_metric`（41–53）

读取一个metrics JSON并提取6个核心指标：

```text
node accuracy/macro-F1/balanced
tier3 accuracy/macro-F1/balanced
```

### `legacy_model_root`（55–60）

根据scope返回原M0–M3的正确目录：

- normal-only → `history_models/retrained_normal_only/normal_only`
- all-runs → `history_models/retrained_all_runs/all_runs`

### `collect_rows`（62–141）

对4人×3seed×2scope×3direct×3split：

1. 读取Direct metrics生成216行；
2. 读取同scope M0和对应原M1/M2/M3；
3. 生成每个Direct对两个references的432行严格差值；
4. complete-grid开启时任何缺失立即停止。

### `build_aggregate`（143–224）

分组键：

```text
scope + direct model + reference + split
```

先participant内平均3 seeds，再跨4人均值与样本标准差，同时保留Direct绝对指标与delta。

### `main`（226–287）

默认participants A/D/J/M、seeds 1/2/42、两个scope；写：

```text
direct_head_metrics.csv
direct_head_paired_deltas.csv
direct_head_aggregate.csv
completed.json
```
