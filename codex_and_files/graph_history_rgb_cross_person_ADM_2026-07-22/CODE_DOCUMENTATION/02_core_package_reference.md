# 02 `graph_history`底层包逐文件参考

本卷按“无状态工具 → 视频backbone → 数据集 → 图结构 → 协议 → 指标 → 评估”的顺序解释。
模型和训练engine放在下一卷。

## 1. `graph_history/__init__.py`

包初始化文件只暴露包版本/基本标识，不执行训练、不读数据、不写文件。它的主要作用是让
`graph_history`成为可导入Python包。

## 2. `constants.py`（1–21）

### 2.1 第1行

`from __future__ import annotations`延迟类型注解求值，使`Path | str`等注解不会在导入阶段产生
不必要的运行时依赖。

### 2.2 第3–7行

| 变量 | 内容 |
|---|---|
| `NUM_GRAPH_NODES` | 35 |
| `NUM_TIER3_CLASSES` | 31 |
| `DEFAULT_CAMERA_ID` | `"001484412812"` |
| `DEFAULT_RGB_MEAN` | 三通道均值 |
| `DEFAULT_RGB_STD` | 三通道标准差 |

mean/std是长度3的tuple，顺序对应RGB通道，用于`Normalize`。

### 2.3 第9–17行

`MODEL_NAMES`把实验短名映射为描述名。它是命名/校验表，不负责实例化模型：

```text
m0 current_only
m1 history_no_position
m2 actual_history
m3 graph_valid_shuffle
m4 candidate_no_graph
m5 graph_oracle
m6 graph_predicted
```

### 2.4 第19–20行

`RELATION_TO_ID`固定`I/M/O/X/S → 0/1/2/3/4`。`ID_TO_RELATION`用字典推导反转映射。
ID是relation embedding/bias索引，不代表强弱顺序。

## 3. `utils.py`（1–175）

### `seed_everything`（14–22）

- 将输入转为int；
- 设置Python `random`、NumPy和PyTorch CPU随机种子；
- 若CUDA可用，对所有GPU设置seed；
- 配置cuDNN deterministic/benchmark状态，降低重复运行差异。

它控制模型初始化、DataLoader相关随机性和graph-valid排序的基础seed，但不能保证不同硬件/算子
100%位级一致。

### `ensure_dir`（24–28）

把输入转成`Path`，以`parents=True, exist_ok=True`创建目录并返回Path。允许目录已存在，适合普通
日志/测试结果子目录，不提供防覆盖。

### `ensure_new_output_dir`（30–39）

这是实验安全核心：

- 目标不存在：创建；
- 目标存在但为空：可继续；
- 目标非空且`overwrite=False`：抛出异常；
- 只有显式`overwrite=True`才允许复用非空目录。

标准BAT/Slurm入口不传`--overwrite`。

### `read_json` / `write_json`（41–51）

- UTF-8读取/写入；
- `write_json`先确保父目录存在；
- 使用缩进和非ASCII保留，便于中文/路径人工检查。

### `read_jsonl`（53–65）

逐行读取JSONL：

- 去除首尾空白；
- 跳过空行；
- 每个非空行独立`json.loads`；
- 返回`list[dict]`。

输入manifest通常只有几千行，因此当前实现整体载入内存。

### `write_jsonl`（67–74）

逐个序列化row并写换行。`rows`接受任意Iterable，因此可传list或生成器。

### `resolve_manifest`（76–81）

若manifest是绝对路径直接返回；否则拼到dataset root。该函数只解析路径，不检查文件一定存在。

### `run_key`（83–85）

返回：

```python
(str(row["participant"]), str(row["run"]))
```

这是历史分组与fault run识别的关键复合键。

### `select_device`（87–91）

- `requested != "auto"`：直接构造`torch.device(requested)`；
- `auto`：CUDA可用则`cuda`，否则`cpu`。

### `extract_state_dict`（93–105）

兼容不同checkpoint包装：

- checkpoint本身可能就是state dict；
- 也可能在`state_dict`、`model_state`或`model`字段；
- 返回值必须是`dict[str, Tensor]`；
- 无可识别结构时抛出错误。

### `strip_state_prefixes`（107–121）

规范checkpoint key，移除常见包装前缀，例如DataParallel产生的`module.`。值tensor不复制，只重建
键字典。

### `load_compatible_state`（123–140）

步骤：

1. CPU加载checkpoint；
2. 提取并清理state dict；
3. 与目标模型当前state逐key比较；
4. 只保留“键存在且shape完全相同”的tensor；
5. `strict=False`加载；
6. 返回已加载、缺失、意外和shape不兼容键的报告。

Tier3迁移到35-node时，最终`fc.weight/fc.bias`因输出维度31→35而自动跳过。

### `save_checkpoint`（142–161）

保存模型、optimizer、epoch、配置和可选extra。模型tensor由`state_dict()`产生；若需要断点续训，
optimizer state也一并保留。当前正式协议仍以最后epoch完成后保存的`last.pth`为最终模型。

### `append_csv`（163–172）

追加单行CSV：

- 文件不存在时写header；
- 存在时只追加row；
- 字段顺序取当前row的keys；
- 适合训练摘要，不用于需要原子替换的完整汇总。

### `env_or`（174–175）

读取环境变量；不存在时返回default。用于BAT/Slurm覆盖默认路径和超参数。

## 4. `backbone.py`（1–185）

实现3D ResNet，输入维度顺序为`[B,C,T,H,W]`。

### `get_inplanes`（15–17）

返回`[64,128,256,512]`，对应四个ResNet stage的输出channel。

### `conv3x3x3`（19–21）

构造3D卷积：

```text
kernel=3×3×3
padding=1
bias=False
stride可配置
```

保持空间/时间尺寸或按stride下采样。

### `conv1x1x1`（23–25）

1×1×1投影卷积，用于shortcut channel/stride对齐。

### `BasicBlock`（27–47）

`expansion=1`。初始化建立：

```text
conv3x3x3 → BatchNorm3d → ReLU
→ conv3x3x3 → BatchNorm3d
→ 加identity/downsample
→ ReLU
```

`forward`逐行含义：

1. `residual=x`保存shortcut；
2. 第一卷积、BN、ReLU；
3. 第二卷积、BN；
4. 若downsample存在，对residual投影；
5. 与主分支相加；
6. 最终ReLU并返回。

### `Bottleneck`（49–72）

`expansion=4`，结构是1×1降/调channel、3×3处理、1×1扩展。当前ResNet3D-18使用BasicBlock，
Bottleneck保留给更深模型。

### `ResNet.__init__`（75–119）

关键成员：

```text
conv1: 7×7×7, input channels=3, output=64
bn1/relu/maxpool
layer1..layer4
avgpool: AdaptiveAvgPool3d((1,1,1))
fc: Linear(512*expansion, n_classes)
```

权重初始化：

- Conv3d使用Kaiming normal；
- BatchNorm scale=1、bias=0。

### `_downsample_basic_block`（121–127）

用于shortcut type A：用平均池化降采样，再用零channel补齐。`out.data`切断该分支旧式autograd关系；
当前配置应以构造参数中的shortcut type为准。

### `_make_layer`（129–146）

决定是否需要downsample，然后：

- 第一个block处理stride/channel变化；
- 剩余blocks保持尺寸；
- 更新`self.in_planes`；
- 返回`nn.Sequential`。

### `forward_stem`（148–150）

依次执行`conv1 → bn1 → relu → maxpool`，输出五维feature map。

### `forward_features`（152–160）

执行stem和四个residual stages，adaptive average pool后`flatten(1)`：

```text
[B,512,1,1,1] → [B,512]
```

这是feature extraction使用的接口。

### `forward_head`（162–165）

对`[B,512]`先dropout再fc，得到`[B,n_classes]`。

### `forward`（167–169）

组合`forward_features`和`forward_head`。

### `generate_model`（171–185）

按depth选择block与每stage block数。正式配置`18`对应BasicBlock `[2,2,2,2]`。不支持的depth抛出
ValueError。

## 5. `data.py`（1–281）

### `safe_torch_load`（18–22）

优先`weights_only=True`降低pickle对象加载风险；旧版PyTorch不支持该参数时捕获TypeError回退。
统一`map_location="cpu"`，设备搬运在DataLoader之后完成。

### `uniform_frame_indices`（25–28）

先检查原始帧数和目标帧数均大于0，再用`linspace(0,T_raw-1,T)`生成均匀位置并转int64列表。
当原视频少于16帧时索引可能重复，这是均匀重采样而非报错。

### `RGBVideoTransform`（31–69）

训练和评估构造两套`torchvision.transforms.v2`流水线。输入`video_tchw`为`[T,3,H,W]`，
包装成`tv_tensors.Video`保证同一个随机空间变换一致应用到所有帧。最终转普通Tensor并
`contiguous()`。

`__call__`（68–69）使transform实例可以像函数一样调用；它不改变维度顺序，只执行已构造的变换并
保证返回内存连续tensor。

### `RGBClipDataset.__init__`（73–101）

逐项动作：

1. 规范dataset root和manifest路径；
2. 保存camera、帧数；
3. 整体读取manifest；
4. 构造transform；
5. 可选逐样本检查`<camera>_rgb`；
6. 最多收集10个缺失示例后抛FileNotFoundError。

### `RGBClipDataset.__getitem__`（106–128）

1. 按index取manifest行；
2. 拼出RGB文件路径并安全加载；
3. 字典对象取`frames`，tensor对象直接使用；
4. 验证`[T,3,H,W]`；
5. 均匀取16帧；
6. 变换后从`[T,3,H,W]`换为`[3,T,H,W]`；
7. 返回视频、两个标签和样本metadata；
8. `node_idx-1`完成1-based→0-based转换。

### `HistoryExample`（131–136）

冻结dataclass保存cache索引和manifest行，不重复保存feature tensor。

### `load_feature_cache`（139–146）

要求四个字段`features/tier3_logits/records/metadata`，并验证record数等于feature第一维。

### `FeatureHistoryDataset.__init__`（150–212）

逐段：

- 校验`history_order ∈ {actual,graph_valid}`；
- cache feature强制float32；
- graph-valid模式必须提供`TaskGraphSpec`；
- 建立`sample_name → cache index`；
- selection manifest中任一样本不在cache则停止；
- 按`run_key`分组；
- 每组按annotation index排序；
- 当前位置之前的rows构成历史；
- graph-valid时用sample稳定seed重排；
- 将当前/历史cache索引封装为HistoryExample；
- 最终按participant、run、annotation index排序，保证评估输出稳定。

### `feature_dim` / `__len__`（214–219）

feature维度取`features.shape[1]`；长度为构造出的当前样本数。

### `__getitem__`（221–248）

- 有历史时用long索引`index_select`，结果`[l,F]`；
- 空历史创建同dtype/device的`[0,F]`；
- position ID为`l..1`，1代表presented sequence最后一个/最近历史；
- 历史node转0-based long tensor；
- 返回当前feature、历史、targets和可追溯sample names。

### `collate_history_batch`（251–280）

先计算`B/F/L`，创建padding张量，再逐样本写入前`length`位置并把mask设False。最后stack当前feature、
构造target tensors，同时保留字符串列表。该函数是所有M0–M6和Direct模型DataLoader的
`collate_fn`。

## 6. `graph.py`（1–167）

### `TaskGraphSpec.load`（29–89）

1. 读取两个JSON；
2. 建立1-based node字典；
3. 验证1..35完整存在；
4. 生成`node_to_tier3 [35]`；
5. 生成`node_to_stage [35]`；
6. 根据relation JSON显式列顺序建立column lookup；
7. 双循环构造`relation_ids [35,35]`；
8. `.`规范化为`X`，未知关系立即报错；
9. 读取全部必须前序与立即前序；
10. 读取atomic sequences；
11. 返回不可变TaskGraphSpec。

关系矩阵赋值使用`current_node-1, previous_node-1`，因此模型内部完全0-based。

### `stable_sample_seed`（92–94）

把`base_seed:sample_name`做SHA-256，取前8字节作为无符号整数。与Python内置`hash()`不同，它不受
进程hash随机化影响，所以同seed、同sample始终得到相同graph-valid排列。

### `randomized_graph_valid_history`（97–167）

功能是对“已经观察到的历史rows”产生一个随机但合法的拓扑序：

- 0/1个历史直接复制返回；
- 历史出现重复node时回退实际顺序，避免单个node ID无法区分多次实例；
- 只在已观察node之间建约束边；
- 使用`all_must_previous`和atomic sequence约束；
- `immediate_previous`虽然由`TaskGraphSpec.load`保存，但当前排序函数没有单独读取它；
- Kahn式拓扑排序每一步从当前可用node中用本地`random.Random(seed)`选择；
- 若观察子图出现环或无法完成，抛出`RuntimeError`，不会静默改回实际顺序；
- 不接收当前目标node，因此不会用当前标签决定历史排列。

## 7. `protocols.py`（1–144）

### `find_fault_manifest`（12–22）

为指定participant定位`fault_run_test_manifest.jsonl`。尝试预定义候选位置；找不到时抛出明确错误。

### `load_global_fault_run_keys`（24–30）

对A/D/J/M读取fault manifests，把每行转换为`(participant,run)`集合。集合去重保证一个fault run
无论含多少clip只记录一次。

### `_validate_rows`（32–52）

检查协议行关键字段、node/Tier3/stage范围以及排序所需信息；错误中携带source，便于定位原manifest
还是生成协议。

### `_sorted`（54–63）

返回新list，按participant、run、annotation index和sample name稳定排序。

### `_summary`（65–86）

统计samples、run数、node/Tier3支持度、缺失类别等，供`protocol_report.json`使用。

### `prepare_protocols`（88–144）

读取完整manifest并产生：

- held-out participant的test normal/fault/all；
- 其余participants的normal-only train；
- 其余participants的all-runs train；
- 各split摘要。

fault划分以run为单位，而不是只移除fault run中的某几个clip，防止同run泄漏到normal-only训练。

## 8. `metrics.py`（1–60）

### `confusion_matrix`（9–14）

创建`[K,K] int64`零矩阵，逐样本把`matrix[truth,pred] += 1`。越界标签被忽略，正常数据验证应保证
不会发生。

### `metrics_from_confusion`（17–42）

- `support`按行求和；
- `predicted`按列求和；
- `tp`取对角线；
- 安全除法计算recall/precision/F1，零分母输出0；
- `present=support>0`；
- accuracy对所有样本；
- macro-F1和balanced accuracy只平均present classes；
- 返回per-class数组、support和完整matrix。

### `classification_metrics`（45–48）

把Python list转int64 NumPy数组，再组合前两个函数。

### `aggregate_node_probabilities`（51–60）

创建前导维度不变、末维31的零tensor；把`node_to_tier3`reshape/expand到与node probabilities相同
shape；最后沿末维`scatter_add_`。输入输出保留device和dtype。

## 9. `video_evaluation.py`（1–203）

### `_stage_metrics`（15–30）

按stage筛选真实/预测列表，对每个stage分别调用classification metrics。

### `_write_predictions`（32–41）

确保目录存在，使用首行keys作为CSV header；空rows时不写数据行。

### `evaluate_tier3_video_model`（43–109）

用于E2E-Tier3：

- `@torch.no_grad()`；
- 模型eval；
- 遍历RGB loader并把tensor移到device；
- 可选AMP；
- logits `[B,31]`→softmax；
- 收集真实/预测、stage和逐样本metadata；
- 计算overall/per-stage Tier3；
- 写metrics、predictions和probability tensor；
- 不伪造node指标。

### `evaluate_node_video_model`（111–203）

用于E2E-Node：

- RGB模型输出`[B,35]`；
- softmax得到node probabilities；
- 调用固定映射聚合`[B,31]`；
- 同时收集node和Tier3；
- 写overall/per-stage、predictions以及两种概率。
