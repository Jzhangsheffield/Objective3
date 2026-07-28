# A/run_7 与 J/run_12 三模型实时 Demo 说明

## 1. Demo 目标

本目录提供两个可在会议中直接播放的真实测试run，并通过启动选择器并排展示三个模型：

- **M0（Current Frozen RGB Feature）**：当前动作由同一RGB backbone提取512维特征，再直接预测35个Task Graph node；不使用历史。
- **M3（Graph-valid History）**：使用当前动作的 RGB 特征，以及此前已经完成动作的真实 `node_idx`，按 Task Graph 规则对历史进行重排后预测当前 node。
- **E2E-Node-Scratch**：只使用当前动作的 RGB 帧，不使用历史。

可选择的profile：

| Profile | 数据 | Seed | 动作数 | 时长 |
|---|---|---:|---:|---:|
| A / run_7 | `run_sample_000005` / camera `001484412812` | 1 | 25 | 138.43 s |
| J / run_12 | `run_sample_000058` / camera `001484412812` | 1 | 24 | 102.42 s |

两个Demo都按原始时间戳以 **1×** 速度播放。每个动作段结束后才开始预测；当前动作的真实标签在M0、M3和E2E都完成预测后才显示，并且只在此后进入M3历史。

> 重要说明：这个版本使用人工标注提供的 action/background 边界，因此 background 直接来自标注，不经过模型，也不使用置信度阈值。它展示的是“已知分段边界下的在线顺序动作识别”，不是完整的无边界在线检测系统。

## 2. 快速使用

### 2.1 启动可视化 Demo

双击：

```text
run_demo.bat
```

首先出现选择页面，点击：

```text
Run A / run_7
```

或：

```text
Run J / run_12
```

等待模型加载完成、顶部状态变为 `Ready` 后，点击：

```text
Play 1×
```

Demo 窗口会自动最大化，以便完整显示视频、预测卡和实时 Task Graph。视频会按当前可用区域动态等比例放大；全屏时不再固定显示为 960×540，只保留由窗口与视频宽高比不同造成的最小 letterbox。

界面支持：

- `Play / Pause / Resume`：按 1× 时间戳速度播放或暂停；
- `Restart`：清空预测结果和 M3 历史，从头播放；
- 当前 segment 状态：显示 background 或 action 是否正在进行；
- M0、M3与E2E三张预测卡：明确显示 `Predicted Node`、完整 `node_id`、occurrence、confidence、top-3，以及文字化的 `CORRECT / INCORRECT`；
- Ground truth：只在三个预测完成后揭示，并独立显示 `Ground-truth Node` 和完整 `node_id`；
- Live Task Graph：显示 37 个节点和 48 条直接依赖边；已完成真实节点按预测结果使用绿/黄/红填充，当前 Ground Truth 使用亮色边框；
- Consecutive execution：必须相邻执行的 atomic sequence 使用紫色加粗箭头和组下方的整体方括线表示；
- 图上模型标记：M0 为橙色、M3 为绿色、E2E 为蓝色，三个标签显示在各自预测节点上方；预测相同时在同一节点上纵向排列，预测不同时分别落在不同节点；
- 节点悬停信息：鼠标移到节点上可查看 node、stage、Tier-3 label，以及当前预测到该节点的模型 confidence；
- Completed action history：显示已完成动作、三个模型的结果和推理时间；最近两行直接可见，其余可滚动查看。

模型加载需要额外时间，取决于磁盘和GPU状态。

当前版本为两个profile都预生成了 **960×540 H.264显示视频**并使用后台顺序解码。如果视频缺失，运行 `build_all_display_videos.bat` 重新生成。

### 2.2 运行完整一致性验证

双击：

```text
validate_all_demos.bat
```

该脚本会依次重跑A的25个动作和J的24个动作，并将三个模型的新预测与原实验保存结果逐项比较。结果分别写入：

```text
outputs/validation_predictions.jsonl
outputs/validation_summary.json
profiles/j_run12_seed1/outputs/validation_predictions.jsonl
profiles/j_run12_seed1/outputs/validation_summary.json
```

### 2.3 重新生成派生元数据

通常不需要手动运行。如果复制的标注快照或配置发生变化，可双击：

```text
prepare_demo_data.bat
```

`demo.py` 每次启动时也会先执行同一套数据检查和派生过程。

### 2.4 重新生成流畅播放视频

双击：

```text
build_all_display_videos.bat
```

它会为两个profile分别生成：

```text
derived/display_960x540_h264.mp4
profiles/j_run12_seed1/derived/display_960x540_h264.mp4
```

这些视频只用于界面显示，三个模型仍使用原始JPEG产生模型输入。重新编码不会改变M0、M3或E2E的预测。

## 3. 目录结构

```text
task_graph_realtime_demo_A_run7_2026-07-27/
├─ README.md
├─ demo_profiles.json
├─ launcher.py
├─ config.json
├─ demo.py
├─ demo_core.py
├─ task_graph_view.py
├─ prepare_demo_metadata.py
├─ build_display_video.py
├─ display_video.py
├─ benchmark_playback.py
├─ render_preview.py
├─ run_demo.bat
├─ run_a_demo.bat
├─ run_j_demo.bat
├─ validate_all_demos.bat
├─ build_all_display_videos.bat
├─ validate_demo.bat
├─ prepare_demo_data.bat
├─ build_display_video.bat
├─ source_metadata/
│  ├─ manifest.source.jsonl
│  ├─ run_sample_000005_frame_annotation.source.csv
│  ├─ run_sample_000005_segmentation_annotation.source.csv
│  └─ raw_trim_tracking.source.csv
├─ derived/
│  ├─ demo_manifest.json
│  ├─ demo_segments.jsonl
│  ├─ data_validation_report.json
│  ├─ display_960x540_h264.mp4
│  ├─ display_video_report.json
│  └─ playback_benchmark.json
├─ outputs/
│  ├─ validation_predictions.jsonl
│  └─ validation_summary.json
├─ profiles/
│  └─ j_run12_seed1/
│     ├─ config.json
│     ├─ source_metadata/
│     │  ├─ run_sample_000058_frame_annotation.source.csv
│     │  └─ run_sample_000058_segmentation_annotation.source.csv
│     ├─ derived/
│     │  ├─ demo_manifest.json
│     │  ├─ demo_segments.jsonl
│     │  ├─ data_validation_report.json
│     │  ├─ display_960x540_h264.mp4
│     │  └─ display_video_report.json
│     └─ outputs/
│        ├─ validation_predictions.jsonl
│        └─ validation_summary.json
└─ previews/
   ├─ a_run7_seed1_preview.png
   └─ j_run12_seed1_preview.png
```

原始RGB帧没有复制到本目录。Demo只读访问原始帧目录。A显示视频约10 MB，J显示视频约7 MB，二者都是独立派生缓存，不会覆盖或改变任何原始JPEG。

## 4. 使用的数据

### 4.1 A/run_7 原始数据

数据集根目录：

```text
D:\Junxi_data\MULTISENSOR_DATA_COLLECTION_Stage2_structured_data\Action_Segmentation_Dataset
```

本 Demo 使用的 RGB 帧：

```text
D:\Junxi_data\MULTISENSOR_DATA_COLLECTION_Stage2_structured_data\Action_Segmentation_Dataset\raw\run_sample_000005\001484412812
```

来源标注：

```text
annotations\run_sample_000005_frame_annotation.csv
annotations\run_sample_000005_segmentation_annotation.csv
```

来源 manifest：

```text
manifest.jsonl
```

对应关系为：

| 字段 | 值 |
|---|---|
| participant | `A` |
| source run | `run_7` |
| run sample | `run_sample_000005` |
| reference camera | `001484412812` |
| 实际 JPEG 数 | 4,152 |
| 当前帧编号 | 1–4,152 |
| 原始帧编号 | 677–4,828 |
| 首帧 | `20260313_101225_419908.jpg` |
| 末帧 | `20260313_101443_849216.jpg` |
| 按时间戳计算的时长 | 138.4293 s |
| 有效帧率 | 29.9864 FPS |
| segments | 51 |
| action segments | 25 |
| background segments | 26 |
| action frames | 1,127 |
| background frames | 3,025（72.856%） |

逐帧标注共有 4,152 行，并与当前目录中的 4,152 个 JPEG 文件名一一精确对应。

### 4.2 J/run_12 原始数据

J profile使用：

```text
D:\Junxi_data\MULTISENSOR_DATA_COLLECTION_Stage2_structured_data\Action_Segmentation_Dataset\raw\run_sample_000058\001484412812
```

来源标注：

```text
annotations\run_sample_000058_frame_annotation.csv
annotations\run_sample_000058_segmentation_annotation.csv
```

| 字段 | 值 |
|---|---|
| participant | `J` |
| source run | `run_12` |
| run sample | `run_sample_000058` |
| reference camera | `001484412812` |
| 实际JPEG数 | 3,073 |
| 当前帧编号 | 1–3,073 |
| 原始帧编号 | 623–3,695 |
| segments | 48 |
| action segments | 24 |
| background segments | 24 |
| action frames | 1,759 |
| background frames | 1,314（42.760%） |
| 时长 | 102.4194 s |
| 有效帧率 | 29.9943 FPS |
| 原实验action samples | `sample_000922`–`sample_000945` |

来源manifest保留裁剪前的4,066帧记录；当前raw保留原始帧623–3,695，因此派生索引规则为：

```text
current_frame_idx = original_frame_idx - 622
```

J/run_12的24个动作覆盖Stage 2和Stage 3。派生脚本已验证24/24标签、24/24动作帧数以及48/48 segment边界。

### 4.3 A/run_7 的25个动作映射

这些动作与原实验测试协议中的 `sample_000379`–`sample_000403` 精确对应。下表中的“当前帧”是 Demo 实际播放目录中的 1-based 编号；“原始帧”是裁剪前标注使用的编号。

| # | 原实验 sample | node | Tier-3 label | 当前帧 | 原始帧 | 帧数 |
|---:|---|---:|---|---|---|---:|
| 1 | `sample_000379` | 1 | unlock crimper | 31–92 | 707–768 | 62 |
| 2 | `sample_000380` | 2 | put lock on table | 269–293 | 945–969 | 25 |
| 3 | `sample_000381` | 3 | turn on main switch | 504–533 | 1180–1209 | 30 |
| 4 | `sample_000382` | 4 | turn on crimper | 694–715 | 1370–1391 | 22 |
| 5 | `sample_000383` | 8 | turn on extractor fan | 867–886 | 1543–1562 | 20 |
| 6 | `sample_000384` | 7 | turn on water pump | 1032–1046 | 1708–1722 | 15 |
| 7 | `sample_000385` | 6 | turn on air compressor | 1166–1188 | 1842–1864 | 23 |
| 8 | `sample_000386` | 10 | remove protection cover from crimper | 1325–1364 | 2001–2040 | 40 |
| 9 | `sample_000387` | 11 | put protection cover on ground | 1468–1508 | 2144–2184 | 41 |
| 10 | `sample_000388` | 9 | move pedal to safe location | 1613–1662 | 2289–2338 | 50 |
| 11 | `sample_000389` | 5 | adjust parameters | 1802–1856 | 2478–2532 | 55 |
| 12 | `sample_000390` | 12 | take plier from table | 2009–2037 | 2685–2713 | 29 |
| 13 | `sample_000391` | 13 | grip sample from table | 2134–2182 | 2810–2858 | 49 |
| 14 | `sample_000392` | 14 | place sample under electrodes | 2416–2478 | 3092–3154 | 63 |
| 15 | `sample_000393` | 15 | press pedal | 2871–2964 | 3547–3640 | 94 |
| 16 | `sample_000394` | 16 | put sample on machine table | 3053–3073 | 3729–3749 | 21 |
| 17 | `sample_000395` | 17 | grip sample from machine table | 3153–3190 | 3829–3866 | 38 |
| 18 | `sample_000396` | 18 | reverse sample | 3218–3312 | 3894–3988 | 95 |
| 19 | `sample_000397` | 19 | put sample on machine table | 3332–3361 | 4008–4037 | 30 |
| 20 | `sample_000398` | 20 | grip sample from machine table | 3436–3489 | 4112–4165 | 54 |
| 21 | `sample_000399` | 21 | place sample under electrodes | 3539–3584 | 4215–4260 | 46 |
| 22 | `sample_000400` | 22 | press pedal | 3621–3720 | 4297–4396 | 100 |
| 23 | `sample_000401` | 23 | inspect sample | 3798–3864 | 4474–4540 | 67 |
| 24 | `sample_000402` | 24 | put sample on table | 3994–4021 | 4670–4697 | 28 |
| 25 | `sample_000403` | 25 | put plier on table | 4093–4122 | 4769–4798 | 30 |

## 5. 对原数据集做了什么修改

### 5.1 原数据集本身：没有任何修改

以下目录中的文件没有被写入、重命名、移动或删除：

```text
D:\Junxi_data\MULTISENSOR_DATA_COLLECTION_Stage2_structured_data\Action_Segmentation_Dataset
```

Demo 对原始 JPEG 只做读取。A profile 根目录 `source_metadata/` 中有四个来源快照，J profile 的 `profiles/j_run12_seed1/source_metadata/` 中有两个来源快照。后续处理只针对这些快照以及各 profile 的 `derived/` 新文件。

来源快照的 SHA-256 已分别写入对应 profile 的 `derived/data_validation_report.json`：

| 快照 | SHA-256 |
|---|---|
| `manifest.source.jsonl` | `fe7e6b52de2e3b5b101f3121b976a072d646fc1c19882d3f7bde4033ac445194` |
| `run_sample_000005_frame_annotation.source.csv` | `0a65349f33213be2aba47e5e759113d6d2f53bdb3ba14a6f2f8ea15f060b9367` |
| `run_sample_000005_segmentation_annotation.source.csv` | `eb60b10c2ab4cbb82ab030bee1abd06041ecb245478b160325dcd0dc687ecbba` |
| `raw_trim_tracking.source.csv` | `16f9527520fbc8aebd5b27abb0b114e11ff2f68220a0ae8b47999044a04b1839` |
| `run_sample_000058_frame_annotation.source.csv` | `7fba531898814faa3803616af75a801c3520daa2f3634fa19c72fa48c1687dc0` |
| `run_sample_000058_segmentation_annotation.source.csv` | `dd449d43adef83d1d3eea6320f6afbaf812f19bdcfe453a098769b9755c17e69` |

### 5.2 为什么需要派生元数据

以 A / run_7 为例，来源 manifest 和 segmentation CSV 记录的是裁剪前的 reference camera 范围：

```text
1–4,971
```

当前 raw 目录实际保留的是原始编号：

```text
677–4,828
```

因此来源 manifest 中的 `reference_frame_count = 4971` 对当前 raw 目录已经过时，而当前真实帧数为 4,152。这个差异来自数据整理阶段的前后裁剪，不是 Demo 删除帧造成的。

J / run_12 同样采用来源标注与当前 raw 文件交集：当前 raw 保留原始编号 `623–3695`，共 3,073 帧，因此使用 `current_frame_idx = original_frame_idx - 622`。它最终包含 48 个 segment，其中 24 个 action、24 个 background。

### 5.3 派生文件中的调整

`prepare_demo_metadata.py` 为每个 profile 生成以下两个核心文件：

- `derived/demo_manifest.json`
- `derived/demo_segments.jsonl`

以 A / run_7 为例，调整规则为：

```text
current_frame_idx = original_frame_idx - 676
```

具体处理：

1. 保留每一帧的 `original_frame_idx`，同时使用当前目录连续的 `frame_idx = 1–4152`；
2. 将 segmentation 的原始边界与当前保留范围 `677–4828` 求交；
3. 将交集边界重排为当前帧编号；
4. 对首尾 background 做截断：
   - 开头来源 background 为 1–706，当前实际保留 677–706，对应当前 1–30；
   - 结尾来源 background 为 4799–4971，当前实际保留 4799–4828，对应当前 4123–4152；
5. 保留来源边界、当前边界、帧名、时间戳、action/object/mark 等 provenance 字段；
6. 将 25 个 action segment 与既有测试协议逐项对齐，附加：
   - `original_action_sample_name`
   - `node_idx`
   - `node_id`
   - `stage_id`
   - `tier3_id`
   - `tier3_label`
7. 自动检查文件名、帧数、标签和边界是否一致。

检查结果：

- A / run_7：51 个 segment、25 个动作的重排、标签和帧数检查全部通过；
- J / run_12：48 个 segment、24 个动作的重排、标签和帧数检查全部通过。

## 6. 模型与推理配置

### 6.1 使用的实验设置

| 项目 | 设置 |
|---|---|
| test participant | A 或 J（由启动选择器决定） |
| run | A / run_7；J / run_12 |
| camera | 001484412812 |
| seed | 1（两个 profile 相同） |
| training scope | all-runs |
| M0 | RGB feature → node classifier，不使用历史 |
| M3 | retrained all-runs / graph-valid history |
| E2E-Node-Scratch | 当前 RGB clip → node classifier，不使用历史 |
| 输出空间 | Task Graph node |

权重、Task Graph、关系矩阵和原实验预测结果由根目录 `config.json`（A）或 `profiles/j_run12_seed1/config.json`（J）指向既有实验包：

```text
D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\graph_history_rgb_cross_person_ADM_2026-07-22
```

Demo 不会改动该实验包。

### 6.2 RGB 预处理

为了复现原实验，当前动作段使用与既有 map-style 测试输入一致的处理：

1. 从 action segment 读取原始 RGB JPEG；
2. 使用 PIL `BILINEAR` resize 到 `256 × 256`；
3. 在完整动作段内使用 inclusive uniform linspace 采样 **16 帧**；
4. 模型输入 resize 到 `224 × 224`；
5. 使用：

```text
mean = [0.5369, 0.5295, 0.5208]
std  = [0.2311, 0.2360, 0.2363]
```

M0、M3 和 E2E 对当前动作使用相同的 16 帧采样规则。M0 与 M3 共用冻结 RGB backbone 得到的当前动作特征；M0 只对当前特征分类，M3 额外使用历史和 Task Graph 重排。E2E-Node-Scratch 则从当前 RGB clip 端到端预测 node。

### 6.3 M3 历史策略

M3 的状态更新顺序是：

1. action segment 结束；
2. 从当前 action 采样 16 帧并提取当前特征；
3. 读取此前已完成动作的历史；
4. 使用这些历史动作的真实 `node_idx` 做 graph-valid reordering；
5. M0 使用当前 RGB feature 独立预测当前 node；
6. M3 使用当前 feature 与重排后的历史预测当前 node；
7. E2E 使用当前 RGB clip 独立预测；
8. 三个预测均完成后，才揭示当前 ground truth；
9. 当前动作的真实 `node_idx` 和当前 RGB feature 被加入历史，供后续动作使用。

因此，当前动作的真实标签不会泄漏到当前预测中；但历史真实 node 是本次会议 Demo 明确采用的 oracle-history 条件。

### 6.4 Node 优先的界面语义

Task Graph node 是本 Demo 的预测目标，Tier-3 只作为辅助动作描述。界面不再把共享的 Tier-3 文本当作主要预测结果。

例如：

```text
Node 15 = node_15_press_pedal_1 = press pedal，第一次
Node 22 = node_22_press_pedal_2 = press pedal，第二次
```

以及：

```text
Node 14 = node_14_put_sample_under_electrodes_1
Node 21 = node_21_put_sample_under_electrodes_2
```

每个模型卡片分别显示：

1. `Predicted Node N`；
2. 完整 Task Graph `node_id`；
3. `CORRECT` 或 `INCORRECT`，错误时同时写出真正的 Ground-truth Node；
4. Tier-3 辅助标签与 `occurrence 1/2/3`；
5. confidence 和以 node 为单位的 top-3。

Ground Truth 区域使用中性颜色独立展示真实 node，不再依靠预测文字的绿/红颜色让观众反推真实标签。

### 6.5 实时 Task Graph 状态

右下角图直接读取 `integrated_task_graph_latest.json`，绘制 37 个节点和 48 条 `direct_must_previous_nodes` 边。节点按 Stage 1、Stage 2、Stage 3 三条固定 lane 排列，因此播放过程中不会发生自动布局跳动。

状态规则：

1. 未执行节点使用深色；
2. 每个动作完成并揭示 Ground Truth 后，对应真实 node 持续点亮；
   - 绿色：M0、M3、E2E 三个模型都预测正确；
   - 黄色：M3 预测正确，但 M0 或 E2E 至少一个预测错误；
   - 红色：M3 预测错误；
3. 最近完成的真实 node 额外使用亮色边框；
4. M0、M3、E2E 的当前 top-1 分别用橙色、绿色、蓝色标签显示在预测 node 上方；
5. 新动作开始时清除上一动作的三个临时预测标签，但保留已经完成的真实路径；
6. `Restart` 同时清空模型历史、列表历史和图上的已完成路径；
7. 鼠标悬停节点时，图底部显示完整动作含义和命中该节点的当前模型 confidence。

Task Graph 中共有 17 条 `must_immediately_previous_node` 约束。为避免逐对方括号互相重叠，界面按照来源 JSON 中的 `atomic_sequences` 合并显示为 5 个最大连续执行组：

```text
[N1 → N2]
[N10 → N11]
[N12 → N13 → … → N25]
[N26 → N27]
[N34 → N35]
```

组内直接箭头使用紫色加粗线，组下方使用带左右端点的紫色括线并标注 `IMMEDIATE`；普通依赖仍使用细灰色箭头。小窗口高度不足时会隐藏重复的 `IMMEDIATE` 文字，但保留括线、紫色箭头和图例。鼠标悬停连续组内节点时，底部详情也会显示该节点所属的完整 consecutive sequence。

### 6.6 Background 与边界

本版本的 `background_policy` 为：

```text
supplied_annotations_not_model_prediction
```

即：

- 标注为 background 的段只播放，界面显示 `Background / not classified`；
- 标注为 action 的段结束后才调用三个模型；
- 没有对 action confidence 使用 background threshold。

这样做避免了用“node 分类 confidence”替代“动作是否存在”的检测分数。若后续要演示未知边界的视频流，应单独增加 actionness/background 模型，或使用独立校准集确定阈值，而不是在这个测试 run 上调阈值。

## 7. 实时播放与显示时机

播放调度使用 JPEG 文件名中的采集时间戳，而不是简单地固定每 33.33 ms 显示一帧。因此：

- 速度为真实时间的 1×；
- 原始采集时间间隔被保留；
- 如果界面短时落后，会直接显示当前时间点应显示的最新帧，以保持时间轴同步；
- 模型推理在单独的工作线程中运行，不阻塞 Tk 界面；
- 推理任务串行执行，保证 M3 历史顺序稳定。

### 7.1 流畅播放优化

初版播放器在Tk主线程中对每一帧执行：

```text
打开单张JPEG → JPEG解码 → Lanczos缩放 → 创建Tk图像
```

当主线程短时落后时，初版还会在同一次界面刷新中处理所有积压帧，形成“越补帧越卡”的现象。优化版改为：

```text
FFmpeg预编码H.264
        ↓
OpenCV后台线程顺序解码
        ↓
最多缓存120帧RGB
        ↓
Tk主线程每次刷新最多创建一张图像
```

具体变化：

1. 每个 profile 的 JPEG 被一一映射为等量 H.264 视频帧；
2. 显示视频预编码为960×540；运行时根据窗口大小使用 OpenCV `INTER_LINEAR` 快速缩放到最大的等比例显示尺寸；
3. H.264解码在独立线程中提前进行，主线程只获取已经准备好的RGB帧；动态缩放基准约为0.83 ms/帧（960×540放大到1600×900）；
4. 界面落后时不再逐张补画所有积压帧，只显示时间轴所需的最新帧；
5. action边界和模型触发仍根据原逐帧标注与时间戳计算，而不是依赖视频压缩后的画面内容；
6. 模型的16帧输入仍从原始1280×720 JPEG读取，显示视频从不进入模型。

显示视频校验结果：

| Profile | 输入 JPEG / 编码帧 | 分辨率 | codec | 文件大小 | 视频时长 |
|---|---:|---:|---:|---:|---:|
| A / run_7 | 4,152 / 4,152 | 960×540 | H.264 | 10,097,331 bytes | 138.4627 s |
| J / run_12 | 3,073 / 3,073 | 960×540 | H.264 | 7,169,850 bytes | 102.4527 s |

本机600帧基准测试：

| 显示源处理 | 平均耗时/帧 |
|---|---:|
| 初版：打开JPEG、解码、Lanczos缩放 | 14.42 ms |
| 优化版：H.264顺序解码并转RGB | 1.00 ms |
| 显示源阶段加速 | 14.36× |
| 约30 FPS的总帧预算 | 33.35 ms |

该测试未把两种方案都需要的Tk图像提交耗时计入对比，因此它反映的是被本次修改消除的JPEG文件访问、解码和缩放开销。

会议展示时建议先完成一次 `validate_all_demos.bat`，然后关闭其他占用 GPU 的程序，再启动 `run_demo.bat`。

## 8. 已完成的验证结果

验证环境：

```text
Python 3.12.9
PyTorch 2.6.0+cu126
GPU: NVIDIA GeForce RTX 4090
```

2026-07-27 的全动作验证结果：

| Profile | M0 | M3 | E2E-Node-Scratch | 与既有 top-1 一致 |
|---|---:|---:|---:|---:|
| A / run_7 / seed 1 | 19/25（76%） | 22/25（88%） | 18/25（72%） | 三模型均 25/25 |
| J / run_12 / seed 1 | 19/24（79.17%） | 23/24（95.83%） | 21/24（87.5%） | 三模型均 24/24 |

| Profile | 最大 M0 confidence 差 | 最大 M3 confidence 差 | 最大 E2E confidence 差 | 平均三模型推理时间 |
|---|---:|---:|---:|---:|
| A / run_7 | 0.000401 | 0.000927 | 0.000931 | 约 158 ms/action |
| J / run_12 | 0.000732 | 0.000354 | 0.000922 | 约 156 ms/action |

两个 profile 的类别预测都与既有实验完全一致。小于 0.001 的 confidence 差异属于当前 GPU 运行下的浮点数值差异，不改变任何 top-1 node。

J / run_12 存在 seed 1、2、42 三个严格 LOSO 结果。本 Demo 选择 seed 1，是因为它与 A profile 保持相同 seed，且在 J/run_12 上三模型结果完整，M3（23/24）和 E2E（21/24）也优于另外两个候选 seed，适合现场展示。该选择只决定展示哪一份已有模型结果，没有在 J/run_12 上重新调参。

错误分布：

- M3 错误：action 13、20、24；
- E2E 错误：action 13、14、16、21、22、24、25；
- action 14 是一个适合会议讲解的例子：M3 正确预测 node 14，而只看当前 RGB 的 E2E 预测为 node 21。这两个 node 的可见动作语义相同，历史与 Task Graph 提供了步骤位置区分。

## 9. 输出文件说明

### `derived/data_validation_report.json`

记录：

- 来源快照哈希；
- raw JPEG 与逐帧标注的一致性；
- 当前/原始帧范围；
- segment/action/background 数量；
- action 与原测试协议的标签、帧数匹配；
- 时长、有效 FPS 和 background 比例；
- 所有检查是否通过。

### `outputs/validation_predictions.jsonl`

每个 action 一行，包含：

- true node、label、stage；
- M0/M3/E2E top-1、confidence、top-3；
- M3 使用的 graph-valid history node 顺序；
- 16 个采样帧的局部索引；
- 推理时间；
- 与既有实验结果的 node/confidence 差异。

### `outputs/validation_summary.json`

完整 run 的三模型准确率、复现匹配数、最大 confidence 差和平均推理时间摘要。A 的文件位于根目录 `outputs/`，J 的文件位于 `profiles/j_run12_seed1/outputs/`。

### `derived/display_video_report.json`

记录显示视频的输入/输出帧数、codec、分辨率、时长、文件大小、构建耗时和完整FFmpeg命令。A 的 `source_frames` 与 `encoded_frames` 均为 4,152；J 均为 3,073。

### `derived/playback_benchmark.json`

记录逐JPEG显示链路与H.264顺序解码链路的600帧性能对照。

### `previews/a_run7_seed1_preview.png` 与 `previews/j_run12_seed1_preview.png`

两个文件都使用对应 run 的真实帧和实测三模型预测结果生成，用于快速检查布局；不是手工伪造的占位内容。

## 10. 环境与依赖

所有 `.bat` 启动文件当前固定使用：

```text
C:\Users\digit\anaconda3\envs\Pytorch\python.exe
```

主要依赖：

- Python 3.12；
- PyTorch / torchvision；
- Pillow；
- NumPy；
- OpenCV；
- FFmpeg / ffprobe；
- Tkinter（Python 标准 GUI 组件）。

程序会优先使用 CUDA；无 CUDA 时可回退 CPU，但 CPU 推理可能无法稳定跟上 1× 播放，不建议用于会议现场。

如果环境位置改变，需要同时修改 `.bat` 中的 Python 路径。模型和数据位置改变时，分别修改 A 或 J 的 profile config。

## 11. 当前版本的边界与后续扩展

这个 Demo 可以可靠支持本周会议所需的模型对比，但解读时应明确：

1. **边界是给定的**：没有在线检测动作起止点；
2. **background 是给定的**：没有训练或预测 background 类；
3. **M3 使用真实历史 node**：属于 oracle completed-history 设置；
4. **只演示两个指定 participant/run**：A / run_7 与 J / run_12，camera 均为 001484412812；
5. **动作完成后预测**：不是动作尚未完成时的 early recognition；
6. **准确率来自这个代表性 run**：不替代完整测试集统计。

后续更接近真实部署的版本可以按以下顺序扩展：

1. 增加滑动 window 与 actionness/background 头；
2. 在独立 calibration set 上确定 background 阈值；
3. 用模型预测的已完成 node 替代真实历史 node；
4. 增加 boundary debounce、最短动作长度和多窗口投票；
5. 将当前双 profile 选择器扩展到更多 run/camera；
6. 分别报告 supplied-boundary、predicted-boundary 和 predicted-history 三种设置。

## 12. 源数据保护结论

本 Demo 的代码、复制的 metadata、派生文件、验证输出和预览图全部位于：

```text
D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\task_graph_realtime_demo_A_run7_2026-07-27
```

原始 `Action_Segmentation_Dataset` 中的 manifest、annotation CSV 和 raw 数据均未改动。
