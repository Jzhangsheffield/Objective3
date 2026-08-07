# 标注与标签生成依据

## 使用的版本

正式输入目录为：

`D:\Junxi_data\MULTISENSOR_DATA_COLLECTION_Stage2_structured_data\Action_Segmentation_Dataset\annotations\action_recognition_boundaries_with_background_v1`

这个版本以动作识别标注的动作区间为主，并从原 segmentation 标注恢复动作间 background。其审计结果记录在该目录的 `generation_audit.csv`、`generation_summary.json` 和 `annotation_set_manifest.jsonl`。

已确认的整体规模：103 runs、1895 个动作。1886 个动作区间与动作识别边界完全一致；9 个相邻动作存在区间重叠，采用“前一动作 end = 后一动作 start - 1”消除 26 个重叠帧。45 个短 background 融合差异被恢复，共 113 帧。恢复后不足 10 帧的 background 为 123 段、676 帧。

## 本实验如何解释 54 条边界差异

这些差异不能统一当成“错误时间戳”。其中主要部分来自标注策略：两个动作之间的 background 少于约 8/10 帧时，动作识别数据曾把这些帧合并到前一个或后一个动作。动作类别本身没有错误。

本版本采用以下原则：

1. 动作边界服从动作识别标注，使 M3 所看到的动作语义区间保持一致。
2. 被策略性融合的短 background 恢复为独立 background。
3. 仅当两个动作标注发生真实帧重叠时才做确定性去重；不删除动作。
4. 无效时间戳不直接作为训练标签；逐帧标签最终依赖可验证的 frame name、frame index 和连续覆盖。

## 训练标签

- state：`action` 字段不等于 `background` 时为 1，否则为 0。
- start：动作 segment 的第一帧为 1。
- end：动作 segment 的最后一帧为 1。
- action→action：即使中间没有 background，前一 end 和后一 start 仍是两个独立事件。
- 缓存 stride=1 时边界完全保持逐帧位置。
- stride>1 仅用于调试：边界量化到事件发生后的第一个 anchor，避免提前使用未来事件。

`boundary_label_radius_frames` 只扩张训练 target，不改变评估 GT。评估从 cache 中的未扩张状态转换和原始帧号计算。

## 必须持续检查的条件

- 每个 frame annotation 的 `frame_name` 在指定相机目录真实存在；
- `original_frame_idx` 严格递增；
- segmentation 覆盖区间连续且无空洞、无重叠；
- 短 background 不因在线 `merge_gap_steps` 被再次合并；
- annotation 版本及哈希写入 feature cache metadata。
