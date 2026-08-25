# cache.py
```python
def resample_lc_to_cl(value: torch.Tensor, channels: int, length: int, name: str) -> torch.Tensor:
    value = torch.as_tensor(value).float()
    if value.ndim != 2 or value.shape[1] != channels:
        raise ValueError(f"{name} must be [L,{channels}], got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return F.interpolate(
        value.transpose(0, 1).unsqueeze(0), size=length, mode="linear", align_corners=False
    ).squeeze(0).contiguous()
```
.transpose(0, 1)和.transpose(1, 0)是一样的，两者都指对第0维和第一维进行交换。
F.interpolate() 对于1D信号要求输入形状:[B, C, L], 所以要进行.unsqueeze(0), 同时align_corners表示是否输入和输出的端点值是否要严格的进行对其。

整个函数的作用是检查输入的维度，并将输入长度进行变化到所需要的长度。
输入: [L, C] -> [C, L] -> [0, C, L] -> [0, C, L_new] -> [C, L_new]

```python
def _load_right_signals(dataset_root: Path, row: dict[str, Any], emg_len: int, imu_len: int):
    loaded = safe_load(dataset_root / row["mindrove"])
    if not isinstance(loaded, dict):
        raise TypeError(f"MindRove file for {row['sample_name']} is not a dict")
    emg = resample_lc_to_cl(loaded["right_emg"], 8, emg_len, "right_emg")
    acc = torch.as_tensor(loaded["right_acc"]).float()
    gyro = torch.as_tensor(loaded["right_gyro"]).float()
    if acc.shape != gyro.shape or acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError(f"Invalid right IMU shape for {row['sample_name']}: {acc.shape}, {gyro.shape}")
    imu = resample_lc_to_cl(torch.cat([acc, gyro], dim=1), 6, imu_len, "right_imu")
    return emg, imu
```
首先明确mindrove.pt里面的数据结构是什么样的：右手 `mindrove.pt` 的原始字段为 `right_emg [L,8]`、`right_acc [L,3]`、`right_gyro [L,3]`。IMU 在通道维拼成 `[L,6]`。每个 clip 先线性重采样为 EMG `[8,512]`、IMU `[6,256]`。

因此该函数先读取一个mindrove.pt文件，然后将右手emg数据长度变化为"emg_len"，
对于acc 和 gyro数据，将他们沿着通道维度拼接起来，然后将其变化为"imu_len"长度。

该函数的输出是一个元组，元组里面的元素是两个张量，维度分别为[8, emg_len]和[6, imu_len]

```python
def _channel_stats(values: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    merged = torch.cat(values, dim=1).double()
    mean = merged.mean(dim=1).float()
    std = merged.std(dim=1, unbiased=False).float().clamp_min(1e-6)
    return mean, std
```
这是用于计算逐通道的均值和标准差。输入一个列表，里面的元素是每个样本的emg/imu张量，每个样本维度为[8, emg_len] 或 [6, imu_len]


```python
def build_signal_caches(
    dataset_root: str | Path,
    train_manifest: str | Path,
    test_manifest: str | Path,
    output_dir: str | Path,
    emg_length: int = 512,
    imu_length: int = 256,
    overwrite: bool = False,
) -> dict[str, str]:
    """Compute train-only normalization and cache standardized right-hand tensors."""
    dataset_root, output_dir = Path(dataset_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {split: output_dir / f"{split}_right_signals.pt" for split in ("train", "test")}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError(f"Signal cache already exists under {output_dir}")
    rows_by_split = {"train": read_jsonl(train_manifest), "test": read_jsonl(test_manifest)}
    raw: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    for split, rows in rows_by_split.items():
        emg_values, imu_values = [], []
        for index, row in enumerate(rows, 1):
            emg, imu = _load_right_signals(dataset_root, row, emg_length, imu_length)
            # emg: [8， 512]， imu: [6, 256]
            emg_values.append(emg)
            imu_values.append(imu)
            if index % 250 == 0:
                print(f"{split}: loaded {index}/{len(rows)}", flush=True)
        raw[split] = emg_values, imu_values
    emg_mean, emg_std = _channel_stats(raw["train"][0])
    imu_mean, imu_std = _channel_stats(raw["train"][1])
    # 计算emg和imu的逐通道均值和标准差
    stats = {
        "right_emg_mean": emg_mean.tolist(), "right_emg_std": emg_std.tolist(),
        "right_imu_mean": imu_mean.tolist(), "right_imu_std": imu_std.tolist(),
        "emg_target_length": emg_length, "imu_target_length": imu_length,
        "normalization_source": str(Path(train_manifest).resolve()),
    }
    for split, rows in rows_by_split.items():
        emg_values, imu_values = raw[split]
        emg = torch.stack([(x - emg_mean[:, None]) / emg_std[:, None] for x in emg_values])
        imu = torch.stack([(x - imu_mean[:, None]) / imu_std[:, None] for x in imu_values])
        torch.save({"records": rows, "right_emg": emg, "right_imu": imu, "stats": stats}, outputs[split])
    write_json(output_dir / "right_signal_stats.json", stats)
    return {key: str(value) for key, value in outputs.items()}
```
用训练集的数据计算emg和imu的逐通道均值和标准差，然后将标准化后的训练数据和测试数据保存成.pt文件。
torch.stack()会沿着新的维度进行拼接，所以会形成[num_samples, 8, emg_len]/[num_sample, 6, imu_len]的张量。

# data.py
```python
@dataclass(frozen=True)
class Example:
    current: str
    history: tuple[str, ...]
    row: dict[str, Any]
```
Example 数据类，他们都写名了类型注释，所以他们都是实例属性。

```python
def _index(records: list[dict[str, Any]]) -> dict[str, int]:
    result = {str(row["sample_name"]): index for index, row in enumerate(records)}
    if len(result) != len(records):
        raise ValueError("A cache contains duplicate sample_name values")
    return result
```
建立样本名"sample_name"和 index 的映射。

```python
def _zero_shift(signal: torch.Tensor, fraction: float) -> torch.Tensor:
    amount = int(round(signal.shape[-1] * fraction))
    # 最后一个维度是emg_len 或 imu_len, 假设amount变成了400
    if amount == 0:
        return signal
    shifted = torch.zeros_like(signal)
    # shifted维度会是[C, L]
    if amount > 0 and amount < signal.shape[-1]:
        shifted[..., amount:] = signal[..., :-amount]
    elif amount < 0 and -amount < signal.shape[-1]:
        shifted[..., :amount] = signal[..., -amount:]
    # 以amount=400为例，这里会取shifted [:, 400:到最后]共113个值，而signal [...,:-amout]会取0到-400也是113个值，
    # 但是[..., 0:400] 这写值都被设置为0了，被丢弃了。
    return shifted
```

```python
class MultimodalHistoryDataset(Dataset):
    def __init__(
        self,
        primary_cache: str | Path,
        secondary_cache: str | Path,
        signal_cache: str | Path,
        selection_manifest: str | Path,
        drop_modalities: tuple[str, ...] = (),
        sensor_offset_fraction: float = 0.0,
        emg_offset_fraction: float | None = None,
        imu_offset_fraction: float | None = None,
        training: bool = False,
        time_shift_augmentation_probability: float = 0.0,
        time_shift_augmentation_max_fraction: float = 0.0,
    ) -> None:
        self.primary = load_feature_cache(primary_cache)
        self.secondary = load_feature_cache(secondary_cache)
        self.signals = load_signal_cache(signal_cache)
        self.rows = read_jsonl(selection_manifest)
        self.drop_modalities = set(drop_modalities)
        self.emg_offset_fraction = float(sensor_offset_fraction if emg_offset_fraction is None else emg_offset_fraction)
        self.imu_offset_fraction = float(sensor_offset_fraction if imu_offset_fraction is None else imu_offset_fraction)
        self.training = bool(training)
        self.time_shift_augmentation_probability = float(time_shift_augmentation_probability)
        self.time_shift_augmentation_max_fraction = float(time_shift_augmentation_max_fraction)
        self.lookup = {
            "primary": _index(self.primary["records"]),
            "secondary": _index(self.secondary["records"]),
            "signal": _index(self.signals["records"]),
        }
        # 这里的值是一个字典，字典的键是样本名"sample_name", 值是该样本在records中的详细信息。
        sample_names = {str(row["sample_name"]) for row in self.rows}
        # 构建一个含有样本名称的集合
        for name, lookup in self.lookup.items():
            missing = sorted(sample_names - set(lookup))
            if missing:
                raise KeyError(f"{name} cache misses {len(missing)} samples, e.g. {missing[:5]}")
        # 要确保缓存中的样本名称包含读取的manifest的的样本名称，否则会有样本没有对应的特征数据
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in self.rows:
            grouped.setdefault((str(row["participant"]), str(row["run"])), []).append(row)
            # 将同一个参与人的同一个run的数据放到同一个列表中
        self.examples: list[Example] = []
        for run_rows in grouped.values():
            run_rows.sort(key=lambda value: int(value["annotation_row_index"]))
            # 将同一个人的同一个run的片段按照其标注的index也就是实际发生的顺序进行排序。
            for position, row in enumerate(run_rows):
                self.examples.append(Example(str(row["sample_name"]), tuple(
                    str(previous["sample_name"]) for previous in run_rows[:position]
                ), row))
                # 对于每一个样本当前的样本样本名，其之前的样本的样本名和当前row的数据保存下来。
        self.examples.sort(key=lambda item: (
            str(item.row["participant"]), str(item.row["run"]), int(item.row["annotation_row_index"])
        ))
        # 将example中的数据优先按参与人进行排序，同一个参与人按run排序，同一个run按annotation_row_index排序。

    def __len__(self) -> int:
        return len(self.examples)

    def _one(self, sample_name: str) -> dict[str, torch.Tensor]:
        p = self.lookup["primary"][sample_name]
        s = self.lookup["secondary"][sample_name]
        m = self.lookup["signal"][sample_name]
        # 取出对应的特征
        emg_offset, imu_offset = self.emg_offset_fraction, self.imu_offset_fraction
        if self.training and torch.rand(()) < self.time_shift_augmentation_probability:
            emg_offset += float(torch.empty(()).uniform_(
                -self.time_shift_augmentation_max_fraction, self.time_shift_augmentation_max_fraction
            ))
        if self.training and torch.rand(()) < self.time_shift_augmentation_probability:
            imu_offset += float(torch.empty(()).uniform_(
                -self.time_shift_augmentation_max_fraction, self.time_shift_augmentation_max_fraction
            ))
        emg = _zero_shift(self.signals["right_emg"][m].float(), emg_offset)
        imu = _zero_shift(self.signals["right_imu"][m].float(), imu_offset)
        return {
            "primary": self.primary["features"][p].float(),
            "secondary": self.secondary["features"][s].float(),
            "emg": emg,
            "imu": imu,
            "secondary_available": torch.tensor("secondary" not in self.drop_modalities),
            "emg_available": torch.tensor("emg" not in self.drop_modalities),
            "imu_available": torch.tensor("imu" not in self.drop_modalities),
        }
        # 返回一个字典，保存着没有shifted的信号值和shifted的信号值。

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        current = self._one(example.current)
        history = [self._one(sample_name) for sample_name in example.history]
        row = example.row
        return {
            "current": current, "history": history,
            "history_position_ids": torch.arange(len(history), 0, -1, dtype=torch.long),
            "node_target": int(row["node_idx"]) - 1,
            "tier3_target": int(row["tier3_id"]), "stage_id": int(row["stage_id"]),
            "sample_name": str(row["sample_name"]), "participant": str(row["participant"]),
            "run": str(row["run"]), "annotation_row_index": int(row["annotation_row_index"]),
        }
        # 返回一个字典， 而这里的current是字典，history是列表，里面的元素是字典。字典的内容就是self._one()返回的内容。
```

```python
def collate_multimodal(batch: list[dict[str, Any]]) -> dict[str, Any]:
    size = len(batch)
    max_history = max(len(item["history"]) for item in batch)
    current_keys = list(batch[0]["current"])
    # 这里得到["primary", "secondary", "emg", "imu", "secondary_available", "emg_available", "imu_available"]
    result: dict[str, Any] = {}
    for key in current_keys:
        result[f"current_{key}"] = torch.stack([item["current"][key] for item in batch])
        # 这里是直接将当前样本的"primary"的内容拼接起来维度变成[Batch_size, C, L]
        shape = tuple(batch[0]["current"][key].shape)
        # (C, L)
        history_tensor = torch.zeros((size, max_history, *shape), dtype=result[f"current_{key}"].dtype)
        # [batch_size, max_h_len, C, L]
        for row_index, item in enumerate(batch):
            if item["history"]:
                history_tensor[row_index, :len(item["history"])] = torch.stack( # 这里stack后维度为[h_len, C, L]
                    [entry[key] for entry in item["history"]]
                )
                # 完成后history_tensor 维度还是[batch_size, max_h_len, C, L], 但又历史信息的值会被填充，否则为0
        result[f"history_{key}"] = history_tensor
    result["history_position_ids"] = torch.zeros((size, max_history), dtype=torch.long)
    result["history_padding_mask"] = torch.ones((size, max_history), dtype=torch.bool)
    for row_index, item in enumerate(batch):
        length = len(item["history"])
        if length:
            result["history_position_ids"][row_index, :length] = item["history_position_ids"]
            result["history_padding_mask"][row_index, :length] = False
            #  有历史信息的位置的mask值设置为0表示此处不进行mask
    for key in ("node_target", "tier3_target", "stage_id"):
        result[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
    for key in ("sample_name", "participant", "run", "annotation_row_index"):
        result[key] = [item[key] for item in batch]
    return result
```
最后返回的result 是一个字典结构如下:
{
    "current_primary": [B, C, L],
    "history_primary": [B, max_h_len, C, L],
    "current_secondary": [B, C, L],
    "history_secondary": [B, max_h_len, C, L],
    "current_emg": [B, C, L],
    "history_emg": [B, max_h_len, C, L],
    "current_imu": [B, C, L],
    "history_imu": [B, max_h_len, C, L],
    "current_secondary_availabl" : [B],
    "history_secondary_availabl": [B, max_h_len]
    ...
    "history_position_ids": [B, max_h_len],
    "history_padding_mask": [B, max_h_len],
    "node_target": [B],
    "tier3_target": [B],
    ...
}