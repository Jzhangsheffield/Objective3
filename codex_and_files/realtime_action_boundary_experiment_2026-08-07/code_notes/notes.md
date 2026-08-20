# 项目理解梳理

## 阶段1：项目说明文件
浏览D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\realtime_action_boundary_experiment_2026-08-07下的README.md和FIRST_ROUND_RESULTS_ANALYSIS_2026-08-10.md文件。
浏览D:\Junxi_data\MULTISENSOR_DATA_COLLECTION_Stage2_structured_data\Action_Segmentation_Dataset\annotations\action_recognition_boundaries_with_background_v1下的README.md文件
浏览D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\realtime_action_boundary_experiment_2026-08-07\docs下的ANNOTATIONS.md，EXPERIMENT_PROTOCOL.md，VALIDATION_REPORT_2026-08-07.md文件。

## 阶段2：项目配置文件
### config_windows.bat
里面保存着项目路径，数据集路径，保存路径，使用多少个workers等，其将为base.json提供一些变量或者覆盖一些变量.

### base.json
里面保存着更加详细的配置信息，一些变量可以通过config_windows.bat进行配置。

### somke_stride4.json
里面保存着一些进行smoke训练所需要配置的变量。

### resolved_config.json
每个训练会有一个这个文件，其内保存着本次训练实际的变量配置。

## 阶段3：数据集manifest.jsonl
数据集manifest.jsonl文件保存着每一个样本的信息，包括该样本数据哪一个实验者的第几个run，其数据保存在哪个位置等。

比如:
```text
{
  "sample_name": "run_sample_000001",
  "participant": "A",
  "source_run": "run_1",
  "camera_dirs": {
    "001484412812": "raw/run_sample_000001/001484412812"
  }
}
```

## 阶段4：数据集标注和读取代码
### run_sample_000001_segmentation_annotation.csv
其内容如下：
```text
No	action	object	start_idx	end_idx	start	end	mark
1	background	none	1	1301	20260313_093615_031922	20260313_093658_214425	none
2	turn on	main switch	1302	1346	20260313_093658_251584	20260313_093659_751198	none
3	background	none	1347	1474	20260313_093659_783645	20260313_093703_977627	none
4	turn on	water pump	1475	1502	20260313_093704_009473	20260313_093704_906247	none
5	background	none	1503	1624	20260313_093704_938795	20260313_093708_967278	none
6	turn on	air compressor	1625	1655	20260313_093709_008309	20260313_093710_027171	none
7	background	none	1656	1755	20260313_093710_063859	20260313_093713_349850	none
8	turn on	extractor fan	1756	1785	20260313_093713_374461	20260313_093714_355978	none
9	background	none	1786	1909	20260313_093714_378917	20260313_093718_534421	none
```
是一种动作识别类型的标注，一条视频被分割成一段一段的，标注出每一段的动作，起止时间戳等信息。其中No表示该动作片段在整条视频中发生顺序。

### run_sample_000002_frame_annotation.csv
是run_sample_000001_segmentation_annotation.csv的一种拓展，其将分割的动作片段展开，为每一帧提供标注。
内容如下:
```text
frame_idx	original_frame_idx	frame_name	timestamp	action	object	mark	segment_no	segment_start_idx	segment_end_idx	segment_start	segment_end
1	693	20260313_094131_491632.jpg	20260313_094131_491632	background	none	none	1	1	722	20260313_094108_601837	20260313_094132_472343
2	694	20260313_094131_525542.jpg	20260313_094131_525542	background	none	none	1	1	722	20260313_094108_601837	20260313_094132_472343
3	695	20260313_094131_558963.jpg	20260313_094131_558963	background	none	none	1	1	722	20260313_094108_601837	20260313_094132_472343
4	696	20260313_094131_589882.jpg	20260313_094131_589882	background	none	none	1	1	722	20260313_094108_601837	20260313_094132_472343
5	697	20260313_094131_638282.jpg	20260313_094131_638282	background	none	none	1	1	722	20260313_094108_601837	20260313_094132_472343
6	698	20260313_094131_673182.jpg	20260313_094131_673182	background	none	none	1	1	722	20260313_094108_601837	20260313_094132_472343
7	699	20260313_094131_701107.jpg	20260313_094131_701107	background	none	none	1	1	722	20260313_094108_601837	20260313_094132_472343
8	700	20260313_094131_731029.jpg	20260313_094131_731029	background	none	none	1	1	722	20260313_094108_601837	20260313_094132_472343
9	701	20260313_094131_765933.jpg	20260313_094131_765933	background	none	none	1	1	722	20260313_094108_601837	20260313_094132_472343
10	702	20260313_094131_798948.jpg	20260313_094131_798948	background	none	none	1	1	722	20260313_094108_601837	20260313_094132_472343
11	703	20260313_094131_829865.jpg	20260313_094131_829865	background	none	none	1	1	722	20260313_094108_601837	20260313_094132_472343
12	704	20260313_094131_861783.jpg	20260313_094131_861783	background	none	none	1	1	722	20260313_094108_601837	20260313_094132_472343
13	705	20260313_094131_893692.jpg	20260313_094131_893692	background	none	none	1	1	722	20260313_094108_601837	20260313_094132_472343
14	706	20260313_094131_922614.jpg	20260313_094131_922614	background	none	none	1	1	722	20260313_094108_601837	20260313_094132_472343
15	707	20260313_094131_956524.jpg	20260313_094131_956524	background	none	none	1	1	722	20260313_094108_601837	20260313_094132_472343
16	708	20260313_094131_990433.jpg	20260313_094131_990433	background	none	none	1	1	722	20260313_094108_601837	20260313_094132_472343
```

### annotations.py
读取数据集标注csv代码。
``` python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .utils import read_csv, read_jsonl
```
导入包
其中read_csv和read_jsonl要说明一下，read_csv代码如下: 
```text
def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))
```
这里使用了csv.DictReader()可以将每一行作为一个字典，键默认来自第一行表头。
而csv.reader()则每一行返回一个list。
这里read_csv()返回一个列表，列表里面元素为字典，每个字典是csv文件中的一行。
```text
def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
```
返回一个列表，列表里面元素是字典，每一个字典的内容就是文件每一行的内容。

```python
@dataclass(frozen=True)
class RunInfo:
    sample_name: str
    participant: str
    source_run: str
    raw_dir: Path
    camera_dir: Path
    frame_annotation: Path
```
dataclass 类。

```python
def load_run_index(dataset_root: str | Path, annotation_root: str | Path, camera_id: str) -> dict[str, RunInfo]:
    dataset_root = Path(dataset_root)
    annotation_root = Path(annotation_root)
    result: dict[str, RunInfo] = {}
    for row in read_jsonl(dataset_root / "manifest.jsonl"):
        # 遍历manifest中的row
        sample_name = str(row["sample_name"])
        camera_rel = row.get("camera_dirs", {}).get(camera_id, f"raw/{sample_name}/{camera_id}")
        # 获得每个row中样本的名字，以及样本RGB数据的保存路径，如果找不到键就返回一个空的列表，再在保存路径中找对应相机id的数据，如果找不到对应相机的camera_id，则返回f"raw/{sample_name}/{camera_id}。
        result[sample_name] = RunInfo(
            sample_name=sample_name,
            participant=str(row["participant"]),
            source_run=str(row["source_run"]),
            raw_dir=dataset_root / str(row["raw_dir"]),
            camera_dir=dataset_root / str(camera_rel),
            frame_annotation=annotation_root / f"{sample_name}_frame_annotation.csv",
        )
        # 将一个样本数据打包成一个dataclass, 添加到字典中。
    return result
```
返回一个字典，字典的键是manifest中的"sample_name", 值是包含该样本后续所需信息的dataclass实例。

```python
def load_frame_table(info: RunInfo) -> dict[str, Any]:
    rows = read_csv(info.frame_annotation)
    # 读取一个run的逐帧标注csv
    if not rows:
        raise ValueError(f"Empty frame annotation: {info.frame_annotation}")
    # 空的则报错
    frame_paths = [info.camera_dir / row["frame_name"] for row in rows]
    # 获得该run每一帧的路径
    missing = [str(path) for path in frame_paths if not path.is_file()]
    # 检测每一帧的路径是否为一个文件
    if missing:
        raise FileNotFoundError(f"{len(missing)} annotated frames missing; examples={missing[:3]}")
    # 如果列表不是空的，则报错说明有些帧是缺失的。
    is_action = np.asarray([row["action"].strip().lower() != "background" for row in rows], dtype=np.int64)
    # 将每一帧是否是动作帧添加到数组中
    original_idx = np.asarray([int(row["original_frame_idx"]) for row in rows], dtype=np.int64)
    # 将每一帧的"original_frame_idx"添加到数组
    frame_idx = np.asarray([int(row["frame_idx"]) for row in rows], dtype=np.int64)
    # 将每一帧的"frame_idx"添加到数组
    if not np.all(np.diff(frame_idx) == 1):
        raise ValueError(f"frame_idx is not continuous in {info.frame_annotation}")
    if not np.all(np.diff(original_idx) > 0):
        raise ValueError(f"original_frame_idx is not strictly increasing in {info.frame_annotation}")
    # np.diff()计算相邻元素插值，保证帧的连续和递增。
    action = [row["action"] for row in rows]
    obj = [row["object"] for row in rows]
    segment_no = np.asarray([int(row["segment_no"]) for row in rows], dtype=np.int64)
    # 获得每一帧所属的action, object, 以及segment 列表。
    starts = np.zeros(len(rows), dtype=np.float32)
    ends = np.zeros(len(rows), dtype=np.float32)
    for i in range(len(rows)):
        if is_action[i] and (i == 0 or not is_action[i - 1] or segment_no[i] != segment_no[i - 1]):
            starts[i] = 1.0
            # 如果一帧是动作帧并且(其是第一帧，或者其前一帧不是动作帧，或者其和前一帧不在一个片段中)
            # 那这一帧是其实帧。jiangq
        if is_action[i] and (i == len(rows) - 1 or not is_action[i + 1] or segment_no[i] != segment_no[i + 1]):
            ends[i] = 1.0
            # 如果一帧是动作并且（其是最后一帧，或者其后一帧不是动作，或者其后一帧和他不在一个片段中）
            # 那这一帧是结束帧，将其设置为1
    return {
        "rows": rows,
        "frame_paths": frame_paths,
        "frame_idx": frame_idx,
        "original_frame_idx": original_idx,
        "timestamps": [row["timestamp"] for row in rows],
        "state": is_action,
        "start": starts,
        "end": ends,
        "action": action,
        "object": obj,
        "segment_no": segment_no,
    }
```
返回一个字典，字典中的值都是对应每一帧的信息。


## 阶段5：生成Leave-one-subject-out实验协议
### prepare_protocols.py
```python
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from boundary_experiment.annotations import load_run_index
from boundary_experiment.config import load_config
from boundary_experiment.protocols import prepare_boundary_protocols
```
其中的load_config和prepare_boundary_protocols代码如下:
```text
def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    cfg = read_json(path)
    if "extends" in cfg:
        parent = Path(cfg.pop("extends"))
        # cfg中含有extends, 则将extends的内容复制给parent, 并将其从cfg中删除。
        if not parent.is_absolute():
            parent = path.parent / parent
            # 如果parent不是绝对路径，就用path的父路径+parent构成新的路径。
            # 这里extends里保存的是另一个文件的路径，可能是绝对路径，也可能是相对路径，但是这个文件和一开始的json文件是在同一个文件夹中的，也就是通过这样将parent 变成了extends中文件的绝对路径。
        cfg = _deep_update(load_config(parent), cfg)
        # 递归的内容进行更新
    missing = REQUIRED_SECTIONS - set(cfg)
    # 检查更新后的cfg文件键的内容是否满足要求。
    if missing:
        raise ValueError(f"Config missing sections: {sorted(missing)}")
        # 不满足要求则报错。
    return copy.deepcopy(_expand_environment(cfg))

def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
        # 如果要覆盖的cfg的值是字典，并且base的键值对也是字典
            _deep_update(base[key], value)
        # 则继续进行递归，直到两个字典对应的值都不是字典
        else:
            base[key] = value
            # 使用cfg的值更新base的值。或者添加新的键值对。
    return base

def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "%RAB_" in expanded:
            raise EnvironmentError(
                f"Unresolved Windows config variable in {value!r}. "
                "Run through scripts\\*.bat or CALL config_windows.bat first."
            )
        return expanded
    return value
```
这一部分的内容所含的思想十分高级。
首先.resolve()用法用于将Path路径对象转变为统一规范的绝对路径。其会自动去处理..或者.等内容，将路径变成绝对路径。
.is_absolute()用于判断路径是否是绝对路径。
copy.deepcopy()常用于复制嵌套含有可变对象的变量。 比如一个字典a，里面嵌套一个列表。如果使用"="号，那那字典会增加一个指向b. 现在a和b都指向同一个字典。如果使用copy.copy()进行浅拷贝，那b会指向一个新的字典，但是新旧字典里面保存的都是同一个列表的引用，因此如果改动内部列表，则两个字典都会发生变化。使用copy.deepcopy()则会完全复制所有内容，两个字典变得完全独立。
.expandvars()把字符串中的环境变量替换成操作系统中对应的值。比如os.environ["RAB_DATA"] = r"D:\dataset"， value = r"%RAB_DATA%\train"， os.path.expandvars(value)， 得到D:\dataset\train。

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Translate existing Atomic-tail LOSO protocols to continuous runs")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    runs = load_run_index(cfg["paths"]["dataset_root"], cfg["paths"]["annotation_root"], cfg["data"]["camera_id"])
    report = prepare_boundary_protocols(
        runs, cfg["paths"]["atomic_project_root"], cfg["paths"]["protocol_root"],
        cfg["data"]["camera_id"], list(cfg["data"]["participants"]),
    )
    print(f"Prepared {len(report['folds'])} LOSO folds at {cfg['paths']['protocol_root']}")


if __name__ == "__main__":
    main()
```
### protocols.py
```python
def _run_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(str(row["participant"]), str(row.get("run", row.get("source_run")))) for row in rows}


def prepare_boundary_protocols(
    run_index: dict[str, RunInfo],
    atomic_project_root: str | Path,
    output_root: str | Path,
    camera_id: str,
    participants: list[str],
) -> dict[str, Any]:
    atomic_project_root = Path(atomic_project_root)
    output_root = Path(output_root)
    by_key = {(info.participant, info.source_run): info for info in run_index.values()}
    # run_index是一个字典，字典的键是"sample_name"，只是该样本对应的RunInfo dataclass
    # by_key也是一个字典，键是有"participant和source_run"组成的元组，值还是该样本也可以称为该run对应的RunInfo dataclass
    report: dict[str, Any] = {"folds": {}}
    for heldout in participants:
        source_root = atomic_project_root / "outputs" / f"{heldout}_as_test" / f"cam_{camera_id}" / "protocols"
        report["folds"][heldout] = {}
        # 对于每一个参与人员，将其拿出来作为测试集。
        for scope in ("normal_only", "all_runs"):
            scope_report: dict[str, Any] = {}
            split_keys: dict[str, set[tuple[str, str]]] = {}
            for split in ("train", "test_normal", "test_fault", "test_all"):
                source = source_root / scope / f"{split}.jsonl"
                # 将对应分割的.jsonl文件读取
                keys = _run_keys(read_jsonl(source))
                split_keys[split] = keys
                missing = sorted(keys - set(by_key))
                # 确保分割中的run没有缺失
                if missing:
                    raise KeyError(f"Structured dataset is missing run keys from {source}: {missing}")
                rows = [
                    {
                        "sample_name": by_key[key].sample_name,
                        "participant": key[0],
                        "source_run": key[1],
                        "split": split,
                        "train_scope": scope,
                        "heldout_participant": heldout,
                    }
                    for key in sorted(keys)
                ]
                target = output_root / f"{heldout}_as_test" / scope / f"{split}.jsonl"
                write_jsonl(target, rows)
                scope_report[split] = {"runs": len(rows), "path": str(target), "source": str(source)}
            if any(participant == heldout for participant, _ in split_keys["train"]):
                raise ValueError(f"LOSO leakage: held-out {heldout} appears in {scope} train")
            if any(participant != heldout for participant, _ in split_keys["test_all"]):
                raise ValueError(f"Non-held-out run appears in {heldout}/{scope} test_all")
            if split_keys["train"] & split_keys["test_all"]:
                raise ValueError(f"Train/test run overlap in {heldout}/{scope}")
            if split_keys["test_normal"] & split_keys["test_fault"]:
                raise ValueError(f"Normal/fault test overlap in {heldout}/{scope}")
            if split_keys["test_normal"] | split_keys["test_fault"] != split_keys["test_all"]:
                raise ValueError(f"Normal+fault does not equal test_all in {heldout}/{scope}")
            report["folds"][heldout][scope] = scope_report
    write_json(output_root / "protocol_report.json", report)
    return report


def load_protocol_runs(protocol_path: str | Path) -> list[str]:
    return [str(row["sample_name"]) for row in read_jsonl(protocol_path)]
```
暂时跳过，用于Leave one subject out的实验配置协议。


### extract_boundary_features.py
```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Extract causal RGB window features for one LOSO condition")
    parser.add_argument("--config", required=True)
    parser.add_argument("--heldout", required=True, choices=["A", "D", "J", "M"])
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--scope", required=True, choices=["normal_only", "all_runs"])
    parser.add_argument("--splits", nargs="+", default=["train", "test_all"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["training"]["device"])
    checkpoint = format_path(cfg["paths"]["backbone_checkpoint_template"], heldout=args.heldout, seed=args.seed, scope=args.scope)
    model, load_report = build_frozen_backbone(cfg["paths"]["atomic_project_root"], checkpoint, device)
    cache_root = format_path(
        cfg["paths"]["feature_cache_template"], heldout=args.heldout, seed=args.seed,
        scope=args.scope, stride=cfg["features"]["stride_frames"],
    )
    run_index = load_run_index(cfg["paths"]["dataset_root"], cfg["paths"]["annotation_root"], cfg["data"]["camera_id"])
    names: set[str] = set()
    for split in args.splits:
        protocol = Path(cfg["paths"]["protocol_root"]) / f"{args.heldout}_as_test" / args.scope / f"{split}.jsonl"
        names.update(load_protocol_runs(protocol))
    records = {}
    for position, name in enumerate(sorted(names), 1):
        output = cache_root / f"{name}.pt"
        if output.is_file() and not args.overwrite:
            print(f"[{position}/{len(names)}] skip {name}")
            continue
        print(f"[{position}/{len(names)}] extract {name}")
        records[name] = extract_run_features(run_index[name], model, output, device, cfg["features"], checkpoint)
    write_json(cache_root / "extraction_manifest.json", {"load_report": load_report, "runs": records})


if __name__ == "__main__":
    main()
```
这个脚本是提取特征的入口，特征提取流程在features.py文件中。
这个文件的关键是明确各个变量的数据结构和保存的内容。
cfg主要是读取的D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\realtime_action_boundary_experiment_2026-08-07\configs\base.json的内容。

device: 
默认是"auto"
checkpoint: 
```text
def format_path(template: str, **values: Any) -> Path:
    return Path(template.format(**values))
```
cfg["paths"]["backbone_checkpoint_template"] 是这样的："%RAB_FEATURE_CACHE_ROOT%\\{heldout}_as_test\\{scope}\\seed_{seed}\\stride_{stride}"
，其路径中{}里面的内容使用heldout=args.heldout, seed=args.seed, scope=args.scope进行填充。
model和load_report:
在后续的features.py中进行讲解。
cache_root:
同样也是一个路径，和checkpoint格式类似
run_index:
获得一个字典，字典结构如下: dict[str, RunInfo], 其键是样本的名称"sample_name"，值是RunInfo数据类。
protocol:
也是一个是一个列表，里面的元素是一个一个的字典，字典保存着类似如下的内容
```text
{
"sample_name": "run_sample_000001",
"participant": "A",
"source_run": "run_1",
"split": "test_all",
"train_scope": "all_runs",
"heldout_participant": "A"
}
```
names:
是一个集合，
```text
def load_protocol_runs(protocol_path: str | Path) -> list[str]:
    return [str(row["sample_name"]) for row in read_jsonl(protocol_path)]
```
所以names是一个由"sample_name"组成的集合
records:
是一个字典，其里面的内容在features.py中进行详解。

## features.py
### class RGBFrameDataset(Dataset):
```python
class RGBFrameDataset(Dataset):
    """Decode and normalize each source frame exactly once, in chronological order."""

    def __init__(self, frame_paths: list[Path], size: int, mean: list[float], std: list[float]):
        self.frame_paths = frame_paths
        self.size = int(size)
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = read_image(str(self.frame_paths[index])).float().div_(255.0)
        image = resize(image, [self.size, self.size], antialias=True)
        return (image - self.mean) / self.std
```
这是一个RGB帧数据集，其主要作用就是读取一帧RGB图片将其进行归一化。

```python
def _import_atomic_modules(project_root: str | Path):
    root = str(Path(project_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from graph_history.backbone import generate_model
    from graph_history.utils import load_compatible_state
    return generate_model, load_compatible_state
```
其作用就是获得生成模型和载入模型权重的函数

```python
def build_frozen_backbone(project_root: str | Path, checkpoint: str | Path, device: torch.device):
    generate_model, load_compatible_state = _import_atomic_modules(project_root)
    model = generate_model(18, num_classes=31)
    report = load_compatible_state(model, checkpoint)
    if report["missing_keys"] or report["unexpected_keys"] or report["loaded_keys"] != report["model_keys"]:
        raise RuntimeError(f"Backbone checkpoint is not an exact architecture match: {report}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, report
```
初始化模型并载入预训练的权重，并将模型送到指定设置比，变成evaluation模式，将参数冻结。
返回的report则是预训练权重文件，和模型不匹配的模块信息。

```python
def _load_rgb(path: Path, size: int, mean: list[float], std: list[float]) -> torch.Tensor:
    image = read_image(str(path)).float().div_(255.0)
    image = resize(image, [size, size], antialias=True)
    mean_t = torch.tensor(mean, dtype=image.dtype)[:, None, None]
    std_t = torch.tensor(std, dtype=image.dtype)[:, None, None]
    return (image - mean_t) / std_t
```
该函数用于读取一帧，将其缩放到指定大小并归一化。

```python
def causal_clip_indices(anchor: int, clip_frames: int) -> list[int]:
    return [max(0, anchor - clip_frames + 1 + offset) for offset in range(clip_frames)]
```
目前没用先空着。

```python
def _events_on_anchor_grid(exact_target, anchors: list[int]) -> torch.Tensor:
    result = torch.zeros(len(anchors), dtype=torch.float32)
    anchor_tensor = torch.tensor(anchors)
    for event in torch.from_numpy(exact_target).nonzero().flatten():
        candidates = (anchor_tensor >= event).nonzero().flatten()
        result[int(candidates[0] if len(candidates) else len(anchors) - 1)] = 1.0
    return result
```
这个函数很关键，用于如果采用stride取帧的话，将原本的逐帧标注的时间映射到stride的时间轴上。
一般exact_target是逐帧的标注，比如[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
而anchor则是stride取的帧号[0, 4, 8, 12, 16]
要解决的是实际在第6帧由各动作，那应该映射到stride序列的第几帧上呢？
torch.from_numpy(exact_target).nonzero().flatten(): 中的nonzero()返回非0值的索引所以这里返回的是[6, 12]
(anchor_tensor >= event).nonzero().flatten() 这里先是[False，False, True, True, True], 然后nonzero()得到[2, 3, 4]
result[int(candidates[0] if len(candidates) else len(anchors) - 1)] = 1.0, 如果candidates不为空，则将其第一个元素代表的索引设置为1.
也就是8对应的那个位置2.


```python
@torch.inference_mode()
def extract_closed_segment_feature(
    frame_paths: list[Path], start_row: int, end_row: int, model: torch.nn.Module,
    device: torch.device, feature_cfg: dict[str, Any],
) -> torch.Tensor:
    """Extract the M3-compatible 16-frame feature only after a segment has ended."""
    if end_row < start_row:
        raise ValueError("end_row must be >= start_row")
    count = int(feature_cfg["clip_frames"])
    positions = torch.linspace(start_row, end_row, count).round().long().tolist()
    frames = [
        _load_rgb(frame_paths[index], int(feature_cfg["rgb_size"]), list(feature_cfg["mean"]), list(feature_cfg["std"]))
        for index in positions
    ]
    clip = torch.stack(frames, dim=1).unsqueeze(0).to(device)
    return model.forward_features(clip)[0].cpu()
```

```python
@torch.inference_mode()
def extract_run_features(
    info: RunInfo,
    model: torch.nn.Module,
    output_path: str | Path,
    device: torch.device,
    feature_cfg: dict[str, Any],
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    table = load_frame_table(info)
    # 载入一个run的RunInfo数据类
    stride = int(feature_cfg["stride_frames"])
    # 从cfg文件中读取的对于正常是1，对于somketest是4
    anchors = list(range(0, len(table["frame_paths"]), stride))
    # 一个run包含的一个列表，从0到一个run包含的帧数，间隔stride采样。
    batch_size = int(feature_cfg["batch_size"])
    clip_frames = int(feature_cfg["clip_frames"])
    num_workers = int(feature_cfg.get("num_workers", 0))
    frame_loader_batch_size = int(feature_cfg.get("frame_loader_batch_size", max(batch_size, 1)))
    pin_memory = bool(feature_cfg.get("pin_memory", True)) and device.type == "cuda"
    loader_kwargs: dict[str, Any] = {
        "batch_size": frame_loader_batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    # 提取出一些训练配置
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(feature_cfg.get("prefetch_factor", 2))
        loader_kwargs["persistent_workers"] = bool(feature_cfg.get("persistent_workers", True))
    frame_loader = DataLoader(
        RGBFrameDataset(
            table["frame_paths"], int(feature_cfg["rgb_size"]),
            list(feature_cfg["mean"]), list(feature_cfg["std"]),
        ),
        **loader_kwargs,
    )
    # 初始化一个RGB帧数据集

    features: list[torch.Tensor] = []
    rolling: deque[torch.Tensor] = deque(maxlen=clip_frames)
    # 一个双端队列
    pending_clips: list[torch.Tensor] = []

    def flush_pending() -> None:
        if not pending_clips:
            return
        batch = torch.stack(pending_clips)
        if pin_memory and not batch.is_pinned():
            batch = batch.pin_memory()
        features.append(model.forward_features(batch.to(device, non_blocking=pin_memory)).cpu())
        pending_clips.clear()

    row_index = 0
    for frame_batch in frame_loader:
        for frame in frame_batch:
            if not rolling:
                rolling.extend([frame] * clip_frames)
            # 如果rolling 为空，则用frame这一帧将其填充
            else:
                rolling.append(frame)
            # 如果不为空，则将该帧放在最后。
            # 这里dequeue的特性，当达到最大容量后，左侧填充帧，则最右侧帧会被弹出。
            if row_index % stride == 0:
                pending_clips.append(torch.stack(list(rolling), dim=1))
                # 只有符合stride位置的帧的数据被添加到pending_clips
                # 这里使用stack会创建一个新的维度并拼接，所以拼接后维度为[c, t, h, w]
                if len(pending_clips) >= batch_size:
                    flush_pending()
                    # 当pending_clips里面含有足够的片段后，统一使用flush_pending()提取特征
            row_index += 1
    flush_pending()
    # 将最后可能剩余的片段进行特征提取
    if row_index != len(table["frame_paths"]):
        raise RuntimeError(f"Frame loader returned {row_index} frames, expected {len(table['frame_paths'])}")
    # 判断是否遍历的所有的帧。

    feature_tensor = torch.cat(features, dim=0)
    # feature中是一个一个的张量，维度为[b, feat_dim], 所以沿着第一维度进行拼接。
    if feature_tensor.shape[0] != len(anchors):
        raise RuntimeError(f"Extracted {feature_tensor.shape[0]} anchors, expected {len(anchors)}")
    # 如果拼接后第一维的大小和要提取的stride帧数量不一样则报错。
    anchor_tensor = torch.tensor(anchors, dtype=torch.long)
    # 0, 4, 8, 12, 16
    radius_frames = int(feature_cfg.get("boundary_label_radius_frames", 0))
    # 默认值是2
    radius_anchors = (radius_frames + stride - 1) // stride
    # (2 + 4 -1 )// 4, 结果是1
    anchor_start = _events_on_anchor_grid(table["start"], anchors)
    anchor_end = _events_on_anchor_grid(table["end"], anchors)
    # 获得新stride时间轴上每一帧对应的事件起止。
    payload = {
        "sample_name": info.sample_name,
        "participant": info.participant,
        "source_run": info.source_run,
        "features": feature_tensor,
        "anchor_row_index": anchor_tensor,
        "frame_idx": torch.from_numpy(table["frame_idx"][anchors]),
        "original_frame_idx": torch.from_numpy(table["original_frame_idx"][anchors]),
        "timestamps": [table["timestamps"][index] for index in anchors],
        "state": torch.from_numpy(table["state"][anchors]).long(),
        "start": torch.from_numpy(dilate_binary_targets(anchor_start.numpy(), radius_anchors)),
        "end": torch.from_numpy(dilate_binary_targets(anchor_end.numpy(), radius_anchors)),
        "exact_start": anchor_start,
        "exact_end": anchor_end,
        "action": [table["action"][index] for index in anchors],
        "object": [table["object"][index] for index in anchors],
        "segment_no": torch.from_numpy(table["segment_no"][anchors]).long(),
        "metadata": {
            "causal": True,
            "clip_frames": clip_frames,
            "stride_frames": stride,
            "rgb_size": int(feature_cfg["rgb_size"]),
            "feature_dim": int(feature_tensor.shape[1]),
            "extract_batch_size": batch_size,
            "frame_loader_batch_size": frame_loader_batch_size,
            "frame_loader_num_workers": num_workers,
            "frame_loader_prefetch_factor": int(feature_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None,
            "frame_loader_pin_memory": pin_memory,
            "backbone_checkpoint": str(Path(checkpoint_path).resolve()),
            "backbone_checkpoint_sha256": sha256_file(checkpoint_path),
            "annotation_file": str(info.frame_annotation.resolve()),
            "annotation_sha256": sha256_file(info.frame_annotation),
            "available_frame_count": len(table["rows"]),
            "anchor_count": len(anchors),
        },
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return payload["metadata"]
    # 将一个run提取的特征保存下来。
```

## 阶段7：检查feature cache
### 一个run_sample_000001.pt内包含的内容
```python
import torch

run_1_sample_data = torch.load(r"D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\realtime_action_boundary_experiment_2026-08-07\cache\features\A_as_test\all_runs\seed_1\stride_4\run_sample_000001.pt",
                               map_location="cpu")

for key, value in run_1_sample_data.items():
    if hasattr(value, "shape"):
        print(f"{key}: {value.shape}")
    elif isinstance(value, str):
        print(f"{key}: {value}")
    else:
        print(f"{key}: {len(value)}")

print(run_1_sample_data["state"][:10])
print(run_1_sample_data["start"][:10])
print(run_1_sample_data["end"][:10])
print(run_1_sample_data["timestamps"][:10])
```
通过以上代码可以获得其.pt的结构和一些输出如下:
```text
sample_name: run_sample_000001
participant: A
source_run: run_1
features: torch.Size([1182, 512])
anchor_row_index: torch.Size([1182])
frame_idx: torch.Size([1182])
original_frame_idx: torch.Size([1182])
timestamps: 1182
state: torch.Size([1182])
start: torch.Size([1182])
end: torch.Size([1182])
exact_start: torch.Size([1182])
exact_end: torch.Size([1182])
action: 1182
object: 1182
segment_no: torch.Size([1182])
metadata: 11

tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
tensor([0., 0., 0., 0., 0., 0., 0., 1., 1., 1.])
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
['20260313_093657_246168', '20260313_093657_369124', '20260313_093657_497449', '20260313_093657_651785', '20260313_093657_782819', '20260313_093657_901520', '20260313_093658_046423', '20260313_093658_180996', '20260313_093658_303230', '20260313_093658_445744']
```

## 阶段8 理解Boundary训练样本如何切chunk
### data.py
```python
class BoundaryChunkDataset(Dataset):
    def __init__(self, cache_root: str | Path, run_names: list[str], chunk_length: int, chunk_overlap: int):
        self.cache_root = Path(cache_root)
        self.chunk_length = int(chunk_length)
        step = self.chunk_length - int(chunk_overlap)
        # 默认chunk_length = 256
        # chunk_overlap = 124
        # step = 132
        if step <= 0:
            raise ValueError("chunk_overlap must be smaller than chunk_length")
        self.caches = {name: load_feature_cache(self.cache_root / f"{name}.pt") for name in run_names}
        # 载入提取的每一个run的特征.pt文件
        # 这里键是样本的名称"sample_names", 值是一个字典（和阶段7一样的）
        self.index: list[tuple[str, int, int]] = []
        for name in run_names:
            length = int(self.caches[name]["features"].shape[0])
            # 返回一个run有多少帧
            for start in range(0, length, step):
                end = min(length, start + self.chunk_length)
                self.index.append((name, start, end))
                if end == length:
                    break

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        name, start, end = self.index[index]
        cache = self.caches[name]
        return {
            "sample_name": name,
            "start_offset": start,
            "features": cache["features"][start:end].float(),
            "state": cache["state"][start:end].long(),
            "start": cache["start"][start:end].float(),
            "end": cache["end"][start:end].float(),
        }
```
返回一个切分的chunk样本

```python 
def collate_chunks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # 这里rows里面的dict就是上面数据集返回的一个字典。
    max_len = max(row["features"].shape[0] for row in rows)
    # 获得最长的chunk的长度
    feature_dim = rows[0]["features"].shape[1]
    # 获得特征维度
    batch = len(rows)
    features = torch.zeros(batch, max_len, feature_dim)
    state = torch.zeros(batch, max_len, dtype=torch.long)
    start = torch.zeros(batch, max_len)
    end = torch.zeros(batch, max_len)
    mask = torch.zeros(batch, max_len, dtype=torch.bool)
    # 创建好张量方面后面进行填充
    for i, row in enumerate(rows):
        length = row["features"].shape[0]
        features[i, :length] = row["features"]
        state[i, :length] = row["state"]
        start[i, :length] = row["start"]
        end[i, :length] = row["end"]
        mask[i, :length] = True
    # 使用真实值进行填充
    return {
        "features": features,
        "state": state,
        "start": start,
        "end": end,
        "mask": mask,
        "sample_name": [row["sample_name"] for row in rows],
        "start_offset": [row["start_offset"] for row in rows],
    }
```
其最终返回一个字典，这个字典中 "features" 维度为[batch_size, max_len, feature_dim], "state", "start", "end", "mask"等维度为[batch, max_len]
```text
features [B,L,512]
state    [B,L]
start    [B,L]
end      [B,L]
mask     [B,L]
```

## 阶段9：理解Causal Boundary TCN
### models.py
```python
class CausalConv1d(nn.Conv1d):
    # 直接继承 nn.Conv1d
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.left_padding = self.dilation[0] * (self.kernel_size[0] - 1)
        # 输入维度是[B, D, T] 其中 d=512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_padding, 0)))
```
这里首先F.pad(x, (self.left_padding, 0))表示，对x的最后一个维度进行补0，左侧补self.left_padding个0， 右侧不补0.
为什么补self.left_padding = self.dilation[0] * (self.kernel_size[0] - 1) 个0，就变成casual?
```text
首先如果不补0，则比如输入 x1, x2, x3, ...
然后卷积第一个时间步对应的是x1, x2, x3相当如看了未来信息，再补0后变成 0, 0, x1, x2, x3, ...
则第一个时间不卷积对应的是0, 0, x1，这样第一个卷积就看到x1，没有使用未来信息。

因为直接继承自nn.Conv1d, 所以self.dilation都会在__init__()中创建好，并且回将其变成元组，也就是(2, )而不是直接的2。这也就是为什么用self.dilation[0]的原因。
```

```python
class TimewiseLayerNorm(nn.Module):
    """Normalize channels independently at each time step, preserving causality."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)
```
创建一个LayerNorm层，先将维度转置[B, D, T] -> [B, T, D], 然后对channel维度进行layernorm

```python
class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = TimewiseLayerNorm(channels)
        self.norm2 = TimewiseLayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        x = self.dropout(F.gelu(self.norm2(self.conv2(x))))
        return x + residual
```
```text
                            x ———————————————
                            |               |  
                            |               |
                        CasualConv1d        |
                            |               |
                            |               |   
                        LayerNorm           |
                            |               |
                            |               |
                          gelu              |
                            |               |
                            |               |
                        dropout             |
                            |               |
                            |               |
                        CasualConv1d        |
                            |               |
                            |               |
                        LayerNorm           |
                            |               |
                            |               |
                          gelu              |
                            |               |
                            |               |
                        dropout——————————————
```

```python
class CausalBoundaryTCN(nn.Module):
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(feature_dim, hidden_dim, 1) # in, out, kernel_size
        # 先将特征从512映射到256
        self.blocks = nn.ModuleList(
            CausalResidualBlock(hidden_dim, kernel_size, 2**layer, dropout)
            for layer in range(num_layers)
        )
        # 构建5个CausalResidualBlock
        self.state_head = nn.Conv1d(hidden_dim, 2, 1)
        self.boundary_head = nn.Conv1d(hidden_dim, 2, 1)
        # 构建状态头和分类头

    @property
    def receptive_field_steps(self) -> int:
        total = 1
        for block in self.blocks:
            total += 2 * block.conv1.left_padding
        return total

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(f"Expected [B,L,D], got {tuple(features.shape)}")
        x = self.input_projection(features.transpose(1, 2))
        # 维度 [B, T, D] -> [B, D, T]
        for block in self.blocks:
            x = block(x)
        state = self.state_head(x).transpose(1, 2)
        boundary = self.boundary_head(x).transpose(1, 2)
        # 维度又变回 [B, T, D] 此时D变成了2
        return {"state_logits": state, "start_logits": boundary[..., 0], "end_logits": boundary[..., 1]}
```
最后boundary[..., 0]...表示前面全取，最后一个维度取第一个，表示每一帧属于start 的概率，而"end_logits": boundary[..., 1]则表示每一帧属于end的概率。

```python
def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
    positive_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["mask"]
    state_loss = F.cross_entropy(outputs["state_logits"][mask], batch["state"][mask])
    start_loss = F.binary_cross_entropy_with_logits(
        outputs["start_logits"][mask], batch["start"][mask],
        pos_weight=torch.tensor(positive_weights["start"], device=mask.device),
    )
    end_loss = F.binary_cross_entropy_with_logits(
        outputs["end_logits"][mask], batch["end"][mask],
        pos_weight=torch.tensor(positive_weights["end"], device=mask.device),
    )
    total = weights["state"] * state_loss + weights["start"] * start_loss + weights["end"] * end_loss
    return total, {"loss": float(total.detach()), "state_loss": float(state_loss.detach()), "start_loss": float(start_loss.detach()), "end_loss": float(end_loss.detach())}
```
batch["state"][mask]：由0，1组成，0表示不是动作是background, 1表示是动作
batch["start"][mask],batch["end"][mask]：同样由0，1组成，0表示是开始或者结束，1表示不是开始或者结束。
F.binary_cross_entropy_with_logits(...)自动将logits经过sigmoid后再进行损失计算。
而F.cross_entropy() 则自动计算softmax后再进行损失计算。
F.binary_cross_entropy_with_logits(...)自动将logits经过sigmoid后再进行损失计算。中传入pos_weight会给是开始或或者结束的帧在计算损失时增大权重，因为一个视频中，开始和结束帧远远少于非开始和结束帧，所以不适用该权重，模型会直接将所有帧拟合为非开始或者结束帧，就能获得高的准确率。
返回损失

## 阶段10：理解训练循环与best checkpoint
### engine.py
```python
def build_model(cfg: dict[str, Any]) -> CausalBoundaryTCN:
    return CausalBoundaryTCN(
        feature_dim=int(cfg["feature_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        kernel_size=int(cfg["kernel_size"]),
        dropout=float(cfg["dropout"]),
    )
```
构建模型

```python 
def make_loader(cache_root: Path, runs: list[str], cfg: dict[str, Any], shuffle: bool) -> DataLoader:
    dataset = BoundaryChunkDataset(
        cache_root, runs, int(cfg["chunk_length_steps"]), int(cfg["chunk_overlap_steps"]),
    )
    return DataLoader(
        dataset, batch_size=int(cfg["batch_size"]), shuffle=shuffle,
        num_workers=int(cfg["num_workers"]), collate_fn=collate_chunks,
        pin_memory=torch.cuda.is_available(),
    )
```
构建数据集和DataLoader

```python
def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        tensors = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
        with torch.set_grad_enabled(training):
            outputs = model(tensors["features"])
            loss, values = compute_loss(
                outputs, tensors, loss_cfg["weights"], loss_cfg["positive_weights"],
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(loss_cfg["gradient_clip_norm"]))
                optimizer.step()
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}
```
这里dataloder加上collate_chunks, 返回的一个batch是一个字典。
计算完损失后，values是字典用于统计state, start, end的损失。

```python
def save_boundary_checkpoint(path: Path, model, optimizer, epoch: int, config: dict, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "epoch": epoch,
            "config": config,
            "metrics": metrics,
        },
        path,
    )
```
用于保存模型

```python
def load_boundary_checkpoint(path: str | Path, device: torch.device):
    checkpoint = safe_torch_load(path, device)
    model = build_model(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint
```
用于载入模型

```python
@torch.inference_mode()
def infer_run(model: torch.nn.Module, cache: dict[str, Any], device: torch.device) -> dict[str, np.ndarray]:
    features = cache["features"].float().unsqueeze(0).to(device)
    outputs = model(features)
    return {
        "state_probability": torch.softmax(outputs["state_logits"], dim=-1)[0, :, 1].cpu().numpy(),
        "start_probability": torch.sigmoid(outputs["start_logits"])[0].cpu().numpy(),
        "end_probability": torch.sigmoid(outputs["end_logits"])[0].cpu().numpy(),
    }
```
用于测试模型，cache["features"]维度为[T, D]，所以要unsqueeze()增加一个维度让其变成[B, T, D]

```python
def evaluate_caches(
    model: torch.nn.Module,
    cache_root: str | Path,
    runs: list[str],
    device: torch.device,
    online_cfg: dict[str, Any],
    evaluation_cfg: dict[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    per_run: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for run in runs:
        cache = load_feature_cache(Path(cache_root) / f"{run}.pt")
        # 载入一个run提取的特征
        probabilities = infer_run(model, cache, device)
        # 输出run中每一个时间步的state, start, end 预测状态，是一个字典。
        segments = run_state_machine(**probabilities, settings=online_cfg)
        pred_state = np.zeros(len(cache["state"]), dtype=np.int64)
        for segment in segments:
            pred_state[segment["start_index"] : segment["end_index"] + 1] = 1
        original = cache["original_frame_idx"].numpy()
        gt_state = cache["state"].numpy()
        gt_segments = []
        start = None
        for index, value in enumerate(list(gt_state) + [0]):
            if value and start is None:
                start = index
            elif not value and start is not None:
                gt_segments.append((start, index - 1)); start = None
        gt_starts = [int(original[index]) for index in np.flatnonzero(cache["exact_start"].numpy() > 0)]
        gt_ends = [int(original[index]) for index in np.flatnonzero(cache["exact_end"].numpy() > 0)]
        pred_segments_anchor = [(int(x["start_index"]), int(x["end_index"])) for x in segments]
        pred_segments_frames = [(int(original[x["start_index"]]), int(original[x["end_index"]])) for x in segments]
        per_run[run] = evaluate_run(
            gt_state, pred_state, gt_starts, gt_ends, pred_segments_anchor,
            [int(x) for x in evaluation_cfg["boundary_tolerance_frames"]],
            [segment[0] for segment in pred_segments_frames],
            [segment[1] for segment in pred_segments_frames],
        )
        emission_delays = [
            int(original[min(x["emitted_at_index"], len(original) - 1)]) - int(original[x["end_index"]])
            for x in segments
        ]
        per_run[run]["emission_delay_frames"] = float(np.mean(emission_delays)) if emission_delays else float("nan")
        for segment, frame_segment in zip(segments, pred_segments_frames):
            prediction_rows.append({"sample_name": run, **segment, "start_original_frame_idx": frame_segment[0], "end_original_frame_idx": frame_segment[1]})
    metrics = _macro_average(per_run)
    result = {"runs": len(runs), "macro": metrics, "per_run": per_run}
    write_json(output_root / "metrics.json", result)
    with (output_root / "predicted_segments.jsonl").open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return result



