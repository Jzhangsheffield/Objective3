
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
resplit_mapstyle_manifest_by_rules.py

作用：
------
1. 读取当前 map-style 数据集下已有的 train / val / test 三个 manifest.jsonl
2. 将三份 manifest 的样本先合并成一个总样本池
3. 按“规则”重新划分样本到新的 train / val / test
4. 生成新的 train_manifest.jsonl / val_manifest.jsonl / test_manifest.jsonl
5. 生成详细检查报告，帮助你确认：
   - 每个 split 的样本数量
   - 每个 split 中 person / lighting / pos 的分布
   - 每个 split 中 tier1 / tier2 / tier3 的类别覆盖情况
   - 哪些 tier 标签只出现在某一个 split 中
   - 哪些 tier 标签在某些 split 中缺失
   - 哪些样本只对应“单一 split 独占标签”，便于后续人工检查或训练时跳过

设计原则：
----------
- 保持 manifest 字段结构不变，兼容你现在的 dataloader
- 支持非常灵活的规则配置：
    例如：
        J + normal -> test
        J + left/right -> val
        其他 -> train

- 也支持更复杂的组合条件：
        person / lighting / pos / action / tier1 / tier2 / tier3
- 规则按顺序匹配，先匹配先分配
- 如果没有命中任何规则，则进入 default_split

注意：
-----
1) 本脚本默认“只重写 manifest”，不会复制物理数据。
   因此新 manifest 中的路径仍可能指向旧的 train/val/test 目录。
   这是允许的，因为你的 dataloader 是按 jsonl 中的相对路径去找文件的。

2) 如果你希望所有数据物理上合并到一个新目录，再做重划分，
   请使用配套脚本：merge_mapstyle_dataset_to_unified_root.py

3) 当前脚本默认保留原始 sample_name，不主动改名。
   如果你担心不同旧 split 中 sample_name 重复导致后续分析不方便，
   可以通过参数选择自动加前缀，使 sample_name 在新 split 中更容易区分。

示例：
-----
1. 先写一个规则文件 split_rules_example.json

{
  "default_split": "train",
  "rules": [
    {
      "name": "J_normal_to_test",
      "split": "test",
      "person": ["J"],
      "lighting": ["normal"]
    },
    {
      "name": "J_left_right_to_val",
      "split": "val",
      "person": ["J"],
      "lighting": ["left", "right"]
    }
  ]
}

2. 运行：
python resplit_mapstyle_manifest_by_rules.py \
    --dataset_root "L:/Dataset_thermal_crimper/mapstyle_dataset" \
    --rules_json "L:/Dataset_thermal_crimper/mapstyle_dataset/split_rules_example.json" \
    --output_dir "L:/Dataset_thermal_crimper/mapstyle_dataset_resplit_case1"

输出：
-----
output_dir/
    train_manifest.jsonl
    val_manifest.jsonl
    test_manifest.jsonl
    split_stats.txt
    missing_labels_report.txt
    exclusive_labels_report.txt
    assignment_trace.txt
    duplicate_original_key_report.txt
    summary.json
"""

import argparse
import copy
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ============================================================
# 一些固定的字段定义
# ============================================================

# 允许用于“规则匹配”的字段。
# 你当前明确提到需要支持 person / lighting / pos，
# 这里额外顺手支持 action / tier1 / tier2 / tier3，后续更灵活。
MATCHABLE_FIELDS = [
    "person",
    "lighting",
    "pos",
    "action",
    "tier1",
    "tier2",
    "tier3",
]

# 三个 tier 字段，用于后面覆盖率检查。
TIER_FIELDS = ["tier1", "tier2", "tier3"]

# 合法 split 名称
VALID_SPLITS = {"train", "val", "test"}


# ============================================================
# 基础 I/O
# ============================================================

def load_jsonl_records(jsonl_path: Path, source_manifest_name: str) -> List[Dict]:
    """
    读取单个 jsonl 文件，返回样本字典列表。

    参数
    ----
    jsonl_path:
        manifest.jsonl 文件路径
    source_manifest_name:
        例如 "train_manifest.jsonl"，用于记录样本原始来源

    返回
    ----
    records: List[Dict]
        每个元素都是一条 manifest 记录
    """
    records: List[Dict] = []
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Manifest not found: {jsonl_path}")

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON decode error in {jsonl_path} at line {line_idx}: {exc}"
                ) from exc

            # 记录来源，后续调试和报告很有用
            # 注意：这是额外的内部辅助字段，不会写回最终 manifest
            obj["__source_manifest__"] = source_manifest_name

            # 尝试从 manifest 名称猜测原 split
            if source_manifest_name.startswith("train"):
                obj["__source_split__"] = "train"
            elif source_manifest_name.startswith("val"):
                obj["__source_split__"] = "val"
            elif source_manifest_name.startswith("test"):
                obj["__source_split__"] = "test"
            else:
                obj["__source_split__"] = "unknown"

            records.append(obj)

    return records


def save_jsonl_records(records: List[Dict], out_path: Path) -> None:
    """
    将样本列表写成 jsonl 文件。

    注意：
    ----
    写回前会去掉内部辅助字段（以 __ 开头），以保持 manifest 格式干净，
    并最大程度兼容你现在的 dataloader。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            clean_record = {
                k: v for k, v in record.items()
                if not k.startswith("__")
            }
            f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")


def load_rules(rules_json_path: Path) -> Dict:
    """
    读取规则文件。

    规则文件格式：
    -------------
    {
      "default_split": "train",
      "rules": [
        {
          "name": "J_normal_to_test",
          "split": "test",
          "person": ["J"],
          "lighting": ["normal"]
        },
        {
          "name": "J_left_right_to_val",
          "split": "val",
          "person": ["J"],
          "lighting": ["left", "right"]
        }
      ]
    }

    每条规则中：
    - split 是目标集合
    - 其余 MATCHABLE_FIELDS 中出现的键，表示筛选条件
    - 条件之间是 AND 关系
    - 列表中的多个值是 OR 关系
    """
    if not rules_json_path.exists():
        raise FileNotFoundError(f"Rules JSON not found: {rules_json_path}")

    with rules_json_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if "default_split" not in config:
        raise ValueError("Rules JSON must contain 'default_split'")

    default_split = config["default_split"]
    if default_split not in VALID_SPLITS:
        raise ValueError(f"Invalid default_split: {default_split}")

    rules = config.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError("'rules' must be a list")

    for idx, rule in enumerate(rules):
        if "split" not in rule:
            raise ValueError(f"Rule index {idx} missing required key: 'split'")
        if rule["split"] not in VALID_SPLITS:
            raise ValueError(f"Rule index {idx} has invalid split: {rule['split']}")

        for key in rule:
            if key in {"name", "split"}:
                continue
            if key not in MATCHABLE_FIELDS:
                raise ValueError(
                    f"Rule index {idx} contains unsupported field '{key}'. "
                    f"Supported fields: {MATCHABLE_FIELDS}"
                )

            if not isinstance(rule[key], list):
                raise ValueError(
                    f"Rule index {idx}, field '{key}' must be a list. "
                    f"Example: 'person': ['J', 'M']"
                )

    return config


# ============================================================
# 规则匹配相关
# ============================================================

def record_matches_rule(record: Dict, rule: Dict) -> bool:
    """
    判断一条样本是否匹配某条规则。

    规则逻辑：
    --------
    - 一条规则中的不同字段之间：AND
    - 同一个字段给多个值之间：OR

    例如：
    rule = {
        "split": "val",
        "person": ["J"],
        "lighting": ["left", "right"]
    }

    表示：
        person == "J" AND lighting in {"left", "right"}
    """
    for field in MATCHABLE_FIELDS:
        if field not in rule:
            # 规则里没写该字段，说明该字段“不限制”
            continue

        allowed_values = rule[field]
        value = record.get(field, None)

        if value not in allowed_values:
            return False

    return True


def assign_split_by_rules(
    record: Dict,
    rules_config: Dict,
) -> Tuple[str, str]:
    """
    根据规则为单条样本分配目标 split。

    返回
    ----
    assigned_split:
        "train" / "val" / "test"
    rule_name:
        命中的规则名称；如果没命中任何规则，则返回 "DEFAULT"
    """
    rules = rules_config["rules"]
    default_split = rules_config["default_split"]

    for idx, rule in enumerate(rules):
        if record_matches_rule(record, rule):
            rule_name = rule.get("name", f"RULE_{idx:03d}")
            return rule["split"], rule_name

    return default_split, "DEFAULT"


# ============================================================
# sample_name 处理
# ============================================================

def update_sample_name_if_needed(record: Dict, mode: str) -> Dict:
    """
    按需要修改 sample_name。

    mode 可选：
    ----------
    preserve:
        保持原来的 sample_name 不变

    source_prefix:
        在 sample_name 前面加原始来源 split 前缀，例如：
            sample_0001 -> train__sample_0001

    original_key_hashless:
        用 original_key 直接替换斜杠为双下划线，得到更稳定且通常唯一的 sample_name，例如：
            J/adjust_slider/run_15_clip_000003_right_elbow
            -> J__adjust_slider__run_15_clip_000003_right_elbow
    """
    new_record = copy.deepcopy(record)

    old_sample_name = str(new_record.get("sample_name", "unknown_sample"))
    source_split = str(new_record.get("__source_split__", "unknown"))
    original_key = str(new_record.get("original_key", old_sample_name))

    if mode == "preserve":
        return new_record

    if mode == "source_prefix":
        new_record["sample_name"] = f"{source_split}__{old_sample_name}"
        return new_record

    if mode == "original_key_hashless":
        safe_name = original_key.replace("/", "__").replace("\\", "__")
        new_record["sample_name"] = safe_name
        return new_record

    raise ValueError(f"Unsupported sample_name_mode: {mode}")


# ============================================================
# 各类统计与检查
# ============================================================

def count_distribution(records: List[Dict], field: str) -> Counter:
    """统计某个字段在样本列表中的分布。"""
    c = Counter()
    for r in records:
        c[str(r.get(field, "MISSING"))] += 1
    return c


def collect_unique_labels(records: List[Dict], field: str) -> set:
    """收集某个 tier 字段的唯一标签集合。"""
    labels = set()
    for r in records:
        value = r.get(field, None)
        if value is not None:
            labels.add(str(value))
    return labels


def build_label_to_samples(records: List[Dict], field: str) -> Dict[str, List[Dict]]:
    """
    构建标签到样本的映射。

    例如：
        label_to_samples["adjust"] = [record1, record2, ...]
    """
    mapping = defaultdict(list)
    for r in records:
        value = r.get(field, None)
        if value is not None:
            mapping[str(value)].append(r)
    return mapping


def summarize_split_records(split_records: Dict[str, List[Dict]]) -> Dict:
    """
    生成机器可读的汇总字典 summary。

    后续会写入 summary.json，方便你程序化检查。
    """
    summary: Dict[str, Dict] = {}

    for split_name, records in split_records.items():
        item = {
            "num_samples": len(records),
            "person_distribution": dict(count_distribution(records, "person")),
            "lighting_distribution": dict(count_distribution(records, "lighting")),
            "pos_distribution": dict(count_distribution(records, "pos")),
            "tier1_num_classes": len(collect_unique_labels(records, "tier1")),
            "tier2_num_classes": len(collect_unique_labels(records, "tier2")),
            "tier3_num_classes": len(collect_unique_labels(records, "tier3")),
        }
        summary[split_name] = item

    return summary


def write_split_stats_txt(split_records: Dict[str, List[Dict]], out_path: Path) -> None:
    """
    生成人工可读的 split_stats.txt。

    内容包括：
    - 每个 split 的样本数
    - person / lighting / pos 分布
    - tier1 / tier2 / tier3 的类别数量
    """
    lines: List[str] = []
    lines.append("=== Split Statistics ===")
    lines.append("")

    for split_name in ["train", "val", "test"]:
        records = split_records.get(split_name, [])
        lines.append(f"[{split_name}]")
        lines.append(f"num_samples: {len(records)}")
        lines.append("")

        for field in ["person", "lighting", "pos"]:
            dist = count_distribution(records, field)
            lines.append(f"{field}_distribution:")
            for key, value in sorted(dist.items()):
                lines.append(f"  {key}: {value}")
            lines.append("")

        for tier in TIER_FIELDS:
            label_set = sorted(collect_unique_labels(records, tier))
            lines.append(f"{tier}_num_classes: {len(label_set)}")
            lines.append(f"{tier}_labels:")
            for label in label_set:
                lines.append(f"  {label}")
            lines.append("")

        lines.append("-" * 80)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_missing_labels_report(
    split_records: Dict[str, List[Dict]],
    out_path: Path,
) -> Dict:
    """
    生成 missing_labels_report.txt

    逻辑：
    ----
    对每个 tier：
        1) 先求 train/val/test 的标签并集
        2) 再检查每个 split 缺哪些标签

    这可以帮助你确认：
    - 某些类别是否完全没有进入 val
    - 某些类别是否完全没有进入 test
    - 某些类别是否只有 train 有
    """
    report: Dict[str, Dict[str, List[str]]] = {}
    lines: List[str] = []
    lines.append("=== Missing Labels Report ===")
    lines.append("")

    for tier in TIER_FIELDS:
        split_to_labels = {
            split: collect_unique_labels(records, tier)
            for split, records in split_records.items()
        }
        union_labels = set().union(*split_to_labels.values())

        report[tier] = {}
        lines.append(f"[{tier}]")
        lines.append(f"union_num_labels: {len(union_labels)}")

        for split in ["train", "val", "test"]:
            missing = sorted(union_labels - split_to_labels.get(split, set()))
            report[tier][split] = missing
            lines.append(f"{split}_missing_num: {len(missing)}")
            for label in missing:
                lines.append(f"  {label}")
            lines.append("")

        lines.append("-" * 80)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return report


def write_exclusive_labels_report(
    split_records: Dict[str, List[Dict]],
    out_path: Path,
) -> Dict:
    """
    生成 exclusive_labels_report.txt

    逻辑：
    ----
    对每个 tier 标签，检查它出现在哪些 split 中。
    如果某个标签只出现在一个 split 中，就把它列出来。

    同时会把对应样本的 original_key / sample_name / person / lighting / pos 写出来，
    方便你后续人工检查这些“独占标签”是否需要跳过。
    """
    report: Dict[str, Dict[str, Dict]] = {}
    lines: List[str] = []
    lines.append("=== Exclusive Labels Report ===")
    lines.append("")

    for tier in TIER_FIELDS:
        report[tier] = {}
        lines.append(f"[{tier}]")

        # 先构建：label -> 出现在哪些 split
        label_presence = defaultdict(set)
        split_label_to_samples = {}

        for split_name, records in split_records.items():
            label_to_samples = build_label_to_samples(records, tier)
            split_label_to_samples[split_name] = label_to_samples
            for label in label_to_samples:
                label_presence[label].add(split_name)

        exclusive_labels = []
        for label, present_splits in label_presence.items():
            if len(present_splits) == 1:
                only_split = list(present_splits)[0]
                exclusive_labels.append((label, only_split))

        exclusive_labels.sort(key=lambda x: (x[1], x[0]))
        lines.append(f"exclusive_label_num: {len(exclusive_labels)}")
        lines.append("")

        for label, only_split in exclusive_labels:
            samples = split_label_to_samples[only_split][label]
            report[tier][label] = {
                "only_split": only_split,
                "num_samples": len(samples),
                "samples": [
                    {
                        "sample_name": s.get("sample_name", ""),
                        "original_key": s.get("original_key", ""),
                        "person": s.get("person", ""),
                        "lighting": s.get("lighting", ""),
                        "pos": s.get("pos", ""),
                        "source_manifest": s.get("__source_manifest__", ""),
                    }
                    for s in samples
                ],
            }

            lines.append(f"label: {label}")
            lines.append(f"only_split: {only_split}")
            lines.append(f"num_samples: {len(samples)}")
            lines.append("samples:")
            for s in samples:
                lines.append(
                    "  - sample_name={sample_name}, original_key={original_key}, "
                    "person={person}, lighting={lighting}, pos={pos}, source_manifest={source_manifest}".format(
                        sample_name=s.get("sample_name", ""),
                        original_key=s.get("original_key", ""),
                        person=s.get("person", ""),
                        lighting=s.get("lighting", ""),
                        pos=s.get("pos", ""),
                        source_manifest=s.get("__source_manifest__", ""),
                    )
                )
            lines.append("")

        lines.append("-" * 80)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return report


def write_assignment_trace(
    all_records: List[Dict],
    out_path: Path,
) -> None:
    """
    生成 assignment_trace.txt

    用途：
    ----
    这个文件相当于“划分轨迹”，用于回看每条样本是如何被分到新 split 的。
    非常适合后续排查规则是否写错。

    每行记录：
    - sample_name
    - original_key
    - person / lighting / pos
    - 原始来源 split
    - 新 split
    - 命中的规则名
    """
    lines: List[str] = []
    lines.append("=== Assignment Trace ===")
    lines.append("")

    for r in all_records:
        lines.append(
            "sample_name={sample_name} | original_key={original_key} | person={person} | "
            "lighting={lighting} | pos={pos} | source_split={source_split} | "
            "assigned_split={assigned_split} | matched_rule={matched_rule}".format(
                sample_name=r.get("sample_name", ""),
                original_key=r.get("original_key", ""),
                person=r.get("person", ""),
                lighting=r.get("lighting", ""),
                pos=r.get("pos", ""),
                source_split=r.get("__source_split__", ""),
                assigned_split=r.get("__assigned_split__", ""),
                matched_rule=r.get("__matched_rule__", ""),
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_duplicate_original_key_report(
    split_records: Dict[str, List[Dict]],
    out_path: Path,
) -> Dict[str, List[str]]:
    """
    检查同一个 split 中是否出现重复 original_key。

    理论上你原始数据集每条样本的 original_key 应该能作为较稳定的“样本身份标识”。
    如果在同一个新 split 中出现重复 original_key，通常说明：
    - 原 manifest 本身有重复
    - 或者你重复合并了某些文件

    这个检查不是必须，但非常实用。
    """
    report: Dict[str, List[str]] = {}
    lines: List[str] = []
    lines.append("=== Duplicate original_key Report ===")
    lines.append("")

    for split_name, records in split_records.items():
        c = Counter(str(r.get("original_key", "")) for r in records)
        duplicates = sorted([k for k, v in c.items() if v > 1])
        report[split_name] = duplicates

        lines.append(f"[{split_name}] duplicate_num: {len(duplicates)}")
        for key in duplicates:
            lines.append(f"  {key}")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return report



# ============================================================
# 主逻辑
# ============================================================

def load_input_records(dataset_root: Path, input_mode: str) -> List[Dict]:
    """
    按指定模式读取输入样本。

    input_mode:
    -----------
    three_manifests:
        读取 dataset_root 下的
        - train_manifest.jsonl
        - val_manifest.jsonl
        - test_manifest.jsonl

    all_manifest:
        读取 dataset_root 下的
        - all_manifest.jsonl

    返回：
    ----
    一个合并后的总样本列表
    """
    if input_mode == "three_manifests":
        manifest_paths = [
            dataset_root / "train_manifest_with_mindrove.jsonl",
            dataset_root / "val_manifest_with_mindrove.jsonl",
            dataset_root / "test_manifest_with_mindrove.jsonl",
        ]
    elif input_mode == "all_manifest":
        manifest_paths = [dataset_root / "all_manifest.jsonl"]
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    all_records: List[Dict] = []
    for path in manifest_paths:
        all_records.extend(load_jsonl_records(path, path.name))
    return all_records


def load_all_manifests(dataset_root: Path) -> List[Dict]:
    """
    读取 dataset_root 下当前的 train/val/test 三份 manifest，并合并。

    支持两种读取模式：

    1) three_manifests
       从 dataset_root 读取：
       - train_manifest.jsonl
       - val_manifest.jsonl
       - test_manifest.jsonl

    2) all_manifest
       从 dataset_root 读取：
       - all_manifest.jsonl

    这样当你先执行“物理合并脚本”后，也可以直接对 unified_root/all_manifest.jsonl 再做重划分。
    """
    raise RuntimeError("This function has been replaced by load_input_records().")


def resplit_records(
    all_records: List[Dict],
    rules_config: Dict,
    sample_name_mode: str,
) -> Dict[str, List[Dict]]:
    """
    对总样本池执行重划分。

    返回
    ----
    split_records:
        {
            "train": [...],
            "val": [...],
            "test": [...]
        }
    """
    split_records = {"train": [], "val": [], "test": []}

    for record in all_records:
        # 先为样本分配目标 split
        assigned_split, matched_rule = assign_split_by_rules(record, rules_config)

        new_record = update_sample_name_if_needed(record, sample_name_mode)

        # 记录内部辅助字段，便于后续出报告
        new_record["__assigned_split__"] = assigned_split
        new_record["__matched_rule__"] = matched_rule

        split_records[assigned_split].append(new_record)

    return split_records


def write_summary_json(
    split_records: Dict[str, List[Dict]],
    rules_config: Dict,
    duplicate_report: Dict[str, List[str]],
    missing_report: Dict,
    exclusive_report: Dict,
    out_path: Path,
) -> None:
    """
    写一个 summary.json，便于你后续写别的分析脚本直接读取。
    """
    summary = {
        "rules_config": rules_config,
        "split_summary": summarize_split_records(split_records),
        "duplicate_original_key_report": duplicate_report,
        "missing_labels_report": missing_report,
        "exclusive_labels_report": exclusive_report,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-split map-style manifest files by flexible rules."
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        type=str,
        help="当前 mapstyle_dataset 根目录。里面需要有 train_manifest.jsonl / val_manifest.jsonl / test_manifest.jsonl",
    )
    parser.add_argument(
        "--rules_json",
        required=True,
        type=str,
        help="重划分规则文件（JSON）路径",
    )
    parser.add_argument(
        "--input_mode",
        default="three_manifests",
        choices=["three_manifests", "all_manifest"],
        help=(
            "输入模式。three_manifests=读取 train/val/test 三份 manifest；"
            "all_manifest=读取 all_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="输出目录。新的 manifest 和报告文件都会保存到这里",
    )
    parser.add_argument(
        "--sample_name_mode",
        default="preserve",
        choices=["preserve", "source_prefix", "original_key_hashless"],
        help=(
            "如何处理 sample_name。"
            "preserve=保持原样；"
            "source_prefix=前面加旧 split 前缀；"
            "original_key_hashless=用 original_key 替换斜杠后作为 sample_name"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    rules_json = Path(args.rules_json)
    output_dir = Path(args.output_dir)

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读取规则
    rules_config = load_rules(rules_json)

    # 2) 读取输入样本
    all_records = load_input_records(dataset_root, args.input_mode)

    # 3) 执行重划分
    split_records = resplit_records(
        all_records=all_records,
        rules_config=rules_config,
        sample_name_mode=args.sample_name_mode,
    )

    # 4) 保存新的 manifest
    save_jsonl_records(split_records["train"], output_dir / "train_manifest.jsonl")
    save_jsonl_records(split_records["val"], output_dir / "val_manifest.jsonl")
    save_jsonl_records(split_records["test"], output_dir / "test_manifest.jsonl")

    # 5) 保存划分轨迹与各类报告
    #    注意：这里的 all_records 需要用“已分配后的版本”来写轨迹
    assigned_all_records = (
        split_records["train"] + split_records["val"] + split_records["test"]
    )

    write_assignment_trace(assigned_all_records, output_dir / "assignment_trace.txt")
    write_split_stats_txt(split_records, output_dir / "split_stats.txt")
    duplicate_report = write_duplicate_original_key_report(
        split_records, output_dir / "duplicate_original_key_report.txt"
    )
    missing_report = write_missing_labels_report(
        split_records, output_dir / "missing_labels_report.txt"
    )
    exclusive_report = write_exclusive_labels_report(
        split_records, output_dir / "exclusive_labels_report.txt"
    )
    write_summary_json(
        split_records=split_records,
        rules_config=rules_config,
        duplicate_report=duplicate_report,
        missing_report=missing_report,
        exclusive_report=exclusive_report,
        out_path=output_dir / "summary.json",
    )

    # 6) 控制台打印简要结果
    print("=" * 80)
    print("Re-splitting finished.")
    print(f"Output directory: {output_dir}")
    print(f"train num_samples: {len(split_records['train'])}")
    print(f"val   num_samples: {len(split_records['val'])}")
    print(f"test  num_samples: {len(split_records['test'])}")
    print("=" * 80)


if __name__ == "__main__":
    main()
