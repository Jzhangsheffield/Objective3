#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
split_merged_manifest_clip_stratified.py

从 merged_manifest.jsonl 中生成 train_manifest.jsonl、val_manifest.jsonl、test_manifest.jsonl。

本脚本适用于当前这种设置：

    4 个参与者；
    每次指定 1 个参与者作为完全独立 test set；
    剩余 3 个参与者作为 train / val 候选；
    在 train / val 候选中，按照动作类别逐类抽取约 20% clips 作为 validation set。

划分原则
========
1. test participant 完全独立
   - 通过 --test_person 指定测试参与者，例如 J / M / MR / N
   - 该参与者的所有样本全部进入 test_manifest.jsonl
   - test participant 不会出现在 train 或 val 中

2. train / val 按 clip 级别划分
   - 不再强制 run-level grouping
   - 也就是说，同一个 run 的不同 clip 可能会分别进入 train 和 val
   - 这样做的目的是缓解验证集中的类别极度不均衡问题

3. 每个动作类别单独抽取 validation clips
   - 默认使用 tier1 作为动作类别
   - 每个类别中抽取 round(n_class * val_ratio) 个 clip 作为 val
   - 默认 val_ratio = 0.20

4. 每个类别内部，validation clips 尽量覆盖：
   - 不同 participant
   - 不同 lighting
   - 不同 pos

5. 对极小类别的处理
   - 如果某个类别在非测试数据中只有 1 个样本，则不抽 validation，全部保留在 train
   - 如果某个类别至少有 2 个样本，则至少抽 1 个样本进入 val
   - 同时保证每个类别在 train 中至少保留 1 个样本

输入 manifest 每行应为 JSON object，并至少包含：
    original_key
    tier1 / tier2 / tier3
    lighting
    pos

例如：
    {
      "sample_id": "sample_000001",
      "original_key": "M/adjust_slider/run_11_clip_000002_normal_mid",
      "tier1": "adjust",
      "tier2": "adjust_slider",
      "tier3": "adjust_slider",
      "lighting": "normal",
      "pos": "mid",
      ...
    }

participant 默认从 original_key 的第一级路径解析：
    M/adjust_slider/run_11_clip_000002_normal_mid
    -> participant = M
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


@dataclass(frozen=True)
class ClipInfo:
    """
    保存一个 clip 在划分时需要用到的关键信息。
    """

    index: int
    class_label: str
    participant: str
    lighting: str
    pos: str


def read_jsonl(path: Path) -> List[dict]:
    """
    读取 jsonl manifest。

    每一行必须是一个 JSON object。
    遇到格式错误时直接报错，不做静默跳过。
    """

    records: List[dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_no} in {path}: {e}"
                ) from e

            if not isinstance(obj, dict):
                raise ValueError(
                    f"Line {line_no} in {path} is not a JSON object."
                )

            records.append(obj)

    if len(records) == 0:
        raise ValueError(f"No valid records found in: {path}")

    return records


def write_jsonl(records: List[dict], indices: List[int], path: Path) -> None:
    """
    按照给定 index 顺序写出 jsonl manifest。
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for idx in indices:
            json.dump(records[idx], f, ensure_ascii=False)
            f.write("\n")


def require_field(record: dict, field_name: str, idx: int) -> str:
    """
    读取必需字段。

    如果字段不存在、为 None、为空字符串，则直接报错。
    这样可以尽早发现 manifest 格式问题。
    """

    if field_name not in record:
        raise KeyError(
            f"Record index {idx} does not contain required field: {field_name}"
        )

    value = record[field_name]

    if value is None:
        raise ValueError(
            f"Record index {idx} has None value for field: {field_name}"
        )

    value = str(value)

    if value == "":
        raise ValueError(
            f"Record index {idx} has empty value for field: {field_name}"
        )

    return value


def get_participant_from_original_key(
    record: dict,
    idx: int,
    original_key_field: str,
) -> str:
    """
    从 original_key 中解析 participant。

    默认假设 original_key 格式类似：
        M/adjust_slider/run_11_clip_000002_normal_mid

    则 participant = M。
    """

    original_key = require_field(record, original_key_field, idx)
    parts = original_key.split("/")

    if len(parts) < 2:
        raise ValueError(
            f"Record index {idx} has invalid original_key: {original_key}\n"
            f"Expected format like: M/adjust_slider/run_11_clip_000002_normal_mid"
        )

    participant = parts[0]

    if participant == "":
        raise ValueError(
            f"Record index {idx} has empty participant parsed from original_key: {original_key}"
        )

    return participant


def build_clip_infos(
    records: List[dict],
    indices: List[int],
    class_key: str,
    lighting_key: str,
    pos_key: str,
    original_key_field: str,
) -> List[ClipInfo]:
    """
    把 records 中指定 indices 的样本转换成 ClipInfo。
    """

    infos: List[ClipInfo] = []

    for idx in indices:
        record = records[idx]

        class_label = require_field(record, class_key, idx)
        lighting = require_field(record, lighting_key, idx)
        pos = require_field(record, pos_key, idx)
        participant = get_participant_from_original_key(
            record=record,
            idx=idx,
            original_key_field=original_key_field,
        )

        infos.append(
            ClipInfo(
                index=idx,
                class_label=class_label,
                participant=participant,
                lighting=lighting,
                pos=pos,
            )
        )

    return infos


def compute_val_count_for_class(
    n_class: int,
    val_ratio: float,
    min_val_per_class: int,
    min_train_per_class: int,
) -> int:
    """
    计算某个类别应该抽多少个 clip 进入 val。

    规则：
    1. 如果类别样本数不足以同时满足 train 和 val，则 val_count = 0
    2. 否则按照 round(n_class * val_ratio) 计算
    3. 至少抽 min_val_per_class 个
    4. 但必须给 train 至少保留 min_train_per_class 个

    例如：
        n_class = 100, val_ratio = 0.2 -> 20
        n_class = 5,   val_ratio = 0.2 -> 1
        n_class = 2,   val_ratio = 0.2 -> 1
        n_class = 1,   val_ratio = 0.2 -> 0
    """

    if n_class <= min_train_per_class:
        return 0

    raw = math.floor(n_class * val_ratio + 0.5)
    val_count = max(min_val_per_class, raw)

    max_allowed = n_class - min_train_per_class
    val_count = min(val_count, max_allowed)

    return val_count


def select_val_clips_for_one_class(
    clips: List[ClipInfo],
    val_count: int,
    rng: random.Random,
) -> List[int]:
    """
    对单个类别选择 validation clips。

    目标：
        在固定 val_count 的前提下，尽量覆盖更多 participant / lighting / pos。

    实现方式：
        使用贪心覆盖选择。

    每一步从剩余 clips 中选一个，使得它对当前 val 子集带来的新增覆盖最大。
    覆盖目标包括：
        participant
        lighting
        pos
        participant-lighting pair
        participant-pos pair
        lighting-pos pair
        participant-lighting-pos triple

    注意：
        这里是 clip-level split，不保证 run 不被拆分。
    """

    if val_count <= 0:
        return []

    if val_count >= len(clips):
        raise ValueError(
            "val_count must be smaller than the number of clips in this class, "
            "because train should keep at least one sample."
        )

    remaining = list(clips)
    selected: List[ClipInfo] = []

    covered_participants: Set[str] = set()
    covered_lightings: Set[str] = set()
    covered_positions: Set[str] = set()

    covered_participant_lighting: Set[Tuple[str, str]] = set()
    covered_participant_pos: Set[Tuple[str, str]] = set()
    covered_lighting_pos: Set[Tuple[str, str]] = set()
    covered_full_combo: Set[Tuple[str, str, str]] = set()

    # 为了让 tie-breaking 可复现，用固定 rng 生成随机值。
    # 注意这不是为了改变算法目标，只是在多个候选分数完全相同时随机打破平局。
    tie_breaker = {
        clip.index: rng.random()
        for clip in remaining
    }

    def score(clip: ClipInfo) -> Tuple[int, int, int, int, int, int, int, float]:
        """
        分数越大，越优先选择。

        前三项优先保证单独维度覆盖：
            participant, lighting, pos

        后四项用于进一步提高组合覆盖：
            participant-lighting
            participant-pos
            lighting-pos
            participant-lighting-pos
        """

        participant_lighting = (clip.participant, clip.lighting)
        participant_pos = (clip.participant, clip.pos)
        lighting_pos = (clip.lighting, clip.pos)
        full_combo = (clip.participant, clip.lighting, clip.pos)

        new_participant = int(clip.participant not in covered_participants)
        new_lighting = int(clip.lighting not in covered_lightings)
        new_pos = int(clip.pos not in covered_positions)

        new_participant_lighting = int(participant_lighting not in covered_participant_lighting)
        new_participant_pos = int(participant_pos not in covered_participant_pos)
        new_lighting_pos = int(lighting_pos not in covered_lighting_pos)
        new_full_combo = int(full_combo not in covered_full_combo)

        return (
            new_participant,
            new_lighting,
            new_pos,
            new_full_combo,
            new_participant_lighting,
            new_participant_pos,
            new_lighting_pos,
            tie_breaker[clip.index],
        )

    for _ in range(val_count):
        best_clip = max(remaining, key=score)

        selected.append(best_clip)
        remaining.remove(best_clip)

        covered_participants.add(best_clip.participant)
        covered_lightings.add(best_clip.lighting)
        covered_positions.add(best_clip.pos)

        covered_participant_lighting.add((best_clip.participant, best_clip.lighting))
        covered_participant_pos.add((best_clip.participant, best_clip.pos))
        covered_lighting_pos.add((best_clip.lighting, best_clip.pos))
        covered_full_combo.add((best_clip.participant, best_clip.lighting, best_clip.pos))

    return [clip.index for clip in selected]


def split_train_val_by_class(
    candidate_infos: List[ClipInfo],
    val_ratio: float,
    min_val_per_class: int,
    min_train_per_class: int,
    seed: int,
) -> Tuple[List[int], List[int], dict]:
    """
    从非测试参与者样本中划分 train / val。

    对每个动作类别单独处理：
        1. 计算该类别 val_count
        2. 用 coverage-aware greedy selection 选择 val clips
        3. 剩余样本进入 train

    返回：
        train_indices
        val_indices
        class_report
    """

    rng = random.Random(seed)

    class_to_clips: Dict[str, List[ClipInfo]] = defaultdict(list)

    for info in candidate_infos:
        class_to_clips[info.class_label].append(info)

    train_indices_set: Set[int] = set()
    val_indices_set: Set[int] = set()

    class_report: Dict[str, dict] = {}

    for class_label in sorted(class_to_clips.keys()):
        clips = class_to_clips[class_label]
        n_class = len(clips)

        val_count = compute_val_count_for_class(
            n_class=n_class,
            val_ratio=val_ratio,
            min_val_per_class=min_val_per_class,
            min_train_per_class=min_train_per_class,
        )

        val_indices_for_class = select_val_clips_for_one_class(
            clips=clips,
            val_count=val_count,
            rng=rng,
        )

        val_indices_for_class_set = set(val_indices_for_class)

        for clip in clips:
            if clip.index in val_indices_for_class_set:
                val_indices_set.add(clip.index)
            else:
                train_indices_set.add(clip.index)

        all_participants = sorted({clip.participant for clip in clips})
        all_lightings = sorted({clip.lighting for clip in clips})
        all_positions = sorted({clip.pos for clip in clips})

        val_clips = [
            clip for clip in clips
            if clip.index in val_indices_for_class_set
        ]

        val_participants = sorted({clip.participant for clip in val_clips})
        val_lightings = sorted({clip.lighting for clip in val_clips})
        val_positions = sorted({clip.pos for clip in val_clips})

        class_report[class_label] = {
            "total_samples_in_train_val_candidates": n_class,
            "train_samples": n_class - val_count,
            "val_samples": val_count,
            "actual_val_ratio": val_count / n_class if n_class > 0 else None,
            "available_participants": all_participants,
            "val_participants": val_participants,
            "missing_val_participants": sorted(set(all_participants) - set(val_participants)),
            "available_lightings": all_lightings,
            "val_lightings": val_lightings,
            "missing_val_lightings": sorted(set(all_lightings) - set(val_lightings)),
            "available_positions": all_positions,
            "val_positions": val_positions,
            "missing_val_positions": sorted(set(all_positions) - set(val_positions)),
            "note": (
                "No validation sample selected because this class has too few samples."
                if val_count == 0
                else ""
            ),
        }

    train_indices = sorted(train_indices_set)
    val_indices = sorted(val_indices_set)

    if train_indices_set & val_indices_set:
        overlap = sorted(train_indices_set & val_indices_set)
        raise RuntimeError(
            f"Internal split error: train and val overlap. First overlaps: {overlap[:10]}"
        )

    return train_indices, val_indices, class_report


def count_by_field(
    records: List[dict],
    indices: List[int],
    field_name: str,
) -> Dict[str, int]:
    counter = Counter()

    for idx in indices:
        value = require_field(records[idx], field_name, idx)
        counter[value] += 1

    return dict(sorted(counter.items(), key=lambda x: x[0]))


def count_by_participant(
    records: List[dict],
    indices: List[int],
    original_key_field: str,
) -> Dict[str, int]:
    counter = Counter()

    for idx in indices:
        participant = get_participant_from_original_key(
            record=records[idx],
            idx=idx,
            original_key_field=original_key_field,
        )
        counter[participant] += 1

    return dict(sorted(counter.items(), key=lambda x: x[0]))


def get_unique_values_by_field(
    records: List[dict],
    indices: List[int],
    field_name: str,
) -> List[str]:
    values = {
        require_field(records[idx], field_name, idx)
        for idx in indices
    }

    return sorted(values)


def check_test_person_independent(
    records: List[dict],
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    test_person: str,
    original_key_field: str,
) -> None:
    """
    检查 test participant 是否完全独立。
    """

    for split_name, indices in [
        ("train", train_indices),
        ("val", val_indices),
    ]:
        for idx in indices:
            participant = get_participant_from_original_key(
                record=records[idx],
                idx=idx,
                original_key_field=original_key_field,
            )

            if participant == test_person:
                raise RuntimeError(
                    f"Split error: test_person={test_person} appears in {split_name}. "
                    f"record index = {idx}"
                )

    for idx in test_indices:
        participant = get_participant_from_original_key(
            record=records[idx],
            idx=idx,
            original_key_field=original_key_field,
        )

        if participant != test_person:
            raise RuntimeError(
                f"Split error: non-test participant appears in test split. "
                f"expected={test_person}, got={participant}, record index={idx}"
            )


def make_report(
    records: List[dict],
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    class_report: dict,
    args: argparse.Namespace,
) -> dict:
    """
    生成划分统计报告，方便你检查划分是否合理。
    """

    train_val_total = len(train_indices) + len(val_indices)

    report = {
        "config": {
            "merged_manifest": str(args.merged_manifest),
            "output_dir": str(args.output_dir),
            "test_person": args.test_person,
            "class_key": args.class_key,
            "lighting_key": args.lighting_key,
            "pos_key": args.pos_key,
            "original_key_field": args.original_key_field,
            "val_ratio": args.val_ratio,
            "min_val_per_class": args.min_val_per_class,
            "min_train_per_class": args.min_train_per_class,
            "seed": args.seed,
            "split_level": "clip",
            "run_grouping": False,
            "note": (
                "This split is class-stratified at clip level. "
                "The test participant is strictly independent, but train/val may share clips from the same run."
            ),
        },
        "sample_counts": {
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_indices),
            "train_val_total": train_val_total,
            "val_ratio_in_train_val": (
                len(val_indices) / train_val_total
                if train_val_total > 0
                else None
            ),
        },
        "participants": {
            "train": get_unique_values_by_participant(records, train_indices, args.original_key_field),
            "val": get_unique_values_by_participant(records, val_indices, args.original_key_field),
            "test": get_unique_values_by_participant(records, test_indices, args.original_key_field),
        },
        "sample_distribution": {
            "train_by_participant": count_by_participant(
                records, train_indices, args.original_key_field
            ),
            "val_by_participant": count_by_participant(
                records, val_indices, args.original_key_field
            ),
            "test_by_participant": count_by_participant(
                records, test_indices, args.original_key_field
            ),
            "train_by_class": count_by_field(records, train_indices, args.class_key),
            "val_by_class": count_by_field(records, val_indices, args.class_key),
            "test_by_class": count_by_field(records, test_indices, args.class_key),
            "train_by_lighting": count_by_field(records, train_indices, args.lighting_key),
            "val_by_lighting": count_by_field(records, val_indices, args.lighting_key),
            "test_by_lighting": count_by_field(records, test_indices, args.lighting_key),
            "train_by_pos": count_by_field(records, train_indices, args.pos_key),
            "val_by_pos": count_by_field(records, val_indices, args.pos_key),
            "test_by_pos": count_by_field(records, test_indices, args.pos_key),
        },
        "per_class_split": class_report,
    }

    return report


def get_unique_values_by_participant(
    records: List[dict],
    indices: List[int],
    original_key_field: str,
) -> List[str]:
    values = {
        get_participant_from_original_key(
            record=records[idx],
            idx=idx,
            original_key_field=original_key_field,
        )
        for idx in indices
    }

    return sorted(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split merged_manifest.jsonl into train / val / test manifests. "
            "Test split is participant-independent. "
            "Validation split is class-stratified at clip level."
        )
    )

    parser.add_argument(
        "--merged_manifest",
        type=Path,
        required=True,
        help="Path to merged_manifest.jsonl.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for train_manifest.jsonl, val_manifest.jsonl, test_manifest.jsonl, and split_report.json.",
    )

    parser.add_argument(
        "--test_person",
        type=str,
        required=True,
        help="Participant used as test set, e.g. J, M, MR, or N.",
    )

    parser.add_argument(
        "--class_key",
        type=str,
        default="tier1",
        help="Class label key used for stratified validation split. Default: tier1.",
    )

    parser.add_argument(
        "--lighting_key",
        type=str,
        default="lighting",
        help="Manifest key for lighting condition. Default: lighting.",
    )

    parser.add_argument(
        "--pos_key",
        type=str,
        default="pos",
        help="Manifest key for position condition. Default: pos.",
    )

    parser.add_argument(
        "--original_key_field",
        type=str,
        default="original_key",
        help="Manifest key used to parse participant. Default: original_key.",
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.20,
        help="Validation ratio for each class among non-test participants. Default: 0.20.",
    )

    parser.add_argument(
        "--min_val_per_class",
        type=int,
        default=1,
        help="Minimum validation samples per class if the class has enough samples. Default: 1.",
    )

    parser.add_argument(
        "--min_train_per_class",
        type=int,
        default=1,
        help="Minimum training samples per class after validation split. Default: 1.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for tie-breaking in coverage-aware selection. Default: 42.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.merged_manifest.exists():
        raise FileNotFoundError(
            f"merged_manifest does not exist: {args.merged_manifest}"
        )

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError(
            f"val_ratio must be in (0, 1), got {args.val_ratio}"
        )

    if args.min_val_per_class < 0:
        raise ValueError(
            f"min_val_per_class must be >= 0, got {args.min_val_per_class}"
        )

    if args.min_train_per_class < 1:
        raise ValueError(
            f"min_train_per_class must be >= 1, got {args.min_train_per_class}"
        )

    records = read_jsonl(args.merged_manifest)

    all_participants = sorted(
        {
            get_participant_from_original_key(
                record=record,
                idx=idx,
                original_key_field=args.original_key_field,
            )
            for idx, record in enumerate(records)
        }
    )

    if args.test_person not in all_participants:
        raise ValueError(
            f"test_person={args.test_person} not found in manifest.\n"
            f"Available participants: {all_participants}"
        )

    test_indices: List[int] = []
    candidate_indices: List[int] = []

    for idx, record in enumerate(records):
        participant = get_participant_from_original_key(
            record=record,
            idx=idx,
            original_key_field=args.original_key_field,
        )

        if participant == args.test_person:
            test_indices.append(idx)
        else:
            candidate_indices.append(idx)

    if len(test_indices) == 0:
        raise ValueError(
            f"No test samples found for test_person={args.test_person}"
        )

    if len(candidate_indices) == 0:
        raise ValueError(
            "No train/val candidate samples found after removing test_person."
        )

    candidate_infos = build_clip_infos(
        records=records,
        indices=candidate_indices,
        class_key=args.class_key,
        lighting_key=args.lighting_key,
        pos_key=args.pos_key,
        original_key_field=args.original_key_field,
    )

    train_indices, val_indices, class_report = split_train_val_by_class(
        candidate_infos=candidate_infos,
        val_ratio=args.val_ratio,
        min_val_per_class=args.min_val_per_class,
        min_train_per_class=args.min_train_per_class,
        seed=args.seed,
    )

    check_test_person_independent(
        records=records,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        test_person=args.test_person,
        original_key_field=args.original_key_field,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.output_dir / "train_manifest.jsonl"
    val_path = args.output_dir / "val_manifest.jsonl"
    test_path = args.output_dir / "test_manifest.jsonl"
    report_path = args.output_dir / "split_report.json"

    write_jsonl(records, train_indices, train_path)
    write_jsonl(records, val_indices, val_path)
    write_jsonl(records, test_indices, test_path)

    report = make_report(
        records=records,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        class_report=class_report,
        args=args,
    )

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    train_val_total = len(train_indices) + len(val_indices)
    actual_val_ratio = len(val_indices) / train_val_total if train_val_total > 0 else 0.0

    print("Done.")
    print(f"Available participants: {all_participants}")
    print(f"Test person: {args.test_person}")
    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples:   {len(val_indices)}")
    print(f"Test samples:  {len(test_indices)}")
    print(f"Actual val ratio in non-test data: {actual_val_ratio:.4f}")
    print()
    print(f"Train manifest: {train_path}")
    print(f"Val manifest:   {val_path}")
    print(f"Test manifest:  {test_path}")
    print(f"Split report:   {report_path}")
    print()
    print("Note:")
    print("  This script performs clip-level class-stratified validation split.")
    print("  Test participant is strictly independent.")
    print("  Train and val may contain clips from the same run by design.")


if __name__ == "__main__":
    main()