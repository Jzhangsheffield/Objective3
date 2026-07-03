#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_mapstyle_splits.py

将 train / val / test 三个划分的 map-style 数据集合并成一个统一数据集，
并重新生成一个整体 manifest。

合并后的目录结构示例
====================

output_root/
    merged_dataset/
        sample_0001/
            rgb.pt
            depth.pt
            label.txt
            mindrove.pt
        sample_0002/
            rgb.pt
            depth.pt
            label.txt
            mindrove.pt
        ...
    merged_manifest.jsonl

新的 merged_manifest.jsonl 中，路径会写成：

    "rgb": "merged_dataset/sample_0001/rgb.pt"
    "depth": "merged_dataset/sample_0001/depth.pt"
    "label_txt": "merged_dataset/sample_0001/label.txt"
    "mindrove": "merged_dataset/sample_0001/mindrove.pt"

注意
====
1. 不会新增 source_split / original_sample_name 等字段。
2. 除 sample_name、rgb、depth、label_txt、mindrove 以外，其他字段保持不变。
3. 默认不覆盖已有的 merged_dataset 或 merged_manifest.jsonl。
4. 会严格检查每个样本的 rgb / depth / label_txt / mindrove 是否存在。
5. 会复制整个原始样本文件夹，因此如果 sample_xxxx 下面还有其他文件，也会一起复制。
"""

import argparse
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Dict, Any, List, Tuple


PATH_FIELDS = ("rgb", "depth", "label_txt", "mindrove")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge train/val/test map-style dataset splits into one dataset."
    )

    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help=(
            "原始数据集根目录。manifest 中的相对路径会基于这个目录解析。"
            "例如 manifest 中 rgb=train/sample_0001/rgb.pt，"
            "则实际文件路径为 input_root/train/sample_0001/rgb.pt。"
        ),
    )

    parser.add_argument(
        "--train_manifest",
        type=str,
        required=True,
        help="train manifest jsonl 文件路径。可以是绝对路径，也可以是相对于 input_root 的路径。",
    )

    parser.add_argument(
        "--val_manifest",
        type=str,
        required=True,
        help="val manifest jsonl 文件路径。可以是绝对路径，也可以是相对于 input_root 的路径。",
    )

    parser.add_argument(
        "--test_manifest",
        type=str,
        required=True,
        help="test manifest jsonl 文件路径。可以是绝对路径，也可以是相对于 input_root 的路径。",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="输出根目录。merged_dataset 和 merged_manifest.jsonl 会放在这个目录下。",
    )

    parser.add_argument(
        "--merged_dataset_name",
        type=str,
        default="merged_dataset",
        help="合并后数据集文件夹名称，默认 merged_dataset。",
    )

    parser.add_argument(
        "--merged_manifest_name",
        type=str,
        default="merged_manifest.jsonl",
        help="合并后 manifest 文件名，默认 merged_manifest.jsonl。",
    )

    parser.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="新 sample 编号起始值，默认 1，即 sample_0001。",
    )

    parser.add_argument(
        "--num_digits",
        type=int,
        default=4,
        help="sample 编号位数，默认 4，即 sample_0001。",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果指定，则删除已有的 merged_dataset 和 merged_manifest.jsonl 后重新生成。",
    )

    return parser.parse_args()


def resolve_path(path_str: str, base_dir: Path) -> Path:
    """
    如果 path_str 是绝对路径，则直接返回；
    如果是相对路径，则基于 base_dir 解析。
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return base_dir / path


def load_jsonl(manifest_path: Path) -> List[Dict[str, Any]]:
    """
    读取 JSONL manifest。
    每一行必须是合法 JSON object。
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")

    records: List[Dict[str, Any]] = []

    with manifest_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                raise ValueError(
                    f"Empty line found in manifest: {manifest_path}, line {line_idx}"
                )

            record = json.loads(line)

            if not isinstance(record, dict):
                raise TypeError(
                    f"Each JSONL line must be a JSON object. "
                    f"Got {type(record)} in {manifest_path}, line {line_idx}"
                )

            records.append(record)

    return records


def check_required_fields(record: Dict[str, Any], record_desc: str) -> None:
    """
    检查当前样本是否包含必要字段。
    """
    if "sample_name" not in record:
        raise KeyError(f"Missing key 'sample_name' in {record_desc}")

    for field in PATH_FIELDS:
        if field not in record:
            raise KeyError(f"Missing key '{field}' in {record_desc}")

        if not isinstance(record[field], str):
            raise TypeError(
                f"Field '{field}' must be a string path in {record_desc}, "
                f"but got {type(record[field])}"
            )


def get_source_files(
    record: Dict[str, Any],
    input_root: Path,
    record_desc: str,
) -> Dict[str, Path]:
    """
    根据 manifest 中的相对路径，解析出实际源文件路径。
    """
    source_files: Dict[str, Path] = {}

    for field in PATH_FIELDS:
        src_file = resolve_path(record[field], input_root)

        if not src_file.is_file():
            raise FileNotFoundError(
                f"File for field '{field}' does not exist in {record_desc}: {src_file}"
            )

        source_files[field] = src_file

    return source_files


def get_unique_sample_dir(
    source_files: Dict[str, Path],
    record_desc: str,
) -> Path:
    """
    检查 rgb / depth / label_txt / mindrove 是否来自同一个样本文件夹。

    例如：
        train/sample_0001/rgb.pt
        train/sample_0001/depth.pt
        train/sample_0001/label.txt
        train/sample_0001/mindrove.pt

    这些文件的 parent 都应该是 train/sample_0001。
    """
    parent_dirs = {path.parent.resolve() for path in source_files.values()}

    if len(parent_dirs) != 1:
        msg = "\n".join(str(p) for p in sorted(parent_dirs))
        raise ValueError(
            f"Path fields are not located in the same sample directory in {record_desc}.\n"
            f"Detected parent directories:\n{msg}"
        )

    return next(iter(parent_dirs))


def make_sample_name(index: int, num_digits: int) -> str:
    return f"sample_{index:0{num_digits}d}"


def update_manifest_record_paths(
    record: Dict[str, Any],
    new_sample_name: str,
    merged_dataset_name: str,
    source_files: Dict[str, Path],
) -> Dict[str, Any]:
    """
    生成新的 manifest record。

    只修改：
        sample_name
        rgb
        depth
        label_txt
        mindrove

    其他字段保持原样。
    """
    new_record = dict(record)

    new_record["sample_name"] = new_sample_name

    for field in PATH_FIELDS:
        filename = source_files[field].name

        # 使用 PurePosixPath 保证 manifest 中始终使用 /，避免 Windows 下写成反斜杠。
        new_relative_path = PurePosixPath(
            merged_dataset_name,
            new_sample_name,
            filename,
        )

        new_record[field] = str(new_relative_path)

    return new_record


def update_label_txt_sample_id(
    target_sample_dir: Path,
    new_sample_name: str,
) -> None:
    """
    修改合并后样本目录中的 label.txt。

    原始 label.txt 中通常包含：

        sample_name: sample_0569

    合并后需要改成：

        sample_id: sample_0001

    只允许修改以 sample_name: 开头的那一行。
    如果没有找到 sample_name:，则直接报错。
    """
    label_path = target_sample_dir / "label.txt"

    if not label_path.is_file():
        raise FileNotFoundError(f"label.txt does not exist: {label_path}")

    with label_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    found = False
    new_lines = []

    for line in lines:
        if line.startswith("sample_name:"):
            new_lines.append(f"sample_id: {new_sample_name}\n")
            found = True
        else:
            new_lines.append(line)

    if not found:
        raise ValueError(f"No 'sample_name:' line found in label.txt: {label_path}")

    with label_path.open("w", encoding="utf-8", newline="") as f:
        f.writelines(new_lines)


def prepare_output_paths(
    output_root: Path,
    merged_dataset_name: str,
    merged_manifest_name: str,
    overwrite: bool,
) -> Tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)

    merged_dataset_dir = output_root / merged_dataset_name
    merged_manifest_path = output_root / merged_manifest_name

    if merged_dataset_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset directory already exists: {merged_dataset_dir}\n"
                f"Use --overwrite if you want to remove it and regenerate."
            )
        shutil.rmtree(merged_dataset_dir)

    if merged_manifest_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output manifest already exists: {merged_manifest_path}\n"
                f"Use --overwrite if you want to remove it and regenerate."
            )
        merged_manifest_path.unlink()

    merged_dataset_dir.mkdir(parents=True, exist_ok=False)

    return merged_dataset_dir, merged_manifest_path


def merge_one_record(
    record: Dict[str, Any],
    input_root: Path,
    merged_dataset_dir: Path,
    merged_dataset_name: str,
    new_sample_name: str,
    record_desc: str,
) -> Dict[str, Any]:
    """
    合并单个样本：
    1. 检查字段；
    2. 解析源文件；
    3. 检查四个路径是否位于同一个样本文件夹；
    4. 复制整个样本文件夹；
    5. 更新 manifest record 中的路径。
    """
    check_required_fields(record, record_desc)

    source_files = get_source_files(
        record=record,
        input_root=input_root,
        record_desc=record_desc,
    )

    source_sample_dir = get_unique_sample_dir(
        source_files=source_files,
        record_desc=record_desc,
    )

    target_sample_dir = merged_dataset_dir / new_sample_name

    if target_sample_dir.exists():
        raise FileExistsError(f"Target sample directory already exists: {target_sample_dir}")

    shutil.copytree(source_sample_dir, target_sample_dir, copy_function=shutil.copy2)

    # 复制完成后，严格检查目标文件是否存在。
    for field, src_file in source_files.items():
        target_file = target_sample_dir / src_file.name
        if not target_file.is_file():
            raise FileNotFoundError(
                f"Copied file is missing for field '{field}' in {record_desc}: "
                f"{target_file}"
            )
        
    update_label_txt_sample_id(
    target_sample_dir=target_sample_dir,
    new_sample_name=new_sample_name,
    )

    new_record = update_manifest_record_paths(
        record=record,
        new_sample_name=new_sample_name,
        merged_dataset_name=merged_dataset_name,
        source_files=source_files,
    )

    return new_record


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root does not exist or is not a directory: {input_root}")

    train_manifest = resolve_path(args.train_manifest, input_root).resolve()
    val_manifest = resolve_path(args.val_manifest, input_root).resolve()
    test_manifest = resolve_path(args.test_manifest, input_root).resolve()

    merged_dataset_dir, merged_manifest_path = prepare_output_paths(
        output_root=output_root,
        merged_dataset_name=args.merged_dataset_name,
        merged_manifest_name=args.merged_manifest_name,
        overwrite=args.overwrite,
    )

    split_manifests = [
        ("train", train_manifest),
        ("val", val_manifest),
        ("test", test_manifest),
    ]

    all_new_records: List[Dict[str, Any]] = []

    next_index = args.start_index
    split_counts: Dict[str, int] = {}

    for split_name, manifest_path in split_manifests:
        records = load_jsonl(manifest_path)
        split_counts[split_name] = len(records)

        for local_idx, record in enumerate(records, start=1):
            new_sample_name = make_sample_name(next_index, args.num_digits)

            record_desc = (
                f"split={split_name}, "
                f"manifest={manifest_path}, "
                f"line={local_idx}, "
                f"old_sample_name={record.get('sample_name', '<missing>')}, "
                f"new_sample_name={new_sample_name}"
            )

            new_record = merge_one_record(
                record=record,
                input_root=input_root,
                merged_dataset_dir=merged_dataset_dir,
                merged_dataset_name=args.merged_dataset_name,
                new_sample_name=new_sample_name,
                record_desc=record_desc,
            )

            all_new_records.append(new_record)
            next_index += 1

    with merged_manifest_path.open("w", encoding="utf-8") as f:
        for record in all_new_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_count = len(all_new_records)

    print("Merge completed successfully.")
    print(f"Input root: {input_root}")
    print(f"Output dataset: {merged_dataset_dir}")
    print(f"Output manifest: {merged_manifest_path}")
    print(f"Train samples: {split_counts.get('train', 0)}")
    print(f"Val samples: {split_counts.get('val', 0)}")
    print(f"Test samples: {split_counts.get('test', 0)}")
    print(f"Total merged samples: {total_count}")

    if total_count > 0:
        first_sample = make_sample_name(args.start_index, args.num_digits)
        last_sample = make_sample_name(args.start_index + total_count - 1, args.num_digits)
        print(f"New sample range: {first_sample} -> {last_sample}")


if __name__ == "__main__":
    main()