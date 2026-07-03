#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_sample_key_to_index.py

作用：
------------------------------------------------
从 train.jsonl / val.jsonl / test.jsonl 之类的样本清单中，
读取每一行的 "key" 字段，生成一个连续编号映射：

    sample_key -> compact_index   (0, 1, 2, ..., N-1)

这份映射可以在 WebDataset loader 中使用，
从而避免使用哈希 int64 作为 idx，导致 prototype 刷新阶段
按 max_index+1 分配超大张量而溢出。

输入：
------------------------------------------------
每行一个 JSON 对象，例如：
{
  "key": "J_open_wire_cutter_run_15_clip_000009_right_elbow",
  ...
}

输出：
------------------------------------------------
1) key_to_index.json
   形式：
   {
     "J_open_wire_cutter_run_15_clip_000009_right_elbow": 0,
     "J_open_wire_cutter_run_16_clip_000004_right_elbow": 1,
     ...
   }

2) index_to_key.json   （可选，方便调试）
   形式：
   [
     "J_open_wire_cutter_run_15_clip_000009_right_elbow",
     "J_open_wire_cutter_run_16_clip_000004_right_elbow",
     ...
   ]

3) key_to_index.csv    （可选，方便人工查看）
   列：
   index,key

使用示例：
------------------------------------------------
python build_sample_key_to_index.py ^
    --input_jsonl F:\\path\\to\\train_samples.jsonl ^
    --out_dir F:\\path\\to\\sample_index_map

如果你想把 train/val/test 全部统一编号，也可以传多个 jsonl：
python build_sample_key_to_index.py ^
    --input_jsonl train.jsonl val.jsonl test.jsonl ^
    --out_dir sample_index_map
"""

import os
import json
import csv
import argparse
from typing import List, Dict


def read_jsonl_keys(jsonl_paths: List[str]) -> List[str]:
    """
    从一个或多个 jsonl 文件中读取所有样本 key。

    规则：
    - 每行必须是合法 JSON
    - JSON 中必须包含 "key" 字段
    - 若遇到重复 key，默认只保留第一次出现的顺序
    """
    keys = []
    seen = set()

    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"JSON 解析失败: file={path}, line={line_num}, error={e}"
                    )

                if "key" not in obj:
                    raise KeyError(
                        f'缺少 "key" 字段: file={path}, line={line_num}'
                    )

                key = obj["key"]
                if not isinstance(key, str) or len(key) == 0:
                    raise ValueError(
                        f'非法 key: file={path}, line={line_num}, key={key}'
                    )

                if key not in seen:
                    seen.add(key)
                    keys.append(key)

    return keys


def build_key_to_index(keys: List[str]) -> Dict[str, int]:
    """
    按出现顺序生成连续映射：
        key -> 0..N-1
    """
    return {k: i for i, k in enumerate(keys)}


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(keys: List[str], path: str):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "key"])
        for i, k in enumerate(keys):
            writer.writerow([i, k])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        nargs="+",
        required=True,
        help="一个或多个输入 jsonl 文件路径"
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="输出目录"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    keys = read_jsonl_keys(args.input_jsonl)
    if len(keys) == 0:
        raise RuntimeError("没有从输入 jsonl 中读到任何 key。")

    key_to_index = build_key_to_index(keys)
    index_to_key = keys

    save_json(key_to_index, os.path.join(args.out_dir, "key_to_index.json"))
    save_json(index_to_key, os.path.join(args.out_dir, "index_to_key.json"))
    save_csv(keys, os.path.join(args.out_dir, "key_to_index.csv"))

    print("=" * 60)
    print(f"总样本数: {len(keys)}")
    print(f"映射已保存到: {args.out_dir}")
    print("生成文件:")
    print(" - key_to_index.json")
    print(" - index_to_key.json")
    print(" - key_to_index.csv")
    print("=" * 60)

    # 打印前几个样本，方便检查
    print("前 10 个映射示例:")
    for i, k in enumerate(keys[:10]):
        print(f"{i:6d}  {k}")


if __name__ == "__main__":
    main()