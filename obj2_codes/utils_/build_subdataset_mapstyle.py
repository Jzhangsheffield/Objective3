#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
subset_split_jsonl_sampler.py
------------------------------------------------------------
从 JSONL 样本清单中抽取一个子集，逻辑与之前的 JSON 版脚本保持一致，
只是把输入/输出格式改成了 JSONL（每行一个样本 dict）。

============================================================
一、支持的输入格式
============================================================
输入文件是 jsonl，每一行类似：

{"sample_name": "sample_0001", "original_key": "...", "person": "J",
 "tier1": "adjust", "tier2": "adjust_slider", "tier3": "adjust_slider",
 "lighting": "right", "pos": "elbow", "rgb": "...", "depth": "...", ...}

也就是说：
- 整个文件不是一个大 JSON dict
- 而是“每行一个样本记录”

============================================================
二、支持的功能
============================================================
1) 可按 tier1 / tier2 / tier3 其中之一作为类别进行抽样
   --label_level tier1|tier2|tier3

2) 可指定每个类别需要保留多少个样本
   --per_class "adjust=100,take=100,put=100"

   规则与之前保持一致：
   - 没指定的类：默认“全要”
   - 如果指定数量 > 该类实际可用数：则该类全部保留
   - 不会补伪样本

3) person 维度：
   --person_mode all      : 不限制
   --person_mode select   : 只保留指定 persons
   --person_mode uniform  : 尽可能均匀覆盖不同 person

4) env 维度（定义为 lighting + "_" + pos）：
   --env_mode all
   --env_mode select
   --env_mode uniform

   例如：
   - right_elbow
   - left_mid
   - normal_elbow

5) 可复现：
   --seed 固定随机种子

6) 输出：
   - 新的 subset jsonl
   - report.txt（抽样前后类别 / person / env 分布）

============================================================
三、均匀抽样的定义（与之前一致）
============================================================
当 person_mode=uniform / env_mode=uniform 时，本脚本采用：
“分层 + round-robin” 的尽可能均匀抽样方式。

- 如果只开 person uniform：
    按 person 分层
- 如果只开 env uniform：
    按 env 分层
- 如果两个都开：
    按 (person, env) 联合分层

然后：
- 每个层内部先随机打乱
- 再轮流从每个层取 1 个样本
- 直到达到目标数量，或者可用样本耗尽

这样做的目的，是尽量避免某些 person / env 样本很多时被“选爆”，
从而使子集尽可能保持均衡。

============================================================
四、使用示例
============================================================

(1) 按 tier1 抽样，对 adjust / take / put 限制数量，其余类全要；
    并尽量让 person 和 env 都均匀：
python subset_split_jsonl_sampler.py ^
  --in_jsonl "F:\\data\\train.jsonl" ^
  --out_jsonl "F:\\data\\train_subset.jsonl" ^
  --label_level tier1 ^
  --per_class "adjust=100,take=100,put=100" ^
  --person_mode uniform ^
  --env_mode uniform ^
  --seed 42

(2) 只保留 J,N 两个参与者，只保留 right_elbow 和 left_mid 两种环境；
    按 tier2 每类最多取 50：
python subset_split_jsonl_sampler.py ^
  --in_jsonl "F:\\data\\train.jsonl" ^
  --out_jsonl "F:\\data\\train_subset_selected.jsonl" ^
  --label_level tier2 ^
  --per_class "adjust_slider=50,insert_wire=50" ^
  --person_mode select --persons "J,N" ^
  --env_mode select --envs "right_elbow,left_mid" ^
  --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set


# ============================================================
# 0) 基础 IO
# ============================================================

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    读取 JSONL 文件。

    返回：
        records: List[dict]
    其中每个元素就是一行样本记录。
    """
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 解析失败：第 {line_idx} 行，错误：{e}") from e

            if not isinstance(obj, dict):
                raise ValueError(f"JSONL 第 {line_idx} 行不是 dict。")
            records.append(obj)
    return records


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    """
    将记录列表写回 JSONL。
    每个样本一行 JSON。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1) 参数解析
# ============================================================

def parse_per_class_spec(spec: str) -> Dict[str, int]:
    """
    解析：
        --per_class "adjust=100,take=100,put=100"

    返回：
        {"adjust": 100, "take": 100, "put": 100}
    """
    out: Dict[str, int] = {}
    spec = (spec or "").strip()
    if not spec:
        return out

    parts = [x.strip() for x in spec.split(",") if x.strip()]
    for item in parts:
        if "=" not in item:
            raise ValueError(f"--per_class 格式错误：{item}，应为 name=count")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()

        if not k:
            raise ValueError(f"--per_class 中类别名为空：{item}")

        try:
            n = int(v)
        except ValueError as e:
            raise ValueError(f"--per_class 中数量不是整数：{item}") from e

        if n < 0:
            raise ValueError(f"--per_class 中数量必须 >= 0：{item}")

        out[k] = n

    return out


def parse_csv_list(s: str) -> List[str]:
    """
    解析逗号分隔字符串，例如：
        "J,N,MR"
        "right_elbow,left_mid"
    """
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--in_jsonl", type=str, required=True, help="输入 JSONL 路径")
    ap.add_argument("--out_jsonl", type=str, required=True, help="输出 subset JSONL 路径")
    ap.add_argument("--out_dir", type=str, default="", help="输出目录；默认使用 out_jsonl 所在目录")

    ap.add_argument(
        "--label_level",
        type=str,
        default="tier1",
        choices=["tier1", "tier2", "tier3"],
        help="使用哪个字段作为类别标签"
    )

    ap.add_argument(
        "--per_class",
        type=str,
        default="",
        help='指定每类保留数量，如 "adjust=100,take=100"；未指定的类默认全要'
    )

    ap.add_argument(
        "--person_mode",
        type=str,
        default="all",
        choices=["all", "select", "uniform"],
        help="person 策略：all / select / uniform"
    )
    ap.add_argument(
        "--persons",
        type=str,
        default="",
        help='person_mode=select 时生效，例如 "J,N,MR"'
    )

    ap.add_argument(
        "--env_mode",
        type=str,
        default="all",
        choices=["all", "select", "uniform"],
        help="env 策略：all / select / uniform"
    )
    ap.add_argument(
        "--envs",
        type=str,
        default="",
        help='env_mode=select 时生效，例如 "right_elbow,left_mid,right_mid"'
    )

    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--report_name", type=str, default="subset_report.txt", help="报告文件名")

    return ap


# ============================================================
# 2) 样本字段提取
# ============================================================

def get_sample_key(sample_entry: Dict[str, Any], fallback_idx: int) -> str:
    """
    取得样本唯一 key。

    优先使用：
        1) sample_name
        2) original_key
        3) 自动生成 line_xxxxxx

    这样做的目的是：
    - 后续抽样过程内部要有一个稳定的 key
    - 输出 JSONL 时仍然写原始 record，不会改动字段内容
    """
    if "sample_name" in sample_entry and sample_entry["sample_name"] is not None:
        return str(sample_entry["sample_name"])
    if "original_key" in sample_entry and sample_entry["original_key"] is not None:
        return str(sample_entry["original_key"])
    return f"line_{fallback_idx:06d}"


def get_label(sample_entry: Dict[str, Any], label_level: str) -> str:
    """
    取 tier1 / tier2 / tier3 作为类别名。
    """
    v = sample_entry.get(label_level, None)
    if v is None:
        raise KeyError(f"sample 缺少字段 {label_level}")
    return str(v)


def get_person(sample_entry: Dict[str, Any]) -> str:
    """
    取 person 字段，例如 J / N / MR / M。
    """
    p = sample_entry.get("person", None)
    if p is None:
        raise KeyError("sample 缺少字段 person")
    return str(p)


def get_env(sample_entry: Dict[str, Any]) -> str:
    """
    环境定义为 lighting_pos，例如：
        right_elbow
        left_mid
        normal_elbow
    """
    lighting = sample_entry.get("lighting", None)
    pos = sample_entry.get("pos", None)
    if lighting is None or pos is None:
        raise KeyError("sample 缺少字段 lighting 或 pos")
    return f"{lighting}_{pos}"


# ============================================================
# 3) 过滤逻辑
# ============================================================

def filter_by_person_and_env(
    items: List[Tuple[str, Dict[str, Any]]],
    person_mode: str,
    persons_select: List[str],
    env_mode: str,
    envs_select: List[str],
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    先对 select 模式做“硬过滤”。

    说明：
    - person_mode=select 时，只保留指定 persons
    - env_mode=select 时，只保留指定 envs
    - uniform 模式不在这里过滤，而是在后面的抽样过程中尽量均衡

    输入：
        items: [(sample_key, sample_entry), ...]

    输出：
        过滤后的 items
    """
    out: List[Tuple[str, Dict[str, Any]]] = []
    persons_set = set(persons_select)
    envs_set = set(envs_select)

    for k, e in items:
        if person_mode == "select":
            if get_person(e) not in persons_set:
                continue

        if env_mode == "select":
            if get_env(e) not in envs_set:
                continue

        out.append((k, e))

    return out


# ============================================================
# 4) 分层 + round-robin 抽样
# ============================================================

def stratified_round_robin_sample(
    items: List[Tuple[str, Dict[str, Any]]],
    target_n: int,
    balance_person: bool,
    balance_env: bool,
    rng: random.Random,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    从一组同类别样本 items 中抽取 target_n 个，尽可能在指定维度上均匀。

    参数：
        items:
            当前类别下的样本列表，元素为 (sample_key, sample_entry)

        target_n:
            目标抽取数量

        balance_person:
            是否尽量在 person 维度均匀

        balance_env:
            是否尽量在 env 维度均匀

        rng:
            随机数发生器（使用固定 seed，保证可复现）

    分层策略：
        - 两个都 False：所有样本放在同一层，等价于普通随机采样
        - 只 balance_person：按 person 分层
        - 只 balance_env：按 env 分层
        - 两个都 True：按 (person, env) 联合分层

    抽取方式：
        1) 每层内部随机打乱
        2) active strata 轮流每层取 1 个
        3) 直到取满 target_n 或没有样本可取

    这样比简单 random.sample 更适合“尽量均匀覆盖多个条件”的场景。
    """
    if target_n <= 0:
        return []

    if target_n >= len(items):
        # 如果目标数量 >= 可用数量，则全部保留
        tmp = items[:]
        rng.shuffle(tmp)
        return tmp

    # ----------------------------------------
    # 1) 建立 strata
    # ----------------------------------------
    strata: Dict[Tuple[str, ...], List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)

    for k, e in items:
        key_parts: List[str] = []

        if balance_person:
            key_parts.append(get_person(e))
        if balance_env:
            key_parts.append(get_env(e))

        # 如果两个维度都不平衡，则全部放在同一个层里
        if key_parts:
            stratum_key = tuple(key_parts)
        else:
            stratum_key = ("__all__",)

        strata[stratum_key].append((k, e))

    # ----------------------------------------
    # 2) 每层内部打乱
    # ----------------------------------------
    for sk in list(strata.keys()):
        rng.shuffle(strata[sk])

    # ----------------------------------------
    # 3) round-robin 取样
    # ----------------------------------------
    selected: List[Tuple[str, Dict[str, Any]]] = []
    active_keys = list(strata.keys())

    while len(selected) < target_n and active_keys:
        new_active_keys: List[Tuple[str, ...]] = []

        for sk in active_keys:
            if len(selected) >= target_n:
                break

            bucket = strata[sk]
            if not bucket:
                continue

            # 每轮从该层取一个
            selected.append(bucket.pop())

            # 若该层还有剩余，则下一轮继续参与
            if bucket:
                new_active_keys.append(sk)

        active_keys = new_active_keys

    return selected


# ============================================================
# 5) 统计函数
# ============================================================

@dataclass
class DistStats:
    class_counter: Counter
    person_counter: Counter
    env_counter: Counter
    class_person_counter: Dict[str, Counter]
    class_env_counter: Dict[str, Counter]


def compute_stats(
    items: List[Tuple[str, Dict[str, Any]]],
    label_level: str
) -> DistStats:
    """
    统计类别 / person / env 分布，以及每个类别内部的 person/env 分布。
    """
    c_class = Counter()
    c_person = Counter()
    c_env = Counter()
    c_class_person: Dict[str, Counter] = defaultdict(Counter)
    c_class_env: Dict[str, Counter] = defaultdict(Counter)

    for _, e in items:
        cls = get_label(e, label_level)
        person = get_person(e)
        env = get_env(e)

        c_class[cls] += 1
        c_person[person] += 1
        c_env[env] += 1
        c_class_person[cls][person] += 1
        c_class_env[cls][env] += 1

    return DistStats(
        class_counter=c_class,
        person_counter=c_person,
        env_counter=c_env,
        class_person_counter=c_class_person,
        class_env_counter=c_class_env,
    )


def format_counter(counter: Counter, topk: int = 999999) -> str:
    """
    将 Counter 整理成易读字符串。
    """
    lines = []
    for k, v in counter.most_common(topk):
        lines.append(f"  {k}: {v}")
    return "\n".join(lines) if lines else "  (empty)"


def write_report(
    out_path: Path,
    before: DistStats,
    after: DistStats,
    label_level: str,
    per_class_targets: Dict[str, int],
    person_mode: str,
    persons_select: List[str],
    env_mode: str,
    envs_select: List[str],
    seed: int,
    num_before: int,
    num_after: int,
) -> None:
    """
    写抽样报告 txt。
    """
    with out_path.open("w", encoding="utf-8") as f:
        f.write("========== Subset Sampling Report (JSONL) ==========\n")
        f.write(f"label_level : {label_level}\n")
        f.write(f"seed        : {seed}\n")
        f.write(f"num_before  : {num_before}\n")
        f.write(f"num_after   : {num_after}\n\n")

        f.write("[Config] per_class_targets (未列出的类=默认全要)\n")
        if per_class_targets:
            for k, v in per_class_targets.items():
                f.write(f"  {k}: {v}\n")
        else:
            f.write("  (none)\n")
        f.write("\n")

        f.write(f"[Config] person_mode: {person_mode}\n")
        if person_mode == "select":
            f.write(f"  persons: {persons_select}\n")

        f.write(f"[Config] env_mode: {env_mode}\n")
        if env_mode == "select":
            f.write(f"  envs: {envs_select}\n")
        f.write("\n")

        f.write("---------- BEFORE ----------\n")
        f.write("[Class distribution]\n")
        f.write(format_counter(before.class_counter) + "\n\n")
        f.write("[Person distribution]\n")
        f.write(format_counter(before.person_counter) + "\n\n")
        f.write("[Env distribution]\n")
        f.write(format_counter(before.env_counter) + "\n\n")

        f.write("---------- AFTER ----------\n")
        f.write("[Class distribution]\n")
        f.write(format_counter(after.class_counter) + "\n\n")
        f.write("[Person distribution]\n")
        f.write(format_counter(after.person_counter) + "\n\n")
        f.write("[Env distribution]\n")
        f.write(format_counter(after.env_counter) + "\n\n")

        f.write("---------- Per-class breakdown (AFTER) ----------\n")
        for cls, n in after.class_counter.most_common():
            f.write(f"\n[Class: {cls}] total={n}\n")
            f.write("  persons:\n")
            f.write(format_counter(after.class_person_counter.get(cls, Counter())) + "\n")
            f.write("  envs:\n")
            f.write(format_counter(after.class_env_counter.get(cls, Counter())) + "\n")


# ============================================================
# 6) 核心抽样流程
# ============================================================

def build_subset(
    records: List[Dict[str, Any]],
    label_level: str,
    per_class_targets: Dict[str, int],
    person_mode: str,
    persons_select: List[str],
    env_mode: str,
    envs_select: List[str],
    seed: int,
) -> List[Dict[str, Any]]:
    """
    从 records 中构造子集，返回 subset_records。

    逻辑与之前版本一致：
    1) 先把 records 转成 [(sample_key, entry), ...]
    2) 对 select 模式做硬过滤
    3) 按类别分组
    4) 每个类别单独抽样
    5) 最后把被选中的样本按“原始输入顺序”输出

    为什么最后按原始顺序输出？
    - JSONL 本身是按行组织的
    - 保持原顺序通常更直观，也更利于后续排查
    """
    rng = random.Random(seed)

    # ----------------------------------------
    # 1) 转成 (key, entry) 列表
    # ----------------------------------------
    items_all: List[Tuple[str, Dict[str, Any]]] = []
    seen_keys: Set[str] = set()

    for idx, e in enumerate(records):
        k = get_sample_key(e, idx)

        if k in seen_keys:
            raise ValueError(f"发现重复样本 key：{k}。请确保 sample_name 唯一。")

        seen_keys.add(k)
        items_all.append((k, e))

    # ----------------------------------------
    # 2) 先做 select 的硬过滤
    # ----------------------------------------
    items_filtered = filter_by_person_and_env(
        items_all,
        person_mode=person_mode,
        persons_select=persons_select,
        env_mode=env_mode,
        envs_select=envs_select,
    )

    # ----------------------------------------
    # 3) 按 class 分组
    # ----------------------------------------
    by_class: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for k, e in items_filtered:
        cls = get_label(e, label_level)
        by_class[cls].append((k, e))

    # ----------------------------------------
    # 4) 每个类别内部做抽样
    # ----------------------------------------
    subset_key_set: Set[str] = set()

    balance_person = (person_mode == "uniform")
    balance_env = (env_mode == "uniform")

    # 按类别名排序，保证同 seed 下结果更稳定
    for cls in sorted(by_class.keys()):
        group = by_class[cls]

        # 未指定该类数量 -> 默认全要
        if cls not in per_class_targets:
            target_n = len(group)
        else:
            target_n = per_class_targets[cls]

        # 组内先打乱，再做分层抽样
        tmp = group[:]
        rng.shuffle(tmp)

        chosen = stratified_round_robin_sample(
            items=tmp,
            target_n=target_n,
            balance_person=balance_person,
            balance_env=balance_env,
            rng=rng,
        )

        for k, _ in chosen:
            subset_key_set.add(k)

    # ----------------------------------------
    # 5) 按原始顺序输出被选中的记录
    # ----------------------------------------
    subset_records: List[Dict[str, Any]] = []
    for idx, e in enumerate(records):
        k = get_sample_key(e, idx)
        if k in subset_key_set:
            subset_records.append(e)

    return subset_records


# ============================================================
# 7) main
# ============================================================

def main():
    args = build_argparser().parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)

    per_class_targets = parse_per_class_spec(args.per_class)
    persons_select = parse_csv_list(args.persons)
    envs_select = parse_csv_list(args.envs)

    # 输出目录
    if args.out_dir.strip():
        out_dir = Path(args.out_dir)
    else:
        out_dir = out_path.parent
    ensure_dir(out_dir)

    # ----------------------------------------
    # 1) 读取 JSONL
    # ----------------------------------------
    records = read_jsonl(in_path)

    # ----------------------------------------
    # 2) 统计抽样前分布
    #    注意：这里统计的是“原始全量 records”
    # ----------------------------------------
    items_before = [(get_sample_key(e, i), e) for i, e in enumerate(records)]
    before_stats = compute_stats(items_before, args.label_level)

    # ----------------------------------------
    # 3) 构造子集
    # ----------------------------------------
    subset_records = build_subset(
        records=records,
        label_level=args.label_level,
        per_class_targets=per_class_targets,
        person_mode=args.person_mode,
        persons_select=persons_select,
        env_mode=args.env_mode,
        envs_select=envs_select,
        seed=args.seed,
    )

    # ----------------------------------------
    # 4) 统计抽样后分布
    # ----------------------------------------
    items_after = [(get_sample_key(e, i), e) for i, e in enumerate(subset_records)]
    after_stats = compute_stats(items_after, args.label_level)

    # ----------------------------------------
    # 5) 写出子集 JSONL
    # ----------------------------------------
    write_jsonl(subset_records, out_path)

    # ----------------------------------------
    # 6) 写出报告
    # ----------------------------------------
    report_path = out_dir / args.report_name
    write_report(
        out_path=report_path,
        before=before_stats,
        after=after_stats,
        label_level=args.label_level,
        per_class_targets=per_class_targets,
        person_mode=args.person_mode,
        persons_select=persons_select,
        env_mode=args.env_mode,
        envs_select=envs_select,
        seed=args.seed,
        num_before=len(records),
        num_after=len(subset_records),
    )

    # ----------------------------------------
    # 7) 控制台输出
    # ----------------------------------------
    print("========== Done ==========")
    print("Input JSONL :", str(in_path))
    print("Output JSONL:", str(out_path))
    print("Report      :", str(report_path))
    print(f"Samples before: {len(records)}")
    print(f"Samples after : {len(subset_records)}")

    print("\n[AFTER] class distribution:")
    for k, v in after_stats.class_counter.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()