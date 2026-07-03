#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
temporal_augmentation.py

目标
====
提供两个最简洁、可直接给你现有 dataset / dataloader 使用的时间采样函数：

1) sample_indices_strict(T, n)
   - 单视角使用
   - 全局均匀采样
   - 当 T < n 时，不再 last-frame padding
   - 改为用 linspace 做“全局均匀上采样”

2) sample_two_views_indices(T, n, rng=None)
   - two-view 使用
   - 当 T <= n 时：
       view1 = 全局均匀上采样
       view2 = 在 view1 基础上做轻微抖动
   - 当 T > n 时：
       两个 view 各自独立做 adaptive temporal sampling：
         a) 随机决定 span
         b) 随机决定 start
         c) 在 span 内分成 n 个小区间，每段随机取一帧
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np


def _check_inputs(T: int, n: int) -> None:
    if T <= 0:
        raise ValueError(f"T must be > 0, but got T={T}")
    if n <= 0:
        raise ValueError(f"n must be > 0, but got n={n}")


def _clip_and_sort(idxs: List[int], T: int) -> List[int]:
    """
    保证索引不越界，并且时间顺序非递减。
    """
    arr = np.asarray(idxs, dtype=np.int64)
    arr = np.clip(arr, 0, T - 1)
    arr = np.sort(arr)
    return arr.astype(int).tolist()


def sample_indices_strict(T: int, n: int) -> List[int]:
    """
    单视角全局均匀采样。

    新规则：
    - T >= n: 在整段视频上均匀采 n 帧
    - T <  n: 不做 last-frame padding，而是用 linspace 全局均匀上采样

    这样短视频不会出现后面一大段都重复最后一帧的问题。
    """
    _check_inputs(T, n)

    # 统一都用 linspace：长视频=下采样，短视频=上采样
    idxs = np.linspace(0, T - 1, n).astype(int).tolist()
    return idxs


def _sample_one_adaptive_view(T: int, n: int, rng: random.Random) -> List[int]:
    """
    从长度为 T 的视频中采一个长度为 n 的自适应 temporal view。

    做法：
    1) 随机选 span ∈ [n, T]
    2) 随机选 start
    3) 在 [start, start+span) 内分成 n 段
    4) 每段随机采一帧
    """
    span = rng.randint(n, T)
    start = rng.randint(0, T - span)
    end_exclusive = start + span

    # 用 [start, end_exclusive] 分成 n 个区间边界
    ticks = np.linspace(start, end_exclusive, n + 1)

    idxs: List[int] = []
    for i in range(n):
        low = int(np.floor(ticks[i]))
        high = int(np.floor(ticks[i + 1]))

        # 当前区间实际采样范围为 [low, high-1]
        if high <= low:
            idxs.append(min(low, T - 1))
        else:
            idxs.append(rng.randint(low, min(high, T) - 1))

    return _clip_and_sort(idxs, T)


def sample_two_views_indices(
    T: int,
    n: int,
    rng: Optional[random.Random] = None,
) -> Tuple[List[int], List[int]]:
    """
    生成 two-view 时间采样索引。

    参数
    ----
    T : 视频总帧数
    n : 每个 view 需要采的帧数
    rng : 可选随机数生成器；不传则使用全局 random

    返回
    ----
    idxs1, idxs2 : 两个长度均为 n 的索引列表
    """
    _check_inputs(T, n)

    if rng is None:
        rng = random

    # --------------------------------------------------
    # 短视频：全局均匀上采样 + 轻微抖动
    # --------------------------------------------------
    if T <= n:
        idxs1 = sample_indices_strict(T, n)

        idxs2: List[int] = []
        for v in idxs1:
            # 50% 概率做轻微偏移
            if rng.random() > 0.5:
                shift = rng.choice([-1, 0, 1])
            else:
                shift = 0

            idxs2.append(max(0, min(T - 1, v + shift)))

        idxs2 = _clip_and_sort(idxs2, T)
        return idxs1, idxs2

    # --------------------------------------------------
    # 长视频：两个 view 独立做 adaptive temporal sampling
    # --------------------------------------------------
    idxs1 = _sample_one_adaptive_view(T, n, rng)
    idxs2 = _sample_one_adaptive_view(T, n, rng)

    return idxs1, idxs2


if __name__ == "__main__":
    r = random.Random(0)

    for T in [8, 12, 16, 20, 80]:
        a = sample_indices_strict(T, 16)
        b1, b2 = sample_two_views_indices(T, 16, rng=r)

        print(f"\nT={T}")
        print("single:", a)
        print("view1 :", b1)
        print("view2 :", b2)