from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ASSETS = HERE / "report_assets"
ASSETS.mkdir(parents=True, exist_ok=True)
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")

NAVY = "#17324D"
BLUE = "#2E74B5"
TEAL = "#2A9D8F"
GOLD = "#E9A23B"
RED = "#C94C4C"
GRAY = "#6B7280"
LIGHT = "#E8EEF5"
GRID = "#D7DEE8"
WHITE = "#FFFFFF"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = Path(r"C:\Windows\Fonts\msyhbd.ttc") if bold else FONT_PATH
    return ImageFont.truetype(str(path), size=size)


def canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1600, 900), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((90, 55), title, fill=NAVY, font=font(42, True))
    draw.text((90, 112), subtitle, fill=GRAY, font=font(23))
    return image, draw


def save(image: Image.Image, name: str) -> None:
    image.save(ASSETS / name, dpi=(160, 160))


def chart_order_sensitivity() -> None:
    df = pd.read_csv(HERE / "order_sensitivity_summary.csv")
    native = df[(df.analysis_scope == "graph_permutation_changed_only") & (df.variant == "graph_valid_native")]
    other = df[(df.analysis_scope == "eligible_history_len_ge_2") & df.variant.isin(["random_presented", "reverse_presented"])]
    models = ["M2-Direct-RealOrder", "A1-Legacy-Once", "A3-DualPos-Once"]
    labels = ["M2 RealOrder", "A1 Once", "DualPos Once"]
    series = {
        "Graph-valid native\n(changed only)": [100 * float(native[native.model == model].tier3_top1_change_rate.iloc[0]) for model in models],
        "Random presented": [100 * float(other[(other.model == model) & (other.variant == "random_presented")].tier3_top1_change_rate.iloc[0]) for model in models],
        "Reverse presented": [100 * float(other[(other.model == model) & (other.variant == "reverse_presented")].tier3_top1_change_rate.iloc[0]) for model in models],
    }
    image, draw = canvas("测试时顺序敏感性", "Tier3 top-1 改变率；graph-valid 只统计实际发生换序的样本")
    left, top, right, bottom = 150, 220, 1500, 760
    maximum = 24.0
    for tick in range(0, 25, 5):
        y = bottom - (bottom - top) * tick / maximum
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((82, y - 14), f"{tick}%", fill=GRAY, font=font(20))
    colors = [BLUE, GOLD, RED]
    group_width = (right - left) / len(models)
    bar_width = 105
    gap = 18
    for i, label in enumerate(labels):
        center = left + group_width * (i + 0.5)
        for j, (name, values) in enumerate(series.items()):
            value = values[i]
            x0 = center + (j - 1) * (bar_width + gap) - bar_width / 2
            y0 = bottom - (bottom - top) * value / maximum
            draw.rounded_rectangle((x0, y0, x0 + bar_width, bottom), radius=8, fill=colors[j])
            draw.text((x0 + bar_width / 2, y0 - 30), f"{value:.2f}%", anchor="mm", fill=colors[j], font=font(19, True))
        draw.text((center, bottom + 32), label, anchor="ma", fill=NAVY, font=font(23, True))
    legend_x = 240
    for j, name in enumerate(series):
        x = legend_x + j * 430
        draw.rounded_rectangle((x, 820, x + 28, 848), radius=5, fill=colors[j])
        draw.multiline_text((x + 40, 808), name, fill=NAVY, font=font(18), spacing=2)
    save(image, "01_order_sensitivity.png")


def chart_group_deltas() -> None:
    df = pd.read_csv(HERE / "grouped_performance_model_deltas_summary.csv")
    df = df[(df.comparison_model == "A1-Legacy-Once") & (df.condition == "all")]
    keys = [
        ("local_prefix_3", "seen", "Local-3 seen"),
        ("local_prefix_3", "unseen", "Local-3 unseen"),
        ("history_length", "3-5", "History 3-5"),
        ("history_length", "6-10", "History 6-10"),
        ("history_length", "11-20", "History 11-20"),
        ("stage", "1", "Stage 1"),
        ("stage", "2", "Stage 2"),
        ("stage", "3", "Stage 3"),
        ("active_tail", "True", "Active tail"),
        ("active_tail", "False", "No active tail"),
    ]
    values = []
    for grouping, group, label in keys:
        row = df[(df.grouping == grouping) & (df.group.astype(str) == group)].iloc[0]
        values.append((label, 100 * float(row.tier3_accuracy_delta_mean)))
    image, draw = canvas("A1-Legacy-Once 相对 M2 的分组增益", "All split，Tier3 Accuracy 配对 fold×seed 平均差（百分点）")
    left, center, right = 430, 850, 1490
    top, row_h = 205, 59
    scale = 120
    draw.line((center, top - 10, center, top + row_h * len(values)), fill=NAVY, width=3)
    for index, (label, value) in enumerate(values):
        y = top + index * row_h
        draw.text((left - 25, y + 15), label, anchor="rm", fill=NAVY, font=font(23))
        width = abs(value) * scale
        color = TEAL if value >= 0 else RED
        x0, x1 = (center, center + width) if value >= 0 else (center - width, center)
        draw.rounded_rectangle((x0, y, x1, y + 34), radius=7, fill=color)
        draw.text((x1 + 12 if value >= 0 else x0 - 12, y + 17), f"{value:+.2f} pp", anchor="lm" if value >= 0 else "rm", fill=color, font=font(21, True))
    draw.text((center, 825), "0", anchor="mm", fill=GRAY, font=font(18))
    save(image, "02_group_deltas.png")


def chart_augmentation_quality() -> None:
    df = pd.read_csv(HERE / "augmentation_history_audit_summary.csv")
    df = df[df.analysis_scope == "changed_views_only"].groupby("model").mean(numeric_only=True)
    models = ["A1-Legacy-Once", "A1-Legacy-Every10-Replace", "A3-DualPos-Once", "A3-DualPos-Every10"]
    labels = ["A1 Once", "A1 Every10", "DualPos Once", "DualPos Every10"]
    novel = [100 * float(df.loc[model, "novel_transition_fraction"]) for model in models]
    real = [100 * float(df.loc[model, "real_prefix_fraction"]) for model in models]
    image, draw = canvas("增强历史：图合法不等于经验真实", "仅统计真正换序的视图；相邻转移和真实训练 prefix 支持率")
    left, top, right, bottom = 160, 220, 1500, 750
    maximum = 30.0
    for tick in range(0, 31, 5):
        y = bottom - (bottom - top) * tick / maximum
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((88, y - 13), f"{tick}%", fill=GRAY, font=font(20))
    group_width = (right - left) / len(models)
    for i, label in enumerate(labels):
        center = left + group_width * (i + 0.5)
        for j, (value, color) in enumerate(((novel[i], RED), (real[i], BLUE))):
            x0 = center + (j - 0.5) * 105 - 42
            y0 = bottom - (bottom - top) * value / maximum
            draw.rounded_rectangle((x0, y0, x0 + 84, bottom), radius=7, fill=color)
            draw.text((x0 + 42, y0 - 28), f"{value:.1f}%", anchor="mm", fill=color, font=font(19, True))
        draw.text((center, bottom + 35), label, anchor="ma", fill=NAVY, font=font(21, True))
    for x, color, label in ((500, RED, "未在训练折出现的相邻转移"), (950, BLUE, "等于某个真实训练 prefix")):
        draw.rounded_rectangle((x, 835, x + 28, 863), radius=5, fill=color)
        draw.text((x + 40, 837), label, fill=NAVY, font=font(19))
    save(image, "03_augmentation_quality.png")


def chart_diversity() -> None:
    df = pd.read_csv(HERE / "augmentation_diversity_by_sample.csv")
    summary = df.groupby("model").agg(views=("views", "mean"), unique=("unique_sequences", "mean"), duplicate=("duplicate_ratio", "mean"), unchanged=("all_unchanged", "mean"))
    models = ["A1-Legacy-Once", "A1-Legacy-Every10-Replace", "A3-DualPos-Once", "A3-DualPos-Every10"]
    labels = ["A1 Once", "A1 Every10", "DualPos Once", "DualPos Every10"]
    image, draw = canvas("增强视图的有效多样性", "每个训练样本跨 seed/refresh 的平均唯一历史数；括号为生成视图数")
    left, top, right, bottom = 180, 220, 1490, 750
    maximum = 9.0
    for tick in range(0, 10, 1):
        y = bottom - (bottom - top) * tick / maximum
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((125, y - 12), str(tick), fill=GRAY, font=font(18))
    group_width = (right - left) / len(models)
    for i, (model, label) in enumerate(zip(models, labels)):
        center = left + group_width * (i + 0.5)
        value = float(summary.loc[model, "unique"])
        views = int(round(summary.loc[model, "views"]))
        y0 = bottom - (bottom - top) * value / maximum
        draw.rounded_rectangle((center - 68, y0, center + 68, bottom), radius=10, fill=BLUE if "A1" in model else TEAL)
        draw.text((center, y0 - 34), f"{value:.2f} / {views}", anchor="mm", fill=NAVY, font=font(23, True))
        draw.text((center, bottom + 35), label, anchor="ma", fill=NAVY, font=font(21, True))
        draw.text((center, bottom + 82), f"重复 {100*summary.loc[model, 'duplicate']:.1f}%", anchor="ma", fill=RED, font=font(18))
    save(image, "04_diversity.png")


def chart_test_composition() -> None:
    df = pd.read_csv(HERE / "test_group_metadata.csv")
    total = len(df)
    local = df.local_prefix3_status.value_counts()
    exact = df.exact_full_prefix_seen.value_counts()
    active = df.active_tail.value_counts()
    rows = [
        ("Exact full history prefix", [("Seen", exact.get(True, 0)), ("Unseen", exact.get(False, 0))]),
        ("Local-3 prefix", [("Seen", local.get("seen", 0)), ("Unseen", local.get("unseen", 0)), ("Length < 3", local.get("insufficient", 0))]),
        ("Active tail", [("Yes", active.get(True, 0)), ("No", active.get(False, 0))]),
    ]
    colors = [BLUE, RED, GOLD]
    image, draw = canvas("测试样本的局部顺序覆盖", f"四个 held-out fold 共 {total} 个唯一测试 clip；不重复计算 seed")
    left, right = 430, 1480
    bar_w = right - left
    for row_index, (label, parts) in enumerate(rows):
        y = 260 + row_index * 175
        draw.text((left - 35, y + 35), label, anchor="rm", fill=NAVY, font=font(25, True))
        cursor = left
        for index, (name, count) in enumerate(parts):
            width = bar_w * count / total
            draw.rectangle((cursor, y, cursor + width, y + 70), fill=colors[index])
            if width > 110:
                draw.text((cursor + width / 2, y + 35), f"{name}\n{count/total:.1%}", anchor="mm", fill=WHITE, font=font(19, True), align="center")
            elif width > 55:
                draw.text((cursor + width / 2, y + 35), f"{count/total:.1%}", anchor="mm", fill=WHITE, font=font(16, True))
            cursor += width
    draw.text((90, 810), "Local-3 = 当前动作之前紧邻的最后 3 个历史节点；Length < 3 单列。", fill=GRAY, font=font(20))
    save(image, "05_test_composition.png")


def main() -> None:
    chart_order_sensitivity()
    chart_group_deltas()
    chart_augmentation_quality()
    chart_diversity()
    chart_test_composition()
    print(ASSETS)


if __name__ == "__main__":
    main()
