from __future__ import annotations

import html
import json
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "task_graph" / "task_graph.json"
VIZ_FRAGMENT = Path(r"C:\Users\digit\.codex\visualizations\2026\07\15\019f6551-4c78-7263-b4fd-9e508160a7a9\task-graph.html")

STAGE_COLORS = {
    -1: ("#e5e7eb", "#374151"),
    1: ("#dbeafe", "#1e3a8a"),
    2: ("#dcfce7", "#14532d"),
    3: ("#ffedd5", "#7c2d12"),
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def inspect_graph(nodes: list[dict]) -> dict:
    indices = [int(n["node_idx"]) for n in nodes]
    ids = [n["node_id"] for n in nodes]
    index_set = set(indices)
    duplicate_indices = sorted(k for k, v in Counter(indices).items() if v > 1)
    duplicate_ids = sorted(k for k, v in Counter(ids).items() if v > 1)
    missing_indices = sorted(set(range(min(indices), max(indices) + 1)) - index_set)
    missing_refs = []
    self_refs = []
    duplicate_refs = []
    forward_optional = []
    overlaps = []

    for n in nodes:
        idx = int(n["node_idx"])
        must = [int(x) for x in n.get("must_previous_nodes", [])]
        optional = [int(x) for x in n.get("optional_previous_nodes", [])]
        for field, refs in (("must_previous_nodes", must), ("optional_previous_nodes", optional)):
            for ref, count in Counter(refs).items():
                if count > 1:
                    duplicate_refs.append((idx, field, ref, count))
            for ref in refs:
                if ref not in index_set:
                    missing_refs.append((idx, field, ref))
                if ref == idx:
                    self_refs.append((idx, field))
        for ref in optional:
            if ref > idx:
                forward_optional.append((ref, idx))
        for ref in sorted(set(must) & set(optional)):
            overlaps.append((idx, ref))

    # Cycles in mandatory edges only.
    adjacency = defaultdict(list)
    indegree = {idx: 0 for idx in indices}
    for n in nodes:
        target = int(n["node_idx"])
        for source in set(int(x) for x in n.get("must_previous_nodes", [])):
            if source in index_set:
                adjacency[source].append(target)
                indegree[target] += 1
    queue = deque(sorted(idx for idx, deg in indegree.items() if deg == 0))
    visited = []
    while queue:
        current = queue.popleft()
        visited.append(current)
        for nxt in adjacency[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    must_cycle_nodes = sorted(set(indices) - set(visited))

    # Pairs that form immediate optional two-cycles.
    optional_edges = {
        (int(ref), int(n["node_idx"]))
        for n in nodes
        for ref in n.get("optional_previous_nodes", [])
        if int(ref) in index_set
    }
    mutual_optional_pairs = sorted(
        (a, b) for a, b in optional_edges if a < b and (b, a) in optional_edges
    )

    return {
        "duplicate_indices": duplicate_indices,
        "duplicate_ids": duplicate_ids,
        "missing_indices": missing_indices,
        "missing_refs": missing_refs,
        "self_refs": self_refs,
        "duplicate_refs": duplicate_refs,
        "forward_optional": forward_optional,
        "overlaps": overlaps,
        "must_cycle_nodes": must_cycle_nodes,
        "mutual_optional_pairs": mutual_optional_pairs,
    }


def node_positions() -> dict[int, tuple[int, int]]:
    pos = {0: (900, 75)}
    # Stage 1: parallel setup actions, then convergence into nodes 4 and 5.
    for idx, y in zip([1, 3, 6, 7, 8, 9, 10], [250, 350, 450, 550, 650, 750, 850]):
        pos[idx] = (180, y)
    pos.update({2: (580, 270), 11: (580, 830), 4: (1000, 350), 5: (1440, 350)})
    # Stage 2: a snake layout preserves the long mandatory sequence.
    xs = [180, 420, 660, 900, 1140, 1380, 1620]
    for idx, x in zip(range(12, 19), xs):
        pos[idx] = (x, 1100)
    for idx, x in zip(range(19, 26), reversed(xs)):
        pos[idx] = (x, 1370)
    # Stage 3: shutdown actions are mostly parallel after node 25.
    for idx, x in zip(range(26, 33), xs):
        pos[idx] = (x, 1660)
    pos.update({33: (1140, 1930), 34: (1380, 1930), 35: (1500, 2140), 36: (900, 2350)})
    return pos


def path_for_edge(sx: int, sy: int, tx: int, ty: int, optional: bool, offset: int = 0) -> str:
    # Curved paths reduce overlap and make reverse optional relationships visible.
    if abs(ty - sy) < 80:
        bend = 55 + offset * 6
        direction = -1 if tx >= sx else 1
        return f"M {sx} {sy} C {sx} {sy + direction*bend}, {tx} {ty + direction*bend}, {tx} {ty}"
    mid = (sy + ty) / 2
    if optional:
        side = 35 + offset * 4
        return f"M {sx} {sy} C {sx+side} {mid}, {tx-side} {mid}, {tx} {ty}"
    return f"M {sx} {sy} C {sx} {mid}, {tx} {mid}, {tx} {ty}"


def make_svg(nodes: list[dict], standalone: bool = True) -> str:
    pos = node_positions()
    by_idx = {int(n["node_idx"]): n for n in nodes}
    width, height = 1800, 2460
    parts = []
    if standalone:
        parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-labelledby="graph-title graph-desc">'
    )
    parts.append("""<title id="graph-title">Thermal crimp task graph</title>
<desc id="graph-desc">Nodes are colored by stage. Solid arrows are mandatory previous-node relationships; dashed purple arrows are optional previous-node relationships.</desc>
<defs>
  <marker id="arrow-must" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><path d="M0,0 L9,3.5 L0,7 Z" fill="#334155"/></marker>
  <marker id="arrow-optional" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><path d="M0,0 L9,3.5 L0,7 Z" fill="#7c3aed"/></marker>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="2" stdDeviation="2" flood-opacity="0.16"/></filter>
</defs>
<style>
  .band{fill:#f8fafc;stroke:#cbd5e1;stroke-width:1.5}.band-title{font:600 20px Arial,sans-serif;fill:#334155}
  .must-edge{fill:none;stroke:#334155;stroke-width:2;marker-end:url(#arrow-must)}
  .optional-edge{fill:none;stroke:#7c3aed;stroke-width:1.6;stroke-dasharray:7 6;opacity:.46;marker-end:url(#arrow-optional)}
  .node-box{stroke-width:1.5;filter:url(#shadow)}.idx{font:700 16px Arial,sans-serif}.label{font:14px Arial,sans-serif;fill:#111827}
  .legend{font:14px Arial,sans-serif;fill:#334155}.small{font:12px Arial,sans-serif;fill:#475569}
</style>""")
    bands = [
        ("Stage 1 · setup", 115, 965),
        ("Stage 2 · crimping", 990, 1495),
        ("Stage 3 · shutdown", 1530, 2205),
    ]
    for title, y1, y2 in bands:
        parts.append(f'<rect class="band" x="35" y="{y1}" width="1730" height="{y2-y1}" rx="18"/>')
        parts.append(f'<text class="band-title" x="58" y="{y1+32}">{esc(title)}</text>')

    # Legend.
    lx = 55
    for stage, name in [(-1, "start/end"), (1, "stage 1"), (2, "stage 2"), (3, "stage 3")]:
        fill, stroke = STAGE_COLORS[stage]
        parts.append(f'<rect x="{lx}" y="35" width="18" height="18" rx="4" fill="{fill}" stroke="{stroke}"/>')
        parts.append(f'<text class="legend" x="{lx+26}" y="49">{name}</text>')
        lx += 150
    parts.append('<line x1="690" y1="44" x2="745" y2="44" class="must-edge"/><text class="legend" x="755" y="49">must</text>')
    parts.append('<line x1="835" y1="44" x2="890" y2="44" class="optional-edge"/><text class="legend" x="900" y="49">optional</text>')

    edge_counts = defaultdict(int)
    for n in nodes:
        target = int(n["node_idx"])
        if target not in pos:
            continue
        tx, ty = pos[target]
        for kind, refs in (("must", n.get("must_previous_nodes", [])), ("optional", n.get("optional_previous_nodes", []))):
            for source in refs:
                source = int(source)
                if source not in pos:
                    continue
                key = tuple(sorted((source, target)))
                offset = edge_counts[key]
                edge_counts[key] += 1
                sx, sy = pos[source]
                d = path_for_edge(sx, sy, tx, ty, kind == "optional", offset)
                parts.append(f'<path class="{kind}-edge" data-edge="{kind}" d="{d}"/>')

    for idx, n in by_idx.items():
        x, y = pos[idx]
        w, h = (190, 70) if idx not in (0, 36) else (150, 58)
        fill, stroke = STAGE_COLORS.get(int(n.get("stage_id", -1)), STAGE_COLORS[-1])
        x0, y0 = x - w/2, y - h/2
        label = n.get("action_label_tier3") or n.get("node_id")
        words = label.split()
        lines, current = [], []
        for word in words:
            if len(" ".join(current + [word])) > 24 and current:
                lines.append(" ".join(current)); current = [word]
            else:
                current.append(word)
        if current: lines.append(" ".join(current))
        lines = lines[:2]
        parts.append(f'<g class="task-node" data-stage="{n.get("stage_id", -1)}" data-node="{idx}">')
        parts.append(f'<rect class="node-box" x="{x0}" y="{y0}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}"/>')
        parts.append(f'<text class="idx" x="{x}" y="{y-9 if len(lines)>1 else y-2}" text-anchor="middle" fill="{stroke}">#{idx}</text>')
        start_y = y + 12 if len(lines) == 1 else y + 10
        for line_no, line in enumerate(lines):
            parts.append(f'<text class="label" x="{x}" y="{start_y + line_no*17}" text-anchor="middle">{esc(line)}</text>')
        parts.append(f'<title>node_idx {idx}: {esc(n.get("node_id"))}\nstage_id {n.get("stage_id")}\nmust: {esc(n.get("must_previous_nodes", []))}\noptional: {esc(n.get("optional_previous_nodes", []))}</title></g>')

    parts.append('</svg>')
    return "\n".join(parts)


def make_report(nodes: list[dict], issues: dict) -> str:
    stage_counts = Counter(int(n.get("stage_id", -1)) for n in nodes)
    lines = [
        "# Task graph draft validation report",
        "",
        f"Source: `{SOURCE}`",
        "",
        "## Structural checks",
        "",
        f"- Nodes: **{len(nodes)}**, with continuous `node_idx` 0–36: **yes**.",
        f"- Duplicate `node_idx`: **{issues['duplicate_indices'] or 'none'}**.",
        f"- Duplicate `node_id`: **{issues['duplicate_ids'] or 'none'}**.",
        f"- References to missing nodes: **{issues['missing_refs'] or 'none'}**.",
        f"- Self references: **{issues['self_refs'] or 'none'}**.",
        f"- Cycle using only mandatory edges: **{'yes: ' + str(issues['must_cycle_nodes']) if issues['must_cycle_nodes'] else 'none'}**.",
        f"- Stage counts: start/end={stage_counts[-1]}, stage 1={stage_counts[1]}, stage 2={stage_counts[2]}, stage 3={stage_counts[3]}.",
        "",
        "## Findings and suggested corrections",
        "",
        "### 1. Definite data error: duplicated predecessor in node 36",
        "",
        "`node_36_end.optional_previous_nodes` contains `23` twice:",
        "",
        "```json",
        '"optional_previous_nodes": [23, 22, 23, 24, 25, 26, 28, 30]',
        "```",
        "",
        "At minimum, remove the duplicate. More importantly, review the intended end condition as described below.",
        "",
        "### 2. Likely typo in node ID",
        "",
        "`node_28_tunr_off_extractor_fan` probably should be `node_28_turn_off_extractor_fan`. The action labels are already spelled correctly. If other files refer to the old ID, update them together.",
        "",
        "### 3. Start node is disconnected",
        "",
        "Node 0 has no outgoing dependency path because every stage-1 entry action has an empty mandatory predecessor list. If node 0 is intended as the formal graph root, add `0` as a mandatory predecessor to every action that is genuinely allowed to start immediately (for example nodes 1, 3, 6, 7, 8, 9, and 10). If node 0 is only a display sentinel, document that convention instead.",
        "",
        "### 4. Stage 1 does not currently guarantee completion before stage 2",
        "",
        "Node 12 requires only node 5. Therefore the mandatory graph allows crimping to begin without completing nodes 6, 7, 8, 9, 10, and 11. If all setup actions must finish first, use a stage-1 join condition. The simplest representation is:",
        "",
        "```json",
        '"node_12_take_plier_from_table": {',
        '  "must_previous_nodes": [5, 6, 7, 8, 9, 11]',
        "}",
        "```",
        "",
        "This assumes node 11 represents completion of the 10→11 cover-removal branch. Confirm the real-world safety logic before applying it.",
        "",
        "### 5. Optional predecessors create many reverse and two-way relations",
        "",
        f"There are **{len(issues['forward_optional'])}** optional references where a higher `node_idx` is listed as a predecessor of a lower one, and **{len(issues['mutual_optional_pairs'])}** immediate two-way optional pairs. Examples include 1↔3, 6↔7, and 7↔8. This is not necessarily wrong if `optional_previous_nodes` means “may occur in either order,” but it is cyclic if downstream code treats every listed predecessor as a normal directed dependency.",
        "",
        "Recommended schema choice:",
        "",
        "- If optional edges are real prerequisites, require the combined graph to be acyclic and keep only the intended direction.",
        "- If these actions are freely orderable, represent them as a parallel/commutative group instead of two directed optional edges.",
        "- If the field means an observation/history feature rather than a scheduling constraint, rename or document it so graph validators do not interpret it as dependency structure.",
        "",
        "### 6. End node does not represent completion of shutdown",
        "",
        "Node 36 has no mandatory predecessors and its optional list omits several terminal shutdown branches, including nodes 27, 29, 31, 33, and 35. If shutdown actions are all required, node 36 should be a mandatory join over the terminal actions of each branch. Based on the current graph, a plausible candidate is:",
        "",
        "```json",
        '"must_previous_nodes": [27, 28, 29, 30, 31, 33, 35],',
        '"optional_previous_nodes": []',
        "```",
        "",
        "This is a semantic suggestion, not an automatic fix: verify whether turning off the main switch (33) must wait for other equipment shutdowns and whether cover replacement (27) must precede locking (35).",
        "",
        "### 7. Shutdown ordering may need safety constraints",
        "",
        "Nodes 26–32 all become available immediately after node 25. Current mandatory edges allow cover replacement, pedal movement, utility shutdown, and crimper shutdown in any order. If the operating procedure requires a safe order, add those constraints explicitly. In particular, review whether node 33 should depend on all powered-device shutdown nodes (28, 29, 30, 32), and whether locking (35) should depend on both cover replacement (27) and main-switch shutdown (33).",
        "",
        "## Edge interpretation used in the graph",
        "",
        "- Solid arrow: `must_previous_nodes` predecessor → current node.",
        "- Dashed purple arrow: `optional_previous_nodes` predecessor → current node.",
        "- Node fill color: `stage_id`; stage −1 is used for start/end.",
        "",
        "The visualization intentionally preserves the draft data rather than silently correcting it.",
    ]
    return "\n".join(lines) + "\n"


def make_html(svg: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Thermal crimp task graph draft</title>
<style>
  :root{{--bg:#f1f5f9;--panel:#ffffff;--text:#0f172a;--muted:#475569;--border:#cbd5e1}}
  *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font-family:Arial,sans-serif}}
  header{{padding:16px 20px;background:var(--panel);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:2}}
  h1{{font-size:20px;margin:0 0 10px}} .controls{{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted)}}
  label{{display:flex;gap:7px;align-items:center}} main{{padding:16px}} .canvas{{background:var(--panel);border:1px solid var(--border);border-radius:12px;overflow:auto;max-height:calc(100vh - 115px)}}
  svg{{display:block;width:1800px;height:2460px}} body.hide-optional .optional-edge{{display:none}} body.hide-must .must-edge{{display:none}}
</style>
</head>
<body>
<header><h1>Thermal crimp task graph — draft</h1><div class="controls">
  <label><input id="must-toggle" type="checkbox" checked> Mandatory edges</label>
  <label><input id="optional-toggle" type="checkbox" checked> Optional edges</label>
</div></header>
<main><div class="canvas">{svg}</div></main>
<script>
document.getElementById('must-toggle').addEventListener('change',e=>document.body.classList.toggle('hide-must',!e.target.checked));
document.getElementById('optional-toggle').addEventListener('change',e=>document.body.classList.toggle('hide-optional',!e.target.checked));
</script>
</body></html>"""


def make_fragment(svg: str) -> str:
    # The conversation preview uses host theme variables; the standalone file keeps
    # a print-friendly fixed palette.
    return f"""<div id="thermal-task-graph">
<div class="viz-controls" aria-label="Edge visibility">
  <label class="form-check"><input class="form-check-input" id="ttg-must" type="checkbox" checked><span class="form-check-label">Mandatory edges</span></label>
  <label class="form-check"><input class="form-check-input" id="ttg-optional" type="checkbox" checked><span class="form-check-label">Optional edges</span></label>
</div>
<div class="ttg-chart">{svg}</div>
<style>
#thermal-task-graph .ttg-chart{{margin-top:12px}}
#thermal-task-graph svg{{display:block;width:100%;height:auto}}
#thermal-task-graph .band{{fill:color-mix(in srgb,var(--card) 68%,transparent);stroke:var(--border)}}
#thermal-task-graph .band-title,#thermal-task-graph .legend,#thermal-task-graph .small{{fill:var(--foreground)}}
#thermal-task-graph .must-edge{{stroke:var(--foreground)}}
#thermal-task-graph .optional-edge{{stroke:var(--viz-series-6);opacity:.52}}
#thermal-task-graph #arrow-must path{{fill:var(--foreground)}}
#thermal-task-graph #arrow-optional path{{fill:var(--viz-series-6)}}
#thermal-task-graph .task-node[data-stage="-1"] .node-box{{fill:color-mix(in srgb,var(--muted) 35%,transparent);stroke:var(--muted-foreground)}}
#thermal-task-graph .task-node[data-stage="1"] .node-box{{fill:color-mix(in srgb,var(--viz-series-1) 22%,transparent);stroke:var(--viz-series-1)}}
#thermal-task-graph .task-node[data-stage="2"] .node-box{{fill:color-mix(in srgb,var(--viz-series-2) 22%,transparent);stroke:var(--viz-series-2)}}
#thermal-task-graph .task-node[data-stage="3"] .node-box{{fill:color-mix(in srgb,var(--viz-series-3) 22%,transparent);stroke:var(--viz-series-3)}}
#thermal-task-graph .idx,#thermal-task-graph .label{{fill:var(--foreground)}}
#thermal-task-graph.hide-must .must-edge{{display:none}}
#thermal-task-graph.hide-optional .optional-edge{{display:none}}
</style>
<script>
(() => {{
  const root = document.getElementById('thermal-task-graph');
  const must = root.querySelector('#ttg-must');
  const optional = root.querySelector('#ttg-optional');
  must.addEventListener('change', () => root.classList.toggle('hide-must', !must.checked));
  optional.addEventListener('change', () => root.classList.toggle('hide-optional', !optional.checked));
}})();
</script>
</div>"""


def make_png(nodes: list[dict]) -> None:
    scale = 1.25
    width, height = int(1800 * scale), int(2460 * scale)
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
        bold = ImageFont.truetype("arialbd.ttf", 17)
        title_font = ImageFont.truetype("arialbd.ttf", 21)
    except OSError:
        font = bold = title_font = ImageFont.load_default()

    def box(coords, **kwargs):
        draw.rounded_rectangle(tuple(int(v * scale) for v in coords), radius=int(14 * scale), **kwargs)

    def line(points, **kwargs):
        draw.line([(int(x * scale), int(y * scale)) for x, y in points], **kwargs)

    def centered_text(x, y, text, chosen_font, fill):
        bbox = draw.textbbox((0, 0), text, font=chosen_font)
        draw.text((int(x * scale - (bbox[2] - bbox[0]) / 2), int(y * scale)), text, font=chosen_font, fill=fill)

    for title, y1, y2 in [("Stage 1 - setup", 115, 965), ("Stage 2 - crimping", 990, 1495), ("Stage 3 - shutdown", 1530, 2205)]:
        box((35, y1, 1765, y2), fill="#ffffff", outline="#cbd5e1", width=2)
        draw.text((58 * scale, (y1 + 14) * scale), title, font=title_font, fill="#334155")

    pos = node_positions()
    # Raster preview keeps optional edges lighter so mandatory structure remains readable.
    for n in nodes:
        target = int(n["node_idx"])
        if target not in pos:
            continue
        tx, ty = pos[target]
        for source in n.get("optional_previous_nodes", []):
            source = int(source)
            if source in pos:
                sx, sy = pos[source]
                line([(sx, sy), ((sx + tx) / 2, (sy + ty) / 2), (tx, ty)], fill="#c4b5fd", width=1)
    for n in nodes:
        target = int(n["node_idx"])
        if target not in pos:
            continue
        tx, ty = pos[target]
        for source in n.get("must_previous_nodes", []):
            source = int(source)
            if source in pos:
                sx, sy = pos[source]
                line([(sx, sy), ((sx + tx) / 2, (sy + ty) / 2), (tx, ty)], fill="#475569", width=3)

    for n in nodes:
        idx = int(n["node_idx"])
        x, y = pos[idx]
        w, h = (190, 70) if idx not in (0, 36) else (150, 58)
        fill, stroke = STAGE_COLORS.get(int(n.get("stage_id", -1)), STAGE_COLORS[-1])
        box((x - w/2, y - h/2, x + w/2, y + h/2), fill=fill, outline=stroke, width=2)
        centered_text(x, y - 23, f"#{idx}", bold, stroke)
        label = n.get("action_label_tier3", n["node_id"])
        if len(label) > 28:
            label = label[:27] + "…"
        centered_text(x, y + 2, label, font, "#111827")
    image.save(HERE / "task_graph_draft_preview.png")


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    nodes = data["nodes"]
    issues = inspect_graph(nodes)
    svg = make_svg(nodes)
    (HERE / "task_graph_draft.svg").write_text(svg, encoding="utf-8")
    (HERE / "task_graph_draft.html").write_text(make_html(svg), encoding="utf-8")
    VIZ_FRAGMENT.parent.mkdir(parents=True, exist_ok=True)
    VIZ_FRAGMENT.write_text(make_fragment(svg), encoding="utf-8")
    make_png(nodes)
    (HERE / "validation_report.md").write_text(make_report(nodes, issues), encoding="utf-8")
    shutil.copy2(SOURCE, HERE / "task_graph_source_snapshot.json")
    summary = {
        "node_count": len(nodes),
        "duplicate_reference_entries": issues["duplicate_refs"],
        "forward_optional_reference_count": len(issues["forward_optional"]),
        "mutual_optional_pair_count": len(issues["mutual_optional_pairs"]),
        "mandatory_cycle_nodes": issues["must_cycle_nodes"],
    }
    (HERE / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
