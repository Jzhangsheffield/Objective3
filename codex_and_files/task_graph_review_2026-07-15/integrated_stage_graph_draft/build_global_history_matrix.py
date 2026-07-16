from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "integrated_task_graph_latest.json"
CSV_OUT = HERE / "integrated_feature_history_matrix.csv"
JSON_OUT = HERE / "integrated_feature_history_matrix.json"
FRAGMENT_OUT = HERE / "integrated-history-matrix.html"


def stage_of(idx: int) -> str:
    if idx in (0, 36):
        return "sentinel"
    if 1 <= idx <= 11:
        return "stage1"
    if 12 <= idx <= 25:
        return "stage2"
    return "stage3"


def build_codes(nodes: list[dict]) -> list[list[str]]:
    by_idx = {int(node["node_idx"]): node for node in nodes}
    matrix = []
    for current in range(37):
        node = by_idx[current]
        execution = node["execution_constraints"]
        history = node["feature_history_constraints"]
        immediate = execution["must_immediately_previous_node"]
        mandatory = set(history["all_must_previous_nodes"])
        optional = set(history["optional_previous_nodes"])
        row = []
        for previous in range(37):
            if previous == current:
                code = "S"
            elif previous == immediate:
                code = "I"
            elif previous in mandatory:
                code = "M"
            elif previous in optional:
                code = "O"
            else:
                code = "."
            row.append(code)
        matrix.append(row)
    return matrix


def write_csv(matrix: list[list[str]]) -> None:
    with CSV_OUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["current_node\\previous_node", *range(37)])
        for idx, row in enumerate(matrix):
            writer.writerow([idx, *row])


def write_json(nodes: list[dict], matrix: list[list[str]]) -> None:
    labels = {int(node["node_idx"]): node["action_label_tier3"] for node in nodes}
    payload = {
        "matrix_semantics": {
            "row": "current node_idx",
            "column": "candidate previous node_idx",
            "M": "mandatory direct or transitive history",
            "I": "must immediately precede the current node",
            "O": "possible optional history in at least one legal execution",
            "S": "same node / diagonal",
            ".": "cannot occur before the current node",
        },
        "node_labels": labels,
        "column_node_idx": list(range(37)),
        "rows": [
            {"current_node_idx": idx, "values": row}
            for idx, row in enumerate(matrix)
        ],
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def svg_fragment(nodes: list[dict], matrix: list[list[str]]) -> str:
    cell = 22
    left = 66
    top = 66
    grid = cell * 37
    width = left + grid + 12
    height = top + grid + 12
    parts = [
        '<div id="integrated-history-matrix">',
        '  <div class="ihm-legend text-small" aria-label="Matrix legend">',
        '    <span><b class="mandatory">M</b> mandatory</span>',
        '    <span><b class="immediate">I</b> immediate</span>',
        '    <span><b class="optional">O</b> possible optional</span>',
        '    <span><b class="self">—</b> same node</span>',
        '    <span><b class="impossible"></b> impossible</span>',
        '  </div>',
        f'  <svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="ihm-title ihm-desc">',
        '    <title id="ihm-title">Integrated node history matrix for nodes 0 through 36</title>',
        '    <desc id="ihm-desc">Rows are current nodes and columns are candidate previous nodes. Cell letters identify mandatory, immediate, optional, self, or impossible relationships.</desc>',
    ]

    for idx in range(37):
        x = left + idx * cell
        y = top + idx * cell
        stage = stage_of(idx)
        parts.append(f'    <rect class="axis-label {stage}" x="{x}" y="{top-cell}" width="{cell}" height="{cell}"/>')
        parts.append(f'    <text class="axis-text" x="{x + cell/2}" y="{top-7}" text-anchor="middle">{idx}</text>')
        parts.append(f'    <rect class="axis-label {stage}" x="{left-cell}" y="{y}" width="{cell}" height="{cell}"/>')
        parts.append(f'    <text class="axis-text" x="{left-7}" y="{y + 15}" text-anchor="end">{idx}</text>')

    class_for = {"M": "mandatory", "I": "immediate", "O": "optional", "S": "self", ".": "impossible"}
    visible_for = {"M": "M", "I": "I", "O": "O", "S": "—", ".": ""}
    for current, row in enumerate(matrix):
        for previous, code in enumerate(row):
            x = left + previous * cell
            y = top + current * cell
            parts.append(f'    <rect class="matrix-cell {class_for[code]}" x="{x}" y="{y}" width="{cell}" height="{cell}"/>')
            visible = visible_for[code]
            if visible:
                parts.append(f'      <text class="cell-text {class_for[code]}-text" x="{x + cell/2}" y="{y + 15}" text-anchor="middle">{visible}</text>')

    for boundary in (1, 12, 26, 36):
        position = left + boundary * cell
        parts.append(f'    <line class="stage-boundary" x1="{position}" y1="{top}" x2="{position}" y2="{top+grid}"/>')
        position_y = top + boundary * cell
        parts.append(f'    <line class="stage-boundary" x1="{left}" y1="{position_y}" x2="{left+grid}" y2="{position_y}"/>')
    parts.extend([
        '  </svg>',
        '  <div class="ihm-mobile text-small" aria-label="Stage-block summary">',
        '    <div></div><b>previous S1</b><b>previous S2</b><b>previous S3</b>',
        '    <b>current S1</b><span>mixed M/I/O</span><span>impossible</span><span>impossible</span>',
        '    <b>current S2</b><span>all M</span><span>linear M/I</span><span>impossible</span>',
        '    <b>current S3</b><span>all M</span><span>all M</span><span>mixed M/I/O</span>',
        '  </div>',
        '  <style>',
        '    #integrated-history-matrix { color:var(--foreground); }',
        '    #integrated-history-matrix .ihm-legend { display:flex; justify-content:center; flex-wrap:wrap; gap:14px; margin-bottom:8px; }',
        '    #integrated-history-matrix .ihm-legend b { display:inline-block; width:22px; height:22px; text-align:center; border-radius:3px; }',
        '    #integrated-history-matrix svg { display:block; width:100%; height:auto; }',
        '    #integrated-history-matrix .matrix-cell { stroke:var(--border); stroke-width:.5; }',
        '    #integrated-history-matrix .mandatory { fill:var(--primary); background:var(--primary); color:var(--primary-foreground); }',
        '    #integrated-history-matrix .immediate { fill:var(--destructive); background:var(--destructive); color:var(--background); }',
        '    #integrated-history-matrix .optional { fill:var(--accent); background:var(--accent); color:var(--accent-foreground); }',
        '    #integrated-history-matrix .self { fill:var(--secondary); background:var(--secondary); color:var(--secondary-foreground); }',
        '    #integrated-history-matrix .impossible { fill:var(--muted); background:var(--muted); border:1px solid var(--border); }',
        '    #integrated-history-matrix .cell-text { font-weight:500; pointer-events:none; }',
        '    #integrated-history-matrix .mandatory-text { fill:var(--primary-foreground); }',
        '    #integrated-history-matrix .immediate-text { fill:var(--background); }',
        '    #integrated-history-matrix .optional-text { fill:var(--accent-foreground); }',
        '    #integrated-history-matrix .self-text { fill:var(--secondary-foreground); }',
        '    #integrated-history-matrix .axis-label.sentinel { fill:var(--muted); }',
        '    #integrated-history-matrix .axis-label.stage1 { fill:color-mix(in srgb,var(--viz-series-1) 30%,transparent); }',
        '    #integrated-history-matrix .axis-label.stage2 { fill:color-mix(in srgb,var(--viz-series-2) 30%,transparent); }',
        '    #integrated-history-matrix .axis-label.stage3 { fill:color-mix(in srgb,var(--viz-series-3) 30%,transparent); }',
        '    #integrated-history-matrix .axis-text { fill:var(--foreground); font-weight:500; }',
        '    #integrated-history-matrix .stage-boundary { stroke:var(--foreground); stroke-width:1.5; }',
        '    #integrated-history-matrix .ihm-mobile { display:none; grid-template-columns:repeat(4,minmax(0,1fr)); gap:3px; text-align:center; }',
        '    #integrated-history-matrix .ihm-mobile > * { padding:6px 2px; background:var(--muted); color:var(--muted-foreground); }',
        '    #integrated-history-matrix .ihm-mobile b { color:var(--foreground); font-weight:500; }',
        '    @media (max-width:520px) { #integrated-history-matrix svg { display:none; } #integrated-history-matrix .ihm-mobile { display:grid; } }',
        '  </style>',
        '</div>',
    ])
    return "\n".join(parts) + "\n"


def main() -> None:
    document = json.loads(SOURCE.read_text(encoding="utf-8"))
    nodes = document["nodes"]
    matrix = build_codes(nodes)
    write_csv(matrix)
    write_json(nodes, matrix)
    FRAGMENT_OUT.write_text(svg_fragment(nodes, matrix), encoding="utf-8")

    assert len(matrix) == 37 and all(len(row) == 37 for row in matrix)
    for idx, row in enumerate(matrix):
        assert row[idx] == "S"
    assert matrix[2][1] == "I"
    assert matrix[13][12] == "I"
    assert matrix[27][26] == "I"
    assert matrix[35][34] == "I"
    assert matrix[36][0:36] == ["M"] * 36


if __name__ == "__main__":
    main()
