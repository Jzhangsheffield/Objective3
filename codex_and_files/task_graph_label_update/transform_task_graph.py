"""Normalize task_graph.json fields using the canonical hierarchical label map."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_GRAPH_PATH = PROJECT_ROOT / "task_graph" / "task_graph.json"
LABEL_MAP_PATH = Path(
    r"C:\Junxi_data_for_training_speedup\Stage_2_Mapstyle_Dataset\label_map.json"
)
BACKUP_PATH = Path(__file__).with_name("task_graph_original_backup.json")
REPORT_PATH = Path(__file__).with_name("validation_report.json")

# These node IDs intentionally remain unchanged. Only their canonical action fields
# are corrected according to label_map.json.
TIER3_CORRECTIONS = {
    14: "place sample under electrodes",
    21: "place sample under electrodes",
    28: "turn off extractor fan",
    31: "move pedal to original location",
}

CONTROL_LABELS = {0: "start", 36: "end"}


def longest_prefix(label: str, candidates: dict[str, int]) -> str:
    matches = [candidate for candidate in candidates if label == candidate or label.startswith(candidate + " ")]
    if not matches:
        raise ValueError(f"No hierarchical parent found for tier3 label: {label!r}")
    return max(matches, key=len)


def extract_source_nodes(raw_text: str) -> list[dict[str, object]]:
    """Extract the fields that must be retained even when the source JSON is invalid."""
    object_pattern = re.compile(r"\{\s*\"node_id\".*?\}\s*,?", re.DOTALL)
    nodes: list[dict[str, object]] = []
    for block in object_pattern.findall(raw_text):
        node_id_match = re.search(r'\"node_id\"\s*:\s*\"([^\"]+)\"', block)
        stage_id_match = re.search(r'\"stage_id\"\s*:\s*(-?\d+)', block)
        if not node_id_match or not stage_id_match:
            raise ValueError(f"Could not extract required fields from node block:\n{block}")
        nodes.append(
            {
                "node_id": node_id_match.group(1),
                "stage_id": int(stage_id_match.group(1)),
            }
        )
    if not nodes:
        raise ValueError("No nodes found in task graph")
    return nodes


def parse_node_id(node_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"node_(\d+)_(.+)", node_id)
    if not match:
        raise ValueError(f"Unexpected node_id format: {node_id!r}")
    node_idx = int(match.group(1))
    action_part = re.sub(r"_\d+$", "", match.group(2))
    return node_idx, action_part.replace("_", " ")


def main() -> None:
    raw_text = TASK_GRAPH_PATH.read_text(encoding="utf-8")
    label_map = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    source_nodes = extract_source_nodes(raw_text)

    if not BACKUP_PATH.exists():
        shutil.copyfile(TASK_GRAPH_PATH, BACKUP_PATH)

    transformed_nodes: list[dict[str, object]] = []
    report_nodes: list[dict[str, object]] = []

    for source_node in source_nodes:
        node_id = str(source_node["node_id"])
        node_idx, parsed_tier3 = parse_node_id(node_id)

        if node_idx in CONTROL_LABELS:
            control_label = CONTROL_LABELS[node_idx]
            transformed = {
                "node_id": node_id,
                "node_idx": node_idx,
                "action_label_tier1": control_label,
                "action_label_tier2": control_label,
                "action_label_tier3": control_label,
                "action_id_tier1": -1,
                "action_id_tier2": -1,
                "action_id_tier3": -1,
                "stage_id": source_node["stage_id"],
                "must_previous_nodes": [],
                "optional_previous_nodes": [],
            }
            status = "control_node"
            canonical_tier3 = control_label
        else:
            canonical_tier3 = TIER3_CORRECTIONS.get(node_idx, parsed_tier3)
            if canonical_tier3 not in label_map["tier3"]:
                raise ValueError(
                    f"Node {node_id!r} does not resolve to a canonical tier3 label: "
                    f"{canonical_tier3!r}"
                )
            tier2 = longest_prefix(canonical_tier3, label_map["tier2"])
            tier1 = longest_prefix(tier2, label_map["tier1"])
            transformed = {
                "node_id": node_id,
                "node_idx": node_idx,
                "action_label_tier1": tier1,
                "action_label_tier2": tier2,
                "action_label_tier3": canonical_tier3,
                "action_id_tier1": label_map["tier1"][tier1],
                "action_id_tier2": label_map["tier2"][tier2],
                "action_id_tier3": label_map["tier3"][canonical_tier3],
                "stage_id": source_node["stage_id"],
                "must_previous_nodes": [],
                "optional_previous_nodes": [],
            }
            status = "corrected_to_canonical" if node_idx in TIER3_CORRECTIONS else "matched"

        transformed_nodes.append(transformed)
        report_nodes.append(
            {
                "node_id": node_id,
                "parsed_tier3_from_node_id": parsed_tier3,
                "canonical_tier3_used": canonical_tier3,
                "status": status,
            }
        )

    node_indices = [node["node_idx"] for node in transformed_nodes]
    if len(node_indices) != len(set(node_indices)):
        raise ValueError("Duplicate node_idx values found")

    TASK_GRAPH_PATH.write_text(
        json.dumps({"nodes": transformed_nodes}, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        json.dumps(
            {
                "source_task_graph": str(TASK_GRAPH_PATH),
                "label_map": str(LABEL_MAP_PATH),
                "node_count": len(transformed_nodes),
                "matched_action_nodes": sum(item["status"] == "matched" for item in report_nodes),
                "corrected_action_nodes": sum(
                    item["status"] == "corrected_to_canonical" for item in report_nodes
                ),
                "control_nodes": sum(item["status"] == "control_node" for item in report_nodes),
                "nodes": report_nodes,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
