from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_MANIFEST = Path(r"C:\Junxi_data_for_training_speedup\Stage_2_Mapstyle_Dataset\3_camera_mindrove_manifest.jsonl")
DEFAULT_GRAPH = Path(r"D:\Junxi_data\Objective3_thermal_crimp\codex_and_files\task_graph_review_2026-07-15\integrated_stage_graph_draft\integrated_task_graph_latest.json")
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "3_camera_mindrove_manifest_with_graph_nodes.jsonl"
DEFAULT_REPORT = HERE / "enrichment_report.json"


def align_stage2_segment(labels: list[str], template: list[str]) -> tuple[list[set[int]], int]:
    """Return possible template positions on globally minimum-gap alignments."""
    if not labels:
        return [], 0
    size = len(template)
    positions_by_label: dict[str, list[int]] = defaultdict(list)
    for position, label in enumerate(template):
        positions_by_label[label].append(position)

    first_states = positions_by_label[labels[0]]
    if not first_states:
        raise ValueError(f"Stage 2 label is absent from template: {labels[0]!r}")

    # forward[i][absolute_template_position] = minimum skipped actions so far.
    forward: list[dict[int, int]] = [{position: 0 for position in first_states}]
    max_cycle = len(labels) + 2
    for label in labels[1:]:
        candidates = [position + cycle * size for cycle in range(max_cycle) for position in positions_by_label[label]]
        current: dict[int, int] = {}
        for candidate in candidates:
            best = None
            for previous, cost in forward[-1].items():
                if candidate <= previous:
                    continue
                value = cost + candidate - previous - 1
                if best is None or value < best:
                    best = value
            if best is not None:
                current[candidate] = best
        if not current:
            raise ValueError(f"No monotonic Stage 2 alignment for labels: {labels}")
        minimum = min(current.values())
        # Higher-cost states can never become optimal because future costs are non-negative.
        forward.append({position: cost for position, cost in current.items() if cost <= minimum + size})

    # backward[i][state] = minimum additional skipped actions to reach the end.
    backward: list[dict[int, int]] = [{} for _ in labels]
    backward[-1] = {state: 0 for state in forward[-1]}
    for i in range(len(labels) - 2, -1, -1):
        current: dict[int, int] = {}
        for state in forward[i]:
            values = [
                next_state - state - 1 + remaining
                for next_state, remaining in backward[i + 1].items()
                if next_state > state
            ]
            if values:
                current[state] = min(values)
        backward[i] = current

    optimum = min(forward[-1].values())
    possible_positions = []
    for i in range(len(labels)):
        positions = {
            state % size
            for state, prefix_cost in forward[i].items()
            if state in backward[i] and prefix_cost + backward[i][state] == optimum
        }
        possible_positions.append(positions)
    return possible_positions, optimum


def insert_graph_fields(sample: dict, node: dict) -> dict:
    """Place graph fields immediately after tier3 for easier line inspection."""
    enriched = {}
    inserted = False
    for key, value in sample.items():
        enriched[key] = value
        if key == "tier3":
            enriched["node_id"] = node["node_id"]
            enriched["node_idx"] = node["node_idx"]
            enriched["stage_id"] = node["stage_id"]
            inserted = True
    if not inserted:
        enriched["node_id"] = node["node_id"]
        enriched["node_idx"] = node["node_idx"]
        enriched["stage_id"] = node["stage_id"]
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--diagnose-only", action="store_true")
    args = parser.parse_args()

    graph_document = json.loads(args.graph.read_text(encoding="utf-8"))
    graph_nodes = graph_document["nodes"]
    nodes_by_idx = {int(node["node_idx"]): node for node in graph_nodes}
    nodes_by_label: dict[str, list[dict]] = defaultdict(list)
    for node in graph_nodes:
        nodes_by_label[node["action_label_tier3"]].append(node)

    records = []
    with args.manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            sample = json.loads(line)
            records.append({"line_number": line_number, "sample": sample})

    unmatched = [
        {"line_number": record["line_number"], "sample_name": record["sample"].get("sample_name"), "tier3": record["sample"].get("tier3")}
        for record in records
        if record["sample"].get("tier3") not in nodes_by_label
    ]

    assignment: dict[int, int] = {}
    for record in records:
        label = record["sample"].get("tier3")
        matches = nodes_by_label.get(label, [])
        if len(matches) == 1:
            assignment[record["line_number"]] = int(matches[0]["node_idx"])

    stage2_nodes = [nodes_by_idx[idx] for idx in range(12, 26)]
    stage2_template = [node["action_label_tier3"] for node in stage2_nodes]
    stage2_labels = set(stage2_template)
    stage2_allowed_successors = {
        12: {13}, 13: {14}, 14: {15}, 15: {16}, 16: {17},
        17: {18}, 18: {19}, 19: {20}, 20: {21}, 21: {22},
        # Some manifest runs contain an additional crimp/reverse cycle.
        # It reuses nodes 16–22 before continuing to inspection.
        22: {16, 23}, 23: {24}, 24: {25}, 25: set(),
    }
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        sample = record["sample"]
        key = (sample.get("participant"), sample.get("run"))
        grouped[key].append(record)

    alignment_segments = []
    unresolved = []
    observed_stage2_repeat_cycles = 0
    for group_key, group_records in grouped.items():
        ordered = sorted(group_records, key=lambda record: (record["sample"].get("annotation_row_index", 10**12), record["line_number"]))
        segments: list[list[dict]] = []
        current: list[dict] = []
        for record in ordered:
            if record["sample"].get("tier3") in stage2_labels:
                current.append(record)
            elif current:
                segments.append(current)
                current = []
        if current:
            segments.append(current)

        for segment_no, segment in enumerate(segments, start=1):
            labels = [record["sample"]["tier3"] for record in segment]
            possible_positions, gap_cost = align_stage2_segment(labels, stage2_template)
            candidate_nodes = [{12 + position for position in positions} for positions in possible_positions]
            # Resolve equal-cost duplicate-label alignments using exact neighboring
            # transitions. This preserves the normal linear chain and also handles
            # the observed 22 -> 16 repeat cycle without guessing by occurrence count.
            changed = True
            while changed:
                changed = False
                for i in range(1, len(candidate_nodes)):
                    if len(candidate_nodes[i - 1]) != 1 or len(candidate_nodes[i]) <= 1:
                        continue
                    previous = next(iter(candidate_nodes[i - 1]))
                    narrowed = candidate_nodes[i] & stage2_allowed_successors.get(previous, set())
                    if narrowed and narrowed != candidate_nodes[i]:
                        candidate_nodes[i] = narrowed
                        changed = True
                for i in range(len(candidate_nodes) - 2, -1, -1):
                    if len(candidate_nodes[i + 1]) != 1 or len(candidate_nodes[i]) <= 1:
                        continue
                    following = next(iter(candidate_nodes[i + 1]))
                    narrowed = {
                        candidate
                        for candidate in candidate_nodes[i]
                        if following in stage2_allowed_successors.get(candidate, set())
                    }
                    if narrowed and narrowed != candidate_nodes[i]:
                        candidate_nodes[i] = narrowed
                        changed = True
            segment_info = {
                "participant": group_key[0],
                "run": group_key[1],
                "segment_no": segment_no,
                "sample_count": len(segment),
                "alignment_gap_cost": gap_cost,
            }
            alignment_segments.append(segment_info)
            observed_stage2_repeat_cycles += sum(
                1
                for previous, following in zip(candidate_nodes, candidate_nodes[1:])
                if previous == {22} and following == {16}
            )
            for record, candidates in zip(segment, candidate_nodes):
                line_number = record["line_number"]
                if len(candidates) == 1:
                    node_idx = next(iter(candidates))
                    existing = assignment.get(line_number)
                    if existing is not None and existing != node_idx:
                        raise AssertionError(f"Unique-label assignment disagrees with Stage 2 alignment at line {line_number}: {existing} vs {node_idx}")
                    assignment[line_number] = node_idx
                else:
                    unresolved.append({
                        "line_number": line_number,
                        "sample_name": record["sample"].get("sample_name"),
                        "participant": group_key[0],
                        "run": group_key[1],
                        "annotation_row_index": record["sample"].get("annotation_row_index"),
                        "tier3": record["sample"].get("tier3"),
                        "candidate_node_idx": sorted(candidates),
                    })

    report = {
        "source_manifest": str(args.manifest),
        "source_graph": str(args.graph),
        "source_line_count": len(records),
        "unique_tier3_count": len({record["sample"].get("tier3") for record in records}),
        "unmatched_count": len(unmatched),
        "unmatched": unmatched,
        "stage2_alignment_segment_count": len(alignment_segments),
        "stage2_alignment_gap_cost_distribution": dict(sorted(Counter(item["alignment_gap_cost"] for item in alignment_segments).items())),
        "observed_stage2_repeat_transition_22_to_16_count": observed_stage2_repeat_cycles,
        "deviation_interpretation": "Observed skipped or repeated actions belong to intentionally erroneous runs. They are retained as observations and must not be added to the standard normal-run task graph.",
        "standard_task_graph_modified": False,
        "unresolved_ambiguous_count": len(unresolved),
        "unresolved_ambiguous": unresolved,
        "assigned_count": len(assignment),
        "output_manifest": None if args.diagnose_only else str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if unmatched or unresolved or len(assignment) != len(records):
        raise SystemExit(
            f"Enrichment stopped safely: unmatched={len(unmatched)}, unresolved={len(unresolved)}, assigned={len(assignment)}/{len(records)}. See {args.report}"
        )
    if args.diagnose_only:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            node = nodes_by_idx[assignment[record["line_number"]]]
            enriched = insert_graph_fields(record["sample"], node)
            handle.write(json.dumps(enriched, ensure_ascii=False, separators=(",", ":")) + "\n")

    # Full read-back validation.
    output_records = []
    with args.output.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            sample = json.loads(line)
            output_records.append(sample)
            expected_node = nodes_by_idx[assignment[line_number]]
            assert sample["node_id"] == expected_node["node_id"]
            assert sample["node_idx"] == expected_node["node_idx"]
            assert sample["stage_id"] == expected_node["stage_id"]
    assert len(output_records) == len(records)
    for original_record, enriched_sample in zip(records, output_records):
        original_fields = {
            key: value
            for key, value in enriched_sample.items()
            if key not in {"node_id", "node_idx", "stage_id"}
        }
        assert original_fields == original_record["sample"]

    report["output_line_count"] = len(output_records)
    report["validated"] = True
    report["original_fields_preserved"] = True
    report["node_idx_distribution"] = dict(sorted(Counter(sample["node_idx"] for sample in output_records).items()))
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
