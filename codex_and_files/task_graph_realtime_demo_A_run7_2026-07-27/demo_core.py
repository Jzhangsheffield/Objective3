from __future__ import annotations

import csv
import json
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DEMO_ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else DEMO_ROOT / path


def timestamp_seconds(value: str) -> float:
    return datetime.strptime(str(value), "%Y%m%d_%H%M%S_%f").timestamp()


@dataclass(frozen=True)
class DemoData:
    config: dict[str, Any]
    frames: list[dict[str, Any]]
    segments: list[dict[str, Any]]
    segments_by_no: dict[int, dict[str, Any]]
    frames_by_segment: dict[int, list[dict[str, Any]]]
    raw_frames_dir: Path
    first_timestamp_seconds: float
    duration_seconds: float


def load_demo_data(config_path: Path | None = None) -> DemoData:
    config_path = (config_path or DEMO_ROOT / "config.json").resolve()
    config = read_json(config_path)
    paths = {key: resolve(value) for key, value in config["paths"].items()}
    if not paths["derived_segments"].is_file():
        raise FileNotFoundError(
            f"Derived metadata is missing. Run prepare_demo_metadata.py first: "
            f"{paths['derived_segments']}"
        )

    with paths["source_frame_annotation"].open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        frames = list(csv.DictReader(handle))
    for row in frames:
        row["frame_idx"] = int(row["frame_idx"])
        row["original_frame_idx"] = int(row["original_frame_idx"])
        row["segment_no"] = int(row["segment_no"])
        row["timestamp_seconds"] = timestamp_seconds(row["timestamp"])

    segments = read_jsonl(paths["derived_segments"])
    segments_by_no = {int(row["segment_no"]): row for row in segments}
    frames_by_segment: dict[int, list[dict[str, Any]]] = {}
    for row in frames:
        frames_by_segment.setdefault(row["segment_no"], []).append(row)

    first_timestamp = frames[0]["timestamp_seconds"]
    duration = frames[-1]["timestamp_seconds"] - first_timestamp
    return DemoData(
        config=config,
        frames=frames,
        segments=segments,
        segments_by_no=segments_by_no,
        frames_by_segment=frames_by_segment,
        raw_frames_dir=paths["raw_frames_dir"],
        first_timestamp_seconds=first_timestamp,
        duration_seconds=duration,
    )


class InferenceEngine:
    def __init__(self, data: DemoData, device: str = "auto") -> None:
        self.data = data
        self.config = data.config
        self.paths = {
            key: resolve(value) for key, value in self.config["paths"].items()
        }
        self.history_policy = str(
            self.config["demo"].get(
                "history_policy", "completed_action_ground_truth_node_idx"
            )
        )
        self.history_order = str(
            self.config["demo"].get("history_order", "graph_valid")
        )
        supported_history_policies = {
            "completed_action_ground_truth_node_idx",
            "completed_action_m3_predicted_node_idx",
        }
        if self.history_policy not in supported_history_policies:
            raise ValueError(f"Unsupported history_policy: {self.history_policy}")
        if self.history_order not in {"graph_valid", "actual"}:
            raise ValueError(f"Unsupported history_order: {self.history_order}")
        package_root = self.paths["experiment_package"]
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))

        from graph_history.backbone import generate_model
        from graph_history.constants import NUM_GRAPH_NODES, NUM_TIER3_CLASSES
        from graph_history.data import RGBVideoTransform, uniform_frame_indices
        from graph_history.graph import (
            TaskGraphSpec,
            randomized_graph_valid_history,
            stable_sample_seed,
        )
        from graph_history.models import FeatureNodeClassifier, build_context_model
        from graph_history.utils import load_compatible_state

        self.NUM_GRAPH_NODES = NUM_GRAPH_NODES
        self.NUM_TIER3_CLASSES = NUM_TIER3_CLASSES
        self.uniform_frame_indices = uniform_frame_indices
        self.randomized_graph_valid_history = randomized_graph_valid_history
        self.stable_sample_seed = stable_sample_seed
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if device == "auto" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.amp = self.device.type == "cuda"

        self.graph = TaskGraphSpec.load(
            self.paths["task_graph"], self.paths["relation_matrix"]
        )
        self.transform = RGBVideoTransform(train=False, size=224)
        self.node_metadata = {
            int(node["node_idx"]): node
            for node in self.graph.graph_json["nodes"]
            if 1 <= int(node["node_idx"]) <= NUM_GRAPH_NODES
        }

        self.backbone = generate_model(
            18, num_classes=NUM_TIER3_CLASSES
        ).to(self.device)
        backbone_report = load_compatible_state(
            self.backbone, self.paths["rgb_backbone_checkpoint"]
        )
        self.backbone.eval()

        self.m0 = FeatureNodeClassifier(feature_dim=512).to(self.device)
        m0_report = load_compatible_state(
            self.m0, self.paths["m0_checkpoint"]
        )
        self.m0.eval()

        self.e2e = generate_model(
            18, num_classes=NUM_GRAPH_NODES
        ).to(self.device)
        e2e_report = load_compatible_state(
            self.e2e, self.paths["e2e_node_scratch_checkpoint"]
        )
        self.e2e.eval()

        baseline = FeatureNodeClassifier(feature_dim=512)
        self.m3 = build_context_model(
            model_name="m3",
            baseline=baseline,
            relation_ids=self.graph.relation_ids,
            feature_dim=512,
            d_model=256,
            num_heads=4,
            max_history=35,
            dropout=0.1,
        ).to(self.device)
        m3_report = load_compatible_state(self.m3, self.paths["m3_checkpoint"])
        self.m3.eval()

        for name, report in (
            ("backbone", backbone_report),
            ("m0", m0_report),
            ("e2e", e2e_report),
            ("m3", m3_report),
        ):
            if report["missing_keys"] or report["unexpected_keys"]:
                raise RuntimeError(f"Incomplete {name} checkpoint load: {report}")

        self.load_reports = {
            "backbone": backbone_report,
            "m0": m0_report,
            "e2e": e2e_report,
            "m3": m3_report,
        }
        self._history: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def device_name(self) -> str:
        if self.device.type == "cuda":
            return torch.cuda.get_device_name(self.device)
        return str(self.device)

    def reset(self) -> None:
        with self._lock:
            self._history.clear()

    def _video_tensor(self, segment_no: int) -> tuple[torch.Tensor, list[int]]:
        frame_rows = self.data.frames_by_segment[int(segment_no)]
        indices = self.uniform_frame_indices(len(frame_rows), 16)
        selected: list[torch.Tensor] = []
        for index in indices:
            frame_path = self.data.raw_frames_dir / frame_rows[index]["frame_name"]
            with Image.open(frame_path) as image:
                image = image.convert("RGB").resize(
                    (256, 256), Image.Resampling.BILINEAR
                )
                array = np.asarray(image).copy()
            selected.append(torch.from_numpy(array).permute(2, 0, 1))
        video_tchw = torch.stack(selected, dim=0)
        video_cthw = self.transform(video_tchw).permute(1, 0, 2, 3).contiguous()
        return video_cthw, indices

    def _node_result(
        self, probabilities: torch.Tensor
    ) -> tuple[int, str, float, list[dict[str, Any]]]:
        top_values, top_indices = probabilities.topk(3)
        top3 = []
        for value, index in zip(top_values.tolist(), top_indices.tolist()):
            node_idx = int(index) + 1
            node_info = self._node_info(node_idx)
            top3.append(
                {
                    "node_idx": node_idx,
                    "node_id": node_info["node_id"],
                    "label": node_info["tier3_label"],
                    "occurrence": node_info["occurrence"],
                    "display_name": node_info["display_name"],
                    "confidence": float(value),
                }
            )
        best = top3[0]
        return best["node_idx"], best["label"], best["confidence"], top3

    def _node_info(self, node_idx: int) -> dict[str, Any]:
        metadata = self.node_metadata[int(node_idx)]
        node_id = str(metadata["node_id"])
        tier3_label = str(metadata["action_label_tier3"])
        semantic_id = re.sub(rf"^node_{int(node_idx)}_", "", node_id)
        occurrence_match = re.search(r"_(\d+)$", semantic_id)
        occurrence = (
            int(occurrence_match.group(1)) if occurrence_match is not None else None
        )
        readable_label = tier3_label
        if occurrence is not None:
            readable_label = f"{tier3_label} · occurrence {occurrence}"
        return {
            "node_idx": int(node_idx),
            "node_id": node_id,
            "tier3_label": tier3_label,
            "stage_id": int(metadata["stage_id"]),
            "occurrence": occurrence,
            "display_name": f"Node {int(node_idx)} · {readable_label}",
        }

    @torch.inference_mode()
    def predict(self, segment: dict[str, Any]) -> dict[str, Any]:
        if segment["is_background"]:
            raise ValueError("Background segments are not sent to the action classifiers")
        started = time.perf_counter()
        video, selected_indices = self._video_tensor(int(segment["segment_no"]))
        video = video.unsqueeze(0).to(self.device, non_blocking=True)

        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp,
        ):
            current_feature = self.backbone.forward_features(video)
            e2e_logits = self.e2e(video)
        current_feature = current_feature.float()
        m0_probabilities = F.softmax(self.m0(current_feature).float(), dim=-1)[0]
        e2e_probabilities = F.softmax(e2e_logits.float(), dim=-1)[0]

        with self._lock:
            stored_history_rows = list(self._history)
            if self.history_order == "graph_valid":
                history_rows = self.randomized_graph_valid_history(
                    stored_history_rows,
                    graph=self.graph,
                    seed=self.stable_sample_seed(
                        int(self.config["demo"]["seed"]),
                        str(segment["original_action_sample_name"]),
                    ),
                )
            else:
                history_rows = stored_history_rows
            if history_rows:
                history_features = torch.stack(
                    [row["feature"] for row in history_rows], dim=0
                ).unsqueeze(0)
                history_nodes = torch.tensor(
                    [[int(row["node_idx"]) - 1 for row in history_rows]],
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                history_features = current_feature.new_zeros((1, 0, 512))
                history_nodes = torch.empty(
                    (1, 0), dtype=torch.long, device=self.device
                )
            history_length = history_features.shape[1]
            history_positions = torch.arange(
                history_length, 0, -1, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            history_mask = torch.zeros(
                (1, history_length), dtype=torch.bool, device=self.device
            )
            m3_logits, _ = self.m3(
                current_feature=current_feature,
                history_features=history_features,
                history_position_ids=history_positions,
                history_node_classes=history_nodes,
                history_padding_mask=history_mask,
            )
            m3_probabilities = F.softmax(m3_logits.float(), dim=-1)[0]

            m0_node, m0_label, m0_confidence, m0_top3 = self._node_result(
                m0_probabilities
            )
            m3_node, m3_label, m3_confidence, m3_top3 = self._node_result(
                m3_probabilities
            )
            e2e_node, e2e_label, e2e_confidence, e2e_top3 = self._node_result(
                e2e_probabilities
            )

            true_node = int(segment["node_idx"])
            if self.history_policy == "completed_action_ground_truth_node_idx":
                history_node_for_future = true_node
                history_node_source = "ground_truth"
            else:
                history_node_for_future = m3_node
                history_node_source = "m3_prediction"

            # The current action enters history only after all current predictions.
            self._history.append(
                {
                    "node_idx": int(history_node_for_future),
                    "true_node_idx": true_node,
                    "history_node_source": history_node_source,
                    "feature": current_feature[0].detach(),
                    "sample_name": segment["original_action_sample_name"],
                    "annotation_row_index": int(segment["annotation_row_index"]),
                }
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        true_node_info = self._node_info(true_node)
        m0_node_info = self._node_info(m0_node)
        m3_node_info = self._node_info(m3_node)
        e2e_node_info = self._node_info(e2e_node)
        return {
            "segment_no": int(segment["segment_no"]),
            "annotation_row_index": int(segment["annotation_row_index"]),
            "sample_name": segment["original_action_sample_name"],
            "true_node_idx": true_node,
            "true_node_id": true_node_info["node_id"],
            "true_label": segment["tier3_label"],
            "true_occurrence": true_node_info["occurrence"],
            "true_display_name": true_node_info["display_name"],
            "stage_id": int(segment["stage_id"]),
            "history_length_before_current": len(history_rows),
            "history_policy": self.history_policy,
            "history_order": self.history_order,
            "history_node_order": [int(row["node_idx"]) for row in history_rows],
            "history_entries_before_current": [
                {
                    "annotation_row_index": int(row["annotation_row_index"]),
                    "sample_name": str(row["sample_name"]),
                    "history_node_idx": int(row["node_idx"]),
                    "true_node_idx": int(row.get("true_node_idx", row["node_idx"])),
                    "history_node_source": str(
                        row.get("history_node_source", "ground_truth")
                    ),
                }
                for row in history_rows
            ],
            "graph_valid_history_node_order": [
                int(row["node_idx"]) for row in history_rows
            ]
            if self.history_order == "graph_valid"
            else None,
            "history_node_added_for_future": int(history_node_for_future),
            "history_node_added_source": history_node_source,
            "selected_local_frame_indices_zero_based": selected_indices,
            "m0": {
                "pred_node_idx": m0_node,
                "pred_node_id": m0_node_info["node_id"],
                "label": m0_label,
                "occurrence": m0_node_info["occurrence"],
                "display_name": m0_node_info["display_name"],
                "confidence": m0_confidence,
                "correct": m0_node == true_node,
                "top3": m0_top3,
            },
            "m3": {
                "pred_node_idx": m3_node,
                "pred_node_id": m3_node_info["node_id"],
                "label": m3_label,
                "occurrence": m3_node_info["occurrence"],
                "display_name": m3_node_info["display_name"],
                "confidence": m3_confidence,
                "correct": m3_node == true_node,
                "top3": m3_top3,
            },
            "e2e": {
                "pred_node_idx": e2e_node,
                "pred_node_id": e2e_node_info["node_id"],
                "label": e2e_label,
                "occurrence": e2e_node_info["occurrence"],
                "display_name": e2e_node_info["display_name"],
                "confidence": e2e_confidence,
                "correct": e2e_node == true_node,
                "top3": e2e_top3,
            },
            "inference_ms": elapsed_ms,
        }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def validate_all(
    data: DemoData,
    engine: InferenceEngine,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir or resolve(data.config["paths"]["outputs_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    engine.reset()
    results = [
        engine.predict(segment)
        for segment in data.segments
        if not segment["is_background"]
    ]

    existing_m0 = _prediction_lookup(
        resolve(data.config["paths"]["existing_m0_predictions"])
    )
    compare_m3_to_existing = (
        data.config["demo"].get("history_policy")
        == "completed_action_ground_truth_node_idx"
        and data.config["demo"].get("history_order", "graph_valid") == "graph_valid"
    )
    existing_m3 = (
        _prediction_lookup(resolve(data.config["paths"]["existing_m3_predictions"]))
        if compare_m3_to_existing
        else None
    )
    existing_e2e = _prediction_lookup(
        resolve(data.config["paths"]["existing_e2e_predictions"])
    )
    history_length_sequence_valid = all(
        int(result["history_length_before_current"]) == result_index
        for result_index, result in enumerate(results)
    )
    history_membership_and_node_source_valid = True
    actual_history_order_valid: bool | None = (
        True
        if data.config["demo"].get("history_order", "graph_valid") == "actual"
        else None
    )
    for result_index, result in enumerate(results):
        completed_results = results[:result_index]
        history_entries = result["history_entries_before_current"]
        expected_by_action = {
            int(completed["annotation_row_index"]): (
                int(completed["true_node_idx"])
                if data.config["demo"].get("history_policy")
                == "completed_action_ground_truth_node_idx"
                else int(completed["m3"]["pred_node_idx"])
            )
            for completed in completed_results
        }
        observed_by_action = {
            int(entry["annotation_row_index"]): int(entry["history_node_idx"])
            for entry in history_entries
        }
        expected_source = (
            "ground_truth"
            if data.config["demo"].get("history_policy")
            == "completed_action_ground_truth_node_idx"
            else "m3_prediction"
        )
        history_membership_and_node_source_valid = (
            history_membership_and_node_source_valid
            and observed_by_action == expected_by_action
            and all(
                entry["history_node_source"] == expected_source
                for entry in history_entries
            )
            and int(result["history_node_added_for_future"])
            == (
                int(result["true_node_idx"])
                if expected_source == "ground_truth"
                else int(result["m3"]["pred_node_idx"])
            )
            and result["history_node_added_source"] == expected_source
        )
        if actual_history_order_valid is not None:
            actual_history_order_valid = (
                actual_history_order_valid
                and [
                    int(entry["annotation_row_index"])
                    for entry in history_entries
                ]
                == [
                    int(completed["annotation_row_index"])
                    for completed in completed_results
                ]
            )

    for result in results:
        sample_name = result["sample_name"]
        result["existing_result_match"] = {
            "m0_node": (
                result["m0"]["pred_node_idx"]
                == int(existing_m0[sample_name]["pred_node_idx"])
            ),
            "m3_node": (
                result["m3"]["pred_node_idx"]
                == int(existing_m3[sample_name]["pred_node_idx"])
            )
            if existing_m3 is not None
            else None,
            "e2e_node": (
                result["e2e"]["pred_node_idx"]
                == int(existing_e2e[sample_name]["pred_node_idx"])
            ),
            "m0_confidence_abs_diff": abs(
                result["m0"]["confidence"]
                - float(existing_m0[sample_name]["node_confidence"])
            ),
            "m3_confidence_abs_diff": (
                abs(
                    result["m3"]["confidence"]
                    - float(existing_m3[sample_name]["node_confidence"])
                )
                if existing_m3 is not None
                else None
            ),
            "e2e_confidence_abs_diff": abs(
                result["e2e"]["confidence"]
                - float(existing_e2e[sample_name]["node_confidence"])
            ),
        }

    summary = {
        "schema_version": "task-graph-realtime-demo-inference-validation-v3",
        "profile_id": data.config["demo"].get("profile_id"),
        "participant": data.config["demo"]["participant"],
        "run": data.config["demo"]["source_run"],
        "seed": data.config["demo"]["seed"],
        "history_policy": data.config["demo"].get("history_policy"),
        "history_order": data.config["demo"].get("history_order", "graph_valid"),
        "device": engine.device_name,
        "actions": len(results),
        "m0_correct": sum(int(row["m0"]["correct"]) for row in results),
        "m0_accuracy": sum(int(row["m0"]["correct"]) for row in results) / len(results),
        "m3_correct": sum(int(row["m3"]["correct"]) for row in results),
        "m3_accuracy": sum(int(row["m3"]["correct"]) for row in results) / len(results),
        "e2e_correct": sum(int(row["e2e"]["correct"]) for row in results),
        "e2e_accuracy": sum(int(row["e2e"]["correct"]) for row in results) / len(results),
        "m0_existing_node_matches": sum(
            int(row["existing_result_match"]["m0_node"]) for row in results
        ),
        "m3_reference_comparison_applicable": compare_m3_to_existing,
        "m3_existing_node_matches": (
            sum(int(row["existing_result_match"]["m3_node"]) for row in results)
            if compare_m3_to_existing
            else None
        ),
        "e2e_existing_node_matches": sum(
            int(row["existing_result_match"]["e2e_node"]) for row in results
        ),
        "max_m0_confidence_abs_diff": max(
            row["existing_result_match"]["m0_confidence_abs_diff"] for row in results
        ),
        "max_m3_confidence_abs_diff": (
            max(
                row["existing_result_match"]["m3_confidence_abs_diff"]
                for row in results
            )
            if compare_m3_to_existing
            else None
        ),
        "max_e2e_confidence_abs_diff": max(
            row["existing_result_match"]["e2e_confidence_abs_diff"] for row in results
        ),
        "mean_inference_ms": sum(row["inference_ms"] for row in results) / len(results),
        "all_applicable_predictions_match_existing": all(
            row["existing_result_match"]["m0_node"]
            and row["existing_result_match"]["e2e_node"]
            and (
                row["existing_result_match"]["m3_node"]
                if compare_m3_to_existing
                else True
            )
            for row in results
        ),
        "all_predictions_match_existing": (
            all(
                row["existing_result_match"]["m0_node"]
                and row["existing_result_match"]["m3_node"]
                and row["existing_result_match"]["e2e_node"]
                for row in results
            )
            if compare_m3_to_existing
            else None
        ),
        "first_m3_error_action": next(
            (
                int(row["annotation_row_index"])
                for row in results
                if not row["m3"]["correct"]
            ),
            None,
        ),
        "history_protocol_checks": {
            "history_length_matches_completed_action_count": (
                history_length_sequence_valid
            ),
            "history_membership_and_node_source_valid": (
                history_membership_and_node_source_valid
            ),
            "actual_history_order_valid": actual_history_order_valid,
            "all_pass": (
                history_length_sequence_valid
                and history_membership_and_node_source_valid
                and (
                    actual_history_order_valid
                    if actual_history_order_valid is not None
                    else True
                )
            ),
        },
    }
    write_jsonl(output_dir / "validation_predictions.jsonl", results)
    write_json(output_dir / "validation_summary.json", summary)
    if not summary["history_protocol_checks"]["all_pass"]:
        raise RuntimeError(
            "History protocol audit failed; inspect validation_summary.json"
        )
    return summary


def _prediction_lookup(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["sample_name"]: row for row in csv.DictReader(handle)}
