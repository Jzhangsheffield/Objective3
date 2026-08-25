from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from phase_a.config import load_config
from phase_a.io import file_sha256, read_jsonl, write_json
from phase_a.paths import a0_result_dir, primary_feature_cache, protocol_dir, secondary_feature_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase A data and immutable LOSO protocol inputs")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_a.json"))
    parser.add_argument("--output", default=str(PACKAGE_ROOT / "audit" / "dataset_audit.json"))
    parser.add_argument("--load-tensors", action="store_true", help="Also torch.load every MindRove file")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["dataset_root"])
    global_manifest = root / "3_camera_mindrove_manifest.jsonl"
    rows = read_jsonl(global_manifest)
    cameras = [config["primary_camera_id"], config["secondary_camera_id"], config["alternative_secondary_camera_id"]]
    report = {
        "status": "METADATA_PASS_TENSOR_AUDIT_PENDING", "tensor_audit": "PENDING",
        "dataset_root": str(root), "global_manifest": str(global_manifest),
        "global_manifest_sha256": file_sha256(global_manifest), "samples": len(rows),
        "participants": dict(Counter(str(row["participant"]) for row in rows)),
        "unique_sample_names": len({str(row["sample_name"]) for row in rows}),
        "missing_files": {}, "shape_metadata": {}, "folds": {}, "a0": {}, "warnings": [], "errors": [],
    }
    for camera in cameras:
        report["missing_files"][camera] = sum(
            not (root / row.get(f"{camera}_rgb", "__missing__")).is_file() for row in rows
        )
    report["missing_files"]["mindrove"] = sum(
        not (root / row.get("mindrove", "__missing__")).is_file() for row in rows
    )
    right_lengths = [int(row["mindrove_right_emg_shape"][0]) for row in rows]
    frame_deltas = [max(int(row[f"{camera}_num_rgb_frames"]) for camera in cameras) -
                    min(int(row[f"{camera}_num_rgb_frames"]) for camera in cameras) for row in rows]
    rate_rows = []
    for row in rows:
        duration = float(row["mindrove_right_end_board_ts"]) - float(row["mindrove_right_start_board_ts"])
        rate = float(row["mindrove_right_emg_shape"][0]) / duration if duration > 0 else float("nan")
        rate_rows.append((rate, str(row["sample_name"]), duration, int(row["mindrove_right_emg_shape"][0])))
    rates = sorted(value[0] for value in rate_rows)
    rate_outliers = [
        {"sample_name": sample, "samples": length, "duration_seconds": duration, "effective_hz": rate}
        for rate, sample, duration, length in rate_rows if rate < 480.0 or rate > 520.0
    ]
    report["shape_metadata"] = {
        "right_emg_channels": sorted({int(row["mindrove_right_emg_shape"][1]) for row in rows}),
        "right_acc_channels": sorted({int(row["mindrove_right_acc_shape"][1]) for row in rows}),
        "right_gyro_channels": sorted({int(row["mindrove_right_gyro_shape"][1]) for row in rows}),
        "right_length_min_median_max": [min(right_lengths), statistics.median(right_lengths), max(right_lengths)],
        "right_emg_acc_gyro_length_mismatches": sum(
            row["mindrove_right_emg_shape"][0] != row["mindrove_right_acc_shape"][0] or
            row["mindrove_right_emg_shape"][0] != row["mindrove_right_gyro_shape"][0] for row in rows
        ),
        "three_camera_frame_delta_median_max": [statistics.median(frame_deltas), max(frame_deltas)],
    }
    report["timing_metadata"] = {
        "right_effective_hz_min_median_max": [min(rates), statistics.median(rates), max(rates)],
        "right_effective_hz_outside_480_520_count": len(rate_outliers),
        "right_effective_hz_outliers": sorted(rate_outliers, key=lambda value: value["effective_hz"]),
    }
    if any(report["missing_files"].values()):
        report["errors"].append("One or more modality files are missing")
    if any("node_idx" not in row or "stage_id" not in row for row in rows):
        report["warnings"].append(
            "Global 3_camera_mindrove_manifest lacks node_idx/stage_id; enriched fold manifests/protocols are required."
        )
    if rate_outliers:
        report["warnings"].append(
            f"{len(rate_outliers)} clips have manifest-derived right-hand effective sampling rate outside 480-520 Hz."
        )
    label_rows = []
    for participant in config["participants"]:
        train_path, test_path = root / f"{participant}_as_test" / "train_manifest.jsonl", root / f"{participant}_as_test" / "test_manifest.jsonl"
        train_rows, test_rows = read_jsonl(train_path), read_jsonl(test_path)
        label_rows.extend(test_rows)
        train_names, test_names = {row["sample_name"] for row in train_rows}, {row["sample_name"] for row in test_rows}
        protocol = protocol_dir(config, participant)
        protocol_train, protocol_test = read_jsonl(protocol / "train.jsonl"), read_jsonl(protocol / "test_all.jsonl")
        fold = {
            "train_samples": len(train_rows), "test_samples": len(test_rows),
            "train_participants": sorted({row["participant"] for row in train_rows}),
            "test_participants": sorted({row["participant"] for row in test_rows}),
            "train_test_overlap": len(train_names & test_names),
            "source_protocol_train_identical": train_names == {row["sample_name"] for row in protocol_train},
            "source_protocol_test_identical": test_names == {row["sample_name"] for row in protocol_test},
            "protocol_sha256": {name: file_sha256(protocol / name) for name in (
                "train.jsonl", "test_all.jsonl", "test_normal.jsonl", "test_fault.jsonl"
            )},
        }
        report["folds"][participant] = fold
        if fold["train_test_overlap"] or not fold["source_protocol_train_identical"] or not fold["source_protocol_test_identical"]:
            report["errors"].append(f"Manifest/protocol mismatch in {participant}_as_test")
        for seed in config["seeds"]:
            key = f"{participant}_seed{seed}"
            result_dir = a0_result_dir(config, participant, seed)
            required = [result_dir / f"test_{split}_{suffix}" for split in ("all", "normal", "fault")
                        for suffix in ("metrics.json", "predictions.csv", "probabilities.pt")]
            report["a0"][key] = {
                "complete": all(path.is_file() for path in required),
                "primary_train_cache": primary_feature_cache(config, participant, seed, "train").is_file(),
                "primary_test_cache": primary_feature_cache(config, participant, seed, "test").is_file(),
                "secondary_train_cache": secondary_feature_cache(config, participant, seed, "train").is_file(),
                "secondary_test_cache": secondary_feature_cache(config, participant, seed, "test").is_file(),
            }
            if not report["a0"][key]["complete"]:
                report["errors"].append(f"A0 incomplete: {key}")
    node_mapping, mapping_conflicts = {}, []
    for row in label_rows:
        node = int(row["node_idx"])
        value = (int(row["tier3_id"]), int(row["stage_id"]))
        if node in node_mapping and node_mapping[node] != value:
            mapping_conflicts.append({"node_idx": node, "first": node_mapping[node], "other": value})
        node_mapping[node] = value
    report["label_integrity"] = {
        "union_test_samples": len(label_rows),
        "union_test_unique_samples": len({row["sample_name"] for row in label_rows}),
        "node_ids": sorted(node_mapping),
        "tier3_ids": sorted({int(row["tier3_id"]) for row in label_rows}),
        "stage_counts": dict(Counter(str(row["stage_id"]) for row in label_rows)),
        "node_to_tier3_stage_conflicts": mapping_conflicts,
    }
    if len(node_mapping) != 35 or len(report["label_integrity"]["tier3_ids"]) != 31 or mapping_conflicts:
        report["errors"].append("Node/Tier3/stage mapping is incomplete or inconsistent")
    if args.load_tensors:
        import torch
        tensor_errors = []
        for index, row in enumerate(rows, 1):
            try:
                value = torch.load(root / row["mindrove"], map_location="cpu", weights_only=False)
                for key, channels in (("right_emg", 8), ("right_acc", 3), ("right_gyro", 3)):
                    tensor = value[key]
                    if tensor.ndim != 2 or tensor.shape[1] != channels or not torch.isfinite(tensor).all():
                        raise ValueError(f"{key}: {tuple(tensor.shape)} or non-finite")
            except Exception as error:
                tensor_errors.append({"sample_name": row["sample_name"], "error": repr(error)})
            if index % 250 == 0:
                print(f"tensor audit {index}/{len(rows)}", flush=True)
        report["tensor_errors"] = tensor_errors
        report["tensor_audit"] = "PASS" if not tensor_errors else "FAIL"
        if tensor_errors:
            report["errors"].append(f"MindRove tensor errors: {len(tensor_errors)}")
    if report["errors"]:
        report["status"] = "FAIL"
    elif args.load_tensors:
        report["status"] = "PASS"
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
