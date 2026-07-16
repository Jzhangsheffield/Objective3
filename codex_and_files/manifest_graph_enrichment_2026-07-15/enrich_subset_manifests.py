from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
MASTER = HERE / "3_camera_mindrove_manifest_with_graph_nodes.jsonl"
BACKUP_ROOT = HERE / "subset_manifests_backup_2026-07-15"
STAGED_ROOT = HERE / "subset_manifests_enriched_staged"
REPORT = HERE / "subset_manifest_enrichment_report.json"

TARGET_DIRS = [
    Path(r"C:\Junxi_data_for_training_speedup\Stage_2_Mapstyle_Dataset\Only_falut_run_as_test_M"),
    Path(r"C:\Junxi_data_for_training_speedup\Stage_2_Mapstyle_Dataset\A_as_test"),
    Path(r"C:\Junxi_data_for_training_speedup\Stage_2_Mapstyle_Dataset\D_as_test"),
    Path(r"C:\Junxi_data_for_training_speedup\Stage_2_Mapstyle_Dataset\J_as_test"),
    Path(r"C:\Junxi_data_for_training_speedup\Stage_2_Mapstyle_Dataset\M_as_test"),
    Path(r"C:\Junxi_data_for_training_speedup\Stage_2_Mapstyle_Dataset\Only_falut_run_as_test_A"),
    Path(r"C:\Junxi_data_for_training_speedup\Stage_2_Mapstyle_Dataset\Only_falut_run_as_test_D"),
    Path(r"C:\Junxi_data_for_training_speedup\Stage_2_Mapstyle_Dataset\Only_falut_run_as_test_J"),
]

GRAPH_FIELDS = ("node_id", "node_idx", "stage_id")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}: {error}") from error
    return rows


def without_graph_fields(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in GRAPH_FIELDS}


def add_graph_fields(row: dict, reference: dict) -> dict:
    output = {}
    inserted = False
    for key, value in row.items():
        if key in GRAPH_FIELDS:
            continue
        output[key] = value
        if key == "tier3":
            output["node_id"] = reference["node_id"]
            output["node_idx"] = reference["node_idx"]
            output["stage_id"] = reference["stage_id"]
            inserted = True
    if not inserted:
        for key in GRAPH_FIELDS:
            output[key] = reference[key]
    return output


def manifest_files() -> list[Path]:
    files = []
    for directory in TARGET_DIRS:
        if not directory.is_dir():
            raise FileNotFoundError(f"Target directory not found: {directory}")
        files.extend(sorted(path for path in directory.rglob("*.jsonl") if "manifest" in path.name.lower()))
    return files


def load_master() -> dict[str, dict]:
    rows = read_jsonl(MASTER)
    by_sample = {}
    for row in rows:
        sample_name = row.get("sample_name")
        if not sample_name:
            raise ValueError("Master manifest contains a row without sample_name")
        if sample_name in by_sample:
            raise ValueError(f"Duplicate sample_name in master: {sample_name}")
        by_sample[sample_name] = row
    return by_sample


def stage_and_backup() -> dict:
    master = load_master()
    files = manifest_files()
    file_reports = []
    all_missing = []
    all_content_mismatches = []

    for source in files:
        directory_name = source.parent.name
        backup = BACKUP_ROOT / directory_name / source.name
        staged = STAGED_ROOT / directory_name / source.name
        rows = read_jsonl(source)

        sample_names = [row.get("sample_name") for row in rows]
        duplicates = sorted({name for name in sample_names if name is not None and sample_names.count(name) > 1})
        missing = sorted({name for name in sample_names if name not in master})
        mismatches = []
        existing_graph_field_rows = 0
        enriched_rows = []
        for line_number, row in enumerate(rows, start=1):
            sample_name = row.get("sample_name")
            if any(key in row for key in GRAPH_FIELDS):
                existing_graph_field_rows += 1
            if sample_name not in master:
                continue
            reference = master[sample_name]
            if without_graph_fields(row) != without_graph_fields(reference):
                mismatches.append({"line_number": line_number, "sample_name": sample_name})
                continue
            enriched_rows.append(add_graph_fields(row, reference))

        all_missing.extend({"file": str(source), "sample_name": name} for name in missing)
        all_content_mismatches.extend({"file": str(source), **item} for item in mismatches)
        if missing or mismatches or len(enriched_rows) != len(rows):
            file_reports.append({
                "source": str(source),
                "line_count": len(rows),
                "missing_sample_count": len(missing),
                "content_mismatch_count": len(mismatches),
                "duplicate_sample_names": duplicates,
                "existing_graph_field_rows": existing_graph_field_rows,
                "status": "not_staged",
            })
            continue

        backup.parent.mkdir(parents=True, exist_ok=True)
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, backup)
        with staged.open("w", encoding="utf-8", newline="\n") as handle:
            for row in enriched_rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

        staged_rows = read_jsonl(staged)
        assert len(staged_rows) == len(rows)
        for original, enriched in zip(rows, staged_rows):
            sample_name = original["sample_name"]
            reference = master[sample_name]
            assert without_graph_fields(enriched) == without_graph_fields(original)
            assert all(enriched[key] == reference[key] for key in GRAPH_FIELDS)

        file_reports.append({
            "source": str(source),
            "backup": str(backup),
            "staged": str(staged),
            "line_count": len(rows),
            "missing_sample_count": 0,
            "content_mismatch_count": 0,
            "duplicate_sample_names": duplicates,
            "existing_graph_field_rows": existing_graph_field_rows,
            "source_sha256": sha256(source),
            "backup_sha256": sha256(backup),
            "staged_sha256": sha256(staged),
            "status": "staged_and_backed_up",
        })

    report = {
        "master_manifest": str(MASTER),
        "master_sample_count": len(master),
        "target_directory_count": len(TARGET_DIRS),
        "manifest_file_count": len(files),
        "total_subset_rows": sum(item["line_count"] for item in file_reports),
        "all_samples_found": len(all_missing) == 0,
        "all_original_records_match_master": len(all_content_mismatches) == 0,
        "missing_samples": all_missing,
        "content_mismatches": all_content_mismatches,
        "backup_root": str(BACKUP_ROOT),
        "staged_root": str(STAGED_ROOT),
        "applied": False,
        "files": file_reports,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if all_missing or all_content_mismatches or any(item["status"] != "staged_and_backed_up" for item in file_reports):
        raise SystemExit(f"Stopped safely before modifying targets. See {REPORT}")
    return report


def apply_staged() -> dict:
    if not REPORT.exists():
        raise FileNotFoundError("Run staging and backup before --apply")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if not report.get("all_samples_found") or not report.get("all_original_records_match_master"):
        raise SystemExit("Report does not authorize applying staged files")

    for item in report["files"]:
        source = Path(item["source"])
        backup = Path(item["backup"])
        staged = Path(item["staged"])
        if sha256(backup) != item["source_sha256"]:
            raise RuntimeError(f"Backup hash mismatch: {backup}")
        if sha256(source) != item["source_sha256"]:
            raise RuntimeError(f"Target changed after backup; refusing to overwrite: {source}")
        if sha256(staged) != item["staged_sha256"]:
            raise RuntimeError(f"Staged file changed unexpectedly: {staged}")

    for item in report["files"]:
        source = Path(item["source"])
        staged = Path(item["staged"])
        temporary = source.with_name(source.name + ".graph_enrichment_tmp")
        shutil.copy2(staged, temporary)
        os.replace(temporary, source)

    for item in report["files"]:
        source = Path(item["source"])
        item["applied_sha256"] = sha256(source)
        item["applied_matches_staged"] = item["applied_sha256"] == item["staged_sha256"]
        item["status"] = "applied_and_verified" if item["applied_matches_staged"] else "verification_failed"
    report["applied"] = all(item["applied_matches_staged"] for item in report["files"])
    report["applied_file_count"] = sum(item["applied_matches_staged"] for item in report["files"])
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not report["applied"]:
        raise SystemExit("One or more applied files failed verification")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        apply_staged()
    else:
        stage_and_backup()


if __name__ == "__main__":
    main()
