from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected object in {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_package_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else package_root() / "config" / "experiment_config.json"
    config_path = config_path.resolve()
    config = read_json(config_path)
    config["_config_path"] = str(config_path)
    config["_package_root"] = str(config_path.parents[1])
    required = {"paths", "grid", "sequence_isolation", "experiments"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Package config is missing sections: {missing}")
    experiment_ids = [str(item["id"]) for item in config["experiments"]]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("Experiment IDs must be unique")
    return config


def _format_until_stable(value: str, variables: dict[str, Any]) -> str:
    result = str(value)
    for _ in range(8):
        previous = result
        result = result.format_map(variables)
        if result == previous:
            return result
    raise ValueError(f"Path template did not stabilize: {value}")


def resolve_paths(
    config: dict[str, Any], participant: str | None = None, seed: int | None = None
) -> dict[str, Path | str]:
    variables: dict[str, Any] = {
        "package_root": config["_package_root"],
        "participant": participant or "{participant}",
        "seed": seed if seed is not None else "{seed}",
        "camera_id": config["grid"]["camera_id"],
        "scope": config["grid"]["train_scope"],
    }
    for key, template in config["paths"].items():
        variables[key] = _format_until_stable(str(template), variables)
    resolved: dict[str, Path | str] = {}
    for key in config["paths"]:
        value = variables[key]
        resolved[key] = value if key == "python_executable" else Path(str(value)).resolve()
    return resolved


def run_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["participant"]), str(row["run"])


def group_runs(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[run_key(row)].append(row)
    for run_rows in grouped.values():
        run_rows.sort(key=lambda row: int(row["annotation_row_index"]))
    return dict(sorted(grouped.items()))


def sequence_values(
    rows: Iterable[dict[str, Any]], field: str, collapse_consecutive: bool = False
) -> tuple[int, ...]:
    values = tuple(int(row[field]) for row in rows)
    if not collapse_consecutive:
        return values
    collapsed: list[int] = []
    for value in values:
        if not collapsed or collapsed[-1] != value:
            collapsed.append(value)
    return tuple(collapsed)


def sequence_hash(values: Iterable[int]) -> str:
    payload = ",".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def manifest_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = group_runs(rows)
    node_support = Counter(int(row["node_idx"]) for row in rows)
    tier3_support = Counter(int(row["tier3_id"]) for row in rows)
    return {
        "samples": len(rows),
        "runs": len(grouped),
        "participants": sorted({str(row["participant"]) for row in rows}),
        "node_idx_support": {str(key): value for key, value in sorted(node_support.items())},
        "tier3_id_support": {str(key): value for key, value in sorted(tier3_support.items())},
        "missing_node_idx": [value for value in range(1, 36) if node_support[value] == 0],
        "missing_tier3_id": [value for value in range(31) if tier3_support[value] == 0],
    }

