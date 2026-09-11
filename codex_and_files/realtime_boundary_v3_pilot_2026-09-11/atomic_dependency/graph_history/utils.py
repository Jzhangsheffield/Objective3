from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_new_output_dir(path: str | Path, overwrite: bool = False) -> Path:
    """Create a dedicated output directory, refusing to overwrite prior experiments by default."""
    path = Path(path)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use --overwrite only for this dedicated new output."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def resolve_manifest(dataset_root: str | Path, manifest: str | Path) -> Path:
    manifest_path = Path(manifest)
    if manifest_path.is_absolute():
        return manifest_path
    return Path(dataset_root) / manifest_path


def run_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["participant"]), str(row["run"])


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary")
    for key in ("model_state_dict", "state_dict", "model", "net", "network"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if checkpoint and all(isinstance(k, str) for k in checkpoint):
        tensor_values = [value for value in checkpoint.values() if torch.is_tensor(value)]
        if tensor_values:
            return checkpoint
    raise ValueError("Unable to locate a model state_dict in checkpoint")


def strip_state_prefixes(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("module.", "model.", "backbone.", "encoder.", "encoder_q.")
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def load_compatible_state(model: torch.nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = strip_state_prefixes(extract_state_dict(checkpoint))
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    message = model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "loaded_keys": len(compatible),
        "model_keys": len(model_state),
        "missing_keys": list(message.missing_keys),
        "unexpected_keys": list(message.unexpected_keys),
    }


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    args: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "args": args,
    }
    if extra:
        payload["extra_info"] = extra
    torch.save(payload, path)


def append_csv(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def env_or(name: str, default: str) -> str:
    return os.environ.get(name, default)
