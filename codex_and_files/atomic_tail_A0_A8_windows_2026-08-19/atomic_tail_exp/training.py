from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import write_json
from .data import MultiViewHistoryDataset, collate_multiview_batch, safe_torch_load
from .graph import TaskGraph
from .losses import (
    aggregate_node_probabilities,
    symmetric_kl_consistency,
    tail_order_loss,
    tier3_nll,
)
from .metrics import classification_metrics
from .model import build_model


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def select_device(requested: str) -> torch.device:
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else requested if requested != "auto" else "cpu")


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def forward_view(model: torch.nn.Module, batch: dict[str, Any], view: str):
    return model(
        batch["current_feature"],
        batch[f"{view}_history_features"],
        batch[f"{view}_history_position_ids"],
        batch[f"{view}_history_padding_mask"],
        batch[f"{view}_history_shift_ids"],
    )


def build_loader(dataset, training_config: dict[str, Any], shuffle: bool, device: torch.device) -> DataLoader:
    workers = int(training_config["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        collate_fn=collate_multiview_batch,
    )


def load_checkpoint_compatible(model: torch.nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load this package's checkpoint or the legacy M2-Direct checkpoint by name/shape."""
    checkpoint = safe_torch_load(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dictionary: {checkpoint_path}")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise ValueError(f"No model state_dict found in {checkpoint_path}")
    remapped = {}
    for key, value in state.items():
        normalized = str(key)
        while normalized.startswith("module.") or normalized.startswith("model."):
            normalized = normalized.split(".", 1)[1]
        normalized = normalized.replace("node_classifier.norm.", "node_classifier.0.")
        normalized = normalized.replace("node_classifier.fc.", "node_classifier.1.")
        remapped[normalized] = value
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in remapped.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    result = model.load_state_dict(compatible, strict=False)
    required = {"fusion.weight", "node_classifier.1.weight", "attention.in_proj_weight"}
    absent = sorted(required - set(compatible))
    if absent:
        raise ValueError(f"Checkpoint is not compatible with M2-Direct; missing required tensors: {absent}")
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "loaded_tensor_count": len(compatible),
        "model_tensor_count": len(model_state),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "legacy_node_head_names_remapped": any("node_classifier.norm." in str(key) for key in state),
    }


def configure_trainable_parameters(model: torch.nn.Module, mode: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True
    if mode == "shift_only":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("shift_embedding.")
    elif mode == "base_only":
        for name, parameter in model.named_parameters():
            if name.startswith("shift_embedding."):
                parameter.requires_grad = False
    elif mode != "all":
        raise ValueError(f"Unsupported trainability mode: {mode}")


def build_phase_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    shift_learning_rate: float | None = None,
) -> torch.optim.Optimizer:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("Training phase has no trainable parameters")
    if shift_learning_rate is None:
        return torch.optim.AdamW(
            [parameter for _, parameter in trainable],
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
    base_parameters = [parameter for name, parameter in trainable if not name.startswith("shift_embedding.")]
    shift_parameters = [parameter for name, parameter in trainable if name.startswith("shift_embedding.")]
    groups = []
    if base_parameters:
        groups.append({"params": base_parameters, "lr": float(learning_rate)})
    if shift_parameters:
        groups.append({"params": shift_parameters, "lr": float(shift_learning_rate)})
    return torch.optim.AdamW(groups, weight_decay=float(weight_decay))


def train_phase(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    graph: TaskGraph,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    augmentation_config: dict[str, Any],
    experiment: dict[str, Any],
    epochs: int,
    phase: str,
    starting_epoch: int,
) -> list[dict[str, Any]]:
    amp = bool(training_config["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    node_to_tier3 = torch.tensor(graph.node_to_tier3, dtype=torch.long, device=device)
    logs = []
    calibration = phase == "actual_calibration"
    for local_epoch in range(1, int(epochs) + 1):
        global_epoch = starting_epoch + local_epoch
        loader.dataset.set_epoch(global_epoch)
        model.train()
        started = time.time()
        totals = {"loss": 0.0, "node_ce": 0.0, "consistency": 0.0, "tail_order": 0.0, "tier3": 0.0}
        total = 0
        correct = 0
        consistency_samples = 0
        tail_samples = 0
        for raw_batch in loader:
            batch = move_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                if calibration or experiment["train_view"] == "actual":
                    actual_logits, actual_extra = forward_view(model, batch, "actual")
                    primary_logits = actual_logits
                    node_ce = F.cross_entropy(actual_logits, batch["node_target"])
                    tier3_component = (
                        tier3_nll(actual_logits, batch["tier3_target"], node_to_tier3, int(model_config["num_tier3_classes"]))
                        if experiment["tier3_aux"]
                        else actual_logits.new_tensor(0.0)
                    )
                    consistency_component = actual_logits.new_tensor(0.0)
                    order_component = actual_logits.new_tensor(0.0)
                elif experiment["paired_views"]:
                    actual_logits, actual_extra = forward_view(model, batch, "actual")
                    augmented_logits, augmented_extra = forward_view(model, batch, "augmented")
                    primary_logits = augmented_logits
                    actual_ce_weight = float(experiment.get("actual_ce_weight", 0.5))
                    augmented_ce_weight = 1.0 - actual_ce_weight
                    node_ce = (
                        actual_ce_weight * F.cross_entropy(actual_logits, batch["node_target"])
                        + augmented_ce_weight * F.cross_entropy(augmented_logits, batch["node_target"])
                    )
                    consistency_component, selected = (
                        symmetric_kl_consistency(
                            actual_logits,
                            augmented_logits,
                            float(augmentation_config["consistency_confidence_threshold"]),
                        )
                        if experiment["consistency"]
                        else (actual_logits.new_tensor(0.0), 0)
                    )
                    consistency_samples += selected
                    if experiment["tail_order_aux"]:
                        _, corrupted_extra = forward_view(model, batch, "corrupted")
                        valid_order_logits = model.tail_order_logits(augmented_extra["history_context"])
                        corrupted_order_logits = model.tail_order_logits(corrupted_extra["history_context"])
                        order_component, selected = tail_order_loss(
                            valid_order_logits, corrupted_order_logits, batch["corruption_valid"]
                        )
                        tail_samples += selected
                    else:
                        order_component = actual_logits.new_tensor(0.0)
                    tier3_component = (
                        0.5 * (
                            tier3_nll(actual_logits, batch["tier3_target"], node_to_tier3, int(model_config["num_tier3_classes"]))
                            + tier3_nll(augmented_logits, batch["tier3_target"], node_to_tier3, int(model_config["num_tier3_classes"]))
                        )
                        if experiment["tier3_aux"]
                        else actual_logits.new_tensor(0.0)
                    )
                else:
                    augmented_logits, augmented_extra = forward_view(model, batch, "augmented")
                    primary_logits = augmented_logits
                    node_ce = F.cross_entropy(augmented_logits, batch["node_target"])
                    consistency_component = augmented_logits.new_tensor(0.0)
                    order_component = augmented_logits.new_tensor(0.0)
                    tier3_component = (
                        tier3_nll(augmented_logits, batch["tier3_target"], node_to_tier3, int(model_config["num_tier3_classes"]))
                        if experiment["tier3_aux"]
                        else augmented_logits.new_tensor(0.0)
                    )
                loss = (
                    node_ce
                    + float(augmentation_config["consistency_weight"]) * consistency_component
                    + float(augmentation_config["tail_order_loss_weight"]) * order_component
                    + float(augmentation_config["tier3_loss_weight"]) * tier3_component
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_config["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(batch["node_target"].shape[0])
            total += batch_size
            correct += int((primary_logits.argmax(dim=-1) == batch["node_target"]).sum())
            for name, value in (
                ("loss", loss), ("node_ce", node_ce), ("consistency", consistency_component),
                ("tail_order", order_component), ("tier3", tier3_component),
            ):
                totals[name] += float(value.detach()) * batch_size
        row = {
            "epoch": global_epoch,
            "phase": phase,
            "train_loss": totals["loss"] / max(1, total),
            "train_node_ce": totals["node_ce"] / max(1, total),
            "train_consistency_loss": totals["consistency"] / max(1, total),
            "train_tail_order_loss": totals["tail_order"] / max(1, total),
            "train_tier3_loss": totals["tier3"] / max(1, total),
            "train_node_accuracy": correct / max(1, total),
            "consistency_selected": consistency_samples,
            "tail_order_selected": tail_samples,
            "seconds": time.time() - started,
        }
        logs.append(row)
        print(
            f"[{experiment['experiment_id']}] {phase} epoch={global_epoch:03d} "
            f"loss={row['train_loss']:.5f} node_acc={row['train_node_accuracy']:.4f} "
            f"seconds={row['seconds']:.1f}",
            flush=True,
        )
    return logs


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    graph: TaskGraph,
    model_config: dict[str, Any],
    output_dir: Path,
    split_name: str,
) -> dict[str, Any]:
    model.eval()
    node_to_tier3 = torch.tensor(graph.node_to_tier3, dtype=torch.long, device=device)
    node_true: list[int] = []
    node_pred: list[int] = []
    tier3_true: list[int] = []
    tier3_pred: list[int] = []
    stages: list[int] = []
    prediction_rows = []
    all_probabilities = []
    for raw_batch in loader:
        batch = move_to_device(raw_batch, device)
        logits, _ = forward_view(model, batch, "actual")
        node_probabilities = F.softmax(logits, dim=-1)
        tier3_probabilities = aggregate_node_probabilities(
            node_probabilities, node_to_tier3, int(model_config["num_tier3_classes"])
        )
        predicted_nodes = node_probabilities.argmax(dim=-1)
        predicted_tier3 = tier3_probabilities.argmax(dim=-1)
        all_probabilities.append(node_probabilities.cpu())
        for index in range(logits.shape[0]):
            true_node = int(batch["node_target"][index])
            pred_node = int(predicted_nodes[index])
            true_tier3 = int(batch["tier3_target"][index])
            pred_tier3 = int(predicted_tier3[index])
            stage = int(batch["stage_id"][index])
            node_true.append(true_node)
            node_pred.append(pred_node)
            tier3_true.append(true_tier3)
            tier3_pred.append(pred_tier3)
            stages.append(stage)
            prediction_rows.append({
                "sample_name": raw_batch["sample_name"][index],
                "participant": raw_batch["participant"][index],
                "run": raw_batch["run"][index],
                "annotation_row_index": raw_batch["annotation_row_index"][index],
                "stage_id": stage,
                "true_node_idx": true_node + 1,
                "pred_node_idx": pred_node + 1,
                "true_tier3_id": true_tier3,
                "pred_tier3_id": pred_tier3,
                "node_confidence": float(node_probabilities[index, pred_node]),
                "tier3_confidence": float(tier3_probabilities[index, pred_tier3]),
            })
    metrics: dict[str, Any] = {
        "split": split_name,
        "samples": len(node_true),
        "history_order": "actual_chronological",
        "node": classification_metrics(node_true, node_pred, int(model_config["num_nodes"])),
        "tier3": classification_metrics(tier3_true, tier3_pred, int(model_config["num_tier3_classes"])),
        "per_stage": {},
    }
    for stage in (1, 2, 3):
        indices = [index for index, value in enumerate(stages) if value == stage]
        metrics["per_stage"][str(stage)] = {
            "samples": len(indices),
            "node": classification_metrics([node_true[i] for i in indices], [node_pred[i] for i in indices], int(model_config["num_nodes"])),
            "tier3": classification_metrics([tier3_true[i] for i in indices], [tier3_pred[i] for i in indices], int(model_config["num_tier3_classes"])),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"{split_name}_metrics.json", metrics)
    with (output_dir / f"{split_name}_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        if prediction_rows:
            writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
            writer.writeheader()
            writer.writerows(prediction_rows)
    torch.save({"node_probabilities": torch.cat(all_probabilities) if all_probabilities else torch.empty(0), "rows": prediction_rows}, output_dir / f"{split_name}_probabilities.pt")
    return metrics


def run_training(config: dict[str, Any], spec: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    output_dir = Path(spec["output_dir"])
    completed_path = output_dir / "completed.json"
    if completed_path.is_file() and not overwrite:
        return {"status": "skipped", "output_dir": str(output_dir)}
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Incomplete non-empty output directory exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(spec["seed"]), bool(config["training"].get("deterministic", True)))
    device = select_device(str(config["training"]["device"]))
    paths = spec["paths"]
    graph = TaskGraph.load(paths["task_graph"], paths["relation_matrix"])
    train_manifest = Path(paths["protocol_root"]) / spec["scope"] / "train.jsonl"
    train_dataset = MultiViewHistoryDataset(
        paths["train_cache"], train_manifest, graph, spec,
        config["augmentation"], int(spec["seed"]), training=True,
    )
    if train_dataset.feature_dim != int(config["model"]["feature_dim"]):
        raise ValueError(f"Configured feature_dim={config['model']['feature_dim']}, cache has {train_dataset.feature_dim}")
    train_loader = build_loader(train_dataset, config["training"], True, device)
    model = build_model(config["model"])
    warm_start_report = None
    if spec.get("warm_start_checkpoint"):
        checkpoint_path = Path(spec["warm_start_checkpoint"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Warm-start checkpoint missing: {checkpoint_path}")
        warm_start_report = load_checkpoint_compatible(model, checkpoint_path)
    shared_a0_reuse = (
        spec["experiment_id"] == "A0"
        and bool(spec.get("reuse_shared_a0_checkpoint"))
    )
    if shared_a0_reuse:
        shared_checkpoint = Path(spec["shared_a0_checkpoint"])
        if not shared_checkpoint.is_file():
            raise FileNotFoundError(f"Shared A0 checkpoint missing: {shared_checkpoint}")
        warm_start_report = load_checkpoint_compatible(model, shared_checkpoint)
    model = model.to(device)
    schedule = spec["schedule"]
    if shared_a0_reuse:
        phases = []
    elif schedule == "baseline":
        phases = [{
            "name": "baseline", "epochs": int(config["training"]["baseline_epochs"]),
            "learning_rate": float(config["training"]["learning_rate"]), "trainability": "all",
        }]
    elif schedule == "scratch":
        phases = [{
            "name": "scratch", "epochs": int(config["training"]["scratch_epochs"]),
            "learning_rate": float(config["training"]["learning_rate"]), "trainability": "all",
        }]
    elif schedule == "dualpos_finetune_calibrate":
        phases = [
            {
                "name": "dualpos_shift_warmup",
                "epochs": int(spec["shift_warmup_epochs"]),
                "learning_rate": float(spec["shift_warmup_learning_rate"]),
                "shift_learning_rate": float(spec["shift_warmup_learning_rate"]),
                "trainability": "shift_only",
            },
            {
                "name": "dualpos_mixed_finetune",
                "epochs": int(spec["mixed_finetune_epochs"]),
                "learning_rate": float(spec["finetune_learning_rate"]),
                "shift_learning_rate": float(spec["shift_learning_rate"]),
                "trainability": "all",
            },
            {
                "name": "actual_calibration",
                "epochs": int(spec["actual_calibration_epochs"]),
                "learning_rate": float(spec["calibration_learning_rate"]),
                "trainability": "base_only",
            },
        ]
    else:
        phases = [
            {
                "name": "mixed_finetune",
                "epochs": int(spec.get("mixed_finetune_epochs", config["training"]["mixed_finetune_epochs"])),
                "learning_rate": float(spec.get("finetune_learning_rate", config["training"]["finetune_learning_rate"])),
                "trainability": "all",
            },
            {
                "name": "actual_calibration",
                "epochs": int(spec.get("actual_calibration_epochs", config["training"]["actual_calibration_epochs"])),
                "learning_rate": float(spec.get("calibration_learning_rate", config["training"]["calibration_learning_rate"])),
                "trainability": "all",
            },
        ]
    write_json(output_dir / "resolved_run_config.json", {"global": config, "run": spec, "device": str(device), "warm_start_report": warm_start_report})
    audit = train_dataset.audit()
    write_json(output_dir / "augmentation_audit.json", audit)
    logs: list[dict[str, Any]] = []
    epoch_offset = 0
    for phase_spec in phases:
        phase_name = str(phase_spec["name"])
        phase_epochs = int(phase_spec["epochs"])
        learning_rate = float(phase_spec["learning_rate"])
        if phase_epochs <= 0:
            continue
        configure_trainable_parameters(model, str(phase_spec.get("trainability", "all")))
        optimizer = build_phase_optimizer(
            model,
            learning_rate,
            float(config["training"]["weight_decay"]),
            float(phase_spec["shift_learning_rate"]) if "shift_learning_rate" in phase_spec else None,
        )
        phase_logs = train_phase(
            model, train_loader, optimizer, device, graph, config["model"], config["training"],
            config["augmentation"], spec, phase_epochs, phase_name, epoch_offset,
        )
        logs.extend(phase_logs)
        epoch_offset += phase_epochs
        write_json(output_dir / "train_log.json", logs)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "experiment_id": spec["experiment_id"],
                "epochs": epoch_offset,
                "completed_phase": phase_name,
                "run_spec": spec,
                "model_config": config["model"],
            },
            output_dir / f"after_{phase_name}.pth",
        )
    write_json(output_dir / "train_log.json", logs)
    checkpoint_path = Path(spec["shared_a0_checkpoint"]) if shared_a0_reuse else output_dir / "last.pth"
    if not shared_a0_reuse:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "experiment_id": spec["experiment_id"],
                "epochs": epoch_offset,
                "run_spec": spec,
                "model_config": config["model"],
                "train_log": logs,
            },
            checkpoint_path,
        )
    test_result_dir = output_dir / "test_results_actual_order"
    metrics = {}
    for split_name in config["grid"]["test_splits"]:
        manifest = Path(paths["protocol_root"]) / spec["scope"] / f"{split_name}.jsonl"
        dataset = MultiViewHistoryDataset(
            paths["test_cache"], manifest, graph, spec,
            config["augmentation"], int(spec["seed"]), training=False,
        )
        loader = build_loader(dataset, config["training"], False, device)
        metrics[split_name] = evaluate(model, loader, device, graph, config["model"], test_result_dir, split_name)
        print(
            f"[{spec['experiment_id']}] {split_name}: "
            f"node_acc={metrics[split_name]['node']['accuracy']:.4f} "
            f"tier3_acc={metrics[split_name]['tier3']['accuracy']:.4f}",
            flush=True,
        )
    completed = {
        "status": "completed",
        "experiment_id": spec["experiment_id"],
        "participant": spec["participant"],
        "seed": spec["seed"],
        "scope": spec["scope"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_reused_without_copy": shared_a0_reuse,
        "evaluation_history_order": "actual_chronological",
        "uses_current_target_for_reordering": False,
        "metrics": metrics,
    }
    write_json(completed_path, completed)
    return completed
