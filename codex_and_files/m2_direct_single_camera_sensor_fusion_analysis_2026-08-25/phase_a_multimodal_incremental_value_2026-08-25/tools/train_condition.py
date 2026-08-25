from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import torch
from torch.utils.data import DataLoader

from phase_a.config import load_config, validate_condition
from phase_a.data import MultimodalHistoryDataset, collate_multimodal
from phase_a.engine import evaluate, move_batch, train_model
from phase_a.io import read_jsonl, seed_everything, write_json
from phase_a.metrics import derive_node_to_tier3
from phase_a.models import PhaseAM2Direct
from phase_a.paths import a0_checkpoint, model_dir, primary_feature_cache, protocol_dir, secondary_feature_cache, signal_cache


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one A1/A3-A7 fold×seed condition")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_a.json"))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--participant", required=True, choices=list("ADJM"))
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    condition = validate_condition(args.condition)
    if condition not in {"A1", "A3", "A4", "A5", "A6", "A7"}:
        raise ValueError("A0 is imported and A2 is probability fusion; train_condition supports A1/A3-A7")
    config = load_config(args.config)
    seed_everything(args.seed)
    output = model_dir(config, condition, args.participant, args.seed)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    protocols = protocol_dir(config, args.participant)
    node_to_tier3 = derive_node_to_tier3(read_jsonl(protocols / "train.jsonl"))
    datasets = {}
    for split, manifest in (("train", "train.jsonl"), ("test_all", "test_all.jsonl"),
                            ("test_normal", "test_normal.jsonl"), ("test_fault", "test_fault.jsonl")):
        cache_split = "train" if split == "train" else "test"
        secondary_path = (secondary_feature_cache(config, args.participant, args.seed, cache_split)
                          if condition in {"A1", "A3", "A7"}
                          else primary_feature_cache(config, args.participant, args.seed, cache_split))
        datasets[split] = MultimodalHistoryDataset(
            primary_feature_cache(config, args.participant, args.seed, cache_split),
            secondary_path,
            signal_cache(config, args.participant, cache_split), protocols / manifest,
            training=split == "train",
            time_shift_augmentation_probability=config["sensor_time_shift_augmentation_probability"],
            time_shift_augmentation_max_fraction=config["sensor_time_shift_augmentation_max_fraction"],
        )
    device = select_device(args.device)
    train_loader = DataLoader(
        datasets["train"], batch_size=config["batch_size"], shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_multimodal,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    model = PhaseAM2Direct(
        condition, config["feature_dim"], config["d_model"], config["num_heads"],
        config["max_history"], config["dropout"], config["modality_dropout"],
    ).to(device)
    initialization = "scratch"
    fallback_equivalence_max_abs_error = None
    if condition in {"A3", "A4", "A5", "A6", "A7"}:
        baseline_checkpoint = torch.load(a0_checkpoint(config, args.participant, args.seed),
                                         map_location="cpu", weights_only=False)
        state = baseline_checkpoint.get("model_state_dict", baseline_checkpoint.get("model"))
        if not isinstance(state, dict):
            raise ValueError("Unable to locate A0 M2 model state_dict")
        message = model.load_state_dict(state, strict=False)
        unexpected = list(message.unexpected_keys)
        missing_core = [key for key in message.missing_keys if not key.startswith("observation.")]
        if unexpected or missing_core:
            raise RuntimeError(f"A0 initialization mismatch: unexpected={unexpected}, missing_core={missing_core}")
        initialization = str(a0_checkpoint(config, args.participant, args.seed))
        if config["freeze_a0_core_for_A3_A7"]:
            model.freeze_m2_core()
        reference = PhaseAM2Direct(
            "A1", config["feature_dim"], config["d_model"], config["num_heads"],
            config["max_history"], config["dropout"], config["modality_dropout"],
        ).to(device)
        reference.load_state_dict(state, strict=True)
        check_batch = move_batch(next(iter(train_loader)), device)
        for modality in ("secondary", "emg", "imu"):
            check_batch[f"current_{modality}_available"].zero_()
            check_batch[f"history_{modality}_available"].zero_()
        reference_batch = dict(check_batch)
        reference_batch["current_secondary"] = check_batch["current_primary"]
        reference_batch["history_secondary"] = check_batch["history_primary"]
        model.eval(); reference.eval()
        with torch.inference_mode():
            candidate_logits, _ = model(check_batch)
            reference_logits, _ = reference(reference_batch)
        fallback_equivalence_max_abs_error = float((candidate_logits - reference_logits).abs().max())
        if fallback_equivalence_max_abs_error > 1e-5:
            raise RuntimeError(f"Primary-only fallback is not A0-equivalent: {fallback_equivalence_max_abs_error}")
    accumulation = max(1, config["effective_batch_size"] // config["batch_size"])
    log, optimizer = train_model(
        model, train_loader, device, config["epochs"], config["learning_rate"],
        config["weight_decay"], config["action_loss_weight"], node_to_tier3, accumulation,
    )
    fallback_equivalence_after_training = None
    if condition in {"A3", "A4", "A5", "A6", "A7"} and config["freeze_a0_core_for_A3_A7"]:
        model.eval(); reference.eval()
        with torch.inference_mode():
            candidate_logits, _ = model(check_batch)
            reference_logits, _ = reference(reference_batch)
        fallback_equivalence_after_training = float((candidate_logits - reference_logits).abs().max())
        if fallback_equivalence_after_training > 1e-5:
            raise RuntimeError(f"Trained primary-only fallback drifted from A0: {fallback_equivalence_after_training}")
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": config["epochs"],
        "condition": condition, "participant": args.participant, "seed": args.seed,
        "config": config, "node_to_tier3": node_to_tier3,
        "initialization": initialization,
        "fallback_equivalence_max_abs_error_before_training": fallback_equivalence_max_abs_error,
        "fallback_equivalence_max_abs_error_after_training": fallback_equivalence_after_training,
    }, output / "last.pth")
    write_json(output / "train_log.json", log)
    for split in ("test_all", "test_normal", "test_fault"):
        loader = DataLoader(
            datasets[split], batch_size=config["batch_size"], shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_multimodal,
            pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
        )
        evaluate(model, loader, device, node_to_tier3, output / "test_results", split)
    write_json(output / "completed.json", {
        "condition": condition, "participant": args.participant, "seed": args.seed,
        "checkpoint": str(output / "last.pth"), "splits": ["test_all", "test_normal", "test_fault"],
        "initialization": initialization,
        "fallback_equivalence_max_abs_error_before_training": fallback_equivalence_max_abs_error,
        "fallback_equivalence_max_abs_error_after_training": fallback_equivalence_after_training,
    })


if __name__ == "__main__":
    main()
