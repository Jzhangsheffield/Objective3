from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from phase_b.config import load_config, validate_experiment
from phase_b.data import TokenHistoryDataset, collate_token_history
from phase_b.evaluation import write_probability_evaluation
from phase_b.io import read_jsonl, seed_everything, write_json
from phase_b.metrics import derive_node_to_tier3
from phase_b.models import JointFusionModel, symmetric_contrastive_loss
from phase_b.paths import experiment_root, outer_protocol, temporal_cache_root
from phase_b.training import move_batch, select_device


def cache_paths(root: Path, split: str) -> dict[str, Path]:
    return {name: root / f"{name}_{split}.pt" for name in ("cam0", "cam1", "imu")}


@torch.no_grad()
def evaluate(model, loader, device, mapping, output, split) -> None:
    model.eval()
    probabilities, rows = [], []
    for batch in loader:
        rows.extend(batch["rows"])
        moved = move_batch(batch, device)
        logits, _ = model(moved)
        probabilities.append(F.softmax(logits, dim=-1).cpu())
    write_probability_evaluation(rows, torch.cat(probabilities), mapping, output, split)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train B3/B4/B5 symmetric joint three-modal fusion")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_b.json"))
    parser.add_argument("--condition", required=True, choices=["B3", "B4", "B5"])
    parser.add_argument("--participant", required=True, choices=list("ADJM"))
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    condition = validate_experiment(args.condition)
    seed_everything(args.seed)
    spec = config[condition.lower()]
    settings = config["joint_fusion"]
    output = experiment_root(config, condition, args.participant, args.seed)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    protocols = outer_protocol(config, args.participant)
    temporal_root = temporal_cache_root(config, args.participant, args.seed)
    for path in cache_paths(temporal_root, "train").values():
        if not path.is_file():
            raise FileNotFoundError(path)
    train_dataset = TokenHistoryDataset(
        cache_paths(temporal_root, "train"), protocols / "train.jsonl", bool(spec["use_history"])
    )
    device = select_device(args.device)
    loader = DataLoader(
        train_dataset, batch_size=int(settings["batch_size"]), shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_token_history,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    model = JointFusionModel(
        temporal_dims={"cam0": 128, "cam1": 128, "imu": 512},
        use_history=bool(spec["use_history"]),
        modality_dropout=float(settings["modality_dropout_probability"]),
        global_dim=int(config["feature_dim"]), d_model=int(config["d_model"]),
        num_heads=int(config["num_heads"]), bottleneck_tokens=int(settings["bottleneck_tokens"]),
        layers=int(settings["fusion_layers"]), dropout=float(config["dropout"]),
        soft_alignment=bool(spec["use_soft_temporal_alignment"]), output_dim=int(config["feature_dim"]),
        num_nodes=int(config["num_nodes"]), max_history=int(config["max_history"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"])
    )
    if int(settings["effective_batch_size"]) != int(settings["batch_size"]):
        raise ValueError("B3-B5 now require direct batch training: effective_batch_size must equal batch_size")
    if int(settings.get("gradient_accumulation_steps", 1)) != 1:
        raise ValueError("B3-B5 gradient_accumulation_steps must be 1")
    log = []
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        totals = {"loss": 0.0, "node": 0.0, "aux": 0.0, "contrastive": 0.0, "samples": 0}
        for batch in loader:
            moved = move_batch(batch, device)
            logits, diagnostics = model(moved)
            target = moved["node_target"]
            node_loss = F.cross_entropy(logits, target)
            aux_loss = torch.stack([
                F.cross_entropy(value, target) for value in diagnostics["unimodal_logits"].values()
            ]).mean()
            contrastive = logits.new_zeros(())
            if float(spec["contrastive_loss_weight"]) > 0:
                contrastive = symmetric_contrastive_loss(
                    diagnostics["camera_embedding"], diagnostics["imu_embedding"],
                    float(spec["contrastive_temperature"]),
                )
            loss = node_loss + float(settings["auxiliary_unimodal_loss_weight"]) * aux_loss \
                + float(spec["contrastive_loss_weight"]) * contrastive
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip_norm"]))
            optimizer.step()
            count = target.numel()
            for key, value in (("loss", loss), ("node", node_loss), ("aux", aux_loss), ("contrastive", contrastive)):
                totals[key] += float(value.detach()) * count
            totals["samples"] += count
        row = {"epoch": epoch, **{key: totals[key] / totals["samples"] for key in ("loss", "node", "aux", "contrastive")}}
        log.append(row)
        print(f"epoch={epoch}/{settings['epochs']} loss={row['loss']:.6f}", flush=True)
    checkpoint = output / "last.pth"
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": int(settings["epochs"]),
        "condition": condition, "participant": args.participant, "seed": args.seed,
        "model_config": {"condition_spec": spec, "joint_fusion": settings},
    }, checkpoint)
    write_json(output / "train_log.json", log)
    mapping = derive_node_to_tier3(read_jsonl(protocols / "train.jsonl"))
    for split in config["evaluation"]["splits"]:
        dataset = TokenHistoryDataset(
            cache_paths(temporal_root, "test"), protocols / f"{split}.jsonl", bool(spec["use_history"])
        )
        test_loader = DataLoader(
            dataset, batch_size=int(settings["batch_size"]), shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_token_history,
            pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
        )
        evaluate(model, test_loader, device, mapping, output / "test_results", split)
    write_json(output / "completed.json", {
        "condition": condition, "checkpoint": str(checkpoint), "seed": args.seed,
        "participant": args.participant, "splits": config["evaluation"]["splits"],
    })


if __name__ == "__main__":
    main()
