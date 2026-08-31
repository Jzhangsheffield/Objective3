from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import torch
import torch.nn.functional as F

from phase_b.calibration import (
    QualityGate, apply_quality_gate, apply_static_fusion, fit_static_fusion,
    probabilities_to_logits, quality_features,
)
from phase_b.config import load_config
from phase_b.evaluation import align_probability_files, write_probability_evaluation
from phase_b.io import read_jsonl, seed_everything, write_json
from phase_b.metrics import derive_node_to_tier3
from phase_b.paths import (
    a0_probabilities, b0_condition_root, crossfit_root, experiment_root,
    expert_root, outer_protocol,
)


def inner_files(config: dict, outer: str, inner: str, seed: int) -> list[Path]:
    root = crossfit_root(config, outer, inner, seed)
    return [
        root / "cam0_m2" / "all_runs" / "m2_direct" / "test_results" / "test_all_probabilities.pt",
        root / "cam1_m2" / "all_runs" / "m2_direct" / "test_results" / "test_all_probabilities.pt",
        root / "imu_direct_node" / "test_results" / "test_all_probabilities.pt",
    ]


def outer_files(config: dict, outer: str, seed: int, split: str) -> list[Path]:
    return [
        a0_probabilities(config, outer, seed, split),
        b0_condition_root(config, "A1", outer, seed) / "test_results" / f"{split}_probabilities.pt",
        expert_root(config, "outer_experts", outer, seed, "imu_direct_node")
        / "test_results" / f"{split}_probabilities.pt",
    ]


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prerequisite probability files:\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit B1 and B2 only on strict inner-LOSO OOF predictions")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "phase_b.json"))
    parser.add_argument("--participant", required=True, choices=list("ADJM"))
    parser.add_argument("--seed", required=True, type=int, choices=[1, 2, 42])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(args.seed)
    inners = [value for value in config["participants"] if value != args.participant]
    all_rows, all_probabilities = [], []
    for inner in inners:
        files = inner_files(config, args.participant, inner, args.seed)
        require_files(files)
        rows, values = align_probability_files(files)
        all_rows.extend(rows)
        all_probabilities.append(torch.stack(values, dim=1).float())
    probabilities = torch.cat(all_probabilities)
    targets = torch.tensor([
        int(row.get("true_node_idx", row.get("node_idx"))) - 1 for row in all_rows
    ], dtype=torch.long)
    if len({str(row["sample_name"]) for row in all_rows}) != len(all_rows):
        raise RuntimeError("Duplicate samples in cross-fit OOF predictions")
    b1_config = config["b1"]
    static = fit_static_fusion(
        probabilities, targets, steps=int(b1_config["steps"]),
        learning_rate=float(b1_config["learning_rate"]),
        uniform_l2=float(b1_config["weight_l2_to_uniform"]),
        epsilon=float(b1_config["probability_epsilon"]),
    )
    b1_root = experiment_root(config, "B1", args.participant, args.seed)
    b2_root = experiment_root(config, "B2", args.participant, args.seed)
    for root in (b1_root, b2_root):
        if root.exists() and any(root.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty output: {root}")
        root.mkdir(parents=True, exist_ok=True)
    torch.save(static, b1_root / "fusion_parameters.pt")
    write_json(b1_root / "fit_summary.json", {
        "training": "strict_inner_loso_out_of_fold", "outer_participant": args.participant,
        "inner_participants": inners, "oof_samples": len(all_rows), "seed": args.seed,
        "temperatures": static["log_temperature"].exp().tolist(),
        "weights": F.softmax(static["weight_logits"], dim=0).tolist(), "history": static["history"],
    })

    b2_config = config["b2"]
    gate = QualityGate(hidden_dim=int(b2_config["hidden_dim"]), dropout=float(b2_config["dropout"]))
    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=float(b2_config["learning_rate"]),
        weight_decay=float(b2_config["weight_decay"]),
    )
    availability = torch.ones(probabilities.shape[:2], dtype=torch.bool)
    calibrated = probabilities_to_logits(probabilities, float(b1_config["probability_epsilon"])) \
        / static["log_temperature"].exp().view(1, -1, 1)
    features = quality_features(probabilities, availability)
    generator = torch.Generator().manual_seed(args.seed)
    max_steps = int(b2_config["max_optimizer_steps"])
    batch_size = int(b2_config["batch_size"])
    gate_log = []
    optimizer_step = 0
    epoch = 0
    while optimizer_step < max_steps:
        epoch += 1
        gate.train()
        permutation = torch.randperm(len(targets), generator=generator)
        for start in range(0, len(targets), batch_size):
            index = permutation[start:start + batch_size]
            weights = gate(features[index], availability[index])
            fused = (weights.unsqueeze(-1) * calibrated[index]).sum(dim=1)
            entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=1).mean()
            cross_entropy = F.cross_entropy(fused, targets[index])
            loss = cross_entropy - float(b2_config["gate_entropy_weight"]) * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_step += 1
            if optimizer_step == 1 or optimizer_step % 50 == 0 or optimizer_step == max_steps:
                gate_log.append({
                    "optimizer_step": optimizer_step, "data_epoch": epoch,
                    "loss": float(loss.detach()), "cross_entropy": float(cross_entropy.detach()),
                    "mean_gate_entropy": float(entropy.detach()),
                })
            if optimizer_step >= max_steps:
                break
    torch.save({
        "model": gate.state_dict(), "log_temperature": static["log_temperature"],
        "seed": args.seed, "input_dim": 15, "hidden_dim": int(b2_config["hidden_dim"]),
    }, b2_root / "fusion_parameters.pt")
    gate.eval()
    with torch.no_grad():
        _, oof_weights = apply_quality_gate(gate, probabilities, static["log_temperature"], availability)
    write_json(b2_root / "fit_summary.json", {
        "training": "strict_inner_loso_out_of_fold", "outer_participant": args.participant,
        "inner_participants": inners, "oof_samples": len(all_rows), "seed": args.seed,
        "max_optimizer_steps": max_steps,
        "temperatures": static["log_temperature"].exp().tolist(),
        "mean_dynamic_weights": oof_weights.mean(dim=0).tolist(), "history": gate_log,
    })

    mapping = derive_node_to_tier3(read_jsonl(outer_protocol(config, args.participant) / "train.jsonl"))
    for split in config["evaluation"]["splits"]:
        files = outer_files(config, args.participant, args.seed, split)
        require_files(files)
        rows, values = align_probability_files(files)
        outer_probabilities = torch.stack(values, dim=1).float()
        b1_probability = apply_static_fusion(
            outer_probabilities, static["log_temperature"], static["weight_logits"],
            float(b1_config["probability_epsilon"]),
        )
        with torch.no_grad():
            b2_probability, weights = apply_quality_gate(
                gate, outer_probabilities, static["log_temperature"], epsilon=float(b1_config["probability_epsilon"])
            )
        write_probability_evaluation(rows, b1_probability, mapping, b1_root / "test_results", split)
        write_probability_evaluation(rows, b2_probability, mapping, b2_root / "test_results", split)
        torch.save({"weights": weights, "rows": rows}, b2_root / "test_results" / f"{split}_gate_weights.pt")
    write_json(b1_root / "completed.json", {"condition": "B1", "seed": args.seed})
    write_json(b2_root / "completed.json", {"condition": "B2", "seed": args.seed})


if __name__ == "__main__":
    main()
