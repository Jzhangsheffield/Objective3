from __future__ import annotations

from pathlib import Path


def outer_protocol(config: dict, participant: str) -> Path:
    return (
        Path(config["m2_project_root"])
        / "outputs"
        / f"{participant}_as_test"
        / f"cam_{config['primary_camera_id']}"
        / "protocols"
        / config["train_scope"]
    )


def primary_fold_root(config: dict, participant: str, seed: int) -> Path:
    return (
        Path(config["m2_project_root"])
        / "outputs"
        / f"{participant}_as_test"
        / f"cam_{config['primary_camera_id']}"
        / f"seed_{seed}"
    )


def a0_checkpoint(config: dict, participant: str, seed: int) -> Path:
    return (
        primary_fold_root(config, participant, seed)
        / "history_models"
        / "direct_head_fusion"
        / config["train_scope"]
        / "m2_direct"
        / "last.pth"
    )


def a0_probabilities(config: dict, participant: str, seed: int, split: str) -> Path:
    return a0_checkpoint(config, participant, seed).parent / "test_results" / f"{split}_probabilities.pt"


def primary_global_cache(config: dict, participant: str, seed: int, split: str) -> Path:
    filename = "train_all.pt" if split == "train" else "test_all.pt"
    return primary_fold_root(config, participant, seed) / "features" / "retrained_all_runs" / filename


def primary_backbone(config: dict, participant: str, seed: int) -> Path:
    return primary_fold_root(config, participant, seed) / "backbone" / config["train_scope"] / "last.pth"


def b0_root(config: dict) -> Path:
    return Path(config["output_root"]) / "B0_phase_a"


def b0_condition_root(config: dict, condition: str, participant: str, seed: int) -> Path:
    return b0_root(config) / condition / f"{participant}_as_test" / f"seed_{seed}"


def b0_secondary_backbone(config: dict, participant: str, seed: int) -> Path:
    return (
        b0_root(config)
        / "upstream"
        / f"{participant}_as_test"
        / f"cam_{config['secondary_camera_id']}"
        / f"seed_{seed}"
        / "backbone"
    )


def b0_secondary_global_cache(config: dict, participant: str, seed: int, split: str) -> Path:
    filename = "train_all.pt" if split == "train" else "test_all.pt"
    return b0_secondary_backbone(config, participant, seed).parent / "features" / filename


def crossfit_root(config: dict, outer: str, inner: str, seed: int) -> Path:
    return Path(config["output_root"]) / "crossfit" / f"outer_{outer}" / f"heldout_{inner}" / f"seed_{seed}"


def crossfit_protocol(config: dict, outer: str, inner: str) -> Path:
    return Path(config["output_root"]) / "crossfit_protocols" / f"outer_{outer}" / f"heldout_{inner}" / "all_runs"


def expert_root(config: dict, scope: str, participant: str, seed: int, expert: str) -> Path:
    return Path(config["output_root"]) / scope / f"{participant}_as_test" / f"seed_{seed}" / expert


def temporal_cache_root(config: dict, participant: str, seed: int) -> Path:
    return Path(config["output_root"]) / "temporal_caches" / f"{participant}_as_test" / f"seed_{seed}"


def experiment_root(config: dict, condition: str, participant: str, seed: int) -> Path:
    return Path(config["output_root"]) / condition / f"{participant}_as_test" / f"seed_{seed}"
