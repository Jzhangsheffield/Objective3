from __future__ import annotations

from pathlib import Path


def protocol_dir(config: dict, participant: str) -> Path:
    return (Path(config["m2_project_root"]) / "outputs" / f"{participant}_as_test" /
            f"cam_{config['primary_camera_id']}" / "protocols" / config["train_scope"])


def primary_feature_cache(config: dict, participant: str, seed: int, split: str) -> Path:
    filename = "train_all.pt" if split == "train" else "test_all.pt"
    return (Path(config["m2_project_root"]) / "outputs" / f"{participant}_as_test" /
            f"cam_{config['primary_camera_id']}" / f"seed_{seed}" / "features" /
            f"retrained_{config['train_scope']}" / filename)


def secondary_feature_cache(config: dict, participant: str, seed: int, split: str) -> Path:
    filename = "train_all.pt" if split == "train" else "test_all.pt"
    return (Path(config["output_root"]) / "upstream" / f"{participant}_as_test" /
            f"cam_{config['secondary_camera_id']}" / f"seed_{seed}" / "features" / filename)


def secondary_backbone_dir(config: dict, participant: str, seed: int) -> Path:
    return (Path(config["output_root"]) / "upstream" / f"{participant}_as_test" /
            f"cam_{config['secondary_camera_id']}" / f"seed_{seed}" / "backbone")


def signal_cache(config: dict, participant: str, split: str) -> Path:
    return Path(config["output_root"]) / "signal_cache" / f"{participant}_as_test" / f"{split}_right_signals.pt"


def model_dir(config: dict, condition: str, participant: str, seed: int) -> Path:
    return Path(config["output_root"]) / condition / f"{participant}_as_test" / f"seed_{seed}"


def a0_result_dir(config: dict, participant: str, seed: int) -> Path:
    return (Path(config["m2_project_root"]) / "outputs" / f"{participant}_as_test" /
            f"cam_{config['primary_camera_id']}" / f"seed_{seed}" / "history_models" /
            "direct_head_fusion" / config["train_scope"] / "m2_direct" / "test_results")


def a0_checkpoint(config: dict, participant: str, seed: int) -> Path:
    return a0_result_dir(config, participant, seed).parent / "last.pth"
