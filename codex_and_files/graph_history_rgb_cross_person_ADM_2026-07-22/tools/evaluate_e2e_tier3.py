from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from graph_history.backbone import generate_model
from graph_history.constants import DEFAULT_CAMERA_ID, NUM_TIER3_CLASSES
from graph_history.data import RGBClipDataset
from graph_history.utils import (
    ensure_new_output_dir,
    load_compatible_state,
    seed_everything,
    select_device,
    write_json,
)
from graph_history.video_evaluation import evaluate_tier3_video_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluation-only retest of an existing per-fold E2E Tier-3 last.pth"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--protocol-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-scope", default="normal_only", choices=["normal_only", "all_runs"])
    parser.add_argument("--camera-id", default=DEFAULT_CAMERA_ID)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--splits", nargs="+", default=["test_normal", "test_fault", "test_all"])
    parser.add_argument("--max-test-samples", type=int, default=-1, help="Debug only")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = select_device(args.device)
    protocol_root = Path(args.protocol_root).resolve()
    output_dir = ensure_new_output_dir(args.output_dir, overwrite=args.overwrite)
    model = generate_model(18, num_classes=NUM_TIER3_CLASSES).to(device)
    load_report = load_compatible_state(model, args.checkpoint)
    if load_report["missing_keys"] or load_report["unexpected_keys"]:
        raise RuntimeError(f"Existing Tier-3 checkpoint is not fully compatible: {load_report}")
    write_json(
        output_dir / "evaluation_config.json",
        {
            **vars(args),
            "model": "e2e_tier3_scratch",
            "mode": "evaluation_only_existing_checkpoint",
            "checkpoint_load_report": load_report,
        },
    )
    test_root = output_dir / "test_results"
    for split_name in args.splits:
        manifest = protocol_root / args.train_scope / f"{split_name}.jsonl"
        dataset = RGBClipDataset(args.dataset_root, manifest, args.camera_id, train=False)
        if args.max_test_samples > 0:
            dataset.rows = dataset.rows[: args.max_test_samples]
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        metrics = evaluate_tier3_video_model(
            model,
            loader,
            device,
            test_root,
            split_name,
            "e2e_tier3_scratch",
            args.checkpoint,
            amp=args.amp,
        )
        print(
            f"{split_name}: tier3_acc={metrics['tier3']['accuracy']:.4f} "
            f"tier3_macro_f1={metrics['tier3']['macro_f1']:.4f}",
            flush=True,
        )
    write_json(
        output_dir / "completed.json",
        {
            "model": "e2e_tier3_scratch",
            "checkpoint": str(args.checkpoint),
            "splits": args.splits,
        },
    )
    print(f"Saved evaluation-only Tier-3 results: {output_dir}")


if __name__ == "__main__":
    main()
