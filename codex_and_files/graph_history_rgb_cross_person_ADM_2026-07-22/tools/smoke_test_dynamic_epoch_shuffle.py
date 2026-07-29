from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from graph_history.data import FeatureHistoryDataset, collate_history_batch
from graph_history.dynamic_data import (
    EpochGraphValidHistoryDataset,
    stable_epoch_sample_seed,
)
from graph_history.dynamic_engine import train_epoch_shuffled_feature_model
from graph_history.dynamic_models import build_joint_head_delta_model
from graph_history.models import (
    FeatureNodeClassifier,
    build_context_model,
    build_direct_context_model,
)
from graph_history.utils import write_jsonl


def build_fixture(root: Path) -> tuple[Path, Path, SimpleNamespace]:
    records = []
    rows = []
    for index in range(6):
        sample_name = f"sample_{index + 1}"
        records.append({"sample_name": sample_name})
        rows.append(
            {
                "sample_name": sample_name,
                "participant": "P",
                "run": "run_1",
                "annotation_row_index": index,
                "node_idx": index + 1,
                "tier3_id": index,
                "stage_id": 1,
            }
        )
    cache_path = root / "features.pt"
    torch.save(
        {
            "features": torch.randn(6, 512),
            "tier3_logits": torch.randn(6, 31),
            "records": records,
            "metadata": {"fixture": True},
        },
        cache_path,
    )
    manifest_path = root / "train.jsonl"
    write_jsonl(manifest_path, rows)
    graph = SimpleNamespace(
        atomic_sequences=(),
        all_must_previous={
            node: ((1,) if node == 3 else ())
            for node in range(1, 36)
        },
        node_to_tier3=torch.arange(35, dtype=torch.long) % 31,
    )
    return cache_path, manifest_path, graph


def assert_dynamic_dataset() -> None:
    with tempfile.TemporaryDirectory(prefix="dynamic_graph_shuffle_") as temp:
        cache_path, manifest_path, graph = build_fixture(Path(temp))
        dataset = EpochGraphValidHistoryDataset(
            cache_path,
            manifest_path,
            graph=graph,
            shuffle_seed=42,
        )
        last_index = len(dataset) - 1
        orders: list[tuple[str, ...]] = []
        for epoch in range(1, 13):
            dataset.set_epoch(epoch)
            first = tuple(dataset[last_index]["history_sample_names"])
            second = tuple(dataset[last_index]["history_sample_names"])
            assert first == second
            assert first.index("sample_1") < first.index("sample_3")
            orders.append(first)
        assert len(set(orders)) > 1

        replica = EpochGraphValidHistoryDataset(
            cache_path,
            manifest_path,
            graph=graph,
            shuffle_seed=42,
        )
        replica.set_epoch(7)
        assert tuple(replica[last_index]["history_sample_names"]) == orders[6]
        assert stable_epoch_sample_seed(42, 1, "sample_6") != stable_epoch_sample_seed(
            42, 2, "sample_6"
        )

        static_dataset = FeatureHistoryDataset(
            cache_path,
            manifest_path,
            history_order="graph_valid",
            graph=graph,
            shuffle_seed=42,
        )
        static_first = tuple(static_dataset[last_index]["history_sample_names"])
        static_second = tuple(static_dataset[last_index]["history_sample_names"])
        assert static_first == static_second

        audit = dataset.audit_epochs(12)
        assert audit["samples_with_multiple_orders"] >= 1
        assert audit["epochs_audited"] == 12
        print(
            "dynamic_dataset=ok",
            f"unique_orders={len(set(orders))}",
            "static_eval=ok",
            "audit=ok",
        )


def assert_epoch_training_engine() -> None:
    with tempfile.TemporaryDirectory(prefix="dynamic_graph_engine_") as temp:
        cache_path, manifest_path, graph = build_fixture(Path(temp))
        dataset = EpochGraphValidHistoryDataset(
            cache_path,
            manifest_path,
            graph=graph,
            shuffle_seed=7,
        )
        loader = DataLoader(
            dataset,
            batch_size=3,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
            collate_fn=collate_history_batch,
        )
        model = build_joint_head_delta_model(
            feature_dim=512,
            d_model=32,
            num_heads=4,
            max_history=35,
            dropout=0.0,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_log = train_epoch_shuffled_feature_model(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=torch.device("cpu"),
            node_to_tier3=graph.node_to_tier3,
            epochs=2,
            action_loss_weight=0.0,
            amp=False,
        )
        assert dataset.epoch == 2
        assert [int(row["history_shuffle_epoch"]) for row in train_log] == [1, 2]
        assert all(torch.isfinite(torch.tensor(row["train_loss"])) for row in train_log)
        print("dynamic_training_engine=ok epoch_state=ok")


def assert_models() -> None:
    batch_size, history_length, feature_dim = 3, 5, 512
    current = torch.randn(batch_size, feature_dim)
    history = torch.randn(batch_size, history_length, feature_dim)
    positions = torch.arange(history_length, 0, -1).repeat(batch_size, 1)
    nodes = torch.randint(0, 35, (batch_size, history_length))
    mask = torch.zeros((batch_size, history_length), dtype=torch.bool)
    mask[0, -1] = True
    target = torch.tensor([0, 1, 2])

    baseline = FeatureNodeClassifier(feature_dim=feature_dim)
    frozen_model = build_context_model(
        "m3",
        baseline=baseline,
        relation_ids=torch.zeros((35, 35), dtype=torch.long),
        feature_dim=feature_dim,
        d_model=256,
        num_heads=4,
        max_history=35,
        dropout=0.0,
    )
    joint_model = build_joint_head_delta_model(
        feature_dim=feature_dim,
        d_model=256,
        num_heads=4,
        max_history=35,
        dropout=0.0,
    )
    direct_model = build_direct_context_model(
        "m3_direct",
        feature_dim=feature_dim,
        d_model=256,
        num_heads=4,
        max_history=35,
        dropout=0.0,
    )
    models = {
        "m3_dynamic_frozen_m0_delta": frozen_model,
        "m3_dynamic_joint_head_delta": joint_model,
        "m3_dynamic_direct_fusion": direct_model,
    }
    for model_name, model in models.items():
        logits, aux = model(
            current_feature=current,
            history_features=history,
            history_position_ids=positions,
            history_node_classes=nodes,
            history_padding_mask=mask,
        )
        assert logits.shape == (batch_size, 35)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        if model_name == "m3_dynamic_frozen_m0_delta":
            assert all(
                not parameter.requires_grad
                for parameter in model.baseline.parameters()
            )
        if model_name == "m3_dynamic_joint_head_delta":
            assert any(
                parameter.grad is not None
                for parameter in model.node_classifier.parameters()
            )
        if model_name == "m3_dynamic_direct_fusion":
            assert "history_delta" not in aux
            assert torch.allclose(
                aux["fused_feature"],
                current,
                atol=1e-6,
                rtol=1e-6,
            )
        print(model_name, "forward=ok backward=ok")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic checks for dynamic graph-valid epoch-shuffle experiments"
    )
    parser.parse_args()
    assert_dynamic_dataset()
    assert_epoch_training_engine()
    assert_models()
    print("Dynamic epoch-shuffle smoke test passed.")


if __name__ == "__main__":
    main()
