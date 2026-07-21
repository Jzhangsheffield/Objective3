from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from graph_history.graph import TaskGraphSpec
from graph_history.models import FeatureNodeClassifier, build_context_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic forward passes for M0-M6")
    parser.add_argument("--task-graph", required=True)
    parser.add_argument("--relation-matrix", required=True)
    args = parser.parse_args()
    graph = TaskGraphSpec.load(args.task_graph, args.relation_matrix)
    batch_size, history_length, feature_dim = 3, 8, 512
    current = torch.randn(batch_size, feature_dim)
    history = torch.randn(batch_size, history_length, feature_dim)
    positions = torch.arange(history_length, 0, -1).repeat(batch_size, 1)
    nodes = torch.randint(0, 35, (batch_size, history_length))
    mask = torch.zeros((batch_size, history_length), dtype=torch.bool)
    mask[0, -2:] = True
    baseline = FeatureNodeClassifier(feature_dim)
    assert baseline(current).shape == (batch_size, 35)
    for model_name in ("m1", "m2", "m3", "m4", "m5", "m6"):
        baseline = FeatureNodeClassifier(feature_dim)
        model = build_context_model(
            model_name, baseline, graph.relation_ids, feature_dim, 256, 4, 35, 0.1
        )
        logits, aux = model(
            current_feature=current,
            history_features=history,
            history_position_ids=positions,
            history_node_classes=nodes,
            history_padding_mask=mask,
        )
        assert logits.shape == (batch_size, 35)
        loss = F.cross_entropy(logits, torch.tensor([0, 1, 2]))
        loss.backward()
        trainable_gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert trainable_gradients and all(torch.isfinite(grad).all() for grad in trainable_gradients)

        # The first action in every run has no history; the null-history path must work.
        empty_logits, _ = model(
            current_feature=current,
            history_features=history[:, :0],
            history_position_ids=positions[:, :0],
            history_node_classes=nodes[:, :0],
            history_padding_mask=mask[:, :0],
        )
        assert empty_logits.shape == (batch_size, 35)
        print(model_name, tuple(logits.shape), sorted(aux), "backward=ok empty_history=ok")
    print("Synthetic forward/backward model smoke test passed.")


if __name__ == "__main__":
    main()
