from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import torch
import torch.nn.functional as F

from graph_history.models import build_direct_context_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run synthetic forward/backward checks for direct-head M1-M3"
    )
    parser.parse_args()
    batch_size, history_length, feature_dim = 3, 8, 512
    current = torch.randn(batch_size, feature_dim)
    history = torch.randn(batch_size, history_length, feature_dim)
    positions = torch.arange(history_length, 0, -1).repeat(batch_size, 1)
    nodes = torch.randint(0, 35, (batch_size, history_length))
    mask = torch.zeros((batch_size, history_length), dtype=torch.bool)
    mask[0, -2:] = True

    for model_name in ("m1_direct", "m2_direct", "m3_direct"):
        model = build_direct_context_model(model_name, feature_dim, 256, 4, 35, 0.1)
        logits, aux = model(
            current_feature=current,
            history_features=history,
            history_position_ids=positions,
            history_node_classes=nodes,
            history_padding_mask=mask,
        )
        assert logits.shape == (batch_size, 35)
        # Identity initialization makes the first fused representation exactly
        # equal to the frozen Tier-3 feature before learning history effects.
        assert torch.allclose(aux["fused_feature"], current, atol=1e-6, rtol=1e-6)
        loss = F.cross_entropy(logits, torch.tensor([0, 1, 2]))
        loss.backward()
        classifier_gradients = [
            parameter.grad
            for parameter in model.node_classifier.parameters()
            if parameter.grad is not None
        ]
        fusion_gradients = [
            parameter.grad for parameter in model.fusion.parameters() if parameter.grad is not None
        ]
        assert classifier_gradients and fusion_gradients
        assert all(
            torch.isfinite(gradient).all()
            for gradient in classifier_gradients + fusion_gradients
        )

        empty_logits, empty_aux = model(
            current_feature=current,
            history_features=history[:, :0],
            history_position_ids=positions[:, :0],
            history_node_classes=nodes[:, :0],
            history_padding_mask=mask[:, :0],
        )
        assert empty_logits.shape == (batch_size, 35)
        assert empty_aux["attention"].shape[-1] == 1
        print(
            model_name,
            tuple(logits.shape),
            sorted(aux),
            "identity_init=ok backward=ok empty_history=ok",
        )
    print("Direct-head synthetic forward/backward smoke test passed.")


if __name__ == "__main__":
    main()
