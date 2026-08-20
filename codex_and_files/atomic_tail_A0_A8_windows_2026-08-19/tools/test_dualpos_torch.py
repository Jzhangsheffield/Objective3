from __future__ import annotations

import sys
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from atomic_tail_exp.model import DirectHistoryClassifier


def main() -> int:
    torch.manual_seed(7)
    model = DirectHistoryClassifier(
        feature_dim=16,
        num_nodes=5,
        d_model=8,
        num_heads=2,
        max_history=4,
        dropout=0.0,
        shift_embedding_init_std=0.02,
    ).eval()
    current = torch.randn(2, 16)
    history = torch.randn(2, 4, 16)
    true_positions = torch.tensor([[4, 3, 2, 1], [4, 3, 2, 1]])
    zero_shifts = torch.zeros_like(true_positions)
    padding = torch.zeros((2, 4), dtype=torch.bool)

    with torch.no_grad():
        logits_without_shift, extra_without_shift = model(
            current, history, true_positions, padding
        )
        logits_zero_shift, extra_zero_shift = model(
            current, history, true_positions, padding, zero_shifts
        )
    assert torch.equal(logits_without_shift, logits_zero_shift)
    assert torch.equal(extra_without_shift["history_context"], extra_zero_shift["history_context"])

    permutation = torch.tensor([1, 3, 0, 2])
    permuted_history = history[:, permutation]
    permuted_true = true_positions[:, permutation]
    presented = torch.tensor([[4, 3, 2, 1], [4, 3, 2, 1]])
    displacement = presented - permuted_true
    with torch.no_grad():
        _, true_only_extra = model(
            current, permuted_history, permuted_true, padding, torch.zeros_like(displacement)
        )
        _, dualpos_extra = model(
            current, permuted_history, permuted_true, padding, displacement
        )
    assert torch.allclose(
        extra_zero_shift["history_context"], true_only_extra["history_context"], atol=1e-6, rtol=1e-6
    )
    assert not torch.allclose(
        true_only_extra["history_context"], dualpos_extra["history_context"], atol=1e-7, rtol=1e-7
    )
    assert torch.count_nonzero(model.shift_embedding.weight[model.zero_shift_index]) == 0
    print("DualPos torch tests passed: zero-shift compatibility, true-only invariance, displacement visibility.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
