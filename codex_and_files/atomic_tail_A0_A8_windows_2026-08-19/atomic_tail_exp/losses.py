from __future__ import annotations

import torch
import torch.nn.functional as F


def aggregate_node_probabilities(
    node_probabilities: torch.Tensor, node_to_tier3: torch.Tensor, num_tier3: int
) -> torch.Tensor:
    result = node_probabilities.new_zeros((node_probabilities.shape[0], int(num_tier3)))
    return result.scatter_add(1, node_to_tier3.unsqueeze(0).expand(node_probabilities.shape[0], -1), node_probabilities)


def tier3_nll(
    logits: torch.Tensor, targets: torch.Tensor, node_to_tier3: torch.Tensor, num_tier3: int
) -> torch.Tensor:
    probabilities = aggregate_node_probabilities(F.softmax(logits, dim=-1), node_to_tier3, num_tier3)
    selected = probabilities.gather(1, targets[:, None]).squeeze(1).clamp_min(1e-12)
    return -selected.log().mean()


def symmetric_kl_consistency(
    actual_logits: torch.Tensor,
    augmented_logits: torch.Tensor,
    confidence_threshold: float,
) -> tuple[torch.Tensor, int]:
    actual_probability = F.softmax(actual_logits.detach(), dim=-1)
    augmented_probability = F.softmax(augmented_logits, dim=-1)
    mask = actual_probability.max(dim=-1).values >= float(confidence_threshold)
    if not bool(mask.any()):
        return augmented_logits.new_tensor(0.0), 0
    p = actual_probability[mask].clamp_min(1e-8)
    q = augmented_probability[mask].clamp_min(1e-8)
    loss = 0.5 * (
        (p * (p.log() - q.log())).sum(dim=-1).mean()
        + (q * (q.log() - p.log())).sum(dim=-1).mean()
    )
    return loss, int(mask.sum())


def tail_order_loss(
    valid_logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    eligible: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if not bool(eligible.any()):
        return valid_logits.new_tensor(0.0), 0
    valid = valid_logits[eligible]
    corrupted = corrupted_logits[eligible]
    logits = torch.cat([valid, corrupted])
    targets = torch.cat([torch.ones_like(valid), torch.zeros_like(corrupted)])
    return F.binary_cross_entropy_with_logits(logits, targets), int(eligible.sum())
