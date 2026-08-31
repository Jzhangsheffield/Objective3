from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F
from torch import nn


def probabilities_to_logits(probabilities: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    return probabilities.clamp_min(float(epsilon)).log()


def calibrated_logits(probabilities: torch.Tensor, log_temperature: torch.Tensor, epsilon: float) -> torch.Tensor:
    return probabilities_to_logits(probabilities, epsilon) / log_temperature.exp().clamp(0.05, 20.0)


def fit_static_fusion(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    steps: int = 1500,
    learning_rate: float = 0.03,
    uniform_l2: float = 0.001,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor | list[float]]:
    """Fit three temperatures and simplex weights on strictly OOF predictions."""
    if probabilities.ndim != 3:
        raise ValueError("probabilities must be [N,M,C]")
    modalities = probabilities.shape[1]
    log_temperature = nn.Parameter(torch.zeros(modalities, dtype=probabilities.dtype))
    weight_logits = nn.Parameter(torch.zeros(modalities, dtype=probabilities.dtype))
    optimizer = torch.optim.Adam([log_temperature, weight_logits], lr=learning_rate)
    history = []
    uniform = torch.full((modalities,), 1.0 / modalities, dtype=probabilities.dtype)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = probabilities_to_logits(probabilities, epsilon) / log_temperature.exp().view(1, -1, 1)
        weights = F.softmax(weight_logits, dim=0)
        fused = (weights.view(1, -1, 1) * logits).sum(dim=1)
        nll = F.cross_entropy(fused, targets)
        loss = nll + uniform_l2 * (weights - uniform).square().sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_temperature.clamp_(math.log(0.05), math.log(20.0))
        if step == 1 or step % 50 == 0 or step == steps:
            history.append({
                "step": step,
                "loss": float(loss.detach()),
                "nll": float(nll.detach()),
                "temperatures": log_temperature.detach().exp().tolist(),
                "weights": F.softmax(weight_logits.detach(), dim=0).tolist(),
            })
    return {
        "log_temperature": log_temperature.detach(),
        "weight_logits": weight_logits.detach(),
        "history": history,
    }


def apply_static_fusion(
    probabilities: torch.Tensor,
    log_temperature: torch.Tensor,
    weight_logits: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    logits = probabilities_to_logits(probabilities, epsilon) / log_temperature.exp().view(1, -1, 1)
    weights = F.softmax(weight_logits, dim=0)
    return F.softmax((weights.view(1, -1, 1) * logits).sum(dim=1), dim=-1)


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()).sum(dim=-1)


def quality_features(probabilities: torch.Tensor, availability: torch.Tensor | None = None) -> torch.Tensor:
    """Return [entropy,max,margin] per expert + three pairwise JS + availability."""
    if probabilities.ndim != 3 or probabilities.shape[1] != 3:
        raise ValueError("B2 expects [N,3,C] probabilities")
    top2 = probabilities.topk(2, dim=-1).values
    features = []
    classes = probabilities.shape[-1]
    for index in range(3):
        value = probabilities[:, index]
        features.extend([
            _entropy(value).unsqueeze(1) / math.log(classes),
            top2[:, index, 0].unsqueeze(1),
            (top2[:, index, 0] - top2[:, index, 1]).unsqueeze(1),
        ])
    for first, second in itertools.combinations(range(3), 2):
        left, right = probabilities[:, first], probabilities[:, second]
        middle = 0.5 * (left + right)
        js = 0.5 * (
            (left * (left.clamp_min(1e-8).log() - middle.clamp_min(1e-8).log())).sum(-1)
            + (right * (right.clamp_min(1e-8).log() - middle.clamp_min(1e-8).log())).sum(-1)
        )
        features.append(js.unsqueeze(1))
    if availability is None:
        availability = torch.ones((probabilities.shape[0], 3), dtype=probabilities.dtype)
    features.append(availability.to(probabilities.dtype))
    return torch.cat(features, dim=1)


class QualityGate(nn.Module):
    def __init__(self, input_dim: int = 15, hidden_dim: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        logits = self.network(features).masked_fill(~availability.bool(), -1e4)
        return F.softmax(logits, dim=-1)


def apply_quality_gate(
    gate: QualityGate,
    probabilities: torch.Tensor,
    log_temperature: torch.Tensor,
    availability: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if availability is None:
        availability = torch.ones(probabilities.shape[:2], dtype=torch.bool, device=probabilities.device)
    features = quality_features(probabilities, availability)
    weights = gate(features, availability)
    logits = probabilities_to_logits(probabilities, epsilon) / log_temperature.exp().view(1, -1, 1)
    return F.softmax((weights.unsqueeze(-1) * logits).sum(dim=1), dim=-1), weights
