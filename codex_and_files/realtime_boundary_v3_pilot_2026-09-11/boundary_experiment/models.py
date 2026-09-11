from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.left_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_padding, 0)))


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = TimewiseLayerNorm(channels)
        self.norm2 = TimewiseLayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        x = self.dropout(F.gelu(self.norm2(self.conv2(x))))
        return x + residual


class TimewiseLayerNorm(nn.Module):
    """Normalize channels independently at each time step, preserving causality."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalBoundaryTCN(nn.Module):
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(feature_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList(
            CausalResidualBlock(hidden_dim, kernel_size, 2**layer, dropout)
            for layer in range(num_layers)
        )
        self.state_head = nn.Conv1d(hidden_dim, 2, 1)
        self.boundary_head = nn.Conv1d(hidden_dim, 2, 1)

    @property
    def receptive_field_steps(self) -> int:
        total = 1
        for block in self.blocks:
            total += 2 * block.conv1.left_padding
        return total

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(f"Expected [B,L,D], got {tuple(features.shape)}")
        x = self.input_projection(features.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        state = self.state_head(x).transpose(1, 2)
        boundary = self.boundary_head(x).transpose(1, 2)
        return {"state_logits": state, "start_logits": boundary[..., 0], "end_logits": boundary[..., 1]}


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
    positive_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["mask"]
    state_loss = F.cross_entropy(outputs["state_logits"][mask], batch["state"][mask])
    start_loss = F.binary_cross_entropy_with_logits(
        outputs["start_logits"][mask], batch["start"][mask],
        pos_weight=torch.tensor(positive_weights["start"], device=mask.device),
    )
    end_loss = F.binary_cross_entropy_with_logits(
        outputs["end_logits"][mask], batch["end"][mask],
        pos_weight=torch.tensor(positive_weights["end"], device=mask.device),
    )
    total = weights["state"] * state_loss + weights["start"] * start_loss + weights["end"] * end_loss
    return total, {"loss": float(total.detach()), "state_loss": float(state_loss.detach()), "start_loss": float(start_loss.detach()), "end_loss": float(end_loss.detach())}
