from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.left_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_padding, 0)))


class TimewiseLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel, dilation=dilation)
        self.norm1 = TimewiseLayerNorm(channels)
        self.norm2 = TimewiseLayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        x = self.dropout(F.gelu(self.norm2(self.conv2(x))))
        return x + residual


class LocalWindowEncoder(nn.Module):
    """Encode a past-only sensor window into one feature at its right edge."""

    def __init__(self, input_channels: int, channels: list[int], feature_dim: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        current = input_channels
        for output in channels:
            layers.extend(
                [
                    nn.Conv1d(current, output, kernel_size=5, stride=2, padding=2),
                    nn.GroupNorm(1, output),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current = output
        self.network = nn.Sequential(*layers)
        self.projection = nn.Linear(current, feature_dim)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        batch, length, width, channels = windows.shape
        x = windows.reshape(batch * length, width, channels).transpose(1, 2)
        x = self.network(x).mean(dim=-1)
        return self.projection(x).reshape(batch, length, -1)


class CausalSensorBoundaryModel(nn.Module):
    def __init__(self, cfg: dict[str, Any], mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
        super().__init__()
        input_channels = int(cfg["input_channels"])
        self.register_buffer("normalization_mean", torch.zeros(input_channels) if mean is None else mean.float())
        self.register_buffer("normalization_std", torch.ones(input_channels) if std is None else std.float())
        feature_dim = int(cfg["local_feature_dim"])
        hidden = int(cfg["tcn_hidden_dim"])
        self.local_encoder = LocalWindowEncoder(
            input_channels,
            [int(x) for x in cfg["local_encoder_channels"]],
            feature_dim,
            float(cfg["dropout"]),
        )
        self.input_projection = nn.Conv1d(feature_dim, hidden, 1)
        self.blocks = nn.ModuleList(
            ResidualBlock(hidden, int(cfg["tcn_kernel_size"]), 2**index, float(cfg["dropout"]))
            for index in range(int(cfg["tcn_num_layers"]))
        )
        self.state_head = nn.Conv1d(hidden, 2, 1)
        self.boundary_head = nn.Conv1d(hidden, 2, 1)

    @property
    def receptive_field_steps(self) -> int:
        return 1 + sum(2 * block.conv1.left_padding for block in self.blocks)

    def forward(self, windows: torch.Tensor) -> dict[str, torch.Tensor]:
        if windows.ndim != 4:
            raise ValueError(f"Expected [B,L,W,C], got {tuple(windows.shape)}")
        windows = (windows - self.normalization_mean) / self.normalization_std.clamp_min(1e-6)
        x = self.local_encoder(windows).transpose(1, 2)
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        state = self.state_head(x).transpose(1, 2)
        boundary = self.boundary_head(x).transpose(1, 2)
        return {
            "state_logits": state,
            "start_logits": boundary[..., 0],
            "end_logits": boundary[..., 1],
        }


def compute_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict[str, Any]):
    mask = batch["mask"]
    state_weight = torch.tensor([1.0, float(cfg["state_positive_weight"])], device=mask.device)
    state_loss = F.cross_entropy(outputs["state_logits"][mask], batch["state"][mask], weight=state_weight)
    start_loss = F.binary_cross_entropy_with_logits(
        outputs["start_logits"][mask], batch["start"][mask],
        pos_weight=torch.tensor(float(cfg["start_positive_weight"]), device=mask.device),
    )
    end_loss = F.binary_cross_entropy_with_logits(
        outputs["end_logits"][mask], batch["end"][mask],
        pos_weight=torch.tensor(float(cfg["end_positive_weight"]), device=mask.device),
    )
    weights = cfg["weights"]
    total = float(weights["state"]) * state_loss + float(weights["start"]) * start_loss + float(weights["end"]) * end_loss
    values = {
        "loss": float(total.detach()),
        "state_loss": float(state_loss.detach()),
        "start_loss": float(start_loss.detach()),
        "end_loss": float(end_loss.detach()),
    }
    return total, values
