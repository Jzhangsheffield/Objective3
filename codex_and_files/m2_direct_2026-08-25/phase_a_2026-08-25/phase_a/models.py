from __future__ import annotations

import torch
from torch import nn


class TemporalSignalEncoder(nn.Module):
    """Small dilated Conv1d encoder for one already normalized sensor clip."""

    def __init__(self, channels: int, output_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        widths = (64, 128, output_dim)
        blocks: list[nn.Module] = []
        previous = channels
        for index, width in enumerate(widths):
            blocks.extend(
                [
                    nn.Conv1d(previous, width, kernel_size=7, stride=2, padding=3),
                    nn.GroupNorm(8, width),
                    nn.GELU(),
                    nn.Conv1d(
                        width,
                        width,
                        kernel_size=5,
                        padding=2 * (2**index),
                        dilation=2**index,
                    ),
                    nn.GroupNorm(8, width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = width
        self.network = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.pool(self.network(signal)).squeeze(-1)


class GatedResidualBranch(nn.Module):
    """Add one modality as a bounded residual; zero initialization preserves the anchor."""

    def __init__(self, anchor_dim: int, context_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.context = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, anchor_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta = nn.Linear(anchor_dim, anchor_dim)
        self.gate = nn.Sequential(
            nn.Linear(anchor_dim * 2 + 1, anchor_dim),
            nn.GELU(),
            nn.Linear(anchor_dim, anchor_dim),
        )
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -3.0)

    def forward(
        self,
        anchor: torch.Tensor,
        context: torch.Tensor,
        available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.context(context)
        quality = available.to(anchor.dtype).unsqueeze(-1)
        gate = torch.sigmoid(self.gate(torch.cat([anchor, projected, quality], dim=-1)))
        gate = gate * quality
        return anchor + gate * self.delta(projected), gate


class CrossViewContext(nn.Module):
    """Primary-query cross-attention over primary and secondary view tokens."""

    def __init__(self, feature_dim: int = 512, d_model: int = 256, heads: int = 4) -> None:
        super().__init__()
        self.primary = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))
        self.secondary = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))
        self.view_embedding = nn.Parameter(torch.zeros(2, d_model))
        nn.init.normal_(self.view_embedding, std=0.02)
        self.attention = nn.MultiheadAttention(d_model, heads, batch_first=True)

    def forward(
        self, primary: torch.Tensor, secondary: torch.Tensor, secondary_available: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        primary_token = self.primary(primary) + self.view_embedding[0]
        secondary_token = self.secondary(secondary) + self.view_embedding[1]
        tokens = torch.stack([primary_token, secondary_token], dim=1)
        # Primary is always valid; the secondary token is masked when absent.
        padding = torch.stack(
            [torch.zeros_like(secondary_available, dtype=torch.bool), ~secondary_available.bool()], dim=1
        )
        context, weights = self.attention(
            primary_token.unsqueeze(1), tokens, tokens, key_padding_mask=padding,
            need_weights=True, average_attn_weights=False,
        )
        return context.squeeze(1), weights


class FeatureNodeClassifier(nn.Module):
    """State-dict-compatible copy of the existing M2-Direct 35-node head."""

    def __init__(self, feature_dim: int = 512, num_nodes: int = 35) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(feature_dim, num_nodes)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.norm(value)))


class MultimodalObservationEncoder(nn.Module):
    """Create the 512-D current/history observation consumed by the unchanged M2 block."""

    def __init__(
        self,
        condition: str,
        feature_dim: int = 512,
        d_model: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
        modality_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.condition = condition.upper()
        self.modality_dropout = float(modality_dropout)
        self.use_secondary = self.condition in {"A3", "A7"}
        self.use_emg = self.condition in {"A5", "A6", "A7"}
        self.use_imu = self.condition in {"A4", "A6", "A7"}
        self.secondary_only = self.condition == "A1"
        if self.condition not in {"A1", "A3", "A4", "A5", "A6", "A7"}:
            raise ValueError(f"This encoder trains only A1/A3-A7, received {condition}")

        if self.use_secondary:
            self.cross_view = CrossViewContext(feature_dim, d_model, heads)
            self.view_residual = GatedResidualBranch(feature_dim, d_model, dropout)
        if self.use_emg:
            self.emg_encoder = TemporalSignalEncoder(8, d_model, dropout)
            self.emg_residual = GatedResidualBranch(feature_dim, d_model, dropout)
        if self.use_imu:
            self.imu_encoder = TemporalSignalEncoder(6, d_model, dropout)
            self.imu_residual = GatedResidualBranch(feature_dim, d_model, dropout)

    def _training_dropout(self, available: torch.Tensor) -> torch.Tensor:
        if not self.training or self.modality_dropout <= 0:
            return available.bool()
        keep = torch.rand(available.shape, device=available.device) >= self.modality_dropout
        return available.bool() & keep

    def forward(
        self,
        primary: torch.Tensor,
        secondary: torch.Tensor | None = None,
        emg: torch.Tensor | None = None,
        imu: torch.Tensor | None = None,
        secondary_available: torch.Tensor | None = None,
        emg_available: torch.Tensor | None = None,
        imu_available: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.secondary_only:
            if secondary is None:
                raise ValueError("A1 requires secondary camera features")
            return secondary, {}
        fused = primary
        diagnostics: dict[str, torch.Tensor] = {}
        if self.use_secondary:
            if secondary is None:
                raise ValueError(f"{self.condition} requires secondary camera features")
            available = self._training_dropout(secondary_available)
            context, attention = self.cross_view(primary, secondary, available)
            fused, gate = self.view_residual(fused, context, available)
            diagnostics.update(view_gate=gate, cross_view_attention=attention)
        if self.use_emg:
            if emg is None:
                raise ValueError(f"{self.condition} requires right-hand EMG")
            available = self._training_dropout(emg_available)
            context = self.emg_encoder(emg)
            fused, gate = self.emg_residual(fused, context, available)
            diagnostics["emg_gate"] = gate
        if self.use_imu:
            if imu is None:
                raise ValueError(f"{self.condition} requires right-hand IMU")
            available = self._training_dropout(imu_available)
            context = self.imu_encoder(imu)
            fused, gate = self.imu_residual(fused, context, available)
            diagnostics["imu_gate"] = gate
        return fused, diagnostics


class PhaseAM2Direct(nn.Module):
    """Multimodal observation fusion followed by the exact M2-Direct history flow."""

    def __init__(
        self,
        condition: str,
        feature_dim: int = 512,
        d_model: int = 256,
        num_heads: int = 4,
        max_history: int = 35,
        dropout: float = 0.1,
        modality_dropout: float = 0.2,
        num_nodes: int = 35,
    ) -> None:
        super().__init__()
        self.condition = condition.upper()
        self.feature_dim = feature_dim
        self.max_history = max_history
        self.observation = MultimodalObservationEncoder(
            condition, feature_dim, d_model, num_heads, dropout, modality_dropout
        )
        self.current_projection = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))
        self.history_projection = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))
        self.position_embedding = nn.Embedding(max_history + 1, d_model)
        self.null_history = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.null_history, std=0.02)
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.fusion = nn.Linear(feature_dim + d_model, feature_dim)
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.bias.zero_()
            self.fusion.weight[:, :feature_dim].copy_(torch.eye(feature_dim))
        self.node_classifier = FeatureNodeClassifier(feature_dim, num_nodes)

    def freeze_m2_core(self) -> None:
        for name, parameter in self.named_parameters():
            if not name.startswith("observation."):
                parameter.requires_grad = False

    def _encode_observations(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict]:
        current, current_diag = self.observation(
            batch["current_primary"], batch.get("current_secondary"),
            batch.get("current_emg"), batch.get("current_imu"),
            batch.get("current_secondary_available"), batch.get("current_emg_available"),
            batch.get("current_imu_available"),
        )
        batch_size, history_length = batch["history_primary"].shape[:2]
        if history_length == 0:
            history = current.new_zeros((batch_size, 0, self.feature_dim))
            return current, history, {"current": current_diag, "history": {}}
        flat = lambda name: batch[name].flatten(0, 1) if name in batch else None
        history, history_diag = self.observation(
            flat("history_primary"), flat("history_secondary"), flat("history_emg"),
            flat("history_imu"), flat("history_secondary_available"),
            flat("history_emg_available"), flat("history_imu_available"),
        )
        return current, history.view(batch_size, history_length, -1), {
            "current": current_diag, "history": history_diag
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, object]]:
        current, history_features, modality_diagnostics = self._encode_observations(batch)
        query = self.current_projection(current)
        history = self.history_projection(history_features)
        if history.shape[1]:
            positions = batch["history_position_ids"].clamp(0, self.max_history)
            history = history + self.position_embedding(positions)
        null = self.null_history.expand(current.shape[0], -1, -1)
        keys = torch.cat([null, history], dim=1)
        null_mask = torch.zeros((current.shape[0], 1), dtype=torch.bool, device=current.device)
        padding = torch.cat([null_mask, batch["history_padding_mask"]], dim=1)
        context, attention = self.attention(
            query.unsqueeze(1), keys, keys, key_padding_mask=padding,
            need_weights=True, average_attn_weights=False,
        )
        fused = self.fusion(torch.cat([current, context.squeeze(1)], dim=-1))
        return self.node_classifier(fused), {
            "observation": current,
            "fused_feature": fused,
            "history_context": context.squeeze(1),
            "history_attention": attention,
            "modalities": modality_diagnostics,
        }
