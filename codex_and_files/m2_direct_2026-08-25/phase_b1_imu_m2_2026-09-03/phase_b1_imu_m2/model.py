from __future__ import annotations

import torch
from torch import nn


class IMUM2Direct(nn.Module):
    """Camera-M2-compatible direct actual-history head over frozen IMU features."""

    def __init__(
        self,
        feature_dim: int = 512,
        d_model: int = 256,
        num_heads: int = 4,
        max_history: int = 35,
        dropout: float = 0.1,
        num_nodes: int = 35,
    ) -> None:
        super().__init__()
        self.max_history = int(max_history)
        self.current_projection = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))
        self.history_projection = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))
        self.position_embedding = nn.Embedding(self.max_history + 1, d_model)
        self.null_history = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.null_history, std=0.02)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.fusion = nn.Linear(feature_dim + d_model, feature_dim)
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.bias.zero_()
            self.fusion.weight[:, :feature_dim].copy_(torch.eye(feature_dim))
        self.node_norm = nn.LayerNorm(feature_dim)
        self.node_head = nn.Linear(feature_dim, num_nodes)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        current = batch["current_feature"]
        history = self.history_projection(batch["history_features"])
        if history.shape[1]:
            positions = batch["history_position_ids"].clamp(0, self.max_history)
            history = history + self.position_embedding(positions)
        query = self.current_projection(current)
        null = self.null_history.expand(current.shape[0], -1, -1)
        keys = torch.cat([null, history], dim=1)
        padding = torch.cat([
            torch.zeros((current.shape[0], 1), dtype=torch.bool, device=current.device),
            batch["history_padding_mask"],
        ], dim=1)
        context, attention = self.attention(
            query.unsqueeze(1), keys, keys, key_padding_mask=padding,
            need_weights=True, average_attn_weights=False,
        )
        fused = self.fusion(torch.cat([current, context.squeeze(1)], dim=-1))
        logits = self.node_head(self.node_norm(fused))
        return logits, {
            "observation": current,
            "history_context": context.squeeze(1),
            "history_attention": attention,
            "fused_feature": fused,
        }
