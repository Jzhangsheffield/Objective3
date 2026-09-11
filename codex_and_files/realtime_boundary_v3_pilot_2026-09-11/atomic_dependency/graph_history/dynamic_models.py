from __future__ import annotations

import torch
import torch.nn as nn

from .constants import NUM_GRAPH_NODES
from .models import FeatureNodeClassifier


class JointHeadDeltaHistoryModel(nn.Module):
    """Train a new current-only node head jointly with a history logit delta."""

    def __init__(
        self,
        feature_dim: int = 512,
        d_model: int = 256,
        num_heads: int = 4,
        max_history: int = 35,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_history = int(max_history)
        self.node_classifier = FeatureNodeClassifier(
            feature_dim=feature_dim,
            num_nodes=NUM_GRAPH_NODES,
            dropout=0.0,
        )
        self.current_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.history_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.position_embedding = nn.Embedding(max_history + 1, d_model)
        self.null_history = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.null_history, std=0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_GRAPH_NODES),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        self.history_scale_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        current_feature: torch.Tensor,
        history_features: torch.Tensor,
        history_position_ids: torch.Tensor,
        history_padding_mask: torch.Tensor,
        **_: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        baseline_logits = self.node_classifier(current_feature)
        current = self.current_projection(current_feature)
        history = self.history_projection(history_features)
        if history.shape[1] > 0:
            positions = history_position_ids.clamp(min=0, max=self.max_history)
            history = history + self.position_embedding(positions)

        null = self.null_history.expand(current.shape[0], -1, -1)
        history = torch.cat([null, history], dim=1)
        null_mask = torch.zeros(
            (current.shape[0], 1),
            dtype=torch.bool,
            device=current.device,
        )
        key_padding_mask = torch.cat([null_mask, history_padding_mask], dim=1)
        context, attention_weights = self.attention(
            current.unsqueeze(1),
            history,
            history,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        context = context.squeeze(1)
        delta = self.delta_head(torch.cat([current, context], dim=-1))
        scale = torch.sigmoid(self.history_scale_logit)
        logits = baseline_logits + scale * delta
        return logits, {
            "baseline_logits": baseline_logits,
            "history_delta": delta,
            "history_scale": scale.detach(),
            "attention": attention_weights,
        }


def build_joint_head_delta_model(
    feature_dim: int,
    d_model: int,
    num_heads: int,
    max_history: int,
    dropout: float,
) -> JointHeadDeltaHistoryModel:
    return JointHeadDeltaHistoryModel(
        feature_dim=feature_dim,
        d_model=d_model,
        num_heads=num_heads,
        max_history=max_history,
        dropout=dropout,
    )
