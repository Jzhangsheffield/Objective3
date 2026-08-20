from __future__ import annotations

import torch
import torch.nn as nn


class DirectHistoryClassifier(nn.Module):
    """Independent implementation of the M2-Direct feature-history classifier."""

    def __init__(
        self,
        feature_dim: int = 512,
        num_nodes: int = 35,
        d_model: int = 256,
        num_heads: int = 4,
        max_history: int = 35,
        dropout: float = 0.1,
        shift_embedding_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.max_history = int(max_history)
        self.current_projection = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))
        self.history_projection = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))
        self.position_embedding = nn.Embedding(max_history + 1, d_model)
        self.zero_shift_index = self.max_history - 1
        self.shift_embedding = nn.Embedding(
            2 * self.max_history - 1,
            d_model,
            padding_idx=self.zero_shift_index,
        )
        nn.init.normal_(self.shift_embedding.weight, std=float(shift_embedding_init_std))
        with torch.no_grad():
            self.shift_embedding.weight[self.zero_shift_index].zero_()
        self.null_history = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.null_history, std=0.02)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.fusion = nn.Linear(feature_dim + d_model, feature_dim)
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.bias.zero_()
            self.fusion.weight[:, :feature_dim].copy_(torch.eye(feature_dim))
        self.node_classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_nodes),
        )
        self.tail_order_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(32, d_model // 4)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, d_model // 4), 1),
        )

    def forward(
        self,
        current_feature: torch.Tensor,
        history_features: torch.Tensor,
        history_position_ids: torch.Tensor,
        history_padding_mask: torch.Tensor,
        history_shift_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        current = self.current_projection(current_feature)
        history = self.history_projection(history_features)
        if history.shape[1] > 0:
            history = history + self.position_embedding(
                history_position_ids.clamp(min=0, max=self.max_history)
            )
            if history_shift_ids is not None:
                maximum_shift = self.max_history - 1
                shift_indices = (
                    history_shift_ids.clamp(min=-maximum_shift, max=maximum_shift)
                    + self.zero_shift_index
                )
                history = history + self.shift_embedding(shift_indices)
        null = self.null_history.expand(current_feature.shape[0], -1, -1)
        keys = torch.cat([null, history], dim=1)
        null_mask = torch.zeros((current_feature.shape[0], 1), dtype=torch.bool, device=current_feature.device)
        padding = torch.cat([null_mask, history_padding_mask], dim=1)
        context, attention = self.attention(
            current.unsqueeze(1), keys, keys, key_padding_mask=padding,
            need_weights=True, average_attn_weights=False,
        )
        context = context.squeeze(1)
        fused = self.fusion(torch.cat([current_feature, context], dim=-1))
        logits = self.node_classifier(fused)
        return logits, {"history_context": context, "fused_feature": fused, "attention": attention}

    def tail_order_logits(self, history_context: torch.Tensor) -> torch.Tensor:
        return self.tail_order_head(history_context).squeeze(-1)


def build_model(config: dict) -> DirectHistoryClassifier:
    return DirectHistoryClassifier(
        feature_dim=int(config["feature_dim"]),
        num_nodes=int(config["num_nodes"]),
        d_model=int(config["d_model"]),
        num_heads=int(config["num_heads"]),
        max_history=int(config["max_history"]),
        dropout=float(config["dropout"]),
        shift_embedding_init_std=float(config.get("shift_embedding_init_std", 0.02)),
    )
