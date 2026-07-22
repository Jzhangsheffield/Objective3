from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import NUM_GRAPH_NODES, RELATION_TO_ID


class FeatureNodeClassifier(nn.Module):
    """M0: current frozen RGB feature -> 35 graph-node logits."""

    def __init__(self, feature_dim: int = 512, num_nodes: int = NUM_GRAPH_NODES, dropout: float = 0.0):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_nodes = int(num_nodes)
        self.norm = nn.LayerNorm(self.feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.feature_dim, self.num_nodes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.norm(features)))


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


class SingleQueryHistoryModel(nn.Module):
    """M1-M3: one current query attends to the same-run causal history."""

    def __init__(
        self,
        baseline: FeatureNodeClassifier,
        feature_dim: int = 512,
        d_model: int = 256,
        num_heads: int = 4,
        max_history: int = 35,
        dropout: float = 0.1,
        use_position: bool = True,
    ) -> None:
        super().__init__()
        self.baseline = baseline
        freeze_module(self.baseline)
        self.use_position = bool(use_position)
        self.max_history = int(max_history)
        self.current_projection = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))
        self.history_projection = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))
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

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline.eval()
        return self

    def forward(
        self,
        current_feature: torch.Tensor,
        history_features: torch.Tensor,
        history_position_ids: torch.Tensor,
        history_padding_mask: torch.Tensor,
        **_: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        current = self.current_projection(current_feature)
        history = self.history_projection(history_features)
        if self.use_position and history.shape[1] > 0:
            positions = history_position_ids.clamp(min=0, max=self.max_history)
            history = history + self.position_embedding(positions)

        null = self.null_history.expand(current.shape[0], -1, -1)
        history = torch.cat([null, history], dim=1)
        null_mask = torch.zeros((current.shape[0], 1), dtype=torch.bool, device=current.device)
        key_padding_mask = torch.cat([null_mask, history_padding_mask], dim=1)
        context, attention_weights = self.attention(
            current.unsqueeze(1), history, history,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        context = context.squeeze(1)
        delta = self.delta_head(torch.cat([current, context], dim=-1))
        scale = torch.sigmoid(self.history_scale_logit)
        with torch.no_grad():
            baseline_logits = self.baseline(current_feature)
        logits = baseline_logits + scale * delta
        return logits, {
            "baseline_logits": baseline_logits,
            "history_delta": delta,
            "history_scale": scale.detach(),
            "attention": attention_weights,
        }


class CandidateHistoryModel(nn.Module):
    """M4-M6: 35 candidate queries with optional task-graph relation bias."""

    def __init__(
        self,
        baseline: FeatureNodeClassifier,
        relation_ids: torch.Tensor,
        graph_source: str,
        feature_dim: int = 512,
        d_model: int = 256,
        num_heads: int = 4,
        max_history: int = 35,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if graph_source not in {"none", "oracle", "predicted"}:
            raise ValueError(f"Unsupported graph_source: {graph_source}")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.baseline = baseline
        freeze_module(self.baseline)
        self.graph_source = graph_source
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.max_history = int(max_history)
        self.current_projection = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))
        self.history_projection = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))
        self.position_embedding = nn.Embedding(max_history + 1, d_model)
        self.candidate_embedding = nn.Embedding(NUM_GRAPH_NODES, d_model)
        self.null_history = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.null_history, std=0.02)
        nn.init.normal_(self.candidate_embedding.weight, std=0.02)

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.register_buffer("relation_ids", relation_ids.long().clone(), persistent=True)

        initial = torch.tensor([0.2, 0.1, 0.0, -0.2, -0.1], dtype=torch.float32)
        self.relation_bias = nn.Parameter(initial.repeat(num_heads, 1))
        self.immediate_not_last_bias = nn.Parameter(torch.full((num_heads,), -0.2))

        self.delta_head = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        self.history_scale_logit = nn.Parameter(torch.tensor(-2.0))

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline.eval()
        return self

    def _history_node_probabilities(
        self,
        history_features: torch.Tensor,
        history_node_classes: torch.Tensor,
        history_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, history_length, _ = history_features.shape
        if history_length == 0:
            return history_features.new_zeros((batch_size, 0, NUM_GRAPH_NODES))
        if self.graph_source == "oracle":
            safe_classes = history_node_classes.clamp(min=0)
            probabilities = F.one_hot(safe_classes, num_classes=NUM_GRAPH_NODES).float()
        elif self.graph_source == "predicted":
            with torch.no_grad():
                probabilities = F.softmax(self.baseline(history_features), dim=-1)
        else:
            probabilities = history_features.new_zeros(
                (batch_size, history_length, NUM_GRAPH_NODES)
            )
        return probabilities.masked_fill(history_padding_mask.unsqueeze(-1), 0.0)

    def _graph_bias(
        self,
        history_probabilities: torch.Tensor,
        history_position_ids: torch.Tensor,
        history_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, history_length, _ = history_probabilities.shape
        if self.graph_source == "none" or history_length == 0:
            return history_probabilities.new_zeros(
                (batch_size, self.num_heads, NUM_GRAPH_NODES, history_length)
            )
        # [H,V,U]: learned scalar for every head and fixed candidate/history relation.
        pair_bias = self.relation_bias[:, self.relation_ids]
        graph_bias = torch.einsum("blu,hvu->bhvl", history_probabilities, pair_bias)

        # I is fully meaningful only when the observed history token is last (distance=1).
        immediate_matrix = (self.relation_ids == RELATION_TO_ID["I"]).to(history_probabilities.dtype)
        immediate_probability = torch.einsum(
            "blu,vu->bvl", history_probabilities, immediate_matrix
        )
        not_last = (
            (history_position_ids != 1) & (~history_padding_mask)
        ).to(history_probabilities.dtype)
        graph_bias = graph_bias + (
            self.immediate_not_last_bias.view(1, self.num_heads, 1, 1)
            * immediate_probability.unsqueeze(1)
            * not_last.unsqueeze(1).unsqueeze(1)
        )
        return graph_bias

    def forward(
        self,
        current_feature: torch.Tensor,
        history_features: torch.Tensor,
        history_position_ids: torch.Tensor,
        history_node_classes: torch.Tensor,
        history_padding_mask: torch.Tensor,
        **_: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = current_feature.shape[0]
        history_length = history_features.shape[1]
        current = self.current_projection(current_feature)
        history = self.history_projection(history_features)
        if history_length:
            positions = history_position_ids.clamp(min=0, max=self.max_history)
            history = history + self.position_embedding(positions)

        candidates = self.candidate_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        queries = self.query_projection(current.unsqueeze(1) + candidates)
        null = self.null_history.expand(batch_size, -1, -1)
        history_with_null = torch.cat([null, history], dim=1)
        keys = self.key_projection(history_with_null)
        values = self.value_projection(history_with_null)

        queries = queries.view(batch_size, NUM_GRAPH_NODES, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, history_length + 1, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, history_length + 1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.einsum("bhnc,bhlc->bhnl", queries, keys) / math.sqrt(self.head_dim)

        history_probabilities = self._history_node_probabilities(
            history_features, history_node_classes, history_padding_mask
        )
        graph_bias = self._graph_bias(
            history_probabilities, history_position_ids, history_padding_mask
        )
        null_bias = graph_bias.new_zeros((batch_size, self.num_heads, NUM_GRAPH_NODES, 1))
        scores = scores + torch.cat([null_bias, graph_bias], dim=-1)
        null_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=current_feature.device)
        key_padding_mask = torch.cat([null_mask, history_padding_mask], dim=1)
        scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attention = self.attention_dropout(F.softmax(scores, dim=-1))
        context = torch.einsum("bhnl,bhlc->bhnc", attention, values)
        context = context.transpose(1, 2).contiguous().view(batch_size, NUM_GRAPH_NODES, self.d_model)
        context = self.output_projection(context)

        current_expanded = current.unsqueeze(1).expand(-1, NUM_GRAPH_NODES, -1)
        delta_input = torch.cat([current_expanded, context, candidates], dim=-1)
        delta = self.delta_head(delta_input).squeeze(-1)
        scale = torch.sigmoid(self.history_scale_logit)
        with torch.no_grad():
            baseline_logits = self.baseline(current_feature)
        logits = baseline_logits + scale * delta
        return logits, {
            "baseline_logits": baseline_logits,
            "history_delta": delta,
            "history_scale": scale.detach(),
            "attention": attention,
            "graph_bias": graph_bias,
            "history_node_probabilities": history_probabilities,
        }


def build_context_model(
    model_name: str,
    baseline: FeatureNodeClassifier,
    relation_ids: torch.Tensor,
    feature_dim: int,
    d_model: int,
    num_heads: int,
    max_history: int,
    dropout: float,
) -> nn.Module:
    if model_name == "m1":
        return SingleQueryHistoryModel(
            baseline, feature_dim, d_model, num_heads, max_history, dropout, use_position=False
        )
    if model_name in {"m2", "m3"}:
        return SingleQueryHistoryModel(
            baseline, feature_dim, d_model, num_heads, max_history, dropout, use_position=True
        )
    graph_sources = {"m4": "none", "m5": "oracle", "m6": "predicted"}
    if model_name in graph_sources:
        return CandidateHistoryModel(
            baseline=baseline,
            relation_ids=relation_ids,
            graph_source=graph_sources[model_name],
            feature_dim=feature_dim,
            d_model=d_model,
            num_heads=num_heads,
            max_history=max_history,
            dropout=dropout,
        )
    raise ValueError(f"Not a context model: {model_name}")

