from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _conv3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv1d:
    return nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = _conv3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value if self.downsample is None else self.downsample(value)
        value = self.relu(self.bn1(self.conv1(value)))
        value = self.bn2(self.conv2(value))
        return self.relu(value + identity)


class IMUResNet10(nn.Module):
    """Right-hand IMU encoder with an eight-token temporal output for length 256."""

    def __init__(self, in_channels: int = 6, base_channels: int = 64, num_nodes: int = 35) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, base_channels, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        self.layer1 = BasicBlock1D(base_channels, base_channels)
        self.layer2 = BasicBlock1D(base_channels, base_channels * 2, 2)
        self.layer3 = BasicBlock1D(base_channels * 2, base_channels * 4, 2)
        self.layer4 = BasicBlock1D(base_channels * 4, base_channels * 8, 2)
        self.feature_dim = base_channels * 8
        self.head = nn.Linear(self.feature_dim, num_nodes)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_tokens(self, signal: torch.Tensor) -> torch.Tensor:
        value = self.maxpool(self.relu(self.bn1(self.conv1(signal))))
        value = self.layer4(self.layer3(self.layer2(self.layer1(value))))
        return value.transpose(1, 2).contiguous()

    def forward_features(self, signal: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(signal).mean(dim=1)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(signal))


class M2HistoryHead(nn.Module):
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
        self.max_history = max_history
        self.current_projection = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))
        self.history_projection = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))
        self.position_embedding = nn.Embedding(max_history + 1, d_model)
        self.null_history = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.null_history, std=0.02)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + d_model, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(feature_dim, num_nodes)

    def forward(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
        history_padding_mask: torch.Tensor,
        history_position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.current_projection(current)
        projected = self.history_projection(history)
        if history.shape[1]:
            projected = projected + self.position_embedding(history_position_ids.clamp(0, self.max_history))
        null = self.null_history.expand(current.shape[0], -1, -1)
        keys = torch.cat([null, projected], dim=1)
        padding = torch.cat([
            torch.zeros((current.shape[0], 1), dtype=torch.bool, device=current.device),
            history_padding_mask,
        ], dim=1)
        context, _ = self.attention(query.unsqueeze(1), keys, keys, key_padding_mask=padding)
        fused = self.fusion(torch.cat([current, context.squeeze(1)], dim=-1))
        return self.head(fused), fused


class CameraContextExpert(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.m2 = M2HistoryHead(**kwargs)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.m2(
            batch["current"], batch["history"], batch["history_padding_mask"],
            batch["history_position_ids"],
        )[0]


class BottleneckBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, bottleneck: torch.Tensor, tokens: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        update, _ = self.attention(
            self.norm_q(bottleneck), self.norm_kv(tokens), self.norm_kv(tokens),
            key_padding_mask=padding,
        )
        bottleneck = bottleneck + update
        return bottleneck + self.ffn(bottleneck)


class SymmetricTokenFusion(nn.Module):
    MODALITIES = ("cam0", "cam1", "imu")

    def __init__(
        self,
        temporal_dims: dict[str, int],
        global_dim: int = 512,
        d_model: int = 256,
        num_heads: int = 4,
        bottleneck_tokens: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        soft_alignment: bool = False,
        output_dim: int = 512,
        num_nodes: int = 35,
    ) -> None:
        super().__init__()
        self.soft_alignment = bool(soft_alignment)
        self.global_projection = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(global_dim, d_model), nn.LayerNorm(d_model))
            for name in self.MODALITIES
        })
        self.temporal_projection = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(int(temporal_dims[name]), d_model), nn.LayerNorm(d_model))
            for name in self.MODALITIES
        })
        self.modality_embedding = nn.Parameter(torch.zeros(len(self.MODALITIES), d_model))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.bottleneck = nn.Parameter(torch.zeros(1, bottleneck_tokens, d_model))
        nn.init.normal_(self.bottleneck, std=0.02)
        self.blocks = nn.ModuleList([BottleneckBlock(d_model, num_heads, dropout) for _ in range(layers)])
        if self.soft_alignment:
            self.cam_to_imu = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            self.imu_to_cam = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            self.align_norm = nn.LayerNorm(d_model)
        self.output = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_dim))
        self.unimodal_heads = nn.ModuleDict({name: nn.Linear(d_model, num_nodes) for name in self.MODALITIES})

    def forward(
        self,
        values: dict[str, dict[str, torch.Tensor]],
        available: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        projected: dict[str, torch.Tensor] = {}
        temporal: dict[str, torch.Tensor] = {}
        for index, name in enumerate(self.MODALITIES):
            global_token = self.global_projection[name](values[name]["global"]).unsqueeze(1)
            temporal[name] = self.temporal_projection[name](values[name]["temporal"])
            projected[name] = torch.cat([global_token, temporal[name]], dim=1) + self.modality_embedding[index]
        alignment = {}
        if self.soft_alignment:
            camera = torch.cat([temporal["cam0"], temporal["cam1"]], dim=1)
            cam_update, cam_weights = self.cam_to_imu(camera, temporal["imu"], temporal["imu"])
            imu_update, imu_weights = self.imu_to_cam(temporal["imu"], camera, camera)
            camera = self.align_norm(camera + cam_update)
            temporal["imu"] = self.align_norm(temporal["imu"] + imu_update)
            cam0_length = temporal["cam0"].shape[1]
            projected["cam0"] = torch.cat([
                projected["cam0"][:, :1], camera[:, :cam0_length] + self.modality_embedding[0]
            ], dim=1)
            projected["cam1"] = torch.cat([
                projected["cam1"][:, :1], camera[:, cam0_length:] + self.modality_embedding[1]
            ], dim=1)
            projected["imu"] = torch.cat([
                projected["imu"][:, :1], temporal["imu"] + self.modality_embedding[2]
            ], dim=1)
            alignment = {"cam_to_imu": cam_weights, "imu_to_cam": imu_weights}
        token_values = torch.cat([projected[name] for name in self.MODALITIES], dim=1)
        token_masks = []
        for index, name in enumerate(self.MODALITIES):
            token_masks.append((~available[:, index]).unsqueeze(1).expand(-1, projected[name].shape[1]))
        padding = torch.cat(token_masks, dim=1)
        bottleneck = self.bottleneck.expand(token_values.shape[0], -1, -1)
        for block in self.blocks:
            bottleneck = block(bottleneck, token_values, padding)
        observation = self.output(bottleneck.mean(dim=1))
        unimodal_logits = {
            name: self.unimodal_heads[name](projected[name].mean(dim=1))
            for name in self.MODALITIES
        }
        camera_embedding = F.normalize(
            0.5 * (projected["cam0"].mean(dim=1) + projected["cam1"].mean(dim=1)), dim=-1
        )
        imu_embedding = F.normalize(projected["imu"].mean(dim=1), dim=-1)
        return observation, {
            "unimodal_logits": unimodal_logits,
            "camera_embedding": camera_embedding,
            "imu_embedding": imu_embedding,
            "alignment": alignment,
        }


class JointFusionModel(nn.Module):
    def __init__(
        self,
        temporal_dims: dict[str, int],
        use_history: bool,
        modality_dropout: float,
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_history = bool(use_history)
        self.modality_dropout = float(modality_dropout)
        output_dim = int(kwargs.pop("output_dim", 512))
        num_nodes = int(kwargs.pop("num_nodes", 35))
        max_history = int(kwargs.pop("max_history", 35))
        d_model = int(kwargs.get("d_model", 256))
        num_heads = int(kwargs.get("num_heads", 4))
        dropout = float(kwargs.get("dropout", 0.1))
        self.observation = SymmetricTokenFusion(
            temporal_dims=temporal_dims, output_dim=output_dim, num_nodes=num_nodes, **kwargs
        )
        if self.use_history:
            self.classifier = M2HistoryHead(
                feature_dim=output_dim,
                d_model=d_model,
                num_heads=num_heads,
                max_history=max_history,
                dropout=dropout,
                num_nodes=num_nodes,
            )
        else:
            self.classifier = nn.Sequential(nn.LayerNorm(output_dim), nn.Linear(output_dim, num_nodes))

    def _availability(self, available: torch.Tensor) -> torch.Tensor:
        available = available.bool()
        if not self.training or self.modality_dropout <= 0:
            return available
        keep = torch.rand_like(available.float()) >= self.modality_dropout
        result = available & keep
        none = ~result.any(dim=1)
        if none.any():
            first = available.float().argmax(dim=1)
            result[none, first[none]] = True
        return result

    @staticmethod
    def _current_values(batch: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
        return {
            name: {
                "global": batch[f"current_{name}_global"],
                "temporal": batch[f"current_{name}_temporal"],
            }
            for name in SymmetricTokenFusion.MODALITIES
        }

    @staticmethod
    def _history_values(batch: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
        return {
            name: {
                "global": batch[f"history_{name}_global"].flatten(0, 1),
                "temporal": batch[f"history_{name}_temporal"].flatten(0, 1),
            }
            for name in SymmetricTokenFusion.MODALITIES
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        available = self._availability(batch["available"])
        current, diagnostics = self.observation(self._current_values(batch), available)
        if not self.use_history:
            return self.classifier(current), diagnostics
        batch_size, history_length = batch["history_cam0_global"].shape[:2]
        if history_length:
            history_available = available.unsqueeze(1).expand(-1, history_length, -1).flatten(0, 1)
            history, _ = self.observation(self._history_values(batch), history_available)
            history = history.view(batch_size, history_length, -1)
        else:
            history = current.new_zeros((batch_size, 0, current.shape[-1]))
        logits, fused = self.classifier(
            current, history, batch["history_padding_mask"], batch["history_position_ids"]
        )
        diagnostics["fused_feature"] = fused
        return logits, diagnostics


def symmetric_contrastive_loss(camera: torch.Tensor, imu: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = camera @ imu.T / max(float(temperature), 1e-6)
    target = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
