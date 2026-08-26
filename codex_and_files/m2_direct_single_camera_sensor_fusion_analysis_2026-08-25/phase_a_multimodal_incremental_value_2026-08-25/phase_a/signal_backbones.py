from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from .models import TemporalSignalEncoder


def _conv3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv1d:
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv1d:
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock1D(nn.Module):
    """The BasicBlock used by the Objective-2 torchvision-style ResNet1D."""

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm1d,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value
        value = self.relu(self.bn1(self.conv1(value)))
        value = self.bn2(self.conv2(value))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(value + identity)


class ResNet10SignalBackbone(nn.Module):
    """Exact ResNet10-1D topology used by the Objective-2 signal scripts."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        stem_kernel_size: int = 7,
        stem_stride: int = 2,
        use_stem_pool: bool = True,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()
        self.inplanes = int(base_channels)
        self.conv1 = nn.Conv1d(
            in_channels, base_channels, kernel_size=stem_kernel_size, stride=stem_stride,
            padding=stem_kernel_size // 2, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1) if use_stem_pool else nn.Identity()
        self.layer1 = self._make_layer(base_channels, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = base_channels * 8
        self._initialize(zero_init_residual)

    def _make_layer(self, planes: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(_conv1(self.inplanes, planes, stride), nn.BatchNorm1d(planes))
        block = BasicBlock1D(self.inplanes, planes, stride, downsample)
        self.inplanes = planes
        return nn.Sequential(block)

    def _initialize(self, zero_init_residual: bool) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicBlock1D):
                    nn.init.zeros_(module.bn2.weight)

    def forward_features(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3:
            raise ValueError(f"ResNet10SignalBackbone expects [B,C,L], got {tuple(signal.shape)}")
        value = self.maxpool(self.relu(self.bn1(self.conv1(signal))))
        value = self.layer4(self.layer3(self.layer2(self.layer1(value))))
        return self.avgpool(value).flatten(1)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.forward_features(signal)


class DilatedSignalBackbone(nn.Module):
    """The package's existing dilated encoder followed by a trained 256->512 projection."""

    def __init__(self, in_channels: int, output_dim: int = 256, feature_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = TemporalSignalEncoder(in_channels, output_dim, dropout)
        self.projection = nn.Sequential(nn.Linear(output_dim, feature_dim), nn.LayerNorm(feature_dim))
        self.feature_dim = feature_dim

    def forward_features(self, signal: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encoder(signal))

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.forward_features(signal)


def build_signal_backbone(config: dict, encoder: str, in_channels: int) -> nn.Module:
    if encoder == "resnet10_1d":
        settings = config["resnet10_1d"]
        model = ResNet10SignalBackbone(
            in_channels=in_channels,
            base_channels=int(settings["base_channels"]),
            stem_kernel_size=int(settings["stem_kernel_size"]),
            stem_stride=int(settings["stem_stride"]),
            use_stem_pool=bool(settings["use_stem_pool"]),
            zero_init_residual=bool(settings["zero_init_residual"]),
        )
    elif encoder == "dilated_conv1d":
        settings = config["dilated_conv1d"]
        model = DilatedSignalBackbone(
            in_channels=in_channels,
            output_dim=int(settings["encoder_output_dim"]),
            feature_dim=int(settings["feature_dim"]),
            dropout=float(settings["dropout"]),
        )
    else:
        raise ValueError(f"Unsupported signal encoder: {encoder}")
    if int(model.feature_dim) != int(config["signal_feature_dim"]):
        raise ValueError(f"Feature dimension mismatch: {model.feature_dim} != {config['signal_feature_dim']}")
    return model
