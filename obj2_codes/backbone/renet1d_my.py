#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
resnet1d.py

标准 torchvision 风格的 1D ResNet，用于时序信号（如 EMG / IMU）建模。

设计目标
========
1. 输入统一为 [B, C, L]
2. 最终分类头统一命名为 self.fc
   - 便于后续 MoCo 脚本把 fc 替换成 projection head
   - 便于 finetune / test 脚本按 model.fc 做 head_only 冻结
3. 提供 forward_features()
   - 返回池化后的 backbone 特征 [B, feature_dim]
   - 便于后续特征提取 / prototype / 可视化 / 消融实验
4. 不兼容旧版参数接口
   - 不再支持 base_filters / n_block / downsample_gap / increasefilter_gap
   - 统一使用 block + layers + base_channels 的新接口

典型用法
========
1) 直接构建标准 ResNet-18 1D:
    model = resnet18_1d(in_channels=28, num_classes=17, base_channels=64)

2) 自定义更浅网络:
    model = ResNet1D(
        block=BasicBlock1D,
        layers=[1, 1, 1, 1],
        in_channels=28,
        num_classes=17,
        base_channels=32,
    )

3) 提取 backbone 特征:
    feats = model.forward_features(x)   # [B, feature_dim]

4) 普通分类:
    logits = model(x)                   # [B, num_classes]
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """
    1D 版本的 3x3 卷积，对应 2D ResNet 里的 conv3x3。
    使用 padding=1 以在 stride=1 时保持时序长度不变。
    """
    return nn.Conv1d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """
    1D 版本的 1x1 卷积。
    主要用于 projection shortcut（维度匹配）和 bottleneck 的通道变换。
    """
    return nn.Conv1d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock1D(nn.Module):
    """
    标准 1D ResNet BasicBlock。

    结构:
        x
        -> conv1 -> bn1 -> relu
        -> conv2 -> bn2
        -> add(shortcut)
        -> relu
    """
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.conv1 = conv3x1(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x1(planes, planes, stride=1)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class Bottleneck1D(nn.Module):
    """
    标准 1D ResNet Bottleneck。

    结构:
        x
        -> 1x1 -> bn -> relu
        -> 3x1 -> bn -> relu
        -> 1x1 -> bn
        -> add(shortcut)
        -> relu
    """
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.conv1 = conv1x1(inplanes, planes, stride=1)
        self.bn1 = norm_layer(planes)

        self.conv2 = conv3x1(planes, planes, stride=stride)
        self.bn2 = norm_layer(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion, stride=1)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """
    torchvision 风格的 1D ResNet。

    参数
    ----
    block:
        残差块类型，通常为 BasicBlock1D 或 Bottleneck1D。

    layers:
        4 个 stage 的 block 数量，例如:
        - [2,2,2,2] -> ResNet-18 风格
        - [1,1,1,1] -> 更浅的版本

    in_channels:
        输入通道数，例如：
        - 双手 EMG: 16
        - 双手 IMU: 12
        - 双手 EMG+IMU: 28

    num_classes:
        最终分类输出维度。
        在普通分类任务里它就是类别数；
        在 MoCo 里，这个位置会被用作 projection 维度，或者 fc 会进一步被替换为 MLP head。

    base_channels:
        stem 和第一 stage 的基础通道数。

    stem_kernel_size:
        stem 的卷积核大小，默认 7。

    stem_stride:
        stem 卷积步长，默认 2。

    use_stem_pool:
        是否在 stem 后使用 MaxPool1d(kernel=3, stride=2, padding=1)。
        默认 True，与标准 ResNet 接近。

    norm_layer:
        归一化层类型，默认 BatchNorm1d。

    zero_init_residual:
        是否将每个残差分支最后一个 BN 的权重初始化为 0。
        对训练稳定性有时有帮助。
    """

    def __init__(
        self,
        block: Type[nn.Module],
        layers: Sequence[int],
        in_channels: int,
        num_classes: int,
        base_channels: int = 64,
        stem_kernel_size: int = 7,
        stem_stride: int = 2,
        use_stem_pool: bool = True,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        zero_init_residual: bool = False,
        l2_normalize_before_fc: bool = False,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        if len(layers) != 4:
            raise ValueError(
                f"'layers' must contain 4 integers for 4 stages, got {layers}"
            )
        if any(int(x) <= 0 for x in layers):
            raise ValueError(f"All entries in 'layers' must be positive, got {layers}")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}")
        if stem_kernel_size <= 0 or stem_kernel_size % 2 == 0:
            raise ValueError(
                f"stem_kernel_size should be a positive odd integer, got {stem_kernel_size}"
            )
        if stem_stride <= 0:
            raise ValueError(f"stem_stride must be positive, got {stem_stride}")

        self._norm_layer = norm_layer
        self.block = block
        self.layers_cfg = tuple(int(x) for x in layers)

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.base_channels = int(base_channels)
        self.stem_kernel_size = int(stem_kernel_size)
        self.stem_stride = int(stem_stride)
        self.use_stem_pool = bool(use_stem_pool)

        # 是否在最终分类头 fc 之前对全局池化后的特征做 L2 normalize。
        # 默认 False，保持原始 ResNet1D 行为；训练脚本可以通过
        # --l2_normalize_before_fc 在模型创建后覆盖该属性。
        self.l2_normalize_before_fc = bool(l2_normalize_before_fc)

        # 当前 stage 的输入通道数（会在 _make_layer 中逐步更新）
        self.inplanes = self.base_channels

        # ---------------- stem ----------------
        self.conv1 = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.base_channels,
            kernel_size=self.stem_kernel_size,
            stride=self.stem_stride,
            padding=self.stem_kernel_size // 2,
            bias=False,
        )
        self.bn1 = norm_layer(self.base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = (
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            if self.use_stem_pool
            else nn.Identity()
        )

        # ---------------- 4 residual stages ----------------
        self.layer1 = self._make_layer(block, planes=self.base_channels,     blocks=self.layers_cfg[0], stride=1)
        self.layer2 = self._make_layer(block, planes=self.base_channels * 2, blocks=self.layers_cfg[1], stride=2)
        self.layer3 = self._make_layer(block, planes=self.base_channels * 4, blocks=self.layers_cfg[2], stride=2)
        self.layer4 = self._make_layer(block, planes=self.base_channels * 8, blocks=self.layers_cfg[3], stride=2)

        # ---------------- global pooling + head ----------------
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = self.base_channels * 8 * block.expansion
        self.fc = nn.Linear(self.feature_dim, self.num_classes)

        self._initialize_weights(zero_init_residual=zero_init_residual)

    def _initialize_weights(self, zero_init_residual: bool = False) -> None:
        """
        参考标准 ResNet 的初始化方式。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[nn.Module],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """
        构建一个 stage。
        该 stage 的第一个 block 可能负责:
        - 下采样（stride != 1）
        - 通道数变换（inplanes != planes * expansion）
        """
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                norm_layer=norm_layer,
            )
        ]

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes=self.inplanes,
                    planes=planes,
                    stride=1,
                    downsample=None,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward_stem(self, x: torch.Tensor) -> torch.Tensor:
        """
        经过 stem 后的特征。
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回池化后的 backbone 特征，shape = [B, feature_dim]。
        """
        if x.ndim != 3:
            raise ValueError(f"ResNet1D expects input [B,C,L], got shape={tuple(x.shape)}")

        x = self.forward_stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)      # [B, C, D1]
        x = torch.flatten(x, 1)  # [B, D]
        return x

    def forward_head(self, feat: torch.Tensor) -> torch.Tensor:
        """
        将 backbone 特征送入最终分类头。

        当 self.l2_normalize_before_fc=True 时，会先对每个样本的
        backbone feature 做 L2 normalize，再送入 fc：
            feat = feat / ||feat||_2

        这对对比学习预训练后的特征有时有帮助，因为它让分类器更
        关注特征方向而不是特征模长。默认关闭，因此不改变原始行为。
        """
        if getattr(self, "l2_normalize_before_fc", False):
            feat = F.normalize(feat, p=2, dim=1, eps=1e-12)
        return self.fc(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        logits = self.forward_head(feat)
        return logits

    def reset_classifier(self, num_classes: int) -> None:
        """
        重置最终分类头。
        在某些迁移学习场景下会方便一些。
        """
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        self.num_classes = int(num_classes)
        self.fc = nn.Linear(self.feature_dim, self.num_classes)

    @property
    def classifier(self) -> nn.Module:
        """
        语义化别名。
        但真正给后续脚本使用的主名字仍然是 self.fc。
        """
        return self.fc


def build_resnet1d(
    arch: str,
    in_channels: int,
    num_classes: int,
    base_channels: int = 64,
    stem_kernel_size: int = 7,
    stem_stride: int = 2,
    use_stem_pool: bool = True,
    zero_init_residual: bool = False,
    l2_normalize_before_fc: bool = False,
) -> ResNet1D:
    """
    通过字符串名称构建常见 ResNet1D。

    支持:
        - resnet10_1d
        - resnet18_1d
        - resnet34_1d
        - resnet50_1d
    """
    arch = str(arch).lower().strip()

    if arch == "resnet10_1d":
        block = BasicBlock1D
        layers = [1, 1, 1, 1]
    elif arch == "resnet18_1d":
        block = BasicBlock1D
        layers = [2, 2, 2, 2]
    elif arch == "resnet34_1d":
        block = BasicBlock1D
        layers = [3, 4, 6, 3]
    elif arch == "resnet50_1d":
        block = Bottleneck1D
        layers = [3, 4, 6, 3]
    else:
        raise ValueError(
            f"Unsupported arch='{arch}'. "
            f"Expected one of: resnet10_1d, resnet18_1d, resnet34_1d, resnet50_1d"
        )

    return ResNet1D(
        block=block,
        layers=layers,
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        stem_kernel_size=stem_kernel_size,
        stem_stride=stem_stride,
        use_stem_pool=use_stem_pool,
        zero_init_residual=zero_init_residual,
        l2_normalize_before_fc=l2_normalize_before_fc,
    )


def resnet10_1d(
    in_channels: int,
    num_classes: int,
    base_channels: int = 64,
    stem_kernel_size: int = 7,
    stem_stride: int = 2,
    use_stem_pool: bool = True,
    l2_normalize_before_fc: bool = False,
) -> ResNet1D:
    return build_resnet1d(
        arch="resnet10_1d",
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        stem_kernel_size=stem_kernel_size,
        stem_stride=stem_stride,
        use_stem_pool=use_stem_pool,
        l2_normalize_before_fc=l2_normalize_before_fc,
    )


def resnet18_1d(
    in_channels: int,
    num_classes: int,
    base_channels: int = 64,
    stem_kernel_size: int = 7,
    stem_stride: int = 2,
    use_stem_pool: bool = True,
    l2_normalize_before_fc: bool = False,
) -> ResNet1D:
    return build_resnet1d(
        arch="resnet18_1d",
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        stem_kernel_size=stem_kernel_size,
        stem_stride=stem_stride,
        use_stem_pool=use_stem_pool,
        l2_normalize_before_fc=l2_normalize_before_fc,
    )


def resnet34_1d(
    in_channels: int,
    num_classes: int,
    base_channels: int = 64,
    stem_kernel_size: int = 7,
    stem_stride: int = 2,
    use_stem_pool: bool = True,
    l2_normalize_before_fc: bool = False,
) -> ResNet1D:
    return build_resnet1d(
        arch="resnet34_1d",
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        stem_kernel_size=stem_kernel_size,
        stem_stride=stem_stride,
        use_stem_pool=use_stem_pool,
        l2_normalize_before_fc=l2_normalize_before_fc,
    )


def resnet50_1d(
    in_channels: int,
    num_classes: int,
    base_channels: int = 64,
    stem_kernel_size: int = 7,
    stem_stride: int = 2,
    use_stem_pool: bool = True,
    l2_normalize_before_fc: bool = False,
) -> ResNet1D:
    return build_resnet1d(
        arch="resnet50_1d",
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        stem_kernel_size=stem_kernel_size,
        stem_stride=stem_stride,
        use_stem_pool=use_stem_pool,
        l2_normalize_before_fc=l2_normalize_before_fc,
    )


if __name__ == "__main__":
    model = resnet18_1d(
        in_channels=28,
        num_classes=17,
        base_channels=64,
        stem_kernel_size=7,
        stem_stride=2,
        use_stem_pool=True,
    )

    x = torch.randn(4, 28, 256)
    y = model(x)
    f = model.forward_features(x)

    print("logits:", y.shape)         # [4, 17]
    print("feat:", f.shape)           # [4, feature_dim]
    print("feature_dim:", model.feature_dim)
    print("fc.in_features:", model.fc.in_features)
    print("fc.out_features:", model.fc.out_features)