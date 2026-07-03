r"""
Weighted sum fusion
"""
from __future__ import annotations

from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSumFusion(nn.Module):
    """
    多模态可学习加权融合模块

    功能
    ----
    1) 支持多个模态输入
    2) 支持两种权重形式：
       - scalar  : 每个模态一个标量权重，shape = [M]
       - feature : 每个模态每个特征维一个权重，shape = [M, D]
    3) 支持两种融合方式：
       - normalize=True  : 对模态权重做 softmax，形成归一化加权平均
       - normalize=False : 不归一化，直接做可学习加权和
    4) 支持选择是否先将不同模态映射到统一隐藏空间：
       - use_projection=True  : 各模态先映射到 hidden_dim
       - use_projection=False : 要求所有模态输入维度一致，直接融合

    输入
    ----
    modalities:
        可以是:
        - dict[str, Tensor], 每个 Tensor 形状 [B, D_i]
        - list[Tensor], 每个 Tensor 形状 [B, D_i]

    输出
    ----
    fused: Tensor
        融合后的特征
        shape [B, fusion_dim]

    fusion_weights: Tensor
        融合权重
        - scalar 模式: [M]
        - feature 模式: [M, fusion_dim]

    projected_features: Tensor
        所有模态用于融合的特征
        shape [B, M, fusion_dim]
    """

    def __init__(
        self,
        input_dims: Union[Dict[str, int], List[int]],
        hidden_dim: int,
        sum_method: str = "scalar",   # "scalar" or "feature"
        normalize: bool = True,
        use_projection: bool = False,
        activation: str = "identity",
        use_pre_bn: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        if sum_method not in {"scalar", "feature"}:
            raise ValueError("sum_method must be 'scalar' or 'feature'")

        if activation not in {"tanh", "relu", "gelu", "identity"}:
            raise ValueError(
                f"activation must be one of ['tanh', 'relu', 'gelu', 'identity'], got {activation}"
            )

        self.sum_method = sum_method
        self.normalize = normalize
        self.use_projection = use_projection
        self.use_pre_bn = use_pre_bn
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if isinstance(input_dims, dict):
            self.modality_names = list(input_dims.keys())
            self.input_dims = dict(input_dims)
        elif isinstance(input_dims, list):
            self.modality_names = [f"modality_{i}" for i in range(len(input_dims))]
            self.input_dims = {
                name: dim for name, dim in zip(self.modality_names, input_dims)
            }
        else:
            raise TypeError("input_dims must be dict[str, int] or list[int]")

        self.num_modalities = len(self.modality_names)
        if self.num_modalities < 2:
            raise ValueError("At least 2 modalities are required for fusion")

        input_dim_values = [self.input_dims[name] for name in self.modality_names]

        if use_projection:
            self.fusion_dim = hidden_dim
        else:
            unique_dims = set(input_dim_values)
            if len(unique_dims) != 1:
                raise ValueError(
                    "When use_projection=False, all modality input dims must be identical. "
                    f"Got dims: {input_dim_values}"
                )
            self.fusion_dim = input_dim_values[0]

        self.pre_bns = nn.ModuleDict()
        for name in self.modality_names:
            self.pre_bns[name] = nn.BatchNorm1d(self.input_dims[name])

        self.projectors = nn.ModuleDict()
        if use_projection:
            for name in self.modality_names:
                self.projectors[name] = nn.Linear(
                    self.input_dims[name], self.fusion_dim, bias=True
                )

        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

        if self.sum_method == "scalar":
            self.weights = nn.Parameter(torch.ones(self.num_modalities))
        else:
            self.weights = nn.Parameter(
                torch.ones(self.num_modalities, self.fusion_dim)
            )

    def _normalize_input(
        self,
        modalities: Union[Dict[str, torch.Tensor], List[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        if isinstance(modalities, dict):
            missing = [name for name in self.modality_names if name not in modalities]
            extra = [name for name in modalities if name not in self.modality_names]
            if missing:
                raise ValueError(f"Missing modalities: {missing}")
            if extra:
                raise ValueError(f"Unexpected modalities: {extra}")
            return modalities

        if isinstance(modalities, list):
            if len(modalities) != self.num_modalities:
                raise ValueError(
                    f"Expected {self.num_modalities} modalities, got {len(modalities)}"
                )
            return {name: x for name, x in zip(self.modality_names, modalities)}

        raise TypeError("modalities must be dict[str, Tensor] or list[Tensor]")

    def _project_one(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(
                f"Modality '{name}' must be a 2D tensor [B, D], but got shape {tuple(x.shape)}"
            )

        expected_dim = self.input_dims[name]
        if x.size(1) != expected_dim:
            raise ValueError(
                f"Modality '{name}' expected feature dim {expected_dim}, got {x.size(1)}"
            )

        if self.use_pre_bn:
            x = self.pre_bns[name](x)

        if self.use_projection:
            x = self.projectors[name](x)
            x = self.act(x)
            x = self.dropout(x)

        return x

    def _get_fusion_weights(self) -> torch.Tensor:
        """
        返回融合权重。

        scalar 模式:
            normalize=True  -> [M]
            normalize=False -> [M]

        feature 模式:
            normalize=True  -> [M, D]
            normalize=False -> [M, D]
        """
        if self.normalize:
            return F.softmax(self.weights, dim=0)
        return self.weights

    def fusion(
        self,
        modalities: Union[Dict[str, torch.Tensor], List[torch.Tensor]],
        return_projected: bool = True,
    ):
        modalities = self._normalize_input(modalities)

        projected_list = []
        for name in self.modality_names:
            h = self._project_one(name, modalities[name])   # [B, fusion_dim]
            projected_list.append(h)

        # [B, M, fusion_dim]
        projected = torch.stack(projected_list, dim=1)

        weights = self._get_fusion_weights()

        if self.sum_method == "scalar":
            # weights: [M] -> [1, M, 1]
            fused = (projected * weights.view(1, self.num_modalities, 1)).sum(dim=1)
        else:
            # weights: [M, D] -> [1, M, D]
            fused = (projected * weights.unsqueeze(0)).sum(dim=1)

        if return_projected:
            return fused, weights, projected
        return fused, weights

    def get_weights(self):
        with torch.no_grad():
            return self._get_fusion_weights().detach().cpu()

    def forward(
        self,
        modalities: Union[Dict[str, torch.Tensor], List[torch.Tensor]],
        return_projected: bool = False,
    ):
        return self.fusion(modalities, return_projected=return_projected)


class WeightedSumFusionClassifier(nn.Module):
    """
    多模态 weighted-sum fusion + 分类头
    """

    def __init__(
        self,
        input_dims: Union[Dict[str, int], List[int]],
        hidden_dim: int,
        hidden_fc_dim: int,
        num_classes: int,
        sum_method: str = "scalar",
        normalize: bool = True,
        use_projection: bool = False,
        activation: str = "identity",
        use_pre_bn: bool = False,
        fusion_dropout: float = 0.0,
        classifier_dropout: float = 0.0,
    ):
        super().__init__()

        self.fusion = WeightedSumFusion(
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            sum_method=sum_method,
            normalize=normalize,
            use_projection=use_projection,
            activation=activation,
            use_pre_bn=use_pre_bn,
            dropout=fusion_dropout,
        )

        fusion_dim = self.fusion.fusion_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_fc_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_fc_dim, num_classes),
        )

    def forward(
        self,
        modalities: Union[Dict[str, torch.Tensor], List[torch.Tensor]],
        return_details: bool = False,
    ):
        fused, weights, projected = self.fusion(
            modalities, return_projected=True
        )
        logits = self.classifier(fused)

        if return_details:
            return logits, weights, projected, fused
        return logits
    


##############################################
#                   测试函数
##############################################
def test_weighted_sum_scalar_no_projection():
    print("=== test_weighted_sum_scalar_no_projection ===")
    batch_size = 4

    fusion = WeightedSumFusion(
        input_dims={"rgb": 128, "depth": 128, "emg": 128},
        hidden_dim=64,
        sum_method="scalar",
        normalize=True,
        use_projection=False,
    )

    inputs = {
        "rgb": torch.randn(batch_size, 128),
        "depth": torch.randn(batch_size, 128),
        "emg": torch.randn(batch_size, 128),
    }

    fused, weights, projected = fusion(inputs, return_projected=True)

    print("fused.shape =", fused.shape)          # [4, 128]
    print("weights.shape =", weights.shape)      # [3]
    print("projected.shape =", projected.shape)  # [4, 3, 128]

    assert fused.shape == (batch_size, 128)
    assert weights.shape == (3,)
    assert projected.shape == (batch_size, 3, 128)

    # normalize=True 时模态权重和应为 1
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
    print("passed.\n")


def test_weighted_sum_feature_with_projection():
    print("=== test_weighted_sum_feature_with_projection ===")
    batch_size = 4

    fusion = WeightedSumFusion(
        input_dims={"rgb": 512, "depth": 256, "emg": 64},
        hidden_dim=128,
        sum_method="feature",
        normalize=True,
        use_projection=True,
        activation="relu",
        use_pre_bn=True,
        dropout=0.1,
    )

    inputs = {
        "rgb": torch.randn(batch_size, 512),
        "depth": torch.randn(batch_size, 256),
        "emg": torch.randn(batch_size, 64),
    }

    fused, weights, projected = fusion(inputs, return_projected=True)

    print("fused.shape =", fused.shape)          # [4, 128]
    print("weights.shape =", weights.shape)      # [3, 128]
    print("projected.shape =", projected.shape)  # [4, 3, 128]

    assert fused.shape == (batch_size, 128)
    assert weights.shape == (3, 128)
    assert projected.shape == (batch_size, 3, 128)

    # feature normalize=True 时，每个维度在模态维上和为 1
    s = weights.sum(dim=0)  # [128]
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
    print("passed.\n")


def test_weighted_sum_classifier_forward():
    print("=== test_weighted_sum_classifier_forward ===")
    batch_size = 5

    model = WeightedSumFusionClassifier(
        input_dims={"rgb": 256, "depth": 256},
        hidden_dim=128,
        hidden_fc_dim=64,
        num_classes=6,
        sum_method="feature",
        normalize=False,
        use_projection=True,
        classifier_dropout=0.1,
    )

    inputs = {
        "rgb": torch.randn(batch_size, 256),
        "depth": torch.randn(batch_size, 256),
    }

    logits, weights, projected, fused = model(inputs, return_details=True)

    print("logits.shape =", logits.shape)        # [5, 6]
    print("weights.shape =", weights.shape)      # [2, 128]
    print("projected.shape =", projected.shape)  # [5, 2, 128]
    print("fused.shape =", fused.shape)          # [5, 128]

    assert logits.shape == (batch_size, 6)
    assert weights.shape == (2, 128)
    assert projected.shape == (batch_size, 2, 128)
    assert fused.shape == (batch_size, 128)

    target = torch.randint(0, 6, (batch_size,))
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

    assert model.fusion.weights.grad is not None
    print("passed.\n")


def test_error_when_no_projection_and_dims_differ():
    print("=== test_error_when_no_projection_and_dims_differ ===")
    try:
        _ = WeightedSumFusion(
            input_dims={"rgb": 128, "depth": 64},
            hidden_dim=32,
            sum_method="scalar",
            normalize=True,
            use_projection=False,
        )
    except ValueError as e:
        print("caught expected error:", e)
        print("passed.\n")
        return

    raise AssertionError("Expected ValueError was not raised.")


def test_error_when_missing_modality():
    print("=== test_error_when_missing_modality ===")
    fusion = WeightedSumFusion(
        input_dims={"rgb": 128, "depth": 128},
        hidden_dim=64,
        sum_method="scalar",
        normalize=True,
        use_projection=False,
    )

    bad_inputs = {
        "rgb": torch.randn(4, 128),
    }

    try:
        _ = fusion(bad_inputs)
    except ValueError as e:
        print("caught expected error:", e)
        print("passed.\n")
        return

    raise AssertionError("Expected ValueError was not raised.")


if __name__ == "__main__":
    torch.manual_seed(42)

    test_weighted_sum_scalar_no_projection()
    test_weighted_sum_feature_with_projection()
    test_weighted_sum_classifier_forward()
    test_error_when_no_projection_and_dims_differ()
    test_error_when_missing_modality()

    print("All tests passed.")



