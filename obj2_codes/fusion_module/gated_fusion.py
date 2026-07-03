import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union

class GatedFusion(nn.Module):
    """
    Multi-modal GMU-style gated fusion.

    功能
    ----
    1) 支持多个模态输入
    2) 支持两种 gate 模式:
       - scalar: 每个样本对每个模态给一个标量权重
       - vector: 每个样本对每个模态的每个隐藏维度给一个权重
    3) 支持选择是否先将不同模态映射到统一隐藏空间:
       - use_projection=True: 各模态先映射到 hidden_dim
       - use_projection=False: 要求所有模态输入维度一致，直接做 fusion

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

    gate_weights: Tensor
        gate 权重
        - scalar 模式: [B, M]
        - vector 模式: [B, M, fusion_dim]

    projected_features: Tensor
        所有模态用于融合的特征
        shape [B, M, fusion_dim]
    """
    def __init__(self, input_dims: Union[Dict[str, int], List[int]],
                 hidden_dim: int = 128,
                 gate_type: str = "vector",
                 activation: str = 'tanh',
                 use_projection: bool = False,
                 use_pre_bn: bool = False,
                 use_post_fusion_proj: bool = False,
                 dropout: float = 0.0):
        super(GatedFusion, self).__init__()

        if gate_type not in {"scalar", "vector"}:
            raise ValueError(f"gate_type must be 'scalar' or 'vector', got {gate_type}")
        
        if activation not in {"tanh", "relu", "gelu", "identity"}:
            raise ValueError(f"activation must be one of ['tanh', 'relu', 'gelu', 'identity'], got {activation}")

        self.gate_type = gate_type
        self.use_projection = use_projection
        self.use_pre_bn = use_pre_bn
        self.use_post_fusion_proj = use_post_fusion_proj
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if isinstance(input_dims, dict):
            self.modality_names = list(input_dims.keys())
            self.input_dims = dict(input_dims)
        elif isinstance(input_dims, list):
            self.modality_names = [f"modality_{i}" for i in range(len(input_dims))]
            self.input_dims = {name: dim for name, dim in zip(self.modality_names, input_dims)}
        else:
            raise TypeError("input_dims must be dict[str, int] or list[int]")

        self.num_modalities = len(self.modality_names)
        if self.num_modalities < 2:
            raise ValueError("at least 2 modalitites are required for the fusion")

        input_dim_values = [self.input_dims[name] for name in self.modality_names]
        
        if use_projection:
            self.fusion_dim = hidden_dim
        else:
            unique_dims = set(input_dim_values)
            if len(unique_dims) != 1:
                raise ValueError(
                    "when use_projection=False, all modality input dims must be the same."
                    f"got dims: {input_dim_values}"
                )
            self.fusion_dim = input_dim_values[0]

        # 预处理 BN
        self.pre_bns = nn.ModuleDict()
        for name in self.modality_names:
            self.pre_bns[name] = nn.BatchNorm1d(self.input_dims[name])

        # 投影层
        self.projectors = nn.ModuleDict()
        if use_projection:
            for name in self.modality_names:
                self.projectors[name] = nn.Linear(self.input_dims[name], self.fusion_dim, bias=True)

        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

        gate_input_dim = self.num_modalities * self.fusion_dim

        if gate_type == "scalar":
            gate_output_dim = self.num_modalities
        else:
            gate_output_dim = self.num_modalities * self.fusion_dim

        self.gate_net = nn.Linear(gate_input_dim, gate_output_dim, bias=True)

        if use_post_fusion_proj:
            self.post_fusion = nn.Sequential(
                nn.Linear(self.fusion_dim, self.fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.post_fusion = nn.Identity()


    def _normalize_input(self, modalities: Union[Dict[str, torch.Tensor], List[torch.Tensor]]):
        if isinstance(modalities, dict):
            missing = [name for name in self.modality_names if name not in modalities]
            extra = [name for name in modalities if name not in self.modality_names]
            if missing:
                raise ValueError(f"missing modalities: {missing}")
            if extra:
                raise ValueError(f"unexpected modalities: {extra}")
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
    

    def forward(self,  
                modalities: Union[Dict[str, torch.Tensor], List[torch.Tensor]],
                return_projected: bool = True ):
        modalities = self._normalize_input(modalities)

        projected_list = []
        for name in self.modality_names:
            h = self._project_one(name, modalities[name]) # [B, fusion_dim]
            projected_list.append(h)

        projected = torch.stack(projected_list, dim=1) # [B, M, fusion_dim] 沿着新的维度拼接
        
        gate_input = projected.reshape(projected.size(0), -1) # [B, M * fusion_dim]

        gate_logits = self.gate_net(gate_input)

        if self.gate_type == "scalar":
            gate_logits = gate_logits.view(projected.size(0), self.num_modalities)
            gate_weights = F.softmax(gate_logits, dim=1) # [B, M]
            
            fused = (projected * gate_weights.unsqueeze(-1)).sum(dim=1) # [B, fusion_dim]

        else:
            gate_logits = gate_logits.view(projected.size(0), self.num_modalities, self.fusion_dim)
            gate_weights = F.softmax(gate_logits, dim=1)

            fused = (projected * gate_weights).sum(dim=1)

        if return_projected:
            return fused, gate_weights, projected
        
        return fused, gate_weights
    


class GatedFusionClassifier(nn.Module):
    def __init__(
        self,
        input_dims: Union[Dict[str, int], List[int]],
        hidden_dim: int,
        num_classes: int,
        gate_type: str = "vector",
        activation: str = "tanh",
        use_projection: bool = True,
        use_pre_bn: bool = False,
        fusion_dropout: float = 0.0,
        classifier_dropout: float = 0.0,
    ):
        super().__init__()

        self.fusion = GatedFusion(
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            gate_type=gate_type,
            activation=activation,
            use_projection=use_projection,
            use_pre_bn=use_pre_bn,
            use_post_fusion_proj=False,
            dropout=fusion_dropout,
        )

        fusion_dim = self.fusion.fusion_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, modalities, return_details: bool = False):
        fused, gate_weights, projected = self.fusion(
            modalities, return_projected=True
        )
        logits = self.classifier(fused)

        if return_details:
            return logits, gate_weights, projected, fused
        return logits
    


############################################
#                  测试代码
###########################################

# 假设上面的类定义已经在同一个文件里
# 如果你把它们放在 gated_fusion.py 里，就改成：
# from gated_fusion import GatedFusion, MultiModalGMUClassifier


def test_fusion_scalar_no_projection():
    print("=== test_fusion_scalar_no_projection ===")
    batch_size = 4
    fusion = GatedFusion(
        input_dims={"rgb": 128, "depth": 128, "emg": 128},
        hidden_dim=64,              # use_projection=False 时这个不会实际用于投影
        gate_type="scalar",
        use_projection=False,
        use_pre_bn=False,
    )

    inputs = {
        "rgb": torch.randn(batch_size, 128),
        "depth": torch.randn(batch_size, 128),
        "emg": torch.randn(batch_size, 128),
    }

    fused, gate_weights, projected = fusion(inputs, return_projected=True)

    print("fused.shape =", fused.shape)              # [4, 128]
    print("gate_weights.shape =", gate_weights.shape) # [4, 3]
    print("projected.shape =", projected.shape)      # [4, 3, 128]

    assert fused.shape == (batch_size, 128)
    assert gate_weights.shape == (batch_size, 3)
    assert projected.shape == (batch_size, 3, 128)

    # scalar gate 按模态维求和应接近 1
    s = gate_weights.sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
    print("passed.\n")


def test_fusion_vector_with_projection():
    print("=== test_fusion_vector_with_projection ===")
    batch_size = 4
    fusion = GatedFusion(
        input_dims={"rgb": 512, "depth": 256, "emg": 64},
        hidden_dim=128,
        gate_type="vector",
        use_projection=True,
        use_pre_bn=True,
        dropout=0.1,
    )

    inputs = {
        "rgb": torch.randn(batch_size, 512),
        "depth": torch.randn(batch_size, 256),
        "emg": torch.randn(batch_size, 64),
    }

    fused, gate_weights, projected = fusion(inputs, return_projected=True)

    print("fused.shape =", fused.shape)               # [4, 128]
    print("gate_weights.shape =", gate_weights.shape) # [4, 3, 128]
    print("projected.shape =", projected.shape)       # [4, 3, 128]

    assert fused.shape == (batch_size, 128)
    assert gate_weights.shape == (batch_size, 3, 128)
    assert projected.shape == (batch_size, 3, 128)

    # vector gate 在模态维求和应接近 1
    s = gate_weights.sum(dim=1)   # [B, 128]
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
    print("passed.\n")


def test_classifier_forward():
    print("=== test_classifier_forward ===")
    batch_size = 5
    model = GatedFusionClassifier(
        input_dims={"rgb": 256, "depth": 256},
        hidden_dim=128,
        num_classes=6,
        gate_type="vector",
        use_projection=True,
        classifier_dropout=0.1,
    )

    inputs = {
        "rgb": torch.randn(batch_size, 256),
        "depth": torch.randn(batch_size, 256),
    }

    logits, gate_weights, projected, fused = model(inputs, return_details=True)

    print("logits.shape =", logits.shape)             # [5, 6]
    print("gate_weights.shape =", gate_weights.shape) # [5, 2, 128]
    print("projected.shape =", projected.shape)       # [5, 2, 128]
    print("fused.shape =", fused.shape)               # [5, 128]

    assert logits.shape == (batch_size, 6)
    assert gate_weights.shape == (batch_size, 2, 128)
    assert projected.shape == (batch_size, 2, 128)
    assert fused.shape == (batch_size, 128)
    print("passed.\n")


def test_error_when_no_projection_and_dims_differ():
    print("=== test_error_when_no_projection_and_dims_differ ===")
    try:
        _ = GatedFusion(
            input_dims={"rgb": 128, "depth": 64},
            hidden_dim=32,
            gate_type="scalar",
            use_projection=False,
        )
    except ValueError as e:
        print("caught expected error:", e)
        print("passed.\n")
        return

    raise AssertionError("Expected ValueError was not raised.")


def test_error_when_missing_modality():
    print("=== test_error_when_missing_modality ===")
    fusion = GatedFusion(
        input_dims={"rgb": 128, "depth": 128},
        hidden_dim=64,
        gate_type="scalar",
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

    test_fusion_scalar_no_projection()
    test_fusion_vector_with_projection()
    test_classifier_forward()
    test_error_when_no_projection_and_dims_differ()
    test_error_when_missing_modality()

    print("All tests passed.")