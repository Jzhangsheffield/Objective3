from __future__ import annotations

from typing import Dict, List, Union, Optional

import torch
import torch.nn as nn

class ConcatMLPFusion(nn.Module):
    """
    Multi-modal concat + MLP fusion.

    功能
    ----
    1) 支持多个模态输入
    2) 支持选择是否先将不同模态映射到统一隐藏空间:
       - use_projection=True  : 各模态先映射到 hidden_dim
       - use_projection=False : 要求所有模态输入维度一致，直接融合
    3) 将所有模态特征拼接后，通过 MLP 得到融合特征

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
        shape [B, fusion_out_dim]

    fusion_info: None
        为了与 GatedFusion / WeightedSumFusion 接口尽量保持一致，这里返回 None

    projected_features: Tensor
        所有模态用于融合的特征
        shape [B, M, fusion_dim]
    """
        
    def __init__(self, input_dims: Union[Dict[str, int], List[int]],
                hidden_dim: int,
                mlp_hidden_dim: int,
                fusion_out_dim: int,
                use_projection: bool = False,
                activation: str = "relu",
                use_pre_bn: bool = False,
                projection_dropout: float = 0.0,
                mlp_dropout: float = 0.0,):
            super(ConcatMLPFusion, self).__init__()

            if activation not in {"tanh", "relu", "gelu", "identity"}:
                    raise ValueError(f"activation must be one of ['tanh', 'relu', 'gelu', 'identity'], got {activation}")
            
            self.use_projection = use_projection
            self.use_pre_bn = use_pre_bn
            self.projection_dropout = nn.Dropout(projection_dropout) if projection_dropout > 0 else nn.Identity()

            if isinstance(input_dims, dict):
                    self.modality_names = list(input_dims.keys())
                    self.input_dims = dict(input_dims)
            elif isinstance(input_dims, list):
                    self.modality_names = [f"modality_{i}" for i in range(len(input_dims))]
                    self.input_dims = {name: dim for name, dim in zip(self.modality_names, input_dims)}
            else:
                    raise TypeError("input_dims must be ditc[str, int] or list[int]")

            self.num_modalities = len(self.modality_names)
            if self.num_modalities < 2:
                    raise ValueError("at least 2 modalities are required for fusion")

            input_dim_values = [self.input_dims[name] for name in self.modality_names]

            if use_projection:
                self.fusion_dim = hidden_dim
            else:
                unique_dims = set(input_dim_values)
                if len(unique_dims) != 1:
                    raise ValueError("when use_projection=False, all modality input dims must be identical."
                                        f"got dims: {input_dim_values}")

                self.fusion_dim = input_dim_values[0]

            self.pre_bns = nn.ModuleDict()
            for name in self.modality_names:
                self.pre_bns[name] = nn.BatchNorm1d(self.input_dims[name])
            
            self.projectors = nn.ModuleDict()
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

            concat_dim = self.num_modalities * self.fusion_dim

            self.mlp = nn.Sequential(nn.Linear(concat_dim, mlp_hidden_dim),
                                    nn.ReLU(),
                                    nn.Dropout(mlp_dropout),
                                    nn.Linear(mlp_hidden_dim,fusion_out_dim),
                                    )

            

    def _normalize_input(self, modalities: Union[Dict[str, torch.Tensor], List[torch.Tensor]]):
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
            x = self.projection_dropout(x)

        return x
    


    def fusion(self, modalities: Union[Dict[str, torch.Tensor], List[torch.Tensor]], return_projected: bool = True):
        modalities = self._normalize_input(modalities)

        projected_list = []
        for name in self.modality_names:
             h = self._project_one(name, modalities[name])
             projected_list.append(h)  # [B, fusion_dim]

        projected = torch.stack(projected_list, dim=1) # [B, M, fusion_dim] 仅仅是用于检查

        concat_features = torch.cat(projected_list, dim=1)

        fused = self.mlp(concat_features)

        if return_projected:
             return fused, None, projected
        return fused, None
    


    def forward(self, modalities:  Union[Dict[str, torch.Tensor], List[torch.Tensor]], return_projected: bool = True):
         return self.fusion(modalities, return_projected=return_projected)





class ConcatMLPFusionClassifier(nn.Module):
    """
    多模态 concat + MLP fusion + 分类头
    """

    def __init__(
        self,
        input_dims: Union[Dict[str, int], List[int]],
        hidden_dim: int,
        mlp_hidden_dim: int,
        fusion_out_dim: int,
        num_classes: int,
        use_projection: bool = False,
        activation: str = "relu",
        use_pre_bn: bool = False,
        projection_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        classifier_dropout: float = 0.0,
    ):
        super().__init__()

        self.fusion = ConcatMLPFusion(
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            fusion_out_dim=fusion_out_dim,
            use_projection=use_projection,
            activation=activation,
            use_pre_bn=use_pre_bn,
            projection_dropout=projection_dropout,
            mlp_dropout=mlp_dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_out_dim, fusion_out_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(fusion_out_dim, num_classes),
        )

    def forward(
        self,
        modalities: Union[Dict[str, torch.Tensor], List[torch.Tensor]],
        return_details: bool = False,
    ):
        fused, fusion_info, projected = self.fusion(modalities, return_projected=True)
        logits = self.classifier(fused)

        if return_details:
            return logits, fusion_info, projected, fused
        return logits
    



##################################################
#                    测试代码
##################################################
def test_concat_mlp_no_projection():
    print("=== test_concat_mlp_no_projection ===")
    batch_size = 4

    fusion = ConcatMLPFusion(
        input_dims={"rgb": 128, "depth": 128, "emg": 128},
        hidden_dim=64,
        mlp_hidden_dim=96,
        fusion_out_dim=80,
        use_projection=False,
        activation="relu",
    )

    inputs = {
        "rgb": torch.randn(batch_size, 128),
        "depth": torch.randn(batch_size, 128),
        "emg": torch.randn(batch_size, 128),
    }

    fused, fusion_info, projected = fusion(inputs, return_projected=True)

    print("fused.shape =", fused.shape)            # [4, 80]
    print("fusion_info =", fusion_info)            # None
    print("projected.shape =", projected.shape)    # [4, 3, 128]

    assert fused.shape == (batch_size, 80)
    assert fusion_info is None
    assert projected.shape == (batch_size, 3, 128)
    print("passed.\n")


def test_concat_mlp_with_projection():
    print("=== test_concat_mlp_with_projection ===")
    batch_size = 4

    fusion = ConcatMLPFusion(
        input_dims={"rgb": 512, "depth": 256, "emg": 64},
        hidden_dim=128,
        mlp_hidden_dim=192,
        fusion_out_dim=96,
        use_projection=True,
        activation="relu",
        use_pre_bn=True,
        projection_dropout=0.1,
        mlp_dropout=0.1,
    )

    inputs = {
        "rgb": torch.randn(batch_size, 512),
        "depth": torch.randn(batch_size, 256),
        "emg": torch.randn(batch_size, 64),
    }

    fused, fusion_info, projected = fusion(inputs, return_projected=True)

    print("fused.shape =", fused.shape)            # [4, 96]
    print("fusion_info =", fusion_info)            # None
    print("projected.shape =", projected.shape)    # [4, 3, 128]

    assert fused.shape == (batch_size, 96)
    assert fusion_info is None
    assert projected.shape == (batch_size, 3, 128)
    print("passed.\n")


def test_concat_mlp_classifier_forward():
    print("=== test_concat_mlp_classifier_forward ===")
    batch_size = 5

    model = ConcatMLPFusionClassifier(
        input_dims={"rgb": 256, "depth": 256},
        hidden_dim=128,
        mlp_hidden_dim=160,
        fusion_out_dim=96,
        num_classes=6,
        use_projection=True,
        classifier_dropout=0.1,
    )

    inputs = {
        "rgb": torch.randn(batch_size, 256),
        "depth": torch.randn(batch_size, 256),
    }

    logits, fusion_info, projected, fused = model(inputs, return_details=True)

    print("logits.shape =", logits.shape)          # [5, 6]
    print("fusion_info =", fusion_info)            # None
    print("projected.shape =", projected.shape)    # [5, 2, 128]
    print("fused.shape =", fused.shape)            # [5, 96]

    assert logits.shape == (batch_size, 6)
    assert fusion_info is None
    assert projected.shape == (batch_size, 2, 128)
    assert fused.shape == (batch_size, 96)

    target = torch.randint(0, 6, (batch_size,))
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()
    print("passed.\n")


def test_error_when_no_projection_and_dims_differ():
    print("=== test_error_when_no_projection_and_dims_differ ===")
    try:
        _ = ConcatMLPFusion(
            input_dims={"rgb": 128, "depth": 64},
            hidden_dim=32,
            mlp_hidden_dim=64,
            fusion_out_dim=48,
            use_projection=False,
        )
    except ValueError as e:
        print("caught expected error:", e)
        print("passed.\n")
        return

    raise AssertionError("Expected ValueError was not raised.")


def test_error_when_missing_modality():
    print("=== test_error_when_missing_modality ===")
    fusion = ConcatMLPFusion(
        input_dims={"rgb": 128, "depth": 128},
        hidden_dim=64,
        mlp_hidden_dim=80,
        fusion_out_dim=48,
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

    test_concat_mlp_no_projection()
    test_concat_mlp_with_projection()
    test_concat_mlp_classifier_forward()
    test_error_when_no_projection_and_dims_differ()
    test_error_when_missing_modality()

    print("All tests passed.")