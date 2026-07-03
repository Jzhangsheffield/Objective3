from __future__ import annotations

from typing import Dict, List, Union, Optional

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Multi-modal cross-attention fusion.

    功能
    ----
    1) 支持多个模态输入
    2) 支持选择是否先将不同模态映射到统一隐藏空间:
       - use_projection=True  : 各模态先映射到 hidden_dim
       - use_projection=False : 要求所有模态输入最后一维一致，直接融合
    3) 支持输入为:
       - [B, D]
       - [B, T, D]
    4) 对每个目标模态 q，让其分别 cross-attend 到其他模态 kv
    5) 将所有 query 模态的聚合表示再做融合，得到最终 fused feature

    输入
    ----
    modalities:
        可以是:
        - dict[str, Tensor]
        - list[Tensor]

        每个 Tensor 可以是:
        - [B, D]
        - [B, T, D]

    输出
    ----
    fused: Tensor
        融合后的特征
        shape [B, fusion_out_dim]

    attention_info: dict
        包含一些中间 cross-attention 结果，便于调试和检查

    projected_features: dict[str, Tensor]
        每个模态投影后的特征
        shape:
        - [B, 1, fusion_dim]  if input was [B, D]
        - [B, T, fusion_dim]  if input was [B, T, D]
    """

    def __init__(self,
                 input_dims:  Union[Dict[str, int], List[int]],
                 hidden_dim: int,
                 num_heads: int,
                 fusion_out_dim: int,
                 use_projection: bool = False,
                 activation: str = "identity",
                 use_pre_bn: bool = False,
                 projection_dropout: float = 0.0,
                 attention_dropout: float =0.0,
                 mlp_dropout: float = 0.0):
        
        super().__init__()

        if activation not in {"tanh", "relu", "gelu", "identity"}:
            raise ValueError(
                f"activation must be one of ['tanh', 'relu', 'gelu', 'identity'], got {activation}"
            )

        self.use_projection = use_projection
        self.use_pre_bn = use_pre_bn
        self.projection_dropout = (
            nn.Dropout(projection_dropout) if projection_dropout > 0 else nn.Identity()
        )

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

        if self.fusion_dim % num_heads != 0:
            raise ValueError(
                f"fusion_dim ({self.fusion_dim}) must be divisible by num_heads ({num_heads})"
            )

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

        # 为每对 (query_modality, kv_modality) 建一个 cross-attention 层
        self.cross_attn = nn.ModuleDict()
        self.cross_layernorm = nn.ModuleDict()

        for q_name in self.modality_names:
            for kv_name in self.modality_names:
                if q_name == kv_name:
                    continue
                key = f"{q_name}_attend_to_{kv_name}"
                self.cross_attn[key] = nn.MultiheadAttention(
                    embed_dim=self.fusion_dim,
                    num_heads = num_heads,
                    dropout=attention_dropout,
                    batch_first=True
                )
                self.cross_layernorm[key] = nn.LayerNorm(self.fusion_dim)

        # 每个 query 模态，把它 attend 到其他模态后的多个结果做整合
        self.query_fusion = nn.ModuleDict()
        for q_name in self.modality_names:
            self.query_fusion[q_name] = nn.Sequential(
                nn.Linear(self.fusion_dim, self.fusion_dim),
                nn.ReLU(),
                nn.Dropout(mlp_dropout)
            )
        
        # 最终将所有 query 模态的聚合表示再融合
        final_in_dim = self.num_modalities * self.fusion_dim
        self.final_mlp = nn.Sequential(
            nn.Linear(final_in_dim, final_in_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(final_in_dim, fusion_out_dim),
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
    


    def _ensure_3d(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if x.dim() == 2:
            # [B, D] -> [B, 1, D]
            return x.unsqueeze(1)
        if x.dim() == 3:
            return x
        raise ValueError(
            f"Modality '{name}' must be [B, D] or [B, T, D], got shape {tuple(x.shape)}"
        )
    


    def _project_one(self, name: str, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_3d(x, name)  # [B, T, D]

        expected_dim = self.input_dims[name]
        if x.size(-1) != expected_dim:
            raise ValueError(
                f"Modality '{name}' expected feature dim {expected_dim}, got {x.size(-1)}"
            )

        if self.use_pre_bn:
            b, t, d = x.shape
            x = x.reshape(b * t, d)
            x = self.pre_bns[name](x)
            x = x.reshape(b, t, d)

        if self.use_projection:
            x = self.projectors[name](x)
            x = self.act(x)
            x = self.projection_dropout(x)

        return x
    

    def fusion(self,
               modalities: Union[Dict[str, torch.Tensor], List[torch.Tensor]],
               return_projected: bool = True,):
        
        modalities = self._normalize_input(modalities)

        projected = {}
        for name in self.modality_names:
            projected[name] = self._project_one(name, modalities[name]) # [B, T, fusion_dim]

        query_outputs = {}
        attention_maps = {}

        for q_name in self.modality_names:
            pair_outputs = []

            q = projected[q_name] # [B, Tq, D]

            for kv_name in self.modality_names:
                if q_name == kv_name:
                    continue
                
                kv = projected[kv_name] # [B, Tk, D]
                key = f"{q_name}_attend_to_{kv_name}"

                attn_out, attn_weights = self.cross_attn[key](
                    query=q,
                    key=kv,
                    value=kv,
                    need_weights=True,
                    average_attn_weights=False
                )

                # residual + layer norm
                attn_out = self.cross_layernorm[key](q + attn_out)

                pooled = attn_out.mean(dim=1)  # [B, D]
                pair_outputs.append(pooled  )

                attention_maps[key] = attn_weights

                # 对同一个 query 模态产生的多个 pairwise 输出求平均
                # [num_other_modalities, B, D] -> [B, D]
            stacked = torch.stack(pair_outputs, dim=0).mean(dim=0)
            query_outputs[q_name] = self.query_fusion[q_name](stacked)

        # [B, M*D]  
        final_concat = torch.cat([query_outputs[name] for name in self.modality_names], dim=1)      

        fused = self.final_mlp(final_concat)

        attention_info = {
            "query_outputs": query_outputs,
            "attention_maps": attention_maps
        }

        if return_projected:
            return fused, attention_info, projected
        return fused, attention_info



    def forward(
        self,
        modalities: Union[Dict[str, torch.Tensor], List[torch.Tensor]],
        return_projected: bool = False,
    ):
        return self.fusion(modalities, return_projected=return_projected)
    




class CrossAttentionFusionClassifier(nn.Module):
    """
    多模态 cross-attention fusion + 分类头
    """

    def __init__(
        self,
        input_dims: Union[Dict[str, int], List[int]],
        hidden_dim: int,
        num_heads: int,
        fusion_out_dim: int,
        num_classes: int,
        use_projection: bool = False,
        activation: str = "identity",
        use_pre_bn: bool = False,
        projection_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        fusion_mlp_dropout: float = 0.0,
        classifier_dropout: float = 0.0,
    ):
        super().__init__()

        self.fusion = CrossAttentionFusion(
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            fusion_out_dim=fusion_out_dim,
            use_projection=use_projection,
            activation=activation,
            use_pre_bn=use_pre_bn,
            projection_dropout=projection_dropout,
            attention_dropout=attention_dropout,
            mlp_dropout=fusion_mlp_dropout,
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
        fused, attention_info, projected = self.fusion(
            modalities, return_projected=True
        )
        logits = self.classifier(fused)

        if return_details:
            return logits, attention_info, projected, fused
        return logits


############################################
#                 测试代码
###########################################
def test_cross_attention_no_projection_vector_inputs():
    print("=== test_cross_attention_no_projection_vector_inputs ===")
    batch_size = 4

    fusion = CrossAttentionFusion(
        input_dims={"rgb": 128, "depth": 128, "emg": 128},
        hidden_dim=64,
        num_heads=8,
        fusion_out_dim=96,
        use_projection=False,
    )

    inputs = {
        "rgb": torch.randn(batch_size, 128),   # [B, D]
        "depth": torch.randn(batch_size, 128), # [B, D]
        "emg": torch.randn(batch_size, 128),   # [B, D]
    }

    fused, attention_info, projected = fusion(inputs, return_projected=True)

    print("fused.shape =", fused.shape)  # [4, 96]
    print("projected[rgb].shape =", projected["rgb"].shape)  # [4, 1, 128]

    assert fused.shape == (batch_size, 96)
    assert projected["rgb"].shape == (batch_size, 1, 128)
    assert projected["depth"].shape == (batch_size, 1, 128)
    assert projected["emg"].shape == (batch_size, 1, 128)
    

    expected_pairs = 3 * 2
    assert len(attention_info["attention_maps"]) == expected_pairs
    assert len(attention_info["query_outputs"]) == 3
    for name, tensor in attention_info["query_outputs"].items():
        assert tensor.shape == (batch_size, 128)
    print("passed.\n")


def test_cross_attention_with_projection_sequence_inputs():
    print("=== test_cross_attention_with_projection_sequence_inputs ===")
    batch_size = 4

    fusion = CrossAttentionFusion(
        input_dims={"rgb": 256, "depth": 128, "emg": 64},
        hidden_dim=128,
        num_heads=8,
        fusion_out_dim=80,
        use_projection=True,
        activation="relu",
        use_pre_bn=True,
        projection_dropout=0.1,
        attention_dropout=0.1,
        mlp_dropout=0.1,
    )

    inputs = {
        "rgb": torch.randn(batch_size, 5, 256),    # [B, T, D]
        "depth": torch.randn(batch_size, 7, 128),  # [B, T, D]
        "emg": torch.randn(batch_size, 10, 64),    # [B, T, D]
    }

    fused, attention_info, projected = fusion(inputs, return_projected=True)

    print("fused.shape =", fused.shape)  # [4, 80]
    print("projected[rgb].shape =", projected["rgb"].shape)      # [4, 5, 128]
    print("projected[depth].shape =", projected["depth"].shape)  # [4, 7, 128]
    print("projected[emg].shape =", projected["emg"].shape)      # [4, 10, 128]

    assert fused.shape == (batch_size, 80)
    assert projected["rgb"].shape == (batch_size, 5, 128)
    assert projected["depth"].shape == (batch_size, 7, 128)
    assert projected["emg"].shape == (batch_size, 10, 128)

    for k, v in attention_info["query_outputs"].items():
        assert v.shape == (batch_size, 128)

    print("passed.\n")


def test_cross_attention_classifier_forward():
    print("=== test_cross_attention_classifier_forward ===")
    batch_size = 3

    model = CrossAttentionFusionClassifier(
        input_dims={"rgb": 256, "depth": 256},
        hidden_dim=128,
        num_heads=8,
        fusion_out_dim=96,
        num_classes=6,
        use_projection=True,
        classifier_dropout=0.1,
    )

    inputs = {
        "rgb": torch.randn(batch_size, 6, 256),
        "depth": torch.randn(batch_size, 4, 256),
    }

    logits, attention_info, projected, fused = model(inputs, return_details=True)

    print("logits.shape =", logits.shape)  # [3, 6]
    print("fused.shape =", fused.shape)    # [3, 96]

    assert logits.shape == (batch_size, 6)
    assert fused.shape == (batch_size, 96)

    target = torch.randint(0, 6, (batch_size,))
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

    print("passed.\n")


def test_error_when_no_projection_and_dims_differ():
    print("=== test_error_when_no_projection_and_dims_differ ===")
    try:
        _ = CrossAttentionFusion(
            input_dims={"rgb": 128, "depth": 64},
            hidden_dim=32,
            num_heads=8,
            fusion_out_dim=48,
            use_projection=False,
        )
    except ValueError as e:
        print("caught expected error:", e)
        print("passed.\n")
        return

    raise AssertionError("Expected ValueError was not raised.")


def test_error_when_hidden_not_divisible_by_heads():
    print("=== test_error_when_hidden_not_divisible_by_heads ===")
    try:
        _ = CrossAttentionFusion(
            input_dims={"rgb": 128, "depth": 128},
            hidden_dim=130,
            num_heads=8,
            fusion_out_dim=64,
            use_projection=True,
        )
    except ValueError as e:
        print("caught expected error:", e)
        print("passed.\n")
        return

    raise AssertionError("Expected ValueError was not raised.")


if __name__ == "__main__":
    torch.manual_seed(42)

    test_cross_attention_no_projection_vector_inputs()
    test_cross_attention_with_projection_sequence_inputs()
    test_cross_attention_classifier_forward()
    test_error_when_no_projection_and_dims_differ()
    test_error_when_hidden_not_divisible_by_heads()

    print("All tests passed.")