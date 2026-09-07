from dataclasses import dataclass
import torch
from typing import List, Tuple, Optional


@dataclass
class DeformableVoxelEncoderOutput:
    """
    StateEncoder 模块的标准输出结构体。

    属性:
        hidden_states: 编码增强后的展平特征序列，形状为 [B, Total_Tokens, hidden_size]
        pos_embed: 融合了 3D 空间与层级编码的位置向量序列，形状为 [B, Total_Tokens, hidden_size]
        spatial_feats: 重构还原为多尺度 3D 网格的特征张量列表 [[B, hidden_size, X_l, Y_l, Z_l], ...]
        spatial_shapes: 各特征层对应的三维网格尺寸列表 [(X_1, Y_1, Z_1), (X_2, Y_2, Z_2), ...]
    """
    hidden_states: torch.Tensor
    pos_embed: torch.Tensor
    spatial_feats: List[torch.Tensor]
    spatial_shapes: List[Tuple[int, int, int]]

    def to(self, device: torch.device | str) -> "DeformableVoxelEncoderOutput":
        return DeformableVoxelEncoderOutput(
            hidden_states=self.hidden_states.to(device),
            pos_embed=self.pos_embed.to(device),
            spatial_feats=[feat.to(device) for feat in self.spatial_feats],
            spatial_shapes=self.spatial_shapes,
        )