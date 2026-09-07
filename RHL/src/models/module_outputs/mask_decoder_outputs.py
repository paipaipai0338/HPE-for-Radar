import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class MaskDecoderOutput:
    """
    MaskDecoder 模块的标准输出结构体。
    
    属性:
        pred_masks: 实例体素掩码 Logits，形状为 [B, Q, X, Y, Z] (如 [B, Q, 48, 48, 32])
        semantic_seg: 全局体素占据 (Occupancy) 预测 Logits，形状为 [B, 1, X, Y, Z] (可选)
    """
    pred_masks: torch.Tensor
    semantic_seg: Optional[torch.Tensor] = None

    def to(self, device: torch.device | str) -> "MaskDecoderOutput":
        return MaskDecoderOutput(
            pred_masks=self.pred_masks.to(device),
            semantic_seg=self.semantic_seg.to(device) if self.semantic_seg is not None else None,
        )