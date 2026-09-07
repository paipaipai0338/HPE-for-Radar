import torch
from dataclasses import dataclass
from typing import  Dict, Optional, Union


@dataclass
class VoxelSegmentationOutput:
    """
    MiliHuPsen 模型的标准输出容器类。

    属性:
        pred_boxes: 各解码层预测的 3D 边界框 [L, B, Q, 6]
        cls_logits: 各解码层预测的实例置信度 Logits [L, B, Q]
        presence_logits: 各解码层预测的全局人体存在性 Logits [L, B, 1]
        pred_masks: 最终层预测的 3D 体素实例掩码 Logits [B, Q, X, Y, Z]
        semantic_seg: 全局前景体素 Occupancy 预测 Logits [B, 1, X, Y, Z] (可选)
        next_h_state: 更新后的时序隐状态 [B, C, X/4, Y/4, Z/4] (若启用时序)
        raw_point_cloud: 透传的当前帧点云张量 [B, P, 4]
        point_valid_mask: 透传的当前帧有效点掩码 [B, P]
        multiscale_features: 骨干网络提取的多尺度特征字典 {"F1": ..., "F2": ..., "F3": ...}
    """
    pred_boxes: torch.Tensor
    cls_logits: torch.Tensor
    presence_logits: torch.Tensor
    pred_masks: torch.Tensor
    semantic_seg: Optional[torch.Tensor] = None
    next_h_state: Optional[torch.Tensor] = None
    raw_point_cloud: Optional[torch.Tensor] = None
    point_valid_mask: Optional[torch.Tensor] = None
    multiscale_features: Optional[Dict[str, torch.Tensor]] = None

    def to(self, device: Union[str, torch.device]) -> "VoxelSegmentationOutput":
        return VoxelSegmentationOutput(
            pred_boxes=self.pred_boxes.to(device),
            cls_logits=self.cls_logits.to(device),
            presence_logits=self.presence_logits.to(device),
            pred_masks=self.pred_masks.to(device),
            semantic_seg=self.semantic_seg.to(device) if self.semantic_seg is not None else None,
            next_h_state=self.next_h_state.to(device) if self.next_h_state is not None else None,
            raw_point_cloud=self.raw_point_cloud.to(device) if self.raw_point_cloud is not None else None,
            point_valid_mask=self.point_valid_mask.to(device) if self.point_valid_mask is not None else None,
            multiscale_features={k: v.to(device) for k, v in self.multiscale_features.items()} if self.multiscale_features is not None else None,
        )

@dataclass
class DetrMiliHupSenOutput:
    """
    MiliHuPsen 模型的标准输出容器类。
    
    属性:
        pred_boxes: 各解码层预测的 3D 边界框 [L, B, Q, 6]
        cls_logits: 各解码层预测的实例置信度 Logits [L, B, Q]
        presence_logits: 各解码层预测的全局人体存在性 Logits [L, B, 1]
        next_h_state: 更新后的时序隐状态 [B, C, X/4, Y/4, Z/4] (若启用时序)
        raw_point_cloud: 透传的当前帧点云张量 [B, P, 4]
        point_valid_mask: 透传的当前帧有效点掩码 [B, P]
        multiscale_features: 骨干网络提取的多尺度特征字典 {"F1": ..., "F2": ..., "F3": ...}
    """
    pred_boxes: torch.Tensor
    cls_logits: torch.Tensor
    presence_logits: torch.Tensor
    next_h_state: Optional[torch.Tensor] = None
    raw_point_cloud: Optional[torch.Tensor] = None
    point_valid_mask: Optional[torch.Tensor] = None
    multiscale_features: Optional[Dict[str, torch.Tensor]] = None
    
    def to(self, device: Union[str, torch.device]) -> "DetrMiliHupSenOutput":
        return DetrMiliHupSenOutput(
            pred_boxes=self.pred_boxes.to(device),
            cls_logits=self.cls_logits.to(device),
            presence_logits=self.presence_logits.to(device),
            next_h_state=self.next_h_state.to(device) if self.next_h_state is not None else None,
            raw_point_cloud=self.raw_point_cloud.to(device) if self.raw_point_cloud is not None else None,
            point_valid_mask=self.point_valid_mask.to(device) if self.point_valid_mask is not None else None,
            multiscale_features={k: v.to(device) for k, v in self.multiscale_features.items()} if self.multiscale_features is not None else None,
        )