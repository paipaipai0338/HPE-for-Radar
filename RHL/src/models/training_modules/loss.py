from src.models import LOSS_LIB
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union


class CustomLoss(nn.Module):
    def __init__(
        self,
        base_weight: float = 2.0,
        schedule_type: str = "constant",
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__()
        self.base_weight = base_weight
        self.current_weight = base_weight if schedule_type == "constant" else 0.0
        self.schedule_type = schedule_type
        self.max_steps = max(1, max_steps) if max_steps is not None else 1

    def step(self, current_step: int):
        """根据当前训练步数更新动态权重"""
        if self.schedule_type == "constant":
            self.current_weight = self.base_weight
            
        elif self.schedule_type == "linear_warmup":
            progress = min(1.0, current_step / self.max_steps)
            self.current_weight = self.base_weight * progress
            
        elif self.schedule_type == "cosine_decay":
            progress = min(1.0, current_step / self.max_steps)
            self.current_weight = self.base_weight * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _focal_loss_core(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        体素级 Sigmoid Focal Loss 基础核心算子。
        在 reduction='mean' 时按有效正样本数量（权重和）进行归一化。
        """
        p = torch.sigmoid(inputs)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        
        loss = bce_loss * ((1.0 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if weights is not None:
            loss = loss * weights

        if reduction == "mean":
            # 统计正样本有效数量
            if weights is not None:
                num_pos = (targets * weights).sum()
            else:
                num_pos = targets.sum()

            # 🎯 修复点：若有正样本，除以正样本数；若全无正样本，除以全图总元素数，防止梯度爆炸
            if num_pos > 0:
                return loss.sum() / num_pos
            else:
                return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
            
        return loss.mean()

    def _dice_loss_core(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """
        体素级软 Dice Loss 核心算子。
        
        Args:
            inputs: 预测 Logits [N, X, Y, Z] 或 [N, 1, X, Y, Z]
            targets: 二值真实掩码 [N, X, Y, Z] 或 [N, 1, X, Y, Z]
            weights: 置信度或空间权重遮蔽 [N, X, Y, Z] 或 [N, 1, X, Y, Z]
            eps: 平滑项，防止除零
            
        Returns:
            loss: 批次内平均的 Dice 损失标量
        """
        # 1. 映射为概率并展平空间维度 [N, V]
        probs = torch.sigmoid(inputs).flatten(1)
        targets = targets.flatten(1)

        # 2. 若传入置信度加权，对预测与真值施加空间遮蔽
        if weights is not None:
            weights = weights.flatten(1)
            probs = probs * weights
            targets = targets * weights

        target_sum = targets.sum(dim=1)  # [N]
        valid_mask = (target_sum > 0).float()  # 仅对存在正样本的目标/样本计算 Dice
        num_valid = valid_mask.sum()

        if num_valid == 0:
            # 🎯 修复点：若整批样本均无正样本，直接返回带计算图的 0.0
            return inputs.sum() * 0.0

        # 3. 逐实例计算交集与并集
        intersection = 2.0 * (probs * targets).sum(dim=1)
        cardinality = probs.sum(dim=1) + targets.sum(dim=1)

        dice_score = (intersection + eps) / (cardinality + eps)
        loss = 1.0 - dice_score
        return loss.mean()

    def _box_cxcyczsxsysz_to_xyzxyz(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        将 [..., 6] 格式的 (cx, cy, cz, sx, sy, sz) 转换为 (x1, y1, z1, x2, y2, z2)
        """
        center, size = boxes[..., :3], boxes[..., 3:]
        min_pt = center - 0.5 * size
        max_pt = center + 0.5 * size
        return torch.cat([min_pt, max_pt], dim=-1)

    def _giou_3d_core(self, boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        计算一一对应的两组 3D 边界框的 3D GIoU 损失 (1 - GIoU)。
        
        Args:
            boxes1: [N, 6] 预测框 (cx, cy, cz, sx, sy, sz)
            boxes2: [N, 6] 真实框 (cx, cy, cz, sx, sy, sz)
            eps: 防除零微小量
            
        Returns:
            loss: [N] 逐样本的 3D GIoU 损失 (取值范围 [0, 2])
        """
        # 确保尺寸非负
        boxes1_size = boxes1[:, 3:].clamp(min=0)
        boxes2_size = boxes2[:, 3:].clamp(min=0)

        b1_xyz = self._box_cxcyczsxsysz_to_xyzxyz(torch.cat([boxes1[:, :3], boxes1_size], dim=-1))
        b2_xyz = self._box_cxcyczsxsysz_to_xyzxyz(torch.cat([boxes2[:, :3], boxes2_size], dim=-1))

        # 1. 计算三维相交区域 (Intersection)
        lt = torch.max(b1_xyz[:, :3], b2_xyz[:, :3])  # [N, 3]
        rb = torch.min(b1_xyz[:, 3:], b2_xyz[:, 3:])  # [N, 3]
        inter_whd = (rb - lt).clamp(min=0)           # [N, 3]
        intersection = inter_whd[:, 0] * inter_whd[:, 1] * inter_whd[:, 2]  # [N]

        # 2. 计算体积与并集 (Union)
        vol1 = boxes1_size[:, 0] * boxes1_size[:, 1] * boxes1_size[:, 2]    # [N]
        vol2 = boxes2_size[:, 0] * boxes2_size[:, 1] * boxes2_size[:, 2]    # [N]
        union = vol1 + vol2 - intersection                                  # [N]

        iou = intersection / union.clamp(min=eps)

        # 3. 计算最小外接立方体 (Enclosing Box)
        enclosing_lt = torch.min(b1_xyz[:, :3], b2_xyz[:, :3])              # [N, 3]
        enclosing_rb = torch.max(b1_xyz[:, 3:], b2_xyz[:, 3:])              # [N, 3]
        enclosing_whd = (enclosing_rb - enclosing_lt).clamp(min=0)          # [N, 3]
        enclosing_vol = enclosing_whd[:, 0] * enclosing_whd[:, 1] * enclosing_whd[:, 2] # [N]

        # 4. 计算 3D GIoU 与损失
        giou = iou - (enclosing_vol - union) / enclosing_vol.clamp(min=eps)
        return 1.0 - giou  # [N]

@LOSS_LIB.register("QueryBCEClassificationLoss")
class QueryBCEClassificationLoss(CustomLoss):
    """
    基于标准二元交叉熵 (BCEWithLogitsLoss) 的 Query 分类损失类。
    适用于小 Query 数量 (如 Q=8) 下的实例存在性二分类监督。
    """
    def __init__(
        self,
        base_weight: float = 2.0,
        pos_weight: Optional[float] = None,
        schedule_type: str = "constant",
        max_steps: int = 1000,
        **kwargs: Any
    ):
        """
        参数:
            base_weight: 基础损失权重
            pos_weight: 正样本权重补偿因子 (如 2.0 表示正样本损失权重翻倍，None 为不补偿)
            schedule_type: 权重调度策略 ('constant', 'linear_warmup', 'cosine_decay')
            max_steps: 调度最大步数
        """
        super().__init__(
            base_weight=base_weight,
            schedule_type=schedule_type,
            max_steps=max_steps,
            **kwargs
        )
        self.name = "QueryBCEClassificationLoss"
        self.pos_weight = pos_weight


    def forward(
        self,
        cls_logits: torch.Tensor,
        matched_indices: List[Tuple[torch.Tensor, torch.Tensor]],
        return_dict: bool = False,
        **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        前向计算二元交叉熵损失。

        Args:
            cls_logits: 最后一层解算出的 Query 分类 Logits [B, Q] (若传入 [L, B, Q] 则自动取最后一层)
            matched_indices: 匈牙利匹配得出的 (src_idx, tgt_idx) 列表，长度为 B
            return_dict: 是否返回详细统计字典

        Returns:
            final_loss: 经过全局动态权重加权后的分类损失标量
        """ 
        if cls_logits.dim() == 3:
            cls_logits = cls_logits[-1]  # [B, Q]
        B, Q = cls_logits.shape
        device = cls_logits.device

        # 1. 构造二值分类标签 targets: [B, Q]
        targets = torch.zeros_like(cls_logits, dtype=torch.float32)
        
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(matched_indices)])
        src_idx = torch.cat([src for (src, _) in matched_indices])
        
        if len(batch_idx) > 0:
            targets[batch_idx, src_idx] = 1.0

        # 2. 正样本权重补偿 (可选)
        pos_weight_tensor = None
        if self.pos_weight is not None:
            pos_weight_tensor = torch.tensor([self.pos_weight], device=device, dtype=cls_logits.dtype)

        # 3. 计算 BCE 损失 (在 Batch 和 Query 维度上求 mean)
        # Reduction='none' 便于分别统计正负样本
        loss_matrix = F.binary_cross_entropy_with_logits(
            cls_logits, 
            targets, 
            pos_weight=pos_weight_tensor, 
            reduction="none"
        )  # [B, Q]

        # 4. 小 Query 场景下推荐直接在整个 [B, Q] 上求 mean
        raw_loss = loss_matrix.mean()

        # 5. 施加动态权重
        final_loss = raw_loss * self.current_weight

        if return_dict:
            with torch.no_grad():
                num_pos = len(src_idx)
                pos_loss = (loss_matrix * targets).sum() / max(num_pos, 1)
                neg_loss = (loss_matrix * (1.0 - targets)).sum() / max((B * Q - num_pos), 1)
                info_dict = {
                    "loss_cls_pos": (pos_loss * self.current_weight).detach(),
                    "loss_cls_neg": (neg_loss * self.current_weight).detach(),
                    "num_pos": torch.tensor(num_pos, device=device),
                }
            return final_loss, info_dict

        return final_loss

@LOSS_LIB.register("VoxelInstanceMaskFocalLoss")
class VoxelInstanceMaskFocalLoss(CustomLoss):
    """
    3D 体素实例掩码 (pred_masks: [B, Q, X, Y, Z]) 的置信度加权 Sigmoid Focal Loss。
    仅针对匈牙利匹配关联到的 (Query <-> GT) 正样本实例进行体素级监督。
    """
    def __init__(
        self,
        base_weight: float = 5.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        schedule_type: str = "constant",
        max_steps: int = 1000,
        **kwargs: Any
    ):
        super().__init__(
            base_weight=base_weight,
            schedule_type=schedule_type,
            max_steps=max_steps,
            **kwargs
        )
        self.name = "VoxelInstanceMaskFocalLoss"
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred_masks: torch.Tensor,                                     # [B, Q, X, Y, Z]
        gt_dict: Dict[str, torch.Tensor],
        matched_indices: List[Tuple[torch.Tensor, torch.Tensor]],
        return_dict: bool = False,
        **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        device = pred_masks.device
        gt_masks = gt_dict["gt_masks"]                       # [B, max_M, X, Y, Z]
        gt_mask_confs = gt_dict["gt_mask_confs"]             # [B, max_M, X, Y, Z]

        # 1. 提取所有匹配上的预测与真实掩码索引
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(matched_indices)])
        src_idx = torch.cat([src for (src, _) in matched_indices])
        tgt_idx = torch.cat([tgt for (_, tgt) in matched_indices])
        
        num_matched = len(src_idx)

        # 2. 实例级体素 Focal Loss 计算
        if num_matched > 0:
            matched_pred_masks = pred_masks[batch_idx, src_idx]            # [num_matched, X, Y, Z]
            matched_gt_masks = gt_masks[batch_idx, tgt_idx]                # [num_matched, X, Y, Z]
            matched_confs = gt_mask_confs[batch_idx, tgt_idx]              # [num_matched, X, Y, Z]

            raw_loss = self._focal_loss_core(
                inputs=matched_pred_masks,
                targets=matched_gt_masks,
                weights=matched_confs,
                alpha=self.alpha,
                gamma=self.gamma,
                reduction="mean",
            )
        else:
            raw_loss = pred_masks.sum() * 0.0

        # 3. 施加动态调度权重
        final_loss = raw_loss * self.current_weight

        if return_dict:
            with torch.no_grad():
                info_dict = {
                    "raw_mask_focal_loss": raw_loss.detach(),
                    "num_matched_masks": torch.tensor(num_matched, device=device),
                }
            return final_loss, info_dict

        return final_loss

@LOSS_LIB.register("VoxelSemanticOccupancyFocalLoss")
class VoxelSemanticOccupancyFocalLoss(CustomLoss):
    """
    3D 全局体素占据 (semantic_seg: [B, 1, X, Y, Z]) 的 Sigmoid Focal Loss 辅助损失。
    无需匈牙利匹配，以全图所有有效实例掩码并集作为二值真值计算全局背景与前景占据。
    """
    def __init__(
        self,
        base_weight: float = 2.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        schedule_type: str = "constant",
        max_steps: int = 1000,
        **kwargs: Any
    ):
        super().__init__(
            base_weight=base_weight,
            schedule_type=schedule_type,
            max_steps=max_steps,
            **kwargs
        )
        self.name = "VoxelSemanticOccupancyFocalLoss"
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        semantic_seg: torch.Tensor,                                   # [B, 1, X, Y, Z]
        gt_dict: Dict[str, torch.Tensor],
        return_dict: bool = False,
        **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if semantic_seg is None:
            device = gt_dict["gt_masks"].device
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return (zero_loss, {}) if return_dict else zero_loss

        device = semantic_seg.device
        gt_masks = gt_dict["gt_masks"]                       # [B, max_M, X, Y, Z]
        gt_valid = gt_dict["gt_instance_valid_mask"]         # [B, max_M] (bool)

        # 1. 构造 GT 全局前景：所有有效实例掩码的并集 [B, 1, X, Y, Z]
        valid_masks = gt_masks * gt_valid[..., None, None, None].float()
        gt_sem = valid_masks.sum(dim=1, keepdim=True).clamp(max=1.0)

        # 2. 复用基类 _focal_loss_core 算子
        raw_loss = self._focal_loss_core(
            inputs=semantic_seg,
            targets=gt_sem,
            weights=None,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction="mean",
        )

        # 3. 施加动态调度权重
        final_loss = raw_loss * self.current_weight

        if return_dict:
            with torch.no_grad():
                info_dict = {
                    "raw_semantic_focal_loss": raw_loss.detach(),
                    "occupancy_ratio": gt_sem.mean().detach(),
                }
            return final_loss, info_dict

        return final_loss

@LOSS_LIB.register("VoxelInstanceMaskDiceLoss")
class VoxelInstanceMaskDiceLoss(CustomLoss):
    """
    3D 体素实例掩码 (pred_masks: [B, Q, X, Y, Z]) 的置信度加权 Dice Loss。
    仅针对匈牙利匹配关联到的 (Query <-> GT) 正样本实例进行体素重合度监督。
    """
    def __init__(
        self,
        base_weight: float = 5.0,
        eps: float = 1e-5,
        schedule_type: str = "constant",
        max_steps: int = 1000,
        **kwargs: Any
    ):
        super().__init__(
            base_weight=base_weight,
            schedule_type=schedule_type,
            max_steps=max_steps,
            **kwargs
        )
        self.name = "VoxelInstanceMaskDiceLoss"
        self.eps = eps

    def forward(
        self,
        pred_masks: torch.Tensor,                                     # [B, Q, X, Y, Z]
        gt_dict: Dict[str, torch.Tensor],
        matched_indices: List[Tuple[torch.Tensor, torch.Tensor]],
        return_dict: bool = False,
        **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        device = pred_masks.device
        gt_masks = gt_dict["gt_masks"]                       # [B, max_M, X, Y, Z]
        gt_mask_confs = gt_dict["gt_mask_confs"]             # [B, max_M, X, Y, Z]

        # 1. 提取所有匹配成功的预测与真实掩码索引
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(matched_indices)])
        src_idx = torch.cat([src for (src, _) in matched_indices])
        tgt_idx = torch.cat([tgt for (_, tgt) in matched_indices])
        
        num_matched = len(src_idx)

        # 2. 实例级体素 Dice Loss 计算
        if num_matched > 0:
            matched_pred_masks = pred_masks[batch_idx, src_idx]            # [num_matched, X, Y, Z]
            matched_gt_masks = gt_masks[batch_idx, tgt_idx]                # [num_matched, X, Y, Z]
            matched_confs = gt_mask_confs[batch_idx, tgt_idx]              # [num_matched, X, Y, Z]

            raw_loss = self._dice_loss_core(
                inputs=matched_pred_masks,
                targets=matched_gt_masks,
                weights=matched_confs,
                eps=self.eps,
            )
        else:
            raw_loss = pred_masks.sum() * 0.0

        # 3. 施加动态调度权重
        final_loss = raw_loss * self.current_weight

        if return_dict:
            with torch.no_grad():
                info_dict = {
                    "raw_mask_dice_loss": raw_loss.detach(),
                    "num_matched_masks": torch.tensor(num_matched, device=device),
                }
            return final_loss, info_dict

        return final_loss

@LOSS_LIB.register("VoxelSemanticOccupancyDiceLoss")
class VoxelSemanticOccupancyDiceLoss(CustomLoss):
    """
    3D 全局体素占据 (semantic_seg: [B, 1, X, Y, Z]) 的 Dice Loss 辅助损失。
    无需匈牙利匹配，直接约束全图预测占据网格与真实前景体素并集的重合度。
    """
    def __init__(
        self,
        base_weight: float = 2.0,
        eps: float = 1e-5,
        schedule_type: str = "constant",
        max_steps: int = 1000,
        **kwargs: Any
    ):
        super().__init__(
            base_weight=base_weight,
            schedule_type=schedule_type,
            max_steps=max_steps,
            **kwargs
        )
        self.name = "VoxelSemanticOccupancyDiceLoss"
        self.eps = eps

    def forward(
        self,
        semantic_seg: torch.Tensor,                                   # [B, 1, X, Y, Z]
        gt_dict: Dict[str, torch.Tensor],
        return_dict: bool = False,
        **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if semantic_seg is None:
            device = gt_dict["gt_masks"].device
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return (zero_loss, {}) if return_dict else zero_loss

        device = semantic_seg.device
        gt_masks = gt_dict["gt_masks"]                       # [B, max_M, X, Y, Z]
        gt_valid = gt_dict["gt_instance_valid_mask"]         # [B, max_M] (bool)

        # 1. 构造 GT 全局前景：所有有效实例掩码的并集 [B, 1, X, Y, Z]
        valid_masks = gt_masks * gt_valid[..., None, None, None].float()
        gt_sem = valid_masks.sum(dim=1, keepdim=True).clamp(max=1.0)

        # 2. 复用基类 _dice_loss_core 算子
        raw_loss = self._dice_loss_core(
            inputs=semantic_seg,
            targets=gt_sem,
            weights=None,
            eps=self.eps,
        )

        # 3. 施加动态调度权重
        final_loss = raw_loss * self.current_weight

        if return_dict:
            with torch.no_grad():
                info_dict = {
                    "raw_semantic_dice_loss": raw_loss.detach(),
                    "occupancy_ratio": gt_sem.mean().detach(),
                }
            return final_loss, info_dict

        return final_loss

@LOSS_LIB.register("LayerWiseBBoxL1Loss")
class LayerWiseBBoxL1Loss(CustomLoss):
    """
    基于全解码层级 (pred_boxes: [L, B, Q, 6]) 的 3D Bounding Box L1 坐标回归损失类。
    
    设计特性:
    1. 继承 CustomLoss 基础调度机制 (constant, linear_warmup, cosine_decay)；
    2. 所有 L 层共享最终层的匈牙利匹配结果 (matched_indices)；
    3. 支持各解码层倒金字塔/自定义辅助权重加权 (layer_weights)；
    4. 以正样本总数归一化，具备无目标场景防除零与梯度保护机制。
    """
    def __init__(
        self,
        base_weight: float = 5.0,
        layer_weights: List[float] = [0.25, 0.5, 1.0],
        loss_type: str = "l1",  # "l1" 或 "smooth_l1"
        beta: float = 1.0 / 9.0, # smooth_l1 的阈值参数
        schedule_type: str = "constant",
        max_steps: int = 1000,
        **kwargs: Any
    ):
        super().__init__(
            base_weight=base_weight,
            schedule_type=schedule_type,
            max_steps=max_steps,
            **kwargs
        )
        self.name = "LayerWiseBBoxL1Loss"
        self.layer_weights = layer_weights
        self.loss_type = loss_type
        self.beta = beta

    def forward(
        self,
        pred_boxes: torch.Tensor,                                     # [L, B, Q, 6] 或 [B, Q, 6]
        gt_dict: Dict[str, torch.Tensor],
        matched_indices: List[Tuple[torch.Tensor, torch.Tensor]],
        return_dict: bool = False,
        **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        前向计算逐层 3D 边界框 L1 坐标回归损失。

        Args:
            pred_boxes: 各解码层预测的 3D 边界框 [L, B, Q, 6] 或单层 [B, Q, 6]
                * 格式: (cx, cy, cz, sx, sy, sz) -> (中心点X, 中心点Y, 中心点Z, 尺寸DX, 尺寸DY, 尺寸DZ)
                * 坐标范围: 归一化至 [0.0, 1.0] 区间（通过 Sigmoid 激活输出）
                * 空间映射: 0.0 和 1.0 分别对应点云体素感兴趣区域 (ROI) 的 min_bound 与 max_bound (XLIM, YLIM, ZLIM)
            gt_dict: 真实标签字典，需包含：
                - gt_bboxes: [B, max_M, 6]，真实 3D 边界框
                    * 格式: (cx, cy, cz, sx, sy, sz)，与 pred_boxes 严格对应
                    * 坐标范围: 归一化至 [0.0, 1.0] 区间（数据预处理阶段执行 (coord - min_bound) / range 归一化）
                - gt_instance_valid_mask: [B, max_M] (bool)，有效 GT 实例标记
            matched_indices: 匈牙利匹配索引列表，长度为 B。每个元素为 (src_idx, tgt_idx) 元组:
                - src_idx: 当前样本中命中目标的模型预测 Query 索引 [num_pos] (范围 [0, Q-1])
                - tgt_idx: 对应的有效 Ground Truth 实例索引 [num_pos] (范围 [0, max_M-1])
            return_dict: 是否返回包含各层详细损失及正样本统计的字典

        Returns:
            final_loss: 经过层级加权和动态调度后的全局 L1 损失标量张量
            info_dict (可选): 包含 raw_bbox_l1_loss、正样本数以及各层分项损失的字典
        """
        # 兼容单层输入维度
        if pred_boxes.dim() == 3:
            pred_boxes = pred_boxes.unsqueeze(0)  # [1, B, Q, 6]

        num_layers, B, Q, _ = pred_boxes.shape
        device = pred_boxes.device
        gt_bboxes = gt_dict["gt_bboxes"]  # [B, max_M, 6]

        # 1. 提取所有匹配成功的索引
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(matched_indices)])
        src_idx = torch.cat([src for (src, _) in matched_indices])
        tgt_idx = torch.cat([tgt for (_, tgt) in matched_indices])
        
        num_pos = len(src_idx)
        normalizer = max(float(num_pos), 1.0)

        # 2. 容错处理：确保层权重数量与实际输入层数一致
        actual_weights = self.layer_weights
        if len(actual_weights) != num_layers:
            actual_weights = [1.0] * num_layers

        total_loss = 0.0
        weight_sum = 0.0
        layer_losses_dict = {}

        # 3. 逐层计算正样本的 Bbox L1 回归损失
        if num_pos > 0:
            target_boxes = gt_bboxes[batch_idx, tgt_idx]  # [num_pos, 6]

            for lvl in range(num_layers):
                layer_pred_boxes = pred_boxes[lvl][batch_idx, src_idx]  # [num_pos, 6]

                if self.loss_type == "smooth_l1":
                    loss_matrix = F.smooth_l1_loss(
                        layer_pred_boxes, target_boxes, beta=self.beta, reduction="none"
                    )
                else:
                    loss_matrix = F.l1_loss(
                        layer_pred_boxes, target_boxes, reduction="none"
                    )

                # 对 6 个维度求和后在正样本上求平均
                lvl_loss = loss_matrix.sum() / normalizer

                w = actual_weights[lvl]
                total_loss += lvl_loss * w
                weight_sum += w

                layer_losses_dict[f"layer_{lvl}_l1"] = (lvl_loss * self.current_weight).detach()
        else:
            total_loss = pred_boxes.sum() * 0.0
            weight_sum = 1.0

        # 4. 加权归一化并施加动态调度权重
        avg_weighted_loss = total_loss / max(weight_sum, 1e-6)
        final_loss = avg_weighted_loss * self.current_weight

        if return_dict:
            with torch.no_grad():
                info_dict = {
                    "raw_bbox_l1_loss": avg_weighted_loss.detach(),
                    "num_pos": torch.tensor(num_pos, device=device),
                    **layer_losses_dict
                }
            return final_loss, info_dict

        return final_loss

@LOSS_LIB.register("LayerWiseBBoxGIoULoss")
class LayerWiseBBoxGIoULoss(CustomLoss):
    """
    基于全解码层级 (pred_boxes: [L, B, Q, 6]) 的 3D Bounding Box Generalized IoU (GIoU) 损失类。
    
    设计特性:
    1. 继承 CustomLoss 基础调度机制 (constant, linear_warmup, cosine_decay)；
    2. 全层共享最终层匈牙利匹配索引 (matched_indices)；
    3. 支持各解码层倒金字塔/自定义辅助权重加权 (layer_weights)；
    4. 采用正样本总数归一化，具备极端稀疏/空目标保护。
    """
    def __init__(
        self,
        base_weight: float = 2.0,
        layer_weights: List[float] = [0.25, 0.5, 1.0],
        schedule_type: str = "constant",
        max_steps: int = 1000,
        eps: float = 1e-7,
        **kwargs: Any
    ):
        super().__init__(
            base_weight=base_weight,
            schedule_type=schedule_type,
            max_steps=max_steps,
            **kwargs
        )
        self.name = "LayerWiseBBoxGIoULoss"
        self.layer_weights = layer_weights
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,                                     # [L, B, Q, 6] 或 [B, Q, 6]
        gt_dict: Dict[str, torch.Tensor],
        matched_indices: List[Tuple[torch.Tensor, torch.Tensor]],
        return_dict: bool = False,
        **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        前向计算逐层 3D 边界框 GIoU 损失。

        Args:
            pred_boxes: 各解码层预测的 3D 边界框 [L, B, Q, 6] 或单层 [B, Q, 6]
                * 格式: (cx, cy, cz, sx, sy, sz) -> (中心点X, 中心点Y, 中心点Z, 尺寸DX, 尺寸DY, 尺寸DZ)
                * 坐标范围: 归一化至 [0.0, 1.0] 区间（通过 Sigmoid 激活输出）
                * 空间映射: 0.0 和 1.0 分别对应点云体素感兴趣区域 (ROI) 的 min_bound 与 max_bound (XLIM, YLIM, ZLIM)
            gt_dict: 真实标签字典，需包含：
                - gt_bboxes: [B, max_M, 6]，真实 3D 边界框
                    * 格式: (cx, cy, cz, sx, sy, sz)，与 pred_boxes 严格对应
                    * 坐标范围: 归一化至 [0.0, 1.0] 区间（数据预处理阶段执行 (coord - min_bound) / range 归一化）
                - gt_instance_valid_mask: [B, max_M] (bool)，有效 GT 实例标记
            matched_indices: 匈牙利匹配索引列表，长度为 B。每个元素为 (src_idx, tgt_idx) 元组:
                - src_idx: 当前样本中命中目标的模型预测 Query 索引 [num_pos] (范围 [0, Q-1])
                - tgt_idx: 对应的有效 Ground Truth 实例索引 [num_pos] (范围 [0, max_M-1])
            return_dict: 是否返回包含各层详细损失及正样本统计的字典

        Returns:
            final_loss: 经过层级加权和动态调度后的全局 3D GIoU 损失标量张量
            info_dict (可选): 包含 raw_bbox_giou_loss、正样本数以及各层分项损失的字典
        """
        if pred_boxes.dim() == 3:
            pred_boxes = pred_boxes.unsqueeze(0)  # [1, B, Q, 6]

        num_layers, B, Q, _ = pred_boxes.shape
        device = pred_boxes.device
        gt_bboxes = gt_dict["gt_bboxes"]  # [B, max_M, 6]

        # 1. 提取所有匹配成功的正样本索引
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(matched_indices)])
        src_idx = torch.cat([src for (src, _) in matched_indices])
        tgt_idx = torch.cat([tgt for (_, tgt) in matched_indices])
        
        num_pos = len(src_idx)
        normalizer = max(float(num_pos), 1.0)

        # 2. 校验层级加权配置
        actual_weights = self.layer_weights
        if len(actual_weights) != num_layers:
            actual_weights = [1.0] * num_layers

        total_loss = 0.0
        weight_sum = 0.0
        layer_losses_dict = {}

        # 3. 逐层计算正样本 3D GIoU 损失
        if num_pos > 0:
            target_boxes = gt_bboxes[batch_idx, tgt_idx]  # [num_pos, 6]

            for lvl in range(num_layers):
                layer_pred_boxes = pred_boxes[lvl][batch_idx, src_idx]  # [num_pos, 6]

                # 计算该层所有匹配正样本的 3D GIoU 损失: [num_pos]
                giou_loss_vec = self._giou_3d_core(layer_pred_boxes, target_boxes, eps=self.eps)
                lvl_loss = giou_loss_vec.sum() / normalizer

                w = actual_weights[lvl]
                total_loss += lvl_loss * w
                weight_sum += w

                layer_losses_dict[f"layer_{lvl}_giou"] = (lvl_loss * self.current_weight).detach()
        else:
            total_loss = pred_boxes.sum() * 0.0
            weight_sum = 1.0

        # 4. 加权平均并施加动态调度权重
        avg_weighted_loss = total_loss / max(weight_sum, 1e-6)
        final_loss = avg_weighted_loss * self.current_weight

        if return_dict:
            with torch.no_grad():
                info_dict = {
                    "raw_bbox_giou_loss": avg_weighted_loss.detach(),
                    "num_pos": torch.tensor(num_pos, device=device),
                    **layer_losses_dict
                }
            return final_loss, info_dict

        return final_loss

# 