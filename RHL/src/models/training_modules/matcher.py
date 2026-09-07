import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Any, Optional


def box_cxcyczsxsysz_to_xyzxyz(boxes: torch.Tensor) -> torch.Tensor:
    """
    将 [..., 6] 格式的 (cx, cy, cz, sx, sy, sz) 转换为 (x1, y1, z1, x2, y2, z2)
    """
    center, size = boxes[..., :3], boxes[..., 3:]
    min_pt = center - 0.5 * size
    max_pt = center + 0.5 * size
    return torch.cat([min_pt, max_pt], dim=-1)


def batch_giou_3d_cost(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    计算两组 3D 边界框之间的成对 3D GIoU 代价矩阵 (-GIoU)。
    
    Args:
        boxes1: [Q, 6] 预测框 (cx, cy, cz, sx, sy, sz)
        boxes2: [M, 6] 真实框 (cx, cy, cz, sx, sy, sz)
        
    Returns:
        cost_giou: [Q, M] 代价矩阵 (范围 [-1, 1])
    """
    Q = boxes1.size(0)
    M = boxes2.size(0)

    boxes1_size = boxes1[:, 3:].clamp(min=0)
    boxes2_size = boxes2[:, 3:].clamp(min=0)

    b1_xyz = box_cxcyczsxsysz_to_xyzxyz(torch.cat([boxes1[:, :3], boxes1_size], dim=-1))  # [Q, 6]
    b2_xyz = box_cxcyczsxsysz_to_xyzxyz(torch.cat([boxes2[:, :3], boxes2_size], dim=-1))  # [M, 6]

    # 1. 广播扩展维度 [Q, M, 3]
    lt = torch.max(b1_xyz[:, None, :3], b2_xyz[None, :, :3])
    rb = torch.min(b1_xyz[:, None, 3:], b2_xyz[None, :, 3:])
    inter_whd = (rb - lt).clamp(min=0)
    intersection = inter_whd[..., 0] * inter_whd[..., 1] * inter_whd[..., 2]  # [Q, M]

    # 2. 计算体积与并集
    vol1 = boxes1_size[:, 0] * boxes1_size[:, 1] * boxes1_size[:, 2]  # [Q]
    vol2 = boxes2_size[:, 0] * boxes2_size[:, 1] * boxes2_size[:, 2]  # [M]
    union = vol1[:, None] + vol2[None, :] - intersection              # [Q, M]

    iou = intersection / union.clamp(min=eps)

    # 3. 最小外接立方体
    enclosing_lt = torch.min(b1_xyz[:, None, :3], b2_xyz[None, :, :3])
    enclosing_rb = torch.max(b1_xyz[:, None, 3:], b2_xyz[None, :, 3:])
    enclosing_whd = (enclosing_rb - enclosing_lt).clamp(min=0)
    enclosing_vol = enclosing_whd[..., 0] * enclosing_whd[..., 1] * enclosing_whd[..., 2]

    # 4. GIoU 计算
    giou = iou - (enclosing_vol - union) / enclosing_vol.clamp(min=eps)
    return -giou  # [Q, M]，值越小匹配度越高

class Hungarian3DMatcher(nn.Module):
    """
    针对 3D 毫米波人体感知的匈牙利二分图匹配器。
    
    重点关注 3D Bbox 定位精度与存在性分类，对掩码重合度赋予低权重/零权重。
    """
    def __init__(
        self,
        cost_cls: float = 1.0,
        cost_bbox_l1: float = 5.0,
        cost_bbox_giou: float = 2.0,
        cost_mask_dice: float = 0.0,
        **kwargs: Any
    ):
        super().__init__()
        self.cost_cls = cost_cls
        self.cost_bbox_l1 = cost_bbox_l1
        self.cost_bbox_giou = cost_bbox_giou
        self.cost_mask_dice = cost_mask_dice

    @torch.no_grad()
    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        gt_dict: Dict[str, torch.Tensor]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向求解 Batch 内每个样本的二分图最佳匹配 (Hungarian Bipartite Matching)。

        Args:
            pred_dict: 模型输出字典，包含：
                - cls_logits: [B, Q] 或 [L, B, Q]，预测的存在性分类 Logits
                - pred_boxes: [B, Q, 6] 或 [L, B, Q, 6]，预测的 3D 边界框
                    * 格式: (cx, cy, cz, sx, sy, sz) -> (中心点X, 中心点Y, 中心点Z, 尺寸DX, 尺寸DY, 尺寸DZ)
                    * 坐标范围: 严格归一化至 [0.0, 1.0] 区间（通过 Sigmoid 激活）
                    * 映射关系: 0.0 和 1.0 分别对应点云体素感兴趣区域 (ROI) 的 min_bound 与 max_bound (XLIM, YLIM, ZLIM)
                - pred_masks: [B, Q, X, Y, Z] (可选，若 cost_mask_dice > 0 时使用)，预测的体素占据 Logits
            gt_dict: 真实标签字典，包含：
                - gt_bboxes: [B, max_M, 6]，真实的 3D 边界框
                    * 格式: (cx, cy, cz, sx, sy, sz)，与 pred_boxes 严格一致
                    * 坐标范围: 归一化至 [0.0, 1.0] 区间（数据加载预处理时完成 (pos - min_bound) / range 归一化）
                - gt_masks: [B, max_M, X, Y, Z] (可选)，真实的 3D 体素二值掩码 (0.0 或 1.0)
                - gt_mask_confs: [B, max_M, X, Y, Z] (可选)，体素置信度权重 (0.0 ~ 1.0)
                - gt_instance_valid_mask: [B, max_M] (bool)，标记当前填充实例是否为有效 GT 目标

        Returns:
            matched_indices: 长度为 B 的列表，每个元素为 (src_idx, tgt_idx) 形式的 1D Tensor (dtype=torch.int64) 元组:
                - src_idx: 当前样本中命中目标的模型预测 Query 索引 [M_valid] (范围 [0, Q-1])
                - tgt_idx: 对应的有效 Ground Truth 实例索引 [M_valid] (范围 [0, max_M-1])
        """
        cls_logits = pred_dict["cls_logits"]
        pred_boxes = pred_dict["pred_boxes"]

        # 统一取最后一层的输出进行匹配
        if cls_logits.dim() == 3:
            cls_logits = cls_logits[-1]  # [B, Q]
        if pred_boxes.dim() == 4:
            pred_boxes = pred_boxes[-1]  # [B, Q, 6]

        B, Q = cls_logits.shape
        device = cls_logits.device

        gt_bboxes = gt_dict["gt_bboxes"]                      # [B, max_M, 6]
        gt_valid = gt_dict["gt_instance_valid_mask"]          # [B, max_M]

        matched_indices = []

        # 逐 Batch 样本独立计算匹配
        for b in range(B):
            valid_mask = gt_valid[b]  # [max_M]
            num_valid = valid_mask.sum().item()

            # 极端情况：当前样本没有任何真实目标
            if num_valid == 0:
                matched_indices.append((
                    torch.empty(0, dtype=torch.int64, device=device),
                    torch.empty(0, dtype=torch.int64, device=device)
                ))
                continue

            # 提取有效目标的真值
            valid_tgt_indices = torch.where(valid_mask)[0]     # [M]
            tgt_boxes = gt_bboxes[b, valid_tgt_indices]        # [M, 6]

            sample_cls = cls_logits[b].sigmoid()               # [Q]
            sample_boxes = pred_boxes[b]                       # [Q, 6]

            # 1. 分类代价: -p (倾向于挑选概率大的 Query)
            cost_cls = -sample_cls[:, None].repeat(1, num_valid)  # [Q, M]

            # 2. 3D Bbox L1 坐标代价: torch.cdist 计算两两曼哈顿距离
            cost_l1 = torch.cdist(sample_boxes, tgt_boxes, p=1)  # [Q, M]

            # 3. 3D GIoU 代价: [Q, M]
            cost_giou = batch_giou_3d_cost(sample_boxes, tgt_boxes)

            # 4. (可选) 掩码形态 Dice 代价
            cost_dice = torch.zeros_like(cost_l1)
            if self.cost_mask_dice > 0 and "pred_masks" in pred_dict and "gt_masks" in gt_dict:
                sample_masks = pred_dict["pred_masks"][b].sigmoid().flatten(1)  # [Q, V]
                tgt_masks = gt_dict["gt_masks"][b, valid_tgt_indices].float().flatten(1)  # [M, V]

                # 矩阵相乘高效求解两两交集
                intersection = 2.0 * torch.matmul(sample_masks, tgt_masks.t())  # [Q, M]
                cardinality = sample_masks.sum(dim=1, keepdim=True) + tgt_masks.sum(dim=1, keepdim=True).t()
                cost_dice = 1.0 - (intersection + 1e-5) / (cardinality + 1e-5)

            # 5. 代价加权合成
            total_cost = (
                self.cost_cls * cost_cls +
                self.cost_bbox_l1 * cost_l1 +
                self.cost_bbox_giou * cost_giou +
                self.cost_mask_dice * cost_dice
            )

            # 6. 调用 scipy 的匈牙利匹配求解全局代价最小二分图
            total_cost_np = total_cost.detach().cpu().numpy()
            src_ind, tgt_ind = linear_sum_assignment(total_cost_np)

            # 将匹配上的 tgt_ind 映射回在原 gt_dict 中的真实位置
            matched_src = torch.as_tensor(src_ind, dtype=torch.int64, device=device)
            matched_tgt = valid_tgt_indices[tgt_ind]

            matched_indices.append((matched_src, matched_tgt))

        return matched_indices