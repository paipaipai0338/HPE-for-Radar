"""
记录与 pose 相关的指标
"""
import sys
import os
import torch
import numpy as np
from typing import Tuple
import torch.nn.functional as F

from utils.COCO import COCO_SKELETON

def get_bce(confidence: torch.tensor, gt_mask: torch.tensor, eps: float=1e-3) -> torch.tensor:
    '''由于模型人数不定，预先给出最大估计人数 max_people，人数需要依赖模型输出的confidence判决'''
    if confidence.shape != gt_mask.shape:
        raise ValueError(
            f"confidence and gt_mask must have same shape, "
            f"got confidence={tuple(confidence.shape)}, gt_mask={tuple(gt_mask.shape)}"
        )
    if confidence.ndim != 3:
        raise ValueError(
            f"confidence must have shape [B, T, K], got {tuple(confidence.shape)}"
        )
    gt_mask = gt_mask.to(dtype=confidence.dtype, device=confidence.device)

    confidence = torch.clamp(confidence, eps, 1 - eps)
    BCE = F.binary_cross_entropy(confidence, gt_mask, reduction='none')
    return BCE

def get_detection_metric(confidence: torch.tensor, gt_mask: torch.tensor, ratio: float=0.5, eps: float=1e-12) -> torch.tensor:
    '''判断模型识别人体存在的准确率'''
    # B, T, K
    confidence = confidence.float()
    gt_mask = gt_mask.float()
    
    pred_mask = (confidence >= ratio).float()
    
    TP = torch.sum((pred_mask == 1) & (gt_mask == 1)).float()  # 预测有人，实际有人
    FP = torch.sum((pred_mask == 1) & (gt_mask == 0)).float()  # 预测有人，实际无人（误检）
    FN = torch.sum((pred_mask == 0) & (gt_mask == 1)).float()  # 预测无人，实际有人（漏检） -> 补全这里
    TN = torch.sum((pred_mask == 0) & (gt_mask == 0)).float()  # 预测无人，实际无人（虽然少用，但一并算出）
    
    total = TP + FP + FN + TN
    
    accuracy = (TP + TN) / (total + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    return accuracy, precision, recall, f1

def get_mpjpe(pre: torch.tensor, gt:torch.tensor, type:str='coco') -> torch.tensor:
    assert pre.shape == gt.shape, 'pre and gt do not have same shape'
    common_shape = gt.shape
    if type.lower() == 'coco':
        num_joint = 17
    assert pre.shape[-2:] == (num_joint, 3), 'pre has wrong shape, epxected ({}, 3), but got {}'.format(num_joint, pre.shape)
    assert gt.shape[-2:] == (num_joint, 3), 'gt has wrong shape, epxected ({}, 3), but got {}'.format(num_joint, gt.shape)

    mpjpe = torch.mean(torch.norm(pre - gt, dim=-1), dim=-1)
    assert mpjpe.shape == common_shape[:-2], 'the finnal results has wrong shape' 
    return mpjpe

def get_pampjpe(
    pre: torch.Tensor,
    gt: torch.Tensor,
    type: str='coco',
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算经过无反射相似变换对齐后的 MPJPE。

    求解 ``min_{s, Q, t} ||s * pre @ Q + t - gt||_F``，其中
    ``Q`` 是行列式为 1 的旋转矩阵。对齐参数使用 detached tensor
    计算：这不改变前向指标，并避免 SVD 在重复奇异值或退化骨架处产生
    NaN 梯度；梯度仍会通过最终的仿射变换传回 ``pre``。
    """
    assert pre.shape == gt.shape, 'pre and gt do not have same shape'
    common_shape = gt.shape
    if type.lower() == 'coco':
        num_joint = 17
    assert pre.shape[-2:] == (num_joint, 3), 'pre has wrong shape, epxected ({}, 3), but got {}'.format(num_joint, pre.shape)
    assert gt.shape[-2:] == (num_joint, 3), 'gt has wrong shape, epxected ({}, 3), but got {}'.format(num_joint, gt.shape)

    if eps <= 0:
        raise ValueError(f'eps must be positive, got {eps}')

    if not torch.isfinite(pre).all() or not torch.isfinite(gt).all():
        raise ValueError('pre and gt must contain only finite values')

    pre_mean = pre.mean(dim=-2, keepdim=True)
    gt_mean = gt.mean(dim=-2, keepdim=True)

    # SVD 的导数在重复奇异值处没有良好定义。最优对齐参数无需参与
    # autograd；最终 pre_aligned 仍然保留到 pre 的梯度路径。
    with torch.no_grad():
        pre_centered = pre.detach() - pre_mean.detach()
        gt_centered = gt.detach() - gt_mean.detach()
        covariance = pre_centered.transpose(-1, -2) @ gt_centered

        # torch.linalg.svd 返回 U, S, Vh，而不是 V。
        U, singular_values, Vh = torch.linalg.svd(covariance)

        # 对行向量约定，最优旋转为 Q = U @ D @ Vh。
        # D 的最后一个元素负责排除镜像反射。
        orientation = torch.det(U @ Vh)
        correction = torch.ones_like(singular_values)
        correction[..., -1] = torch.where(
            orientation < 0,
            -torch.ones_like(orientation),
            torch.ones_like(orientation),
        )
        rotation = (U * correction.unsqueeze(-2)) @ Vh

        variance = pre_centered.square().sum(dim=(-1, -2))
        scale_numerator = (singular_values * correction).sum(dim=-1)
        scale = torch.where(
            variance > eps,
            scale_numerator / variance.clamp_min(eps),
            torch.zeros_like(variance),
        )

    scale = scale.unsqueeze(-1).unsqueeze(-1)
    pre_aligned = scale * ((pre - pre_mean) @ rotation) + gt_mean

    pampjpe = get_mpjpe(pre=pre_aligned, gt=gt, type=type)

    assert pampjpe.shape == common_shape[:-2], 'the finnal results has wrong shape' 
    return pampjpe

def get_bone_length(pre: torch.tensor, gt:torch.tensor, type:str, return_all=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert pre.shape == gt.shape, 'pre and gt do not have same shape'
    common_shape = gt.shape
    if type.lower() == 'coco':
        num_joint = 17
        skeleton = COCO_SKELETON
    assert pre.shape[-2:] == (num_joint, 3), 'pre has wrong shape, epxected ({}, 3), but got {}'.format(num_joint, pre.shape)
    assert gt.shape[-2:] == (num_joint, 3), 'gt has wrong shape, epxected ({}, 3), but got {}'.format(num_joint, gt.shape)
    pre_bone_length, gt_bone_length = [], []
    for s in skeleton:
        start, end = s
        pre_bone_length.append(torch.norm(pre[..., start, :] - pre[..., end, :], dim=-1))
        gt_bone_length.append(torch.norm(gt[..., start, :] - gt[..., end, :], dim=-1))
    pre_bone_length = torch.stack(pre_bone_length, dim=-1)
    gt_bone_length = torch.stack(gt_bone_length, dim=-1)
    
    bone_length_error = torch.norm(pre_bone_length - gt_bone_length, dim=-1)

    assert bone_length_error.shape == common_shape[:-2], 'the finnal results has wrong shape' 
    if return_all:
        return pre_bone_length, gt_bone_length, bone_length_error
    else:
        return bone_length_error


if __name__ == '__main__':
    pre = torch.rand((1, 10, 4, 17, 3))
    gt = torch.rand_like(pre)
    confidence = torch.rand((1, 10, 4))
    gt_mask = torch.rand_like(confidence)
    mpjpe = get_mpjpe(pre, gt, type='COco')
    print('mpjpe', mpjpe.shape)
    pampjpe = get_pampjpe(pre, gt, type='COco')
    print('pampjpe', pampjpe.shape)
    bone_length = get_bone_length(pre, gt, type='COco')
    print('bone_length', bone_length.shape)
    bce = get_bce(confidence, gt_mask)
    print('bce', bce.shape)
