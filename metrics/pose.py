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

def get_BCE(confidence: torch.tensor, gt_mask: torch.tensor, eps: float=1e-12) -> torch.tensor:
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
    

def get_mpjpe(pre: torch.tensor, gt:torch.tensor, type:str) -> torch.tensor:
    assert pre.shape == gt.shape, 'pre and gt do not have same shape'
    common_shape = gt.shape
    if type.lower() == 'coco':
        num_joint = 17
    assert pre.shape[-2:] == (num_joint, 3), 'pre has wrong shape, epxected ({}, 3), but got {}'.format(num_joint, pre.shape)
    assert gt.shape[-2:] == (num_joint, 3), 'gt has wrong shape, epxected ({}, 3), but got {}'.format(num_joint, gt.shape)

    mpjpe = torch.mean(torch.norm(pre - gt, dim=-1), dim=-1)
    assert mpjpe.shape == common_shape[:-2], 'the finnal results has wrong shape' 
    return mpjpe

def get_pampjpe(pre: torch.tensor, gt:torch.tensor, type:str) -> torch.tensor:
    '''min_{s,R,t} || s*R*pre+t - gt||_{F}'''
    assert pre.shape == gt.shape, 'pre and gt do not have same shape'
    common_shape = gt.shape
    if type.lower() == 'coco':
        num_joint = 17
    assert pre.shape[-2:] == (num_joint, 3), 'pre has wrong shape, epxected ({}, 3), but got {}'.format(num_joint, pre.shape)
    assert gt.shape[-2:] == (num_joint, 3), 'gt has wrong shape, epxected ({}, 3), but got {}'.format(num_joint, gt.shape)

    pre_centered = pre - pre.mean(dim=-2, keepdim=True)
    gt_centered = gt - gt.mean(dim=-2, keepdim=True)
    H = pre_centered.transpose(-1,-2) @ gt_centered
    U, S, V = torch.linalg.svd(H)
    R = V @ U.transpose(-1, -2)
    det = torch.det(R)
    V_corrected = V.clone()
    V_corrected = torch.where(
        det[..., None, None] < 0,
        V_corrected * torch.tensor([1, 1, -1], device=V.device),
        V_corrected
    )
    R = V_corrected @ U.transpose(-1, -2)
    s = S.sum(dim=-1) / (pre_centered ** 2).sum(dim=(-1, -2))
    pre_mean = pre.mean(dim=-2, keepdim=True)
    gt_mean = gt.mean(dim=-2, keepdim=True)
    t = gt_mean - s.unsqueeze(-1).unsqueeze(-1) * (pre_mean @ R.transpose(-1, -2))  # (..., 1, 3)
    pre_aligned = s.unsqueeze(-1).unsqueeze(-1) * (pre @ R.transpose(-1, -2)) + t
    

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
    bce = get_BCE(confidence, gt_mask)
    print('bce', bce.shape)

