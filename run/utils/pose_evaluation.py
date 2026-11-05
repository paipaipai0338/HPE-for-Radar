"""Evaluation definitions ported from main (58b9d2d); distances in metres."""
import torch
from utils.skeleton import get_pose_layout
from utils.MOCAP21 import coco_to_mocap

def get_mpjpe(pre: torch.tensor, gt:torch.tensor, type:str='coco') -> torch.tensor:
    assert pre.shape == gt.shape, 'pre and gt do not have same shape'
    common_shape = gt.shape
    num_joint = len(get_pose_layout(type.lower())['names'])
    assert pre.shape[-2:] == (num_joint, 3), 'pre has wrong shape, epxected ({}, 3), but got {}'.format(num_joint, pre.shape)
    assert gt.shape[-2:] == (num_joint, 3), 'gt has wrong shape, epxected ({}, 3), but got {}'.format(num_joint, gt.shape)

    mpjpe = torch.mean(torch.norm(pre - gt, dim=-1), dim=-1)
    assert mpjpe.shape == common_shape[:-2], 'the finnal results has wrong shape'
    return mpjpe

def get_center_mpjpe(pre: torch.Tensor, gt: torch.Tensor, type: str = 'coco') -> torch.Tensor:
    """计算预测与 GT 骨架根节点中心的欧氏距离。"""
    assert pre.shape == gt.shape, 'pre and gt do not have same shape'
    layout = get_pose_layout(type.lower())
    assert pre.shape[-2:] == (len(layout['names']), 3), f'pre has wrong shape: {pre.shape}'
    pre_center = pre[..., layout['root_indices'], :].mean(dim=-2)
    gt_center = gt[..., layout['root_indices'], :].mean(dim=-2)
    return torch.norm(pre_center - gt_center, dim=-1)

def get_relative_pose_mpjpe(pre: torch.Tensor, gt: torch.Tensor, type: str = 'coco') -> torch.Tensor:
    """分别减去骨架根节点中心后计算 MPJPE。"""
    assert pre.shape == gt.shape, 'pre and gt do not have same shape'
    root_indices = get_pose_layout(type.lower())['root_indices']
    pre_center = pre[..., root_indices, :].mean(dim=-2, keepdim=True)
    gt_center = gt[..., root_indices, :].mean(dim=-2, keepdim=True)
    return get_mpjpe(pre - pre_center, gt - gt_center, type=type)

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
    num_joint = len(get_pose_layout(type.lower())['names'])
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


def evaluate_pose_formats(prediction, target):
    if prediction.shape != target.shape or target.shape[-2:] != (17, 3):
        raise ValueError('Expected aligned COCO17 predictions and targets')
    root = target[..., [11, 12], :2].mean(-2)
    inside = ((root-root.new_tensor([2., 0.]))/root.new_tensor([1.6, 2.4])).square().sum(-1) <= 1
    result = {}
    for layout in ('coco', 'mocap'):
        p, y = (prediction, target) if layout == 'coco' else (coco_to_mocap(prediction), coco_to_mocap(target))
        metrics = {name:fn(p,y,type=layout) for name,fn in dict(mpjpe=get_mpjpe,
            centered_mpjpe=get_relative_pose_mpjpe,root_error=get_center_mpjpe,pa_mpjpe=get_pampjpe).items()}
        if layout == 'mocap':
            error=((p-p[..., :1, :])-(y-y[..., :1, :])).norm(dim=-1)
            metrics.update(torso_centered=error[..., :6].mean(-1),
                limbs_centered=error[..., [8,9,10,12,13,14,15,16,17,18,19,20]].mean(-1))
        for region,mask in [('all',torch.ones_like(inside)),('inside',inside),('outside',~inside)]:
            n=int(mask.sum())
            result[layout+'/'+region]=dict(frames=n,**{k:float(v[mask].sum()) for k,v in metrics.items()})
    return result
