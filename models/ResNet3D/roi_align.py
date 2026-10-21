"""Trilinear 3D ROI resampling and scene-coordinate restoration."""
import torch
import torch.nn.functional as F


def crop_spectra_to_gt(cube, target, xyz_limits, roi_size):
    if cube.ndim != 6 or not cube.is_floating_point():
        raise ValueError('Spectra must be real [B,T,C,X,Y,Z]')
    bounds = torch.as_tensor(xyz_limits, device=cube.device, dtype=cube.dtype)
    if bounds.shape != (3, 2) or not torch.isfinite(bounds).all() or (bounds[:, 1] <= bounds[:, 0]).any():
        raise ValueError('xyz_limits must be finite increasing [3,2] bounds')
    if len(roi_size) != 3 or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in roi_size):
        raise ValueError('roi_cube_size must contain three integers >= 2')
    boxes, mask = target['bbox'], target['mask'].bool()
    if boxes.shape != (*mask.shape, 6) or mask.shape[:2] != cube.shape[:2]:
        raise ValueError('GT boxes/mask must align with spectra B,T')
    if not torch.isfinite(boxes[mask]).all() or ((boxes[..., 3:]-boxes[..., :3])[mask] <= 0).any():
        raise ValueError('Valid GT boxes must be finite and have positive extent')
    overlap = (boxes[..., 3:] > bounds[:, 0]).all(-1) & (boxes[..., :3] < bounds[:, 1]).all(-1)
    valid = mask & overlap
    instance_valid = valid.any(dim=1)  # [B,K]，仅移除整个窗口无效的人。
    indices = instance_valid.nonzero(as_tuple=False)
    frame_valid = valid.permute(0, 2, 1)[instance_valid]
    selected = boxes.permute(0, 2, 1, 3)[instance_valid]
    unit_grid = torch.stack(torch.meshgrid(*[
        torch.linspace(0, 1, n, device=cube.device, dtype=cube.dtype) for n in roi_size
    ], indexing='ij'), -1)
    inputs = cube.new_zeros((len(indices), cube.shape[1], cube.shape[2], *roi_size))
    # 保留每个人完整的 T 帧；缺失帧置零，不借用其他帧或其他人的数据。
    for instance, (b, k) in enumerate(indices):
        for t in range(cube.shape[1]):
            if not frame_valid[instance, t]:
                continue
            box = boxes[b, t, k]
            xyz = box[:3] + unit_grid*(box[3:]-box[:3])
            grid = 2*(xyz-bounds[:, 0])/(bounds[:, 1]-bounds[:, 0])-1
            inputs[instance, t] = F.grid_sample(
                cube[b, t][None], grid[..., [2, 1, 0]][None],
                mode='bilinear', padding_mode='zeros', align_corners=True)[0]
    return {'input': inputs, 'mask': frame_valid}, {
        'valid': valid, 'instance_valid': instance_valid, 'boxes': selected,
    }


def restore_spectra_people(prediction, context):
    local = prediction['pose']
    valid, boxes = context['valid'], context['boxes']
    instance_valid = context['instance_valid']
    if (local.ndim != 5 or local.shape[:2] != boxes.shape[:2]
            or local.shape[2] != 1 or local.shape[-1] != 3):
        raise ValueError('Single-person sequence prediction must be [N,T,1,J,3]')
    # Network coordinates are relative to each frame's box centre and half-size.
    frame_valid = valid.permute(0, 2, 1)[instance_valid]
    safe_boxes = boxes.masked_fill(~frame_valid[..., None], 0.)
    center = (safe_boxes[..., :3]+safe_boxes[..., 3:])/2
    half_size = (safe_boxes[..., 3:]-safe_boxes[..., :3])/2
    world = local[:, :, 0].float()*half_size[..., None, :] + center[..., None, :]
    world = world.masked_fill(~frame_valid[..., None, None], 0.)
    batch, frames, people = valid.shape
    pose = world.new_zeros(batch, people, frames, local.shape[3], 3)
    pose[instance_valid] = world
    result = {**prediction, 'pose': pose.permute(0, 2, 1, 3, 4).contiguous(), 'pose_valid': valid}
    if 'confidence' in prediction:
        confidence = prediction['confidence'].new_zeros(batch, people, frames)
        confidence[instance_valid] = prediction['confidence'].squeeze(-1).masked_fill(~frame_valid, 0.)
        result['confidence'] = confidence.permute(0, 2, 1).contiguous()
    return result
