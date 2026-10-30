MOCAP21_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    (4, 7), (7, 8), (8, 9), (9, 10),
    (4, 11), (11, 12), (12, 13), (13, 14),
    (0, 15), (15, 16), (16, 17),
    (0, 18), (18, 19), (19, 20),
]
MOCAP21_NAMES = [
    'pelvis', 'spine1', 'spine2', 'spine3', 'thorax', 'neck', 'head',
    'left_clavicle', 'left_shoulder', 'left_elbow', 'left_wrist',
    'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist',
    'left_hip', 'left_knee', 'left_ankle',
    'right_hip', 'right_knee', 'right_ankle',
]


def coco_to_mocap(pose):
    """将 [...,17,3] 的 COCO 张量转换为本文定义的 21 点骨架。"""
    import torch

    if pose.shape[-2:] != (17, 3):
        raise ValueError(f'COCO pose must end with [17,3], got {tuple(pose.shape)}')
    pelvis = pose[..., [11, 12], :].mean(dim=-2)
    thorax = pose[..., [5, 6], :].mean(dim=-2)
    head = pose[..., :5, :].mean(dim=-2)
    neck = .55 * thorax + .45 * head
    return torch.stack([
        pelvis,
        .75 * pelvis + .25 * thorax,
        .50 * pelvis + .50 * thorax,
        .25 * pelvis + .75 * thorax,
        thorax, neck, head,
        (neck + pose[..., 5, :]) / 2,
        pose[..., 5, :], pose[..., 7, :], pose[..., 9, :],
        (neck + pose[..., 6, :]) / 2,
        pose[..., 6, :], pose[..., 8, :], pose[..., 10, :],
        pose[..., 11, :], pose[..., 13, :], pose[..., 15, :],
        pose[..., 12, :], pose[..., 14, :], pose[..., 16, :],
    ], dim=-2)
