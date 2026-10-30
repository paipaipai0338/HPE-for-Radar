from utils.COCO import COCO_NAMES, COCO_SKELETON
from utils.MOCAP21 import MOCAP21_NAMES, MOCAP21_SKELETON


def get_pose_layout(pose_format):
    if pose_format == 'coco':
        return dict(names=COCO_NAMES, edges=COCO_SKELETON,
                    root_indices=(11, 12), hip_indices=(11, 12), shoulder_indices=(5, 6))
    if pose_format == 'mocap':
        return dict(names=MOCAP21_NAMES, edges=MOCAP21_SKELETON,
                    root_indices=(0,), hip_indices=(15, 18), shoulder_indices=(8, 12))
    raise ValueError(f'pose_format 必须为 coco 或 mocap，当前为 {pose_format!r}')
