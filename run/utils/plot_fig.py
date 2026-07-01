from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from utils.COCO import COCO_SKELETON

def plt_fig(fig_path, pre, gt):
    # pre = {
    #     "pose": pose_pre,                         # [B, T, K, J, 3]
    #     "confidence": confidence,                 # [B, T, K]
    # }
    # gt = {
    #     padded torch.Size([64, 8, 4, 17, 3]),   # [B, T, K, J, 3]
    #     mask torch.Size([64, 8, 4])               # [B, T, K]
    # }
    pose_pre = pre['pose'].detach().cpu().numpy()
    pose_pre_confidence = pre['confidence'].detach().cpu().numpy()

    pose_gt = gt['padded'].detach().cpu().numpy()
    pose_gt_mask = gt['mask'].detach().cpu().numpy()

    b = 0
    T = pose_gt.shape[1]

    fig = plt.figure(figsize=(40, 11))
    for t in range(T):
        ax = fig.add_subplot(2, T, t + 1, projection='3d')
        for person_idx, joints in enumerate(pose_gt[b, t]):
            if pose_gt_mask[b, t, person_idx]:
                ax.scatter(
                    joints[:, 0], joints[:, 1], joints[:, 2], s=5, 
                    c='red', label=f'GT {person_idx}'
                )
                for joint_a, joint_b in COCO_SKELETON:
                    ax.plot(
                        [joints[joint_a, 0], joints[joint_b, 0]],
                        [joints[joint_a, 1], joints[joint_b, 1]],
                        [joints[joint_a, 2], joints[joint_b, 2]],
                        color='red',
                        linewidth=1.5,
                    )
                
        ax.set_xlim(0.0, 6.0)
        ax.set_ylim(-3.0, 3.0)
        ax.set_zlim(-3.0, 3.0)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f'Batch:{b}, Time:{t}')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        ax = fig.add_subplot(2, T, t + T + 1, projection='3d')
        for person_idx, joints in enumerate(pose_pre[b, t]):
            ax.scatter(
                joints[:, 0], joints[:, 1], joints[:, 2], s=5, 
                c='blue', label=f'Pre {person_idx} Confidence {pose_pre_confidence[b, t, person_idx]}'
            )
            for joint_a, joint_b in COCO_SKELETON:
                ax.plot(
                    [joints[joint_a, 0], joints[joint_b, 0]],
                    [joints[joint_a, 1], joints[joint_b, 1]],
                    [joints[joint_a, 2], joints[joint_b, 2]],
                    color='blue',
                    linewidth=1.5,
                )
                
        ax.set_xlim(0.0, 6.0)
        ax.set_ylim(-3.0, 3.0)
        ax.set_zlim(-3.0, 3.0)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f'Batch:{b}, Time:{t}')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()

    fig.tight_layout()
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=400, bbox_inches="tight")

    plt.close(fig)
