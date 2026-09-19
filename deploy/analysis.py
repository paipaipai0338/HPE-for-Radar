import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider, TextBox
from matplotlib.lines import Line2D

plt.switch_backend('webagg')

date = "20260912"
group = "group_051"

ACCEPTANCE_CENTER_XY = np.array([2.0, 0.0])
ACCEPTANCE_RADII_XY = np.array([1.6, 2.4])


def acceptance_mask(poses):
    """低位机坐标系中，GT 髋中心是否位于验收椭圆内。"""
    roots = np.asarray(poses).reshape(-1, 17, 3)[:, [11, 12]].mean(axis=1)
    return np.square((roots[:, :2] - ACCEPTANCE_CENTER_XY) /
                     ACCEPTANCE_RADII_XY).sum(axis=1) <= 1.0


def uid_mpjpe(poses, person_indices, target, inside, align_facing=False):
    """仅按椭圆内 GT UID 配对，计算 17 关节 MPJPE（mm）。"""
    if len(poses) != len(person_indices):
        raise ValueError("预测姿态与 person_indices 数量不一致")
    return {
        int(uid): float(np.linalg.norm(
            (align_facing_to_gt(pose, target[int(uid)]) if align_facing else pose)
            - target[int(uid)], axis=-1).mean() * 1000)
        for pose, uid in zip(poses, person_indices)
        if 0 <= uid < len(target)
        and inside[int(uid)]
        and np.isfinite(pose).all()
        and np.isfinite(target[int(uid)]).all()
    }


def load_extrinsic_npz(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Load p_radar = R_est @ p_camera + t_est."""
    with np.load(path) as data:
        rotation = np.asarray(data["R_est"], dtype=np.float64)
        translation = np.asarray(data["t_est"], dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(f"Unexpected extrinsic shapes in {path}")
    return rotation, translation
def transform_points_between_radars(
    points: np.ndarray,
    source_extrinsic: tuple[np.ndarray, np.ndarray],
    target_extrinsic: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Transform radar points directly between two calibrated radar frames."""
    source_rotation, source_translation = source_extrinsic
    target_rotation, target_translation = target_extrinsic
    relative_rotation = target_rotation @ source_rotation.T
    relative_translation = target_translation - relative_rotation @ source_translation
    return (
        np.asarray(points, dtype=np.float64) @ relative_rotation.T
        + relative_translation
    )



data_path = f"/home/pai/Huawei/temp/Inference_selected_group_{date}_{group}_results.pkl"
with open(data_path, "rb") as source:
    data = pickle.load(source)
if not data:
    raise ValueError("pkl 中没有可播放的帧")
calib_path = f"/mnt/huawei/{date}/calib"
high_extrinsic = load_extrinsic_npz(f"{calib_path}/extrinsic_img_to_radar_high.npz")
low_extrinsic = load_extrinsic_npz(f"{calib_path}/extrinsic_img_to_radar_low.npz")

fig = plt.figure(figsize=(18, 7))
fig.subplots_adjust(bottom=0.24)
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133)

scat_dyn = ax1.scatter([], [], [], c='r', label='dynamic')
scat_sta = ax1.scatter([], [], [], c='b', label='static')
scat_gt_ax1 = ax1.scatter([], [], [], c='g', label='ground truth')
scat_gt_ax2 = ax2.scatter([], [], [], c='g', label='ground truth')
scat_pre_ax2 = ax2.scatter([], [], [], c='#984ea3', label='prediction')

angle = np.linspace(0, 2 * np.pi, 100)
radius = np.linspace(0, 1, 20)
ellipse_x = ACCEPTANCE_CENTER_XY[0] + ACCEPTANCE_RADII_XY[0] * np.outer(radius, np.cos(angle))
ellipse_y = ACCEPTANCE_CENTER_XY[1] + ACCEPTANCE_RADII_XY[1] * np.outer(radius, np.sin(angle))
for ax in (ax1, ax2):
    ax.plot_surface(ellipse_x, ellipse_y, np.full_like(ellipse_x, -2.0),
                    color='#FFF2CC', alpha=0.35, shade=False)
    ax.plot(ellipse_x[-1], ellipse_y[-1], np.full_like(angle, -2.0),
            color='#b59b35', linewidth=1.5)
    ax.set_xlim(0, 6)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    ax.set_zlabel('Z/m')
    ax.legend()


def facing_vector(pose):
    """肩/髋左右轴与躯干向上轴叉乘，估计三维人体前向。"""
    if not np.isfinite(pose[[5, 6, 11, 12]]).all():
        return None
    right = (pose[6] - pose[5]) + (pose[12] - pose[11])
    up = pose[[5, 6]].mean(axis=0) - pose[[11, 12]].mean(axis=0)
    facing = np.cross(up, right)
    length = np.linalg.norm(facing)
    return facing / length if length > 1e-8 else None


def facing_xy(pose):
    facing = facing_vector(pose)
    if facing is None:
        return None
    length = np.linalg.norm(facing[:2])
    return facing[:2] / length if length > 1e-8 else None


def align_facing_to_gt(pose, target):
    """绕预测髋中心做最小三维旋转，将前向对齐 GT，保留位置和尺度。"""
    source_facing, target_facing = facing_vector(pose), facing_vector(target)
    if source_facing is None or target_facing is None:
        return np.full_like(pose, np.nan, dtype=float)
    axis = np.cross(source_facing, target_facing)
    sine = np.linalg.norm(axis)
    cosine = np.clip(np.dot(source_facing, target_facing), -1.0, 1.0)
    if sine <= 1e-8:
        if cosine >= 0:
            return pose.copy()
        # 反向时选择与前向垂直的轴，旋转 180 度。
        axis = np.cross(source_facing, np.eye(3)[np.argmin(np.abs(source_facing))])
        axis /= np.linalg.norm(axis)
        rotation = 2 * np.outer(axis, axis) - np.eye(3)
    else:
        axis /= sine
        x, y, z = axis
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        rotation = cosine * np.eye(3) + (1 - cosine) * np.outer(axis, axis) + sine * skew
    root = pose[[11, 12]].mean(axis=0)
    return (pose - root) @ rotation.T + root


def draw_top_view(frame):
    ax3.clear()
    ax3.fill(ellipse_x[-1], ellipse_y[-1], color='#FFF2CC', alpha=0.35)
    ax3.plot(ellipse_x[-1], ellipse_y[-1], color='#b59b35', linewidth=1.5)
    for stage, color in (('gt', 'g'), ('prediction', '#984ea3')):
        histories = {}
        for index in range(max(0, frame - 4), frame + 1):
            record = data[index]
            if stage == 'gt':
                poses = np.asarray(record['pose_gt']).reshape(-1, 17, 3)
                uids = np.arange(len(poses))
            else:
                raw = record['raw_inference']
                poses = np.asarray(raw['poses']).reshape(-1, 17, 3)
                uids = raw['person_indices']
            poses = transform_points_between_radars(poses, high_extrinsic, low_extrinsic)
            for uid, pose in zip(uids, poses):
                histories.setdefault(int(uid), {})[index] = pose
        for uid, observations in histories.items():
            # 漏失帧保留 NaN，避免跨缺失连接轨迹。
            roots = np.full((min(5, frame + 1), 2), np.nan)
            start = max(0, frame - 4)
            for index, pose in observations.items():
                roots[index - start] = pose[[11, 12], :2].mean(axis=0)
            ax3.plot(*roots.T, color=color, marker='.', linewidth=1, alpha=0.6)
            pose = observations.get(frame)
            if pose is None or not np.isfinite(pose[[11, 12]]).all():
                continue
            root = pose[[11, 12], :2].mean(axis=0)
            ax3.scatter(*root, color=color, s=35)
            ax3.annotate(f'UID {uid} {stage}', root, xytext=(4, 5),
                         textcoords='offset points', color=color, fontsize=7)
            facing = facing_xy(pose)
            if facing is not None:
                ax3.quiver(*root, *facing, color=color, angles='xy',
                           scale_units='xy', scale=2.5, width=0.006)
    ax3.set(xlim=(0, 6), ylim=(-3, 3), xlabel='X/m', ylabel='Y/m',
            title=f'Top view: last 5 frames | Frame {frame + 1}/{len(data)}')
    ax3.set_aspect('equal', adjustable='box')
    ax3.grid(alpha=0.25)
    ax3.legend(handles=[
        Line2D([], [], color='g', marker='.', label='GT'),
        Line2D([], [], color='#984ea3', marker='.', label='Prediction'),
    ])


uid_texts = []
facing_arrows = []


def update(frame):
    record = data[frame]
    pts = np.asarray(record['point_cloud'])
    target = np.asarray(record['pose_gt']).reshape(-1, 17, 3)
    raw = record['raw_inference']
    poses = np.asarray(raw['poses']).reshape(-1, 17, 3)
    person_indices = np.asarray(raw['person_indices'], dtype=int)
    dynamic = pts[:, -1] == 1
    xyz = transform_points_between_radars(
        pts[:, :3], high_extrinsic, low_extrinsic
    )
    pose_gt = transform_points_between_radars(
        np.asarray(record['pose_gt']).reshape(-1, 3), high_extrinsic, low_extrinsic
    )
    pre_pose = transform_points_between_radars(
        np.asarray(record['raw_inference']['poses']).reshape(-1, 3),
        high_extrinsic, low_extrinsic,
    )

    inside = acceptance_mask(pose_gt)
    errors = uid_mpjpe(poses, person_indices, target, inside)
    aligned_errors = uid_mpjpe(poses, person_indices, target, inside, align_facing=True)

    scat_dyn._offsets3d = tuple(xyz[dynamic].T)
    scat_sta._offsets3d = tuple(xyz[~dynamic].T)
    scat_gt_ax1._offsets3d = tuple(pose_gt.T)
    scat_gt_ax2._offsets3d = tuple(pose_gt.T)
    scat_pre_ax2._offsets3d = tuple(pre_pose.T)
    for text in uid_texts:
        text.remove()
    uid_texts.clear()
    for arrow in facing_arrows:
        arrow.remove()
    facing_arrows.clear()
    for axes, skeletons, color in (
        ((ax1, ax2), pose_gt.reshape(-1, 17, 3), 'g'),
        ((ax2,), pre_pose.reshape(-1, 17, 3), '#984ea3'),
    ):
        for pose in skeletons:
            nose = pose[0]
            facing = facing_vector(pose)
            if facing is None or not np.isfinite(nose).all():
                continue
            for ax in axes:
                facing_arrows.append(ax.quiver(
                    *nose, *facing,
                    length=0.4, color=color, arrow_length_ratio=0.3,
                ))
    handles = [
        Line2D([], [], color='g', marker='o', linestyle='None', label='GT'),
        Line2D([], [], color='#984ea3', marker='o', linestyle='None', label='Prediction'),
    ]
    for uid, pose in enumerate(pose_gt.reshape(-1, 17, 3)):
        if not np.isfinite(pose).all():
            continue
        position = pose[[11, 12]].mean(axis=0)
        position[2] = pose[:, 2].max() + 0.15
        for ax in (ax1, ax2):
            uid_texts.append(ax.text(*position, f'UID {uid} GT', color='g', fontsize=7))
        error = ('outside ellipse' if not inside[uid] else
                 f'{errors[uid]:.1f} mm' if uid in errors else 'N/A')
        aligned_error = ('outside ellipse' if not inside[uid] else
                         f'{aligned_errors[uid]:.1f} mm'
                         if uid in aligned_errors and np.isfinite(aligned_errors[uid]) else 'N/A')
        handles.append(Line2D([], [], color='#984ea3', linestyle='None',
                              label=f'UID {uid} | MPJPE: {error} | 3D facing aligned: {aligned_error}'))
    for uid, pose in zip(person_indices, pre_pose.reshape(-1, 17, 3)):
        if not np.isfinite(pose).all():
            continue
        position = pose[[11, 12]].mean(axis=0)
        position[2] = pose[:, 2].max() + 0.15
        uid_texts.append(ax2.text(*position, f'UID {uid} prediction',
                                 color='#984ea3', fontsize=7))
    ax2.legend(handles=handles, fontsize=7.5, loc='upper right')
    ax1.set_title(f"Point cloud + GT | Frame {frame + 1}/{len(data)}")
    ax2.set_title(f"GT + Prediction | Frame {frame + 1}/{len(data)}")
    draw_top_view(frame)
    return scat_dyn, scat_sta, scat_gt_ax1, scat_gt_ax2, scat_pre_ax2


current_frame = 0
playing = True
slider = Slider(fig.add_axes([0.15, 0.13, 0.7, 0.03]), 'Frame',
                1, max(2, len(data)), valinit=1, valstep=1, valfmt='%d')
previous_button = Button(fig.add_axes([0.15, 0.04, 0.12, 0.05]), 'Previous')
play_button = Button(fig.add_axes([0.29, 0.04, 0.12, 0.05]), 'Pause')
next_button = Button(fig.add_axes([0.43, 0.04, 0.12, 0.05]), 'Next')
frame_box = TextBox(fig.add_axes([0.69, 0.04, 0.16, 0.05]), 'Go to frame ', initial='1')


def set_playing(value):
    global playing
    playing = value
    play_button.label.set_text('Pause' if playing else 'Play')
    fig.canvas.draw_idle()


def show_frame(frame):
    global current_frame
    current_frame = max(0, min(int(frame), len(data) - 1))
    # 程序同步控件时不触发跳转回调，避免自动播放被暂停。
    slider.eventson = False
    frame_box.eventson = False
    try:
        slider.set_val(current_frame + 1)
        frame_box.set_val(str(current_frame + 1))
    finally:
        slider.eventson = True
        frame_box.eventson = True
    update(current_frame)
    fig.canvas.draw_idle()


def seek(value):
    set_playing(False)
    show_frame(int(value) - 1)


def submit_frame(value):
    try:
        frame = int(value)
    except ValueError:
        frame = current_frame + 1
    seek(frame)


def step(offset):
    set_playing(False)
    show_frame(current_frame + offset)


def on_key(event):
    if event.inaxes is frame_box.ax:
        return
    if event.key == ' ':
        set_playing(not playing)
    elif event.key == 'left':
        step(-1)
    elif event.key == 'right':
        step(1)


def tick():
    if playing:
        show_frame((current_frame + 1) % len(data))


slider.on_changed(seek)
frame_box.on_submit(submit_frame)
previous_button.on_clicked(lambda event: step(-1))
next_button.on_clicked(lambda event: step(1))
play_button.on_clicked(lambda event: set_playing(not playing))
fig.canvas.mpl_connect('key_press_event', on_key)
timer = fig.canvas.new_timer(interval=120)
timer.add_callback(tick)
show_frame(0)
timer.start()
plt.show()
