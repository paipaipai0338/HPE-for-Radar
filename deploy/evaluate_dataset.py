"""独立的 NAS 同结构数据集评估：PC 点云 / 原始 BIN 频谱，末帧 Keeper+躺姿头脚锁输出计分。

python evaluate_dataset.py --root /mnt/huawei --date 20261023 --group 033 \
  --checkpoint MODEL/checkpoint/last.pth --config MODEL/config/config.yaml

--checkpoint 接受 .pth 或 .onnx，自动选择 PyTorch 或 ONNX Runtime。
--bundle evaluator.zip 打包脚本、配置和模型；ONNX 无需 PyTorch 模型源码。
保存 predictions.npz（mocap21、米、高位机坐标，包含 raw、keeper 和状态）。
stdout：日期\t组号\t全身误差_mm\t躯干误差_mm\t四肢误差_mm。
"""
import argparse
from collections import deque
import importlib.util
import io
import json
from pathlib import Path
import pickle
import sys
import zipfile

import numpy as np
import torch
import yaml

NAMES = ('pelvis spine1 spine2 spine3 thorax neck head left_clavicle left_shoulder '
         'left_elbow left_wrist right_clavicle right_shoulder right_elbow right_wrist '
         'left_hip left_knee left_ankle right_hip right_knee right_ankle').split()
EDGES = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(4,7),(7,8),(8,9),(9,10),
         (4,11),(11,12),(12,13),(13,14),(0,15),(15,16),(16,17),(0,18),(18,19),(19,20)]
PARTS = {'all': list(range(21)), 'torso': list(range(6)),
         'limbs': [8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20]}


def mocap(pose):
    if pose.shape[-2:] == (21, 3):
        return pose.copy()
    if pose.shape[-2:] != (17, 3):
        raise ValueError(f'需要 COCO17 或 MOCAP21，实际 {pose.shape}')
    pelvis = pose[..., [11, 12], :].mean(-2)
    thorax = pose[..., [5, 6], :].mean(-2)
    head = pose[..., :5, :].mean(-2)
    neck = .55 * thorax + .45 * head
    return np.stack([pelvis, .75*pelvis+.25*thorax, .5*pelvis+.5*thorax,
                     .25*pelvis+.75*thorax, thorax, neck, head,
                     (neck+pose[...,5,:])/2, pose[...,5,:], pose[...,7,:], pose[...,9,:],
                     (neck+pose[...,6,:])/2, pose[...,6,:], pose[...,8,:], pose[...,10,:],
                     pose[...,11,:], pose[...,13,:], pose[...,15,:],
                     pose[...,12,:], pose[...,14,:], pose[...,16,:]], axis=-2)


def timestamp_ns(name):
    sec, ns = Path(name).stem.split('_')
    return int(sec)*10**9 + int(ns)


class Frames:
    """普通帧文件或同目录 frames.*pack；不写数据源和对齐缓存。"""
    def __init__(self, directory, suffix):
        self.directory = directory
        self.pack = directory / ('frames'+{'.npy': '.pcpack', '.pkl': '.gtpack', '.bin': '.binpack'}[suffix])
        self.index = {}
        if self.pack.exists():
            with np.load(directory / 'frames_index.npz', allow_pickle=False) as data:
                names = data['frame_names'].astype(str).tolist()
                offsets, lengths = data['offsets'], data['lengths']
            if (len(set(names)) != len(names) or len(names) != len(offsets)
                    or len(names) != len(lengths) or offsets.shape != (len(names),)
                    or lengths.shape != (len(names),) or offsets.dtype.kind not in 'iu'
                    or lengths.dtype.kind not in 'iu' or np.any(lengths < 0)
                    or (len(names) and (offsets[0] != 0
                        or np.any(offsets[1:] != offsets[:-1]+lengths[:-1])
                        or offsets[-1]+lengths[-1] != self.pack.stat().st_size))
                    or any(Path(n).name != n or not n.endswith(suffix) for n in names)):
                raise ValueError(f'打包索引无效：{directory}')
            self.index = dict(zip(names, zip(offsets, lengths)))
        else:
            names = [p.name for p in directory.glob('*'+suffix)]
        self.names = sorted(names, key=timestamp_ns)
        self.times = np.array([timestamp_ns(n) for n in self.names], dtype=np.int64)
        if len(self.times) > 1 and np.any(np.diff(self.times) <= 0):
            raise ValueError(f'时间戳不递增：{directory}')

    def read(self, name):
        if self.index:
            offset, length = self.index[name]
            with self.pack.open('rb') as source:
                source.seek(int(offset))
                content = source.read(int(length))
            if len(content) != length:
                raise ValueError(f'帧数据被截断：{name}')
            source = io.BytesIO(content)
        else:
            source = (self.directory / name).open('rb')
        with source:
            if name.endswith('.bin'):
                return np.frombuffer(source.read(), dtype=np.uint8)
            return np.load(source, allow_pickle=False) if name.endswith('.npy') else pickle.load(source)

    def nearest(self, time):
        if not len(self.times):
            return None
        i = np.searchsorted(self.times, time)
        i = min((j for j in (i-1, i) if 0 <= j < len(self.times)),
                key=lambda j: abs(int(self.times[j])-int(time)))
        return i


def sample_points(points, count, center_xyz=False, center_xy=False):
    center = np.zeros(3, np.float32)
    sampled = np.zeros((count, points.shape[-1]), np.float32)
    mask = np.zeros(count, bool)
    if len(points):
        if center_xyz or center_xy:
            center = points[:, :3].mean(0)
            if center_xy:
                center[2] = 0
        ids = (np.linspace(0, len(points)-1, count).astype(int) if len(points) >= count
               else np.arange(count) % len(points))
        sampled[:] = points[ids]
        sampled[:, :3] -= center
        mask[:] = True
    return sampled, mask, center


# 当前帧质量门控；不使用 80% 动点确认。
KEEPER_PARAMS = {"min_dynamic": 20, "min_total": 30, "confirm_frames": 3,
                 "stable_root_speed_m_s": 1.0, "stable_shape_speed_m_s": 2 / 3,
                 "stable_axis_deg": 30.0, "reversal_deg": 100.0,
                 "reversal_speed_deg_s": 360.0, "lying_reversal_deg": 90.0}


def keeper_metric_protocol():
    return dict(enabled=True, name='keeper_with_lying_head_lock', params=KEEPER_PARAMS.copy(),
                offline_backfill=False, direction_check=False, lying_head_lock=True,
                point_counts='unaccumulated current-frame GT crop; dynamic = TLV tag 1',
                person_ids='GT array slots; not detector track IDs',
                pose_format='mocap21; pelvis=0, head=6, shoulders=8/12, hips=15/18, ankles=17/20',
                description='点数不足或缺 raw 持续保持；方向反转三帧稳定确认，不回填；'
                            '首次接受躺姿锁定头脚轴，躺姿反向>90度拒绝；'
                            '连续三帧站/坐蹲且相邻头脚轴≤30度，经采纳后解锁。')


def vector_angle(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / norm, -1, 1)))) if norm > 1e-10 else 180.


def head_axis(pose):
    axis = pose[6] - pose[[17, 20]].mean(0)
    length = np.linalg.norm(axis)
    return axis / length if length >= .4 else None


def posture_label(pose):
    """重力坐标系 MOCAP21；对应 actionprocess 的默认几何规则，头部使用 head 节点。"""
    shoulder, hip = pose[[8, 12]].mean(0), pose[[15, 18]].mean(0)
    knee, ankle = pose[[16, 19]].mean(0), pose[[17, 20]].mean(0)
    scale = np.linalg.norm(shoulder - hip) + sum(
        np.linalg.norm(pose[h] - pose[k]) + np.linalg.norm(pose[k] - pose[a])
        for h, k, a in ((15, 16, 17), (18, 19, 20))) / 2
    if scale < 1e-6:
        return 'other'
    def tilt(v):
        return 90. if np.linalg.norm(v) < 1e-6 else vector_angle(np.array([0., 0., 1.]), np.r_[v[:2], abs(v[2])])
    height = np.ptp(pose[:, 2]) / scale
    horizontal = np.linalg.norm(pose[:, None, :2] - pose[None, :, :2], axis=-1).max() / scale
    torso, body = tilt(shoulder - hip), tilt(pose[6] - ankle)
    sh, ha, hk, ka = [(a[2] - b[2]) / scale for a, b in
                      ((shoulder, hip), (hip, ankle), (hip, knee), (knee, ankle))]
    angles = []
    for h, k, a in ((15, 16, 17), (18, 19, 20)):
        upper, lower = pose[h] - pose[k], pose[a] - pose[k]
        angles.append(0. if np.linalg.norm(upper) * np.linalg.norm(lower) < 1e-6 else vector_angle(upper, lower))
    support = 0 if pose[17, 2] <= pose[20, 2] else 1
    h, k, a = (15, 16, 17) if support == 0 else (18, 19, 20)
    if horizontal >= .55 and torso >= 68 and ha <= .14:
        return 'lie'
    if torso <= 30 and sh >= .18 and (
        (height >= 1.05 and (pose[h, 2] - pose[a, 2]) / scale >= .40
         and (pose[k, 2] - pose[a, 2]) / scale >= .15 and angles[support] >= 130)
        or (height >= .95 and .35 <= ha <= .48 and abs(angles[0] - angles[1]) >= 15)):
        return 'stand'
    if height >= .40 and sh >= .06 and ha >= .12 and torso <= 70 and (
        (height <= 1.08 and ha <= .52 and hk <= .15
         and min(pose[16, 2] - pose[17, 2], pose[19, 2] - pose[20, 2]) / scale >= .06
         and max(angles) <= 145)
        or (height <= .85 and ha <= .47 and body <= 55 and np.mean(angles) <= 105 and ka >= .12)):
        return 'sit_squat'
    return 'other'


def pose_reversal(raw, keeper, dt):
    threshold = max(KEEPER_PARAMS['reversal_deg'], KEEPER_PARAMS['reversal_speed_deg_s'] * dt)
    def reversed_axis(a, b, minimum):
        return min(np.linalg.norm(a), np.linalg.norm(b)) >= minimum and vector_angle(a, b) > threshold
    return ((reversed_axis(raw[12] - raw[8], keeper[12] - keeper[8], .1)
             and reversed_axis(raw[18] - raw[15], keeper[18] - keeper[15], .1))
            or reversed_axis(raw[6] - raw[[17, 20]].mean(0),
                             keeper[6] - keeper[[17, 20]].mean(0), .4))


def stable_candidate(raw, previous, dt):
    return (np.linalg.norm(raw[0] - previous[0]) <= KEEPER_PARAMS['stable_root_speed_m_s'] * dt
            and np.linalg.norm((raw - raw[0]) - (previous - previous[0]), axis=-1).mean()
            <= KEEPER_PARAMS['stable_shape_speed_m_s'] * dt
            and vector_angle(raw[6] - raw[[17, 20]].mean(0), previous[6] - previous[[17, 20]].mean(0))
            <= KEEPER_PARAMS['stable_axis_deg'])


def keeper_pose_records(records, rotation, translation):
    """因果 Keeper + 躺姿头脚锁；输入/输出均为高位坐标 MOCAP21，不回填。"""
    rotation, translation = np.asarray(rotation), np.asarray(translation)
    if rotation.shape != (3, 3) or translation.shape != (3,) or not (
            np.isfinite(rotation).all() and np.isfinite(translation).all()):
        raise ValueError('Keeper 需要有效的高位到低位标定 R(3,3)、t(3,)')
    states = {}
    for index, record in enumerate(records):
        time = timestamp_ns(record['path'])
        raws = dict(zip(record['pose_indices'], record['poses_raw']))
        output, statuses, snapshots = {}, {}, {}
        for person in record.get('track_ids', record['pose_indices']):
            person = int(person)
            state = states.get(person)
            if state is None or state['last'] != index - 1:
                state = dict(last=index - 1, time=time - 120000000, keeper=None, candidate=None,
                             count=0, previous_axis=None, exit_run=0, locked=False, head_ref=None)
                states[person] = state
            dt = (time - state['time']) * 1e-9
            if dt <= 0:
                raise ValueError('Non-increasing point-cloud timestamp')
            raw = raws.get(person)
            if raw is not None and raw.shape != (21, 3):
                raise ValueError(f'Keeper 需要 MOCAP21，实际 {raw.shape}')
            status = 'kept' if state['keeper'] is not None else 'excluded'
            weak = (record['dynamic_point_counts'].get(person, 0) < KEEPER_PARAMS['min_dynamic']
                    or record['total_point_counts'].get(person, 0) < KEEPER_PARAMS['min_total'])
            if weak or raw is None or not np.isfinite(raw).all():
                state.update(candidate=None, count=0, previous_axis=None, exit_run=0)
                reason = 'point_or_raw_gate'
            else:
                axis = head_axis(raw)
                label = posture_label(raw @ rotation.T + translation)
                upright = label in ('stand', 'sit_squat') and axis is not None
                exit_stable = (axis is not None and state['previous_axis'] is not None
                               and vector_angle(axis, state['previous_axis']) <= 30)
                state['exit_run'] = (state['exit_run'] + 1 if exit_stable else 1) if upright else 0
                state['previous_axis'] = axis
                reason = 'allowed'
                if state['locked']:
                    if label != 'lie' and state['exit_run'] < 3:
                        reason = 'lying_exit_pending'
                    elif label == 'lie' and (axis is None or vector_angle(axis, state['head_ref']) > 90):
                        reason = 'lying_head_reversal'
                permit = reason == 'allowed'
                if permit and (state['keeper'] is None or not pose_reversal(raw, state['keeper'], dt)):
                    state.update(keeper=raw.copy(), candidate=None, count=0)
                    status = 'accepted'
                else:
                    stable = state['candidate'] is not None and stable_candidate(raw, state['candidate'], dt)
                    state['count'] = state['count'] + 1 if stable else 1
                    state['candidate'] = raw.copy()
                    if permit and state['count'] >= KEEPER_PARAMS['confirm_frames']:
                        state.update(keeper=raw.copy(), candidate=None, count=0)
                        status = 'confirmed'
                if status in ('accepted', 'confirmed'):
                    if state['locked'] and state['exit_run'] >= 3:
                        state['locked'] = False
                    if not state['locked'] and axis is not None:
                        state['head_ref'] = axis.copy()
                        state['locked'] = label == 'lie'
            state['last'], state['time'] = index, time
            statuses[person] = status
            snapshots[person] = dict(lying_locked=state['locked'], reason=reason,
                                     head_foot_ref=None if state['head_ref'] is None else state['head_ref'].copy())
            if state['keeper'] is not None:
                output[person] = state['keeper'].copy()
        record['poses'] = np.asarray(list(output.values()), np.float32).reshape(-1, 21, 3)
        record['pose_track_ids'] = np.asarray(list(output), int)
        record['pose_status'], record['keeper_state'] = statuses, snapshots


SKY_WINDOW = [
    1, 5, 11, 20, 31, 44, 60, 78, 99, 122, 147, 175, 205, 238, 272, 309,
    349, 390, 434, 480, 528, 578, 631, 685, 742, 800, 860, 923, 987, 1053, 1121, 1191,
    1262, 1335, 1410, 1487, 1565, 1644, 1725, 1807, 1891, 1976, 2062, 2150, 2239, 2328, 2419, 2511,
    2604, 2698, 2792, 2888, 2984, 3080, 3178, 3275, 3374, 3473, 3572, 3671, 3771, 3871, 3971, 4071,
    4171, 4271, 4371, 4471, 4571, 4670, 4769, 4867, 4966, 5063, 5160, 5257, 5352, 5447, 5541, 5635,
    5727, 5818, 5909, 5998, 6086, 6173, 6259, 6343, 6426, 6508, 6588, 6667, 6744, 6819, 6893, 6966,
    7036, 7105, 7172, 7237, 7301, 7362, 7421, 7479, 7534, 7588, 7639, 7688, 7735, 7780, 7823, 7863,
    7901, 7937, 7971, 8002, 8031, 8058, 8082, 8104, 8123, 8140, 8155, 8167, 8177, 8184, 8189, 8191,
    8191, 8189, 8184, 8177, 8167, 8155, 8140, 8123, 8104, 8082, 8058, 8031, 8002, 7971, 7937, 7901,
    7863, 7823, 7780, 7735, 7688, 7639, 7588, 7534, 7479, 7421, 7362, 7301, 7237, 7172, 7105, 7036,
    6966, 6893, 6819, 6744, 6667, 6588, 6508, 6426, 6343, 6259, 6173, 6086, 5998, 5909, 5818, 5727,
    5635, 5541, 5447, 5352, 5257, 5160, 5063, 4966, 4867, 4769, 4670, 4571, 4471, 4371, 4271, 4171,
    4071, 3971, 3871, 3771, 3671, 3572, 3473, 3374, 3275, 3178, 3080, 2984, 2888, 2792, 2698, 2604,
    2511, 2419, 2328, 2239, 2150, 2062, 1976, 1891, 1807, 1725, 1644, 1565, 1487, 1410, 1335, 1262,
    1191, 1121, 1053, 987, 923, 860, 800, 742, 685, 631, 578, 528, 480, 434, 390, 349,
    309, 272, 238, 205, 175, 147, 122, 99, 78, 60, 44, 31, 20, 11, 5, 1,
]


def decode_bin(raw, radar):
    r, c, m = radar['num_samp']//2, radar['num_chirp'], radar['Tx']*radar['Rx']
    if raw.size != r*c*m*4:
        raise ValueError(f'BIN 字节数 {raw.size} != {r*c*m*4}')
    word = raw.reshape(-1, 8)[:, ::-1].copy().reshape(-1).view('<u4')
    real, imag = (word & 0x3fff).astype(np.int32), ((word >> 14) & 0x3fff).astype(np.int32)
    real[real >= 8192] -= 16384
    imag[imag >= 8192] -= 16384
    value = (real+1j*imag)*np.exp2((word >> 28).astype(np.int32)-13, dtype=np.float32)
    return value.astype(np.complex64).reshape(r, m, c, order='F').transpose(0, 2, 1)


class Spectra:
    """与训练一致的固件 TDM FFT、DBF、DXYZ/KXYZ 和 log1p 功率。"""
    def __init__(self, radar, data, device):
        self.radar, self.data, self.device = radar, data, device
        if (radar['num_chirp'], radar['Tx'], radar['Rx']) != (64, 4, 4):
            raise ValueError('当前 BIN/KXYZ 协议要求 64 chirps、4Tx、4Rx')
        limits = np.asarray(data['xyz_limits'])
        if (limits.shape != (3, 2) or not np.isfinite(limits).all()
                or np.any(limits[:, 0] >= limits[:, 1])
                or len(data['cube_size']) != 3 or min(data['cube_size']) < 1
                or len(data.get('angle_bins', [128, 32])) != 2
                or min(data.get('angle_bins', [128, 32])) < 2
                or radar['num_samp'] < 4):
            raise ValueError('频谱 xyz_limits、cube_size、angle_bins 或 num_samp 无效')
        positions = np.asarray(radar['virtual_channel_positions'])
        if positions.shape != (16, 3) or not np.isfinite(positions).all():
            raise ValueError('virtual_channel_positions 必须为有限 [16,3]')
        self.history = deque(maxlen=7)
        self.geometry = None

    def __call__(self, cube):
        r, cfg, dev = self.radar, self.data, self.device
        current = torch.as_tensor(cube, device=dev)[None, None]
        self.history.append(current)
        differences = None
        window = torch.hann_window(256, periodic=False, device=dev).reshape(64, 4)
        window = window.repeat_interleave(4, dim=1)
        if cfg.get('include_kxyz', False):
            if len(self.history) < 7:
                return None
            # 固件 Q13 窗，与 SignalProcessConfig.winVelPre 相同。
            sky_window = torch.as_tensor(SKY_WINDOW, device=dev, dtype=torch.float32)
            sky_window = sky_window.reshape(64, 4).repeat_interleave(4, dim=1)
            means = [(x*sky_window).mean(3)/sky_window.mean(0) for x in self.history]
            differences = torch.stack([means[-1]-x for x in means[-2::-1]], dim=3)
        if cfg.get('remove_static', False):
            current = current-current.mean(3, keepdim=True)
        doppler = torch.fft.fftshift(torch.fft.fft(current*window, dim=3), dim=3)
        bins = torch.arange(64, device=dev)-32
        tx = torch.arange(16, device=dev)//4
        phase = -2*torch.pi*bins[:, None]*tx[None]/256
        doppler = doppler*torch.exp(1j*phase)/window.sum(0)
        snapshots = torch.cat((doppler, differences), dim=3) if differences is not None else doppler
        if self.geometry is None:
            angles = cfg.get('angle_bins', [128, 32])
            azimuth = torch.linspace(*[torch.pi*v/180 for v in r['azi_deg']], angles[0], device=dev)
            elevation = torch.linspace(*[torch.pi*v/180 for v in r['ele_deg']], angles[1], device=dev)
            az, el = torch.meshgrid(azimuth, elevation, indexing='ij')
            directions = torch.stack((el.cos()*az.cos(), el.cos()*az.sin(), el.sin()), -1).reshape(-1, 3)
            positions = torch.tensor(r['virtual_channel_positions'], device=dev)
            weights = torch.exp(2j*torch.pi/r['lam']*((positions-positions[:1]) @ directions.T))/16
            xyz = torch.meshgrid(*[torch.linspace(*lim, n, device=dev)
                                   for lim, n in zip(cfg['xyz_limits'], cfg['cube_size'])], indexing='ij')
            x, y, z = xyz
            radius = (x*x+y*y+z*z).sqrt()
            az, el = torch.atan2(y, x), torch.atan2(z, (x*x+y*y).sqrt())
            step = r['c']*r['fs']/(2*r['slope']*r['num_samp'])
            norm = lambda value, low, high: 2*(value-low)/(high-low)-1
            grid = torch.stack((norm(el, elevation[0], elevation[-1]),
                                norm(az, azimuth[0], azimuth[-1]),
                                norm(radius, 0, step*(cube.shape[0]-1))), -1)[None]
            self.geometry = weights, grid, angles
        weights, grid, angles = self.geometry
        channels = snapshots[0, 0].permute(1, 0, 2)
        result = []
        # 逐通道限制极坐标临时张量内存，不改变 FFT/插值口径。
        for channel in channels:
            power = torch.cat([(channel @ w).abs().square() for w in weights.split(512, dim=1)], -1)
            polar = power.reshape(1, 1, cube.shape[0], *angles)
            xyz = torch.nn.functional.grid_sample(polar, grid, align_corners=True)
            result.append(xyz[0, 0].clamp_min(0).log1p())
        return torch.stack(result)


def model_source(config_path, name):
    candidates = [config_path.parent/(name+'.py')]
    for parent in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
        candidates.append(parent/'models'/name/(name+'.py'))
    for source in candidates:
        if source.exists():
            return source
    raise FileNotFoundError(f'需要模型源码：{name}.py；请放在 config 同目录')


class OnnxModel:
    def __init__(self, path, device):
        import onnxruntime as ort

        device = torch.device(device)
        providers = ['CPUExecutionProvider']
        if device.type == 'cuda':
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('ONNX CUDA 推理需要 onnxruntime-gpu；或使用 --device cpu')
            providers.insert(0, ('CUDAExecutionProvider', {'device_id': device.index or 0}))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        self.inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        poses = [o for o in outputs if o.name == 'pose']
        if not poses:
            poses = outputs if len(outputs) == 1 else [
                o for o in outputs if len(o.shape) == 5 and o.shape[-2] in (17, 21) and o.shape[-1] == 3]
        if len(poses) != 1:
            raise ValueError('ONNX 需要唯一的 pose 输出，形状为 [B,T,P,17或21,3]')
        self.output = poses[0].name

    def __call__(self, inputs):
        dtypes = {'tensor(float)': np.float32, 'tensor(float16)': np.float16,
                  'tensor(double)': np.float64, 'tensor(bool)': np.bool_, 'tensor(int64)': np.int64}
        feed = {}
        for spec in self.inputs:
            key = 'input' if spec.name == 'points' else spec.name
            if key not in inputs or spec.type not in dtypes:
                raise ValueError(f'无法提供 ONNX 输入 {spec.name!r} ({spec.type})；支持 input/points、mask、bbox、bbox_mask')
            value = inputs[key].detach().cpu().numpy()
            # 导出的点云模型可能仅声明 xyz，原始点云仍保留六列供质量判断。
            if key == 'input' and value.ndim == 4 and isinstance(spec.shape[-1], int):
                value = value[..., :spec.shape[-1]]
            feed[spec.name] = np.ascontiguousarray(value, dtype=dtypes[spec.type])
        output = self.session.run([self.output], feed)[0]
        if output.ndim != 5 or output.shape[-2:] not in ((17, 3), (21, 3)):
            raise ValueError(f'ONNX pose 输出应为 [B,T,P,17或21,3]，实际 {output.shape}')
        return {'pose': torch.from_numpy(output)}


def load_model(args, config):
    if torch.device(args.device).type == 'cuda':
        # pointnet2 扩展在当前 CUDA 设备启动内核。
        torch.cuda.set_device(args.device)
    if args.checkpoint.suffix.lower() == '.onnx':
        return OnnxModel(args.checkpoint, args.device)
    if args.checkpoint.suffix.lower() != '.pth':
        raise ValueError('--checkpoint 仅支持 .pth 或 .onnx')
    name = config['model']['name']
    folder = args.config.parent
    source = model_source(args.config, name)
    # 解压后的包或仓库均可；评估逻辑不导入 run/data2datasets/deploy。
    for parent in (folder, Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
        if (parent / 'models').is_dir():
            sys.path.insert(0, str(parent))
            break
    spec = importlib.util.spec_from_file_location('evaluation_model', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    options = yaml.safe_load((folder/'model_config.yaml').read_text())
    model = getattr(module, name)(**options).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=True)
    return model.eval()


def bundle(args):
    root = Path(__file__).resolve().parents[1]
    if not (root/'models').is_dir():
        root = Path(__file__).resolve().parent
    with zipfile.ZipFile(args.bundle, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(__file__, 'evaluate_dataset.py')
        onnx = args.checkpoint.suffix.lower() == '.onnx'
        checkpoint_name = 'checkpoint.onnx' if onnx else 'checkpoint.pth'
        archive.write(args.checkpoint, checkpoint_name)
        archive.write(args.config, 'config/'+args.config.name)
        if not onnx:
            name = yaml.safe_load(args.config.read_text())['model']['name']
            for file in (args.config.parent/'model_config.yaml', model_source(args.config, name)):
                archive.write(file, 'config/'+file.name)
        radar = args.radar_config or args.config.parent/'radar_config.yaml'
        if radar.exists():
            archive.write(radar, 'config/radar_config.yaml')
        # 模型可能跨目录引用；仅打包模型源码，排除二进制、测试、训练入口。
        if not onnx:
            for file in (root/'models').rglob('*.py'):
                if not file.name.startswith('test_'):
                    archive.write(file, file.relative_to(root))
        documentation = Path(__file__).with_name('evaluate_dataset_README.md')
        if documentation.exists():
            archive.write(documentation, 'README.md')
        archive.writestr('requirements.txt', 'numpy\nPyYAML\ntorch\n'+
                          ('onnxruntime\n' if onnx else 'einops\n'))
        archive.writestr('RUN.txt', 'python evaluate_dataset.py --root DATA_ROOT --date YYYYMMDD '
                          '--group 033 --checkpoint '+checkpoint_name+' --config config/'+args.config.name+'\n'+
                          ('ONNX GPU 推理请安装 onnxruntime-gpu 并设置 --device cuda:0。\n' if onnx else
                           'P4/PCT 等模型需要与目标 PyTorch/CUDA 匹配的 pointnet2_ops。\n'))
    print(f'已打包：{args.bundle}')


def read_cloud(source, name):
    points = np.asarray(source.read(name), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 6:
        raise ValueError(f'点云必须为 [N,6] (xyz,velocity,power,tag)：{name}: {points.shape}')
    return points[np.isfinite(points).all(1)]


def summarize(records, rotation, translation):
    sums, count = np.zeros(3), 0
    for record in records:
        ids = record['pose_track_ids']
        record['metric_valid'] = np.zeros(len(ids), bool)
        for i, (person, pose) in enumerate(zip(ids, record['poses'])):
            gt = record['gt'][person]
            root_low = gt[0] @ rotation.T + translation
            inside = np.square((root_low[:2]-[2., 0.])/[1.6, 2.4]).sum() <= 1
            valid = (inside and record['pose_status'][int(person)] in ('accepted', 'confirmed', 'kept')
                     and np.isfinite(pose).all() and np.isfinite(gt).all())
            record['metric_valid'][i] = valid
            if valid:
                error = np.linalg.norm((pose-pose[:1])-(gt-gt[:1]), axis=-1)*1000
                sums += [error[part].mean() for part in PARTS.values()]
                count += 1
    return {**{key+'_mm': float(value/count) if count else None
               for key, value in zip(PARTS, sums)}, 'evaluated_person_frames': count}


def save_results(output, records, metrics, metadata, max_people):
    shape = (len(records), max_people)
    poses = {key: np.full((*shape, 21, 3), np.nan, np.float32) for key in ('raw', 'keeper', 'gt')}
    status = np.full(shape, 'absent', dtype='<U10')
    eligible = np.zeros(shape, bool)
    dynamic, total = np.zeros(shape, int), np.zeros(shape, int)
    for f, record in enumerate(records):
        poses['gt'][f, :len(record['gt'])] = record['gt']
        for ids, key, source in ((record['pose_indices'], 'raw', 'poses_raw'),
                                  (record['pose_track_ids'], 'keeper', 'poses')):
            poses[key][f, ids] = record[source]
        for person, value in record['pose_status'].items():
            status[f, person] = value
            dynamic[f, person] = record['dynamic_point_counts'][person]
            total[f, person] = record['total_point_counts'][person]
        eligible[f, record['pose_track_ids']] = record['metric_valid']
    offsets = np.cumsum([0]+[len(r['points']) for r in records])
    metadata.update(joint_names=NAMES, edges=EDGES, parts=PARTS, unit='m', coordinate='radar_high',
                    metric='error', prediction_stage='keeper_last',
                    person_ids='GT array slots; not detector track IDs', keeper=keeper_metric_protocol())
    np.savez_compressed(output/'predictions.npz', **poses, status=status, metric_valid=eligible,
                        dynamic_counts=dynamic, total_counts=total, points_offsets=offsets,
                        points=np.concatenate([r['points'] for r in records]),
                        frame_names=np.array([Path(r['path']).name for r in records]),
                        gt_names=np.array([r['gt_name'] for r in records]),
                        frame_reason=np.array([r['reason'] for r in records]),
                        high_to_low_R=metadata['rotation'], high_to_low_t=metadata['translation'],
                        metadata=json.dumps(metadata, ensure_ascii=False))
    values = ['N/A' if metrics[k+'_mm'] is None else f"{metrics[k+'_mm']:.3f}" for k in PARTS]
    return '\t'.join([metadata['date'], metadata['group'], *values])


@torch.inference_mode()
def evaluate(args):
    output = args.output or Path('results')/f'{args.date}_{args.group}'
    if output.exists():
        raise FileExistsError(f'结果目录已存在：{output}')
    config = yaml.safe_load(args.config.read_text())
    data, task = config['data'], config['task']
    precision = task.get('precision', 'FP32')
    dtypes = {'FP32': torch.float32, 'FP16': torch.float16, 'BF16': torch.bfloat16}
    if precision not in dtypes:
        raise ValueError('task.precision 必须为 FP32、FP16 或 BF16')
    is_pth = args.checkpoint.suffix.lower() == '.pth'
    autocast = torch.autocast(device_type=torch.device(args.device).type, dtype=dtypes[precision],
                              enabled=is_pth and precision != 'FP32')
    mode = args.input_format or ('bin' if 'bin' in task['input'] else 'pc')
    if mode == 'bin' and (task['input'] != 'radar_high_bin' or config['model']['name'] != 'ResNet3D'):
        raise ValueError('原始 BIN 当前支持 radar_high_bin → ResNet3D 的 DXYZ / DXYZ+KXYZ 模型')
    if mode == 'pc' and task['input'] != 'radar_high_pc':
        raise ValueError('PC 模式需要 radar_high_pc 模型')
    coordinate = task.get('coordinate', 'low' if task.get('center_xy_in_low') else task.get('coordinate_frame', 'high'))
    if coordinate not in ('high', 'low') or (mode == 'bin' and coordinate != 'high'):
        raise ValueError('coordinate 需要 high/low；BIN 频谱模型仅支持 high')
    if task.get('center_xy_in_low') and coordinate != 'low':
        raise ValueError('center_xy_in_low 要求 coordinate=low')
    if task.get('center_xy_in_low') and task.get('center_on_pointcloud'):
        raise ValueError('两种点云预处理选项不能同时开启')
    T, max_people = data['T'], data['max_people']
    if T < 1 or max_people < 1 or data.get('person_max_points', 200) < 1 or data.get('acc_frame', 0) < 0:
        raise ValueError('T、max_people、person_max_points 必须为正数，acc_frame 必须非负')
    group = args.root / args.date / 'data_collection' / args.group
    pc_path = args.pc_directory or ('fftGen/PC' if data.get('pointcloud_source') == 'fftgen' else 'dpct高位机/PC')
    clouds = Frames(group/pc_path, '.npy')
    frames = Frames(group/'dpct高位机/Bin', '.bin') if mode == 'bin' else clouds
    targets = Frames(group/'camera results/smoothed 3D', '.pkl')
    if not frames.names:
        raise FileNotFoundError(f'没有输入帧：{frames.directory}')
    calib = args.root/args.date/'calib'
    with np.load(calib/'extrinsic_img_to_radar_high.npz') as extrinsic:
        rh, th = extrinsic['R_est'], extrinsic['t_est'].reshape(3)
    with np.load(calib/'extrinsic_img_to_radar_low.npz') as extrinsic:
        rotation = extrinsic['R_est'] @ rh.T
        translation = extrinsic['t_est'].reshape(3)-rotation @ th
    for value in (rh, th, rotation, translation):
        if not np.isfinite(value).all() or value.shape not in ((3,), (3, 3)):
            raise ValueError('外参形状错误或含非有限数值')
    model = load_model(args, config)
    spectra = None
    if mode == 'bin':
        radar_path = args.radar_config or args.config.parent/'radar_config.yaml'
        radar = yaml.safe_load(radar_path.read_text())
        spectra = Spectra(radar, data, args.device)
    window, history = deque(maxlen=T), deque(maxlen=data.get('acc_frame', 0)+1)
    records = []
    for index, name in enumerate(frames.names[:args.max_frames or None]):
        time = timestamp_ns(name)
        record = dict(path=name, gt_name='', reason='', points=np.empty((0, 6), np.float32),
                      gt=np.empty((0, 21, 3), np.float32), track_ids=np.empty(0, int),
                      pose_indices=np.empty(0, int), poses_raw=np.empty((0, 21, 3), np.float32),
                      dynamic_point_counts={}, total_point_counts={})
        records.append(record)
        pc_index = clouds.nearest(time)
        if pc_index is not None and abs(int(clouds.times[pc_index])-time) <= args.max_sync_ms*1e6:
            record['points'] = read_cloud(clouds, clouds.names[pc_index])
        gt_index = targets.nearest(time)
        gt = np.empty((0, 17, 3), np.float32)
        if gt_index is not None and abs(int(targets.times[gt_index])-time) <= args.max_sync_ms*1e6:
            record['gt_name'] = targets.names[gt_index]
            gt = np.asarray(targets.read(record['gt_name']), np.float32)
            if not gt.size:
                gt = np.empty((0, 17, 3), np.float32)
            if gt.ndim == 2:
                gt = gt[None]
            if gt.ndim != 3 or gt.shape[-1] != 3 or gt.shape[-2] not in (17, 21) or len(gt) > max_people:
                raise ValueError(f'GT 格式或人数无效：{record["gt_name"]}: {gt.shape}')
            gt = gt @ rh.T + th
        valid_gt = np.isfinite(gt).all((1, 2))
        people = np.flatnonzero(valid_gt)
        record.update(gt=mocap(gt), track_ids=people)
        points = record['points']
        model_points, model_gt = points.copy(), gt.copy()
        if coordinate == 'low':
            model_points[:, :3] = points[:, :3] @ rotation.T + translation
            model_gt = gt @ rotation.T + translation
        history.append(model_points)
        accumulated = np.concatenate(history)
        inputs, masks, centers, boxes = [], [], [], []
        for person in range(len(gt)):
            box = np.r_[model_gt[person].min(0)-.3, model_gt[person].max(0)+.3]
            boxes.append(box)
            high_box = np.r_[gt[person].min(0)-.3, gt[person].max(0)+.3]
            selected = points[((points[:, :3] >= high_box[:3]) & (points[:, :3] <= high_box[3:])).all(1)]
            record['dynamic_point_counts'][person] = int(np.isclose(selected[:, -1], 1).sum())
            record['total_point_counts'][person] = len(selected)
            cropped = accumulated[((accumulated[:, :3] >= box[:3]) & (accumulated[:, :3] <= box[3:])).all(1)]
            sample, mask, center = sample_points(cropped, data.get('person_max_points', 200),
                                                  task.get('center_on_pointcloud', False), task.get('center_xy_in_low', False))
            inputs.append(sample)
            masks.append(mask)
            centers.append(center)
        frame = dict(people=people, inputs=inputs, masks=masks, centers=centers, boxes=boxes, spectra=None)
        if mode == 'bin':
            cube = decode_bin(frames.read(name), radar)
            frame['spectra'] = spectra(cube)
        window.append(frame)
        common = sorted(set.intersection(*(set(f['people']) for f in window)))
        ready = len(window) == T and (mode == 'pc' or all(f['spectra'] is not None for f in window))
        if not ready or not common:
            record['reason'] = 'warmup' if not ready else 'no_continuous_gt'
            continue
        ids = np.asarray([p for p in common
                          if record['dynamic_point_counts'][p] >= KEEPER_PARAMS['min_dynamic']
                          and record['total_point_counts'][p] >= KEEPER_PARAMS['min_total']], int)
        if len(ids) != len(common):
            record['reason'] = 'low_point_count'
        if not len(ids):
            continue
        if mode == 'pc':
            x = np.array([[f['inputs'][p] for f in window] for p in ids])
            mask = np.array([[f['masks'][p] for f in window] for p in ids])
            model_input = {'input': torch.as_tensor(x, device=args.device),
                           'mask': torch.as_tensor(mask, device=args.device)}
            with autocast:
                prediction = model(model_input)['pose']
            if prediction.shape[:3] != (len(ids), T, 1):
                raise ValueError(f'按人模型输出应为 [P,T,1,J,3]，实际 {prediction.shape}')
            prediction = prediction[:, -1, 0].float().cpu().numpy()+np.array(centers)[ids, None]
            if coordinate == 'low':
                prediction = (prediction-translation) @ rotation
        else:
            x = torch.stack([f['spectra'] for f in window])[None]
            boxes = torch.as_tensor(np.array([[f['boxes'][p] for p in ids] for f in window]),
                                    device=args.device, dtype=torch.float32)[None]
            with autocast:
                prediction = model({'input': x, 'bbox': boxes,
                                    'bbox_mask': torch.ones((1, T, len(ids)), dtype=torch.bool, device=args.device)})['pose']
            if prediction.shape[:3] != (1, T, len(ids)):
                raise ValueError(f'频谱模型输出应为 [1,T,P,J,3]，实际 {prediction.shape}')
            prediction = prediction[0, -1].float().cpu().numpy()
        record.update(pose_indices=ids, poses_raw=mocap(prediction))
        if (index+1) % 100 == 0:
            print(f'{index+1}/{len(frames.names)}', file=sys.stderr)
    keeper_pose_records(records, rotation, translation)
    metrics = summarize(records, rotation, translation)
    output.mkdir(parents=True, exist_ok=False)
    row = save_results(output, records, metrics,
                       dict(date=args.date, group=args.group, input_format=mode,
                            config=config, checkpoint=str(args.checkpoint), max_sync_ms=args.max_sync_ms,
                            inference_precision=precision if is_pth else 'ONNX graph',
                            rotation=rotation.tolist(), translation=translation.tolist()), max_people)
    print(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, help='数据根目录，下面为 日期/data_collection/group_XXX')
    parser.add_argument('--date')
    parser.add_argument('--group')
    parser.add_argument('--checkpoint', type=Path, help='.pth 使用 PyTorch；.onnx 使用 ONNX Runtime')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--radar-config', type=Path, help='BIN 雷达参数，默认 config 同目录 radar_config.yaml')
    parser.add_argument('--input-format', choices=('pc', 'bin'), help='默认根据 task.input 判断')
    parser.add_argument('--pc-directory', help='组目录内的点云子目录；BIN 模式用它判断质量')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-sync-ms', type=float, default=100., help='GT/质量点云与输入帧最大同步偏差')
    parser.add_argument('--max-frames', type=int, default=0, help='只跑前 N 帧用于检查；0 为整组')
    parser.add_argument('--output', type=Path, help='新建结果目录，已存在则拒绝覆盖')
    parser.add_argument('--bundle', type=Path, help='只打包可传递 ZIP，无需数据集')
    args = parser.parse_args()
    if not args.checkpoint or not args.config:
        parser.error('需要 --checkpoint 和 --config')
    if args.checkpoint.suffix.lower() not in ('.pth', '.onnx'):
        parser.error('--checkpoint 仅支持 .pth 或 .onnx')
    if args.bundle:
        bundle(args)
        return
    if not args.root or not args.date or not args.group:
        parser.error('需要 --root、--date 和 --group')
    if len(args.date) != 8 or not args.date.isdigit():
        parser.error('--date 格式应为 YYYYMMDD')
    number = args.group.removeprefix('group_')
    if not number.isdigit():
        parser.error('--group 格式应为 033 或 group_033')
    args.group = f'group_{int(number):03d}'
    if args.max_sync_ms <= 0 or not np.isfinite(args.max_sync_ms) or args.max_frames < 0:
        parser.error('--max-sync-ms 必须为有限正数，--max-frames 不能为负数')
    if args.output and args.output.exists():
        parser.error(f'结果目录已存在：{args.output}')
    torch.set_num_threads(4)
    evaluate(args)


if __name__ == '__main__':
    main()
