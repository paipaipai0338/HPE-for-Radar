"""按组滑窗查看原始点云、D/K 谱、RD、测角诊断，使用 WebAgg。

python deploy/inspect_radar_spectra.py --rd-frames 4
python deploy/inspect_radar_spectra.py --self-test

五列：前三列为点云、DXYZ、KXYZ 三维散点；后两列为原始 RD、方位/俯仰诊断。
动态为蓝色、微动为橙色、GT 为洋红色；谱图颜色越深功率越强，低于 -60 dB 的体素隐藏。
DXYZ 对每个 RD 单元用复数 DPK 选峰、俯仰测角，仅将有效峰功率分配到相邻 XYZ 体素。
KXYZ 使用相同的 DPK 筛峰与体素分配；功率为归一化方位峰值幅度平方。
支持 D/K/点索引滑条、GT/None/Points mask、stride 2 与 lags 1..6 历史切换。
跨帧固定 120 ms 帧周期、80 us chirp、4 TX、BW=1；插值空档不是实测数据。
跨帧相干性尚未验证；原始 BIN/GT 只读。
"""

import argparse
from functools import lru_cache
from copy import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess.gtprocess import get_gt_data, get_gt_boxes_list
from preprocess.radarprocess import (
    Radar_Config, get_bin_data,
    build_azimuth_elevation_directions, build_steering_matrix, conventional_dbf_power,
)
from preprocess.radarpreprocess_HRRadarPose import range_doppler_azi_ele_to_doppler_xyz
from preprocess.radarpreprocess_skyrelay import (
    AlgCfg, StructParam, SignalProcessConfig, StatsInfo, MAX_FRAME_NUM,
    StructBbCfarCfg0, cfarCalc, angleCalcD, framePoints2pc,
    range_cube_to_point_cloud, fft1dMeanCalc, fft2dCalc, fft2dAbsSum, fft3dProc_dense_pc, CfarRst, RadarUserCfg, validate_range_cube,
)

RANGE_RESOLUTION = 0.025
VELOCITY_RESOLUTION = 0.1252

BONES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
         (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
         (13, 15), (12, 14), (14, 16)]


def timestamp_ns(path):
    seconds, nanos = Path(path).stem.split('_')
    return int(seconds) * 10**9 + int(nanos)


def transform_gt(pose, rotation, translation):
    """所有后续 ROI 和二维绘图都使用变换后的雷达坐标。"""
    if pose.ndim != 3 or pose.shape[1:] != (17, 3):
        raise ValueError(f'GT 应为 [P,17,3]，实际为 {pose.shape}')
    return (pose @ rotation.T + translation).astype(np.float32)


def radial_velocity(pose, previous, dt):
    """雷达位于原点；缺失、多人、断点或零距离时不输出速度。"""
    result = np.full(pose.shape[:-1], np.nan, dtype=np.float32)
    # shortcut: GT 没有人员 ID，仅计算单人帧；多人需保留稳定的跟踪 ID 后再扩展。
    if pose.shape != (1, 17, 3) or previous.shape != pose.shape or not 0 < dt <= .18:
        return result
    distance = np.linalg.norm(pose, axis=-1)
    np.divide(np.sum((pose - previous) / dt * pose, axis=-1), distance,
              out=result, where=distance > 1e-6)
    result[~np.isfinite(result)] = np.nan
    return result


def sky_mean(cube, signal):
    fft1d = cube.transpose(1, 2, 0).reshape(64, 4, 4, 256)
    return fft1dMeanCalc(fft1d * signal.winVel.T[:, :, None, None]).astype(np.complex64)



def multiframe_rd(cubes, device='cpu'):
    """[N,R,64,16]: IQ 线性插值 FFT 与真实时间 DFT，共用频率轴和窗增益归一化。"""
    cubes = np.asarray(cubes)
    if (cubes.ndim != 4 or cubes.shape[0] < 2 or cubes.shape[2:] != (64, 16)
            or not np.iscomplexobj(cubes) or not np.isfinite(cubes).all()):
        raise ValueError('多帧 RD 需要有限复数 [N>=2,R,64,16]')
    dt = 4 * 80e-6
    times = (np.arange(len(cubes))[:, None] * .120 + np.arange(64) * dt).ravel()
    samples = cubes.transpose(0, 2, 1, 3).reshape(-1, cubes.shape[1], 16)
    samples = samples - samples.mean(axis=0, keepdims=True)
    uniform = np.arange(int(np.floor(times[-1] / dt)) + 1) * dt
    uniform[-1] = min(uniform[-1], times[-1])  # 浮点舍入可能让末点越过插值上界。
    nfft = 1 << (4 * len(uniform) - 1).bit_length()
    frequencies = np.fft.fftshift(np.fft.fftfreq(nfft, dt))
    # RD 取通道幅度；测角前另按物理 Doppler 频率补偿 TX 时间偏移。
    window = .5 - .5 * np.cos(2 * np.pi * uniform / times[-1])
    filled = interp1d(times, samples, axis=0, assume_sorted=True)(uniform)
    fft = np.fft.fftshift(np.fft.fft(filled * window[:, None, None], n=nfft, axis=0), axes=0)
    interpolated = (np.abs(fft).mean(axis=-1) / window.sum()).T.astype(np.float32)
    observed_window = .5 - .5 * np.cos(2 * np.pi * times / times[-1])
    data = torch.as_tensor((samples * observed_window[:, None, None]).reshape(len(times), -1),
                           dtype=torch.complex64, device=device)
    direct_spectrum = np.empty((nfft, cubes.shape[1], 16), dtype=np.complex64)
    for start in range(0, nfft, 128):
        kernel = torch.as_tensor(np.exp(-2j * np.pi * frequencies[start:start+128, None] * times),
                                 dtype=torch.complex64, device=device)
        transformed = (kernel @ data).reshape(-1, cubes.shape[1], 16)
        direct_spectrum[start:start+128] = (transformed / observed_window.sum()).cpu().numpy()
    wavelength = 3e8 / (56.4e9 + 6.4453e9 / 2)  # BW=1 中心频率。
    direct = np.abs(direct_spectrum).mean(-1).T
    compensation = np.exp(-2j * np.pi * frequencies[:, None] * np.repeat(np.arange(4), 4) * 80e-6)
    fft = (fft / window.sum() * compensation[:, None, :]).astype(np.complex64)
    direct_spectrum *= compensation[:, None, :]
    return interpolated, direct, frequencies * wavelength / 2, fft, direct_spectrum


def rd_spectrum_points(spectrum, velocity, cfg, signal, frames):
    """复用 Skyrelay CFAR、DPK 和俯仰测角，保留新谱的实际速度轴。"""
    amplitude = np.abs(spectrum).mean(-1)
    # 检测时跳过补零细化的相关 bin，CFAR 窗口以约 1/观测时长的频率步长计。
    stride = max(1, round(len(velocity) * .00032 / ((frames - 1) * .120 + 63 * .00032)))
    indices = np.arange(0, len(velocity), stride)
    cfar = StructBbCfarCfg0(cfg, StructParam())
    detections = cfarCalc(amplitude[indices], cfg, cfar)
    if not detections:
        return np.empty((0, 6), dtype=np.float32)
    for detection in detections:
        detection.chirpIdx = int(indices[int(detection.chirpIdx)])
    antennas = np.stack([spectrum[int(d.chirpIdx), int(d.rangeIdx)] for d in detections])
    angle_points = angleCalcD(antennas, detections, cfg, signal)
    local_cfg = copy(cfg)
    local_cfg.fft2dPT = len(velocity)
    cloud = framePoints2pc(angle_points, local_cfg, RANGE_RESOLUTION, float(velocity[1] - velocity[0]))
    return np.column_stack((cloud, np.ones(len(cloud)))).astype(np.float32)

def history_indices(index, history, gap=2):
    return [j for j in range(index - history, index + 1) if j % gap == 0][-MAX_FRAME_NUM:]


def valid_windows(good, times, history, max_gap_seconds=0.18):
    """坏帧和明显采集断点后重新预热，绝不跨断点对消。"""
    intervals = np.diff(times)
    if np.any(intervals <= 0):
        raise ValueError('BIN 时间戳必须严格递增')
    result, run = [], 0
    for i, valid in enumerate(good):
        if i and intervals[i - 1] > max_gap_seconds * 1e9:
            run = 0
        run = run + 1 if valid else 0
        if run >= history + 1 and len(history_indices(i, history)) >= 2:
            result.append(i)
    return result


class Spectra:
    def __init__(self, args, radar, sky):
        self.args, self.radar, self.sky = args, radar, sky
        device = torch.device(args.device)
        self.azimuth = torch.linspace(-np.pi / 2, np.pi / 2, args.angle_bins[0], device=device)
        self.elevation = torch.linspace(-np.pi / 2, np.pi / 2, args.angle_bins[1], device=device)
        self.range = torch.arange(256, device=device) * RANGE_RESOLUTION
        directions = build_azimuth_elevation_directions(self.azimuth, self.elevation)
        self.weights = build_steering_matrix(
            radar.virtual_channel_positions.to(device), directions, radar.lam, normalize=True)
        self.limits = np.asarray(args.xyz_limits).reshape(3, 2).tolist()
        self.axes = [np.linspace(*limits, count, dtype=np.float32)
                     for limits, count in zip(self.limits, args.cube_size)]
        self.grid = np.stack(np.meshgrid(*self.axes, indexing='ij'), axis=-1)

    def xyz(self, snapshots):
        # snapshots 为 [R,D或K,M]；复用 HRRadarPose 的 DBF 几何和 XYZ 插值。
        power = conventional_dbf_power(snapshots, self.weights)
        power = power.reshape(1, 1, 256, snapshots.shape[1],
                              len(self.azimuth), len(self.elevation))
        xyz, *_ = range_doppler_azi_ele_to_doppler_xyz(
            power, self.range, self.azimuth, self.elevation,
            self.limits, tuple(self.args.cube_size))
        return xyz[0, 0].cpu().numpy()

    def dpk_xyz(self, snapshots, signal):
        """每个 RD/RK 单元仅保留 DPK 有效峰，以平方幅度作为功率分配到 XYZ。"""
        data = snapshots.detach().cpu().numpy()
        ranges, dopplers, _ = data.shape
        detections = [CfarRst(r, d, 0) for r in range(ranges) for d in range(dopplers)]
        cfg = copy(self.sky)
        cfg.pc_gen_method = 1
        peaks = angleCalcD(data.reshape(-1, 16), detections, cfg, signal)
        result = np.zeros((dopplers, *self.args.cube_size), dtype=np.float32)
        if not peaks:
            return result
        # 复用点云坐标约定，速度列此处不使用，D bin 由检测索引保留。
        cloud = framePoints2pc(peaks, cfg, RANGE_RESOLUTION, VELOCITY_RESOLUTION)
        bins = np.array([int(p.d2Idx) for p in peaks if int(p.d1Idx) >= cfg.minRangeBin])
        power = (cloud[:, 4] / cfg.numAntA) ** 2
        return self.splat_peaks(cloud[:, :3], bins, power, dopplers)

    def splat_peaks(self, positions, bins, power, dopplers):
        """峰值功率按三线性权重分给相邻体素；范围外的峰直接丢弃。"""
        shape = np.asarray(self.args.cube_size)
        limits = np.asarray(self.limits)
        valid = ((positions >= limits[:, 0]) & (positions <= limits[:, 1])).all(axis=1)
        positions, bins, power = positions[valid], bins[valid], power[valid]
        coordinates = (positions - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) * (shape - 1)
        lower = np.floor(coordinates).astype(int)
        fraction = coordinates - lower
        result = np.zeros((dopplers, *shape), dtype=np.float32)
        for offset in np.ndindex(2, 2, 2):
            offset = np.asarray(offset)
            indices = np.minimum(lower + offset, shape - 1)
            weight = np.prod(np.where(offset, fraction, 1 - fraction), axis=1)
            np.add.at(result, (bins, indices[:, 0], indices[:, 1], indices[:, 2]), power * weight)
        return result

    @torch.inference_mode()
    def snapshots(self, cube, means, signal):
        fft1d = cube.transpose(1, 2, 0).reshape(64, 4, 4, 256)
        doppler = fft2dCalc(fft1d, fft1dMeanCalc(fft1d), RadarUserCfg(), signal, self.sky)
        rd = fft2dAbsSum(doppler).T.copy()
        doppler = doppler.reshape(64, 16, 256).transpose(2, 0, 1)
        velocity = (np.arange(64) - 32) * VELOCITY_RESOLUTION
        selected = abs(velocity) <= self.args.max_velocity
        gain = signal.winVel.sum(axis=1).repeat(4)
        doppler = torch.as_tensor((doppler[:, selected] / gain).astype(np.complex64),
                                 device=self.args.device)
        gain = signal.winVel.mean(axis=1).repeat(4)
        differences = np.stack([means[-1] - old for old in means[-2::-1]])
        snapshots = differences.reshape(-1, 16, 256).transpose(2, 0, 1)
        snapshots = torch.as_tensor((snapshots / gain).astype(np.complex64), device=self.args.device)
        return doppler, snapshots, velocity[selected], rd

    @torch.inference_mode()
    def process(self, cube, means, signal):
        doppler, difference, velocity, rd = self.snapshots(cube, means, signal)
        return self.dpk_xyz(doppler, signal), self.dpk_xyz(difference, signal), velocity, rd

    def masks(self, points, gt, margin):
        roi = np.zeros(self.grid.shape[:-1], dtype=bool)
        for box in get_gt_boxes_list([gt], threshold=margin)[0]:
            roi |= ((self.grid >= box[:3]) & (self.grid <= box[3:])).all(-1)
        # 空点云保持空 mask；界面可切回 GT/None 检查缺失证据。
        pc = np.zeros_like(roi)
        if len(points):
            distances = cKDTree(points[:, :3]).query(self.grid.reshape(-1, 3))[0]
            pc = distances.reshape(roi.shape) <= margin
        return {'None': np.ones_like(roi), 'GT': roi, 'Points': pc}


def spectrum_points(points, velocity, velocity_step, k):
    dynamic = ((points[:, 5] == 1)
               & (points[:, 3] >= velocity - velocity_step / 2)
               & (points[:, 3] < velocity + velocity_step / 2))
    # Skyrelay 标签 1 为动态点，2..7 按历史由近到远对应 K[0]..K[5]。
    return points[dynamic], points[points[:, 5] == k + 2]


class GroupReview:
    def __init__(self, args):
        self.args = args
        group = args.group_path.resolve()
        self.paths = sorted((group / args.sensor / 'Bin').glob('*.bin'), key=timestamp_ns)
        if not self.paths:
            raise FileNotFoundError(f'没有原始 BIN：{group / args.sensor / "Bin"}')
        self.times = np.array([timestamp_ns(p) for p in self.paths], dtype=np.int64)
        self.radar, self.sky = Radar_Config(), AlgCfg(StructParam())
        self.sky.frameGap = 2  # 保留旧流程供对比；默认显示每帧更新的 lags 1..6。
        self.signal = SignalProcessConfig()
        self.signal.init_winVel(self.sky)
        self.signal.init_steeringMatrix(self.sky)
        self.spectra = Spectra(args, self.radar, self.sky)
        bad = set()
        manifest = Path(__file__).resolve().parents[1] / 'data2datasets/bad_bin_frames.json'
        if manifest.exists():
            records = json.loads(manifest.read_text())['bad_frames']
            bad = {r['frame_name'] for r in records if
                   (r['date'], r['group'], r['sensor']) == (group.parent.parent.name, group.name, args.sensor)}
        good = [p.stat().st_size == 256 * 64 * 16 * 4 and p.name not in bad for p in self.paths]
        self.indices = valid_windows(good, self.times, max(args.history, 6, args.rd_frames - 1), self.sky.max_frame_gap_seconds)
        if not self.indices:
            raise ValueError('没有完整有效滑窗；检查 history、坏帧清单及采集间隔')
        if args.start_frame is not None:
            self.indices = [i for i in self.indices if i >= args.start_frame]
            if not self.indices:
                raise ValueError('start-frame 之后没有完整有效滑窗')
        self.gt_paths = sorted((group / 'camera results/smoothed 3D').glob('*.pkl'), key=timestamp_ns)
        if not self.gt_paths:
            raise FileNotFoundError('缺少原始 camera results/smoothed 3D/*.pkl')
        self.gt_times = np.array([timestamp_ns(p) for p in self.gt_paths], dtype=np.int64)
        level = 'high' if args.sensor == 'dpct高位机' else 'low'
        extrinsic = group.parent.parent / 'calib' / f'extrinsic_img_to_radar_{level}.npz'
        with np.load(extrinsic, allow_pickle=False) as data:
            self.rotation = np.asarray(data['R_est'], dtype=np.float32)
            self.translation = np.asarray(data['t_est'], dtype=np.float32).reshape(3)
        if self.rotation.shape != (3, 3) or not np.isfinite(self.rotation).all() or not np.isfinite(self.translation).all():
            raise ValueError('相机到雷达外参无效')
        self.cube = lru_cache(maxsize=max(args.history, 6, args.rd_frames - 1) + 1)(self.read_cube)
        self.frame = lru_cache(maxsize=3)(self.process_frame)
        self.point_trace = lru_cache(maxsize=32)(self.trace_point)
        print('History 对比：stride 2 使用最近偶数帧减步长 2 的历史；lags 1..6 使用当前帧减前 1～6 帧。', flush=True)
        print(f'{group.name}: {len(self.paths)} BIN, {len(self.indices)} 完整滑窗；'
              f'前 {max(args.history, 6, args.rd_frames - 1)} 帧预热，首个显示原始帧 {self.indices[0]}（从 0 编号）', flush=True)
        print(f'GT 外参：{extrinsic}；坐标变换后才投影；不施加扰动。', flush=True)
        if args.history > 12:
            print('注意：Skyrelay 最多缓存 7 个均值快照，history > 12 不增加差分数。', flush=True)

    def read_cube(self, index):
        cube = get_bin_data(self.paths[index], self.radar)
        try:
            return validate_range_cube(cube, self.sky)
        except ValueError as error:
            raise ValueError(f'无效 BIN，停止以免跨坏帧对消：{self.paths[index]} ({error})') from error

    def load_gt(self, index):
        nearest = int(np.argmin(np.abs(self.gt_times - self.times[index])))
        delta_ms = (int(self.gt_times[nearest]) - int(self.times[index])) / 1e6
        if abs(delta_ms) > self.args.gt_tolerance_ms:
            return np.empty((0, 17, 3), dtype=np.float32), delta_ms, np.empty((0, 17))
        gt = transform_gt(get_gt_data(self.gt_paths[nearest]), self.rotation, self.translation)
        previous = (transform_gt(get_gt_data(self.gt_paths[nearest - 1]), self.rotation, self.translation)
                    if nearest else np.empty((0, 17, 3)))
        dt = (int(self.gt_times[nearest]) - int(self.gt_times[nearest - 1])) / 1e9 if nearest else 0
        return gt, delta_ms, radial_velocity(gt, previous, dt)

    def trace_point(self, index, consecutive, point_index):
        record = self.frame(index)
        if consecutive:
            record = record['consecutive']
        point = record['points'][point_index]
        cfg = self.sky
        r = int(round(np.linalg.norm(point[:3]) / RANGE_RESOLUTION))
        d = int(round(point[3] / VELOCITY_RESOLUTION)) + 32
        kind = int(point[5])
        if kind == 1:
            fft1d = self.cube(index).transpose(1, 2, 0).reshape(64, 4, 4, 256)
            antennas = fft2dCalc(fft1d, fft1dMeanCalc(fft1d), RadarUserCfg(), self.signal, cfg)[d, :, :, r].reshape(1, 16)
        else:
            old = record['history_ids'][kind - 2]
            difference = sky_mean(self.cube(record['reference']), self.signal) - sky_mean(self.cube(old), self.signal)
            antennas = difference[:, :, r].reshape(1, 16)
        trace = []
        peaks, _ = fft3dProc_dense_pc(antennas, [CfarRst(r, d, 0)], cfg, trace=trace)
        # 坐标保存为 float32，用角度和峰值联合找回该点对应的 DPK 峰。
        azimuth_bin = int(round(np.sin(np.arctan2(point[1], point[0])) / cfg.azimuthSinSign * 64)) % 128
        candidates = [peak for peak in peaks if peak.d3Idx == azimuth_bin]
        if not candidates:
            raise ValueError('所选点无法匹配 DPK 峰')
        peak = min(candidates, key=lambda value: abs(value.pow3dAbs - point[4]))
        elevation_data = antennas[0, cfg.elevationAntOrder].reshape(cfg.numAntE, cfg.numRX)
        compensated = (elevation_data * self.signal.steeringMatrix[:, peak.d3Idx]).sum(axis=1)
        elevation = np.fft.fft(compensated, n=128)
        return trace, elevation, peak.d3Idx, r, d

    def process_frame(self, index):
        print(f'处理 {index}/{len(self.paths)-1}: {self.paths[index].name}', flush=True)
        # 未抽取为均值的奇数帧也须检查，不能跨过隐藏坏帧。
        for j in range(index - max(self.args.history, 6, self.args.rd_frames - 1), index + 1):
            self.cube(j)
        state = StatsInfo()
        state.frame_cnt = index
        ids = history_indices(index, self.args.history, self.sky.frameGap)
        for j in ids:
            if j < index:
                state.hist_fft1d_winvel_mean.append(sky_mean(self.cube(j), self.signal))
        points = range_cube_to_point_cloud(
            self.cube(index), alg_cfg=self.sky, stats_info=state,
            range_resolution=RANGE_RESOLUTION, velocity_resolution=VELOCITY_RESOLUTION,
            timestamp=self.times[index] / 1e9)
        means = list(state.hist_fft1d_winvel_mean)
        dxyz, kxyz, velocity, rd = self.spectra.process(self.cube(index), means, self.signal)
        rd_interpolated, rd_direct, rd_velocity, fft_spectrum, direct_spectrum = multiframe_rd(
            [self.cube(j) for j in range(index - self.args.rd_frames + 1, index + 1)],
            self.args.device)
        interpolated_points = rd_spectrum_points(fft_spectrum, rd_velocity, self.sky, self.signal, self.args.rd_frames)
        direct_points = rd_spectrum_points(direct_spectrum, rd_velocity, self.sky, self.signal, self.args.rd_frames)
        gt, delta_ms, gt_velocity = self.load_gt(index)
        masks = self.spectra.masks(points, gt, self.args.mask_margin)
        record = dict(interpolated_points=interpolated_points, direct_points=direct_points, rd_interpolated=rd_interpolated, rd_direct=rd_direct, rd_velocity=rd_velocity,
                    index=index, points=points, gt=gt, gt_delta_ms=delta_ms, gt_velocity=gt_velocity,
                    dxyz=dxyz, kxyz=kxyz, rd=rd, velocity=velocity, velocity_step=VELOCITY_RESOLUTION, masks=masks,
                    history_ids=ids[-2::-1], reference=ids[-1],
                    lag_ms=(self.times[ids[-1]] - self.times[ids[-2::-1]]) / 1e6)
        # 仅改变历史取样与参考帧，门限、测角及归一化与原流程相同。
        dense_cfg, dense_state = AlgCfg(StructParam()), StatsInfo()
        dense_cfg.frameGap = 1
        dense_ids = list(range(index - 6, index))
        for j in dense_ids:
            dense_state.hist_fft1d_winvel_mean.append(sky_mean(self.cube(j), self.signal))
        dense_points = range_cube_to_point_cloud(
            self.cube(index), alg_cfg=dense_cfg, stats_info=dense_state,
            range_resolution=RANGE_RESOLUTION, velocity_resolution=VELOCITY_RESOLUTION)
        _, difference, _, _ = self.spectra.snapshots(
            self.cube(index), list(dense_state.hist_fft1d_winvel_mean), self.signal)
        with torch.inference_mode():
            dense_kxyz = self.spectra.dpk_xyz(difference, self.signal)
        record['consecutive'] = dict(
            record, points=dense_points, kxyz=dense_kxyz,
            masks=self.spectra.masks(dense_points, gt, self.args.mask_margin),
            reference=index, history_ids=dense_ids[::-1],
            lag_ms=(self.times[index] - self.times[dense_ids[::-1]]) / 1e6)
        return record


class Viewer:
    def __init__(self, review):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, RadioButtons, Button
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        self.review, self.position = review, 0
        record = review.frame(review.indices[0])
        self.references = [max(float(record['dxyz'].max()), 1e-20),
                           max(float(record['kxyz'].max()), float(record['consecutive']['kxyz'].max()), 1e-20)]
        self.record = record['consecutive']
        self.rd_reference = max(float(record['rd'].max()), 1e-20)
        self.columns = 5
        self.fig, self.axes = plt.subplots(2, self.columns, figsize=(26, 9))
        grid = self.axes[0, 0].get_gridspec()
        self.scatter_axes = []
        for col in range(3):
            for ax in self.axes[:, col]:
                ax.remove()
            self.scatter_axes.append(self.fig.add_subplot(grid[:, col], projection='3d'))
        self.pc_axis, self.d_axis, self.k_axis = self.scatter_axes
        self.fig.subplots_adjust(bottom=.24, top=.86, right=.93, wspace=.25, hspace=.3)
        self.fig.colorbar(ScalarMappable(norm=Normalize(-60, 10), cmap='viridis'),
                          cax=self.fig.add_axes([.95, .32, .012, .42]), label='RD dB / fixed reference')
        self.frame_slider = Slider(self.fig.add_axes([.12, .15, .7, .025]), 'Window',
                                   0, max(1, len(review.indices)-1), valinit=0, valstep=1)
        self.frame_slider.set_active(len(review.indices) > 1)
        d = len(self.record['velocity'])
        self.d_slider = Slider(self.fig.add_axes([.12, .10, .3, .025]), 'D bin',
                               0, max(1, d-1), valinit=int(np.argmin(abs(self.record['velocity']))), valstep=1)
        self.k_slider = Slider(self.fig.add_axes([.53, .10, .29, .025]), 'K index',
                               0, max(1, len(self.record['kxyz'])-1), valinit=0, valstep=1)
        self.d_slider.set_active(d > 1)
        self.radio = RadioButtons(self.fig.add_axes([.86, .04, .1, .14]), ['GT', 'None', 'Points'], active=1)
        self.history_radio = RadioButtons(self.fig.add_axes([.48, .005, .32, .075]), ['stride 2', 'lags 1..6'], active=1)
        self.history_radio.on_clicked(lambda label: self.change_frame(self.position))
        self.buttons = []
        for left, label, step in ((.12, 'Previous', -1), (.24, 'Next', 1)):
            button = Button(self.fig.add_axes([left, .035, .1, .035]), label)
            button.on_clicked(lambda event, step=step: self.move(step))
            self.buttons.append(button)
        self.point_slider = Slider(self.fig.add_axes([.12, .195, .7, .025]), 'Point index',
                                   0, max(1, len(self.record['points'])-1), valinit=0, valstep=1)
        self.point_slider.set_active(len(self.record['points']) > 1)
        self.frame_slider.on_changed(self.change_frame)
        for widget in (self.d_slider, self.k_slider, self.point_slider):
            widget.on_changed(lambda value: self.draw())
        self.radio.on_clicked(lambda label: self.draw())
        self.fig.canvas.mpl_connect('key_press_event', self.key)
        self.draw()

    def move(self, step):
        self.frame_slider.set_val(int(np.clip(self.position + step, 0, len(self.review.indices)-1)))

    def key(self, event):
        if event.key in ('right', ' ', 'left'):
            self.move(-1 if event.key == 'left' else 1)

    def change_frame(self, value):
        position = min(int(value), len(self.review.indices)-1)
        try:
            record = self.review.frame(self.review.indices[position])
            if self.history_radio.value_selected == 'lags 1..6':
                record = record['consecutive']
        except (ValueError, OSError) as error:
            self.fig.suptitle(f'Frame unavailable: {error}', color='red')
            self.fig.canvas.draw_idle()
            return
        self.position, self.record = position, record
        for slider, count in ((self.point_slider, len(record['points'])), (self.k_slider, len(record['kxyz']))):
            slider.valmax = max(1, count-1)
            slider.ax.set_xlim(0, max(1, count-1))
            slider.set_active(count > 1)
            slider.eventson = False
            slider.set_val(min(int(slider.val), max(0, count-1)))
            slider.eventson = True
        self.draw()

    def draw(self):
        record, spectra = self.record, self.review.spectra
        d = min(int(self.d_slider.val), len(record['velocity'])-1)
        k = min(int(self.k_slider.val), len(record['kxyz'])-1)
        selected = min(int(self.point_slider.val), len(record['points'])-1)
        mask = record['masks'][self.radio.value_selected]
        points = record['points']
        for col, ax in enumerate(self.scatter_axes):
            ax.clear()
            if col == 0:
                for valid, color, marker, name in (
                    (points[:, 5] == 1, 'tab:blue', 'o', 'dynamic'),
                    (points[:, 5] > 1, 'tab:orange', '^', 'micro-motion'),
                ):
                    if valid.any():
                        ax.scatter(*points[valid, :3].T, c=color, marker=marker, s=15,
                                   depthshade=False, label=name)
                title = f'Skyrelay points (unmasked)\n{len(points)} points'
            else:
                volume = record['dxyz'][d] if col == 1 else record['kxyz'][k]
                reference = self.references[col-1]
                valid = mask & np.isfinite(volume) & (volume > 0)
                positions = spectra.grid[valid]
                db = 10*np.log10(volume[valid]/reference)
                # 低于显示色标下限的体素不画，避免稠密 K 谱遮挡有效区域。
                visible = db >= -60
                if visible.any():
                    ax.scatter(*positions[visible].T, c=db[visible],
                               cmap='Blues' if col == 1 else 'Oranges', vmin=-60, vmax=10,
                               s=12, alpha=.65, depthshade=False,
                               label='dynamic power (darker=stronger)' if col == 1 else 'micro power (darker=stronger)')
                if col == 1:
                    title = f'DPK D[{d}]: {record["velocity"][d]:+.3f} m/s\n{visible.sum()} voxels'
                else:
                    title = f'K[{k}]: {record["reference"]} - {record["history_ids"][k]}\n{record["lag_ms"][k]:.0f} ms | {visible.sum()} voxels'
            for i, pose in enumerate(record['gt']):
                ax.scatter(*pose.T, s=12, c='magenta', label='GT' if i == 0 else None)
                for a, b in BONES:
                    ax.plot(*pose[[a, b]].T, c='magenta', lw=1)
            if selected >= 0:
                ax.scatter(*points[selected, :3], s=75, facecolors='none', edgecolors='red')
            ax.set(title=title, xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)',
                   xlim=spectra.limits[0], ylim=spectra.limits[1], zlim=spectra.limits[2])
            ax.set_box_aspect(np.diff(np.asarray(spectra.limits), axis=1).ravel())
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7, loc='upper right')

        diagnostic, error = None, None
        if selected >= 0:
            try:
                diagnostic = self.review.point_trace(record['index'], self.history_radio.value_selected == 'lags 1..6', selected)
            except ValueError as exc:
                error = str(exc)
        for row in range(2):
            for col in (3, 4):
                ax = self.axes[row, col]
                ax.clear()
                if col == 4:
                    if diagnostic is None:
                        ax.set_title(error or 'No point available')
                        continue
                    trace, elevation, azimuth_bin, r, doppler_bin = diagnostic
                    if row == 0:
                        for step, residual in enumerate(trace):
                            ax.plot(abs(residual[0]), label='Azimuth FFT' if step == 0 else f'after DPK {step}')
                        ax.axhline(abs(trace[-1][0]).mean()*self.review.sky.dpkThres, color='red', linestyle='--', label='final noise x DPK threshold')
                        ax.axvline(azimuth_bin, color='magenta', linestyle=':')
                        ax.set_title(f'Point {selected}: R[{r}] D[{doppler_bin}] type={int(record["points"][selected, 5])}')
                    else:
                        ax.plot(abs(elevation), label='Elevation FFT (azimuth compensated)')
                        ax.axvline(int(abs(elevation).argmax()), color='magenta', linestyle=':')
                        ax.set_title(f'Elevation FFT | azimuth [{azimuth_bin}]')
                    ax.set(xlabel='FFT bin (unshifted)', ylabel='Amplitude', xlim=(0, 127))
                    ax.legend(fontsize=7)
                    continue
                if col == 3:
                    velocity = (np.arange(64)-32)*VELOCITY_RESOLUTION
                    db = 20*np.log10(np.maximum(record['rd'], 1e-20)/self.rd_reference)
                    ax.pcolormesh(velocity, np.arange(256)*RANGE_RESOLUTION, db, shading='nearest', vmin=-60, vmax=10, cmap='viridis')
                    ax.axvspan(record['velocity'][d]-VELOCITY_RESOLUTION/2, record['velocity'][d]+VELOCITY_RESOLUTION/2, color='yellow', alpha=.15)
                    ax.set_title('RD amplitude [256,64]' if row == 0 else 'RD + points / GT')
                    if row == 1:
                        points = record['points']
                        for valid, color, marker, name in ((points[:, 5] == 1, 'cyan', 'o', 'dynamic'), (points[:, 5] > 1, 'orange', '^', 'micro (v=0 placeholder)')):
                            if valid.any():
                                ax.scatter(points[valid, 3], np.linalg.norm(points[valid, :3], axis=1), s=18, facecolors='none', edgecolors=color, marker=marker, label=name)
                        valid = np.isfinite(record['gt_velocity'])
                        if valid.any():
                            ax.scatter(record['gt_velocity'][valid], np.linalg.norm(record['gt'], axis=-1)[valid], s=16, c='magenta', label='GT radial velocity')
                        if selected >= 0:
                            point = points[selected]
                            ax.scatter(point[3], np.linalg.norm(point[:3]), s=75, facecolors='none', edgecolors='red')
                        if ax.get_legend_handles_labels()[0]:
                            ax.legend(fontsize=7)
                    ax.set(xlabel='Radial velocity (m/s)', ylabel='Range (m)',
                           xlim=(velocity[0]-VELOCITY_RESOLUTION/2, velocity[-1]+VELOCITY_RESOLUTION/2),
                           ylim=(-RANGE_RESOLUTION/2, 255.5*RANGE_RESOLUTION))
                    continue
        self.fig.suptitle(f'{self.review.args.group_path.name} | frame {record["index"]} | {self.review.paths[record["index"]].stem}\n'
                         f'{len(record["points"])} points | {len(record["gt"])} GT, offset {record["gt_delta_ms"]:+.1f} ms | mask={self.radio.value_selected} | {self.history_radio.value_selected}\n'
                         f'DXYZ={record["dxyz"].shape}, KXYZ={record["kxyz"].shape}, RD={record["rd"].shape}; 3D: blue=dynamic, orange=micro, magenta=GT; fixed dB references', fontsize=10)
        self.fig.canvas.draw_idle()


def self_test():
    args = parse_args([])
    args.device, args.angle_bins, args.cube_size = 'cpu', [16, 8], [12, 12, 8]
    cfg, signal = AlgCfg(StructParam()), SignalProcessConfig()
    signal.init_winVel(cfg)
    signal.init_steeringMatrix(cfg)
    cube = np.ones((256, 64, 16), dtype=np.complex64)
    mean = sky_mean(cube, signal)
    assert np.allclose(mean[:, 0, 0], signal.winVel.mean(1))
    spectra = Spectra(args, Radar_Config(), cfg)
    dxyz, kxyz, velocity, rd = spectra.process(cube, [mean, mean], signal)
    assert dxyz.shape == (len(velocity), 12, 12, 8) and kxyz.shape == (1, 12, 12, 8)
    assert rd.shape == (256, 64) and np.count_nonzero(rd) == 0
    assert np.count_nonzero(kxyz) == 0 and np.count_nonzero(dxyz) == 0
    assert np.all(np.diff(velocity) > 0) and np.any(velocity == 0)
    # 体素分配守恒：内部与边界峰均不漏能量，外部峰不进入图中。
    limits = np.asarray(spectra.limits)
    positions = np.array([limits.mean(1), limits[:, 1], limits[:, 1]+1])
    deposited = spectra.splat_peaks(positions, np.array([0, 1, 0]), np.array([2., 3., 99.]), 2)
    np.testing.assert_allclose(deposited.sum(axis=(1, 2, 3)), [2, 3], atol=1e-6)
    # 单个 RD 单元的同相阵列只应在正前方产生稀疏峰，D bin 不串扰。
    sparse_input = torch.zeros((256, 2, 16), dtype=torch.complex64)
    sparse_input[80, 1] = 1
    sparse_xyz = spectra.dpk_xyz(sparse_input, signal)
    assert not sparse_xyz[0].any() and np.count_nonzero(sparse_xyz[1]) <= 8
    np.testing.assert_allclose(sparse_xyz[1].sum(), 1, atol=1e-6)

    tone = cube * np.exp(2j * np.pi * np.arange(64)[None, :, None] / 64).astype(np.complex64)
    shifted, _, _, tone_rd = spectra.process(tone, [mean, mean], signal)
    assert np.argmax(shifted.max(axis=(1, 2, 3))) == np.argmin(abs(velocity)) + 1
    _, changed, _, _ = spectra.process(cube, [mean, 2*mean], signal)
    assert changed.max() > 0 and np.isfinite(changed).all()
    # 历史由近到远的差分幅度为 2、3，筛峰后的 K 功率比应为 4:9。
    _, sparse_k, _, _ = spectra.process(cube, [mean, 2*mean, 4*mean], signal)
    assert sparse_k.shape == (2, 12, 12, 8)
    assert sparse_k[0].sum() > 0 and np.isfinite(sparse_k).all()
    np.testing.assert_allclose(sparse_k[1], sparse_k[0]*2.25, rtol=1e-5, atol=1e-6)

    doppler, difference, bins, rd = spectra.snapshots(tone, [mean, 2*mean, 4*mean], signal)
    np.testing.assert_allclose(difference[:, 0].cpu(), 2, atol=1e-6)
    np.testing.assert_allclose(difference[:, 1].cpu(), 3, atol=1e-6)
    _, consecutive, _, _ = spectra.snapshots(tone, [i * mean for i in range(7)], signal)
    for k in range(6):
        np.testing.assert_allclose(consecutive[:, k].cpu(), k + 1, atol=1e-6)
    # 图中复数 D 快照须与点云入口的 FFT、窗口及 TDM 补偿完全一致。
    fft1d = tone.transpose(1, 2, 0).reshape(64, 4, 4, 256)
    expected = fft2dCalc(fft1d, fft1dMeanCalc(fft1d), RadarUserCfg(), signal, cfg)
    np.testing.assert_allclose(rd, fft2dAbsSum(expected).T, rtol=1e-6)
    assert np.argmax(tone_rd.max(axis=0)) == 33
    expected = expected.reshape(64, 16, 256).transpose(2, 0, 1)
    selected = abs((np.arange(64) - 32) * VELOCITY_RESOLUTION) <= args.max_velocity
    expected = expected[:, selected] / signal.winVel.sum(1).repeat(4)
    np.testing.assert_allclose(doppler.cpu(), expected, rtol=1e-6, atol=1e-7)
    # 非单位外参检验：必须先旋转/平移三维关键点，再取 XY/XZ。
    pose = np.tile([[[1., 2., 3.]]], (1, 17, 1))
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    transformed = transform_gt(pose, rotation, np.array([3., 4., 5.]))
    assert np.allclose(transformed[0, 0], [1., 5., 8.])
    moving = np.tile([[[2., 0., 0.]]], (1, 17, 1))
    np.testing.assert_allclose(radial_velocity(moving, moving - [0.1, 0, 0], .1), 1, atol=1e-5)
    np.testing.assert_allclose(radial_velocity(moving, moving + [0.1, 0, 0], .1), -1, atol=1e-5)
    np.testing.assert_allclose(radial_velocity(moving, moving - [0, .1, 0], .1), 0)
    for previous, dt in ((moving, 0), (moving, .2), (moving[:0], .1)):
        assert np.isnan(radial_velocity(moving, previous, dt)).all()
    assert np.isnan(radial_velocity(moving * 0, moving, .1)).all()
    assert np.isnan(radial_velocity(np.repeat(moving, 2, axis=0), np.repeat(moving, 2, axis=0), .1)).all()
    antennas = np.ones((1, 16), dtype=np.complex64)
    trace = []
    detections = [CfarRst(10, 32, 0)]
    traced, ids = fft3dProc_dense_pc(antennas, detections, cfg, trace=trace)
    ordinary, ordinary_ids = fft3dProc_dense_pc(antennas, detections, cfg)
    assert traced == ordinary and ids == ordinary_ids
    assert len(trace) == cfg.dpkTimes + 1
    np.testing.assert_allclose(trace[0][0], np.fft.fft(np.ones(cfg.numAntA), n=cfg.numAngle))
    for before, after in zip(trace, trace[1:]):
        assert after[0, abs(before[0]).argmax()] == 0
    good = [True]*30
    good[15] = False
    indices = valid_windows(good, np.arange(30)*100_000_000, 4)
    assert indices[0] == 4 and not set(range(15, 20)).intersection(indices)
    jumped = np.arange(30)*100_000_000
    jumped[15:] += 1_000_000_000
    assert not set(range(15, 19)).intersection(valid_windows([True]*30, jumped, 4))
    assert not valid_windows([True]*8, np.arange(8)*200_000_000, 4)
    assert history_indices(12, 12) == [0, 2, 4, 6, 8, 10, 12]
    assert history_indices(13, 12)[-1] == 12  # 沿用 Skyrelay 奇数帧参考前一帧。
    masks = spectra.masks(np.empty((0, 6)), np.empty((0, 17, 3)), .3)
    assert masks['None'].all() and not masks['GT'].any() and not masks['Points'].any()
    points = np.zeros((8, 6), dtype=np.float32)
    points[:, 3] = [-1, -.5, 0, .5, 1, 0, 0, 0]
    points[:, 5] = [1, 1, 1, 1, 1, 2, 3, 7]
    dynamic, micro = spectrum_points(points, 0, 1, 0)
    np.testing.assert_array_equal(dynamic, points[[1, 2]])
    np.testing.assert_array_equal(micro, points[[5]])
    for velocity, index in ((-1, 0), (1, 4)):
        dynamic, micro = spectrum_points(points, velocity, .25, 5)
        np.testing.assert_array_equal(dynamic, points[[index]])
        np.testing.assert_array_equal(micro, points[[7]])
    assert all(len(p) == 0 for p in spectrum_points(points, 5, .25, 2))
    assert all(p.shape == (0, 6) for p in spectrum_points(points[:0], 0, .25, 0))
    # 连续相位慢速单音：非均匀 DFT 应找回真实速度；两方法共用频率网格。
    times = np.arange(2)[:, None] * .120 + np.arange(64) * .00032
    wave = np.exp(2j * np.pi * 20 * times)[:, None, :, None] * np.ones((2, 2, 64, 16))
    filled, direct, vr, fft_complex, direct_complex = multiframe_rd(wave)
    assert filled.shape == direct.shape == (2, len(vr)) and np.isfinite(filled).all()
    wavelength = 3e8 / (56.4e9 + 6.4453e9 / 2)
    assert abs(vr[direct[0].argmax()] - 20 * wavelength / 2) <= np.diff(vr)[0]
    zeros, static, *_ = multiframe_rd(np.ones_like(wave))
    assert not zeros.any() and not static.any()
    # 独立 NumPy 核校验真实时间 DFT 的幅度和归一化。
    f = vr * 2 / wavelength
    centered = wave[:, 0, :, 0].ravel() - wave[:, 0, :, 0].mean()
    w = .5 - .5 * np.cos(2 * np.pi * times.ravel() / times.max())
    expected = abs(np.exp(-2j * np.pi * f[:, None] * times.ravel()) @ (centered * w)) / w.sum()
    np.testing.assert_allclose(direct[0], expected, atol=2e-6)
    peak = int(direct[0].argmax())
    expected_phase = np.exp(-2j*np.pi*f[peak]*np.repeat(np.arange(4), 4)*80e-6)
    np.testing.assert_allclose(direct_complex[peak, 0] / direct_complex[peak, 0, 0],
                               expected_phase, atol=2e-6)
    np.testing.assert_allclose(abs(fft_complex).mean(-1).T, filled, atol=2e-6)

    # 四帧末点曾因 0.38016000000000005 > 0.38016 导致插值越界。
    times4 = np.arange(4)[:, None] * .120 + np.arange(64) * .00032
    wave4 = np.broadcast_to(np.exp(2j * np.pi * 20 * times4)[:, None, :, None], (4, 1, 64, 16))
    filled4, direct4, vr4, *_ = multiframe_rd(wave4)
    assert filled4.shape == direct4.shape == (1, len(vr4))
    assert np.isfinite(filled4).all() and np.isfinite(direct4).all()
    # 测角链空输入及同相阵列的正前方、速度映射。
    signal.init_steeringMatrix(cfg)
    test_spectrum = np.zeros((64, 256, 16), dtype=np.complex64)
    test_velocity = (np.arange(64)-32)*.01
    assert rd_spectrum_points(test_spectrum, test_velocity, cfg, signal, 2).shape == (0, 6)
    test_spectrum[37, 40] = 100
    cloud = rd_spectrum_points(test_spectrum, test_velocity, cfg, signal, 2)
    assert len(cloud) and np.isfinite(cloud).all()
    np.testing.assert_allclose(cloud[:, :3], np.tile([1, 0, 0], (len(cloud), 1)), atol=1e-6)
    np.testing.assert_allclose(cloud[:, 3], .05, atol=1e-6)
    print('Self-test passed: temporal windows, cancellation, D/K axes and point selection, GT transform/radial velocity, full RD amplitude, empty masks, multi-frame RD and angle/velocity mapping.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--group-path', type=Path, default=Path('/mnt/huawei/20260722/data_collection/group_034'))
    parser.add_argument('--sensor', choices=['dpct高位机', 'dpct低位机'], default='dpct高位机')
    parser.add_argument('--history', type=int, default=12, help='前序原始帧数；Skyrelay 最多保存 7 个快照')
    parser.add_argument('--rd-frames', type=int, default=2, help='跨帧 RD 的总帧数，含当前帧；120 ms 帧周期、80 us chirp、4 TX')
    parser.add_argument('--start-frame', type=int, help='从 0 开始的原始帧编号，仍读取之前的历史窗口')
    parser.add_argument('--max-velocity', type=float, default=1.5, help='保留 |v| <= 此值的每个 Doppler bin，m/s')
    parser.add_argument('--cube-size', type=int, nargs=3, default=[64, 64, 32])
    parser.add_argument('--angle-bins', type=int, nargs=2, default=[128, 32])
    parser.add_argument('--xyz-limits', type=float, nargs=6, default=[0, 6, -3, 3, -2, 2])
    parser.add_argument('--mask-margin', type=float, default=.3, help='GT 外扩/点云空间膨胀半径，m')
    parser.add_argument('--gt-tolerance-ms', type=float, default=100)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--port', type=int, default=8988)
    parser.add_argument('--check-frames', type=int, default=0, help='无界面处理指定数量的滑窗并打印形状')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args(argv)
    if args.rd_frames < 2:
        parser.error('rd-frames 必须 >= 2')
    if args.history < 2 or min(*args.cube_size, *args.angle_bins) < 2:
        parser.error('history 和各网格维度必须 >= 2')
    values = [args.max_velocity, args.mask_margin, args.gt_tolerance_ms]
    if not np.isfinite(values + args.xyz_limits).all() or min(values) <= 0:
        parser.error('速度范围、mask 半径、GT 容差必须为有限正数，空间范围必须有限')
    if any(lo >= hi for lo, hi in np.asarray(args.xyz_limits).reshape(3, 2)):
        parser.error('XYZ 范围必须递增')
    if args.check_frames < 0 or (args.start_frame is not None and args.start_frame < 0) or not 1 <= args.port <= 65535:
        parser.error('帧编号/检查数量必须非负，port 必须为 1..65535')
    return args


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    review = GroupReview(args)
    if args.check_frames:
        for index in review.indices[:args.check_frames]:
            record = review.frame(index)
            assert np.isfinite(record['dxyz']).all() and np.isfinite(record['kxyz']).all()
            for key in ('rd_interpolated', 'rd_direct'):
                assert record[key].shape == (256, len(record['rd_velocity'])) and np.isfinite(record[key]).all()
            print(f'Multi-frame RD: N={args.rd_frames}, shape={record["rd_direct"].shape}', flush=True)
            alternative = record['consecutive']
            assert alternative['kxyz'].shape[0] == 6 and np.isfinite(alternative['kxyz']).all()
            print(f'lags 1..6: points={len(alternative["points"])}, KXYZ={alternative["kxyz"].shape}, '
                  f'frames={alternative["reference"]}-{alternative["history_ids"]}', flush=True)
            print(f'frame={index}, DXYZ={record["dxyz"].shape}, KXYZ={record["kxyz"].shape}, '
                  f'points={len(record["points"])}, GT={len(record["gt"])}, '
                  f'GT offset={record["gt_delta_ms"]:+.1f} ms, '
                  f'GT radial velocities (m/s)={record["gt_velocity"].round(3).tolist()}', flush=True)
        return
    import matplotlib
    matplotlib.use('WebAgg')
    matplotlib.rcParams.update({'webagg.port': args.port, 'webagg.address': '127.0.0.1',
                                'webagg.open_in_browser': False})
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_webagg import WebAggApplication
    WebAggApplication.initialize()
    viewer = Viewer(review)
    print(f'WebAgg: http://{WebAggApplication.address}:{WebAggApplication.port}/；方向键/按钮逐帧，滑条切换 D/K，右侧切换 mask。', flush=True)
    plt.show()


if __name__ == '__main__':
    main()
