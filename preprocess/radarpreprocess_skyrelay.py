"""Skyrelay 单帧距离 FFT -> 点云，仅依赖 NumPy。

输入与 data2datasets.dataset 的 get_bin_data 一致：复数 [range, chirp, antenna]，
默认 [256, 64, 16]；已经做过距离 FFT，不再对距离维执行 FFT。
输出 float32 [N, 6]：x/y/z (m)、径向速度 (m/s)、方位峰值幅值、点类型。
类型 1 为动态点，2～7 为六组历史差分微动点；微动速度为 0；第五列为未截位的方位峰值幅值，不是 SNR。
x 向前、y 横向、z 向上；无检测点时返回 [0, 6]。每次调用处理一帧，微动检测需复用历史缓存。

用法：
    from preprocess.radarprocess import Radar_Config, get_bin_data
    from preprocess.radarpreprocess_skyrelay import StatsInfo, range_cube_to_point_cloud
    frame = get_bin_data(bin_path, Radar_Config())
    state = StatsInfo()  # 每个连续雷达序列创建一次
    points = range_cube_to_point_cloud(frame, stats_info=state)

Dataset 的 samples['radar_high_bin'] 为 [T,R,C,M]（批处理后 [B,T,R,C,M]），
需按时间顺序逐帧调用，序列内复用 state：
[range_cube_to_point_cloud(frame, stats_info=state) for frame in sequence]。
不同雷达、不同序列和不同 batch 样本不能共用 state；缓存不足两帧时只输出动态点。
Torch 单帧可先 frame.detach().cpu().numpy()；输出点数可变，不做补点或截断。

算法整合自 temp/wscpcgen：一维 SO-CFAR、复数 sinc 减峰、Q13 俯仰导向矢量。
微动检测默认每帧保存均值，当前帧分别减前 1～6 帧，TLV2～7 按历史由近到远排列。
默认分辨率沿用 skyrelay 的 0.025 m / 0.1252 m/s，可按实际标定覆盖。

from preprocess.radarpreprocess_skyrelay import range_cube_to_point_cloud

# bin_data 是 dataset.py 中 get_bin_data 返回的单帧数据
point_cloud = range_cube_to_point_cloud(bin_data)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Tuple
from numpy.lib.stride_tricks import sliding_window_view
import math

import numpy as np

NUM_TX = 4
NUM_RX = 4
NUM_CHIRP = 64
NUM_SAMP = 512
MAX_FRAME_NUM = 7


class StatsInfo:
    """同一雷达连续序列的历史缓存；新序列需新建实例。"""

    def __init__(self):
        self.hist_fft1d_winvel_mean = deque(maxlen=MAX_FRAME_NUM)
        self.reset()

    def reset(self):
        self.frame_cnt = 0
        self.last_timestamp = None
        self.hist_fft1d_winvel_mean.clear()


@dataclass
class StructParam:
    cfarThVG: int = 4
    cfarThVS: int = 5
    cfarThV: int = 4
    cfarVmode: int = 1  # Keil: 0=GO, 1=SO, 2=CA; 3=OS 软件扩展
    cfarThS: float = 8
    peakenV: int = 0
    dpkThres: int = 8
    dpkTimes: int = 2
    dpkRadius: int = 1


class AlgCfg:

    def __init__(self, stFlashParam: StructParam):
        self.numChirp: int = NUM_CHIRP
        self.numTX: int = NUM_TX
        self.numRX: int = NUM_RX
        self.numSamp: int = NUM_SAMP
        self.numVel: int = NUM_CHIRP
        self.numAngle: int = 128
        self.useRange: int = self.numSamp // 2
        self.fft2dPT: int = self.__cal_fft2dPT()
        self.rangePeakRatio: float = 0.7
        self.numAntA = 8
        self.numAntE = 3
        self.azimuthAntOrder = np.array([8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int32)
        self.elevationAntOrder = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int32)
        self.azimuthSinSign = 1.0
        self.dpkTimes = stFlashParam.dpkTimes
        self.dpkThres = stFlashParam.dpkThres
        self.dpkRadius = stFlashParam.dpkRadius
        self.frameGap = 1
        self.cfarMaxNum = 512
        self.cfar_dim = 1
        self.pc_gen_method = 1
        self.meanEn = True
        self.minRangeBin = 0
        self.max_frame_gap_seconds = 0.18
        self.max_mean_power = 100.0
        self.large_amplitude = 100.0
        self.max_large_fraction = 0.001

    def __cal_fft2dPT(self):
        _range = self.numVel
        return int(_range)


@dataclass
class SignalProcessConfig:
    winVelPre = [
        0x0001, 0x0005, 0x000B, 0x0014, 0x001F, 0x002C, 0x003C, 0x004E,
        0x0063, 0x007A, 0x0093, 0x00AF, 0x00CD, 0x00EE, 0x0110, 0x0135,
        0x015D, 0x0186, 0x01B2, 0x01E0, 0x0210, 0x0242, 0x0277, 0x02AD,
        0x02E6, 0x0320, 0x035C, 0x039B, 0x03DB, 0x041D, 0x0461, 0x04A7,
        0x04EE, 0x0537, 0x0582, 0x05CF, 0x061D, 0x066C, 0x06BD, 0x070F,
        0x0763, 0x07B8, 0x080E, 0x0866, 0x08BF, 0x0918, 0x0973, 0x09CF,
        0x0A2C, 0x0A8A, 0x0AE8, 0x0B48, 0x0BA8, 0x0C08, 0x0C6A, 0x0CCB,
        0x0D2E, 0x0D91, 0x0DF4, 0x0E57, 0x0EBB, 0x0F1F, 0x0F83, 0x0FE7,
        0x104B, 0x10AF, 0x1113, 0x1177, 0x11DB, 0x123E, 0x12A1, 0x1303,
        0x1366, 0x13C7, 0x1428, 0x1489, 0x14E8, 0x1547, 0x15A5, 0x1603,
        0x165F, 0x16BA, 0x1715, 0x176E, 0x17C6, 0x181D, 0x1873, 0x18C7,
        0x191A, 0x196C, 0x19BC, 0x1A0B, 0x1A58, 0x1AA3, 0x1AED, 0x1B36,
        0x1B7C, 0x1BC1, 0x1C04, 0x1C45, 0x1C85, 0x1CC2, 0x1CFD, 0x1D37,
        0x1D6E, 0x1DA4, 0x1DD7, 0x1E08, 0x1E37, 0x1E64, 0x1E8F, 0x1EB7,
        0x1EDD, 0x1F01, 0x1F23, 0x1F42, 0x1F5F, 0x1F7A, 0x1F92, 0x1FA8,
        0x1FBB, 0x1FCC, 0x1FDB, 0x1FE7, 0x1FF1, 0x1FF8, 0x1FFD, 0x1FFF,
        0x1FFF, 0x1FFD, 0x1FF8, 0x1FF1, 0x1FE7, 0x1FDB, 0x1FCC, 0x1FBB,
        0x1FA8, 0x1F92, 0x1F7A, 0x1F5F, 0x1F42, 0x1F23, 0x1F01, 0x1EDD,
        0x1EB7, 0x1E8F, 0x1E64, 0x1E37, 0x1E08, 0x1DD7, 0x1DA4, 0x1D6E,
        0x1D37, 0x1CFD, 0x1CC2, 0x1C85, 0x1C45, 0x1C04, 0x1BC1, 0x1B7C,
        0x1B36, 0x1AED, 0x1AA3, 0x1A58, 0x1A0B, 0x19BC, 0x196C, 0x191A,
        0x18C7, 0x1873, 0x181D, 0x17C6, 0x176E, 0x1715, 0x16BA, 0x165F,
        0x1603, 0x15A5, 0x1547, 0x14E8, 0x1489, 0x1428, 0x13C7, 0x1366,
        0x1303, 0x12A1, 0x123E, 0x11DB, 0x1177, 0x1113, 0x10AF, 0x104B,
        0x0FE7, 0x0F83, 0x0F1F, 0x0EBB, 0x0E57, 0x0DF4, 0x0D91, 0x0D2E,
        0x0CCB, 0x0C6A, 0x0C08, 0x0BA8, 0x0B48, 0x0AE8, 0x0A8A, 0x0A2C,
        0x09CF, 0x0973, 0x0918, 0x08BF, 0x0866, 0x080E, 0x07B8, 0x0763,
        0x070F, 0x06BD, 0x066C, 0x061D, 0x05CF, 0x0582, 0x0537, 0x04EE,
        0x04A7, 0x0461, 0x041D, 0x03DB, 0x039B, 0x035C, 0x0320, 0x02E6,
        0x02AD, 0x0277, 0x0242, 0x0210, 0x01E0, 0x01B2, 0x0186, 0x015D,
        0x0135, 0x0110, 0x00EE, 0x00CD, 0x00AF, 0x0093, 0x007A, 0x0063,
        0x004E, 0x003C, 0x002C, 0x001F, 0x0014, 0x000B, 0x0005, 0x0001,
    ]
    winVel = []
    steeringMatrix = []

    def init_winVel(self, algCfg):
        self.winVel = np.zeros((algCfg.numTX, algCfg.numChirp), dtype=np.uint32)
        for tx in range(min(4, algCfg.numTX)):
            self.winVel[tx, :] = self.winVelPre[tx:algCfg.numChirp * 4:4]

    def init_steeringMatrix(self, algCfg: AlgCfg):
        steering = np.exp(-2j * np.pi * np.arange(algCfg.numRX)[:, None]
                          * np.arange(algCfg.numAngle)[None, :] / algCfg.numAngle)
        # 原表是复数伪浮点 Q13，不能按两个 int16 解码。
        self.steeringMatrix = ((np.trunc(steering.real * 8192)
                               + 1j * np.trunc(steering.imag * 8192)) / 8192).astype(np.complex64)


@dataclass
class RadarUserCfg:
    freqStart: float = 57.5
    Bandwidth: float = 3.8
    sweepTimeA: float = 10
    sweepTimeB: float = 55
    sweepTimeC: float = 15
    sweepWaveNum: int = 0
    numTX: int = NUM_TX
    numRX: int = NUM_TX
    numChirp: int = NUM_CHIRP
    sampPT: int = NUM_SAMP


@dataclass
class StructBbCfarCfg0:

    def __init__(self, alg_cfg: AlgCfg, stFlashParam: StructParam):
        self.cfarVmode = np.uint8(stFlashParam.cfarVmode)
        self.wrapDirA = np.uint8(0)
        self.wrapDirB = np.uint8(0)
        self.sumMode = np.uint8(0)
        self.interCnt = np.uint16(alg_cfg.useRange)
        self.intraCnt = np.uint16(alg_cfg.numVel)
        self.thresDiv = np.uint8(3)
        self.divFac = np.uint8(0)
        self.mulFac = float(stFlashParam.cfarThV)
        self.microThres = float(stFlashParam.cfarThS)
        self.peakenV = np.uint8(stFlashParam.peakenV)
        self.searchSize = np.uint8(stFlashParam.cfarThVS)
        self.guardSize = np.uint8(stFlashParam.cfarThVG)
        self.totalSize = np.uint8(self.guardSize + self.searchSize)


class CfarRst:

    def __init__(self, rangeIdx, chirpIdx, power):
        self.rangeIdx = rangeIdx
        self.chirpIdx = chirpIdx
        self.power = power


@dataclass
class StructFramePoint:
    """帧检测点 — STRUCT_FRAMEPOINT
    对应雷达信号处理后每一帧的检测结果，包含 4 维索引和功率信息
    """
    d1Idx: np.uint16 = 0
    d2Idx: np.uint16 = 0
    d2IdxRslt: np.int16 = 0
    d3Idx: np.uint16 = 0
    d4Idx: np.uint16 = 0
    pow2dAbs: np.uint32 = 0
    pow3dAbs: np.uint32 = 0
    noise: np.uint32 = 0
    snr: np.uint8 = 0


def fft1dMeanCalc(fft1d: np.array):
    """
    :param fft1d: (NUM_CHIRP, NUM_TX, NUM_RX, USE_RANGE)
    """
    fft1dMean = np.stack([fft1d[:, tx].mean(axis=0) for tx in range(fft1d.shape[1])])
    return fft1dMean


def fft2dCalc(fft1d: np.array, fft1dMean: np.array, radar_user_cfg: RadarUserCfg, sgn_process_cfg: SignalProcessConfig, alg_cfg: AlgCfg):
    """
    :param fft1d: (NUM_CHIRP, NUM_TX, NUM_RX, USE_RANGE)
    :param fft1dMean: (NUM_TX, NUM_RX, USE_RANGE)

    Return:
        :fft2d: np.array(fft2dPT,NUM_TX,NUM_RX,USE_RANGE)
    """
    mean_offset_res = fft1d
    if alg_cfg.meanEn:
        mean_offset_res = fft1d - fft1dMean
    fft2d = np.zeros((alg_cfg.fft2dPT, alg_cfg.numTX, alg_cfg.numRX, alg_cfg.useRange), dtype=np.complex64)
    for tx_num in range(radar_user_cfg.numTX):
        current_win = sgn_process_cfg.winVel[tx_num, :]
        src_3d = mean_offset_res[:, tx_num, :, :]
        windowed_src = src_3d * current_win[:, np.newaxis, np.newaxis]
        fft2d_result = np.fft.fft(windowed_src, n=alg_cfg.fft2dPT, axis=0)
        fft2d[:, tx_num, :, :] = fft2d_result
    fft2d = np.fft.fftshift(fft2d, axes=0)
    fft2d = applyTdmPhaseCompensation(fft2d, alg_cfg)
    return fft2d


def applyTdmPhaseCompensation(fft2d: np.array, alg_cfg: AlgCfg):
    """
    在 Doppler FFT 后对全谱做 TDM 相位补偿。
    fft2d: (fft2dPT, NUM_TX, NUM_RX, USE_RANGE)
    """
    num_tx = fft2d.shape[1]
    num_chirp = alg_cfg.numChirp
    fft_size = num_tx * num_chirp
    center_idx = alg_cfg.fft2dPT // 2
    k_vec = np.arange(alg_cfg.fft2dPT, dtype=np.float32) - center_idx
    tx_idx_vec = np.arange(num_tx, dtype=np.float32)
    phase = -2.0 * np.pi * k_vec[:, None] * tx_idx_vec[None, :] / float(fft_size)
    comp = np.exp(1j * phase).astype(np.complex64)
    return fft2d * comp[:, :, None, None]


def fft2dAbsSum(fft2d: np.array):
    """
    :param fft2d: (fft2dPT,NUM_TX,NUM_RX,USE_RANGE)

    Return:
        :fft2dSumPf: np.array(NUM_CHIRP ,USE_RANGE)
    """
    fft2d_abs = np.abs(fft2d)
    sum_abs = np.sum(fft2d_abs, axis=(1, 2))
    fft2dSumPf = sum_abs.astype(np.float32)
    return fft2dSumPf / 128


def estimate_noise(left, right, config, *, sum_cells=False):
    """Keil's 1D reference sums; 2D software mode retains cell averages."""
    if config.cfarVmode == 3:
        cells = np.concatenate((left, right), axis=-1)
        rank = min(int(cells.shape[-1] * getattr(config, 'k_ratio', 0.75)), cells.shape[-1] - 1)
        return np.partition(cells, rank, axis=-1)[..., rank]
    if sum_cells:
        left_noise, right_noise = left.sum(-1), right.sum(-1)
    else:
        left_noise, right_noise = left.mean(-1), right.mean(-1)
    if config.cfarVmode == 2:
        return (left_noise + right_noise) / 2
    operation = np.maximum if config.cfarVmode == 0 else np.minimum
    return operation(left_noise, right_noise)


def cfarCalc(amplitude: np.ndarray, alg_cfg: AlgCfg, config: StructBbCfarCfg0):
    """按扫描顺序限制候选，再做距离峰值检测及相邻 bin 扩展。"""
    if alg_cfg.cfar_dim not in (1, 2) or config.cfarVmode not in (0, 1, 2, 3):
        raise ValueError("CFAR dimension must be 1/2; mode must be 0=GO, 1=SO, 2=CA, 3=OS")
    guard = int(config.guardSize)
    radius = guard + int(config.searchSize)
    padded = np.pad(amplitude, ((radius, radius), (0, 0)), mode="wrap")
    if alg_cfg.cfar_dim == 1:
        windows = sliding_window_view(padded, 2 * radius + 1, axis=0)
        left, right = windows[..., :int(config.searchSize)], windows[..., -int(config.searchSize):]
        alpha = float(config.mulFac) / 2**int(config.thresDiv)
    else:
        padded = np.pad(padded, ((0, 0), (radius, radius)), mode="edge")
        windows = sliding_window_view(padded, (2 * radius + 1, 2 * radius + 1))
        rows, columns = np.indices((2 * radius + 1, 2 * radius + 1))
        training = (abs(rows - radius) > guard) | (abs(columns - radius) > guard)
        first_half = (columns < radius) | ((columns == radius) & (rows < radius))
        left, right = windows[..., training & first_half], windows[..., training & ~first_half]
        alpha = float(config.mulFac)
    noise = estimate_noise(left, right, config, sum_cells=alg_cfg.cfar_dim == 1)
    detected = amplitude > noise * alpha
    if config.peakenV:
        previous, following = np.roll(amplitude, 1, axis=0), np.roll(amplitude, -1, axis=0)
        detected &= ((amplitude >= previous) & (amplitude > following)
                     | (amplitude > previous) & (amplitude >= following))

    initial = np.argwhere(detected)[:alg_cfg.cfarMaxNum]
    detected[:] = False
    detected[tuple(initial.T)] = True

    reflected = np.pad(amplitude, ((0, 0), (1, 1)), mode="reflect")
    previous, following = reflected[:, :-2], reflected[:, 2:]
    detected &= ((amplitude >= previous) & (amplitude > following)
                 | (amplitude > previous) & (amplitude >= following))
    candidates = {}
    last_range = amplitude.shape[1] - 1
    for doppler, distance in np.argwhere(detected):
        neighbors = (distance - 1 if distance else 1, distance,
                     distance + 1 if distance < last_range else last_range - 1)
        for neighbor in neighbors:
            if neighbor == distance or amplitude[doppler, neighbor] >= amplitude[doppler, distance] * alg_cfg.rangePeakRatio:
                candidates.setdefault((int(doppler), int(neighbor)), None)
    return [CfarRst(r, d, amplitude[d, r])
            for d, r in list(candidates)[:alg_cfg.cfarMaxNum]]


def toTargetAllAnt(cfar_rsts: List[CfarRst], fft2d: np.array):
    """
    :param fft2d: (fft2dPT,NUM_TX,NUM_RX,USE_RANGE)

    Return:
        :fft2dSumPf: np.array(NUM_DETECT ,NUM_ANT)
    """
    cfar_ant_rsts = []
    for cfar_rst in cfar_rsts:
        d1_idx = int(cfar_rst.rangeIdx)
        d2_idx = int(cfar_rst.chirpIdx)
        ant_data = fft2d[d2_idx, :, :, d1_idx].flatten()
        cfar_ant_rsts.append(ant_data)
    return np.array(cfar_ant_rsts, dtype=np.complex64)


def estimate_noise_floor_minus_topk(fft3d_abs, dpk_times, peak_radius=1):
    """
    去掉前 dpk_times 个局部峰值（及其邻域）后，对剩余谱求均值。

    Parameters
    ----------
    fft3d_abs : np.ndarray, shape (128,)
        3D-FFT 的幅度谱
    dpk_times : int
        要剔除的峰值个数
    peak_radius : int
        峰值邻域半径。半径=1 表示剔除峰值及其左右各 1 个点，
        共 3 个点；半径=0 表示只剔除峰值本身。

    Returns
    -------
    noise_floor : float
        剩余谱的均值
    """
    N = len(fft3d_abs)
    mask = np.ones(N, dtype=bool)
    local_peaks = []
    for idx in range(N):
        prev_idx = (idx - 1) % N
        next_idx = (idx + 1) % N
        if fft3d_abs[idx] >= fft3d_abs[prev_idx] and fft3d_abs[idx] >= fft3d_abs[next_idx]:
            local_peaks.append((fft3d_abs[idx], idx))
    local_peaks.sort(key=lambda x: x[0], reverse=True)
    topk_peaks = local_peaks[:dpk_times]
    for (_, peak_idx) in topk_peaks:
        for off in range(-peak_radius, peak_radius + 1):
            mask[(peak_idx + off) % N] = False
    if mask.sum() == 0:
        return 0.0
    noise_floor = float(np.mean(fft3d_abs[mask]))
    return noise_floor


def fft3dProc_sparse_pc(cfar_ant_rsts: np.array, cfar_rsts: List[CfarRst], alg_cfg: AlgCfg):
    """
    :param cfar_ant_rsts: (NUM_DETECT,NUM_ANT)

    Return:
        :frame_rsts: List[StructFramePoint]
        :rst_ids: List[int]
    """
    frame_rsts = []
    rst_ids = []
    for (rst_id, cfar_ant_rst) in enumerate(cfar_ant_rsts):
        azimuth_data = cfar_ant_rst[alg_cfg.azimuthAntOrder]
        fft3d = np.fft.fft(azimuth_data, n=alg_cfg.numAngle).astype(np.complex64)
        fft3d_abs = np.abs(fft3d)
        noise_floor = estimate_noise_floor_minus_topk(fft3d_abs, dpk_times=alg_cfg.dpkTimes, peak_radius=alg_cfg.dpkRadius)
        local_peaks = []
        for idx in range(128):
            prev_idx = (idx - 1) % 128
            next_idx = (idx + 1) % 128
            if fft3d_abs[idx] >= fft3d_abs[prev_idx] and fft3d_abs[idx] >= fft3d_abs[next_idx]:
                local_peaks.append((fft3d_abs[idx], idx))
        local_peaks.sort(key=lambda x: x[0], reverse=True)
        for dpk_idx in range(min(len(local_peaks), alg_cfg.dpkTimes)):
            (peak_pow, peak_idx) = local_peaks[dpk_idx]
            if peak_pow > noise_floor * alg_cfg.dpkThres:
                frame_rsts.append(StructFramePoint(d1Idx=cfar_rsts[rst_id].rangeIdx, d2Idx=cfar_rsts[rst_id].chirpIdx, d3Idx=peak_idx, pow2dAbs=cfar_rsts[rst_id].power, pow3dAbs=peak_pow))
                rst_ids.append(rst_id)
    return (frame_rsts, rst_ids)


def fft3dProc_dense_pc(cfar_ant_rsts: np.ndarray, cfar_rsts: List[CfarRst], alg_cfg: AlgCfg, trace=None) -> Tuple[List[StructFramePoint], List[int]]:
    """复数 sinc 减峰；残差幅值均值作为底噪，非硬件逐位仿真。"""
    if not len(cfar_ant_rsts):
        return [], []
    num_angle = alg_cfg.numAngle
    spectrum = np.fft.fft(cfar_ant_rsts[:, alg_cfg.azimuthAntOrder], n=num_angle, axis=1).astype(np.complex64)
    kernel = np.fft.fft(np.ones(alg_cfg.numAntA), n=num_angle) / alg_cfg.numAntA
    scale = np.where(np.maximum(abs(kernel.real), abs(kernel.imag)) >= 1, 4096, 8192)
    sinc = ((np.trunc(kernel.real * scale) + 1j * np.trunc(kernel.imag * scale)) / scale).astype(np.complex64)
    residual = spectrum.copy()
    if trace is not None:
        trace.append(residual.copy())
    rows, bins = np.arange(len(spectrum)), np.arange(num_angle)
    angles, strengths = [], []
    for _ in range(alg_cfg.dpkTimes):
        peak = abs(residual).argmax(axis=1)
        value = residual[rows, peak].copy()
        angles.append(peak)
        strengths.append(abs(value))
        residual -= value[:, None] * sinc[(bins[None] - peak[:, None]) % num_angle]
        residual[rows, peak] = 0
        if trace is not None:
            trace.append(residual.copy())
    angles, strengths = np.stack(angles, axis=1), np.stack(strengths, axis=1)
    noise = abs(residual).mean(axis=1)
    ids, peaks = np.nonzero(strengths > noise[:, None] * alg_cfg.dpkThres)
    points = [StructFramePoint(d1Idx=cfar_rsts[i].rangeIdx, d2Idx=cfar_rsts[i].chirpIdx,
                               d3Idx=angles[i, j], pow2dAbs=cfar_rsts[i].power,
                               pow3dAbs=strengths[i, j]) for i, j in zip(ids, peaks)]
    return points, ids.tolist()


def fft4dProc(cfar_ant_rsts: np.array, rst_ids: List[int], alg_cfg: AlgCfg, frame_rsts: List[StructFramePoint], sgn_process_cfg: SignalProcessConfig):
    """
    :param cfar_ant_rsts: (NUM_DETECT,NUM_ANT)
    :param frame_rsts: 检测结果
    """
    for (idx, frame_rst) in enumerate(frame_rsts):
        d3_idx = frame_rst.d3Idx
        ant_data = cfar_ant_rsts[rst_ids[idx], alg_cfg.elevationAntOrder]
        ant_data_matrix = ant_data.reshape(alg_cfg.numAntE, alg_cfg.numRX)
        steering_vector = sgn_process_cfg.steeringMatrix[:, d3_idx]
        elevation_data_compensated = np.sum(ant_data_matrix * steering_vector, axis=1)
        fft4d = np.fft.fft(elevation_data_compensated, n=128)
        frame_rst.d4Idx = np.argmax(np.abs(fft4d))


def angleCalcD(cfar_ant_rsts: np.array, cfar_rsts: List[CfarRst], alg_cfg: AlgCfg, sgn_process_cfg: SignalProcessConfig):
    """
    :param cfar_ant_rsts: (NUM_DETECT,NUM_ANT)

    Return:
        :frame_rsts: List[StructFramePoint]
    """
    if alg_cfg.pc_gen_method == 1:
        (frame_rsts, rst_ids) = fft3dProc_dense_pc(cfar_ant_rsts, cfar_rsts, alg_cfg=alg_cfg)
    elif alg_cfg.pc_gen_method == 2:
        (frame_rsts, rst_ids) = fft3dProc_sparse_pc(cfar_ant_rsts, cfar_rsts, alg_cfg=alg_cfg)
    else:
        raise ValueError(f'不支持的点云生成方法,仅支持: 1(稠密点云);2(稀疏点云);')
    fft4dProc(cfar_ant_rsts, rst_ids, alg_cfg, frame_rsts, sgn_process_cfg)
    return frame_rsts


def framePoints2pc(frame_points: List[StructFramePoint], alg_cfg: AlgCfg, range_resolution=0.025, velocity_resolution=0.1252):
    """检测结果转换为 [N,5]：x/y/z/velocity/amp；amp 为未截位的角谱峰值幅值。"""
    point_cloud = []
    for frame_point in frame_points:
        idx1 = int(frame_point.d1Idx)
        idx2 = int(frame_point.d2Idx)
        idx3 = int(frame_point.d3Idx)
        idx4 = int(frame_point.d4Idx)
        range_val = float(idx1) * range_resolution
        velocity = (float(idx2) - alg_cfg.fft2dPT // 2) * velocity_resolution
        amp = float(frame_point.pow3dAbs)
        if idx1 < alg_cfg.minRangeBin:
            continue
        azimuth_sign = getattr(alg_cfg, 'azimuthSinSign', 1.0)
        if idx3 <= 63:
            azimuth_angle = math.asin(azimuth_sign * float(idx3) / 64.0)
        elif idx3 <= 127:
            azimuth_angle = math.asin(azimuth_sign * (float(idx3) - 128.0) / 64.0)
        else:
            continue
        if idx4 <= 63:
            elevation_angle = math.asin(float(idx4) / 64.0)
        elif idx4 <= 127:
            elevation_angle = math.asin((float(idx4) - 128.0) / 64.0)
        else:
            continue
        sin_azimuth = math.sin(azimuth_angle)
        cos_azimuth = math.cos(azimuth_angle)
        sin_elevation = math.sin(elevation_angle)
        cos_elevation = math.cos(elevation_angle)
        x = range_val * cos_elevation * cos_azimuth
        y = range_val * sin_azimuth * cos_elevation
        z = range_val * sin_elevation
        point_cloud.append(np.array([x, y, z, velocity, amp]))
    if len(point_cloud) < 1:
        return np.empty(shape=(0, 5))
    return np.stack(point_cloud, axis=0)


def micro_motion_points(fft1d, rd_map, cfg, signal_cfg, cfar_cfg, stats_info):
    """沿用 skyrelay 的加窗均值跨帧差分，返回六组微动检测结果。"""
    history = deque(stats_info.hist_fft1d_winvel_mean, maxlen=MAX_FRAME_NUM)
    if stats_info.frame_cnt % cfg.frameGap == 0:
        window = signal_cfg.winVel.T[:, :, None, None]
        history.append(fft1dMeanCalc(fft1d * window).astype(np.complex64))
    groups = []
    if len(history) >= 2:
        search_size = int(cfar_cfg.searchSize)
        center = cfg.fft2dPT // 2
        left = rd_map[center - search_size:center].T
        right = rd_map[center + 1:center + search_size + 1].T
        threshold = estimate_noise(left, right, cfar_cfg, sum_cells=True) / 8 * cfar_cfg.microThres
        for old_frame in list(history)[-2::-1]:
            diff = history[-1] - old_frame
            magnitude = (np.abs(diff.real) + np.abs(diff.imag)).sum(axis=(0, 1)).astype(np.float32) / 2
            ranges = np.flatnonzero(
                (magnitude[1:-1] >= threshold[1:-1])
                & (magnitude[1:-1] > magnitude[:-2])
                & (magnitude[1:-1] >= magnitude[2:])
            ) + 1
            ranges = ranges[:cfg.cfarMaxNum]
            detections = [CfarRst(int(r), center, 0) for r in ranges]
            antennas = diff.reshape(-1, cfg.useRange)[:, ranges].T
            # 微动分支沿用原项目，始终使用稠密 DPK。
            points, ids = fft3dProc_dense_pc(antennas, detections, cfg)
            fft4dProc(antennas, ids, cfg, points, signal_cfg)
            groups.append(points)
    return groups, history


def validate_range_cube(range_cube, cfg):
    """验证解码结果；预热与完整处理共用坏帧判据。"""
    cube = np.asarray(range_cube)
    if cube.shape != (256, 64, 16) or not np.iscomplexobj(cube):
        raise ValueError(f"range_cube 必须为单帧复数 [256,64,16]，实际为 {cube.shape}")
    cube = np.asarray(cube, dtype=np.complex64)
    if not np.isfinite(cube).all():
        raise ValueError("nonfinite_fft")
    amplitude = abs(cube)
    mean_power = float(np.mean(amplitude**2))
    if mean_power <= 1e-12 or np.ptp(amplitude) == 0:
        raise ValueError("empty_or_constant_fft")
    if (mean_power > cfg.max_mean_power
            or np.mean(amplitude > cfg.large_amplitude) > cfg.max_large_fraction):
        raise ValueError("abnormal_fft_amplitude")
    return cube


def range_cube_to_point_cloud(
    range_cube: np.ndarray,
    *,
    flash_param: StructParam | None = None,
    alg_cfg: AlgCfg | None = None,
    stats_info: StatsInfo | None = None,
    timestamp: float | None = None,
    range_resolution: float = 0.025,
    velocity_resolution: float = 0.1252,
) -> np.ndarray:
    """将 get_bin_data 返回的单帧复数 [256,64,16] 转为 [N,6] 点云。

    第六列：1 动态点，2～7 六组历史差分微动点（按历史由近到远）。
    连续帧复用 stats_info；省略时只有动态点。新序列必须新建缓存。

    flash_param 调整 CFAR/DPK 阈值；alg_cfg 调整 meanEn（默认去静态）、
    cfar_dim（1/2）、pc_gen_method（1 稠密/2 稀疏）、天线顺序和方位符号。
    窗口及测角表固定为 skyrelay 的 4TX/4RX/64chirp/128角度配置。
    坏帧（含全零、恒幅和异常幅值）会清空历史并抛 ValueError。
    timestamp 为秒；重复、逆序或间隔超过 0.18 秒时重建历史。
    不传 timestamp 时按调用顺序处理，调用方须保证帧连续。
    """
    if not all(np.isfinite(value) and value > 0
               for value in (range_resolution, velocity_resolution)):
        raise ValueError("距离和速度分辨率必须为有限正数")

    param = flash_param if flash_param is not None else StructParam()
    cfg = alg_cfg if alg_cfg is not None else AlgCfg(param)
    if (cfg.numTX, cfg.numRX, cfg.numChirp, cfg.numSamp,
        cfg.useRange, cfg.fft2dPT, cfg.numAngle, cfg.numAntE) != (
        4, 4, 64, 512, 256, 64, 128, 3
    ):
        raise ValueError("仅支持 skyrelay 的 4TX/4RX/64chirp/512sample/128angle 配置")
    if cfg.frameGap <= 0 or int(cfg.frameGap) != cfg.frameGap:
        raise ValueError("frameGap 必须为正整数")
    if (param.cfarVmode not in (0, 1, 2, 3) or cfg.cfar_dim not in (1, 2)
            or cfg.pc_gen_method not in (1, 2)):
        raise ValueError("CFAR mode/dimension 或点云生成方法无效")
    for value, minimum, maximum in (
        (param.cfarThVG, 0, 255), (param.cfarThVS, 1, 31),
        (cfg.dpkTimes, 1, 128), (cfg.dpkRadius, 0, 63),
        (cfg.cfarMaxNum, 1, 16384), (cfg.minRangeBin, 0, 256),
    ):
        if not isinstance(value, (int, np.integer)) or not minimum <= value <= maximum:
            raise ValueError("CFAR/测角窗口、候选数量或最小距离 bin 无效")
    if cfg.dpkTimes * (2 * cfg.dpkRadius + 1) >= cfg.numAngle:
        raise ValueError("测角减峰必须保留底噪参考单元")
    if (not 0 <= cfg.rangePeakRatio <= 1 or cfg.azimuthSinSign not in (-1, 1)
            or not 0 <= cfg.max_large_fraction <= 1):
        raise ValueError("峰值比例、方位符号或异常幅值比例无效")
    if not all(np.isfinite(value) and value > 0 for value in (
        param.cfarThV, param.cfarThS, cfg.dpkThres, cfg.max_frame_gap_seconds,
        cfg.max_mean_power, cfg.large_amplitude,
    )):
        raise ValueError("门限与时间间隔必须为有限正数")
    try:
        cube = validate_range_cube(range_cube, cfg)
        if timestamp is not None:
            if not np.isfinite(timestamp):
                raise ValueError("timestamp 必须为有限秒数")
            if stats_info is not None and stats_info.last_timestamp is not None:
                gap = timestamp - stats_info.last_timestamp
                if gap <= 0 or gap > cfg.max_frame_gap_seconds:
                    stats_info.reset()
    except ValueError:
        if stats_info is not None:
            stats_info.reset()
        raise
    cfar_cfg = StructBbCfarCfg0(cfg, param)
    signal_cfg = SignalProcessConfig()
    signal_cfg.init_winVel(cfg)
    signal_cfg.init_steeringMatrix(cfg)
    fft1d = cube.transpose(1, 2, 0).reshape(NUM_CHIRP, NUM_TX, NUM_RX, -1)
    fft2d = fft2dCalc(fft1d, fft1dMeanCalc(fft1d), RadarUserCfg(), signal_cfg, cfg)
    rd_map = fft2dAbsSum(fft2d)
    detections = cfarCalc(rd_map, cfg, cfar_cfg)
    points = angleCalcD(toTargetAllAnt(detections, fft2d), detections, cfg, signal_cfg)
    groups = [points]
    if stats_info is not None:
        micro_groups, history = micro_motion_points(
            fft1d, rd_map, cfg, signal_cfg, cfar_cfg, stats_info
        )
        groups.extend(micro_groups)
    clouds = []
    for point_type, group in enumerate(groups, start=1):
        cloud = framePoints2pc(group, cfg, range_resolution, velocity_resolution)
        clouds.append(np.column_stack((cloud, np.full(len(cloud), point_type))))
    result = np.concatenate(clouds, axis=0).astype(np.float32, copy=False)
    if stats_info is not None:
        stats_info.hist_fft1d_winvel_mean = history
        stats_info.frame_cnt += 1
        stats_info.last_timestamp = timestamp
    return result
