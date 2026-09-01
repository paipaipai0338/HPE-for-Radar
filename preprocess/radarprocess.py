from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import math
import torch
import numpy as np
import os
from typing import *

# *****************************************************************************
# Radar configuration
# *****************************************************************************

@dataclass  # dataclass 可理解为结构体类，加入装饰器(@dataclass)可省略  __init__, __repr__, __eq__ 等必要方法
class Radar_Config:
    # 基础参数
    fs: int = 10_000_000  # 10 MHz，采样率
    chirp_time: float = 80e-6  # 80 μs，脉冲宽度
    B_set: float = 6.4453e9  # 6.4453 GHz，带宽
    time_B: float = 55e-6  # 55 μs，B段扫频时长
    c: float = 3e8  # 光速 m/s
    fc: float = 60e9  # 60 GHz，载频
    d_azi: float = 2170e-6  # 2170 μm，方位向孔径
    d_ele: float = 2400e-6  # 2400 μm，俯仰向孔径
    Tx: int = 4  # 发射通道数
    Rx: int = 4  # 接收通道数
    num_samp: int = 512  # 预设距离维采样点数量
    num_chirp: int = 64  # 预设一帧中chirp数量
    num_channel: int = 16  # 预设虚拟通道数

    channel_layout: Tuple[Tuple[int, ...], ...] = (
        (0,  0,  0,  0,  4,  3,  2,  1),
        (0,  0,  0,  0,  8,  7,  6,  5),
        (16, 15, 14, 13, 12, 11, 10, 9),
    )
    azi_deg: Tuple[float, ...] = (-90, 90)  # 方位角度范围
    ele_deg: Tuple[float, ...] = (-90, 90)  # 俯仰角度范围
    azi_ant_indices: Tuple[int, ...] = (15, 14, 13, 12, 11, 10, 9, 8)   # 方位测算通道索引
    ele_ant_indices: Tuple[int, ...] = (8, 4, 0)                        # 俯仰测算通道索引

    angle_method: Literal["bartlett", "mvdr", "music"] = "bartlett"
    num_angle_beams: int = 128  # 角度谱点数

    # 依赖其他参数的属性（使用 field(init=False)）
    slope: float = None  # 调频率
    lam: float = None  # 波长
    prf: float = None  # 脉冲重复频率

    def __post_init__(self) -> None:
        """
        根据基础雷达配置自动计算派生参数。
        输入:
          self: Radar_Config，包含 fs/chirp_time/B_set/time_B/c/fc/Tx 等标量配置。
        输出:
          None；原地更新 self.slope/self.lam/self.prf，类型均为 float，shape 均为标量。
        """
        self.slope = self.B_set / self.time_B
        self.lam = self.c / self.fc
        self.prf = 1 / (self.chirp_time * self.Tx)
        self.virtual_channel_positions = self._build_virtual_channel_position()

    def _build_virtual_channel_position(self) -> torch.Tensor:
        positions = torch.zeros((self.num_channel, 3), dtype=torch.float32)
        
        num_rows = len(self.channel_layout)
        num_cols = len(self.channel_layout[0])
        center_col = (num_cols - 1) / 2.0
    
        for row_idx, row in enumerate(self.channel_layout):
            z = (num_rows - 1 - row_idx) * self.d_ele
    
            for col_idx, channel_id in enumerate(row):
                if channel_id == 0:
                    continue
    
                y = (col_idx -  center_col) * self.d_azi
    
                positions[channel_id - 1] = torch.tensor([0.0, y, z], dtype=torch.float32)
    
        return positions

# *****************************************************************************
# BIN decoding and sensor I/O
# *****************************************************************************

# 读取1DFFT函数
def bin_buffer_to_cube_range_fft(
    raw: np.ndarray,
    radar_config: Radar_Config,
    source_name: str = '<buffer>',
) -> Optional[np.ndarray]:
    def _pseudo_float_cplx_to_complex(pf_u32: np.ndarray) -> np.ndarray:
        pf = pf_u32.astype(np.uint32)

        exp = (pf >> 28).astype(np.int32)  # 4-bit exponent
        real = (pf & 0x3FFF).astype(np.int32)  # 14-bit signed
        imag = ((pf >> 14) & 0x3FFF).astype(np.int32)  # 14-bit signed

        # two's complement on 14-bit
        real[real >= (1 << 13)] -= (1 << 14)
        imag[imag >= (1 << 13)] -= (1 << 14)

        scale = np.power(2.0, exp - 13).astype(np.float32)
        out = (real.astype(np.float32) + 1j * imag.astype(np.float32)) * scale
        return out.astype(np.complex64)
    num_samp = radar_config.num_samp
    num_chirp = radar_config.num_chirp
    num_ant = radar_config.Tx * radar_config.Rx
    use_range = num_samp // 2
    expected_bytes = use_range * num_chirp * num_ant * 4
    raw = np.asarray(raw, dtype=np.uint8).reshape(-1)
    if raw.size != expected_bytes:
        print(f"[WARN] {source_name} size mismatch: "
              f"{raw.size} != {expected_bytes}, skip.")
        return None
    raw8 = raw.reshape(-1, 8)[:, ::-1].reshape(-1)
    pf_u32 = np.frombuffer(raw8.tobytes(), dtype="<u4")
    vec_cplx = _pseudo_float_cplx_to_complex(pf_u32)
    mcu_timing = vec_cplx.reshape((use_range, num_ant, num_chirp), order="F")
    adc_data_range_FFT = np.transpose(mcu_timing, (0, 2, 1))
    return adc_data_range_FFT


def bin_to_cube_range_fft(file_path: Path|str, radar_config: Radar_Config) -> Optional[np.ndarray]:
    raw = np.fromfile(file_path, dtype=np.uint8)
    return bin_buffer_to_cube_range_fft(
        raw,
        radar_config,
        source_name=os.path.basename(file_path),
    )

# 读取点云数据
def get_pc_data(file_path: Path|str):
    data = np.load(file_path)
    return data

# 读取谱图数据，利用bin_to_cube_range_fft封装
def get_bin_data(file_path: Path|str, radar_config: Radar_Config):
    data = bin_to_cube_range_fft(file_path, radar_config)
    return data

# *****************************************************************************
# Radar resolution and coordinate axes
# *****************************************************************************

# 获取雷达分辨率
def get_radar_res(
    radar_config: Radar_Config,
    doppler_mode: Literal["normal", "firmware_tdm"] = "firmware_tdm",
    azi_num_ant: int = 8,
    ele_num_ant: int = 3,
    aperture_mode: Literal["effective", "physical"] = "effective"
    ) -> Tuple[float, float, float, float]:
    """
    返回:
      range_res: 距离分辨率, m
      velocity_res: 速度分辨率, m/s
      azi_angle_res_deg: 方位理论角分辨率, deg
      ele_angle_res_deg: 俯仰理论角分辨率, deg
    """

    def _array_angle_res_deg(
            lam: float,
            d: float,
            num_ant: int,
            theta_deg: float = 0.0,
            aperture_mode: Literal["effective", "physical"] = "effective",
    ) -> float:
        """
        阵列孔径理论角分辨率，单位 deg。

        aperture_mode:
          "effective":
            使用常见雷达角分辨率公式 A = N * d。
            对应 delta_sin ≈ λ / (N*d)。

          "physical":
            使用物理孔径 A = (N-1) * d。
            更保守一些。
        """
        if num_ant < 2:
            raise RuntimeError(f"num_ant 必须 >= 2，当前为 {num_ant}")
        if d <= 0:
            raise RuntimeError(f"阵元间距 d 必须 > 0，当前为 {d}")

        if aperture_mode == "effective":
            aperture = num_ant * d
        elif aperture_mode == "physical":
            aperture = (num_ant - 1) * d
        else:
            raise RuntimeError(f"未知 aperture_mode={aperture_mode}")

        theta = np.deg2rad(theta_deg)

        # 在 sin(theta) 空间的分辨率
        delta_sin = lam / aperture

        # theta=0° 附近 cos(theta)=1；这里保留一般角度写法
        delta_theta_rad = delta_sin / max(np.cos(theta), 1e-12)

        return float(np.rad2deg(delta_theta_rad))

    # 距离分辨率
    range_res = radar_config.c * radar_config.fs / (
        2.0 * radar_config.slope * radar_config.num_samp
    )

    # 速度分辨率
    if doppler_mode == "firmware_tdm":
        velocity_res = radar_config.lam / (
            2.0 * radar_config.num_chirp * radar_config.Tx * radar_config.chirp_time
        )
    else:
        velocity_res = radar_config.lam * radar_config.prf / (
            2.0 * radar_config.num_chirp
        )

    # 阵列孔径理论角分辨率，0° 附近
    azi_angle_res_deg = _array_angle_res_deg(
        lam=radar_config.lam,
        d=radar_config.d_azi,
        num_ant=azi_num_ant,
        theta_deg=0.0,
        aperture_mode=aperture_mode,
    )

    ele_angle_res_deg = _array_angle_res_deg(
        lam=radar_config.lam,
        d=radar_config.d_ele,
        num_ant=ele_num_ant,
        theta_deg=0.0,
        aperture_mode=aperture_mode,
    )

    return range_res, velocity_res, azi_angle_res_deg, ele_angle_res_deg


# *****************************************************************************
# Torch Doppler processing
# *****************************************************************************

def doppler_fft_torch(
    range_cube: torch.Tensor,
    radar_config: Radar_Config,
    window: bool = True,
    doppler_mode: Literal["normal", "firmware_tdm"] = "firmware_tdm",
    window_periodic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``[B,T,R,D,M]`` Doppler data and its velocity axis."""
    if range_cube.ndim != 5 or not range_cube.is_complex():
        raise ValueError(
            "range_cube 必须为复数 [B,T,R,C,M]，"
            f"实际为 shape={tuple(range_cube.shape)}, dtype={range_cube.dtype}"
        )

    _, _, _, num_chirps, num_antennas = range_cube.shape
    real_dtype = range_cube.real.dtype
    device = range_cube.device
    num_tx = int(getattr(radar_config, "num_tx", radar_config.Tx))
    num_rx = int(getattr(radar_config, "num_rx", radar_config.Rx))
    if num_tx <= 0 or num_rx <= 0 or num_antennas != num_tx * num_rx:
        raise ValueError(
            "Doppler FFT 要求输入天线数等于 Tx * Rx，"
            f"实际 M={num_antennas}, Tx={num_tx}, Rx={num_rx}"
        )

    if doppler_mode == "normal":
        if window:
            doppler_window = torch.hann_window(
                num_chirps,
                periodic=window_periodic,
                dtype=real_dtype,
                device=device,
            )
            range_cube = range_cube * doppler_window.view(1, 1, 1, -1, 1)

        doppler_cube = torch.fft.fftshift(
            torch.fft.fft(range_cube, dim=3),
            dim=3,
        )
        doppler_indices = (
            torch.arange(num_chirps, dtype=real_dtype, device=device)
            - num_chirps // 2
        )
        doppler_frequency = doppler_indices / num_chirps * radar_config.prf
        velocity_axis = radar_config.lam / 2.0 * doppler_frequency
        return doppler_cube, velocity_axis

    if doppler_mode != "firmware_tdm":
        raise ValueError(
            "doppler_mode 必须为 'normal' 或 'firmware_tdm'，"
            f"实际为 {doppler_mode!r}"
        )

    firmware_fft_size = num_chirps * num_tx
    tx_indices = torch.div(
        torch.arange(num_antennas, device=device),
        num_rx,
        rounding_mode="floor",
    )

    if window:
        full_window = torch.hann_window(
            firmware_fft_size,
            periodic=window_periodic,
            dtype=real_dtype,
            device=device,
        )
        sample_indices = (
            torch.arange(num_chirps, device=device)[:, None] * num_tx
            + tx_indices[None, :]
        )
        antenna_windows = full_window[sample_indices]
        range_cube = range_cube * antenna_windows.view(
            1, 1, 1, num_chirps, num_antennas
        )

    doppler_cube = torch.fft.fftshift(
        torch.fft.fft(range_cube, dim=3),
        dim=3,
    )
    signed_bins = (
        torch.arange(num_chirps, dtype=real_dtype, device=device)
        - num_chirps // 2
    )
    phase = (
        -2.0
        * math.pi
        * signed_bins[:, None]
        * tx_indices.to(real_dtype)[None, :]
        / firmware_fft_size
    )
    doppler_cube = doppler_cube * torch.exp(1j * phase).to(range_cube.dtype).view(
        1, 1, 1, num_chirps, num_antennas
    )

    chirp_gap = float(getattr(radar_config, "chirp_gap", radar_config.chirp_time))
    doppler_frequency = signed_bins / firmware_fft_size / chirp_gap
    velocity_axis = radar_config.lam / 2.0 * doppler_frequency
    return doppler_cube, velocity_axis


# *****************************************************************************
# Array geometry and steering vectors
# *****************************************************************************

def build_direction_vectors(
    angle_axis_rad: torch.Tensor,
    plane: Literal["azi", "ele"],
) -> torch.Tensor:
    """Build 1-D azimuth or elevation directions in x-forward coordinates."""
    zeros = torch.zeros_like(angle_axis_rad)
    if plane == "azi":
        return torch.stack(
            (torch.cos(angle_axis_rad), torch.sin(angle_axis_rad), zeros),
            dim=-1,
        )
    if plane == "ele":
        return torch.stack(
            (torch.cos(angle_axis_rad), zeros, torch.sin(angle_axis_rad)),
            dim=-1,
        )
    raise ValueError(f"plane 必须为 'azi' 或 'ele'，实际为 {plane!r}")


def build_azimuth_elevation_directions(
    azimuth_axis_rad: torch.Tensor,
    elevation_axis_rad: torch.Tensor,
) -> torch.Tensor:
    """Build flattened joint azimuth/elevation direction vectors ``[K,3]``."""
    azimuth_grid, elevation_grid = torch.meshgrid(
        azimuth_axis_rad,
        elevation_axis_rad,
        indexing="ij",
    )
    return torch.stack(
        (
            torch.cos(elevation_grid) * torch.cos(azimuth_grid),
            torch.cos(elevation_grid) * torch.sin(azimuth_grid),
            torch.sin(elevation_grid),
        ),
        dim=-1,
    ).reshape(-1, 3)


def build_steering_matrix(
    antenna_positions: torch.Tensor,
    direction_vectors: torch.Tensor,
    wavelength: float,
    normalize: bool = False,
) -> torch.Tensor:
    """Return receive-combining weights ``[M,K]`` with positive phase."""
    if antenna_positions.ndim != 2 or antenna_positions.shape[-1] != 3:
        raise ValueError("antenna_positions 必须为 [M,3]")
    if direction_vectors.ndim != 2 or direction_vectors.shape[-1] != 3:
        raise ValueError("direction_vectors 必须为 [K,3]")
    if wavelength <= 0:
        raise ValueError("wavelength 必须大于 0")

    relative_positions = antenna_positions - antenna_positions[:1]
    path_difference = torch.einsum(
        "mc,kc->mk",
        relative_positions,
        direction_vectors,
    )
    weights = torch.exp(
        1j * (2.0 * math.pi / wavelength) * path_difference
    )
    if normalize:
        weights = weights / float(antenna_positions.shape[0])
    return weights


# *****************************************************************************
# DBF, Bartlett, MVDR and MUSIC
# *****************************************************************************

def conventional_dbf_power(
    snapshots: torch.Tensor,
    steering_weights: torch.Tensor,
    direction_chunk_size: int = 256,
) -> torch.Tensor:
    """Beamform ``[...,M]`` snapshots and return power ``[...,K]``."""
    if snapshots.shape[-1] != steering_weights.shape[0]:
        raise ValueError("snapshot 天线维与 steering matrix 不匹配")
    if direction_chunk_size <= 0:
        raise ValueError("direction_chunk_size 必须大于 0")

    output = torch.empty(
        (*snapshots.shape[:-1], steering_weights.shape[1]),
        dtype=snapshots.real.dtype,
        device=snapshots.device,
    )
    for start in range(0, steering_weights.shape[1], direction_chunk_size):
        stop = min(start + direction_chunk_size, steering_weights.shape[1])
        beamformed = torch.einsum(
            "...m,mk->...k",
            snapshots,
            steering_weights[:, start:stop],
        )
        output[..., start:stop] = beamformed.abs().square()
    return output


def bartlett_spectrum(
    snapshots: torch.Tensor,
    steering_weights: torch.Tensor,
    snapshot_chunk_size: int = 8,
) -> torch.Tensor:
    """Return a Bartlett spectrum for snapshots shaped ``[...,S,M]``."""
    if snapshot_chunk_size <= 0:
        raise ValueError("snapshot_chunk_size 必须大于 0")
    num_snapshots = snapshots.shape[-2]
    num_antennas = snapshots.shape[-1]
    power_sum = torch.zeros(
        (*snapshots.shape[:-2], steering_weights.shape[1]),
        dtype=snapshots.real.dtype,
        device=snapshots.device,
    )
    for start in range(0, num_snapshots, snapshot_chunk_size):
        stop = min(start + snapshot_chunk_size, num_snapshots)
        spectrum_chunk = torch.einsum(
            "...sm,mk->...sk",
            snapshots[..., start:stop, :],
            steering_weights,
        )
        power_sum += spectrum_chunk.abs().square().sum(dim=-2)
    return power_sum / float(num_snapshots * num_antennas**2)


def estimate_spatial_covariance(
    snapshots: torch.Tensor,
    forward_backward: bool = True,
    diagonal_loading: float = 1e-2,
) -> torch.Tensor:
    """Estimate a loaded covariance matrix from ``[...,S,M]`` snapshots."""
    num_snapshots = snapshots.shape[-2]
    num_antennas = snapshots.shape[-1]
    covariance = torch.einsum(
        "...sm,...sn->...mn",
        snapshots,
        snapshots.conj(),
    ) / float(num_snapshots)
    covariance = 0.5 * (covariance + covariance.conj().transpose(-2, -1))

    if forward_backward:
        exchange = torch.eye(
            num_antennas,
            dtype=snapshots.dtype,
            device=snapshots.device,
        ).flip(0)
        covariance_fb = torch.matmul(
            torch.matmul(exchange, covariance.conj()),
            exchange,
        )
        covariance = 0.5 * (covariance + covariance_fb)
        covariance = 0.5 * (
            covariance + covariance.conj().transpose(-2, -1)
        )

    eps = torch.finfo(snapshots.real.dtype).eps
    trace_mean = covariance.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1)
    loading = diagonal_loading * trace_mean.clamp_min(eps) + eps
    identity = torch.eye(
        num_antennas,
        dtype=snapshots.dtype,
        device=snapshots.device,
    )
    return covariance + loading[..., None, None] * identity


def mvdr_spectrum(
    covariance: torch.Tensor,
    steering_vector: torch.Tensor,
) -> torch.Tensor:
    """Return the MVDR spatial spectrum ``[...,K]``."""
    eps = torch.finfo(covariance.real.dtype).eps
    covariance_inverse = torch.linalg.inv(covariance)
    denominator = torch.einsum(
        "mk,...mn,nk->...k",
        steering_vector.conj(),
        covariance_inverse,
        steering_vector,
    ).real.clamp_min(eps)
    return 1.0 / denominator


def music_spectrum(
    covariance: torch.Tensor,
    steering_vector: torch.Tensor,
    num_sources: int = 1,
) -> torch.Tensor:
    """Return a peak-normalized MUSIC spatial spectrum ``[...,K]``."""
    num_antennas = covariance.shape[-1]
    if not 0 < num_sources < num_antennas:
        raise ValueError("num_sources 必须在 [1, M-1] 范围内")
    eps = torch.finfo(covariance.real.dtype).eps
    _, eigenvectors = torch.linalg.eigh(covariance)
    noise_subspace = eigenvectors[..., :num_antennas - num_sources]
    noise_projection = torch.einsum(
        "...mq,mk->...qk",
        noise_subspace.conj(),
        steering_vector,
    )
    denominator = noise_projection.abs().square().sum(dim=-2).clamp_min(eps)
    spectrum = 1.0 / denominator
    return spectrum / spectrum.amax(dim=-1, keepdim=True).clamp_min(eps)


def range_cube_to_range_angle_power(
    range_cube: torch.Tensor,
    radar_config: Radar_Config,
    plane: Literal["azi", "ele"],
    remove_static: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate a 1-D range-angle spectrum from ``[B,T,R,S,M]`` data."""
    if range_cube.ndim != 5 or not range_cube.is_complex():
        raise ValueError("range_cube 必须为复数 [B,T,R,S,M]")
    if plane not in ("azi", "ele"):
        raise ValueError(f"plane 必须为 'azi' 或 'ele'，实际为 {plane!r}")
    if radar_config.angle_method not in ("bartlett", "mvdr", "music"):
        raise ValueError(f"未知 angle_method={radar_config.angle_method}")

    if remove_static:
        range_cube = range_cube - range_cube.mean(dim=3, keepdim=True)

    real_dtype = range_cube.real.dtype
    device = range_cube.device
    positions = torch.as_tensor(
        radar_config.virtual_channel_positions,
        dtype=real_dtype,
        device=device,
    )
    fov = radar_config.azi_deg if plane == "azi" else radar_config.ele_deg
    ant_indices = (
        radar_config.azi_ant_indices
        if plane == "azi"
        else radar_config.ele_ant_indices
    )
    index_tensor = torch.as_tensor(
        tuple(ant_indices),
        dtype=torch.long,
        device=device,
    )
    subarray_cube = torch.index_select(range_cube, dim=-1, index=index_tensor)
    subarray_positions = torch.index_select(positions, dim=0, index=index_tensor)
    angle_axis_rad = torch.linspace(
        math.radians(fov[0]),
        math.radians(fov[1]),
        radar_config.num_angle_beams,
        dtype=real_dtype,
        device=device,
    )
    direction_vectors = build_direction_vectors(angle_axis_rad, plane)
    steering_weights = build_steering_matrix(
        subarray_positions,
        direction_vectors,
        radar_config.lam,
    ).to(range_cube.dtype)

    if radar_config.angle_method == "bartlett":
        return bartlett_spectrum(subarray_cube, steering_weights), angle_axis_rad

    covariance = estimate_spatial_covariance(subarray_cube)
    steering_vector = steering_weights.conj()
    if radar_config.angle_method == "mvdr":
        spectrum = mvdr_spectrum(covariance, steering_vector)
    else:
        spectrum = music_spectrum(covariance, steering_vector, num_sources=1)
        range_power = subarray_cube.abs().square().mean(dim=(-1, -2))
        spectrum = spectrum * range_power.unsqueeze(-1)
    return spectrum, angle_axis_rad


# *****************************************************************************
# Legacy NumPy Doppler and angle processing
# *****************************************************************************

# 对range_cube 在chirp维度做FFT，返回原始与对消后的数据
def doppler_fft(
    data: np.ndarray,
    radar_config: Radar_Config,
    window: bool = True,
    n_fft_doppler: int = 1024,
    doppler_mode: Literal["normal", "firmware_tdm"] = "firmware_tdm",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    doppler_mode:
      "normal":
        普通 Doppler FFT。输出 shape=(range, n_fft_doppler, ant)。

      "firmware_tdm":
        Firmware 风格 TDM Doppler FFT。
        假设 4Tx TDM-MIMO: Tx0, Tx1, Tx2, Tx3, Tx0...
        会把每个 TX 的 64 个 chirp 插回 256 点 TDM 时间轴，
        做 256 点 FFT，再按固件卸载顺序取 64 个 Doppler bin。
        输出 shape=(range, num_chirp, ant)。
        此模式下 n_fft_doppler 不再控制输出点数。
    """
    data = np.asarray(data)

    if data.ndim != 3:
        raise RuntimeError(
            "data must have shape "
            "(num_samp_or_range, num_chirp, num_ant), "
            f"actual shape={data.shape}"
        )

    if n_fft_doppler <= 0:
        raise RuntimeError(
            f"n_fft_doppler 必须大于 0，当前值为 "
            f"{n_fft_doppler}"
        )

    num_samp, num_chirp, num_ant = data.shape

    # ============================================================
    # 均值对消分支
    #
    # 对每个 range bin、每根虚拟天线，
    # 沿慢时间 chirp 维减去复数 IQ 均值。
    #
    # data:       [R, C, A]
    # mean:       [R, 1, A]
    # data_clean: [R, C, A]
    # ============================================================
    data_clean = (
        data
        - np.mean(
            data,
            axis=1,
            keepdims=True,
        )
    )

    # ============================================================
    # 普通 Doppler FFT
    # ============================================================
    if doppler_mode == "normal":
        if window:
            w_d = np.hanning(
                num_chirp
            ).astype(np.float32)

            dop_in = (
                data
                * w_d[None, :, None]
            )

            dop_in_clean = (
                data_clean
                * w_d[None, :, None]
            )
        else:
            dop_in = data
            dop_in_clean = data_clean

        # 原始分支
        dop_fft = np.fft.fft(
            dop_in,
            n=n_fft_doppler,
            axis=1,
        )

        dop_fft = np.fft.fftshift(
            dop_fft,
            axes=1,
        )

        # 均值对消分支
        dop_fft_clean = np.fft.fft(
            dop_in_clean,
            n=n_fft_doppler,
            axis=1,
        )

        dop_fft_clean = np.fft.fftshift(
            dop_fft_clean,
            axes=1,
        )

        Nd = n_fft_doppler

        k = (
            np.arange(Nd)
            - Nd // 2
        )

        f_d = (
            k
            / Nd
            * radar_config.prf
        )

        v_axis = (
            radar_config.lam
            / 2.0
        ) * f_d

        return (
            dop_fft,
            dop_fft_clean,
            v_axis,
        )

    if doppler_mode != "firmware_tdm":
        raise RuntimeError(
            f"未知 doppler_mode={doppler_mode}"
        )

    # ============================================================
    # Firmware 风格 TDM Doppler FFT
    # ============================================================
    num_tx = int(
        getattr(
            radar_config,
            "num_tx",
            getattr(
                radar_config,
                "Tx",
                4,
            ),
        )
    )

    num_rx = int(
        getattr(
            radar_config,
            "num_rx",
            getattr(
                radar_config,
                "Rx",
                num_ant // num_tx,
            ),
        )
    )

    if num_tx <= 0 or num_rx <= 0:
        raise RuntimeError(
            f"num_tx/num_rx 非法: "
            f"num_tx={num_tx}, "
            f"num_rx={num_rx}"
        )

    if num_ant != num_tx * num_rx:
        raise RuntimeError(
            "firmware_tdm 要求 "
            "num_ant == num_tx * num_rx，"
            f"当前 num_ant={num_ant}, "
            f"num_tx={num_tx}, "
            f"num_rx={num_rx}"
        )

    firmware_fft_size = (
        num_chirp
        * num_tx
    )

    # 固件 unload 顺序：
    # 例如 256 点 FFT，取 224..255 和 0..31。
    use_a = (
        (num_tx - 1) * num_chirp
        + num_chirp // 2
    )

    use_b = (
        num_chirp // 2
        - 1
    )

    unload_bins = np.r_[
        use_a:firmware_fft_size,
        0:use_b + 1,
    ]

    # 原始分支和均值对消分支
    dop_fft = np.zeros(
        (
            num_samp,
            num_chirp,
            num_ant,
        ),
        dtype=np.complex128,
    )

    dop_fft_clean = np.zeros(
        (
            num_samp,
            num_chirp,
            num_ant,
        ),
        dtype=np.complex128,
    )

    if window:
        full_win = np.hanning(
            firmware_fft_size
        ).astype(np.float32)

        tx_windows = np.stack(
            [
                full_win[
                    tx_idx:
                    firmware_fft_size:
                    num_tx
                ][:num_chirp]
                for tx_idx in range(num_tx)
            ],
            axis=0,
        )
    else:
        tx_windows = np.ones(
            (
                num_tx,
                num_chirp,
            ),
            dtype=np.float32,
        )

    for ant_idx in range(num_ant):
        tx_idx = (
            ant_idx
            // num_rx
        )

        # 当前虚拟天线对应的真实 TDM 时间槽
        slot_idx = (
            tx_idx
            + np.arange(num_chirp)
            * num_tx
        )

        # --------------------------------------------------------
        # 原始数据分支
        # --------------------------------------------------------
        tdm_input = np.zeros(
            (
                num_samp,
                firmware_fft_size,
            ),
            dtype=np.complex128,
        )

        tdm_input[:, slot_idx] = (
            data[:, :, ant_idx]
            * tx_windows[
                tx_idx
            ][None, :]
        )

        fft_out = np.fft.fft(
            tdm_input,
            n=firmware_fft_size,
            axis=1,
        )

        dop_fft[
            :,
            :,
            ant_idx,
        ] = fft_out[:, unload_bins]

        # --------------------------------------------------------
        # 均值对消分支
        # --------------------------------------------------------
        tdm_input_clean = np.zeros(
            (
                num_samp,
                firmware_fft_size,
            ),
            dtype=np.complex128,
        )

        tdm_input_clean[:, slot_idx] = (
            data_clean[:, :, ant_idx]
            * tx_windows[
                tx_idx
            ][None, :]
        )

        fft_out_clean = np.fft.fft(
            tdm_input_clean,
            n=firmware_fft_size,
            axis=1,
        )

        dop_fft_clean[
            :,
            :,
            ant_idx,
        ] = fft_out_clean[
            :,
            unload_bins,
        ]

    # 固件输出的 num_chirp 个 bin：
    # [-num_chirp/2, ..., num_chirp/2 - 1]
    k = np.arange(
        -num_chirp // 2,
        num_chirp // 2,
    )

    chirp_gap = getattr(
        radar_config,
        "chirp_gap",
        getattr(
            radar_config,
            "chirp_time",
            None,
        ),
    )

    if chirp_gap is not None:
        f_d = (
            k
            / firmware_fft_size
            / float(chirp_gap)
        )
    else:
        f_d = (
            k
            / num_chirp
            * radar_config.prf
        )

    v_axis = (
        radar_config.lam
        / 2.0
    ) * f_d

    return (
        dop_fft,
        dop_fft_clean,
        v_axis,
    )

# 对pytorch传出带有 B T 的tensor进行fft处理
def doppler_fft_batch_T(
    data: torch.Tensor,
    radar_config: Radar_Config,
    window: bool = True,
    n_fft_doppler: int = 1024,
    doppler_mode: Literal[
        "normal",
        "firmware_tdm",
    ] = "firmware_tdm",
    return_to_input_device: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    if data.ndim != 5:
        raise ValueError(
            "data 必须为 "
            "[B,T,num_samp_or_range,num_chirp,num_ant]，"
            f"实际 shape={tuple(data.shape)}"
        )

    B, T = data.shape[:2]

    if B == 0 or T == 0:
        raise ValueError(
            f"B 和 T 必须大于 0，当前 B={B}, T={T}"
        )

    input_device = data.device

    # 当前 NumPy 处理不可微，显式脱离计算图并搬到 CPU。
    data_np = (
        data.detach()
        .cpu()
        .numpy()
    )

    # [B,T,R,C,A] -> [B*T,R,C,A]
    data_stacked = data_np.reshape(
        B * T,
        *data_np.shape[2:],
    )

    doppler_results = []
    doppler_clean_results = []

    reference_output_shape = None
    reference_clean_output_shape = None
    reference_v_axis = None

    for idx in range(B * T):
        (
            dop_fft_frame,
            dop_fft_clean_frame,
            v_axis,
        ) = doppler_fft(
            data=data_stacked[idx],
            radar_config=radar_config,
            window=window,
            n_fft_doppler=n_fft_doppler,
            doppler_mode=doppler_mode,
        )

        dop_fft_frame = np.asarray(
            dop_fft_frame
        )

        dop_fft_clean_frame = np.asarray(
            dop_fft_clean_frame
        )

        v_axis = np.asarray(
            v_axis,
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # 检查单帧输出
        # ---------------------------------------------------------
        if dop_fft_frame.ndim != 3:
            raise RuntimeError(
                f"第 {idx} 帧原始 Doppler FFT 输出应为三维，"
                f"实际 shape={dop_fft_frame.shape}"
            )

        if dop_fft_clean_frame.ndim != 3:
            raise RuntimeError(
                f"第 {idx} 帧均值对消 Doppler FFT 输出应为三维，"
                f"实际 shape={dop_fft_clean_frame.shape}"
            )

        if dop_fft_frame.shape != dop_fft_clean_frame.shape:
            raise RuntimeError(
                f"第 {idx} 帧两个分支输出形状不一致："
                f"raw={dop_fft_frame.shape}, "
                f"clean={dop_fft_clean_frame.shape}"
            )

        # ---------------------------------------------------------
        # 检查不同帧输出形状是否一致
        # ---------------------------------------------------------
        if reference_output_shape is None:
            reference_output_shape = dop_fft_frame.shape
        elif dop_fft_frame.shape != reference_output_shape:
            raise RuntimeError(
                f"原始分支不同帧输出形状不一致："
                f"第一帧={reference_output_shape}，"
                f"第 {idx} 帧={dop_fft_frame.shape}"
            )

        if reference_clean_output_shape is None:
            reference_clean_output_shape = (
                dop_fft_clean_frame.shape
            )
        elif (
            dop_fft_clean_frame.shape
            != reference_clean_output_shape
        ):
            raise RuntimeError(
                f"均值对消分支不同帧输出形状不一致："
                f"第一帧={reference_clean_output_shape}，"
                f"第 {idx} 帧={dop_fft_clean_frame.shape}"
            )

        # ---------------------------------------------------------
        # 检查速度轴是否一致
        # ---------------------------------------------------------
        if reference_v_axis is None:
            reference_v_axis = v_axis.copy()
        else:
            if (
                v_axis.shape != reference_v_axis.shape
                or not np.allclose(
                    v_axis,
                    reference_v_axis,
                    rtol=1e-5,
                    atol=1e-7,
                )
            ):
                raise RuntimeError(
                    f"第 {idx} 帧的速度轴与第一帧不一致"
                )

        # 统一使用 complex64，减少内存占用。
        doppler_results.append(
            dop_fft_frame.astype(
                np.complex64,
                copy=False,
            )
        )

        doppler_clean_results.append(
            dop_fft_clean_frame.astype(
                np.complex64,
                copy=False,
            )
        )

    # =============================================================
    # 原始分支：[B*T,R,D,A] -> [B,T,R,D,A]
    # =============================================================
    dop_fft_np = np.stack(
        doppler_results,
        axis=0,
    )

    dop_fft_np = dop_fft_np.reshape(
        B,
        T,
        *dop_fft_np.shape[1:],
    )

    # =============================================================
    # 均值对消分支：[B*T,R,D,A] -> [B,T,R,D,A]
    # =============================================================
    dop_fft_clean_np = np.stack(
        doppler_clean_results,
        axis=0,
    )

    dop_fft_clean_np = dop_fft_clean_np.reshape(
        B,
        T,
        *dop_fft_clean_np.shape[1:],
    )

    dop_fft_data = torch.from_numpy(
        np.ascontiguousarray(
            dop_fft_np
        )
    )

    dop_fft_clean_data = torch.from_numpy(
        np.ascontiguousarray(
            dop_fft_clean_np
        )
    )

    v_axis_tensor = torch.from_numpy(
        np.ascontiguousarray(
            reference_v_axis,
            dtype=np.float32,
        )
    )

    if return_to_input_device:
        dop_fft_data = dop_fft_data.to(
            input_device,
            non_blocking=True,
        )

        dop_fft_clean_data = (
            dop_fft_clean_data.to(
                input_device,
                non_blocking=True,
            )
        )

        v_axis_tensor = v_axis_tensor.to(
            input_device,
            non_blocking=True,
        )

    return (
        dop_fft_data,
        dop_fft_clean_data,
        v_axis_tensor,
    )

# doppler_fft_batch_T 封装版本
def range_cube_to_doppler_cube(data: torch.Tensor, radar_config: Radar_Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    range_res, _, _, _ = get_radar_res(radar_config)
    r_axis = np.arange(data.shape[2], dtype=float) * range_res
    r_axis = torch.tensor(r_axis)
    doppler_cube, doppler_cube_mean, v_axis = doppler_fft_batch_T(data=data, radar_config=radar_config)
    return doppler_cube, doppler_cube_mean, r_axis, v_axis

def angle_fft(data: np.ndarray, radar_config: Radar_Config, window: bool = True, n_fft_angle: int = 1024, target_index: List = None, channel_index: List = None, type: str = 'ele', method: str = 'fft') -> Tuple[np.ndarray, np.ndarray]:
    """
    对指定距离-多普勒单元和通道做角度谱估计。
    输入:
      data: np.ndarray，距离-多普勒-通道数据，shape=(Nr, Nd, Nch)。
      radar_config: Radar_Config，雷达参数对象，lam/d_ele/d_azi 为 float 标量。
      window: bool，FFT 方法下是否使用 Hann 窗，shape 为标量。
      n_fft_angle: int，角度网格/FFT 点数，shape 为标量。
      target_index: List[int]，目标 [range_index, doppler_index]，shape=(2,)。
      channel_index: List[int]，参与角度估计的通道索引，shape=(M,)。
      type: str，角度类型，'ele' 表示俯仰，'azi' 表示方位，shape 为标量字符串。
      method: str，角度估计方法，'fft' 或 'MVDR'，shape 为标量字符串。
    输出:
      ang_result: np.ndarray，角度谱，FFT 时 dtype=complex，MVDR 时 dtype=float，shape=(n_fft_angle,)。
      az_axis: np.ndarray，dtype=float，角度轴，单位 rad，shape=(n_fft_angle,)。
    """

    # --- 1. 参数与配置 ---
    if type == 'ele':
        d = radar_config.d_ele
    elif type == 'azi':
        d = radar_config.d_azi
    else:
        raise RuntimeError(f"Unknown type: '{type}'. Use 'ele' or 'azi'.")

    data = np.asarray(data)
    if data.ndim != 3:
        raise RuntimeError("data must have shape (Nr, Nd, Nch)")
    if n_fft_angle <= 0:
        raise RuntimeError(f"n_fft_angle 必须大于 0，当前值为 {n_fft_angle}")
    if len(target_index) != 2:
        raise RuntimeError("target_index 必须为 [range_index, doppler_index]")
    if not channel_index:
        raise RuntimeError("channel_index 不能为空")
    if method not in ('fft', 'MVDR'):
        raise RuntimeError("method 必须为 'fft' 或 'MVDR'")
    Nr, Nd, Nch = data.shape
    if target_index[0] < 0 or target_index[0] >= Nr or target_index[1] < 0 or target_index[1] >= Nd:
        raise RuntimeError(f"target_index 越界，当前值为 {target_index}，data shape 为 {data.shape}")
    if min(channel_index) < 0 or max(channel_index) >= Nch:
        raise RuntimeError(f"channel_index 越界，当前值为 {channel_index}，通道数为 {Nch}")
    # data shape假设: [Range, Doppler, Antenna] 或类似
    # 确保拿到正确维度

    num_ant = len(channel_index)

    # --- 2. 数据切片与快拍选取 ---
    if method == 'fft':
        # FFT 模式：通常只取单快拍，或者多快拍相干累加（视具体需求）
        # 这里保持你原有的逻辑：单点切片
        # Shape: (1, num_ant)
        snap_data = data[target_index[0], target_index[1], channel_index].reshape(1, -1)
    else:  # MVDR
        # MVDR 模式：需要多个快拍来估计协方差矩阵 R
        K = 8
        Nd = data.shape[1]
        # 获取快拍索引
        snap_indices = np.arange(target_index[1] - K, target_index[1] + K + 1)
        snap_indices = np.clip(snap_indices, 0, Nd - 1)
        snap_indices = np.unique(snap_indices)

        # 取数据 Shape: (S, num_ant)
        snap_data = data[target_index[0], snap_indices, :]
        snap_data = snap_data[:, channel_index]

    # --- 3. 角度网格生成 (通用) ---
    # 使用 linspace 生成均匀的角度正弦空间 [-1, 1]，比 fftfreq 更直观用于 MVDR
    if method == 'MVDR':
        # MVDR 通常不需要像 FFT 那样凑 2 的幂次，但也可用 n_fft_angle 控制精度
        sin_theta = np.linspace(-1, 1, n_fft_angle)
        az_axis = np.arcsin(sin_theta)  # 注意：这里 sin_theta 为 1/-1 时可能产生极小误差，arcsin 没问题
    else:
        # FFT 模式保持原逻辑，与频率对齐
        u = np.fft.fftshift(np.fft.fftfreq(n_fft_angle, d=1.0))
        sin_theta = (u * radar_config.lam) / d
        # 裁剪以防数值误差导致 arcsin nan
        mask = np.abs(sin_theta) <= 1.0
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        az_axis = np.arcsin(sin_theta)

    # --- 4. 核心算法 ---

    if method == 'fft':
        # === FFT 流程 ===
        # 仅在 FFT 模式下加窗
        process_data = snap_data
        if window:
            w_a = np.hanning(num_ant).astype(process_data.dtype)
            # 广播乘法 (1, M) * (M,)
            process_data = process_data * w_a

        # Zero padding
        # axis=1 是天线维度
        ang_spec = np.fft.fft(process_data, n=n_fft_angle, axis=1)
        ang_spec = np.fft.fftshift(ang_spec, axes=1)

        # 如果是单快拍，去掉第一维
        ang_result = ang_spec[0]

    elif method == 'MVDR':
        # === MVDR 流程 ===
        # 1. 绝对不要加窗 (Keep Raw Data)
        X = snap_data  # Shape (S, M)
        S, M = X.shape

        # 2. 计算协方差矩阵 R = E[x x^H]
        # X.T 是 (M, S), X.conj() 是 (S, M)
        # R 应该是 (M, M)
        # 正确公式: R = (X.conj().T @ X) / S
        R = (X.conj().T @ X) / S

        # 3. 对角加载 (Diagonal Loading)
        # 增强鲁棒性，防止 R 不可逆
        tr = np.trace(R).real
        dl_factor = 1e-3  # 加载因子，可调
        R = R + (dl_factor * tr / M) * np.eye(M, dtype=R.dtype)

        # 4. 求逆
        Rinv = np.linalg.inv(R)

        # 5. 构建导向矢量矩阵 A
        # A shape: (M, n_fft_angle)
        # 假设天线是均匀线阵 (ULA)，索引 0..M-1
        m = np.arange(M).reshape(-1, 1)  # (M, 1)

        # Steering Vector: a(theta) = exp(-j * 2pi * d/lam * m * sin(theta))
        # 注意：这里 sin_theta 使用上面生成的网格
        phase = -2.0j * np.pi * (d / radar_config.lam) * m * sin_theta.reshape(1, -1)
        A = np.exp(phase)

        # 6. 计算 MVDR 空间谱
        # P = 1 / (a^H * Rinv * a)
        # 分母计算技巧：
        # Rinv @ A -> (M, N)
        # conj(A) * (Rinv @ A) -> 逐元素乘
        # sum(..., axis=0) -> 对天线维求和

        den = np.sum(np.conj(A) * (Rinv @ A), axis=0).real
        ang_result = 1.0 / np.maximum(den, 1e-12)  # 避免除以0

        # 如果之前为了 sin_theta 做了 mask (仅针对 FFT 频率超范围情况)，这里其实不需要
        # 因为 linspace(-1, 1) 保证了都在范围内

    return ang_result, az_axis
