from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence, Tuple
import math

import torch
import torch.nn.functional as F


# =============================================================================
# 固定硬件虚拟阵列
#
# 雷达坐标系：
#   x：雷达前方
#   y：雷达左侧
#   z：雷达上方
#
# range_cube 最后一维约定：
#   index 0  <-> 虚拟通道 1
#   ...
#   index 15 <-> 虚拟通道 16
# =============================================================================

AZIMUTH_ELEMENT_SPACING_M = 2170e-6
ELEVATION_ELEMENT_SPACING_M = 2400e-6
NUM_VIRTUAL_CHANNELS = 16

# 从上到下、从图中左到右排列。
# 0 表示该网格位置不存在虚拟通道。
VIRTUAL_CHANNEL_LAYOUT: Tuple[Tuple[int, ...], ...] = (
    (0,  0,  0,  0,  4,  3,  2,  1),
    (0,  0,  0,  0,  8,  7,  6,  5),
    (16, 15, 14, 13, 12, 11, 10, 9),
)


def build_virtual_phase_positions() -> torch.Tensor:
    """
    根据固定虚拟阵列布局生成 16 个虚拟通道的相位坐标。

    Returns:
        positions:
            float32 Tensor，[16, 3]，单位 m。
            positions[channel_id - 1] 对应虚拟通道 channel_id。

    坐标约定：
        x：雷达前方。所有阵元都位于雷达面板，因此 x=0。
        y：雷达左侧为正。图中越靠左，y 越大。
        z：雷达上方为正。最下层 z=0。
    """
    positions = torch.zeros(
        (NUM_VIRTUAL_CHANNELS, 3),
        dtype=torch.float32,
    )

    num_rows = len(VIRTUAL_CHANNEL_LAYOUT)
    num_cols = len(VIRTUAL_CHANNEL_LAYOUT[0])
    center_col = (num_cols - 1) / 2.0

    for row_idx, row in enumerate(VIRTUAL_CHANNEL_LAYOUT):
        z = (
            num_rows - 1 - row_idx
        ) * ELEVATION_ELEMENT_SPACING_M

        for col_idx, channel_id in enumerate(row):
            if channel_id == 0:
                continue

            y = (
                center_col - col_idx
            ) * AZIMUTH_ELEMENT_SPACING_M

            positions[channel_id - 1] = torch.tensor(
                [0.0, y, z],
                dtype=torch.float32,
            )

    return positions


@dataclass
class SingleRadarProjectionConfig:
    """
    固定 4Tx × 4Rx 单雷达的 RPM 风格投影配置。

    方位子阵列：
        虚拟通道 16,15,14,13,12,11,10,9，
        对应底部完整的 8 阵元水平 ULA。

    俯仰子阵列：
        虚拟通道 9,5,1，
        对应最右侧同一列的 3 阵元垂直 ULA。
    """

    virtual_phase_positions: torch.Tensor = field(
        default_factory=build_virtual_phase_positions
    )

    # 0-based Tensor 索引，对应虚拟通道 16...9。
    azi_ant_indices: Tuple[int, ...] = (
        15, 14, 13, 12, 11, 10, 9, 8
    )

    # 0-based Tensor 索引，对应虚拟通道 9,5,1。
    ele_ant_indices: Tuple[int, ...] = (
        8, 4, 0
    )

    azimuth_min_deg: float = -60.0
    azimuth_max_deg: float = 60.0
    elevation_min_deg: float = -45.0
    elevation_max_deg: float = 45.0

    num_azimuth_beams: int = 1024
    num_elevation_beams: int = 1024

    # 8 阵元方位阵列可以使用 Hann 窗。
    azi_apply_array_window: bool = False

    # 3 阵元俯仰阵列不能使用 Hann 窗；
    # 非周期 3 点 Hann 近似为 [0, 1, 0]，会破坏角度信息。
    ele_apply_array_window: bool = False

    # 方位和俯仰阵列的相位方向可能不同，分别配置。
    azimuth_phase_sign: float = -1.0
    elevation_phase_sign: float = 1.0

    # Bartlett 分块处理 chirp，避免一次生成 [B,T,R,C,K] 大张量。
    chirp_chunk_size: int = 8

    # 测角方法：bartlett 为当前常规波束形成，mvdr/music 为超分辨测角。
    angle_method: Literal["bartlett", "mvdr", "music"] = "mvdr"

    # MVDR/MUSIC 协方差矩阵对角加载系数。
    diagonal_loading: float = 1e-2

    # 当前方位和俯仰子阵列均为 ULA，可使用前后向平均增强稳健性。
    forward_backward_average: bool = True

    # MUSIC 必须指定信号源数量。方位可尝试 1~2，俯仰 3 阵元建议固定为 1。
    music_num_sources_azi: int = 1
    music_num_sources_ele: int = 1

    def __post_init__(self) -> None:
        self.virtual_phase_positions = (
            torch.as_tensor(
                self.virtual_phase_positions,
                dtype=torch.float32,
            )
            .detach()
            .clone()
        )

        if self.virtual_phase_positions.shape != (
            NUM_VIRTUAL_CHANNELS,
            3,
        ):
            raise ValueError(
                "virtual_phase_positions 必须为 [16,3]，"
                f"实际为 {tuple(self.virtual_phase_positions.shape)}"
            )

        for name, indices in (
            ("azi_ant_indices", self.azi_ant_indices),
            ("ele_ant_indices", self.ele_ant_indices),
        ):
            if len(indices) < 2:
                raise ValueError(f"{name} 至少需要两个阵元")
            if len(set(indices)) != len(indices):
                raise ValueError(f"{name} 中存在重复索引")
            if min(indices) < 0 or max(indices) >= NUM_VIRTUAL_CHANNELS:
                raise ValueError(
                    f"{name} 必须位于 [0,15]，当前为 {indices}"
                )

        if not (
            self.azimuth_min_deg < self.azimuth_max_deg
            and self.elevation_min_deg < self.elevation_max_deg
        ):
            raise ValueError("角度范围下限必须小于上限")

        if self.num_azimuth_beams < 2:
            raise ValueError("num_azimuth_beams 必须 >= 2")
        if self.num_elevation_beams < 2:
            raise ValueError("num_elevation_beams 必须 >= 2")
        if self.chirp_chunk_size <= 0:
            raise ValueError("chirp_chunk_size 必须 > 0")
        if self.angle_method not in ("bartlett", "mvdr", "music"):
            raise ValueError(f"未知 angle_method={self.angle_method}")
        if self.diagonal_loading < 0:
            raise ValueError("diagonal_loading 必须 >= 0")
        if not 1 <= self.music_num_sources_azi < len(self.azi_ant_indices):
            raise ValueError("music_num_sources_azi 必须位于 [1, 方位阵元数-1]")
        if not 1 <= self.music_num_sources_ele < len(self.ele_ant_indices):
            raise ValueError("music_num_sources_ele 必须位于 [1, 俯仰阵元数-1]")


@dataclass
class RPMProjectionOutput:
    """
    单雷达 RPM 风格投影结果。

    张量形状：
        range_azimuth_power:
            [B,T_out,R,K_azi]

        range_elevation_power:
            [B,T_out,R,K_ele]

        horizontal_xy_power:
            [B,T_out,H_x,W_y]
            行方向为前向 x，列方向为侧向 y。

        vertical_xz_power:
            [B,T_out,H_z,W_x]
            行方向为竖直 z，列方向为前向 x。
    """

    horizontal_xy_power: torch.Tensor
    vertical_xz_power: torch.Tensor

    range_azimuth_power: torch.Tensor
    range_elevation_power: torch.Tensor

    range_axis: torch.Tensor
    azimuth_axis_rad: torch.Tensor
    elevation_axis_rad: torch.Tensor

    horizontal_x_axis: torch.Tensor
    horizontal_y_axis: torch.Tensor
    vertical_x_axis: torch.Tensor
    vertical_z_axis: torch.Tensor

    time_start_index: int


def suppress_static_clutter(
    range_cube: torch.Tensor,
    mode: Literal[
        "none",
        "chirp_mean",
        "frame_difference",
    ] = "frame_difference",
) -> Tuple[torch.Tensor, int]:
    """
    静态杂波抑制。

    Args:
        range_cube:
            复数 Tensor，[B,T,R,C,A]。

        mode:
            "none":
                不处理，T_out=T。

            "chirp_mean":
                沿 chirp 维减复数均值，抑制零 Doppler，T_out=T。

            "frame_difference":
                相邻外层帧做复数差分 X_t-X_{t-1}，
                更接近 RPM 的 consecutive measurement subtraction，
                T_out=T-1。
    """
    if range_cube.ndim != 5:
        raise ValueError(
            "range_cube 必须为 [B,T,R,C,A]，"
            f"实际为 {tuple(range_cube.shape)}"
        )

    if mode == "none":
        return range_cube, 0

    if mode == "chirp_mean":
        return (
            range_cube
            - range_cube.mean(dim=3, keepdim=True),
            0,
        )

    if mode == "frame_difference":
        if range_cube.shape[1] < 2:
            raise ValueError(
                "frame_difference 至少需要两个连续帧"
            )
        return (
            range_cube[:, 1:]
            - range_cube[:, :-1],
            1,
        )

    raise ValueError(f"未知 clutter mode: {mode}")


def _build_direction_vectors(
    angle_axis_rad: torch.Tensor,
    plane: Literal["azimuth", "elevation"],
) -> torch.Tensor:
    """
    按 x 前、y 左、z 上坐标系构造单位方向向量。

    azimuth:
        u(theta) = [cos(theta), sin(theta), 0]

    elevation:
        u(phi) = [cos(phi), 0, sin(phi)]
    """
    zeros = torch.zeros_like(angle_axis_rad)

    if plane == "azimuth":
        return torch.stack(
            (
                torch.cos(angle_axis_rad),
                torch.sin(angle_axis_rad),
                zeros,
            ),
            dim=-1,
        )

    if plane == "elevation":
        return torch.stack(
            (
                torch.cos(angle_axis_rad),
                zeros,
                torch.sin(angle_axis_rad),
            ),
            dim=-1,
        )

    raise ValueError(
        f"plane 必须为 azimuth 或 elevation，当前为 {plane}"
    )


def range_cube_to_range_angle_power(
    range_cube: torch.Tensor,
    virtual_phase_positions: torch.Tensor,
    ant_indices: Sequence[int],
    wavelength: float,
    angle_min_deg: float,
    angle_max_deg: float,
    num_angle_beams: int,
    plane: Literal["azimuth", "elevation"],
    apply_array_window: bool,
    phase_sign: float = 1.0,
    chirp_chunk_size: int = 8,
    angle_method: Literal["bartlett", "mvdr", "music"] = "mvdr",
    diagonal_loading: float = 1e-2,
    forward_backward_average: bool = True,
    music_num_sources: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 Range FFT cube 生成 Range-Angle 功率图。

    输入：
        range_cube: [B,T,R,C,A]，C 作为协方差估计的快拍维。

    输出：
        range_angle_power: [B,T,R,K]
        angle_axis_rad: [K]

    angle_method：
        bartlett：常规波束形成，与原实现一致。
        mvdr：Capon/MVDR 超分辨谱，不需要预先指定目标数量。
        music：MUSIC 伪谱，需要指定 music_num_sources。
    """
    if range_cube.ndim != 5:
        raise ValueError(f"range_cube 必须为 [B,T,R,C,A]，实际为 {tuple(range_cube.shape)}")
    if not torch.is_complex(range_cube):
        raise TypeError(f"range_cube 必须为复数 Tensor，实际 dtype={range_cube.dtype}")
    if wavelength <= 0:
        raise ValueError("wavelength 必须 > 0")
    if num_angle_beams < 2:
        raise ValueError("num_angle_beams 必须 >= 2")
    if chirp_chunk_size <= 0:
        raise ValueError("chirp_chunk_size 必须 > 0")
    if phase_sign not in (-1.0, 1.0):
        raise ValueError("phase_sign 必须为 +1.0 或 -1.0")
    if angle_method not in ("bartlett", "mvdr", "music"):
        raise ValueError(f"未知 angle_method={angle_method}")
    if diagonal_loading < 0:
        raise ValueError("diagonal_loading 必须 >= 0")
    if angle_method != "bartlett" and apply_array_window:
        raise ValueError("MVDR/MUSIC 不建议使用阵列 Hann 窗，请将 apply_array_window=False")

    B, T, R, C, A = range_cube.shape
    device = range_cube.device
    real_dtype = range_cube.real.dtype
    eps = torch.finfo(real_dtype).eps

    positions = torch.as_tensor(virtual_phase_positions, dtype=real_dtype, device=device)
    if positions.shape != (A, 3):
        raise ValueError(f"virtual_phase_positions 期望 ({A},3)，实际 {tuple(positions.shape)}")

    index_tensor = torch.as_tensor(tuple(ant_indices), dtype=torch.long, device=device)
    if index_tensor.numel() < 2:
        raise ValueError("角度估计至少需要两个阵元")
    if int(index_tensor.min()) < 0 or int(index_tensor.max()) >= A:
        raise IndexError(f"天线索引超出范围，A={A}, indices={tuple(ant_indices)}")

    subarray_cube = torch.index_select(range_cube, dim=-1, index=index_tensor)
    subarray_positions = torch.index_select(positions, dim=0, index=index_tensor)
    subarray_positions = subarray_positions - subarray_positions[:1]

    angle_axis_rad = torch.linspace(
        math.radians(angle_min_deg), math.radians(angle_max_deg), num_angle_beams,
        dtype=real_dtype, device=device,
    )

    direction_vectors = _build_direction_vectors(angle_axis_rad, plane)
    path_difference = torch.einsum("mc,kc->mk", subarray_positions, direction_vectors)
    phase = phase_sign * 2.0 * math.pi / wavelength * path_difference

    # steering_weight 保持与原代码一致，用于 y = sum(x_m * steering_weight_m)。
    steering_weight = torch.exp(1j * phase).to(dtype=range_cube.dtype)
    steering_vector = steering_weight.conj()
    num_selected_ant = subarray_cube.shape[-1]

    # -------------------------------------------------------------------------
    # Bartlett：保留原常规波束形成实现，便于与超分辨结果直接对比。
    # -------------------------------------------------------------------------
    if angle_method == "bartlett":
        if apply_array_window:
            if num_selected_ant < 4:
                raise ValueError(f"少于 4 个阵元时不应使用 Hann 窗，当前阵元数={num_selected_ant}")
            array_window = torch.hann_window(
                num_selected_ant, periodic=False, dtype=real_dtype, device=device,
            )
            subarray_cube = subarray_cube * array_window.view(1, 1, 1, 1, num_selected_ant)
            normalization = array_window.sum().square().clamp_min(1e-12)
        else:
            normalization = torch.tensor(float(num_selected_ant ** 2), dtype=real_dtype, device=device)

        power_sum = torch.zeros((B, T, R, num_angle_beams), dtype=real_dtype, device=device)
        for start in range(0, C, chirp_chunk_size):
            stop = min(start + chirp_chunk_size, C)
            angle_spectrum_chunk = torch.einsum(
                "btrcm,mk->btrck", subarray_cube[:, :, :, start:stop, :], steering_weight,
            )
            power_sum += angle_spectrum_chunk.abs().square().sum(dim=3)

        range_angle_power = power_sum / float(C) / normalization
        return range_angle_power, angle_axis_rad

    # -------------------------------------------------------------------------
    # MVDR/MUSIC：使用 C 个 chirp 作为快拍，按每个 range bin 构造空间协方差。
    # Rxx[m,n] = E[x_m * conj(x_n)]。
    # -------------------------------------------------------------------------
    covariance = torch.einsum(
        "btrcm,btrcn->btrmn", subarray_cube, subarray_cube.conj(),
    ) / float(C)
    covariance = 0.5 * (covariance + covariance.conj().transpose(-2, -1))

    if forward_backward_average:
        exchange = torch.eye(num_selected_ant, dtype=range_cube.dtype, device=device).flip(0)
        covariance_fb = torch.matmul(torch.matmul(exchange, covariance.conj()), exchange)
        covariance = 0.5 * (covariance + covariance_fb)
        covariance = 0.5 * (covariance + covariance.conj().transpose(-2, -1))

    trace_mean = covariance.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1)
    loading = diagonal_loading * trace_mean.clamp_min(eps) + eps
    identity = torch.eye(num_selected_ant, dtype=range_cube.dtype, device=device)
    covariance_loaded = covariance + loading[..., None, None] * identity

    if angle_method == "mvdr":
        covariance_inverse = torch.linalg.inv(covariance_loaded)
        denominator = torch.einsum(
            "mk,btrmn,nk->btrk", steering_vector.conj(), covariance_inverse, steering_vector,
        ).real.clamp_min(eps)
        range_angle_power = 1.0 / denominator
        return range_angle_power, angle_axis_rad

    if not 1 <= music_num_sources < num_selected_ant:
        raise ValueError(
            f"music_num_sources 必须位于 [1,{num_selected_ant - 1}]，当前为 {music_num_sources}"
        )

    _, eigenvectors = torch.linalg.eigh(covariance_loaded)
    noise_subspace = eigenvectors[..., :num_selected_ant - music_num_sources]
    noise_projection = torch.einsum(
        "btrmq,mk->btrqk", noise_subspace.conj(), steering_vector,
    )
    denominator = noise_projection.abs().square().sum(dim=-2).clamp_min(eps)
    music_spectrum = 1.0 / denominator

    # MUSIC 是无量纲伪谱。先按每个 range bin 归一化角谱，再用该距离单元能量加权，
    # 避免纯噪声距离单元也出现与强目标相同亮度的尖峰。
    music_spectrum = music_spectrum / music_spectrum.amax(dim=-1, keepdim=True).clamp_min(eps)
    range_power = subarray_cube.abs().square().mean(dim=(-1, -2))
    range_angle_power = music_spectrum * range_power.unsqueeze(-1)

    return range_angle_power, angle_axis_rad

def range_angle_to_cartesian_map(
    range_angle_power: torch.Tensor,
    range_axis: torch.Tensor,
    angle_axis_rad: torch.Tensor,
    forward_limits: Tuple[float, float],
    lateral_limits: Tuple[float, float],
    output_size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将 Range-Angle 图重采样到 Cartesian 平面。

    水平 x-y 图：
        forward=x，lateral=y。

    垂直 x-z 图：
        forward=x，lateral=z。

    Args:
        range_angle_power:
            [B,T,R,K]

        output_size:
            (num_forward_pixels, num_lateral_pixels)

    Returns:
        cartesian_map:
            [B,T,num_forward_pixels,num_lateral_pixels]
    """
    if range_angle_power.ndim != 4:
        raise ValueError(
            "range_angle_power 必须为 [B,T,R,K]，"
            f"实际为 {tuple(range_angle_power.shape)}"
        )

    B, T, R, K = range_angle_power.shape
    H_forward, W_lateral = output_size

    if H_forward <= 0 or W_lateral <= 0:
        raise ValueError("output_size 的两个维度必须 > 0")

    device = range_angle_power.device
    dtype = range_angle_power.dtype

    range_axis = torch.as_tensor(
        range_axis,
        dtype=dtype,
        device=device,
    )
    angle_axis_rad = torch.as_tensor(
        angle_axis_rad,
        dtype=dtype,
        device=device,
    )

    if range_axis.shape != (R,):
        raise ValueError(
            f"range_axis 应为 [{R}]，实际为 {tuple(range_axis.shape)}"
        )
    if angle_axis_rad.shape != (K,):
        raise ValueError(
            f"angle_axis_rad 应为 [{K}]，实际为 "
            f"{tuple(angle_axis_rad.shape)}"
        )
    if not bool(torch.all(range_axis[1:] > range_axis[:-1])):
        raise ValueError("range_axis 必须严格递增")
    if not bool(torch.all(angle_axis_rad[1:] > angle_axis_rad[:-1])):
        raise ValueError("angle_axis_rad 必须严格递增")

    forward_axis = torch.linspace(
        forward_limits[0],
        forward_limits[1],
        H_forward,
        dtype=dtype,
        device=device,
    )
    lateral_axis = torch.linspace(
        lateral_limits[0],
        lateral_limits[1],
        W_lateral,
        dtype=dtype,
        device=device,
    )

    forward_grid, lateral_grid = torch.meshgrid(
        forward_axis,
        lateral_axis,
        indexing="ij",
    )

    range_grid = torch.sqrt(
        forward_grid.square()
        + lateral_grid.square()
    )
    angle_grid = torch.atan2(
        lateral_grid,
        forward_grid,
    )

    angle_normalized = (
        2.0
        * (angle_grid - angle_axis_rad[0])
        / (angle_axis_rad[-1] - angle_axis_rad[0])
        - 1.0
    )
    range_normalized = (
        2.0
        * (range_grid - range_axis[0])
        / (range_axis[-1] - range_axis[0])
        - 1.0
    )

    sampling_grid = torch.stack(
        (angle_normalized, range_normalized),
        dim=-1,
    ).unsqueeze(0).expand(
        B * T,
        -1,
        -1,
        -1,
    )

    polar_input = range_angle_power.reshape(
        B * T,
        1,
        R,
        K,
    )

    cartesian_map = F.grid_sample(
        polar_input,
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(
        B,
        T,
        H_forward,
        W_lateral,
    )

    return cartesian_map, forward_axis, lateral_axis


def range_cube_to_rpm_projection_maps(
    range_cube: torch.Tensor,
    range_axis: torch.Tensor,
    wavelength: float,
    projection_config: SingleRadarProjectionConfig | None = None,
    *,
    xy_limits: Tuple[
        Tuple[float, float],
        Tuple[float, float],
    ] = ((0.2, 5.0), (-2.0, 2.0)),
    xz_limits: Tuple[
        Tuple[float, float],
        Tuple[float, float],
    ] = ((0.2, 5.0), (-1.0, 1.5)),
    xy_size: Tuple[int, int] = (256, 256),
    xz_size: Tuple[int, int] = (256, 256),
    clutter_mode: Literal[
        "none",
        "chirp_mean",
        "frame_difference",
    ] = "frame_difference",
) -> RPMProjectionOutput:
    """
    从单雷达 Range FFT cube 生成 RPM 风格水平和垂直投影图。

    xy_limits:
        ((x_min,x_max), (y_min,y_max))。
        x 为前方，y 为左侧。

    xz_limits:
        ((x_min,x_max), (z_min,z_max))。
        x 为前方，z 为上方。

    xy_size:
        (num_x_pixels, num_y_pixels)。
        输出 horizontal_xy_power 为 [B,T_out,num_x,num_y]。

    xz_size:
        (num_z_pixels, num_x_pixels)。
        输出 vertical_xz_power 为 [B,T_out,num_z,num_x]。
    """
    if projection_config is None:
        projection_config = SingleRadarProjectionConfig()

    processed_cube, time_start_index = suppress_static_clutter(
        range_cube,
        mode=clutter_mode,
    )

    range_azimuth_power, azimuth_axis_rad = (
        range_cube_to_range_angle_power(
            range_cube=processed_cube,
            virtual_phase_positions=(
                projection_config.virtual_phase_positions
            ),
            ant_indices=projection_config.azi_ant_indices,
            wavelength=wavelength,
            angle_min_deg=projection_config.azimuth_min_deg,
            angle_max_deg=projection_config.azimuth_max_deg,
            num_angle_beams=(
                projection_config.num_azimuth_beams
            ),
            plane="azimuth",
            apply_array_window=(
                projection_config.azi_apply_array_window
            ),
            phase_sign=projection_config.azimuth_phase_sign,
            chirp_chunk_size=projection_config.chirp_chunk_size,
            angle_method=projection_config.angle_method,
            diagonal_loading=projection_config.diagonal_loading,
            forward_backward_average=projection_config.forward_backward_average,
            music_num_sources=projection_config.music_num_sources_azi,
        )
    )

    range_elevation_power, elevation_axis_rad = (
        range_cube_to_range_angle_power(
            range_cube=processed_cube,
            virtual_phase_positions=(
                projection_config.virtual_phase_positions
            ),
            ant_indices=projection_config.ele_ant_indices,
            wavelength=wavelength,
            angle_min_deg=projection_config.elevation_min_deg,
            angle_max_deg=projection_config.elevation_max_deg,
            num_angle_beams=(
                projection_config.num_elevation_beams
            ),
            plane="elevation",
            apply_array_window=(
                projection_config.ele_apply_array_window
            ),
            phase_sign=projection_config.elevation_phase_sign,
            chirp_chunk_size=projection_config.chirp_chunk_size,
            angle_method=projection_config.angle_method,
            diagonal_loading=projection_config.diagonal_loading,
            forward_backward_average=projection_config.forward_backward_average,
            music_num_sources=projection_config.music_num_sources_ele,
        )
    )

    # 水平图：[B,T_out,X,Y]。
    horizontal_xy_power, horizontal_x_axis, horizontal_y_axis = (
        range_angle_to_cartesian_map(
            range_angle_power=range_azimuth_power,
            range_axis=range_axis,
            angle_axis_rad=azimuth_axis_rad,
            forward_limits=xy_limits[0],
            lateral_limits=xy_limits[1],
            output_size=xy_size,
        )
    )

    # 垂直图内部先生成 [B,T_out,X,Z]。
    vertical_xz_forward_first, vertical_x_axis, vertical_z_axis = (
        range_angle_to_cartesian_map(
            range_angle_power=range_elevation_power,
            range_axis=range_axis,
            angle_axis_rad=elevation_axis_rad,
            forward_limits=xz_limits[0],
            lateral_limits=xz_limits[1],
            output_size=(xz_size[1], xz_size[0]),
        )
    )

    # [B,T_out,X,Z] -> [B,T_out,Z,X]
    vertical_xz_power = (
        vertical_xz_forward_first
        .transpose(-2, -1)
        .contiguous()
    )

    return RPMProjectionOutput(
        horizontal_xy_power=horizontal_xy_power,
        vertical_xz_power=vertical_xz_power,
        range_azimuth_power=range_azimuth_power,
        range_elevation_power=range_elevation_power,
        range_axis=torch.as_tensor(
            range_axis,
            dtype=range_azimuth_power.dtype,
            device=range_azimuth_power.device,
        ),
        azimuth_axis_rad=azimuth_axis_rad,
        elevation_axis_rad=elevation_axis_rad,
        horizontal_x_axis=horizontal_x_axis,
        horizontal_y_axis=horizontal_y_axis,
        vertical_x_axis=vertical_x_axis,
        vertical_z_axis=vertical_z_axis,
        time_start_index=time_start_index,
    )


def power_to_db(
    power: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """线性功率转换为 dB。"""
    return 10.0 * torch.log10(
        power.clamp_min(eps)
    )


__all__ = [
    "SingleRadarProjectionConfig",
    "RPMProjectionOutput",
    "build_virtual_phase_positions",
    "suppress_static_clutter",
    "range_cube_to_range_angle_power",
    "range_angle_to_cartesian_map",
    "range_cube_to_rpm_projection_maps",
    "power_to_db",
]
