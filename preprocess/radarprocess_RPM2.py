import math
from typing import Literal, Tuple

import torch
import torch.nn.functional as F
from preprocess.radarprocess import (
    Radar_Config,
    build_direction_vectors,
    build_steering_matrix,
    get_radar_res,
)


def range_cube_to_matched_filter_angle_power(
    range_cube: torch.Tensor,
    radar_config: Radar_Config,
    plane: Literal["azi", "ele"],
    remove_static: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 Range FFT 做 chirp 均值对消、空间匹配滤波和功率求和。"""
    if range_cube.ndim != 5 or not range_cube.is_complex():
        raise ValueError(f"range_cube 必须为复数 [B,T,R,C,A]，实际为 {tuple(range_cube.shape)} / {range_cube.dtype}")
    if plane not in ("azi", "ele"):
        raise ValueError(f"plane 必须为 'azi' 或 'ele'，实际为 {plane!r}")

    if remove_static:
        range_cube = range_cube - range_cube.mean(dim=3, keepdim=True)
    real_dtype = range_cube.real.dtype
    device = range_cube.device
    antenna_indices = (
        radar_config.azi_ant_indices
        if plane == "azi"
        else radar_config.ele_ant_indices
    )
    antenna_indices = torch.as_tensor(
        antenna_indices,
        dtype=torch.long,
        device=device,
    )
    subarray_cube = torch.index_select(range_cube, -1, antenna_indices)
    positions = torch.as_tensor(
        radar_config.virtual_channel_positions,
        dtype=real_dtype,
        device=device,
    )
    subarray_positions = torch.index_select(positions, 0, antenna_indices)
    fov = radar_config.azi_deg if plane == "azi" else radar_config.ele_deg
    angle_axis_rad = torch.linspace(
        math.radians(fov[0]),
        math.radians(fov[1]),
        radar_config.num_angle_beams,
        dtype=real_dtype,
        device=device,
    )
    steering_weights = build_steering_matrix(
        subarray_positions,
        build_direction_vectors(angle_axis_rad, plane),
        radar_config.lam,
    ).to(range_cube.dtype)

    range_angle_power = torch.zeros(
        (*subarray_cube.shape[:3], radar_config.num_angle_beams),
        dtype=real_dtype,
        device=device,
    )
    for start in range(0, subarray_cube.shape[3], 8):
        matched = torch.einsum(
            "btrcm,mk->btrck",
            subarray_cube[..., start:start + 8, :],
            steering_weights,
        )
        range_angle_power += matched.abs().square().sum(dim=3)

    return range_angle_power, angle_axis_rad

# range_angle_power B, T, R, K -> cartesian_map B, T, H, W
def range_angle_power_to_cartesian_map(
        range_angle_power: torch.Tensor, 
        range_axis: torch.Tensor, 
        angle_axis_rad: torch.Tensor, 
        xyz_limits: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        map_size: Tuple[int, int],
        plane: Literal["horizontal", "vertical"] = "horizontal",
    ) -> torch.Tensor:
    B, T, R, K = range_angle_power.shape
    H, W = map_size

    if H <= 0 or W <= 0:
        raise ValueError("map_size 的两个维度必须 > 0")

    device = range_angle_power.device
    dtype = range_angle_power.dtype

    range_axis = torch.as_tensor(range_axis, dtype=dtype, device=device)
    angle_axis_rad = torch.as_tensor(angle_axis_rad, dtype=dtype, device=device)

    if plane == "horizontal":
        H_axis = torch.linspace(xyz_limits[0][0], xyz_limits[0][1], H, dtype=dtype, device=device)  # X
        W_axis = torch.linspace(xyz_limits[1][0], xyz_limits[1][1], W, dtype=dtype, device=device)  # Y
    elif plane == "vertical":
        H_axis = torch.linspace(xyz_limits[0][0], xyz_limits[0][1], H, dtype=dtype, device=device)  # X
        W_axis = torch.linspace(xyz_limits[2][0], xyz_limits[2][1], W, dtype=dtype, device=device)  # Z
    else:
        raise ValueError(f"plane 必须为 'horizontal' 或 'vertical'，当前为 {plane}")
    

    H_grid, W_grid = torch.meshgrid(H_axis, W_axis, indexing="ij")

    range_grid = torch.sqrt(H_grid.square() + W_grid.square())
    angle_grid = torch.atan2(W_grid, H_grid)

    # 按照torch.nn.functional.grid_sample(要求转化为坐标范围 [-1,1]
    angle_normalized = (2.0 * (angle_grid - angle_axis_rad[0]) / (angle_axis_rad[-1] - angle_axis_rad[0]) - 1.0)
    range_normalized = (2.0 * (range_grid - range_axis[0]) / (range_axis[-1] - range_axis[0]) - 1.0)

    sampling_grid = torch.stack((angle_normalized, range_normalized), dim=-1,).unsqueeze(0).expand(B * T, -1, -1, -1)

    polar_input = range_angle_power.reshape(B * T, 1, R, K)

    cartesian_map = F.grid_sample(polar_input, sampling_grid, mode="bilinear", padding_mode="zeros", align_corners=True,).reshape(B, T, H, W)

    return cartesian_map, H_axis, W_axis

def range_cube_to_rpm2_maps(
    range_cube: torch.Tensor,
    radar_config: Radar_Config,
    xyz_limits: Tuple[Tuple[float, float], ...],
    map_size: Tuple[int, int],
    remove_static: bool = True,
) -> torch.Tensor:
    """Apply the same BIN wrapper used by training and return [B,T,2,H,W]."""
    range_res, _, _, _ = get_radar_res(radar_config)
    range_axis = (
        torch.arange(
            range_cube.shape[2],
            dtype=range_cube.real.dtype,
            device=range_cube.device,
        )
        * range_res
    )
    range_azi_power, azi_axis_rad = range_cube_to_matched_filter_angle_power(
        range_cube=range_cube,
        radar_config=radar_config,
        plane="azi",
        remove_static=remove_static,
    )
    range_ele_power, ele_axis_rad = range_cube_to_matched_filter_angle_power(
        range_cube=range_cube,
        radar_config=radar_config,
        plane="ele",
        remove_static=remove_static,
    )
    horizontal_map, _, _ = range_angle_power_to_cartesian_map(
        range_angle_power=range_azi_power,
        range_axis=range_axis,
        angle_axis_rad=azi_axis_rad,
        xyz_limits=xyz_limits,
        map_size=map_size,
        plane="horizontal",
    )
    vertical_map, _, _ = range_angle_power_to_cartesian_map(
        range_angle_power=range_ele_power,
        range_axis=range_axis,
        angle_axis_rad=ele_axis_rad,
        xyz_limits=xyz_limits,
        map_size=map_size,
        plane="vertical",
    )
    return torch.stack((horizontal_map, vertical_map), dim=2).contiguous()
