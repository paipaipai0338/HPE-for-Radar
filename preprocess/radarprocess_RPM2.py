from typing import *
import math
import torch
import torch.nn.functional as F
import numpy as np
from preprocess.radarprocess import Radar_Config, get_bin_data, doppler_fft, get_radar_res, angle_fft

def _build_direction_vectors(angle_axis_rad: torch.Tensor, plane: Literal["azi", "ele"]) -> torch.Tensor:
    """
    按 x 前、y 左、z 上坐标系构造单位方向向量。

    azi:
        u(theta) = [cos(theta), sin(theta), 0]

    elevation:
        u(phi) = [cos(phi), 0, sin(phi)]
    """
    zeros = torch.zeros_like(angle_axis_rad)

    if plane == "azi":
        return torch.stack(
            (torch.cos(angle_axis_rad),
                torch.sin(angle_axis_rad),
                zeros,
            ),
            dim=-1,
        )

    if plane == "ele":
        return torch.stack(
            (
                torch.cos(angle_axis_rad),
                zeros,
                torch.sin(angle_axis_rad),
            ),
            dim=-1,
        )

# range_cube B, T, R, C, A  - >  range_angle_power B, T, R, K
def range_cube_to_range_angle_power(
    range_cube: torch.Tensor,
    radar_config: Radar_Config,
    plane: Literal["azi", "ele"],
    remove_static: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:

    assert plane in ("azi", "ele"), f"plane 必须为 'azi' 或 'ele'，当前为 {plane}"
    assert radar_config.angle_method in ("bartlett", "mvdr", "music"), f"未知 angle_method={radar_config.angle_method}"

    B, T, R, C, A = range_cube.shape
    device = range_cube.device
    real_dtype = range_cube.real.dtype
    eps = torch.finfo(real_dtype).eps
    if remove_static:
        range_cube = range_cube - range_cube.mean(dim=3, keepdim=True)

    positions = torch.as_tensor(radar_config.virtual_channel_positions, dtype=real_dtype, device=device)
    fov = radar_config.azi_deg if plane == "azi" else radar_config.ele_deg
    ant_indices = radar_config.azi_ant_indices if plane == "azi" else radar_config.ele_ant_indices

    index_tensor = torch.as_tensor(tuple(ant_indices), dtype=torch.long, device=device)

    subarray_cube = torch.index_select(range_cube, dim=-1, index=index_tensor)
    subarray_positions = torch.index_select(positions, dim=0, index=index_tensor)
    subarray_positions = subarray_positions - subarray_positions[:1]

    angle_axis_rad = torch.linspace(math.radians(fov[0]), math.radians(fov[1]), radar_config.num_angle_beams,dtype=real_dtype, device=device)

    direction_vectors = _build_direction_vectors(angle_axis_rad, plane)
    path_difference = torch.einsum("mc,kc->mk", subarray_positions, direction_vectors)
    phase = 2.0 * math.pi / radar_config.lam * path_difference

    # steering_weight 保持与原代码一致，用于 y = sum(x_m * steering_weight_m)。
    steering_weight = torch.exp(1j * phase).to(dtype=range_cube.dtype)
    steering_vector = steering_weight.conj()
    num_selected_ant = subarray_cube.shape[-1]

    # -------------------------------------------------------------------------
    # Bartlett：保留原常规波束形成实现，便于与超分辨结果直接对比。
    # -------------------------------------------------------------------------
    if radar_config.angle_method == "bartlett":
        
        normalization = torch.tensor(float(num_selected_ant ** 2), dtype=real_dtype, device=device)

        power_sum = torch.zeros((B, T, R, radar_config.num_angle_beams), dtype=real_dtype, device=device)
        for start in range(0, C, 8):
            stop = min(start + 8, C)
            angle_spectrum_chunk = torch.einsum(
                "btrcm,mk->btrck", subarray_cube[:, :, :, start:stop, :], steering_weight,
            )
            power_sum += angle_spectrum_chunk.abs().square().sum(dim=3)

        range_angle_power = power_sum / float(C) / normalization
    else:
        # -------------------------------------------------------------------------
        # MVDR/MUSIC：使用 C 个 chirp 作为快拍，按每个 range bin 构造空间协方差。
        # Rxx[m,n] = E[x_m * conj(x_n)]。
        # -------------------------------------------------------------------------
        covariance = torch.einsum(
            "btrcm,btrcn->btrmn", subarray_cube, subarray_cube.conj(),
        ) / float(C)
        covariance = 0.5 * (covariance + covariance.conj().transpose(-2, -1))

        exchange = torch.eye(num_selected_ant, dtype=range_cube.dtype, device=device).flip(0)
        covariance_fb = torch.matmul(torch.matmul(exchange, covariance.conj()), exchange)
        covariance = 0.5 * (covariance + covariance_fb)
        covariance = 0.5 * (covariance + covariance.conj().transpose(-2, -1))

        trace_mean = covariance.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1)
        loading = 1e-2 * trace_mean.clamp_min(eps) + eps
        identity = torch.eye(num_selected_ant, dtype=range_cube.dtype, device=device)
        covariance_loaded = covariance + loading[..., None, None] * identity

        if radar_config.angle_method == "mvdr":
            covariance_inverse = torch.linalg.inv(covariance_loaded)
            denominator = torch.einsum(
                "mk,btrmn,nk->btrk", steering_vector.conj(), covariance_inverse, steering_vector,
            ).real.clamp_min(eps)
            range_angle_power = 1.0 / denominator

        if radar_config.angle_method == "music":
            _, eigenvectors = torch.linalg.eigh(covariance_loaded)
            noise_subspace = eigenvectors[..., :num_selected_ant - 1]
            noise_projection = torch.einsum(
                "btrmq,mk->btrqk", noise_subspace.conj(), steering_vector,
            )
            denominator = noise_projection.abs().square().sum(dim=-2).clamp_min(eps)
            music_spectrum = 1.0 / denominator
            music_spectrum = music_spectrum / music_spectrum.amax(dim=-1, keepdim=True).clamp_min(eps)
            range_power = subarray_cube.abs().square().mean(dim=(-1, -2))
            range_angle_power = music_spectrum * range_power.unsqueeze(-1)

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

if __name__ == "__main__":
    import os
    from matplotlib import pyplot as plt

    bin_file_path = "/mnt/huawei/20260709/data_collection/group_017/dpct高位机/Bin"
    files = os.listdir(bin_file_path)
    files = [f for f in files if f.endswith('.bin')]
    files.sort()

    def power_to_numpy(power: torch.Tensor) -> np.ndarray:
        """保留算法输出的原始线性谱值。"""
        return power.detach().real.cpu().numpy()

    for file in files:
        print(f"Processing file: {file}")
        file_path = os.path.join(bin_file_path, file)
        fft1d = get_bin_data(file_path)
        if fft1d is None:
            print(f"Skip invalid bin file: {file_path}")
            continue

        fft1d = torch.from_numpy(fft1d)[None, None, ...]
        radar_config = Radar_Config()
        range_res, _, _, _ = get_radar_res(radar_config)
        range_axis = (
            torch.arange(
                fft1d.shape[2],
                dtype=fft1d.real.dtype,
                device=fft1d.device,
            )
            * range_res
        )
        xyz_limits = ((0.2, 5.0), (-2.0, 2.0), (-1.0, 1.5))
        cartesian_size = (256, 256)

        method_results = []
        with torch.no_grad():
            for method in ("bartlett", "mvdr", "music"):
                radar_config.angle_method = method
                range_azi_power, azi_axis_rad = (
                    range_cube_to_range_angle_power(
                        fft1d,
                        radar_config,
                        plane="azi",
                        remove_static=True,
                    )
                )
                horizontal_power, horizontal_x_axis, horizontal_y_axis = (
                    range_angle_power_to_cartesian_map(
                        range_azi_power,
                        range_axis,
                        azi_axis_rad,
                        xyz_limits,
                        cartesian_size,
                        plane="horizontal",
                    )
                )
                range_ele_power, ele_axis_rad = (
                    range_cube_to_range_angle_power(
                        fft1d,
                        radar_config,
                        plane="ele",
                        remove_static=True,
                    )
                )
                vertical_power, vertical_x_axis, vertical_z_axis = (
                    range_angle_power_to_cartesian_map(
                        range_ele_power,
                        range_axis,
                        ele_axis_rad,
                        xyz_limits,
                        cartesian_size,
                        plane="vertical",
                    )
                )
                method_results.append(
                    (
                        method,
                        power_to_numpy(range_azi_power[0, 0]),
                        power_to_numpy(range_ele_power[0, 0]),
                        power_to_numpy(horizontal_power[0, 0]),
                        power_to_numpy(vertical_power[0, 0]),
                    )
                )

        azi_axis_deg = torch.rad2deg(azi_axis_rad).cpu().numpy()
        ele_axis_deg = torch.rad2deg(ele_axis_rad).cpu().numpy()
        num_range_bins = fft1d.shape[2]

        horizontal_x_axis = horizontal_x_axis.cpu().numpy()
        horizontal_y_axis = horizontal_y_axis.cpu().numpy()
        vertical_x_axis = vertical_x_axis.cpu().numpy()
        vertical_z_axis = vertical_z_axis.cpu().numpy()

        fig, axes = plt.subplots(3, 4, figsize=(26, 13))
        for row, (
            method,
            azi_power,
            ele_power,
            horizontal_power,
            vertical_power,
        ) in enumerate(method_results):
            azi_image = axes[row, 0].imshow(
                azi_power,
                aspect="auto",
                extent=[
                    azi_axis_deg[0],
                    azi_axis_deg[-1],
                    0,
                    num_range_bins,
                ],
                origin="lower",
                cmap="jet",
            )
            axes[row, 0].set_title(
                f"{method.upper()} Dynamic Range-Azimuth"
            )
            axes[row, 0].set_xlabel("Azimuth Angle (deg)")
            axes[row, 0].set_ylabel("Range Bin")
            fig.colorbar(azi_image, ax=axes[row, 0], label="Raw spectrum (linear)")

            ele_image = axes[row, 1].imshow(
                ele_power,
                aspect="auto",
                extent=[
                    ele_axis_deg[0],
                    ele_axis_deg[-1],
                    0,
                    num_range_bins,
                ],
                origin="lower",
                cmap="jet",
            )
            axes[row, 1].set_title(
                f"{method.upper()} Dynamic Range-Elevation"
            )
            axes[row, 1].set_xlabel("Elevation Angle (deg)")
            axes[row, 1].set_ylabel("Range Bin")
            fig.colorbar(ele_image, ax=axes[row, 1], label="Raw spectrum (linear)")

            horizontal_image = axes[row, 2].imshow(
                horizontal_power.T,
                aspect="auto",
                extent=[
                    horizontal_x_axis[0],
                    horizontal_x_axis[-1],
                    horizontal_y_axis[0],
                    horizontal_y_axis[-1],
                ],
                origin="lower",
                cmap="jet",
            )
            axes[row, 2].set_title(
                f"{method.upper()} Horizontal XY Projection"
            )
            axes[row, 2].set_xlabel("Forward X (m)")
            axes[row, 2].set_ylabel("Lateral Y (m)")
            fig.colorbar(
                horizontal_image,
                ax=axes[row, 2],
                label="Raw spectrum (linear)",
            )

            vertical_image = axes[row, 3].imshow(
                vertical_power.T,
                aspect="auto",
                extent=[
                    vertical_x_axis[0],
                    vertical_x_axis[-1],
                    vertical_z_axis[0],
                    vertical_z_axis[-1],
                ],
                origin="lower",
                cmap="jet",
            )
            axes[row, 3].set_title(
                f"{method.upper()} Vertical XZ Projection"
            )
            axes[row, 3].set_xlabel("Forward X (m)")
            axes[row, 3].set_ylabel("Height Z (m)")
            fig.colorbar(
                vertical_image,
                ax=axes[row, 3],
                label="Raw spectrum (linear)",
            )

        fig.suptitle(
            f"Static-clutter-removed angle spectra: {file}",
            fontsize=14,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        output_path = f"range_angle_methods_dynamic.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved comparison figure: {output_path}")
