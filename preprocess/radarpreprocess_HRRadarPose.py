import math
from typing import Literal, Tuple

import torch
import torch.nn.functional as F

from preprocess.radarprocess import (
    Radar_Config,
    build_azimuth_elevation_directions,
    build_steering_matrix,
    conventional_dbf_power,
    doppler_fft_torch,
    get_bin_data,
    get_radar_res,
)


def range_cube_to_range_doppler_azi_ele(
    range_cube: torch.Tensor,
    radar_config: Radar_Config,
    remove_static: bool = False,
    num_azimuth_bins: int = 128,
    num_elevation_bins: int = 128,
    window: bool = True,
    doppler_mode: Literal["normal", "firmware_tdm"] = "firmware_tdm",
    angle_chunk_size: int = 256,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Convert a range cube to an HRRadarPose 4-D radar power tensor with 2-D DBF.

    Args:
        range_cube: Complex tensor with shape ``[B, T, R, C, M]``, where
            ``C`` is the chirp dimension and ``M`` is the virtual-antenna
            dimension.
        radar_config: Radar geometry and waveform configuration.
        remove_static: Subtract the complex slow-time mean before Doppler FFT.
        num_azimuth_bins: Number of output azimuth beams. HRRadarPose uses 128.
        num_elevation_bins: Number of output elevation beams. HRRadarPose uses 32.
        window: Apply a Hann window before Doppler FFT.
        doppler_mode: Match either the normal or firmware-TDM Doppler path in
            :mod:`preprocess.radarprocess`.
        angle_chunk_size: Number of joint azimuth/elevation DBF beams evaluated
            at once. This limits temporary GPU memory, not the output shape.

    Returns:
        ``(power, range_axis, velocity_axis, azimuth_axis_rad,
        elevation_axis_rad)``. The real power tensor has shape
        ``[B, T, R, D, A, E]``, where ``D=C``,
        ``A=num_azimuth_bins`` and ``E=num_elevation_bins``.
    """
    if range_cube.ndim != 5 or not range_cube.is_complex():
        raise ValueError(
            "range_cube 必须为复数 [B,T,R,C,M]，"
            f"实际为 shape={tuple(range_cube.shape)}, dtype={range_cube.dtype}"
        )
    if any(size <= 0 for size in range_cube.shape):
        raise ValueError(f"range_cube 的各维必须大于 0，实际为 {tuple(range_cube.shape)}")
    if num_azimuth_bins <= 0 or num_elevation_bins <= 0:
        raise ValueError("num_azimuth_bins 和 num_elevation_bins 必须大于 0")
    if angle_chunk_size <= 0:
        raise ValueError("angle_chunk_size 必须大于 0")

    _, _, _, _, num_antennas = range_cube.shape
    device = range_cube.device
    real_dtype = range_cube.real.dtype

    positions = torch.as_tensor(
        radar_config.virtual_channel_positions,
        dtype=real_dtype,
        device=device,
    )
    if positions.shape != (num_antennas, 3):
        raise ValueError(
            "virtual_channel_positions 必须与输入天线维匹配，"
            f"实际 positions={tuple(positions.shape)}, M={num_antennas}"
        )

    if remove_static:
        range_cube = range_cube - range_cube.mean(dim=3, keepdim=True)

    doppler_cube, velocity_axis = doppler_fft_torch(
        range_cube,
        radar_config,
        window=window,
        doppler_mode=doppler_mode,
        window_periodic=False,
    )

    range_resolution, _, _, _ = get_radar_res(radar_config)
    range_axis = (
        torch.arange(
            range_cube.shape[2],
            dtype=real_dtype,
            device=device,
        )
        * range_resolution
    )

    azimuth_axis_rad = torch.linspace(
        math.radians(radar_config.azi_deg[0]),
        math.radians(radar_config.azi_deg[1]),
        num_azimuth_bins,
        dtype=real_dtype,
        device=device,
    )
    elevation_axis_rad = torch.linspace(
        math.radians(radar_config.ele_deg[0]),
        math.radians(radar_config.ele_deg[1]),
        num_elevation_bins,
        dtype=real_dtype,
        device=device,
    )

    direction_vectors = build_azimuth_elevation_directions(
        azimuth_axis_rad,
        elevation_axis_rad,
    )
    dbf_weights = build_steering_matrix(
        positions,
        direction_vectors,
        radar_config.lam,
        normalize=True,
    ).to(doppler_cube.dtype)
    power_flat = conventional_dbf_power(
        doppler_cube,
        dbf_weights,
        direction_chunk_size=angle_chunk_size,
    )

    power = power_flat.reshape(
        *doppler_cube.shape[:-1],
        num_azimuth_bins,
        num_elevation_bins,
    ).contiguous()

    return (
        power,
        range_axis,
        velocity_axis,
        azimuth_axis_rad,
        elevation_axis_rad,
    )


def range_doppler_azi_ele_to_doppler_xyz(
    range_doppler_azi_ele: torch.Tensor,
    range_axis: torch.Tensor,
    azimuth_axis_rad: torch.Tensor,
    elevation_axis_rad: torch.Tensor,
    xyz_limits: Tuple[
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
    ],
    cube_size: Tuple[int, int, int],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Resample an R-D-A-E tensor into the HRRadarPose Cartesian radar cube.

    Args:
        range_doppler_azi_ele: Real power tensor ``[B, T, R, D, A, E]``.
        range_axis: Uniformly sampled range axis in metres, shape ``[R]``.
        azimuth_axis_rad: Uniformly sampled azimuth axis in radians, shape
            ``[A]``.
        elevation_axis_rad: Uniformly sampled elevation axis in radians, shape
            ``[E]``.
        xyz_limits: ``((x_min, x_max), (y_min, y_max), (z_min, z_max))``.
        cube_size: Cartesian output size ``(X, Y, Z)``.  For example,
            ``(256, 128, 32)`` gives the spatial dimensions used by HRRadarPose.

    Returns:
        ``(doppler_xyz, x_axis, y_axis, z_axis)``. ``doppler_xyz`` has shape
        ``[B, T, D, X, Y, Z]``; the three axes are in metres.
    """
    if range_doppler_azi_ele.ndim != 6:
        raise ValueError(
            "range_doppler_azi_ele 必须为 [B,T,R,D,A,E]，"
            f"实际 shape={tuple(range_doppler_azi_ele.shape)}"
        )
    if (
        range_doppler_azi_ele.is_complex()
        or not range_doppler_azi_ele.is_floating_point()
    ):
        raise ValueError(
            "range_doppler_azi_ele 必须为实数浮点功率张量，"
            f"实际 dtype={range_doppler_azi_ele.dtype}"
        )
    if any(size <= 0 for size in range_doppler_azi_ele.shape):
        raise ValueError(
            "range_doppler_azi_ele 的各维必须大于 0，"
            f"实际 shape={tuple(range_doppler_azi_ele.shape)}"
        )
    if len(xyz_limits) != 3 or any(
        len(limits) != 2 or limits[0] >= limits[1] for limits in xyz_limits
    ):
        raise ValueError(
            "xyz_limits 必须为三个递增区间："
            "((x_min,x_max),(y_min,y_max),(z_min,z_max))"
        )
    if len(cube_size) != 3 or any(size <= 0 for size in cube_size):
        raise ValueError("cube_size 必须为三个正整数 (X,Y,Z)")

    B, T, R, D, A, E = range_doppler_azi_ele.shape
    X, Y, Z = cube_size
    device = range_doppler_azi_ele.device
    dtype = range_doppler_azi_ele.dtype

    def prepare_axis(
        axis: torch.Tensor,
        expected_size: int,
        name: str,
    ) -> torch.Tensor:
        axis = torch.as_tensor(axis, dtype=dtype, device=device)
        if axis.ndim != 1 or axis.numel() != expected_size:
            raise ValueError(
                f"{name} 必须为 [{expected_size}]，实际 shape={tuple(axis.shape)}"
            )
        if expected_size < 2:
            raise ValueError(f"{name} 至少需要两个采样点")
        differences = axis[1:] - axis[:-1]
        if not torch.isfinite(axis).all() or not torch.all(differences > 0):
            raise ValueError(f"{name} 必须有限且严格递增")
        if not torch.allclose(
            differences,
            differences[:1].expand_as(differences),
            rtol=1e-4,
            atol=torch.finfo(dtype).eps * 16,
        ):
            raise ValueError(f"{name} 必须为等间距坐标轴")
        return axis

    range_axis = prepare_axis(range_axis, R, "range_axis")
    azimuth_axis_rad = prepare_axis(azimuth_axis_rad, A, "azimuth_axis_rad")
    elevation_axis_rad = prepare_axis(
        elevation_axis_rad,
        E,
        "elevation_axis_rad",
    )

    x_axis = torch.linspace(
        xyz_limits[0][0],
        xyz_limits[0][1],
        X,
        dtype=dtype,
        device=device,
    )
    y_axis = torch.linspace(
        xyz_limits[1][0],
        xyz_limits[1][1],
        Y,
        dtype=dtype,
        device=device,
    )
    z_axis = torch.linspace(
        xyz_limits[2][0],
        xyz_limits[2][1],
        Z,
        dtype=dtype,
        device=device,
    )

    x_grid, y_grid, z_grid = torch.meshgrid(
        x_axis,
        y_axis,
        z_axis,
        indexing="ij",
    )
    horizontal_range = torch.sqrt(x_grid.square() + y_grid.square())
    range_grid = torch.sqrt(horizontal_range.square() + z_grid.square())
    azimuth_grid = torch.atan2(y_grid, x_grid)
    elevation_grid = torch.atan2(z_grid, horizontal_range)

    def normalize(grid: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        return 2.0 * (grid - axis[0]) / (axis[-1] - axis[0]) - 1.0

    # For 5-D grid_sample the grid components address input W, H and D.
    # The polar input is laid out as [R, A, E], hence the component order is
    # [elevation, azimuth, range].
    sampling_grid = torch.stack(
        (
            normalize(elevation_grid, elevation_axis_rad),
            normalize(azimuth_grid, azimuth_axis_rad),
            normalize(range_grid, range_axis),
        ),
        dim=-1,
    ).unsqueeze(0)

    polar_input = range_doppler_azi_ele.permute(0, 1, 3, 2, 4, 5).reshape(
        B * T * D,
        1,
        R,
        A,
        E,
    )
    doppler_xyz = F.grid_sample(
        polar_input,
        sampling_grid.expand(B * T * D, -1, -1, -1, -1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(B, T, D, X, Y, Z)

    return doppler_xyz.contiguous(), x_axis, y_axis, z_axis


def _power_to_relative_db(
    power: torch.Tensor,
    dynamic_range_db: float,
) -> torch.Tensor:
    """Convert a non-negative power tensor to peak-relative dB."""
    eps = torch.finfo(power.dtype).tiny
    relative_power = power / power.amax().clamp_min(eps)
    power_db = 10.0 * torch.log10(relative_power.clamp_min(eps))
    return power_db.clamp_min(-dynamic_range_db)


def _build_preview_tensors(
    range_doppler_azi_ele: torch.Tensor,
    doppler_xyz: torch.Tensor,
    dynamic_range_db: float,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build six 2-D peak-relative-dB views from one B/T element."""
    polar_power = range_doppler_azi_ele[0, 0]
    cartesian_power = doppler_xyz[0, 0]

    range_doppler = polar_power.mean(dim=(-1, -2))
    range_azimuth = polar_power.amax(dim=(1, 3))
    range_elevation = polar_power.amax(dim=(1, 2))

    # cartesian_power is [D,X,Y,Z]. Max projections retain sparse targets.
    xy_projection = cartesian_power.amax(dim=(0, 3)).transpose(0, 1)
    xz_projection = cartesian_power.amax(dim=(0, 2)).transpose(0, 1)
    yz_projection = cartesian_power.amax(dim=(0, 1)).transpose(0, 1)

    return tuple(
        _power_to_relative_db(view, dynamic_range_db).detach().cpu()
        for view in (
            range_doppler,
            range_azimuth,
            range_elevation,
            xy_projection,
            xz_projection,
            yz_projection,
        )
    )


if __name__ == '__main__':
    from pathlib import Path
    import os
    import numpy as np
    from matplotlib import pyplot as plt

    from data2datasets.dataset import HPE_Dataset
    from preprocess.gtprocess import get_gt_data
    from utils.COCO import COCO_SKELETON

    root_path = Path(r'/mnt/huawei/20260703/data_collection/group_021')
    bin_path = root_path / 'dpct高位机' / 'Bin'
    gt_path = root_path / 'camera results' / 'smoothed 3D'

    bin_files = sorted(file for file in os.listdir(bin_path) if file.endswith('.bin'))
    gt_files = sorted(file for file in os.listdir(gt_path) if file.endswith('.pkl'))

    if not bin_files:
        raise RuntimeError(f'没有找到 BIN 文件: {bin_path}')
    if not gt_files:
        raise RuntimeError(f'没有找到 GT 文件: {gt_path}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    radar_config = Radar_Config()
    xyz_limits = ((0.0, 6.0), (-3.0, 3.0), (-2.0, 2.0))
    cube_size = (128, 128, 96)  # X, Y, Z
    dynamic_range_db = 40.0
    output_dir = Path('temp')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 时间对齐
    # 复用 Dataset 已有的全局单调一对一匹配，避免各自实现一套时间戳逻辑。
    dataset_tools = HPE_Dataset.__new__(HPE_Dataset)
    dataset_tools.suffix_map = {
        'radar_high_bin': '.bin',
        'gt': '.pkl',
    }
    date_path = root_path.parent.parent
    dataset_tools.root_path = date_path.parent
    dataset_tools.calib_cache = {}
    aligned_frames = dataset_tools._align_multi_sensor_files(
        sources={
            'radar_high_bin': bin_path,
            'gt': gt_path,
        },
        max_delta_sec=0.1,
        one_to_one=True,
        base_source='radar_high_bin',
    )
    aligned_bin_files = aligned_frames['radar_high_bin']
    aligned_gt_files = aligned_frames['gt']
    if not aligned_bin_files:
        raise RuntimeError('BIN 与 GT 时间对齐后没有有效帧')

    calibration = dataset_tools._load_calib_T(date_path.name)['gt_to_high']
    print(
        f'files: BIN={len(bin_files)}, GT={len(gt_files)}, '
        f'aligned={len(aligned_bin_files)}, device={device}'
    )

    # 文件读取
    for frame_index, (bin_file, gt_file) in enumerate(
        zip(aligned_bin_files, aligned_gt_files),
        start=1,
    ):
        range_frame = get_bin_data(bin_file, radar_config)
        if range_frame is None:
            print(f'跳过无效 BIN: {bin_file}')
            continue

        gt_camera = get_gt_data(gt_file)
        gt_radar = dataset_tools._transform_gt_sequence(
            [gt_camera],
            R=calibration['R'],
            t=calibration['t'],
        )[0]
        range_cube = torch.from_numpy(range_frame)[None, None].to(
            device=device,
            non_blocking=True,
        )

        # 数据处理与画图
        with torch.inference_mode():
            (
                range_doppler_azi_ele,
                range_axis,
                velocity_axis,
                azimuth_axis_rad,
                elevation_axis_rad,
            ) = range_cube_to_range_doppler_azi_ele(
                range_cube,
                radar_config,
                remove_static=True,
            )
            doppler_xyz, x_axis, y_axis, z_axis = (
                range_doppler_azi_ele_to_doppler_xyz(
                    range_doppler_azi_ele,
                    range_axis,
                    azimuth_axis_rad,
                    elevation_axis_rad,
                    xyz_limits,
                    cube_size,
                )
            )
            previews = _build_preview_tensors(
                range_doppler_azi_ele,
                doppler_xyz,
                dynamic_range_db,
            )

        rdae_shape = tuple(range_doppler_azi_ele.shape)
        dxyz_shape = tuple(doppler_xyz.shape)
        (
            range_doppler_db,
            range_azimuth_db,
            range_elevation_db,
            xy_db,
            xz_db,
            yz_db,
        ) = (preview.numpy() for preview in previews)
        range_values = range_axis.cpu().numpy()
        velocity_values = velocity_axis.cpu().numpy()
        azimuth_values = torch.rad2deg(azimuth_axis_rad).cpu().numpy()
        elevation_values = torch.rad2deg(elevation_axis_rad).cpu().numpy()
        x_values = x_axis.cpu().numpy()
        y_values = y_axis.cpu().numpy()
        z_values = z_axis.cpu().numpy()

        figure, axes = plt.subplots(2, 3, figsize=(19, 11))
        image_specs = (
            (
                axes[0, 0], range_doppler_db,
                (velocity_values[0], velocity_values[-1],
                 range_values[0], range_values[-1]),
                'Range-Doppler', 'Velocity (m/s)', 'Range (m)',
            ),
            (
                axes[0, 1], range_azimuth_db,
                (azimuth_values[0], azimuth_values[-1],
                 range_values[0], range_values[-1]),
                'Range-Azimuth DBF', 'Azimuth (deg)', 'Range (m)',
            ),
            (
                axes[0, 2], range_elevation_db,
                (elevation_values[0], elevation_values[-1],
                 range_values[0], range_values[-1]),
                'Range-Elevation DBF', 'Elevation (deg)', 'Range (m)',
            ),
            (
                axes[1, 0], xy_db,
                (x_values[0], x_values[-1], y_values[0], y_values[-1]),
                'Doppler-XYZ max projection: XY', 'Forward X (m)',
                'Lateral Y (m)',
            ),
            (
                axes[1, 1], xz_db,
                (x_values[0], x_values[-1], z_values[0], z_values[-1]),
                'Doppler-XYZ max projection: XZ', 'Forward X (m)',
                'Height Z (m)',
            ),
            (
                axes[1, 2], yz_db,
                (y_values[0], y_values[-1], z_values[0], z_values[-1]),
                'Doppler-XYZ max projection: YZ', 'Lateral Y (m)',
                'Height Z (m)',
            ),
        )
        for axis, image_data, extent, title, xlabel, ylabel in image_specs:
            image = axis.imshow(
                image_data,
                extent=extent,
                origin='lower',
                aspect='auto',
                cmap='turbo',
                vmin=-dynamic_range_db,
                vmax=0.0,
            )
            axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
            figure.colorbar(image, ax=axis, label='Relative power (dB)')

        def draw_pose(axis, first_coord, second_coord):
            for person_index in range(gt_radar.shape[0]):
                joints = gt_radar[person_index]
                joint_valid = np.isfinite(joints).all(axis=1)
                color = plt.get_cmap('tab10')(person_index % 10)
                axis.scatter(
                    first_coord[person_index][joint_valid],
                    second_coord[person_index][joint_valid],
                    s=18,
                    color=color,
                    edgecolors='white',
                    linewidths=0.4,
                    zorder=3,
                )
                for joint_a, joint_b in COCO_SKELETON:
                    if joint_valid[joint_a] and joint_valid[joint_b]:
                        axis.plot(
                            [first_coord[person_index, joint_a],
                             first_coord[person_index, joint_b]],
                            [second_coord[person_index, joint_a],
                             second_coord[person_index, joint_b]],
                            color=color,
                            linewidth=1.3,
                            zorder=3,
                        )

        if gt_radar.shape[0] > 0:
            gt_range = np.linalg.norm(gt_radar, axis=-1)
            gt_horizontal_range = np.linalg.norm(gt_radar[..., :2], axis=-1)
            gt_azimuth_deg = np.rad2deg(
                np.arctan2(gt_radar[..., 1], gt_radar[..., 0])
            )
            gt_elevation_deg = np.rad2deg(
                np.arctan2(gt_radar[..., 2], gt_horizontal_range)
            )
            draw_pose(axes[0, 1], gt_azimuth_deg, gt_range)
            draw_pose(axes[0, 2], gt_elevation_deg, gt_range)
            draw_pose(axes[1, 0], gt_radar[..., 0], gt_radar[..., 1])
            draw_pose(axes[1, 1], gt_radar[..., 0], gt_radar[..., 2])
            draw_pose(axes[1, 2], gt_radar[..., 1], gt_radar[..., 2])

        bin_timestamp = int(Path(bin_file).stem.replace('_', ''))
        gt_timestamp = int(Path(gt_file).stem.replace('_', ''))
        time_delta_ms = abs(bin_timestamp - gt_timestamp) / 1e6
        figure.suptitle(
            f'HRRadarPose clutter-removed preview '
            f'{frame_index}/{len(aligned_bin_files)} | '
            f'align error={time_delta_ms:.2f} ms\n'
            f'RDAE={rdae_shape}, DXYZ={dxyz_shape}',
            fontsize=13,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

        # 只保留 10 个预览槽位，第 11 帧开始按 frame_index % 10 覆盖。
        output_path = output_dir / f'temp_{(frame_index - 1) % 10}.png'
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        print(
            f'[{frame_index}/{len(aligned_bin_files)}] '
            f'saved (overwrite): {output_path.resolve()}'
        )
