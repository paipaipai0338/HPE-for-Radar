"""DXYZ + Sky-style KXYZ for training. No CFAR or point-cloud generation."""
from functools import lru_cache
import math
import torch
import torch.nn.functional as F
from preprocess.radarprocess import (
    doppler_fft_torch, get_radar_res, build_azimuth_elevation_directions,
    build_steering_matrix, conventional_dbf_power,
)
from preprocess.radarpreprocess_HRRadarPose import _cartesian_sampling_grid
from preprocess.radarpreprocess_skyrelay import AlgCfg, StructParam, SignalProcessConfig, MAX_FRAME_NUM


@lru_cache(maxsize=4)
def _geometry(device, dtype, positions, wavelength, range_step, range_count,
              azi_deg, ele_deg, angle_bins, xyz_limits, cube_size):
    if (range_count < 2 or len(angle_bins) != 2 or min(angle_bins) < 2 or len(cube_size) != 3
            or min(cube_size) <= 0 or len(xyz_limits) != 3
            or any(len(pair) != 2 or not pair[0] < pair[1] for pair in xyz_limits)
            or not azi_deg[0] < azi_deg[1] or not ele_deg[0] < ele_deg[1]
            or not range_step > 0
            or not all(math.isfinite(v) for pair in xyz_limits for v in pair)
            or not all(math.isfinite(v) for v in (*azi_deg,*ele_deg,range_step))):
        raise ValueError('Invalid spectra axes, XYZ limits or cube size')
    azimuth = torch.linspace(*[torch.pi*d/180 for d in azi_deg],angle_bins[0],device=device)
    elevation = torch.linspace(*[torch.pi*d/180 for d in ele_deg],angle_bins[1],device=device)
    directions = build_azimuth_elevation_directions(azimuth,elevation)
    weights = build_steering_matrix(torch.tensor(positions,device=device),directions,wavelength,normalize=True)
    ranges = torch.arange(range_count,device=device)*range_step
    grid,*_ = _cartesian_sampling_grid(ranges.to(dtype),azimuth.to(dtype),elevation.to(dtype),xyz_limits,cube_size)
    return weights, grid


@lru_cache(maxsize=4)
def _sky_window(device, dtype):
    signal = SignalProcessConfig()
    signal.init_winVel(AlgCfg(StructParam()))
    return torch.as_tensor(signal.winVel.T,device=device,dtype=dtype).repeat_interleave(4,dim=1)


@torch.no_grad()
def sky_difference_snapshots(cubes, frame_indices):
    """[B,T+6,R,64,16] -> [B,T,R,6,16], lags 1 through 6.

    Subtract each of the six immediate predecessors from the current frame.
    """
    history = MAX_FRAME_NUM - 1
    if (cubes.ndim != 5 or not cubes.is_complex() or cubes.shape[-2:] != (64,16)
            or cubes.shape[1] <= history):
        raise ValueError('KXYZ requires complex [B,T+6,R,64,16] input')
    indices = torch.as_tensor(frame_indices, device=cubes.device)
    if (indices.shape != cubes.shape[:2] or indices.is_floating_point() or indices.is_complex()
            or indices.dtype == torch.bool or (indices < 0).any()
            or ((indices[:,1:]-indices[:,:-1]) != 1).any()):
        raise ValueError('Frame indices must be consecutive nonnegative integers [B,T+6]')
    window = _sky_window(cubes.device,cubes.real.dtype)
    # Fixed firmware window gain is removed independently for each virtual channel.
    means = (cubes*window).mean(dim=3)/window.mean(dim=0)
    end = torch.arange(history,cubes.shape[1],device=cubes.device)[None].expand(cubes.shape[0],-1)
    offsets = torch.arange(1, MAX_FRAME_NUM, device=cubes.device)
    selected = end[...,None]-offsets
    batch = torch.arange(cubes.shape[0],device=cubes.device)[:,None,None]
    snapshots = means[batch, selected]  # [B,T,6,R,M]
    return (means[:,history:,None]-snapshots).permute(0,1,3,2,4).contiguous()


@torch.no_grad()
def range_cube_to_training_spectra(cubes, radar, xyz_limits, cube_size, angle_bins,
                                   remove_static=False, frame_indices=None):
    """Normalized log1p power, channels [all D bins, optional six K bins]."""
    if (cubes.ndim != 5 or not cubes.is_complex() or cubes.shape[-2:] != (64,16)
            or any(size == 0 for size in cubes.shape)):
        raise ValueError('Training spectra require complex [B,L,R,64,16] cubes')
    if not torch.isfinite(cubes).all():
        raise ValueError('BIN cubes contain NaN/Inf')
    differences = None
    current = cubes
    if frame_indices is not None:
        differences = sky_difference_snapshots(cubes,frame_indices)
        current = cubes[:,-differences.shape[1]:]
    if remove_static:
        current = current-current.mean(dim=3,keepdim=True)
    doppler,_ = doppler_fft_torch(current,radar,doppler_mode='firmware_tdm',window_periodic=False)
    window = torch.hann_window(256,periodic=False,device=cubes.device).reshape(64,4)
    doppler = doppler/window.sum(0).repeat_interleave(4)
    snapshots = torch.cat((doppler,differences),dim=3) if differences is not None else doppler
    weights, grid = _geometry(
        cubes.device, cubes.real.dtype,
        tuple(map(tuple,radar.virtual_channel_positions.tolist())),radar.lam,
        get_radar_res(radar)[0],cubes.shape[2],tuple(radar.azi_deg),tuple(radar.ele_deg),
        tuple(angle_bins),tuple(map(tuple,xyz_limits)),tuple(cube_size))
    B,T,R,C,M = snapshots.shape
    frames = snapshots.permute(0,1,3,2,4).reshape(B*T*C,R,M)
    output = torch.empty((B*T*C,*cube_size),device=cubes.device,dtype=cubes.real.dtype)
    # Flatten frame/channel pairs so each grid_sample thread handles one voxel.
    # Bound polar power to 128 MiB independently of the training batch size.
    chunk = max(1,(128*1024**2)//(R*weights.shape[1]*output.element_size()))
    for start in range(0,len(frames),chunk):
        stop = min(start+chunk,len(frames))
        power = conventional_dbf_power(frames[start:stop],weights,direction_chunk_size=512)
        polar = power.reshape(stop-start,1,R,*angle_bins)
        xyz = F.grid_sample(polar,grid.expand(stop-start,-1,-1,-1,-1),
                            mode='bilinear',padding_mode='zeros',align_corners=True)
        output[start:stop] = xyz[:,0].clamp_min_(0).log1p_()
    return output.reshape(B,T,C,*cube_size)
