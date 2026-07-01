from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch
import numpy as np
from typing import *

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

# 读取1DFFT函数
def bin_to_cube_range_fft(file_path: Path|str, radar_config: Radar_Config) -> Optional[np.ndarray]:
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
    raw = np.fromfile(file_path, dtype=np.uint8)
    if raw.size != expected_bytes:
        print(f"[WARN] {os.path.basename(file_path)} size mismatch: "
              f"{raw.size} != {expected_bytes}, skip.")
        return None
    raw8 = raw.reshape(-1, 8)[:, ::-1].reshape(-1)
    pf_u32 = np.frombuffer(raw8.tobytes(), dtype="<u4")
    vec_cplx = _pseudo_float_cplx_to_complex(pf_u32)
    mcu_timing = vec_cplx.reshape((use_range, num_ant, num_chirp), order="F")
    adc_data_range_FFT = np.transpose(mcu_timing, (0, 2, 1))
    return adc_data_range_FFT

# 读取点云数据
def get_pc_data(file_path: Path|str):
    data = np.load(file_path)
    return data

# 读取谱图数据，利用bin_to_cube_range_fft封装
def get_bin_data(file_path: Path|str):
    radar_config = Radar_Config()
    data = bin_to_cube_range_fft(file_path, radar_config)
    return data

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

# 对pytorch传出带有 B T的tensor进行fft处理
def doppler_fft_batch_np_wrapper(
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

# doppler_fft_batch_np_wrapper的封装版本
def range_cube_to_doppler_cube(data: torch.Tensor, radar_config: Radar_Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    range_res, _, _, _ = get_radar_res(radar_config)
    r_axis = np.arange(data.shape[2], dtype=float) * range_res
    r_axis = torch.tensor(r_axis)
    doppler_cube, doppler_cube_mean, v_axis = doppler_fft_batch_np_wrapper(data=data, radar_config=radar_config)
    return doppler_cube, doppler_cube_mean, r_axis, v_axis

