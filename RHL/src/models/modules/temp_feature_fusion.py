import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from src.models import TFS  # 指定的 TFS 注册器


class ConvGRU3DCell(nn.Module):
    """
    3D ConvGRU 的单步核心计算单元
    职责：维护并更新 3D 时空隐状态 M_t，结合当前帧 F3_t 进行门控融合。
    """
    def __init__(self, channels: int = 128, num_groups: int = 8):
        super().__init__()
        # 门控卷积：同时输入当前特征 x (channels) 和上一时刻隐藏状态 h (channels)，共 2 * channels
        # 使用 3D 卷积保持空间和高度维度的上下文
        self.conv_z = nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1)
        self.conv_r = nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1)
        self.conv_h = nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1)
        
        # 使用 GroupNorm 保证小 batch 下的稳定性
        self.gn_z = nn.GroupNorm(num_groups=min(num_groups, channels), num_channels=channels)
        self.gn_r = nn.GroupNorm(num_groups=min(num_groups, channels), num_channels=channels)
        self.gn_h = nn.GroupNorm(num_groups=min(num_groups, channels), num_channels=channels)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        参数:
            x: 当前帧或某一帧的历史特征 [B, C, X, Y, Z]
            h_prev: 上一时刻的隐状态 [B, C, X, Y, Z]，若为 None 则初始化为全 0
        返回:
            h_next: 更新后的隐状态 [B, C, X, Y, Z]
        """
        B, C, X, Y, Z = x.shape
        if h_prev is None:
            h_prev = torch.zeros_like(x)

        # 拼接当前输入与历史隐状态
        combined = torch.cat([x, h_prev], dim=1)  # [B, 2*C, X, Y, Z]

        # 1. 更新门 (Update Gate)
        z = torch.sigmoid(self.gn_z(self.conv_z(combined)))
        
        # 2. 重置门 (Reset Gate)
        r = torch.sigmoid(self.gn_r(self.conv_r(combined)))
        
        # 3. 候选隐状态 (Candidate Hidden State)
        combined_r = torch.cat([x, r * h_prev], dim=1)
        h_candidate = torch.tanh(self.gn_h(self.conv_h(combined_r)))
        
        # 4. 最终隐状态更新
        h_next = (1.0 - z) * h_prev + z * h_candidate

        return h_next


@TFS.register("ConvGRU3D_Fusion")
class TemporalFeatureFusion(nn.Module):
    """
    体素特征时域信息融合模块 (TemporalFeatureFusion)
    职责：
    1. 接收当前帧深层特征 F3_t 以及历史多帧特征 F3_history。
    2. 通过 3D ConvGRU 顺序迭代更新时空记忆。
    3. 输出具备时域连续性的融合特征 F_temporal: [B, 128, X/4, Y/4, Z/4]。
    """
    def __init__(
        self, 
        config,
    ):
        super().__init__()
        self.channels = config.backbone_channels[-1]
        self.cell = ConvGRU3DCell(channels=self.channels, num_groups=config.num_groups)

    def forward(
        self,
        current_f3: torch.Tensor,
        f3_history: Optional[torch.Tensor] = None,
        h_state: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播接口
        参数:
            current_f3: 当前帧深层特征 F3_t [B, 128, X/4, Y/4, Z/4]
            f3_history: 历史帧深层特征序列 F3_history [B, T-1, 128, X/4, Y/4, Z/4] (若无历史则为 None)
            h_state:    传入的初始隐状态 (若需要外部维护状态时使用)
            **kwargs:   预留扩展参数
        返回:
            F_temporal: 融合后的时域特征 [B, 128, X/4, Y/4, Z/4]
            h_state:    更新后的最新隐状态，可用于下一次迭代
        """
        B, C, X, Y, Z = current_f3.shape

        # 初始化或继承隐状态
        h = h_state if h_state is not None else torch.zeros(
            (B, C, X, Y, Z), dtype=current_f3.dtype, device=current_f3.device
        )

        # 如果存在历史特征序列 (形状: [B, T-1, C, X, Y, Z])
        if f3_history is not None and f3_history.size(1) > 0:
            num_history_frames = f3_history.size(1)
            # 沿着时间维度 (dim=1) 依次喂入 ConvGRUCell 进行时序记忆迭代
            for t in range(num_history_frames):
                hist_frame_feat = f3_history[:, t, :, :, :,]  # [B, C, X, Y, Z]
                h = self.cell(hist_frame_feat, h)

        # 最后将当前帧特征喂入 ConvGRUCell，完成最终的时序记忆融合
        F_temporal = self.cell(current_f3, h)
        
        # 更新隐状态供后续追踪或连续序列使用
        h_state = F_temporal.detach()  # 阻断时序太长导致的梯度截断反向传播过深 (Truncated BPTT 策略)

        return F_temporal, h_state