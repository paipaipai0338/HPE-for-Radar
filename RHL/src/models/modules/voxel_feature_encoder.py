import torch
import torch.nn as nn
from typing import Dict
from src.models import VFES


class ResBlock3D(nn.Module):
    """
    3D 残差块：用于在同一分辨率下加强梯度传播和特征复用。
    引入可选的 Dropout 机制，防止在稀疏雷达数据上过拟合。
    """
    def __init__(self, channels: int, num_groups: int = 8, dropout_p: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(num_groups, channels), num_channels=channels),
            nn.GELU(),
            nn.Dropout3d(dropout_p) if dropout_p > 0 else nn.Identity(), # 空间 Dropout 防御噪声过拟合
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(num_groups, channels), num_channels=channels)
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class DownBlock3D(nn.Module):
    """
    3D 下采样残差块：通过 stride=2 压缩空间分辨率并提升通道数，配有 1x1x1 卷积捷径投影。
    """
    def __init__(self, in_c: int, out_c: int, num_groups: int = 8, dropout_p: float = 0.1):
        super().__init__()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(num_groups, out_c), num_channels=out_c),
            nn.GELU(),
            nn.Dropout3d(dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(num_groups, out_c), num_channels=out_c)
        )
        # 捷径分支：用 stride=2 的 1x1x1 卷积对齐尺寸和通道
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=1, stride=2, padding=0),
            nn.GroupNorm(num_groups=min(num_groups, out_c), num_channels=out_c)
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv_down(x) + self.shortcut(x))


@VFES.register("Res_3DUNet_Encoder")
class VoxelFeatureEncoder(nn.Module):
    """
    当前帧体素特征编码模块 (VoxelFeatureEncoder - 正则化完备版)
    """
    def __init__(
        self, 
        config,
    ):
        super().__init__()
        in_channels = config.in_channels
        num_groups = config.num_groups
        dropout_p = config.dropout_p
        c1, c2, c3 = config.backbone_channels
        # 1. Stem（词干初始卷积层）：66 -> 32
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c1, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(num_groups, 32), num_channels=32),
            nn.GELU()
        )

        # 2. Block-1: 32 -> 32 (输出 F1)
        self.block1 = ResBlock3D(c1, num_groups, dropout_p)
        
        # 3. Down-1: 32 -> 64 (下采样)
        self.down1 = DownBlock3D(c1, c2, num_groups, dropout_p)
        
        # 4. Block-2: 64 -> 64 (输出 F2)
        self.block2 = ResBlock3D(c2, num_groups, dropout_p)
        
        # 5. Down-2: 64 -> 128 (下采样)
        self.down2 = DownBlock3D(c2, c3, num_groups, dropout_p)
        
        # 6. Block-3: 128 -> 128 (输出 F3)
        self.block3 = ResBlock3D(c3, num_groups, dropout_p)

    def forward(self, dense_voxel_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(dense_voxel_features)  # [B, 32, X, Y, Z]
        
        F1 = self.block1(x)                  # [B, 32, 24, 24, 8]
        
        x = self.down1(F1)                   # [B, 64, 12, 12, 4]
        F2 = self.block2(x)                  # [B, 64, 12, 12, 4]
        
        x = self.down2(F2)                   # [B, 128, 6, 6, 2]
        F3 = self.block3(x)                  # [B, 128, 6, 6, 2]

        return {
            "F1": F1,
            "F2": F2,
            "F3": F3
        }