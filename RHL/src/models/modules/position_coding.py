import torch
import torch.nn as nn
import math


class SinePositionEmbedding(nn.Module):
    """
    三维正弦余弦位置编码模块（SinePositionEmbedding）。
    
    支持两种核心功能：
    1. 体素网格空间位置编码（forward）：为 3D 体素特征网格 [B, C, X, Y, Z] 生成对应的 3D 空间位置编码。
    2. 连续 3D 边界框/参考点位置编码（encode_boxes）：为归一化的 6D 边界框 [cx, cy, cz, sx, sy, sz]
       或 3D 参考点 [cx, cy, cz] 生成正弦频率特征。
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: float = 10000.0,
        normalize: bool = True,
        scale: float | None = None,
    ):
        """
        Args:
            num_pos_feats: 每个坐标轴单侧（sin/cos 各占一半）的特征通道数。
                           若对 3D 网格编码，总输出通道为 3 * (2 * num_pos_feats)。
                           若对 6D 边界框编码，总输出通道为 6 * (2 * num_pos_feats)。
            temperature: 正弦函数频率衰减温度系数，默认 10000.0。
            normalize: 是否对坐标进行归一化缩放（乘以 scale）。
            scale: 归一化缩放系数，默认 2 * pi。
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("scale should not be set when normalize is False")
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """为 3D 体素特征生成对应的三维正弦位置编码。

        Args:
            tensor: 3D 体素特征张量，形状为 [B, C, X, Y, Z]
            mask: 可选的体素有效掩码，形状为 [B, X, Y, Z]，True 表示无效/填充，False 表示有效。
                  若为 None，则假设所有体素均有效。

        Returns:
            pos: 三维位置编码张量，形状为 [B, 3 * num_pos_feats, X, Y, Z]
        """
        B, C, X, Y, Z = tensor.shape

        if mask is None:
            mask = torch.zeros((B, X, Y, Z), device=tensor.device, dtype=torch.bool)

        not_mask = ~mask

        # 1. 沿各自真实物理轴向计算累计坐标: mask 形状为 [B, X, Y, Z]
        x_embed = not_mask.cumsum(1, dtype=torch.float32)  # 沿着 dim 1 (X 轴)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)  # 沿着 dim 2 (Y 轴)
        z_embed = not_mask.cumsum(3, dtype=torch.float32)  # 沿着 dim 3 (Z 轴)

        # 2. 归一化到 [0, scale]
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            z_embed = z_embed / (z_embed[:, :, :, -1:] + eps) * self.scale

        # 3. 生成频率基底 (长度为 num_pos_feats)
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=tensor.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        # 4. 频率投影: [B, X, Y, Z, num_pos_feats]
        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t
        pos_z = z_embed.unsqueeze(-1) / dim_t

        # 5. 正弦与余弦交织拼接，保持单轴通道数为 num_pos_feats
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)

        # 6. 沿特征维度拼接并置换为 [B, 3 * num_pos_feats, X, Y, Z]
        pos = torch.cat((pos_x, pos_y, pos_z), dim=-1).permute(0, 4, 1, 2, 3)
        return pos.to(dtype=tensor.dtype)

    def encode_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        对连续归一化三维边界框或空间坐标进行多频正弦编码。

        Args:
            boxes: 归一化输入坐标/边界框，形状为 [..., N_coords]
                   例如 [B, Q, 6] (cx, cy, cz, sx, sy, sz) 或 [B, Q, 3] (cx, cy, cz)

        Returns:
            box_embed: 编码后的高维正弦特征，形状为 [..., N_coords * 2 * num_pos_feats]
        """
        if self.normalize:
            boxes = boxes * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=boxes.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        # [..., N_coords, num_pos_feats]
        encoded = boxes.unsqueeze(-1) / dim_t
        encoded = torch.stack(
            (encoded[..., 0::2].sin(), encoded[..., 1::2].cos()), dim=-1
        ).flatten(-2)

        # 展平坐标维度与特征维度: [..., N_coords * 2 * num_pos_feats]
        box_embed = encoded.flatten(-2)
        return box_embed.to(dtype=boxes.dtype)