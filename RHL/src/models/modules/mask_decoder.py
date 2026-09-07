from transformers import Sam3Model,DetrModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Optional
from src.models.module_outputs.mask_decoder_outputs import MaskDecoderOutput
from src.models import MDS


class VoxelDecoder(nn.Module):
    """
    3D 体素上采样重建解码器 (VoxelDecoder)。

    核心功能：
    1. lateral_convs: 通过 1x1x1 3D 卷积将 Backbone 输出的多尺度特征通道统一映射至 hidden_size；
    2. fusion_blocks: 自顶向下通过三线性插值 (Trilinear) 逐步上采样，与侧边跳跃连接特征相加，
       并经过 3x3x3 卷积 (bias=True) + GroupNorm + ReLU 进行空间平滑与空洞填补；
    3. 统一通过 config 实例化构建。

    输入特征尺度（由细到粗）：
        F1: [B, C1, 48, 48, 32]  (如 C1=32)
        F2: [B, C2, 24, 24, 16]  (如 C2=64)
        F3: [B, C3, 12, 12, 8]   (如 C3=128)

    输出特征尺度：
        out: [B, hidden_size, 48, 48, 32]
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.in_channels_list = config.backbone_channels
        self.num_stages = len(self.in_channels_list)
        # GroupNorm 分组数，默认 8 组（需保证 hidden_size 能被 num_groups 整除）
        self.num_groups = config.num_groups

        # 1. 侧边通道对齐卷积 (1x1x1 3D Conv，配合 GroupNorm 开启 bias=True)
        self.lateral_convs = nn.ModuleList([
            nn.Conv3d(in_c, self.hidden_size, kernel_size=1, bias=True)
            for in_c in self.in_channels_list
        ])

        # 2. 上采样融合平滑块 (3x3x3 3D Conv + GroupNorm + ReLU)
        self.fusion_blocks = nn.ModuleList()
        for _ in range(self.num_stages - 1):
            self.fusion_blocks.append(
                nn.Sequential(
                    nn.Conv3d(
                        self.hidden_size,
                        self.hidden_size,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    ),
                    nn.GroupNorm(num_groups=self.num_groups, num_channels=self.hidden_size),
                    nn.ReLU(inplace=True),
                )
            )

        self._init_weights()

    def _init_weights(self):
        """对卷积和 GroupNorm 层进行标准权重初始化。"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, backbone_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            backbone_features: 骨干网络提取的多尺度 3D 特征列表，由细到粗排列：
                               [F1, F2, F3]
                               F1: [B, C1, 48, 48, 32]
                               F2: [B, C2, 24, 24, 16]
                               F3: [B, C3, 12, 12, 8]

        Returns:
            voxel_embed: 融合重建后的全分辨率体素特征，形状为 [B, hidden_size, 48, 48, 32]
        """
        if len(backbone_features) != self.num_stages:
            raise ValueError(
                f"Expected {self.num_stages} backbone feature maps, but got {len(backbone_features)}."
            )

        # 1. 1x1x1 侧边卷积投影
        mapped_feats = [
            conv(feat) for conv, feat in zip(self.lateral_convs, backbone_features)
        ]

        # 2. 自顶向下 (Top-Down) 逐级三线性插值上采样与侧边残差相加
        prev_feat = mapped_feats[-1]  # [B, hidden_size, 12, 12, 8]

        for layer_idx, skip_feat in enumerate(reversed(mapped_feats[:-1])):
            target_spatial_shape = skip_feat.shape[-3:]  # (X, Y, Z)

            # (1) 三线性插值平滑上采样
            upsampled_feat = F.interpolate(
                prev_feat,
                size=target_spatial_shape,
                mode="trilinear",
                align_corners=False,
            )

            # (2) 侧边残差相加融合
            fused_feat = upsampled_feat + skip_feat

            # (3) 3x3x3 卷积 + GroupNorm + ReLU 平滑
            prev_feat = self.fusion_blocks[layer_idx](fused_feat)

        # 输出全分辨率体素特征 [B, hidden_size, 48, 48, 32]
        return prev_feat


class MaskEmbedder(nn.Module):
    """
    实例 Query 掩码特征变换头 (MaskEmbedder)。

    核心功能：
    将 StateDecoder 输出的实例级 Query 隐状态 (hidden_states)
    映射为与 3D 体素空间特征对齐的动态分类核向量 E_mask。

    输入维度：[B, Q, hidden_size]
    输出维度：[B, Q, hidden_size]
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        # MLP 层数，SAM3 / Mask2Former 标准实现通常为 3 层
        self.num_layers = config.mask_embedder_num_layers

        layers = []
        for i in range(self.num_layers):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size, bias=True))
            # 最后一层输出不加激活函数，中间层加 ReLU
            if i < self.num_layers - 1:
                layers.append(nn.ReLU(inplace=True))

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """对线性层进行标准权重初始化。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, decoder_queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            decoder_queries: StateDecoder 最后一层解算出的实例 Query 隐状态，
                             形状为 [B, Q, hidden_size]

        Returns:
            mask_embeddings: 实例掩码动态嵌入特征 (E_mask)，
                             形状为 [B, Q, mask_dim]
        """
        return self.mlp(decoder_queries)

@MDS.register("Voxel_FPN_Mask_Decoder")
class MaskDecoder(nn.Module):
    """
    3D 毫米波雷达体素实例掩码解码器 (MaskDecoder)。

    设计完全对齐 SAM 3 的 Dual-Path / Two-Stage 分割范式：
    1. _embed_voxels: 对齐 SAM 3 的 _embed_pixels，用时域融合后的 F_temporal 替换最深层视觉特征并解码；
    2. mask_embedder: 将 StateDecoder 输出的 Query 转换为动态掩码权重核 E_mask；
    3. instance_projection & einsum: 动态点积解算 3D 实例体素掩码；
    4. semantic_projection: 全局体素前景 Occupancy 辅助预测。
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # 1. 3D 体素上采样解码器 (FPN-style)
        self.voxel_decoder = VoxelDecoder(config)

        # 2. 实例 Query 掩码特征变换头 (3层 MLP)
        self.mask_embedder = MaskEmbedder(config)

        # 3. 体素空间特征投影层 (1x1x1 Conv3d)
        self.instance_projection = nn.Conv3d(
            self.hidden_size, self.hidden_size, kernel_size=1, bias=True
        )

        # 4. 全局体素 Occupancy 辅助预测头
        self.semantic_projection = nn.Conv3d(
            self.hidden_size, 1, kernel_size=1, bias=True
        )

        self._init_weights()

    def _init_weights(self):
        """对顶层投影卷积进行标准权重初始化。"""
        for m in [self.instance_projection, self.semantic_projection]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _embed_voxels(
        self,
        backbone_features: List[torch.Tensor],
        temporal_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        对标 SAM 3 的 _embed_pixels 方法：
        将多尺度骨干网络特征的最深层粗粒度特征替换为时域融合特征 F_temporal，
        然后输入 VoxelDecoder 进行多尺度三线性插值上采样与融合。

        Args:
            backbone_features: 骨干网络提取的多尺度 3D 特征列表 [F1, F2, F3]
            temporal_feature: 经过 3D ConvGRU 时域融合后的深层特征 F_temporal [B, 128, 12, 12, 8]
                             若为 None，则直接使用骨干网络的原始深层特征 F3。

        Returns:
            voxel_embed: 重建后的全分辨率体素特征 [B, hidden_size, 48, 48, 32]
        """
        # 浅拷贝列表以保护外部输入
        backbone_visual_feats = [feat for feat in backbone_features]

        # 若提供了时域增强特征，直接替换最深层 (最低分辨率) 特征
        if temporal_feature is not None:
            backbone_visual_feats[-1] = temporal_feature

        # 送入体素解码器完成自顶向下的上采样融合
        voxel_embed = self.voxel_decoder(backbone_visual_feats)

        return voxel_embed

    def forward(
        self,
        decoder_queries: torch.Tensor,
        backbone_features: List[torch.Tensor],
        temporal_feature: Optional[torch.Tensor] = None,
    ) -> MaskDecoderOutput:
        """
        Args:
            decoder_queries: StateDecoder 最后一层解算出的实例 Query 特征 [B, Q, hidden_size]
            backbone_features: 骨干网络提取的多尺度 3D 特征列表 [F1, F2, F3]
            temporal_feature: TemporalFeatureFusion 产出的时域融合特征 F_temporal [B, 128, 12, 12, 8]

        Returns:
            MaskDecoderOutput: 包含 pred_masks [B, Q, 48, 48, 32] 与 semantic_seg [B, 1, 48, 48, 32]
        """
        # 1. 通过 _embed_voxels 构建融合时序特征后的全分辨率体素嵌入
        voxel_embed = self._embed_voxels(
            backbone_features=backbone_features,
            temporal_feature=temporal_feature,
        )

        # 2. 投影体素空间特征与 Query 掩码特征
        # instance_voxel_feats: [B, hidden_size, 48, 48, 32]
        instance_voxel_feats = self.instance_projection(voxel_embed)
        # mask_embeddings (E_mask): [B, Q, hidden_size]
        mask_embeddings = self.mask_embedder(decoder_queries)

        # 3. 动态点积解算实例掩码: [B, Q, C] x [B, C, X, Y, Z] -> [B, Q, X, Y, Z]
        pred_masks = torch.einsum("bqc,bcxyz->bqxyz", mask_embeddings, instance_voxel_feats)

        # 4. 全局前景 Occupancy 辅助预测: [B, 1, X, Y, Z]
        semantic_seg = self.semantic_projection(voxel_embed)

        return MaskDecoderOutput(
            pred_masks=pred_masks,
            semantic_seg=semantic_seg,
        )

# 