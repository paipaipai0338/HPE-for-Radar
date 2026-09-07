import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any
from src.models.module_outputs.state_encoder_outputs import DeformableVoxelEncoderOutput
from src.models.modules.position_coding import *
from src.models import SES


class StateEncoderMLP(nn.Module):
    """
    通用多层感知机模块（StateDecoderMLP），用于 State Decoder 内部的各类头结构与投影映射。
    
    常见用途：
    - Box Head (6D 边界框增量回归)
    - Presence Head (全局存在性预测)
    - Query / Reference Point 条件位置编码投影
    - BoxRPB 三轴 Delta 编码映射
    - Mask Embedding 投影
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: str = "relu",
    ):
        """
        Args:
            input_dim: 输入特征通道数
            hidden_dim: 隐藏层特征通道数
            output_dim: 输出特征通道数
            num_layers: 线性层总层数 (num_layers >= 1)
            activation: 隐藏层激活函数类型，支持 'relu' 与 'gelu'
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, but got {num_layers}")

        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

        if activation.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation.lower() == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}. Choose 'relu' or 'gelu'.")

        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 [..., input_dim]
        Returns:
            输出张量，形状为 [..., output_dim]
        """
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MultiScaleChannelProjector(nn.Module):
    """
    多尺度体素特征通道投影与对齐模块。

    输入：多尺度体素特征列表 [F1, F2, F3]，每个张量形状为 [B, C_in_i, X_i, Y_i, Z_i]
    输出：通道数对齐后的特征列表 [P1, P2, P3]，每个张量形状为 [B, hidden_size, X_i, Y_i, Z_i]
    """

    def __init__(
        self,
        config: Any, 
    ):
        """
        Args:
            in_channels_list: 各尺度输入特征的通道数列表。
            hidden_size: 统一对齐的目标通道维度。
            num_groups: GroupNorm 的分组数（默认 8）。
            act_layer: 激活函数类型。
        """
        super().__init__()
        self.config = config
        self.in_channels_list = config.in_channels_list
        self.hidden_size = config.hidden_size
        activation = config.activation

        if activation.lower() == "gelu":
            self.act = nn.GELU()
        elif activation.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}. Choose 'gelu' or 'relu'.")

        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_c, config.hidden_size, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=min(config.num_groups, config.hidden_size), num_channels=config.hidden_size),
                self.act
            )
            for in_c in self.in_channels_list
        ])

    def forward(self, multi_level_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            multi_level_feats: 长度为 N 的多尺度体素特征列表，
                               每个元素形状为 [B, C_in_i, X_i, Y_i, Z_i]。

        Returns:
            projected_feats: 长度为 N 的对齐后特征列表，
                             每个元素形状为 [B, hidden_size, X_i, Y_i, Z_i]。
        """
        assert len(multi_level_feats) == len(self.projectors), (
            f"Expected {len(self.projectors)} feature levels, but got {len(multi_level_feats)}"
        )

        return [proj(feat) for proj, feat in zip(self.projectors, multi_level_feats)]

class MSDeformableAttention3D(nn.Module):
    """
    多尺度 3D 可变形注意力模块 (Multi-Scale 3D Deformable Attention).

    使用纯 PyTorch 原生算子 (F.grid_sample) 实现 3D 空间连续采样，
    无需编译自定义 CUDA 扩展。
    """

    def __init__(
        self,
        config: Any,
    ):
        """
        Args:
            config: 配置对象，需包含以下字段：
                - hidden_size (int): 特征维度，必须能被 num_attention_heads 整除。
                - num_attention_heads (int): 注意力头数。
                - num_levels (int): 特征尺度层级数（如 F1, F2, F3 共 3 层）。
                - num_points (int): 每个 Head 在每个尺度上采样的 3D 关键点数量。
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_levels = config.num_levels
        self.num_points = config.num_points

        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size {self.hidden_size} must be divisible by num_attention_heads {self.num_heads}"
        )
        self.head_dim = self.hidden_size // self.num_heads

        # 采样 3D 偏移量预测: [dx, dy, dz]，总共预测 num_heads * num_levels * num_points * 3 个值
        self.sampling_offsets = nn.Linear(
            self.hidden_size, self.num_heads * self.num_levels * self.num_points * 3
        )
        # 注意力权重预测: 每个采样点对应一个标量权重
        self.attention_weights = nn.Linear(
            self.hidden_size, self.num_heads * self.num_levels * self.num_points
        )
        # 线性变换: 产生 Value 与最终输出投影
        self.value_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self._reset_parameters()

    def _reset_parameters(self):
        # 权重偏置初始化：让初始采样点均匀分布在参考点周围
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        # 小随机初始化偏移偏置
        nn.init.uniform_(self.sampling_offsets.bias.data, -2.0, 2.0)
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: List[Tuple[int, int, int]],
        spatial_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            query: 展平后的 Query 序列 [B, N_query, d_model]
                   (对于自注意力，query = src + pos_embed)
            reference_points: 每个 Query 的 3D 归一化参考坐标 [B, N_query, 3]，取值在 [0, 1] 之间
            spatial_shapes: 多尺度 3D 网格尺寸 [(X1, Y1, Z1), (X2, Y2, Z2), (X3, Y3, Z3)]
            spatial_feats: 通道对齐后的多尺度 3D 特征列表 [[B, d_model, X_l, Y_l, Z_l], ...]

        Returns:
            output: 可变形注意力聚合后的特征 [B, N_query, d_model]
        """
        B, N_q, _ = query.shape

        # 1. 预测 3D 采样偏移量与注意力权重
        # [B, N_q, num_heads, num_levels, num_points, 3]
        offsets = self.sampling_offsets(query).view(
            B, N_q, self.num_heads, self.num_levels, self.num_points, 3
        )
        # [B, N_q, num_heads, num_levels * num_points]
        attn_weights = self.attention_weights(query).view(
            B, N_q, self.num_heads, self.num_levels * self.num_points
        )
        attn_weights = F.softmax(attn_weights, dim=-1).view(
            B, N_q, self.num_heads, self.num_levels, self.num_points
        )

        # 2. 对每个尺度提取 Value 并通过 3D 插值采样
        # 存储各层采样后的加权特征
        sampled_value_list = []

        for lvl, feat in enumerate(spatial_feats):
            X_l, Y_l, Z_l = spatial_shapes[lvl]
            
            # 投影 Value: [B, d_model, X, Y, Z] -> [B, X, Y, Z, num_heads, head_dim]
            v = self.value_proj(feat.permute(0, 2, 3, 4, 1)).view(
                B, X_l, Y_l, Z_l, self.num_heads, self.head_dim
            )
            # 变形以便与 batch 维度合并处理: [B * num_heads, head_dim, X, Y, Z]
            v = v.permute(0, 4, 5, 1, 2, 3).flatten(0, 1)

            # 计算归一化采样网格坐标 (带偏移)
            # reference_points: [B, N_q, 1, 1, 3] -> 归一化 [0, 1] 坐标
            ref_lvl = reference_points.unsqueeze(2).unsqueeze(3)
            
            # offsets 乘以体素分辨率尺度反比，使得预测的 offset 为体素格数单位
            scale = torch.tensor([X_l, Y_l, Z_l], dtype=query.dtype, device=query.device)
            norm_offsets = offsets[:, :, :, lvl, :, :] / scale.view(1, 1, 1, 1, 3)

            # 实际采样物理点坐标: [B, N_q, num_heads, num_points, 3] (取值范围约 [0, 1])
            sampling_loc = ref_lvl + norm_offsets

            # 将 [0, 1] 坐标转换至 F.grid_sample 需要的 [-1, 1] 坐标区间
            grid_loc = sampling_loc * 2.0 - 1.0

            # 整理为 grid_sample 接受的 5D 形状: [B * num_heads, N_q, num_points, 1, 3]
            # grid_sample 3D 格式对应为 (grid_x, grid_y, grid_z) 即 (dim_Z, dim_Y, dim_X)
            # 此处我们将 normalized [cx, cy, cz] 对齐到 grid 的 (z, y, x)
            grid = grid_loc.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(-2)

            # 3D 连续采样: [B * num_heads, head_dim, N_q, num_points, 1]
            sampled_feat = F.grid_sample(
                v,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).squeeze(-1)

            # 重塑形状: [B, num_heads, head_dim, N_q, num_points] -> [B, N_q, num_heads, num_points, head_dim]
            sampled_feat = (
                sampled_feat.view(B, self.num_heads, self.head_dim, N_q, self.num_points)
                .permute(0, 3, 1, 4, 2)
            )

            # 加权求和当前层采样点: weights 为 [B, N_q, num_heads, 1, num_points]
            weights_lvl = attn_weights[:, :, :, lvl, :].unsqueeze(-2)
            # [B, N_q, num_heads, head_dim]
            weighted_feat_lvl = (sampled_feat.permute(0, 1, 2, 4, 3) @ weights_lvl.transpose(-1, -2)).squeeze(-1)

            sampled_value_list.append(weighted_feat_lvl)

        # 3. 跨尺度特征求和并投影输出
        # [B, N_q, num_heads, head_dim]
        total_value = sum(sampled_value_list)
        # 拼回总特征维度: [B, N_q, d_model]
        output = self.output_proj(total_value.flatten(2))

        return output

class DeformableVoxelEncoderFFN(nn.Module):
    """
    状态编码器前馈神经网络模块 (StateEncoderFFN).

    用于在注意力计算后进行特征升维与逐点非线性特征变换。
    """

    def __init__(
        self,
        config: Any,
    ):
        """
        Args:
            config: 配置对象，需包含以下字段：
                - hidden_size (int): 输入与输出的特征通道维度。
                - intermediate_size (int): 前馈网络隐藏层特征维度 (通常为 hidden_size 的 2~4 倍)。
                - dropout (float): Dropout 概率。
                - activation (str): 激活函数类型 ('gelu' 或 'relu')。
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        dropout = getattr(config, "dropout", 0.1)
        activation = getattr(config, "activation", "gelu")

        if activation.lower() == "gelu":
            self.act = nn.GELU()
        elif activation.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}. Choose 'gelu' or 'relu'.")

        self.linear1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(self.intermediate_size, self.hidden_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征张量，形状为 [B, N_tokens, hidden_size]

        Returns:
            out: 变换后的特征张量，形状为 [B, N_tokens, hidden_size]
        """
        return self.dropout2(self.linear2(self.dropout1(self.act(self.linear1(x)))))

class DeformableVoxelEncoderLayer(nn.Module):
    """
    状态编码器单层模块 (StateEncoderLayer).

    采用 Pre-LN 结构，包含：
    1. 多尺度 3D 可变形自注意力子层 (MSDeformableAttention3D) + 残差连接
    2. 前馈神经网络子层 (StateEncoderFFN) + 残差连接
    """

    def __init__(
        self,
        config: Any,
    ):
        """
        Args:
            config: 配置对象，需包含以下字段：
                - hidden_size (int): 特征维度。
                - num_attention_heads (int): 注意力头数。
                - num_levels (int): 特征尺度层级数。
                - num_points (int): 每个 Head 在每个尺度采样的关键点数。
                - intermediate_size (int): FFN 隐藏层维度。
                - dropout (float, optional): Dropout 概率，默认 0.1。
                - activation (str, optional): 激活函数类型，默认 'gelu'。
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        dropout = getattr(config, "dropout", 0.1)

        # 1. 多尺度 3D 可变形自注意力子层
        self.self_attn = MSDeformableAttention3D(config)
        self.dropout = nn.Dropout(dropout)
        self.self_attn_norm = nn.LayerNorm(self.hidden_size)

        # 2. 前馈神经网络子层
        self.ffn = DeformableVoxelEncoderFFN(config)
        self.ffn_norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        src: torch.Tensor,
        pos_embed: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: List[Tuple[int, int, int]],
        spatial_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            src: 展平后的状态特征序列，形状为 [B, N_total, hidden_size]
            pos_embed: 融合了 3D 空间与层级编码的位置向量，形状为 [B, N_total, hidden_size]
            reference_points: 每个 Token 的 3D 归一化中心参考坐标 [B, N_total, 3]，取值在 [0, 1] 区间
            spatial_shapes: 多尺度 3D 网格尺寸列表 [(X1, Y1, Z1), (X2, Y2, Z2), (X3, Y3, Z3)]
            spatial_feats: 当前层对应的多尺度 3D 特征张量列表 [[B, hidden_size, X_l, Y_l, Z_l], ...]

        Returns:
            src: 经过注意力与 FFN 增强后的特征序列，形状为 [B, N_total, hidden_size]
        """
        # --- 1. Multi-Scale 3D Deformable Self-Attention (Pre-LN) ---
        query = src + pos_embed

        attn_out = self.self_attn(
            query=query,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_feats=spatial_feats,
        )
        src = src + self.dropout(attn_out)
        src = self.self_attn_norm(src)

        # --- 2. Feed-Forward Network (Pre-LN) ---
        
        ffn_out = self.ffn(src)
        src = src + ffn_out
        src = self.ffn_norm(src)

        return src

@SES.register("Detr_State_Encoder")
class DeformableVoxelEncoder(nn.Module):
    """
    多尺度体素状态编码器 (StateEncoder).

    接收多尺度 3D 体素特征（如 F1, F2, F3/F_temporal），
    统一通道维度、注入 3D 空间与层级位置编码，
    并通过多层 3D 多尺度可变形自注意力完成全局上下文交互与多尺度特征增强。
    """

    def __init__(
        self,
        config: Any,
    ):
        """
        Args:
            config: 配置对象，需包含以下字段：
                - in_channels_list (List[int]): 各尺度输入特征的通道数列表。
                - hidden_size (int): 统一特征通道维度。
                - num_encoder_layers (int): 编码器层数 (通常为 2~3 层)。
                - num_attention_heads (int): 注意力头数。
                - num_levels (int): 特征尺度层级数 (len(in_channels_list))。
                - num_points (int): 每个 Head 在每个尺度采样的关键点数。
                - intermediate_size (int): FFN 隐藏层维度。
                - num_groups (int, optional): GroupNorm 分组数，默认 8。
                - dropout (float, optional): Dropout 概率，默认 0.1。
                - activation (str, optional): 激活函数类型 ('gelu' 或 'relu')。
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_levels = config.num_levels
        self.num_layers = config.num_layers

        # 1. 多尺度体素通道投影模块
        self.projector = MultiScaleChannelProjector(config)

        # 2. 3D 正弦位置编码模块
        self.position_encoding = SinePositionEmbedding(
            num_pos_feats=self.hidden_size // 2,
            temperature=10000.0,
            normalize=True,
        )
        voxel_pos_in_dim = int(1.5 * self.hidden_size)
        self.voxel_pos_proj = StateEncoderMLP(
            input_dim=voxel_pos_in_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.hidden_size,
            num_layers=2,
            activation=config.activation,
        )

        # 3. 尺度/层级可学习 Embedding (区分不同特征层级)
        self.level_embed = nn.Parameter(torch.Tensor(self.num_levels, self.hidden_size))
        nn.init.normal_(self.level_embed)

        # 4. 堆叠多层 StateEncoderLayer
        self.layers = nn.ModuleList([
            DeformableVoxelEncoderLayer(config) for _ in range(self.num_layers)
        ])

    @staticmethod
    def get_reference_points(
        spatial_shapes: List[Tuple[int, int, int]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        生成多尺度网格各体素中心归一化 3D 参考坐标 (cx, cy, cz) in [0, 1].

        Args:
            spatial_shapes: 各层 3D 网格尺寸 [(X1, Y1, Z1), (X2, Y2, Z2), ...]
            device: 设备
            dtype: 数据类型

        Returns:
            reference_points: 形状为 [1, N_total, 3] 的归一化参考点坐标
        """
        ref_points_list = []
        for X, Y, Z in spatial_shapes:
            # 生成 0.5 到 Shape - 0.5 的体素中心网格
            grid_x, grid_y, grid_z = torch.meshgrid(
                torch.linspace(0.5, X - 0.5, X, dtype=dtype, device=device),
                torch.linspace(0.5, Y - 0.5, Y, dtype=dtype, device=device),
                torch.linspace(0.5, Z - 0.5, Z, dtype=dtype, device=device),
                indexing="ij",
            )
            # 归一化到 [0, 1] 区间
            grid_x = grid_x / X
            grid_y = grid_y / Y
            grid_z = grid_z / Z

            # 沿最后一维堆叠为 [X, Y, Z, 3] -> 展平为 [X*Y*Z, 3]
            ref_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).flatten(0, 2)
            ref_points_list.append(ref_points)

        # 沿 Token 序列维度拼接: [Sum(N_l), 3] -> [1, Sum(N_l), 3]
        reference_points = torch.cat(ref_points_list, dim=0).unsqueeze(0)
        return reference_points

    def forward(
        self,
        multi_level_feats: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[Tuple[int, int, int]]]:
        """
        Args:
            multi_level_feats: 多尺度 3D 特征列表 [F1, F2, F3_temp]，
                               每个张量形状为 [B, C_in_i, X_i, Y_i, Z_i]。

        Returns:
            memory: 编码增强后的展平特征序列 [B, Total_Tokens, hidden_size]
            pos_embed: 融合了 3D 空间与层级编码的位置向量序列 [B, Total_Tokens, hidden_size]
            enhanced_spatial_feats: 重构还原为多尺度 3D 网格的特征列表 [[B, hidden_size, X_l, Y_l, Z_l], ...]
            spatial_shapes: 各层网格尺寸列表 [(X1, Y1, Z1), (X2, Y2, Z2), ...]
        """
        assert len(multi_level_feats) == self.num_levels, (
            f"Expected {self.num_levels} feature levels, but got {len(multi_level_feats)}"
        )

        B = multi_level_feats[0].shape[0]
        device = multi_level_feats[0].device
        dtype = multi_level_feats[0].dtype

        # 1. 统一多尺度通道维度
        projected_feats = self.projector(multi_level_feats)

        flattened_feats = []
        flattened_pos = []
        spatial_shapes = []

        # 2. 提取各层网格尺寸、生成位置编码并展平
        for lvl, feat in enumerate(projected_feats):
            _, _, X, Y, Z = feat.shape
            spatial_shapes.append((X, Y, Z))

            # 3D 正弦位置编码: [B, Pos_C, X, Y, Z]
            pos = self.position_encoding(feat)
            pos = pos.permute(0, 2, 3, 4, 1)  # -> [B, X, Y, Z, Pos_C]
            pos = self.voxel_pos_proj(pos)          # -> [B, X, Y, Z, hidden_size]

            # 注入可学习层级编码
            lvl_emb = self.level_embed[lvl].view(1, 1, 1, 1, self.hidden_size)
            pos_with_lvl = pos + lvl_emb

            # 空间维度展平为 Token 序列: [B, X*Y*Z, hidden_size]
            flat_feat = feat.permute(0, 2, 3, 4, 1).flatten(1, 3)
            flat_pos = pos_with_lvl.flatten(1, 3)

            flattened_feats.append(flat_feat)
            flattened_pos.append(flat_pos)

        # 沿序列维度拼接
        src = torch.cat(flattened_feats, dim=1)        # [B, Total_Tokens, hidden_size]
        pos_embed = torch.cat(flattened_pos, dim=1)    # [B, Total_Tokens, hidden_size]

        # 3. 动态生成多尺度 3D 参考点坐标: [B, Total_Tokens, 3]
        reference_points = self.get_reference_points(spatial_shapes, device=device, dtype=dtype)
        reference_points = reference_points.repeat(B, 1, 1)

        # 4. 逐层 Encoder 增强与 3D Value 特征迭代更新
        current_spatial_feats = projected_feats
        for layer in self.layers:
            # 经过单层可变形自注意力与 FFN 计算
            src = layer(
                src=src,
                pos_embed=pos_embed,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                spatial_feats=current_spatial_feats,
            )

            # 将更新后的 src 切片还原回 3D 特征列表，供下一层可变形注意力采样
            current_spatial_feats = []
            start_idx = 0
            for (X, Y, Z) in spatial_shapes:
                end_idx = start_idx + X * Y * Z
                lvl_tokens = src[:, start_idx:end_idx, :]
                spatial_feat = (
                    lvl_tokens.view(B, X, Y, Z, self.hidden_size)
                    .permute(0, 4, 1, 2, 3)
                    .contiguous()
                )
                current_spatial_feats.append(spatial_feat)
                start_idx = end_idx


        return DeformableVoxelEncoderOutput(
            hidden_states=src,
            pos_embed=pos_embed,
            spatial_feats=current_spatial_feats,
            spatial_shapes=spatial_shapes,
        )
    
# 