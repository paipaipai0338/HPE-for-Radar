import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Any, Optional, Tuple, List
from src.models.module_outputs.state_decoder_outputs import StateDecoderOutput
from src.models import SDS
from src.models.modules.position_coding import *

class StateDecoderMLP(nn.Module):
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


class StateDecoderAttention(nn.Module):
    """
    基于 PyTorch 官方 F.scaled_dot_product_attention 封装的高性能通用注意力模块。
    
    重塑说明：
    1. 内部移除了 dropout，统一由外层 DecoderLayer 的 Dropout 接管（设置 dropout_p = 0.0）；
    2. 显式添加了点积的 scale 缩放系数 (1.0 / sqrt(head_dim))，尽管 PyTorch SDPA 默认处理，
       但显式声明能保证行为完全透明且符合 Transformer 标准定义；
    3. 线性层启用 bias=True，严格通过 config 实例化。
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config

        embed_dim = config.hidden_size
        num_heads = config.num_heads

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"hidden_size ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, Q_len, hidden_size]
            key: [B, KV_len, hidden_size], 若为 None 则默认取 query (Self-Attention)
            value: [B, KV_len, hidden_size], 若为 None 则默认取 query (Self-Attention)
            query_pos: [B, Q_len, hidden_size], 条件位置编码
            key_pos: [B, KV_len, hidden_size], 网格/序列位置编码
            attn_mask: 4D 浮点加性偏置 (3D BoxRPB) [B, num_heads, Q_len, KV_len]
                       或布尔掩码 [B, num_heads, Q_len, KV_len] / [Q_len, KV_len]
        """
        batch_size, q_len, _ = query.shape
        kv_len = key.shape[1]

        # 2. 线性投影并切分多头: [B, L, C] -> [B, num_heads, L, head_dim]
        q = self.q_proj(query).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3.
        if attn_mask is not None and attn_mask.dim() == 3:
            raise ValueError(
                f"{attn_mask} 数据维度错误,其维度应为4:[batch_size, num_heads, num_query, num_key(value)]."
            )

        # 4. 调用底层优化的注意力算子
        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=self.scale,
        )

        # 5. 合并多头并输出映射: [B, num_heads, Q_len, head_dim] -> [B, Q_len, hidden_size]
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.embed_dim)
        out = self.out_proj(out)
        return out

class StateDecoderFFN(nn.Module):
    """
    State Decoder 单层内部的前馈神经网络模块（FFN / MLP Sub-layer）。
    
    结构：
    Linear(hidden_size, intermediate_size) -> Activation -> Dropout -> Linear(intermediate_size, hidden_size)
    """

    def __init__(self, config: Any):
        """
        Args:
            config: 配置对象，需包含：
                - hidden_size (int): 输入与输出特征通道数
                - intermediate_size (int): FFN 隐藏层扩展通道数
                - ffn_dropout (float): FFN 内部激活后的 Dropout 概率（与残差 dropout 进行参数名区分）
                - activation (str): 激活函数类型 ('gelu' 或 'relu')
        """
        super().__init__()
        self.config = config

        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        ffn_dropout = config.dropout
        activation = config.activation

        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.dropout = nn.Dropout(ffn_dropout)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)

        if activation.lower() == "gelu":
            self.act = nn.GELU()
        elif activation.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}. Choose 'gelu' or 'relu'.")

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征张量，形状为 [B, L, hidden_size]
        Returns:
            输出特征张量，形状为 [B, L, hidden_size]
        """
        return self.linear2(self.dropout(self.act(self.linear1(x))))

class StateDecoderLayer(nn.Module):
    """
    单层人体状态解码器模块（StateDecoderLayer）。

    采用经典 Transformer Decoder 架构，按序集成：
    1. 自注意力子层 (Self-Attention): 建模 presence token 与 cls queries 间的多目标交互与去重；
    2. 3D 空间交叉注意力子层 (Cross-Attention): 融合 3D BoxRPB 加性偏置，实现几何软引导的体素特征聚合；
    3. 前馈网络子层 (FFN): 执行非线性特征重组。
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # 1. 自注意力子层
        self.self_attn = StateDecoderAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.self_attn_dropout = nn.Dropout(config.dropout)

        # 2. 3D 空间交叉注意力子层
        self.voxel_cross_attn = StateDecoderAttention(config)
        self.voxel_cross_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.voxel_cross_attn_dropout = nn.Dropout(config.dropout)

        # 3. 前馈神经网络子层
        self.ffn = StateDecoderFFN(config)
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_pos: Optional[torch.Tensor],
        voxel_features: Optional[torch.Tensor],
        voxel_pos_encoding: Optional[torch.Tensor],
        voxel_cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: 输入状态序列，形状为 [B, 1 + Q, hidden_size]
                           (第 0 位为 presence_token，后 Q 位为 cls_tokens)
            query_pos: Query 条件位置编码，形状为 [B, 1 + Q, hidden_size]
                       (presence_token 对应位置为 0，cls_tokens 对应位置为 3D 框的正弦位置投影)
            voxel_features: 展平后的 3D 时空体素特征，形状为 [B, N_voxels, hidden_size] (N_voxels = X * Y * Z)
            voxel_pos_encoding: 展平后的 3D 体素空间位置编码，形状为 [B, N_voxels, hidden_size]
            voxel_cross_attn_mask: 3D BoxRPB 加性偏置矩阵，形状为 [B, num_heads, 1 + Q, N_voxels]
                                    (第 0 行 presence_token 全为 0，后 Q 行为 3D 边界框的几何距离偏置)

        Returns:
            hidden_states: 更新后的状态序列，形状为 [B, 1 + Q, hidden_size]
        """
        # 对齐 query_pos 长度 (若传入的是 Q 个框的 pos，则在第 0 位为 presence_token 补 0)
        if query_pos is not None and query_pos.shape[1] < hidden_states.shape[1]:
            query_pos = F.pad(query_pos, (0, 0, 1, 0), mode="constant", value=0)

        # =========================================================================
        # 1. Self-attention with query position encoding
        # =========================================================================
        residual = hidden_states
        query_with_pos = hidden_states + query_pos

        attn_output = self.self_attn(
            query=query_with_pos,
            key=query_with_pos,
            value=hidden_states,
            attn_mask=None,
        )
        hidden_states = residual + self.self_attn_dropout(attn_output)
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # =========================================================================
        # 2. voxel cross-attention: queries attend to voxel features (with RPB)
        # =========================================================================
        residual = hidden_states
        query_with_pos = hidden_states + query_pos
        key_with_pos = voxel_features + voxel_pos_encoding

        attn_output = self.voxel_cross_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=voxel_features,
            attn_mask=voxel_cross_attn_mask,
        )
        hidden_states = residual + self.voxel_cross_attn_dropout(attn_output)
        hidden_states = self.voxel_cross_attn_layer_norm(hidden_states)


        # =========================================================================
        # 3. FFN Sub-layer
        # =========================================================================
        residual = hidden_states
        mlp_output = self.ffn(hidden_states)
        hidden_states = residual + self.ffn_dropout(mlp_output)
        hidden_states = self.ffn_layer_norm(hidden_states)

        return hidden_states

def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """The inverse function for sigmoid activation function."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def box_cxcyczsxsysz_to_xyzxyz(boxes: torch.Tensor) -> torch.Tensor:
    """
    将 6D 边界框从 [cx, cy, cz, sx, sy, sz] 转换为 [x_min, y_min, z_min, x_max, y_max, z_max]。
    """
    cx, cy, cz, sx, sy, sz = boxes.unbind(-1)
    x_min = cx - 0.5 * sx
    y_min = cy - 0.5 * sy
    z_min = cz - 0.5 * sz
    x_max = cx + 0.5 * sx
    y_max = cy + 0.5 * sy
    z_max = cz + 0.5 * sz
    return torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], dim=-1)

@SDS.register("boxRPB3D_State_Decoder")
class StateDecoder(nn.Module):
    """
    基于 SAM 3 架构演进的 3D 毫米波雷达人体状态解码器（StateDecoder）。

    核心机制：
    1. 统一通过 config 实例化；
    2. 共享头结构：各层共享同一个 box_head、cls_head 与 presence_head；
    3. 逆 Sigmoid 增量微调 (Box Refinement)：逐层更新并在传给下一层采样前截断梯度 (.detach())；
    4. 3D BoxRPB (三维体素加性偏置)：基于 log-scale 编码计算体素网格到 6D 框边界的三轴正交偏置；
    5. 复用 SinePositionEmbedding(num_pos_feats=hidden_size // 2)，6 维坐标编码后输出 3 * hidden_size，
       直接对接输入为 3 * hidden_size 的 ref_point_head；
    6. Presence Token 机制：第 0 位全局存在性预测，享有全图无偏置感受野，输出截断防饱和；
    7. 返回 StateDecoderOutput 数据结构。
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_decoder_layers
        self.num_queries = config.num_queries
        self.num_heads = config.num_heads
        self.clamp_presence_logit_max_val = getattr(config, "clamp_presence_logit_max_val", 10.0)

        # 1. 级联解码器层
        self.layers = nn.ModuleList([StateDecoderLayer(config) for _ in range(self.num_layers)])

        # 2. 查询与参考框可学习参数
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_size)
        self.reference_points = nn.Embedding(self.num_queries, 6)
        self.presence_token = nn.Embedding(1, self.hidden_size)

        # 3. 共享预测头 (Shared Heads across layers)
        self.box_head = StateDecoderMLP(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=6,
            num_layers=3,
            activation=config.activation,
        )
        self.cls_head = StateDecoderMLP(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=1,
            num_layers=3,
            activation=config.activation,
        )
        self.presence_head = StateDecoderMLP(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=1,
            num_layers=3,
            activation=config.activation,
        )

        # 4. 3D 条件位置编码与投影层
        # 单坐标 num_pos_feats = hidden_size // 2，6 个坐标编码后总维度为 6 * (hidden_size // 2) = 3 * hidden_size
        self.position_encoding = SinePositionEmbedding(
            num_pos_feats=self.hidden_size // 2,
            temperature=10000.0,
            normalize=True,
        )
        
        # 3D 体素网格位置编码映射头 (1.5 * hidden_size -> hidden_size)
        voxel_pos_in_dim = int(1.5 * self.hidden_size)
        self.voxel_pos_proj = StateDecoderMLP(
            input_dim=voxel_pos_in_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.hidden_size,
            num_layers=2,
            activation=config.activation,
        )

        self.ref_point_head = StateDecoderMLP(
            input_dim=3 * self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.hidden_size,
            num_layers=2,
            activation=config.activation,
        )

        # 5. 3D BoxRPB 三轴 Delta 映射模块
        self.box_rpb_embed_x = StateDecoderMLP(2, self.hidden_size, self.num_heads, 2, config.activation)
        self.box_rpb_embed_y = StateDecoderMLP(2, self.hidden_size, self.num_heads, 2, config.activation)
        self.box_rpb_embed_z = StateDecoderMLP(2, self.hidden_size, self.num_heads, 2, config.activation)

        self._init_weights()

    def _init_weights(self):
        # 1. 构造符合雷达 3D 人体空间先验的初始 Reference Boxes
        with torch.no_grad():
            init_boxes = torch.zeros_like(self.reference_points.weight)

            # (1) 中心点 [cx, cy, cz]: 覆盖空间的主要探测区域 [0.15, 0.85]
            # 若 Q=8，可以做简单的网格/均匀散布，也可直接 uniform
            init_boxes[:, 0].uniform_(0.15, 0.85)  # cx
            init_boxes[:, 1].uniform_(0.15, 0.85)  # cy
            init_boxes[:, 2].uniform_(0.30, 0.70)  # cz

            # (2) 尺寸 [sx, sy, sz]: 匹配典型人体在归一化雷达空间中的比例
            init_boxes[:, 3].uniform_(0.10, 0.18)  # sx: 约 0.5m ~ 1.1m (基准 6m)
            init_boxes[:, 4].uniform_(0.10, 0.18)  # sy: 约 0.5m ~ 1.1m (基准 6m)
            init_boxes[:, 5].uniform_(0.35, 0.50)  # sz: 约 1.2m ~ 2.0m (基准 4m)

            # (3) 逆 Sigmoid 写入可学习参数权重中
            self.reference_points.weight.copy_(inverse_sigmoid(init_boxes))

        # 2. 初始化 Head 参数
        # 边界框回归头的最后一层权重和偏置清零，保证初始预测出的增量 delta ≈ 0
        nn.init.zeros_(self.box_head.layers[-1].weight)
        nn.init.zeros_(self.box_head.layers[-1].bias)

        # 类别预测头使用 Focal Loss 的标准偏置初始化 (如 -2.19 对应初始正样本概率 ~0.1)
        nn.init.constant_(self.cls_head.layers[-1].bias, -2.19)
        nn.init.constant_(self.presence_head.layers[-1].bias, 0.0)

    def _get_coords(
        self, X: int, Y: int, Z: int, dtype: torch.dtype, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """生成归一化的 3D 体素中心坐标序列。"""
        coords_x = (torch.arange(0, X, device=device, dtype=dtype) + 0.5) / X
        coords_y = (torch.arange(0, Y, device=device, dtype=dtype) + 0.5) / Y
        coords_z = (torch.arange(0, Z, device=device, dtype=dtype) + 0.5) / Z
        return coords_x, coords_y, coords_z

    def _get_rpb_matrix(
        self, reference_boxes: torch.Tensor, spatial_shapes: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        计算 3D BoxRPB (相对位置偏置) 矩阵，采用与 SAM 3 对齐的 log-scale 编码机制。

        Args:
            reference_boxes: 当前层 3D 参考框 [B, Q, 6]，位于 Sigmoid 空间 [0, 1]
            spatial_shapes: 3D 体素网格维度 (X, Y, Z)

        Returns:
            rpb_matrix: [B, num_heads, Q, X * Y * Z]
        """
        X, Y, Z = spatial_shapes
        batch_size, num_queries, _ = reference_boxes.shape
        boxes_min_max = box_cxcyczsxsysz_to_xyzxyz(reference_boxes)  # [B, Q, 6]

        coords_x, coords_y, coords_z = self._get_coords(
            X, Y, Z, dtype=reference_boxes.dtype, device=reference_boxes.device
        )

        # 1. 计算三轴方向体素坐标与包围盒对应最小/最大边界的相对 Delta
        # X 轴: 边界索引为 [0, 3] -> (x_min, x_max)
        deltas_x = coords_x.view(1, -1, 1) - boxes_min_max.reshape(-1, 1, 6)[:, :, 0:4:3]
        deltas_x = deltas_x.view(batch_size, num_queries, -1, 2)  # [B, Q, X, 2]

        # Y 轴: 边界索引为 [1, 4] -> (y_min, y_max)
        deltas_y = coords_y.view(1, -1, 1) - boxes_min_max.reshape(-1, 1, 6)[:, :, 1:5:3]
        deltas_y = deltas_y.view(batch_size, num_queries, -1, 2)  # [B, Q, Y, 2]

        # Z 轴: 边界索引为 [2, 5] -> (z_min, z_max)
        deltas_z = coords_z.view(1, -1, 1) - boxes_min_max.reshape(-1, 1, 6)[:, :, 2:6:3]
        deltas_z = deltas_z.view(batch_size, num_queries, -1, 2)  # [B, Q, Z, 2]

        # 2. SAM 3 风格 Log-scale 非线性压缩
        deltas_x_log = torch.sign(deltas_x * 8.0) * torch.log2(torch.abs(deltas_x * 8.0) + 1.0) / math.log2(8.0)
        deltas_y_log = torch.sign(deltas_y * 8.0) * torch.log2(torch.abs(deltas_y * 8.0) + 1.0) / math.log2(8.0)
        deltas_z_log = torch.sign(deltas_z * 8.0) * torch.log2(torch.abs(deltas_z * 8.0) + 1.0) / math.log2(8.0)

        # 3. MLP 映射至多头偏置通道: [B, Q, Dim, num_heads]
        bias_x = self.box_rpb_embed_x(deltas_x_log)
        bias_y = self.box_rpb_embed_y(deltas_y_log)
        bias_z = self.box_rpb_embed_z(deltas_z_log)

        # 4. 三维正交广播相加: [B, Q, X, Y, Z, num_heads]
        rpb = (
            bias_x.unsqueeze(3).unsqueeze(4)
            + bias_y.unsqueeze(2).unsqueeze(4)
            + bias_z.unsqueeze(2).unsqueeze(3)
        )

        # 5. 展平空间网格并调整多头维度: [B, num_heads, Q, X*Y*Z]
        rpb = rpb.flatten(2, 4)
        return rpb.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        voxel_features: torch.Tensor,
        spatial_shapes: Optional[Tuple[int, int, int]] = None,
        raw_point_cloud: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> StateDecoderOutput:
        """
        State Decoder 前向传播。

        Args:
            voxel_features: 展平后的 3D 时空体素特征，形状为 [B, X*Y*Z, hidden_size]
                            (若传入 5D [B, C, X, Y, Z]，内部会自动展平为 3D)
            voxel_pos_encoding: 3D 体素空间位置编码，形状为 [B, X*Y*Z, hidden_size]
            spatial_shapes: 3D 体素网格维度 (X, Y, Z)
            raw_point_cloud: (可选) 原始点云，透传挂载于输出对象中

        Returns:
            StateDecoderOutput: 包含各层 hidden_states、boxes、presence_logits、cls_logits 的数据类
        """
        # 兼容 5D 体素特征输入 [B, C, X, Y, Z] -> [B, X*Y*Z, C]
        if voxel_features.dim() == 5:
            B, C, X, Y, Z = voxel_features.shape
            spatial_shapes = (X, Y, Z)
            # (1) 计算 3D 体素空间位置编码: [B, 1.5 * hidden_size, X, Y, Z]
            voxel_pos_raw = self.position_encoding(voxel_features)
            
            # (2) 展平为 [B, X*Y*Z, 1.5 * hidden_size] 并通过 MLP 投影至 [B, X*Y*Z, hidden_size]
            voxel_pos_flat = voxel_pos_raw.flatten(2).transpose(1, 2)
            voxel_pos_encoding = self.voxel_pos_proj(voxel_pos_flat)
            # (3) 展平体素特征: [B, X*Y*Z, C]
            voxel_features = voxel_features.flatten(2).transpose(1, 2)
        else:
            raise ValueError(
                f"[StateDecoder] 输入的 `voxel_features` 期望为 5D 密集张量 [B, C, X, Y, Z]，"
                f"但接收到的张量维度为 {voxel_features.dim()}D (shape={tuple(voxel_features.shape)})。"
                f"请确保传入前未被展平，以便内部正确计算 3D 空间正弦位置编码及三维网格形状 (X, Y, Z)。"
            )

        batch_size = voxel_features.shape[0]

        # 1. 扩展可学习 Query 与初始 Reference Boxes
        query_embeds = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [B, Q, C]
        reference_boxes = self.reference_points.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [B, Q, 6]
        reference_boxes = reference_boxes.sigmoid()  # 约束在 [0, 1] 空间

        presence_token = self.presence_token.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 1, C]

        # 拼接 Presence Token 与 Query Embeddings -> [B, 1 + Q, C]
        hidden_states = torch.cat([presence_token, query_embeds], dim=1)

        intermediate_hidden_states = []
        intermediate_boxes = [reference_boxes]
        intermediate_presence_logits = []
        intermediate_cls_logits = []

        # 2. 逐层微调迭代解码
        for layer in self.layers:
            # 2.1 基于当前 reference_boxes 生成 3D 条件位置编码
            # 输出形状: [B, Q, 6 * (hidden_size // 2)] = [B, Q, 3 * hidden_size]
            query_sine_embed = self.position_encoding.encode_boxes(reference_boxes)
            query_pos = self.ref_point_head(query_sine_embed)  # [B, Q, hidden_size]

            # 2.2 计算当前层的 3D BoxRPB 加性偏置矩阵
            voxel_cross_attn_mask = None
            if spatial_shapes is not None:
                rpb_matrix = self._get_rpb_matrix(reference_boxes, spatial_shapes)
                # 为第 0 位的 presence_token 在第 2 维前置补 0 偏置 (赋予全局均匀感受野)
                voxel_cross_attn_mask = F.pad(rpb_matrix, (0, 0, 1, 0), mode="constant", value=0)

            # 2.3 解码层更新
            hidden_states = layer(
                hidden_states=hidden_states,
                query_pos=query_pos,
                voxel_features=voxel_features,
                voxel_pos_encoding=voxel_pos_encoding,
                voxel_cross_attn_mask=voxel_cross_attn_mask,
                **kwargs,
            )

            # 2.4 分离 Presence Token 与 Query States
            presence_hidden = hidden_states[:, :1]  # [B, 1, C]
            query_hidden_states = hidden_states[:, 1:]  # [B, Q, C]

            # 2.5 6D 边界框逆 Sigmoid 增量微调 (Box Refinement)
            reference_boxes_before_sigmoid = inverse_sigmoid(reference_boxes)
            delta_boxes = self.box_head(query_hidden_states)
            new_reference_boxes = (delta_boxes + reference_boxes_before_sigmoid).sigmoid()
            reference_boxes = new_reference_boxes.detach()  # 截断梯度传入下一层

            # 2.7 预测 Query 二分类 Logits
            cls_logits = self.cls_head(query_hidden_states).squeeze(-1)  # [B, Q]

            # 2.8 预测全局 Presence Logits 并执行数值截断防饱和
            presence_logits = self.presence_head(presence_hidden).squeeze(-1)  # [B, 1]
            presence_logits = presence_logits.clamp(
                min=-self.clamp_presence_logit_max_val,
                max=self.clamp_presence_logit_max_val,
            )

            # 2.9 收集当前层输出
            intermediate_hidden_states.append(query_hidden_states)
            intermediate_boxes.append(new_reference_boxes)
            intermediate_presence_logits.append(presence_logits)
            intermediate_cls_logits.append(cls_logits)

        # 3. 堆叠所有解码层的结果 -> [L, B, ...]
        stacked_hidden_states = torch.stack(intermediate_hidden_states, dim=0)
        stacked_boxes = torch.stack(intermediate_boxes[:-1], dim=0)
        stacked_presence_logits = torch.stack(intermediate_presence_logits, dim=0)
        stacked_cls_logits = torch.stack(intermediate_cls_logits, dim=0)

        return StateDecoderOutput(
            intermediate_hidden_states=stacked_hidden_states,
            intermediate_boxes=stacked_boxes,
            intermediate_presence_logits=stacked_presence_logits,
            intermediate_cls_logits=stacked_cls_logits,
            last_presence_token=presence_hidden,
            raw_point_cloud=raw_point_cloud,
        )

class MSDeformableCrossAttention3D(nn.Module):
    """
    解耦双分支多尺度 3D 可变形交叉注意力模块 (MSDeformableCrossAttention3D).

    架构特性：
    1. 双分支解耦：
       - Presence Token: 采用标准 Multi-Head Cross-Attention 与全局展平特征 memory 交互，享有全场景无偏置感受野；
       - Q 个 Cls Query Tokens: 采用 6D Box-Aware 3D 可变形注意力，在多尺度体素网格上进行尺寸自适应采样。
    2. 6D Box-Aware 采样：
       - 利用参考框中心点 (cx, cy, cz) 与尺寸 (sx, sy, sz) 联合缩放采样偏移量，实现尺度自适应局部搜索。
    3. 严格对齐 F.grid_sample 坐标轴系与全局跨层 Softmax 归一化。
    """

    def __init__(self, config: Any):
        """
        Args:
            config: 配置对象，需包含以下字段：
                - hidden_size (int): 特征隐藏维度。
                - num_attention_heads (int): 注意力头数。
                - num_levels (int): 特征尺度层级数 (如 F1, F2, F3 共 3 层)。
                - num_points (int): 每个 Head 在每个尺度采样的 3D 关键点数量 (默认 4)。
                - dropout (float, optional): Dropout 概率，默认 0.1。
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_levels = config.num_levels
        self.num_points = config.num_points
        dropout = config.dropout

        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size {self.hidden_size} must be divisible by num_attention_heads {self.num_heads}"
        )
        self.head_dim = self.hidden_size // self.num_heads

        # --- 1. Presence Token 分支：标准全局多头交叉注意力 ---
        self.presence_cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # --- 2. Query Tokens 分支：6D Box-Aware 3D 可变形交叉注意力 ---
        # 预测 3D 连续采样偏移量 [dx, dy, dz]
        self.sampling_offsets = nn.Linear(
            self.hidden_size, self.num_heads * self.num_levels * self.num_points * 3
        )
        # 跨所有尺度和采样点联合预测注意力权重
        self.attention_weights = nn.Linear(
            self.hidden_size, self.num_heads * self.num_levels * self.num_points
        )
        # 线性映射层
        self.value_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self._reset_parameters()

    def _reset_parameters(self):
        # 偏置与权重初始化
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        # 适度随机初始化初始采样偏移偏置
        nn.init.uniform_(self.sampling_offsets.bias.data, -1.0, 1.0)
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_pos: torch.Tensor,
        reference_boxes: torch.Tensor,
        spatial_shapes: List[Tuple[int, int, int]],
        spatial_feats: List[torch.Tensor],
        memory: torch.Tensor,
        memory_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: 当前层输入 Token 序列 [B, 1 + Q, hidden_size]
                           (第 0 位为 presence_token，第 1: 位为 query_tokens)
            query_pos: 由 6D 参考框生成的 Query 条件位置编码 [B, Q, hidden_size]
            reference_boxes: 当前层 6D 归一化参考框 [B, Q, 6] (取值在 [0, 1] 空间)
            spatial_shapes: 多尺度网格尺寸列表 [(X1, Y1, Z1), (X2, Y2, Z2), (X3, Y3, Z3)]
            spatial_feats: 多尺度 3D 特征列表 [[B, hidden_size, X_l, Y_l, Z_l], ...]
            memory: 展平后的全局编码特征 [B, Total_Tokens, hidden_size]
            memory_pos: 全局编码特征的空间+层级位置编码 [B, Total_Tokens, hidden_size] (可选)

        Returns:
            out_hidden_states: 交叉注意力更新后的 Token 序列 [B, 1 + Q, hidden_size]
        """
        B, N_tokens, _ = hidden_states.shape

        # 分离 Presence Token 与 Query Tokens
        presence_token = hidden_states[:, :1, :]      # [B, 1, hidden_size]
        query_states = hidden_states[:, 1:, :]        # [B, Q, hidden_size]
        Q = query_states.shape[1]

        # ==================== 分支 1: Presence Token 全局标准交叉注意力 ====================
        mem_k = memory if memory_pos is None else (memory + memory_pos)
        presence_out, _ = self.presence_cross_attn(
            query=presence_token,
            key=mem_k,
            value=memory,
            need_weights=False,
        )  # [B, 1, hidden_size]

        # ==================== 分支 2: Query Tokens 6D Box-Aware 3D 可变形交叉注意力 ====================
        # 1. 注入几何位置编码引导采样
        query_total = query_states + query_pos

        # 2. 预测连续 3D 偏移量与注意力权重
        # offsets: [B, Q, num_heads, num_levels, num_points, 3]
        offsets = self.sampling_offsets(query_total).view(
            B, Q, self.num_heads, self.num_levels, self.num_points, 3
        )
        # attention_weights: [B, Q, num_heads, num_levels * num_points] -> 在所有采样点上联合 Softmax
        attn_weights = self.attention_weights(query_total).view(
            B, Q, self.num_heads, self.num_levels * self.num_points
        )
        attn_weights = F.softmax(attn_weights, dim=-1).view(
            B, Q, self.num_heads, self.num_levels, self.num_points
        )

        # 3. 解析 6D 参考框：中心点 (cx, cy, cz) 与尺寸 (sx, sy, sz)
        box_center = reference_boxes[..., :3].unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B, Q, 1, 1, 3]
        box_size = reference_boxes[..., 3:].unsqueeze(2).unsqueeze(3).unsqueeze(4)    # [B, Q, 1, 1, 3]

        # 6D Box 尺寸自适应缩放: loc = center + 0.5 * size * tanh(offset)
        # 取值依然自然约束在 [0, 1] 附近
        sampling_locations = box_center + 0.5 * box_size * torch.tanh(offsets)  # [B, Q, num_heads, num_levels, num_points, 3]

        # 4. 多尺度 3D 插值采样
        sampled_value_list = []
        for lvl, feat in enumerate(spatial_feats):
            X_l, Y_l, Z_l = spatial_shapes[lvl]

            # 投影 Value: [B, hidden_size, X, Y, Z] -> [B * num_heads, head_dim, X, Y, Z]
            v = self.value_proj(feat.permute(0, 2, 3, 4, 1)).view(
                B, X_l, Y_l, Z_l, self.num_heads, self.head_dim
            )
            v = v.permute(0, 4, 5, 1, 2, 3).flatten(0, 1)

            # 当前层归一化采样点: [B, Q, num_heads, num_points, 3] -> (cx, cy, cz)
            loc_lvl = sampling_locations[:, :, :, lvl, :, :]

            # F.grid_sample 5D 期望坐标为 (grid_x, grid_y, grid_z) 对应张量的 (Z, Y, X)
            # 将归一化 [0, 1] 转换为 [-1, 1]
            grid_cx = loc_lvl[..., 0] * 2.0 - 1.0  # 对应 X 轴 (dim_X)
            grid_cy = loc_lvl[..., 1] * 2.0 - 1.0  # 对应 Y 轴 (dim_Y)
            grid_cz = loc_lvl[..., 2] * 2.0 - 1.0  # 对应 Z 轴 (dim_Z)

            # 按照 (Z, Y, X) 顺序组装为 grid: [B, Q, num_heads, num_points, 3]
            grid_3d = torch.stack([grid_cz, grid_cy, grid_cx], dim=-1)

            # 调整为 grid_sample 接受的 5D 张量形状: [B * num_heads, Q, num_points, 1, 3]
            grid_5d = grid_3d.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(-2)

            # 3D 双线性插值采样: [B * num_heads, head_dim, Q, num_points, 1]
            sampled_feat = F.grid_sample(
                v,
                grid_5d,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).squeeze(-1)

            # 重塑维度: [B, num_heads, head_dim, Q, num_points] -> [B, Q, num_heads, num_points, head_dim]
            sampled_feat = (
                sampled_feat.view(B, self.num_heads, self.head_dim, Q, self.num_points)
                .permute(0, 3, 1, 4, 2)
            )

            # 加权求和当前层采样点: [B, Q, num_heads, head_dim]
            weights_lvl = attn_weights[:, :, :, lvl, :].unsqueeze(-2)  # [B, Q, num_heads, 1, num_points]
            weighted_feat_lvl = (sampled_feat.permute(0, 1, 2, 4, 3) @ weights_lvl.transpose(-1, -2)).squeeze(-1)

            sampled_value_list.append(weighted_feat_lvl)

        # 跨尺度加和并投影
        query_out = sum(sampled_value_list)               # [B, Q, num_heads, head_dim]
        query_out = self.output_proj(query_out.flatten(2))  # [B, Q, hidden_size]

        # ==================== 合并双分支输出 ====================
        out_hidden_states = torch.cat([presence_out, query_out], dim=1)  # [B, 1 + Q, hidden_size]

        return out_hidden_states


class DeformableStateDecoderLayer(nn.Module):
    """
    状态解码器单层模块 (StateDecoderLayer).

    采用 Post-LN 结构，包含三个核心子层：
    1. Multi-Head Self-Attention: Token 间全局交互（显式注入 Query 位置编码）
    2. MSDeformableCrossAttention3D: 解耦双分支多尺度 3D 可变形交叉注意力
    3. Feed-Forward Network: 前馈非线性映射
    """

    def __init__(self, config: Any):
        """
        Args:
            config: 配置对象，需包含以下字段：
                - hidden_size (int): 特征维度。
                - num_heads (int): 注意力头数。
                - intermediate_size (int): FFN 隐藏层维度。
                - num_levels (int): 特征尺度层级数。
                - num_points (int, optional): 每个 Head 每层采样的关键点数。
                - dropout (float, optional): Dropout 概率，默认 0.1。
                - activation (str, optional): 激活函数类型 ('gelu' 或 'relu')。
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        dropout = getattr(config, "dropout", 0.1)

        # 1. Self-Attention 子层
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn_norm = nn.LayerNorm(self.hidden_size)

        # 2. Multi-Scale 3D Deformable Cross-Attention 子层
        self.cross_attn = MSDeformableCrossAttention3D(config)
        self.cross_attn_dropout = nn.Dropout(dropout)
        self.cross_attn_norm = nn.LayerNorm(self.hidden_size)

        # 3. FFN 子层 (直接使用 StateEncoderFFN 保持统一实现)
        self.ffn = StateDecoderFFN(config)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_pos: torch.Tensor,
        reference_boxes: torch.Tensor,
        spatial_shapes: List[Tuple[int, int, int]],
        spatial_feats: List[torch.Tensor],
        memory: torch.Tensor,
        memory_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: 当前层输入 Token 序列，形状为 [B, 1 + Q, hidden_size]
                           (第 0 位为 presence_token，第 1: 位为 query_tokens)
            query_pos: 由 6D 参考框生成的 Query 条件位置编码，形状为 [B, Q, hidden_size]
            reference_boxes: 当前层 6D 归一化参考框 [B, Q, 6]，取值在 [0, 1] 空间
            spatial_shapes: 多尺度网格尺寸列表 [(X1, Y1, Z1), (X2, Y2, Z2), (X3, Y3, Z3)]
            spatial_feats: 多尺度 3D 特征列表 [[B, hidden_size, X_l, Y_l, Z_l], ...]
            memory: 展平后的全局编码特征序列 [B, Total_Tokens, hidden_size]
            memory_pos: 全局特征的位置编码 [B, Total_Tokens, hidden_size] (可选)

        Returns:
            hidden_states: 更新后的 Token 序列，形状为 [B, 1 + Q, hidden_size]
        """
        # ================= 1. Self-Attention (Post-LN) =================
        # 为第 0 位的 presence_token 在位置编码前置补 0 (presence 无特定空间坐标)
        # full_query_pos: [B, 1 + Q, hidden_size]
        presence_pos = torch.zeros_like(hidden_states[:, :1, :])
        full_query_pos = torch.cat([presence_pos, query_pos], dim=1)

        # Q 和 K 显式注入位置编码，V 使用纯内容特征
        q = k = hidden_states + full_query_pos
        v = hidden_states

        self_attn_out, _ = self.self_attn(
            query=q,
            key=k,
            value=v,
            need_weights=False,
        )
        hidden_states = hidden_states + self.self_attn_dropout(self_attn_out)
        hidden_states = self.self_attn_norm(hidden_states)

        # ================= 2. Multi-Scale 3D Deformable Cross-Attention (Post-LN) =================
        cross_attn_out = self.cross_attn(
            hidden_states=hidden_states,
            query_pos=query_pos,
            reference_boxes=reference_boxes,
            spatial_shapes=spatial_shapes,
            spatial_feats=spatial_feats,
            memory=memory,
            memory_pos=memory_pos,
        )
        hidden_states = hidden_states + self.cross_attn_dropout(cross_attn_out)
        hidden_states = self.cross_attn_norm(hidden_states)

        # ================= 3. FFN (Post-LN) =================
        # StateEncoderFFN 内部末尾已有 dropout，此处直接相加过 norm
        ffn_out = self.ffn(hidden_states)
        hidden_states = hidden_states + self.ffn_dropout(ffn_out)
        hidden_states = self.ffn_norm(hidden_states)

        return hidden_states

@SDS.register("Deformable_State_Decoder")
class DeformableStateDecoder(nn.Module):
    """
    基于多尺度 3D 可变形空间采样的雷达人体状态解码器 (DeformableStateDecoder).

    核心功能：
    1. 维护 Presence Token 与 Q 个人体 Query Tokens；
    2. 基于 6D 人体几何先验的初始 Reference Boxes；
    3. 堆叠多层 DeformableStateDecoderLayer 逐层细化人体边界框 (Iterative Box Refinement)；
    4. 共享多层预测头 (box_head, cls_head, presence_head) 输出各层辅助监督结果。
    """

    def __init__(self, config: Any):
        """
        Args:
            config: 配置对象，需包含以下字段：
                - hidden_size (int): 统一特征通道维度。
                - num_decoder_layers (int): 解码器层数 (通常为 3~6 层)。
                - num_queries (int): 目标 Query 数量 Q (如 8)。
                - num_attention_heads (int): 注意力头数。
                - num_levels (int): 多尺度层级数。
                - num_points (int, optional): 每个 Head 每层采样的 3D 点数。
                - intermediate_size (int): FFN 隐藏层维度。
                - clamp_presence_logit_max_val (float, optional): Presence Logit 截断阈值，默认 10.0。
                - activation (str, optional): 激活函数 ('gelu' 或 'relu')。
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_decoder_layers
        self.num_queries = config.num_queries
        self.clamp_presence_logit_max_val = getattr(config, "clamp_presence_logit_max_val", 10.0)

        # 1. 级联解码器层
        self.layers = nn.ModuleList([
            DeformableStateDecoderLayer(config) for _ in range(self.num_layers)
        ])

        # 2. 可学习 Query 与初始 6D 参考框
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_size)
        self.reference_points = nn.Embedding(self.num_queries, 6)
        self.presence_token = nn.Embedding(1, self.hidden_size)

        # 3. 6D 边界框正弦条件位置编码与投影头
        # 单轴 num_pos_feats = hidden_size // 2，6 个坐标编码后输出 6 * (2 * (hidden_size // 4)) -> 3 * hidden_size
        self.position_encoding = SinePositionEmbedding(
            num_pos_feats=self.hidden_size // 2,
            temperature=10000.0,
            normalize=True,
        )
        self.ref_point_head = StateDecoderMLP(
            input_dim=3 * self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.hidden_size,
            num_layers=2,
            activation=config.activation,
        )

        # 4. 跨层共享预测头 (Shared Prediction Heads)
        self.box_head = StateDecoderMLP(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=6,
            num_layers=3,
            activation=config.activation,
        )
        self.cls_head = StateDecoderMLP(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=1,
            num_layers=3,
            activation=config.activation,
        )
        self.presence_head = StateDecoderMLP(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=1,
            num_layers=3,
            activation=config.activation,
        )

        self._init_weights()

    def _init_weights(self):
        """初始化网络权重与 3D 雷达人体空间先验。"""
        # 1. 构造人体空间先验初始 6D 边界框 [cx, cy, cz, sx, sy, sz] in [0, 1]
        with torch.no_grad():
            init_boxes = torch.zeros_like(self.reference_points.weight)

            # 2x4 平面网格散布中心点 (以 Q=8 为例)
            xs = torch.linspace(0.25, 0.75, steps=2)
            ys = torch.linspace(0.20, 0.80, steps=4)
            grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
            
            init_boxes[:, 0] = grid_x.flatten()[:self.num_queries]  # cx
            init_boxes[:, 1] = grid_y.flatten()[:self.num_queries]  # cy
            init_boxes[:, 2] = 0.50                                 # cz (人体高度中心)

            # 人体典型尺寸先验比例
            init_boxes[:, 3] = 0.12  # sx: 约 0.7m (针对 6m 探测空间)
            init_boxes[:, 4] = 0.12  # sy: 约 0.7m (针对 6m 探测空间)
            init_boxes[:, 5] = 0.42  # sz: 约 1.7m (针对 4m 探测高度)

            # 写入可学习参数 (经 inverse_sigmoid)
            self.reference_points.weight.copy_(inverse_sigmoid(init_boxes))

        # 2. 初始化 Head 参数
        # 边界框回归头的最后一层权重和偏置清零，保证初始预测出的增量 delta ≈ 0
        nn.init.zeros_(self.box_head.layers[-1].weight)
        nn.init.zeros_(self.box_head.layers[-1].bias)
    
        # 类别预测头使用 Focal Loss 的标准偏置初始化 (如 -2.19 对应初始正样本概率 ~0.1)
        nn.init.constant_(self.cls_head.layers[-1].bias, -2.19)
        nn.init.constant_(self.presence_head.layers[-1].bias, 0.0)

    def forward(
        self,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
        spatial_feats: List[torch.Tensor],
        spatial_shapes: List[Tuple[int, int, int]],
        raw_point_cloud: Optional[torch.Tensor] = None,
    ) -> StateDecoderOutput:
        """
        Args:
            encoder_output: 来自 StateEncoder 的标准输出对象，包含：
                - hidden_states: [B, Total_Tokens, hidden_size]
                - pos_embed: [B, Total_Tokens, hidden_size]
                - spatial_feats: List[[B, hidden_size, X_l, Y_l, Z_l]]
                - spatial_shapes: List[(X_l, Y_l, Z_l)]
            raw_point_cloud: (可选) 原始点云张量，透传挂载

        Returns:
            DeformableStateDecoderOutput: 包含各层预测结果的数据类
        """

        batch_size = memory.shape[0]

        # 1. 扩展 Query 与初始 Reference Boxes
        query_embeds = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)      # [B, Q, hidden_size]
        reference_boxes = self.reference_points.weight.unsqueeze(0).expand(batch_size, -1, -1).sigmoid()  # [B, Q, 6]
        presence_token = self.presence_token.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 1, hidden_size]

        # 拼接 Presence Token 与 Query Embeddings -> [B, 1 + Q, hidden_size]
        hidden_states = torch.cat([presence_token, query_embeds], dim=1)

        intermediate_hidden_states = []
        intermediate_boxes = [reference_boxes]
        intermediate_presence_logits = []
        intermediate_cls_logits = []

        # 2. 逐层微调迭代解码
        for layer in self.layers:
            # 2.1 基于当前 reference_boxes 生成 3D 条件位置编码
            query_sine_embed = self.position_encoding.encode_boxes(reference_boxes)
            query_pos = self.ref_point_head(query_sine_embed)  # [B, Q, hidden_size]

            # 2.2 解码层单层计算 (包含 Self-Attn, MS-Deformable Cross-Attn, FFN)
            hidden_states = layer(
                hidden_states=hidden_states,
                query_pos=query_pos,
                reference_boxes=reference_boxes,
                spatial_shapes=spatial_shapes,
                spatial_feats=spatial_feats,
                memory=memory,
                memory_pos=memory_pos,
            )

            # 2.3 分离 Presence Token 与 Query States
            presence_hidden = hidden_states[:, :1, :]      # [B, 1, hidden_size]
            query_hidden_states = hidden_states[:, 1:, :]  # [B, Q, hidden_size]

            # 2.4 6D 边界框逆 Sigmoid 增量微调 (Box Refinement)
            reference_boxes_inv = inverse_sigmoid(reference_boxes)
            delta_boxes = self.box_head(query_hidden_states)
            new_reference_boxes = (delta_boxes + reference_boxes_inv).sigmoid()
            # 2.5 截断梯度，作为下一层的先验框
            reference_boxes = new_reference_boxes.detach()

            # 2.6 预测 Query 分类与 Presence 存在性 Logits
            cls_logits = self.cls_head(query_hidden_states).squeeze(-1)  # [B, Q]
            presence_logits = self.presence_head(presence_hidden).squeeze(-1)  # [B, 1]
            presence_logits = presence_logits.clamp(
                min=-self.clamp_presence_logit_max_val,
                max=self.clamp_presence_logit_max_val,
            )

            # 2.7 收集当前层输出
            intermediate_hidden_states.append(query_hidden_states)
            intermediate_boxes.append(reference_boxes)
            intermediate_presence_logits.append(presence_logits)
            intermediate_cls_logits.append(cls_logits)

            

        # 3. 堆叠所有解码层的结果 -> [L, B, ...]
        stacked_hidden_states = torch.stack(intermediate_hidden_states, dim=0)
        stacked_boxes = torch.stack(intermediate_boxes[:-1], dim=0)
        stacked_presence_logits = torch.stack(intermediate_presence_logits, dim=0)
        stacked_cls_logits = torch.stack(intermediate_cls_logits, dim=0)

        return StateDecoderOutput(
            intermediate_hidden_states=stacked_hidden_states,
            intermediate_boxes=stacked_boxes,
            intermediate_cls_logits=stacked_cls_logits,
            intermediate_presence_logits=stacked_presence_logits,
            last_presence_token=presence_hidden,
            raw_point_cloud=raw_point_cloud,
        )

#