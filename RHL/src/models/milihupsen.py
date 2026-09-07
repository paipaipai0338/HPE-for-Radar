import torch
import torch.nn as nn
from src.configs.milihupsen_config import (
    MiliHuPsenConfig, DetrMiliHuPsenConfig
)
from transformers import PreTrainedModel
from src.models.build_modules import *
from typing import Any, List, Optional
from src.models.module_outputs.processor_outputs import PocPaddingProcessorOutputs
from src.models.module_outputs.voxel_segmentation_outputs import (
    VoxelSegmentationOutput, DetrMiliHupSenOutput
)
from src.models.module_outputs.state_encoder_outputs import DeformableVoxelEncoderOutput
from src.models.module_outputs.state_decoder_outputs import StateDecoderOutput
from src.models.modules.state_decoder import inverse_sigmoid
import os

class MiliHuPsen(PreTrainedModel):
    config_class = MiliHuPsenConfig

    def __init__(self, config: MiliHuPsenConfig):
        super().__init__(config)
        self.config = config
        self.voxel_feature_precoder = build_voxel_feature_precoder(
            config.voxel_feature_precoder_config
        )
        self.voxel_feature_encoder = build_voxel_feature_encoder(
            config.voxel_feature_encoder_config
        )
        self.histInfo_feature_encoder = build_histInfo_feature_encoder(
            config.histInfo_feature_encoder_config
        )
        self.temp_feature_fusion_module = build_temp_feature_fusion_module(
            config.temp_feature_fusion_config
        )
        self.state_decoder = build_state_decoder(
            config.state_decoder_config
        )
        self.mask_decoder = build_mask_decoder(
            config.mask_decoder_config
        )

        self.time_assist = config.time_assist

    def get_state_dict(self,pt_path):
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"❌ 找不到微调权重文件: {pt_path}")
        if pt_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(pt_path)
        else:
            state_dict = torch.load(pt_path, map_location="cpu")
        return state_dict
    
    def load_pt_in_wrapper(self,pt_path):
        state_dict=self.get_state_dict(pt_path)
        filter_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_key = k.replace("model.", "")
                filter_state_dict[new_key] = v
        missing, unexpected = self.load_state_dict(filter_state_dict, strict=True)
        print(f" ┣━ model: 成功加载 {len(filter_state_dict)} 项 | 缺失 {len(unexpected)} 项")
        if missing:
            print(f" ┃  ┗━ 缺失示例: {missing[:3]}")
    
        # ============= [加载完整权重] ====================== 
    
    def load_pt(self,pt_path: str):
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"❌ 找不到模型权重文件: {pt_path}")
            
        print(f"[*] 正在加载模型权重: {pt_path}")
        
        if str(pt_path).endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(pt_path)
        else:
            state_dict = torch.load(pt_path, map_location="cpu")
        self.load_state_dict(state_dict,strict=True)

    def forward(
        self,
        current_data: PocPaddingProcessorOutputs,
        history_points_list: Optional[List[PocPaddingProcessorOutputs]] = None,
        h_state: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> VoxelSegmentationOutput:
        """
        MiliHuPsen 端到端前向传播。

        Args:
            current_data: 当前帧经过 PocPaddingProcessor 预处理后的标准输出容器
            history_points_list: 历史 T-1 帧 PocPaddingProcessorOutputs 列表 (单帧训练模式下可为 None 或 [])
            h_state: 外部维护传入的 ConvGRU 初始/上一帧隐藏状态 [B, C, X/4, Y/4, Z/4]
            **kwargs: 预留透传参数

        Returns:
            MiliHuPsenOutput: 封装了检测、分类、掩码分割与全局 Occupancy 预测的标准结构体
        """
        # =========================================================================
        # 1. 当前帧体素预编码与多尺度特征提取
        # =========================================================================
        # [B, in_channels, X, Y, Z] (如 [B, 66, 48, 48, 32])
        dense_voxel_features = self.voxel_feature_precoder(current_data)

        # 提取当前帧多尺度体素特征金字塔
        multiscale_feats = self.voxel_feature_encoder(dense_voxel_features)
        f1 = multiscale_feats["F1"]  # [B, 32, X, Y, Z]
        f2 = multiscale_feats["F2"]  # [B, 64, X/2, Y/2, Z/2]
        current_f3 = multiscale_feats["F3"]  # [B, 128, X/4, Y/4, Z/4]

        # =========================================================================
        # 2. 时域特征编码与融合分支 (通过 self.time_assist 控制)
        # =========================================================================
        next_h_state = None
        temporal_feature = None

        if self.time_assist:
            # 提取历史帧深层特征: [B, T-1, 128, X/4, Y/4, Z/4]
            f3_history = self.histInfo_feature_encoder(
                history_points_list=history_points_list,
                shared_precoder=self.voxel_feature_precoder,
                shared_encoder=self.voxel_feature_encoder,
            )

            # ConvGRU 融合时空记忆
            f_temporal, next_h_state = self.temp_feature_fusion_module(
                current_f3=current_f3,
                f3_history=f3_history,
                h_state=h_state,
            )
            # 供下游解码器使用的最深层特征
            deep_features = f_temporal
            temporal_feature = f_temporal
        else:
            # 单帧模式：直接使用当前帧 F3 作为深层语义表征
            deep_features = current_f3
            temporal_feature = None

        # =========================================================================
        # 3. 状态与 3D 边界框解码 (StateDecoder)
        # =========================================================================
        state_output = self.state_decoder(
            voxel_features=deep_features,
            raw_point_cloud=current_data.raw_point_cloud,
            **kwargs,
        )

        all_box_offsets = self.state_decoder.box_head(state_output.intermediate_hidden_states)  # [L, B, Q, 6]
        ref_boxes_inv_sig = inverse_sigmoid(state_output.intermediate_boxes)                        # [L, B, Q, 6]
        all_pred_boxes = (ref_boxes_inv_sig + all_box_offsets).sigmoid()

        # =========================================================================
        # 4. 实例体素掩码解码 (MaskDecoder)
        # =========================================================================
        # 获取 StateDecoder 最后一层解算出的实例 Query 隐状态: [B, Q, hidden_size]
        final_queries = state_output.intermediate_hidden_states[-1]

        # 传入多尺度特征与时域特征 (内部通过 _embed_voxels 完成深层特征替代与上采样重建)
        mask_output = self.mask_decoder(
            decoder_queries=final_queries,
            backbone_features=[f1, f2, current_f3],
            temporal_feature=temporal_feature,
        )

        # =========================================================================
        # 5. 整合打包输出
        # =========================================================================
        return VoxelSegmentationOutput(
            pred_boxes=all_pred_boxes,                                  # [L, B, Q, 6] 完整计算后的 L 层 3D 预测框
            cls_logits=state_output.intermediate_cls_logits,            # [L, B, Q] 各层 Query 分类 Logits
            presence_logits=state_output.intermediate_presence_logits,  # [L, B, 1] 各层全局场景存在性 Logits
            pred_masks=mask_output.pred_masks,                          # [B, Q, X, Y, Z] 最终层实例体素掩码 Logits
            semantic_seg=mask_output.semantic_seg,                      # [B, 1, X, Y, Z] 全局前景体素 Occupancy Logits
            next_h_state=next_h_state,                                  # [B, C, X/4, Y/4, Z/4] 更新后的时序隐状态 (可选)
            raw_point_cloud=current_data.raw_point_cloud,                # [B, P, 4] 原始点云透传
            point_valid_mask=current_data.point_valid_mask,             # [B, P] 有效点掩码透传
            multiscale_features=multiscale_feats,                       # {"F1": ..., "F2": ..., "F3": ...} 多尺度特征字典
        )

class DetrMiliHuPsen(MiliHuPsen):
    config_class = DetrMiliHuPsenConfig
    def __init__(self, config: DetrMiliHuPsenConfig):
        super().__init__(config)
        self.state_encoder = build_state_encoder(
            config.state_encoder_config
        )
        self.mask_decoder = None


    def forward(
        self,
        current_data: PocPaddingProcessorOutputs,
        history_points_list: Optional[List[PocPaddingProcessorOutputs]] = None,
        h_state: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        """
        MiliHuPsen 端到端前向传播。
        
        Args:
            current_data: 当前帧经过 PocPaddingProcessor 预处理后的标准输出容器
            history_points_list: 历史 T-1 帧 PocPaddingProcessorOutputs 列表 (单帧训练模式下可为 None 或 [])
            h_state: 外部维护传入的 ConvGRU 初始/上一帧隐藏状态 [B, C, X/4, Y/4, Z/4]
            **kwargs: 预留透传参数
        
        Returns:
            MiliHuPsenOutput: 封装了检测、分类、掩码分割与全局 Occupancy 预测的标准结构体
        """
        # =========================================================================
        # 1. 当前帧体素预编码与多尺度特征提取
        # =========================================================================
        # [B, in_channels, X, Y, Z] (如 [B, 66, 48, 48, 32])
        dense_voxel_features = self.voxel_feature_precoder(current_data)
        
        # 提取当前帧多尺度体素特征金字塔
        multiscale_feats = self.voxel_feature_encoder(dense_voxel_features)
        f1 = multiscale_feats["F1"]  # [B, 32, X, Y, Z]
        f2 = multiscale_feats["F2"]  # [B, 64, X/2, Y/2, Z/2]
        current_f3 = multiscale_feats["F3"]  # [B, 128, X/4, Y/4, Z/4]
        
        # =========================================================================
        # 2. 时域特征编码与融合分支 (通过 self.time_assist 控制)
        # =========================================================================
        next_h_state = None
        temporal_feature = None
        
        if self.time_assist:
            # 提取历史帧深层特征: [B, T-1, 128, X/4, Y/4, Z/4]
            f3_history = self.histInfo_feature_encoder(
                history_points_list=history_points_list,
                shared_precoder=self.voxel_feature_precoder,
                shared_encoder=self.voxel_feature_encoder,
            )
        
            # ConvGRU 融合时空记忆
            f_temporal, next_h_state = self.temp_feature_fusion_module(
                current_f3=current_f3,
                f3_history=f3_history,
                h_state=h_state,
            )
            # 供下游解码器使用的最深层特征
            deep_features = f_temporal
            temporal_feature = f_temporal
        else:
            # 单帧模式：直接使用当前帧 F3 作为深层语义表征
            deep_features = current_f3
            temporal_feature = None

        # =========================================================================
        # 3. 状态编码 (StateEncoder)
        # =========================================================================
        encoder_input = [f1, f2, deep_features]
        encoder_output: DeformableVoxelEncoderOutput = self.state_encoder(
            encoder_input
        )
        # =========================================================================
        # 4. 状态与 3D 边界框解码 (StateDecoder)
        # =========================================================================
        decoder_output: StateDecoderOutput = self.state_decoder(
            memory = encoder_output.hidden_states,
            memory_pos = encoder_output.pos_embed,
            spatial_feats = encoder_output.spatial_feats,
            spatial_shapes = encoder_output.spatial_shapes,
            raw_point_cloud=current_data.raw_point_cloud,
        )

        all_box_offsets = self.state_decoder.box_head(decoder_output.intermediate_hidden_states)  # [L, B, Q, 6]
        ref_boxes_inv_sig = inverse_sigmoid(decoder_output.intermediate_boxes)                        # [L, B, Q, 6]
        all_pred_boxes = (ref_boxes_inv_sig + all_box_offsets).sigmoid()


        return DetrMiliHupSenOutput(
            pred_boxes=all_pred_boxes,                                  # [L, B, Q, 6] 完整计算后的 L 层 3D 预测框
            cls_logits=decoder_output.intermediate_cls_logits,            # [L, B, Q] 各层 Query 分类 Logits
            presence_logits=decoder_output.intermediate_presence_logits,  # [L, B, 1] 各层全局场景存在性 Logits
            next_h_state=next_h_state,                                  # [B, C, X/4, Y/4, Z/4] 更新后的时序隐状态 (可选)
            raw_point_cloud=current_data.raw_point_cloud,                # [B, P, 4] 原始点云透传
            point_valid_mask=current_data.point_valid_mask,             # [B, P] 有效点掩码透传
            multiscale_features=multiscale_feats,                       # {"F1": ..., "F2": ..., "F3": ...} 多尺度特征字典
        )
        
#       