from transformers import PretrainedConfig


# 全局参数
BACKBONE_CHANNELS= [32, 64, 128]
HIDDEN_SIZE = 128
NUM_LEVELS =3

class VoxelFeaturePrecoderConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.type="MLP_Voxel_Feature_Precoder"
        self.in_dim = 4
        self.hidden_dim1 = 16
        self.hidden_dim2 = 32


class VoxelFeatureEncoderConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "Res_3DUNet_Encoder"
        self.in_channels = 66
        self.num_groups = 8
        self.dropout_p = 0.1
        self.backbone_channels = BACKBONE_CHANNELS

class HistInfoFeatureEncoderConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "Shared_Historical_Encoder"
        self.share_modules = True

class TemporalFeatureFusionConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "ConvGRU3D_Fusion"

        self.backbone_channels = BACKBONE_CHANNELS
        self.num_groups = 8

class StateDecoderConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "boxRPB3D_State_Decoder"

        self.hidden_size = HIDDEN_SIZE
        self.intermediate_size = HIDDEN_SIZE * 2
        self.num_decoder_layers = 3
        self.num_queries = 8
        self.num_heads = 8
        self.clamp_presence_logit_max_val = 10.0
        self.dropout = 0.1
        self.activation = "gelu"

class MaskDecoderConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "Voxel_FPN_Mask_Decoder"
        self.hidden_size = HIDDEN_SIZE
        self.backbone_channels = BACKBONE_CHANNELS
        self.num_groups = 8
        self.mask_embedder_num_layers = 3

class DetrStateEncoderConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "Detr_State_Encoder"

        self.in_channels_list = BACKBONE_CHANNELS
        self.hidden_size = HIDDEN_SIZE
        self.num_levels = NUM_LEVELS
        self.num_groups = 8
        self.intermediate_size = HIDDEN_SIZE * 2
        self.num_layers = 3
        self.num_queries = 8
        self.num_heads = 8
        self.num_points = 4
        self.clamp_presence_logit_max_val = 10.0
        self.dropout = 0.1
        self.activation = "gelu"

class DetrStateDecoderConfig(StateDecoderConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "Deformable_State_Decoder"
        self.num_levels = NUM_LEVELS
        self.num_points = 4


class MiliHuPsenConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.time_assist = False        # 是否使用时域信息辅助
        self.voxel_feature_precoder_config=VoxelFeaturePrecoderConfig()
        self.voxel_feature_encoder_config = VoxelFeatureEncoderConfig()
        self.histInfo_feature_encoder_config = HistInfoFeatureEncoderConfig()
        self.temp_feature_fusion_config = TemporalFeatureFusionConfig()
        self.state_decoder_config = StateDecoderConfig()
        self.mask_decoder_config = MaskDecoderConfig()


class DetrMiliHuPsenConfig(MiliHuPsenConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.state_encoder_config = DetrStateEncoderConfig()
        self.state_decoder_config = DetrStateDecoderConfig()
        




