from src.utils.decorator import *


VFPS=VoxFeatPrecoderRegistry()      # 体素特征预编码模块
VFES=VoxFeatEncoderRegistry()       # 体素特征编码模块
HFES=HistFeatEncoderRegistry()      # 历史信息编码模块
TFS=TempFusionRegistry()            # 时域特征融合模块
SES=StateEncoderRegistry()          # 状态编码模块
SDS=StateDecoderRegistry()          # 状态解码模块
MDS=MaskDecoderRegistry()           # 掩码解码模块


LOSS_LIB=LossRegistry()     # 损失函数
OPTIM = OptimRegistry()     # 优化器
LR_SCHEDULER = LRSchedulerRegistry()    # 学习率调度器

from src.models.modules.histInfo_feature_encoder import *
from src.models.modules.mask_decoder import *
from src.models.modules.state_decoder import *
from src.models.modules.state_encoder import *
from src.models.modules.temp_feature_fusion import *
from src.models.modules.voxel_feature_encoder import *
from src.models.modules.voxel_feature_precoder import *

from src.models.training_modules.loss import *
from src.models.training_modules.optimizer import *
from src.models.training_modules.scheduler import *