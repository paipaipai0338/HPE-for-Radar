import torch
import torch.nn as nn
from typing import List, Optional, Any
from src.models import HFES


@HFES.register("Shared_Historical_Encoder")
class HistoricalInformationFeatureEncoder(nn.Module):
    """
    历史信息特征编码模块 (HistoricalInformationFeatureEncoder - 运行时动态绑定版)
    职责：
    1. 接收历史 T-1 帧的雷达点云序列。
    2. 在 forward 过程中，根据 share_modules 策略与传入的共享模块完成并行前向编码。
    3. 产出多帧历史深层特征 F3_history: [B, T-1, 128, X/4, Y/4, Z/4]。
    """
    def __init__(
        self, 
        config,
    ):
        super().__init__()
        self.share_modules = config.share_modules

    def forward(
        self,
        history_points_list: Optional[List[Any]] = None,
        *,
        shared_precoder: Optional[nn.Module] = None,
        shared_encoder: Optional[nn.Module] = None,
        **kwargs: Any
    ) -> Optional[torch.Tensor]:
        """
        前向传播接口
        参数:
            history_points_list: 包含 T-1 个历史帧 ProcessorOutputs 对象的列表。若为 None 则返回 None。
            shared_precoder:     当前帧共享的 Precoder 模块实例
            shared_encoder:      当前帧共享的 Encoder 模块实例
            **kwargs:            预留扩展参数
        返回:
            F3_history: [B, T-1, 128, X/4, Y/4, Z/4] 的时序特征张量 (若无历史则返回 None)
        """
        if history_points_list is None or len(history_points_list) == 0:
            return None

        # 如果开启了模块共享策略，但未传入对应的共享模块实例，则直接报错
        if self.share_modules and (shared_precoder is None or shared_encoder is None):
            raise RuntimeError(
                "HistoricalInformationFeatureEncoder 配置了 share_modules=True，"
                "但在 forward 过程中未检测到传入的 shared_precoder 或 shared_encoder！"
            )

        batch_f3_history = []

        # 遍历每一帧历史观测 (t-k 到 t-1)
        for hist_outputs in history_points_list:
            # 1. 使用共享的 Precoder 将历史点云转换为稠密网格特征
            hist_dense_feat = shared_precoder(hist_outputs)
            
            # 2. 使用共享的 Encoder 提取多尺度特征金字塔
            hist_multiscale_feats = shared_encoder(hist_dense_feat)
            
            # 3. 提取文档指定的深层抽象特征 F3 (形状: [B, 128, X/4, Y/4, Z/4])
            hist_f3 = hist_multiscale_feats["F3"]
            
            batch_f3_history.append(hist_f3)

        # 4. 在时间维度 (Time Dimension) 上堆叠 -> [B, T-1, 128, X/4, Y/4, Z/4]
        F3_history = torch.stack(batch_f3_history, dim=1)

        return F3_history