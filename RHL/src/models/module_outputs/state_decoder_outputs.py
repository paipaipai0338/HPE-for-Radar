from dataclasses import dataclass
from typing import Any, Optional, Union
import torch


@dataclass
class StateDecoderOutput:
    """3D 人体状态解码器输出数据类（StateDecoderOutput）。

    特性支持：
    1. 属性访问：outputs.intermediate_boxes
    2. 字典式访问：outputs["intermediate_boxes"]、"intermediate_boxes" in outputs、keys() 等
    3. 一键设备迁移：outputs.to(device) 会递归迁移所有包含的 Tensor

    维度符号说明：
    - L: 解码器层数 (num_decoder_layers)
    - B: Batch 大小 (batch_size)
    - Q: 实例 Query 数量 (num_queries)
    - C: 隐藏特征维度 (hidden_size)

    字段含义：
    - intermediate_hidden_states: 逐层 LayerNorm 后的实例 Query 特征，形状为 [L, B, Q, C]。
    - intermediate_boxes: 逐层累加细化后的 3D 边界框 (归一化在 [0, 1] 区间)，形状为 [L, B, Q, 6]。
      参数格式为 [cx, cy, cz, sx, sy, sz]。
    - intermediate_presence_logits: 逐层全局人体存在性预测对数几率，形状为 [L, B, 1]。
    - intermediate_cls_logits: (可选) 逐层实例二分类预测对数几率，形状为 [L, B, Q]。
    - last_presence_token: (可选) 最终层 Presence Token 的隐藏状态，形状为 [B, 1, C]。
    - raw_point_cloud: (可选) 原始点云张量或辅助特征，形状为 [B, P, 4] 或类似结构。
    """

    intermediate_hidden_states: torch.Tensor
    intermediate_boxes: torch.Tensor
    intermediate_presence_logits: torch.Tensor
    intermediate_cls_logits: Optional[torch.Tensor] = None
    last_presence_token: Optional[torch.Tensor] = None
    raw_point_cloud: Optional[torch.Tensor] = None

    # =========================================================================
    # 1. 字典式访问支持 (Dict-like Access)
    # =========================================================================
    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(f"'{self.__class__.__name__}' object has no key '{key}'")
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) and getattr(self, key) is not None

    def keys(self):
        return [k for k, v in self.__dict__.items() if not k.startswith("_") and v is not None]

    def values(self):
        return [v for k, v in self.__dict__.items() if not k.startswith("_") and v is not None]

    def items(self):
        return [(k, v) for k, v in self.__dict__.items() if not k.startswith("_") and v is not None]

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    # =========================================================================
    # 2. 一键设备迁移支持 (.to(device))
    # =========================================================================
    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
        copy: bool = False,
        memory_format: Optional[torch.memory_format] = None,
    ) -> "StateDecoderOutput":
        """将内部所有 Tensor 类型的属性一键迁移至指定的设备 (Device) 或数据类型 (Dtype)。"""
        new_fields = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                new_fields[k] = v.to(
                    device=device,
                    dtype=dtype,
                    non_blocking=non_blocking,
                    copy=copy,
                    memory_format=memory_format,
                )
            else:
                new_fields[k] = v
        return StateDecoderOutput(**new_fields)

    # =========================================================================
    # 3. 便捷属性属性获取 (Properties)
    # =========================================================================
    @property
    def final_hidden_states(self) -> torch.Tensor:
        """获取最后一层的实例 Query 特征: [B, Q, C]"""
        return self.intermediate_hidden_states[-1]

    @property
    def final_boxes(self) -> torch.Tensor:
        """获取最后一层预测的 3D 边界框: [B, Q, 6]"""
        return self.intermediate_boxes[-1]

    @property
    def final_presence_logit(self) -> torch.Tensor:
        """获取最后一层的全局存在性预测: [B, 1]"""
        return self.intermediate_presence_logits[-1]

    @property
    def final_cls_logits(self) -> Optional[torch.Tensor]:
        """获取最后一层的实例分类预测: [B, Q]"""
        if self.intermediate_cls_logits is not None:
            return self.intermediate_cls_logits[-1]
        return None