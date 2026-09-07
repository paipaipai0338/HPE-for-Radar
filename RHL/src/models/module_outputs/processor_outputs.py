import torch
from typing import  List, Union, Tuple, Any

class PocPaddingProcessorOutputs:
    """
    Hugging Face 风格的 Processor 输出容器类。
    特点：
    1. 所有属性均为 torch.Tensor 类型（包括空间配置参数转成的 Tensor）。
    2. 支持属性访问 (outputs.raw_point_cloud) 与字典访问 (outputs["raw_point_cloud"])。
    3. 支持一键 .to(DEVICE) 数据迁移（所有属性均可直接 .to()）。
    """
    def __init__(
        self,
        raw_point_cloud: torch.Tensor,
        point_valid_mask: torch.Tensor,
        xlim: torch.Tensor,
        ylim: torch.Tensor,
        zlim: torch.Tensor,
        voxel_size: torch.Tensor,
        grid_shape: torch.Tensor
    ):
        self.raw_point_cloud = raw_point_cloud        # [B, P, 4] Tensor
        self.point_valid_mask = point_valid_mask      # [B, P] Tensor
        self.xlim = xlim                              # [2] Tensor
        self.ylim = ylim                              # [2] Tensor
        self.zlim = zlim                              # [2] Tensor
        self.voxel_size = voxel_size                  # [3] Tensor
        self.grid_shape = grid_shape                  # [3] Tensor (X, Y, Z)

    def __getitem__(self, key: str) -> torch.Tensor:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"PocPaddingProcessorOutputs has no attribute '{key}'")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def keys(self) -> List[str]:
        return ["raw_point_cloud", "point_valid_mask", "xlim", "ylim", "zlim", "voxel_size", "grid_shape"]

    def items(self) -> List[Tuple[str, torch.Tensor]]:
        return [(k, getattr(self, k)) for k in self.keys()]

    def to(self, device: Union[str, torch.device]) -> "PocPaddingProcessorOutputs":
        """
        全员 Tensor，直接一键全部 .to(device)！
        """
        for k in self.keys():
            val = getattr(self, k)
            if isinstance(val, torch.Tensor):
                setattr(self, k, val.to(device))
        return self