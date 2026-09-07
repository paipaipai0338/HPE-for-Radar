import torch.nn as nn
from typing import List, Dict, Union, Optional

class ModelInfo:
    """
    用于分析和格式化输出 PyTorch 模型参数信息的工具类。
    
    功能涵盖：计算模型总参数量、可训练参数量、预估参数占用的显存体积 (MB)，
    以及遍历打印各顶层子模块的参数分布与占比。
    
    Example:
        >>> import torch.nn as nn
        >>> my_model = nn.Linear(10, 2) # 假设已有一个模型
        >>>
        >>> # 1. 实例化类并自动解析模型参数
        >>> info = ModelInfo(model=my_model)
        >>> 
        >>> # 2. 获取格式化的模型信息字符串
        >>> summary_str = info.get_model_info()
        >>> 
        >>> # 3. 直接在控制台打印模型信息
        >>> info.print_info()
        >>> 
        >>> # 4. 或者更简洁的做法：直接 print 对象本身
        >>> print(info)
    """
    def __init__(self, model: nn.Module):
        """
        初始化 ModelInfo 并自动解析模型结构。
        
        Args:
            model (nn.Module): 需要分析的 PyTorch 模型实例。
        
        Raises:
            TypeError: 如果传入的 model 不是 torch.nn.Module 的实例。
        """
        if not isinstance(model, nn.Module):
            raise TypeError(f"入参 model 必须是 torch.nn.Module 类型，但收到的是: {type(model)}")
            
        self.model: nn.Module = model
        self.total_params: int = 0
        self.trainable_params: int = 0
        self.param_size_mb: float = 0.0
        
        # 记录子模块详情的列表
        self.module_details: List[Dict[str, Union[str, int, float]]] = []
        
        # 初始化时自动计算参数
        self._analyze_model()

    def _analyze_model(self) -> None:
        """内部核心计算逻辑：获取全局参数量、内存占用和各子模块的参数量"""
        
        # 1. 计算整个模型的总参数量、可训练参数量及内存占用
        for p in self.model.parameters():
            numel = p.numel()
            self.total_params += numel
            if p.requires_grad:
                self.trainable_params += numel
            # 计算字节数并转换为 MB ( element_size() 返回单个元素的字节数，如 float32 为 4 )
            self.param_size_mb += (numel * p.element_size()) / (1024 ** 2)

        # 2. 遍历模型的顶层子模块，计算各自的参数量与占比
        for name, child in self.model.named_children():
            child_total = sum(p.numel() for p in child.parameters())
            child_trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
            
            # 计算该模块参数占总参数量的百分比 (防御除以0的情况)
            ratio = (child_total / self.total_params * 100) if self.total_params > 0 else 0.0
            
            self.module_details.append({
                'name': name,
                'class_name': child.__class__.__name__,
                'total_params': child_total,
                'trainable_params': child_trainable,
                'ratio': ratio
            })

    def get_model_info(self) -> str:
        """
        生成格式化的模型信息字符串。
        
        Returns:
            str: 包含对齐表格的格式化字符串。
        """
        lines = []
        separator = "=" * 80
        light_separator = "-" * 80
        
        # 全局信息头部
        lines.append(separator)
        lines.append(f"模型总参数量 (Total Params)   : {self.total_params / 1e6:.3f} M")
        lines.append(f"可训练参数量 (Trainable Params): {self.trainable_params / 1e6:.3f} M")
        lines.append(f"参数内存估算 (Params Size)    : {self.param_size_mb:.2f} MB")
        lines.append(light_separator)
        
        # 动态表头
        header = f"{'模块名称 (Name)':<18} | {'模块类型 (Type)':<18} | {'参数量 (M)':<15} | {'占比 (%)':<8}"
        lines.append(header)
        lines.append(light_separator)
        
        # 详情遍历
        for detail in self.module_details:
            raw_name = str(detail['name'])
            raw_cls = str(detail['class_name'])
            
            # 智能截断超长名称，避免把表格撑变形 (保留前16个字符+..)
            name = raw_name if len(raw_name) <= 18 else raw_name[:16] + ".."
            cls_name = raw_cls if len(raw_cls) <= 18 else raw_cls[:16] + ".."
            
            params = f"{detail['total_params'] / 1e6:.3f} M"
            ratio = f"{detail['ratio']:.2f}%"
            
            lines.append(f"{name:<18} | {cls_name:<18} | {params:<15} | {ratio:<8}")
            
        lines.append(separator)
        return "\n".join(lines)

    def print_info(self) -> None:
        """直接在控制台打印模型信息"""
        print(self.get_model_info())
        
    def __str__(self) -> str:
        """实现魔术方法，支持直接使用 print(对象) 输出"""
        return self.get_model_info()

#下一个类