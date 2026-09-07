
# ==================== 模型装饰器 ====================
class CustomRegistry:
    def __init__(self):
        self._models = {}
    def register(self, name):
        """装饰器：将类注册到字典中"""
        def decorator(cls):
            self._models[name] = cls
            return cls
        return decorator


# ==================== 体素特征预编码装饰器 ====================
class VoxFeatPrecoderRegistry(CustomRegistry):
    def get(self, name):
        """根据名称获取类"""
        if name not in self._models:
            raise KeyError(f"体素特征预编码模块 {name} 未注册！")
        return self._models[name]


# ==================== 体素特征编码装饰器 ====================
class VoxFeatEncoderRegistry(CustomRegistry):
    def get(self, name):
        """根据名称获取类"""
        if name not in self._models:
            raise KeyError(f"体素特征编码模块 {name} 未注册！")
        return self._models[name]


# ==================== 历史信息编码模块装饰器 ====================
class HistFeatEncoderRegistry(CustomRegistry):
    def get(self, name):
        """根据名称获取类"""
        if name not in self._models:
            raise KeyError(f"历史信息编码模块 {name} 未注册！")
        return self._models[name]


# ==================== 时域特征融合模块装饰器 ====================
class TempFusionRegistry(CustomRegistry):
    def get(self, name):
        """根据名称获取类"""
        if name not in self._models:
            raise KeyError(f"时域特征融合模块 {name} 未注册！")
        return self._models[name]

# ==================== 状态编码模块装饰器 ====================
class StateEncoderRegistry(CustomRegistry):
    def get(self, name):
        """根据名称获取类"""
        if name not in self._models:
            raise KeyError(f"状态编码模块 {name} 未注册！")
        return self._models[name]


# ==================== 状态解码模块装饰器 ====================
class StateDecoderRegistry(CustomRegistry):
    def get(self, name):
        """根据名称获取类"""
        if name not in self._models:
            raise KeyError(f"状态解码模块 {name} 未注册！")
        return self._models[name]


# ==================== 掩码解码模块装饰器 ====================
class MaskDecoderRegistry(CustomRegistry):
    def get(self, name):
        """根据名称获取类"""
        if name not in self._models:
            raise KeyError(f"掩码解码模块 {name} 未注册！")
        return self._models[name]



# ==================== 数据集装饰器 ====================
class DatasetRegistry(CustomRegistry):
    def get(self, name):
        """根据名称获取类"""
        if name not in self._models:
            raise KeyError(f"数据集 {name} 未注册！")
        return self._models[name]

# ==================== 数据增强装饰器 ====================
class DataAugmentationRegistry(CustomRegistry):
    def get(self, name):
        """根据名称获取类"""
        if name not in self._models:
            raise KeyError(f"数据增强方法 {name} 未注册！")
        return self._models[name]


# ==================== 损失函数装饰器 ====================
class LossRegistry(CustomRegistry):
    def get(self, name):
        """根据名称获取类"""
        if name not in self._models:
            raise KeyError(f"损失函数: {name} 未注册！")
        return self._models[name]

# ==================== 优化器装饰器 ====================
class OptimRegistry(CustomRegistry):
    def get(self, name):
        """根据名称获取类"""
        if name not in self._models:
            raise KeyError(f"优化器: {name} 未注册！")
        return self._models[name]


# ==================== 学习率调度器装饰器 ====================
class LRSchedulerRegistry(CustomRegistry):
    def get(self, name):
        """根据名称获取类"""
        if name not in self._models:
            raise KeyError(f"优化器: {name} 未注册！")
        return self._models[name]


    