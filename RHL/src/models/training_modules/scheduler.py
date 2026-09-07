from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from src.models import LR_SCHEDULER
from torch.optim.lr_scheduler import ConstantLR

@LR_SCHEDULER.register("LinearLR")
def scheduler_LinearLR(optimizer,start_factor,steps,**kwargs):
    return LinearLR(
        optimizer=optimizer,
        start_factor=start_factor,
        total_iters=steps
    )


@LR_SCHEDULER.register("CosineAnnealingLR")
def scheduler_CosineAnnealingLR(optimizer,steps,**kwargs):
    return CosineAnnealingLR(
        optimizer, 
        T_max=steps, 
        eta_min=1e-6
    )

@LR_SCHEDULER.register("ConstantLR")
def scheduler_ConstantLR(optimizer, steps, **kwargs):
    """
    固定学习率调度器。
    通过将 factor 设置为 1.0，学习率在整个训练过程中保持不变。
    """
    return ConstantLR(
        optimizer, 
        factor=1.0, 
        total_iters=steps
    )


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

@LR_SCHEDULER.register("CosineAnnealingWarmRestarts")
def scheduler_CosineAnnealingWarmRestarts(optimizer, steps, **kwargs):
    """
    带有热重启的余弦退火学习率调度器 (Cosine Annealing with Warm Restarts)
    
    参数:
        optimizer: 优化器
        steps: 当前阶段分配到的总步数 (在这里作为备用的参考值)
        kwargs: 接收来自 YAML 的额外配置，如 T_0, T_mult 等
    """
    T_0 = kwargs.get('T_0', max(1, steps // 5)) 
    T_0_ratio=kwargs.get('T_0_ratio', None)
    if T_0_ratio is not None:
        T_0=max(1,min(int(steps*T_0_ratio),steps))
    
    # T_mult: 每次重启后，下一个周期的步数乘子。默认为 1（每个周期长度相同）
    # 如果设为 2，则周期长度会翻倍，如 T_0, 2*T_0, 4*T_0
    T_mult = kwargs.get('T_mult', 1)
    
    # eta_min: 学习率下限
    eta_min = kwargs.get('eta_min', 1e-6)

    return CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min
    )